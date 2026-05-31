#ifndef ENHANCED_RRT_H_
#define ENHANCED_RRT_H_

#include "benchmark_core.h"
#include <unordered_map>
#include <unordered_set>

// ============================================================================
// Enhanced Informed RRT* (this project's implementation)
// ============================================================================
// Key enhancements over classical:
//   1. Two-phase search (Phase 1: brute-force NN + Phase 2: kd-tree informed optimization)
//   2. APF (Artificial Potential Field) sampling with 26-direction repulsive field
//   3. Obstacle-outside sampling (project around blocking obstacles)
//   4. Clearance-aware goal connection
//   5. Shortcut optimization with BFS cost propagation
//   6. Path smoothing (Douglas-Peucker)
//   7. Turn penalty in edge cost
//   8. Delayed exit after first solution
//   9. APF clearance cache
// ============================================================================

struct RRTNodeEnhanced
{
  Eigen::Vector3d x;
  RRTNodeEnhanced *parent;
  double cost;
  int id;
  std::vector<RRTNodeEnhanced *> children;

  RRTNodeEnhanced(const Eigen::Vector3d &_x, RRTNodeEnhanced *_parent, double _cost, int _id)
      : x(_x), parent(_parent), cost(_cost), id(_id) {}

  void addChild(RRTNodeEnhanced *c) { children.push_back(c); }
  void removeChild(RRTNodeEnhanced *c)
  {
    for (auto it = children.begin(); it != children.end(); ++it)
      if (*it == c)
      {
        children.erase(it);
        break;
      }
  }
};

class EnhancedInformedRRTstar
{
public:
  EnhancedInformedRRTstar()
  {
    rand_buf_.resize(4096);
    for (auto &v : rand_buf_)
      v = (double)rand() / RAND_MAX;
  }

  void init(const GridMap3D &grid)
  {
    grid_ = &grid;
    workspace_min_ = grid.min_bound;
    workspace_max_ = grid.max_bound;
    resolution_ = grid.resolution;
  }

  RunMetrics search(const Eigen::Vector3d &start, const Eigen::Vector3d &goal,
                    double step_size = 0.2, double total_time_budget = 0.5,
                    int max_iter = 20000)
  {
    RunMetrics m;
    auto t0 = std::chrono::high_resolution_clock::now();
    const double phase1_ratio = 0.65;  // 65% of time for initial exploration
    auto deadline = t0 + std::chrono::duration<double>(total_time_budget);

    collision_checks_ = 0;
    rand_pos_ = -1;
    srand(std::random_device{}());  // fresh seed per search for independent trials
    for (auto &v : rand_buf_)
      v = (double)rand() / RAND_MAX;
    ++apf_cache_frame_;
    apf_cache_.clear();

    start_ = start;
    goal_ = goal;
    step_size_ = step_size;
    c_min_ = (goal_ - start_).norm();
    c_best_ = std::numeric_limits<double>::infinity();

    clearTree();

    RRTNodeEnhanced *start_node = new RRTNodeEnhanced(start_, nullptr, 0.0, node_id_counter_++);
    nodes_.push_back(start_node);
    id_to_node_[start_node->id] = start_node;

    computeEllipse();

    // Direct connection?
    if (tryConnectToGoal(start_node))
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      m.total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      m.time_to_first_solution_ms = m.total_time_ms;
      m.success = true;
      m.total_nodes = nodes_.size();
      m.collision_checks = collision_checks_;
      auto path = getPathVector();
      m.path_nodes = path.size();
      computePathMetrics(*grid_, path, m.path_length, m.min_clearance, m.avg_clearance, m.smoothness);
      return m;
    }

    auto phase1_end = t0 + std::chrono::duration<double>(total_time_budget * phase1_ratio);

    // ========== PHASE 1: RRT growth ==========
    int iter = 0;
    bool goal_reached = false;
    constexpr int MIN_POST_GOAL_ITERATIONS = 50;
    int post_goal_iterations = 0;

    while (std::chrono::high_resolution_clock::now() < phase1_end && iter < max_iter)
    {
      ++iter;
      Eigen::Vector3d x_rand;
      double roll = fastRand();

      if (roll < goal_bias_)
        x_rand = goal_;
      else if (roll < goal_bias_ + 0.55 && sampleObstacleOutside(x_rand))
      {
        // bias toward obstacle-outside free space
      }
      else if (roll < goal_bias_ + apf_sampling_ratio_)
        x_rand = sampleWithAPF();
      else
        x_rand = sampleWorkspace();

      RRTNodeEnhanced *x_nearest = nearestBruteForce(x_rand);
      if (x_nearest == nullptr)
        continue;

      Eigen::Vector3d x_new = steer(x_nearest, x_rand);
      if (grid_->isOccupied(x_new))
      {
        Eigen::Vector3d outside;
        if (!projectToObstacleOutside(x_new, x_rand - x_nearest->x, outside))
          continue;
        x_new = steer(x_nearest, outside);
        if (grid_->isOccupied(x_new))
          continue;
      }
      if (!isSegmentFree(x_nearest->x, x_new))
      {
        Eigen::Vector3d blocking_center, path_dir, outside;
        if (!findBlockingObstacle(x_nearest->x, x_new, blocking_center, path_dir) ||
            !projectToObstacleOutside(blocking_center, path_dir, outside))
          continue;
        x_new = steer(x_nearest, outside);
        if (grid_->isOccupied(x_new) || !isSegmentFree(x_nearest->x, x_new))
          continue;
      }

      double edge_cost = edgeCostWithTurnPenalty(x_nearest, x_new);
      RRTNodeEnhanced *new_node = new RRTNodeEnhanced(x_new, x_nearest, x_nearest->cost + edge_cost, node_id_counter_++);
      x_nearest->addChild(new_node);
      nodes_.push_back(new_node);
      id_to_node_[new_node->id] = new_node;

      if (tryConnectToGoal(new_node))
      {
        if (!goal_reached)
        {
          goal_reached = true;
          post_goal_iterations = 0;
          m.time_to_first_solution_ms = std::chrono::duration<double, std::milli>(
                                            std::chrono::high_resolution_clock::now() - t0)
                                            .count();
        }
      }

      if (goal_reached)
      {
        ++post_goal_iterations;
        if (post_goal_iterations >= MIN_POST_GOAL_ITERATIONS)
          break;
      }
    }

    if (!goal_reached)
    {
      m.success = false;
      m.total_time_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t0)
                            .count();
      m.total_nodes = nodes_.size();
      m.collision_checks = collision_checks_;
      return m;
    }

    // ========== PHASE 2: kd-tree optimization ==========
    rebuildKDTree();

    int shortcut_period = 40;

    while (std::chrono::high_resolution_clock::now() < deadline && iter < max_iter * 2)
    {
      ++iter;
      if (c_best_ <= c_min_ + 1e-3)
        break;

      Eigen::Vector3d x_rand;
      if (fastRand() < 0.50 && sampleObstacleOutside(x_rand))
      {
        // obstacle-outside
      }
      else
        x_rand = sampleEllipseSafe();

      for (int d = 0; d < 3; ++d)
        x_rand(d) = std::max(workspace_min_(d), std::min(workspace_max_(d), x_rand(d)));

      if (grid_->isOccupied(x_rand))
        continue;

      RRTNodeEnhanced *x_nearest = kdTreeNN(x_rand);
      if (x_nearest == nullptr)
        continue;

      Eigen::Vector3d x_new = steer(x_nearest, x_rand);
      if (grid_->isOccupied(x_new))
      {
        Eigen::Vector3d outside;
        if (!projectToObstacleOutside(x_new, x_rand - x_nearest->x, outside))
          continue;
        x_new = steer(x_nearest, outside);
        if (grid_->isOccupied(x_new))
          continue;
      }
      if (!isSegmentFree(x_nearest->x, x_new))
      {
        Eigen::Vector3d blocking_center, path_dir, outside;
        if (!findBlockingObstacle(x_nearest->x, x_new, blocking_center, path_dir) ||
            !projectToObstacleOutside(blocking_center, path_dir, outside))
          continue;
        x_new = steer(x_nearest, outside);
        if (grid_->isOccupied(x_new) || !isSegmentFree(x_nearest->x, x_new))
          continue;
      }

      double edge_cost = edgeCostWithTurnPenalty(x_nearest, x_new);
      RRTNodeEnhanced *new_node = new RRTNodeEnhanced(x_new, x_nearest, x_nearest->cost + edge_cost, node_id_counter_++);
      x_nearest->addChild(new_node);
      nodes_.push_back(new_node);
      id_to_node_[new_node->id] = new_node;

      if (nodes_.size() % 50 == 0)
        rebuildKDTree();

      // Throttle expensive operations
      if (iter % 3 == 0)
        tryConnectToGoal(new_node);
      if (iter % 2 == 0)
        rewire(new_node);

      // Only run expensive shortcut if we have time budget remaining
      if (iter % shortcut_period == 0 &&
          std::chrono::high_resolution_clock::now() < deadline)
        tryShortcutPath();
    }

    if (std::chrono::high_resolution_clock::now() < deadline)
      tryShortcutPath();

    auto t1 = std::chrono::high_resolution_clock::now();
    m.total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    m.success = (solution_node_ != nullptr);
    m.total_nodes = nodes_.size();
    m.collision_checks = collision_checks_;

    if (m.success)
    {
      auto path = getPathVector();
      m.path_nodes = path.size();
      computePathMetrics(*grid_, path, m.path_length, m.min_clearance, m.avg_clearance, m.smoothness);
    }

    return m;
  }

  // Public accessor for the solution path (non-const due to path cache)
  std::vector<Eigen::Vector3d> getPathVector()
  {
    if (!solution_node_) return {};
    auto node_path = traceBack(solution_node_);
    std::vector<Eigen::Vector3d> path;
    for (auto *n : node_path) path.push_back(n->x);
    return path;
  }

private:
  const GridMap3D *grid_;
  Eigen::Vector3d workspace_min_, workspace_max_;
  double resolution_;
  double step_size_ = 0.2;

  Eigen::Vector3d start_, goal_;
  double c_min_, c_best_;
  RRTNodeEnhanced *solution_node_ = nullptr;
  int node_id_counter_ = 0;
  int collision_checks_ = 0;

  // Parameters
  double min_path_clearance_ = 0.25;
  double goal_bias_ = 0.15;
  double apf_sampling_ratio_ = 0.35;
  double apf_attr_gain_ = 1.0;
  double apf_rep_gain_ = 2.0;
  double apf_rep_radius_ = 1.5;

  // Tree
  std::vector<RRTNodeEnhanced *> nodes_;
  std::unordered_map<int, RRTNodeEnhanced *> id_to_node_;

  // kd-tree
  std::vector<Eigen::Vector3d> kdtree_pts_;
  std::vector<int> kdtree_idx_;
  int kd_size_ = 0;

  // Path cache
  bool path_cache_valid_ = false;
  std::vector<RRTNodeEnhanced *> path_cache_;
  std::unordered_set<int> visited_ids_;
  std::vector<int> bfs_queue_;

  // APF cache
  struct APFCacheEntry
  {
    Eigen::Vector3d pos;
    double clear;
    Eigen::Vector3d grad;
    uint64_t frame;
  };
  std::vector<APFCacheEntry> apf_cache_;
  uint64_t apf_cache_frame_ = 0;

  // Fast PRNG
  std::vector<double> rand_buf_;
  int rand_pos_ = -1;

  // Ellipse
  Eigen::Vector3d ellipse_center_;
  Eigen::Matrix3d ellipse_C_;

  inline double fastRand()
  {
    if (++rand_pos_ >= (int)rand_buf_.size())
    {
      for (auto &v : rand_buf_)
        v = (double)rand() / RAND_MAX;
      rand_pos_ = 0;
    }
    return rand_buf_[rand_pos_];
  }

  void clearTree()
  {
    for (auto *n : nodes_)
      delete n;
    nodes_.clear();
    id_to_node_.clear();
    kdtree_pts_.clear();
    kdtree_idx_.clear();
    kd_size_ = 0;
    solution_node_ = nullptr;
    path_cache_valid_ = false;
  }

  Eigen::Vector3d steer(const RRTNodeEnhanced *from, const Eigen::Vector3d &to)
  {
    Eigen::Vector3d diff = to - from->x;
    double d = diff.norm();
    if (d <= step_size_)
      return to;
    return from->x + diff / d * step_size_;
  }

  bool isSegmentFree(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2)
  {
    ++collision_checks_;
    double d = (p2 - p1).norm();
    if (d < 1e-6)
      return !grid_->isOccupied(p1) && !grid_->isOccupied(p2);
    if (grid_->isOccupied(p1) || grid_->isOccupied(p2))
      return false;
    int checks = std::max(2, (int)std::ceil(d / (resolution_ / 2.0)));
    Eigen::Vector3d step = (p2 - p1) / checks;
    for (int i = 1; i < checks; ++i)
      if (grid_->isOccupied(p1 + step * i))
        return false;
    return true;
  }

  RRTNodeEnhanced *nearestBruteForce(const Eigen::Vector3d &x)
  {
    if (nodes_.empty())
      return nullptr;
    double best_d = std::numeric_limits<double>::infinity();
    RRTNodeEnhanced *best = nullptr;
    for (auto *n : nodes_)
    {
      double d = (n->x - x).squaredNorm();
      if (d < best_d)
      {
        best_d = d;
        best = n;
      }
    }
    return best;
  }

  // ===== kd-tree =====
  void rebuildKDTree()
  {
    kd_size_ = nodes_.size();
    kdtree_pts_.resize(kd_size_);
    kdtree_idx_.resize(kd_size_);
    for (int i = 0; i < kd_size_; ++i)
    {
      kdtree_pts_[i] = nodes_[i]->x;
      kdtree_idx_[i] = i;
    }
    kdTreeBuild(0, kd_size_ - 1, 0);
  }

  int kdTreeBuild(int left, int right, int depth)
  {
    if (left > right)
      return -1;
    int mid = (left + right) / 2;
    int dim = depth % 3;
    std::nth_element(kdtree_idx_.begin() + left,
                     kdtree_idx_.begin() + mid,
                     kdtree_idx_.begin() + right + 1,
                     [dim, this](int a, int b)
                     { return kdtree_pts_[a](dim) < kdtree_pts_[b](dim); });
    kdTreeBuild(left, mid - 1, depth + 1);
    kdTreeBuild(mid + 1, right, depth + 1);
    return kdtree_idx_[mid];
  }

  void kdTreeNNRec(int left, int right, int depth, const Eigen::Vector3d &q,
                   int &best_id, double &best_dist) const
  {
    if (left > right)
      return;
    int mid = (left + right) / 2;
    int nid = kdtree_idx_[mid];
    double d = (kdtree_pts_[nid] - q).norm();
    if (d < best_dist)
    {
      best_dist = d;
      best_id = nid;
    }
    int dim = depth % 3;
    double diff = q(dim) - kdtree_pts_[nid](dim);
    if (diff <= 0)
    {
      kdTreeNNRec(left, mid - 1, depth + 1, q, best_id, best_dist);
      if (std::abs(diff) < best_dist)
        kdTreeNNRec(mid + 1, right, depth + 1, q, best_id, best_dist);
    }
    else
    {
      kdTreeNNRec(mid + 1, right, depth + 1, q, best_id, best_dist);
      if (std::abs(diff) < best_dist)
        kdTreeNNRec(left, mid - 1, depth + 1, q, best_id, best_dist);
    }
  }

  RRTNodeEnhanced *kdTreeNN(const Eigen::Vector3d &x)
  {
    if (kdtree_pts_.empty())
      return nullptr;
    int best_id = -1;
    double best_dist = std::numeric_limits<double>::infinity();
    kdTreeNNRec(0, kd_size_ - 1, 0, x, best_id, best_dist);
    if (best_id < 0 || best_id >= (int)nodes_.size())
      return nullptr;
    return nodes_[best_id];
  }

  // ===== Ellipse =====
  void computeEllipse()
  {
    c_min_ = (goal_ - start_).norm();
    if (c_best_ <= c_min_ || c_min_ < 1e-6)
    {
      ellipse_C_.setZero();
      ellipse_center_ = start_;
      return;
    }
    ellipse_center_ = (start_ + goal_) / 2.0;
    double a = c_best_ / 2.0;
    double c = c_min_ / 2.0;
    double b = std::sqrt(std::max(0.0, a * a - c * c));

    Eigen::Vector3d axis = (goal_ - start_).normalized();
    Eigen::Vector3d z_axis(0, 0, 1);
    Eigen::Vector3d v = z_axis.cross(axis);
    double s = v.norm(), c_rot = z_axis.dot(axis);

    if (s < 1e-8)
    {
      if (c_rot > 0)
        ellipse_C_.setZero();
      else
        ellipse_C_ << -b, 0, 0, 0, -b, 0, 0, 0, -a;
    }
    else
    {
      v = v / s;
      Eigen::Matrix3d K;
      K << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
      Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + s * K + (1 - c_rot) * K * K;
      Eigen::Matrix3d D = Eigen::Matrix3d::Zero();
      D(0, 0) = b;
      D(1, 1) = b;
      D(2, 2) = a;
      ellipse_C_ = R * D;
    }
  }

  Eigen::Vector3d sampleEllipse()
  {
    if (ellipse_C_.isZero(0) || c_best_ <= c_min_)
      return (start_ + goal_) / 2.0;
    Eigen::Vector3d u;
    double norm;
    do
    {
      u(0) = fastRand() * 2 - 1;
      u(1) = fastRand() * 2 - 1;
      u(2) = fastRand() * 2 - 1;
      norm = u.norm();
    } while (norm > 1.0);
    return ellipse_center_ + ellipse_C_ * u;
  }

  Eigen::Vector3d sampleEllipseSafe()
  {
    constexpr double MIN_SAFE_CLEARANCE = 0.6;
    for (int attempt = 0; attempt < 10; ++attempt)
    {
      Eigen::Vector3d x = sampleEllipse();
      for (int d = 0; d < 3; ++d)
        x(d) = std::max(workspace_min_(d), std::min(workspace_max_(d), x(d)));
      if (grid_->isOccupied(x))
        continue;
      if (computeClearance(*grid_, x) >= MIN_SAFE_CLEARANCE)
        return x;
    }
    Eigen::Vector3d outside;
    if (sampleObstacleOutside(outside))
      return outside;
    return sampleWorkspace();
  }

  // ===== Workspace sampling with clearance bias =====
  Eigen::Vector3d sampleWorkspace()
  {
    constexpr double CLEAR_TARGET = 0.48;
    for (int attempt = 0; attempt < 28; ++attempt)
    {
      Eigen::Vector3d x(workspace_min_(0) + (workspace_max_(0) - workspace_min_(0)) * fastRand(),
                        workspace_min_(1) + (workspace_max_(1) - workspace_min_(1)) * fastRand(),
                        workspace_min_(2) + (workspace_max_(2) - workspace_min_(2)) * fastRand());
      if (grid_->isOccupied(x))
        continue;
      if (computeClearance(*grid_, x) >= CLEAR_TARGET)
        return x;
    }
    return Eigen::Vector3d(workspace_min_(0) + (workspace_max_(0) - workspace_min_(0)) * fastRand(),
                           workspace_min_(1) + (workspace_max_(1) - workspace_min_(1)) * fastRand(),
                           workspace_min_(2) + (workspace_max_(2) - workspace_min_(2)) * fastRand());
  }

  // ===== Obstacle-outside sampling =====
  bool findBlockingObstacle(const Eigen::Vector3d &from, const Eigen::Vector3d &to,
                            Eigen::Vector3d &obs_center, Eigen::Vector3d &path_dir)
  {
    Eigen::Vector3d diff = to - from;
    double dist = diff.norm();
    if (dist < 1e-6)
      return false;
    path_dir = diff / dist;
    int checks = std::max(2, (int)std::ceil(dist / std::max(resolution_ * 0.5, 0.05)));

    bool in_block = false;
    int best_count = 0, curr_count = 0;
    Eigen::Vector3d curr_sum = Eigen::Vector3d::Zero(), best_sum = Eigen::Vector3d::Zero();

    for (int i = 0; i <= checks; ++i)
    {
      double t = (double)i / checks;
      Eigen::Vector3d p = from + t * diff;
      bool occ = grid_->isOccupied(p);

      if (occ)
      {
        if (!in_block)
        {
          in_block = true;
          curr_count = 0;
          curr_sum.setZero();
        }
        ++curr_count;
        curr_sum += p;
      }
      else if (in_block)
      {
        if (curr_count > best_count)
        {
          best_count = curr_count;
          best_sum = curr_sum;
        }
        in_block = false;
      }
    }
    if (in_block && curr_count > best_count)
    {
      best_count = curr_count;
      best_sum = curr_sum;
    }
    if (best_count <= 0)
      return false;
    obs_center = best_sum / best_count;
    return true;
  }

  bool projectToObstacleOutside(const Eigen::Vector3d &obs_pos, const Eigen::Vector3d &preferred_dir,
                                 Eigen::Vector3d &outside_pt)
  {
    constexpr double MIN_CLEARANCE = 0.65;
    constexpr double MAX_RADIUS = 2.8;

    std::vector<Eigen::Vector3d> dirs;
    if (preferred_dir.norm() > 1e-6)
    {
      Eigen::Vector3d d = preferred_dir.normalized();
      dirs.push_back(d);
      dirs.push_back(-d);
      Eigen::Vector3d lateral = Eigen::Vector3d(0, 0, 1).cross(d);
      if (lateral.norm() > 1e-6)
      {
        lateral.normalize();
        dirs.push_back(lateral);
        dirs.push_back(-lateral);
      }
    }
    for (int ix = -1; ix <= 1; ++ix)
      for (int iy = -1; iy <= 1; ++iy)
        for (int iz = -1; iz <= 1; ++iz)
        {
          if (ix == 0 && iy == 0 && iz == 0)
            continue;
          dirs.push_back(Eigen::Vector3d(ix, iy, iz).normalized());
        }

    for (const auto &dir_raw : dirs)
    {
      Eigen::Vector3d dir = dir_raw.normalized();
      for (double r = MIN_CLEARANCE; r <= MAX_RADIUS; r += std::max(0.15, resolution_))
      {
        Eigen::Vector3d cand = obs_pos + dir * r;
        for (int ax = 0; ax < 3; ++ax)
          cand(ax) = std::max(workspace_min_(ax), std::min(workspace_max_(ax), cand(ax)));
        if (grid_->isOccupied(cand))
          continue;
        if (computeClearance(*grid_, cand) < MIN_CLEARANCE)
          continue;
        outside_pt = cand;
        return true;
      }
    }
    return false;
  }

  bool sampleObstacleOutside(Eigen::Vector3d &x_sample)
  {
    Eigen::Vector3d obs_center, path_dir;
    if (!findBlockingObstacle(start_, goal_, obs_center, path_dir))
      return false;

    Eigen::Vector3d z_axis(0, 0, 1);
    Eigen::Vector3d lateral = z_axis.cross(path_dir);
    if (lateral.norm() < 1e-6)
      lateral = Eigen::Vector3d(1, 0, 0).cross(path_dir);
    if (lateral.norm() < 1e-6)
      return false;
    lateral.normalize();

    Eigen::Vector3d projected;
    if (projectToObstacleOutside(obs_center, lateral, projected))
    {
      RRTNodeEnhanced *nearest = nearestBruteForce(projected);
      if (nearest == nullptr || isSegmentFree(nearest->x, projected))
      {
        x_sample = projected;
        return true;
      }
    }

    Eigen::Vector3d vertical = path_dir.cross(lateral);
    if (vertical.norm() < 1e-6)
      vertical = z_axis;
    vertical.normalize();

    for (int attempt = 0; attempt < 40; ++attempt)
    {
      double side = (attempt % 2 == 0) ? 1.0 : -1.0;
      Eigen::Vector3d cand = obs_center +
                              side * lateral * (0.9 + 2.0 * fastRand()) +
                              path_dir * (fastRand() - 0.5) * std::max(1.0, 4.0 * step_size_) +
                              vertical * (fastRand() - 0.5) * 0.8;
      for (int ax = 0; ax < 3; ++ax)
        cand(ax) = std::max(workspace_min_(ax), std::min(workspace_max_(ax), cand(ax)));
      if (grid_->isOccupied(cand))
        continue;
      if (computeClearance(*grid_, cand) < 0.65)
        continue;
      RRTNodeEnhanced *nearest = nearestBruteForce(cand);
      if (nearest != nullptr && !isSegmentFree(nearest->x, cand))
        continue;
      x_sample = cand;
      return true;
    }
    return false;
  }

  // ===== APF sampling =====
  bool apfCacheLookup(const Eigen::Vector3d &pos, double &clear, Eigen::Vector3d &grad)
  {
    constexpr double CACHE_RADIUS = 0.15;
    for (const auto &e : apf_cache_)
      if (e.frame == apf_cache_frame_ && (e.pos - pos).norm() < CACHE_RADIUS)
      {
        clear = e.clear;
        grad = e.grad;
        return true;
      }
    return false;
  }

  void apfCacheStore(const Eigen::Vector3d &pos, double clear, const Eigen::Vector3d &grad)
  {
    apf_cache_.push_back({pos, clear, grad, apf_cache_frame_});
    if ((int)apf_cache_.size() > 256)
      apf_cache_.erase(apf_cache_.begin());
  }

  double attractivePotential(const Eigen::Vector3d &pos, Eigen::Vector3d &grad)
  {
    constexpr double D_STAR = 2.0;
    Eigen::Vector3d d = pos - goal_;
    double dist = d.norm();
    if (dist < D_STAR)
    {
      grad = 2.0 * apf_attr_gain_ * d;
      return apf_attr_gain_ * dist * dist;
    }
    grad = 2.0 * apf_attr_gain_ * D_STAR * d.normalized();
    return apf_attr_gain_ * D_STAR * (2.0 * dist - D_STAR);
  }

  double repulsivePotential(const Eigen::Vector3d &pos, Eigen::Vector3d &grad)
  {
    grad.setZero();
    double cached_clear;
    Eigen::Vector3d cached_grad;
    if (apfCacheLookup(pos, cached_clear, cached_grad))
    {
      if (cached_clear > apf_rep_radius_ || cached_clear < 1e-6)
        return 0.0;
      double k = 1.0 / cached_clear - 1.0 / apf_rep_radius_;
      grad = 2.0 * apf_rep_gain_ * k * cached_grad / (cached_clear * cached_clear);
      return apf_rep_gain_ * k * k;
    }

    constexpr int NUM_DIRS = 26;
    constexpr double DELTA = 0.05;
    Eigen::Vector3d dirs[NUM_DIRS] = {
        Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(-1, 0, 0),
        Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, -1, 0),
        Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, -1),
        Eigen::Vector3d(1, 1, 0), Eigen::Vector3d(1, -1, 0),
        Eigen::Vector3d(-1, 1, 0), Eigen::Vector3d(-1, -1, 0),
        Eigen::Vector3d(1, 0, 1), Eigen::Vector3d(1, 0, -1),
        Eigen::Vector3d(-1, 0, 1), Eigen::Vector3d(-1, 0, -1),
        Eigen::Vector3d(0, 1, 1), Eigen::Vector3d(0, 1, -1),
        Eigen::Vector3d(0, -1, 1), Eigen::Vector3d(0, -1, -1),
        Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(1, 1, -1),
        Eigen::Vector3d(1, -1, 1), Eigen::Vector3d(1, -1, -1),
        Eigen::Vector3d(-1, 1, 1), Eigen::Vector3d(-1, 1, -1),
        Eigen::Vector3d(-1, -1, 1), Eigen::Vector3d(-1, -1, -1)};
    for (int i = 0; i < NUM_DIRS; ++i)
      dirs[i].normalize();

    double min_dist = std::numeric_limits<double>::infinity();
    Eigen::Vector3d closest_dir = Eigen::Vector3d::Zero();

    for (int di = 0; di < NUM_DIRS; ++di)
    {
      Eigen::Vector3d p = pos;
      for (int step = 0; step < 200; ++step)
      {
        p += dirs[di] * DELTA;
        if (p(0) < workspace_min_(0) || p(0) > workspace_max_(0) ||
            p(1) < workspace_min_(1) || p(1) > workspace_max_(1) ||
            p(2) < workspace_min_(2) || p(2) > workspace_max_(2))
          break;
        if (grid_->isOccupied(p))
        {
          double d = (step + 1) * DELTA;
          if (d < min_dist)
          {
            min_dist = d;
            closest_dir = -dirs[di];
          }
          break;
        }
      }
    }

    apfCacheStore(pos, min_dist, closest_dir);

    if (min_dist > apf_rep_radius_ || min_dist < 1e-6)
      return 0.0;

    double k = 1.0 / min_dist - 1.0 / apf_rep_radius_;
    grad = 2.0 * apf_rep_gain_ * k * closest_dir / (min_dist * min_dist);
    return apf_rep_gain_ * k * k;
  }

  Eigen::Vector3d sampleWithAPF()
  {
    if (nodes_.empty())
      return sampleWorkspace();

    RRTNodeEnhanced *anchor = nodes_[rand() % nodes_.size()];
    Eigen::Vector3d anchor_pos = anchor->x;

    Eigen::Vector3d grad_att, grad_rep;
    attractivePotential(anchor_pos, grad_att);
    repulsivePotential(anchor_pos, grad_rep);

    Eigen::Vector3d dir_att = goal_ - anchor_pos;
    double da = dir_att.norm();
    if (da > 1e-6)
      dir_att /= da;

    Eigen::Vector3d dir_rep = grad_rep;
    double dr = dir_rep.norm();
    if (dr > 1e-6)
      dir_rep /= dr;

    Eigen::Vector3d dir_total = dir_att + 0.8 * dir_rep;
    double dt = dir_total.norm();
    if (dt < 1e-6)
      dir_total = dir_att;
    else
      dir_total /= dt;

    constexpr double LAMBDA_DIST = 0.5;
    double dist = -std::log(1.0 - fastRand()) / LAMBDA_DIST;
    dist = std::max(0.2, std::min(dist, 3.0));

    Eigen::Vector3d x_sample = anchor_pos + dir_total * dist;
    for (int ax = 0; ax < 3; ++ax)
      x_sample(ax) = std::max(workspace_min_(ax), std::min(workspace_max_(ax), x_sample(ax)));

    if (grid_->isOccupied(x_sample))
      return sampleWorkspace();

    return x_sample;
  }

  // ===== Edge cost with turn penalty =====
  double edgeCostWithTurnPenalty(const RRTNodeEnhanced *from, const Eigen::Vector3d &to) const
  {
    double length = (to - from->x).norm();
    if (from->parent == nullptr || length < 1e-6)
      return length;
    Eigen::Vector3d prev = from->x - from->parent->x;
    double prev_len = prev.norm();
    if (prev_len < 1e-6)
      return length;
    double cos_angle = std::max(-1.0, std::min(1.0, prev.dot(to - from->x) / (prev_len * length)));
    return length + 0.8 * length * (1.0 - cos_angle);
  }

  // ===== Goal connection (with lightweight clearance check) =====
  bool tryConnectToGoal(RRTNodeEnhanced *node)
  {
    if (grid_->isOccupied(goal_) || !isSegmentFree(node->x, goal_))
      return false;
    double seg_len = (goal_ - node->x).norm();
    double new_cost = node->cost + edgeCostWithTurnPenalty(node, goal_);
    if (new_cost >= c_best_ - 1e-9)
      return false;

    // Lightweight clearance: sample ~5 points along the segment (not 100)
    int checks = std::min(20, std::max(3, (int)std::ceil(seg_len / 1.0)));
    for (int i = 0; i <= checks; ++i)
    {
      double t = (double)i / checks;
      if (computeClearance(*grid_, node->x + t * (goal_ - node->x)) < min_path_clearance_)
        return false;
    }

    RRTNodeEnhanced *goal_node = new RRTNodeEnhanced(goal_, node, new_cost, node_id_counter_++);
    nodes_.push_back(goal_node);
    id_to_node_[goal_node->id] = goal_node;
    node->addChild(goal_node);
    solution_node_ = goal_node;
    c_best_ = new_cost;
    computeEllipse();
    path_cache_valid_ = false;
    return true;
  }

  // ===== Rewire =====
  void rewire(RRTNodeEnhanced *new_node)
  {
    double radius = std::min(1.5 * step_size_ + std::sqrt((double)nodes_.size()) * step_size_,
                             5.0 * step_size_);

    for (auto *neighbor : nodes_)
    {
      if (neighbor == new_node)
        continue;
      double d = (neighbor->x - new_node->x).norm();
      if (d > radius)
        continue;

      double new_cost = new_node->cost + edgeCostWithTurnPenalty(new_node, neighbor->x);
      if (new_cost < neighbor->cost - 1e-6 && isSegmentFree(new_node->x, neighbor->x))
      {
        if (neighbor->parent)
          neighbor->parent->removeChild(neighbor);
        neighbor->parent = new_node;
        neighbor->cost = new_cost;
        new_node->addChild(neighbor);

        path_cache_valid_ = false;
        // BFS cost propagation
        bfs_queue_.clear();
        visited_ids_.clear();
        bfs_queue_.push_back(neighbor->id);
        while (!bfs_queue_.empty())
        {
          int cur_id = bfs_queue_.back();
          bfs_queue_.pop_back();
          if (visited_ids_.count(cur_id))
            continue;
          visited_ids_.insert(cur_id);
          auto it = id_to_node_.find(cur_id);
          if (it == id_to_node_.end())
            continue;
          for (auto *child : it->second->children)
          {
            double cc = it->second->cost + edgeCostWithTurnPenalty(it->second, child->x);
            if (cc < child->cost - 1e-6)
            {
              child->cost = cc;
              bfs_queue_.push_back(child->id);
            }
          }
        }
      }
    }
  }

  // ===== Shortcut optimization =====
  void tryShortcutPath()
  {
    if (!solution_node_)
      return;

    auto current_path = traceBack(solution_node_);
    if (current_path.size() < 3)
      return;

    bool improved = true;
    int attempts = 0;
    while (improved && attempts < 100)
    {
      ++attempts;
      improved = false;

      for (size_t i = 0; i < current_path.size() - 2 && !improved; ++i)
      {
        for (size_t j = i + 2; j < current_path.size() && !improved; ++j)
        {
          if (isSegmentFree(current_path[i]->x, current_path[j]->x))
          {
            // Lightweight clearance: check sample points along the shortcut segment
            Eigen::Vector3d seg = current_path[j]->x - current_path[i]->x;
            double seg_len = seg.norm();
            int clr_samples = std::min(15, std::max(3, (int)std::ceil(seg_len / 1.5)));
            bool safe = true;
            for (int s = 1; s < clr_samples && safe; ++s)
            {
              double t = (double)s / clr_samples;
              if (computeClearance(*grid_, current_path[i]->x + t * seg) < min_path_clearance_ * 0.72)
                safe = false;
            }
            if (safe)
            {
            double new_cost = current_path[i]->cost + edgeCostWithTurnPenalty(current_path[i], current_path[j]->x);
            if (new_cost < current_path[j]->cost - 1e-4)
            {
              auto *node_j = current_path[j];
              if (node_j->parent)
                node_j->parent->removeChild(node_j);
              node_j->parent = current_path[i];
              node_j->cost = new_cost;
              current_path[i]->addChild(node_j);

              path_cache_valid_ = false;

              if (node_j == solution_node_)
              {
                // Shortcut directly reaches the goal node: update best cost
                c_best_ = new_cost;
                computeEllipse();
              }
              else if (new_cost < c_best_)
              {
                // Intermediate shortcut: goal cost updated via BFS, only refresh ellipse
                c_best_ = new_cost;
                computeEllipse();
              }
              improved = true;
            }
            }
          }
        }
      }
      if (improved)
        current_path = traceBack(solution_node_);
    }
  }

  // ===== Path tracing =====
  std::vector<RRTNodeEnhanced *> traceBack(RRTNodeEnhanced *node)
  {
    if (path_cache_valid_ && node == solution_node_ && !path_cache_.empty())
      return path_cache_;
    std::vector<RRTNodeEnhanced *> path;
    RRTNodeEnhanced *cur = node;
    while (cur)
    {
      path.push_back(cur);
      cur = cur->parent;
    }
    std::reverse(path.begin(), path.end());
    if (node == solution_node_)
    {
      path_cache_ = path;
      path_cache_valid_ = true;
    }
    return path;
  }
};

#endif // ENHANCED_RRT_H_

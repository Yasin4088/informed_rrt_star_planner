#ifndef CLASSICAL_RRT_H_
#define CLASSICAL_RRT_H_

#include "benchmark_core.h"

// ============================================================================
// Classical Informed RRT* (Gammell et al. 2014)
// ============================================================================
// Core algorithm:
//   1. Uniform / goal-biased sampling until first solution
//   2. Informed ellipse sampling after solution found
//   3. kd-tree nearest neighbor
//   4. Euclidean-distance edge cost
//   5. Basic rewire (no child-pointer optimization)
//   6. No APF, no obstacle-outside, no clearance checks, no shortcut, no smoothing
// ============================================================================

struct RRTNodeClassical
{
  Eigen::Vector3d x;
  RRTNodeClassical *parent;
  double cost;
  int id;

  RRTNodeClassical(const Eigen::Vector3d &_x, RRTNodeClassical *_parent, double _cost, int _id)
      : x(_x), parent(_parent), cost(_cost), id(_id) {}
};

class ClassicalInformedRRTstar
{
public:
  ClassicalInformedRRTstar() {}

  void init(const GridMap3D &grid)
  {
    grid_ = &grid;
    workspace_min_ = grid.min_bound;
    workspace_max_ = grid.max_bound;
    resolution_ = grid.resolution;
  }

  RunMetrics search(const Eigen::Vector3d &start, const Eigen::Vector3d &goal,
                    double step_size = 0.2, double max_time = 0.5, int max_iter = 20000)
  {
    RunMetrics m;
    auto t0 = std::chrono::high_resolution_clock::now();

    start_ = start;
    goal_ = goal;
    c_min_ = (goal_ - start_).norm();
    c_best_ = std::numeric_limits<double>::infinity();
    solution_node_ = nullptr;
    node_id_counter_ = 0;
    collision_checks_ = 0;

    clearTree();

    RRTNodeClassical *start_node = new RRTNodeClassical(start_, nullptr, 0.0, node_id_counter_++);
    nodes_.push_back(start_node);

    // Direct connection check
    if (isSegmentFree(start_, goal_))
    {
      solution_node_ = new RRTNodeClassical(goal_, start_node,
                                            start_node->cost + (goal_ - start_).norm(),
                                            node_id_counter_++);
      nodes_.push_back(solution_node_);
      c_best_ = solution_node_->cost;

      auto t1 = std::chrono::high_resolution_clock::now();
      m.success = true;
      m.total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      m.time_to_first_solution_ms = m.total_time_ms;
      m.path_length = c_best_;
      m.path_nodes = 2;
      m.total_nodes = nodes_.size();
      m.collision_checks = collision_checks_;
      computePathMetrics(*grid_, getPathVector(), m.path_length, m.min_clearance,
                         m.avg_clearance, m.smoothness);
      return m;
    }

    computeEllipse();
    auto end_time = t0 + std::chrono::duration<double>(max_time);
    bool first_solution_found = false;
    // Stop shortly after first solution so search time reflects scenario difficulty,
    // instead of always saturating the full time budget.
    constexpr int MIN_POST_GOAL_ITERATIONS = 80;
    int post_goal_iterations = 0;
    int iter = 0;

    while (std::chrono::high_resolution_clock::now() < end_time && iter < max_iter)
    {
      ++iter;

      // Sampling strategy
      Eigen::Vector3d x_rand;
      double roll = uniform01_(rng_);

      if (!first_solution_found)
      {
        // Before first solution: uniform + goal bias
        if (roll < 0.10)
          x_rand = goal_;
        else
          x_rand = sampleWorkspace();
      }
      else
      {
        // After first solution: informed ellipse + goal bias
        if (roll < 0.10)
          x_rand = goal_;
        else if (roll < 0.90)
          x_rand = sampleInEllipse();
        else
          x_rand = sampleWorkspace();
      }

      // Nearest neighbor via kd-tree (rebuild periodically)
      if (nodes_.size() % 100 == 0 || kdtree_pts_.empty())
        rebuildKDTree();

      RRTNodeClassical *x_nearest = kdTreeNN(x_rand);
      if (x_nearest == nullptr)
        continue;

      Eigen::Vector3d x_new = steer(x_nearest, x_rand);
      if (grid_->isOccupied(x_new))
        continue;
      if (!isSegmentFree(x_nearest->x, x_new))
        continue;

      double edge_cost = (x_new - x_nearest->x).norm();
      double new_cost = x_nearest->cost + edge_cost;

      // Choose parent: find best among nearby nodes
      RRTNodeClassical *best_parent = x_nearest;
      double best_cost = new_cost;
      double rewire_radius = computeRewireRadius();

      for (auto *node : nodes_)
      {
        if (node == x_nearest)
          continue;
        double d = (node->x - x_new).norm();
        if (d > rewire_radius)
          continue;
        double cand_cost = node->cost + d;
        if (cand_cost < best_cost && isSegmentFree(node->x, x_new))
        {
          best_parent = node;
          best_cost = cand_cost;
        }
      }

      RRTNodeClassical *new_node = new RRTNodeClassical(x_new, best_parent, best_cost, node_id_counter_++);
      nodes_.push_back(new_node);

      // Rewire: check if new_node can be a better parent for existing nodes
      for (auto *node : nodes_)
      {
        if (node == new_node || node == best_parent)
          continue;
        double d = (node->x - new_node->x).norm();
        if (d > rewire_radius)
          continue;
        double cand_cost = new_node->cost + d;
        if (cand_cost < node->cost && isSegmentFree(new_node->x, node->x))
        {
          node->parent = new_node;
          node->cost = cand_cost;
        }
      }

      // Try to connect to goal
      if (isSegmentFree(new_node->x, goal_))
      {
        double goal_cost = new_node->cost + (goal_ - new_node->x).norm();
        if (goal_cost < c_best_)
        {
          if (solution_node_)
            solution_node_->parent = nullptr; // old goal becomes orphaned (memory leaks but harmless for benchmark)
          solution_node_ = new RRTNodeClassical(goal_, new_node, goal_cost, node_id_counter_++);
          nodes_.push_back(solution_node_);
          c_best_ = goal_cost;
          computeEllipse();

          if (!first_solution_found)
          {
            first_solution_found = true;
            post_goal_iterations = 0;
            m.time_to_first_solution_ms = std::chrono::duration<double, std::milli>(
                                              std::chrono::high_resolution_clock::now() - t0)
                                              .count();
          }
        }
      }

      // Similar to enhanced version: keep a short optimization window
      // after first solution, then terminate early.
      if (first_solution_found)
      {
        ++post_goal_iterations;
        if (post_goal_iterations >= MIN_POST_GOAL_ITERATIONS)
          break;
      }
    }

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

  // Public accessor for the solution path
  std::vector<Eigen::Vector3d> getPathVector() const
  {
    std::vector<Eigen::Vector3d> path;
    if (!solution_node_)
      return path;
    std::vector<RRTNodeClassical *> rev;
    RRTNodeClassical *cur = solution_node_;
    while (cur)
    {
      rev.push_back(cur);
      cur = cur->parent;
    }
    for (auto it = rev.rbegin(); it != rev.rend(); ++it)
      path.push_back((*it)->x);
    return path;
  }

private:
  const GridMap3D *grid_;
  Eigen::Vector3d workspace_min_, workspace_max_;
  double resolution_;

  Eigen::Vector3d start_, goal_;
  double c_min_, c_best_;
  RRTNodeClassical *solution_node_ = nullptr;
  int node_id_counter_ = 0;
  int collision_checks_ = 0;

  std::vector<RRTNodeClassical *> nodes_;

  // kd-tree
  std::vector<Eigen::Vector3d> kdtree_pts_;
  std::vector<int> kdtree_idx_;
  int kd_size_ = 0;

  // Ellipse params
  Eigen::Vector3d ellipse_center_;
  Eigen::Matrix3d ellipse_C_;

  // RNG — seeded per-search for independent trials
  std::mt19937 rng_{std::random_device{}()};
  std::uniform_real_distribution<double> uniform01_{0.0, 1.0};

  void clearTree()
  {
    for (auto *n : nodes_)
      delete n;
    nodes_.clear();
    kdtree_pts_.clear();
    kdtree_idx_.clear();
    kd_size_ = 0;
  }

  Eigen::Vector3d sampleWorkspace()
  {
    std::uniform_real_distribution<double> rx(workspace_min_(0), workspace_max_(0));
    std::uniform_real_distribution<double> ry(workspace_min_(1), workspace_max_(1));
    std::uniform_real_distribution<double> rz(workspace_min_(2), workspace_max_(2));

    for (int attempt = 0; attempt < 50; ++attempt)
    {
      Eigen::Vector3d p(rx(rng_), ry(rng_), rz(rng_));
      if (!grid_->isOccupied(p))
        return p;
    }
    return Eigen::Vector3d(rx(rng_), ry(rng_), rz(rng_));
  }

  Eigen::Vector3d steer(const RRTNodeClassical *from, const Eigen::Vector3d &to)
  {
    Eigen::Vector3d diff = to - from->x;
    double d = diff.norm();
    const double step = 0.2;
    if (d <= step)
      return to;
    return from->x + diff / d * step;
  }

  bool isSegmentFree(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2)
  {
    ++collision_checks_;
    double d = (p2 - p1).norm();
    if (d < 1e-6)
      return !grid_->isOccupied(p1);
    int checks = std::max(2, (int)std::ceil(d / (resolution_ / 2.0)));
    Eigen::Vector3d step = (p2 - p1) / checks;
    for (int i = 0; i <= checks; ++i)
      if (grid_->isOccupied(p1 + step * i))
        return false;
    return true;
  }

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

  Eigen::Vector3d sampleInEllipse()
  {
    if (ellipse_C_.isZero(0) || c_best_ <= c_min_)
      return (start_ + goal_) / 2.0;

    for (int attempt = 0; attempt < 30; ++attempt)
    {
      Eigen::Vector3d u(uniform01_(rng_) * 2 - 1,
                        uniform01_(rng_) * 2 - 1,
                        uniform01_(rng_) * 2 - 1);
      if (u.norm() > 1.0)
        continue;
      Eigen::Vector3d p = ellipse_center_ + ellipse_C_ * u;
      for (int d = 0; d < 3; ++d)
        p(d) = std::max(workspace_min_(d), std::min(workspace_max_(d), p(d)));
      if (!grid_->isOccupied(p))
        return p;
    }
    return sampleWorkspace();
  }

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

  RRTNodeClassical *kdTreeNN(const Eigen::Vector3d &x)
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

  double computeRewireRadius() const
  {
    // Standard RRT* rewire radius: gamma * (log(n)/n)^(1/d)
    if (nodes_.size() < 2)
      return 5.0;
    double n = nodes_.size();
    double d = 3.0;
    double gamma = 2.0 * std::pow((1.0 + 1.0 / d), 1.0 / d) *
                   std::pow((workspace_max_ - workspace_min_).norm() / 2.0, 1.0 / d);
    return std::min(5.0, gamma * std::pow(std::log(n) / n, 1.0 / d));
  }
};

#endif // CLASSICAL_RRT_H_

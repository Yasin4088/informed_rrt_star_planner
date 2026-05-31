#ifndef BENCHMARK_ALL_CORE_H_
#define BENCHMARK_ALL_CORE_H_

#include "pointcloud_generator.h"
#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unordered_set>
#include <numeric>

// ============================================================================
// 3D Occupancy Grid (no ROS dependency)
// ============================================================================
struct GridMap3D
{
  double resolution;
  Eigen::Vector3d min_bound, max_bound;
  Eigen::Vector3i grid_size;
  std::vector<bool> data; // true = occupied

  void init(const Eigen::Vector3d &mn, const Eigen::Vector3d &mx, double res)
  {
    resolution = res;
    min_bound = mn;
    max_bound = mx;
    grid_size(0) = static_cast<int>(std::ceil((mx(0) - mn(0)) / res)) + 1;
    grid_size(1) = static_cast<int>(std::ceil((mx(1) - mn(1)) / res)) + 1;
    grid_size(2) = static_cast<int>(std::ceil((mx(2) - mn(2)) / res)) + 1;
    data.assign(grid_size(0) * grid_size(1) * grid_size(2), false);
  }

  inline int idx(int x, int y, int z) const
  {
    return z * grid_size(0) * grid_size(1) + y * grid_size(0) + x;
  }

  inline bool inBounds(int x, int y, int z) const
  {
    return x >= 0 && x < grid_size(0) &&
           y >= 0 && y < grid_size(1) &&
           z >= 0 && z < grid_size(2);
  }

  bool isOccupied(const Eigen::Vector3d &pos) const
  {
    int x = static_cast<int>(std::floor((pos(0) - min_bound(0)) / resolution));
    int y = static_cast<int>(std::floor((pos(1) - min_bound(1)) / resolution));
    int z = static_cast<int>(std::floor((pos(2) - min_bound(2)) / resolution));
    if (!inBounds(x, y, z)) return true;
    return data[idx(x, y, z)];
  }

  void setOccupied(const Eigen::Vector3d &pos, bool occ = true)
  {
    int x = static_cast<int>(std::floor((pos(0) - min_bound(0)) / resolution));
    int y = static_cast<int>(std::floor((pos(1) - min_bound(1)) / resolution));
    int z = static_cast<int>(std::floor((pos(2) - min_bound(2)) / resolution));
    if (inBounds(x, y, z)) data[idx(x, y, z)] = occ;
  }

  // Build occupancy grid from point cloud (simple inflation-based approach)
  // Each point votes for its voxel and neighbors within inflation_radius
  void buildFromPointCloud(const std::vector<Eigen::Vector3d> &points,
                           double inflation_radius = 0.15)
  {
    // Reset
    data.assign(data.size(), false);

    int inflate_steps = std::max(1, (int)std::ceil(inflation_radius / resolution));
    for (const auto &pt : points)
    {
      int cx = static_cast<int>(std::floor((pt(0) - min_bound(0)) / resolution));
      int cy = static_cast<int>(std::floor((pt(1) - min_bound(1)) / resolution));
      int cz = static_cast<int>(std::floor((pt(2) - min_bound(2)) / resolution));

      for (int dx = -inflate_steps; dx <= inflate_steps; ++dx)
        for (int dy = -inflate_steps; dy <= inflate_steps; ++dy)
          for (int dz = -inflate_steps; dz <= inflate_steps; ++dz)
          {
            int nx = cx + dx, ny = cy + dy, nz = cz + dz;
            if (inBounds(nx, ny, nz))
              data[idx(nx, ny, nz)] = true;
          }
    }
  }

  // Add a box obstacle
  void addBox(const Eigen::Vector3d &center, const Eigen::Vector3d &half_extents)
  {
    for (double dx = -half_extents(0); dx <= half_extents(0); dx += resolution * 0.5)
      for (double dy = -half_extents(1); dy <= half_extents(1); dy += resolution * 0.5)
        for (double dz = -half_extents(2); dz <= half_extents(2); dz += resolution * 0.5)
          setOccupied(center + Eigen::Vector3d(dx, dy, dz));
  }

  void addCylinder(const Eigen::Vector3d &center, double radius, double height)
  {
    for (double dx = -radius; dx <= radius; dx += resolution * 0.5)
      for (double dy = -radius; dy <= radius; dy += resolution * 0.5)
      {
        if (dx * dx + dy * dy > radius * radius) continue;
        for (double dz = -height / 2; dz <= height / 2; dz += resolution * 0.5)
          setOccupied(center + Eigen::Vector3d(dx, dy, dz));
      }
  }

  void addRandomPillars(int count, double min_r, double max_r,
                        double min_h, double max_h, int seed)
  {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> rx(min_bound(0) + 2.0, max_bound(0) - 2.0);
    std::uniform_real_distribution<double> ry(min_bound(1) + 2.0, max_bound(1) - 2.0);
    std::uniform_real_distribution<double> rr(min_r, max_r);
    std::uniform_real_distribution<double> rh(min_h, max_h);
    for (int i = 0; i < count; ++i)
    {
      double cx = rx(rng), cy = ry(rng), r = rr(rng), h = rh(rng);
      addCylinder(Eigen::Vector3d(cx, cy, (min_bound(2) + max_bound(2)) / 2), r, h);
    }
  }

  // Add random boxes at various sizes
  void addRandomBoxes(int count, double min_side, double max_side, int seed,
                      bool ground_contact = true)
  {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> rx(min_bound(0) + 2.0, max_bound(0) - 2.0);
    std::uniform_real_distribution<double> ry(min_bound(1) + 2.0, max_bound(1) - 2.0);
    std::uniform_real_distribution<double> rs(min_side, max_side);
    std::uniform_real_distribution<double> rh(min_side * 0.5, max_side * 1.5);

    for (int i = 0; i < count; ++i)
    {
      double sx = rs(rng), sy = rs(rng), sz = rh(rng);
      double z_center;
      if (ground_contact)
        z_center = min_bound(2) + sz / 2 + (max_bound(2) - min_bound(2)) * 0.05;
      else
        z_center = min_bound(2) + sz / 2 + (max_bound(2) - min_bound(2) - sz) *
                   std::uniform_real_distribution<double>(0.0, 1.0)(rng);
      addBox(Eigen::Vector3d(rx(rng), ry(rng), z_center),
             Eigen::Vector3d(sx, sy, sz));
    }
  }

  void blockDirectPath(const Eigen::Vector3d &start, const Eigen::Vector3d &goal,
                       int num_blocks, double block_radius)
  {
    Eigen::Vector3d dir = goal - start;
    double total_dist = dir.norm();
    if (total_dist < 1e-6) return;
    dir /= total_dist;
    for (int i = 0; i < num_blocks; ++i)
    {
      double t = (i + 1.0) / (num_blocks + 1.0);
      Eigen::Vector3d pos = start + dir * total_dist * t;
      for (double dx = -block_radius; dx <= block_radius; dx += resolution * 0.5)
        for (double dy = -block_radius; dy <= block_radius; dy += resolution * 0.5)
          for (double dz = -block_radius; dz <= block_radius; dz += resolution * 0.5)
          {
            Eigen::Vector3d p = pos + Eigen::Vector3d(dx, dy, dz);
            if (p(2) >= min_bound(2) && p(2) <= max_bound(2))
              setOccupied(p);
          }
    }
  }

  void makeCorridor(double corridor_width, double length, int seed)
  {
    std::mt19937 rng(seed);
    double mid_y = (min_bound(1) + max_bound(1)) / 2;
    double half_gap = corridor_width / 2;

    for (double x = min_bound(0); x <= max_bound(0); x += resolution * 0.5)
      for (double z = min_bound(2); z <= max_bound(2); z += resolution * 0.5)
      {
        for (double y = mid_y + half_gap; y <= max_bound(1); y += resolution * 0.5)
          setOccupied(Eigen::Vector3d(x, y, z));
        for (double y = min_bound(1); y <= mid_y - half_gap; y += resolution * 0.5)
          setOccupied(Eigen::Vector3d(x, y, z));
      }

    std::uniform_real_distribution<double> bx(min_bound(0) + 3.0, max_bound(0) - 3.0);
    std::uniform_real_distribution<double> sign(0.0, 1.0);
    int blocks = std::max(1, (int)(length / 6.0));
    for (int i = 0; i < blocks; ++i)
    {
      double cy = mid_y + (sign(rng) > 0.5 ? 1 : -1) * half_gap * 0.2;
      addBox(Eigen::Vector3d(bx(rng), cy, (min_bound(2) + max_bound(2)) / 2),
             Eigen::Vector3d(0.15, half_gap * 0.2, max_bound(2) - min_bound(2)));
    }
  }

  // Build wall with a gap at a specific Y
  void addWallWithGap(double x, double gap_y, double gap_width, double height)
  {
    for (double y = min_bound(1); y <= max_bound(1); y += resolution * 0.5)
    {
      if (std::abs(y - gap_y) < gap_width / 2) continue;
      for (double z = min_bound(2); z <= min_bound(2) + height; z += resolution * 0.5)
        setOccupied(Eigen::Vector3d(x, y, z));
    }
  }

  // Carve a collision-free tunnel along polyline waypoints.
  // Used to guarantee at least one feasible route in very hard synthetic scenes.
  void carveSafePassage(const std::vector<Eigen::Vector3d> &waypoints,
                        double radius_xy,
                        double half_height)
  {
    if (waypoints.size() < 2) return;

    auto carve_point = [&](const Eigen::Vector3d &center)
    {
      for (double dx = -radius_xy; dx <= radius_xy; dx += resolution * 0.5)
        for (double dy = -radius_xy; dy <= radius_xy; dy += resolution * 0.5)
          for (double dz = -half_height; dz <= half_height; dz += resolution * 0.5)
          {
            if (dx * dx + dy * dy > radius_xy * radius_xy) continue;
            setOccupied(center + Eigen::Vector3d(dx, dy, dz), false);
          }
    };

    for (size_t i = 1; i < waypoints.size(); ++i)
    {
      const Eigen::Vector3d &p0 = waypoints[i - 1];
      const Eigen::Vector3d &p1 = waypoints[i];
      double seg_len = (p1 - p0).norm();
      int steps = std::max(2, (int)std::ceil(seg_len / std::max(0.05, resolution * 0.5)));
      for (int s = 0; s <= steps; ++s)
      {
        double t = (double)s / steps;
        carve_point(p0 + t * (p1 - p0));
      }
    }
  }
};

// ============================================================================
// Pipeline metrics: captures performance at each stage
// ============================================================================
// RunMetrics — returned by RRT search (compatible with classical_rrt.h / enhanced_rrt.h)
struct RunMetrics
{
  double total_time_ms = 0;
  double time_to_first_solution_ms = 0;
  double path_length = 0;
  int path_nodes = 0;
  double min_clearance = std::numeric_limits<double>::infinity();
  double avg_clearance = 0;
  double smoothness = 0; // sum of turning angles (rad), smaller = smoother
  int total_nodes = 0;
  int collision_checks = 0;
  bool success = false;
};

struct RRTMetrics
{
  bool success = false;
  double time_ms = 0;
  double path_length = 0;
  int path_nodes = 0;
  int tree_nodes = 0;
  int collision_checks = 0;
  double min_clearance = 0;
  double smoothness_rad = 0;
};

struct BSplineMetrics
{
  int num_control_points = 0;
  double conversion_time_ms = 0;
  double bspline_path_length = 0;
  bool feasible_vel = true;
  bool feasible_acc = true;
  double mean_v = 0, max_vel = 0;
  double mean_acc = 0, max_acc = 0;
};

struct LBFGSMetrics
{
  double opt_time_ms = 0;
  int iterations = 0;
  double final_cost = 0;
  double initial_cost = 0;
  double jerk_cost = 0;
  double smoothness_cost = 0;
  double distance_cost = 0;
  double feasibility_cost = 0;
  int rebound_count = 0;
  bool validated = false;
};

struct TrajectoryMetrics
{
  double length = 0;
  double min_clearance = 0;
  double avg_clearance = 0;
  double smoothness_rad = 0;
  double max_curvature = 0;
  double mean_vel = 0, max_vel = 0;
  double mean_acc = 0, max_acc = 0;
};

struct PipelineResult
{
  std::string algorithm_name;
  RRTMetrics rrt;
  BSplineMetrics bspline;
  LBFGSMetrics lbfgs;
  TrajectoryMetrics traj;
  double total_time_ms = 0;
  std::vector<Eigen::Vector3d> rrt_path;
  std::vector<Eigen::Vector3d> bspline_control_points;
  std::vector<Eigen::Vector3d> optimized_trajectory;
};

// Aggregate across trials
struct AggregatePipelineMetrics
{
  std::string name;
  int trials = 0, successes = 0;

  // RRT stage
  double avg_rrt_time_ms = 0, std_rrt_time_ms = 0;
  double avg_rrt_path_len = 0, std_rrt_path_len = 0;
  double avg_rrt_nodes = 0;
  double avg_rrt_clearance = 0;
  double success_rate = 0;

  // B-spline stage
  double avg_bspline_ctrl_pts = 0;
  double avg_bspline_conv_time_ms = 0;

  // L-BFGS stage
  double avg_lbfgs_time_ms = 0, std_lbfgs_time_ms = 0;
  double avg_lbfgs_iters = 0;
  double avg_lbfgs_cost_reduction = 0; // percentage
  double avg_rebounds = 0;

  // Final trajectory
  double avg_traj_length = 0, std_traj_length = 0;
  double avg_traj_clearance = 0;
  double avg_traj_smoothness = 0;
  double avg_max_vel = 0, avg_max_acc = 0;

  // End-to-end
  double avg_total_time_ms = 0;

  void compute(const std::vector<PipelineResult> &results)
  {
    trials = results.size();
    successes = 0;
    std::vector<double> rrt_t, rrt_l, rrt_n, rrt_c;
    std::vector<double> bs_cp, bs_ct;
    std::vector<double> lb_t, lb_i, lb_cr, lb_rb;
    std::vector<double> tr_l, tr_c, tr_s, tr_v, tr_a;
    std::vector<double> total_t;

    for (const auto &r : results)
    {
      if (!r.rrt.success) continue;
      successes++;
      rrt_t.push_back(r.rrt.time_ms);
      rrt_l.push_back(r.rrt.path_length);
      rrt_n.push_back(r.rrt.tree_nodes);
      rrt_c.push_back(r.rrt.min_clearance);
      bs_cp.push_back(r.bspline.num_control_points);
      bs_ct.push_back(r.bspline.conversion_time_ms);
      lb_t.push_back(r.lbfgs.opt_time_ms);
      lb_i.push_back(r.lbfgs.iterations);
      if (r.lbfgs.initial_cost > 0)
        lb_cr.push_back((r.lbfgs.initial_cost - r.lbfgs.final_cost) / r.lbfgs.initial_cost * 100.0);
      lb_rb.push_back(r.lbfgs.rebound_count);
      tr_l.push_back(r.traj.length);
      tr_c.push_back(r.traj.min_clearance);
      tr_s.push_back(r.traj.smoothness_rad);
      tr_v.push_back(r.traj.max_vel);
      tr_a.push_back(r.traj.max_acc);
      total_t.push_back(r.total_time_ms);
    }

    success_rate = (double)successes / trials;

    auto stats = [](const std::vector<double> &v, double &mean, double &stddev)
    {
      if (v.empty()) { mean = stddev = 0; return; }
      mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
      stddev = 0;
      for (double d : v) stddev += (d - mean) * (d - mean);
      stddev = std::sqrt(stddev / v.size());
    };
    auto avg = [](const std::vector<double> &v) -> double
    {
      if (v.empty()) return 0;
      return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    stats(rrt_t, avg_rrt_time_ms, std_rrt_time_ms);
    stats(rrt_l, avg_rrt_path_len, std_rrt_path_len);
    avg_rrt_nodes = avg(rrt_n);
    avg_rrt_clearance = avg(rrt_c);
    avg_bspline_ctrl_pts = avg(bs_cp);
    avg_bspline_conv_time_ms = avg(bs_ct);
    stats(lb_t, avg_lbfgs_time_ms, std_lbfgs_time_ms);
    avg_lbfgs_iters = avg(lb_i);
    avg_lbfgs_cost_reduction = avg(lb_cr);
    avg_rebounds = avg(lb_rb);
    stats(tr_l, avg_traj_length, std_traj_length);
    avg_traj_clearance = avg(tr_c);
    avg_traj_smoothness = avg(tr_s);
    avg_max_vel = avg(tr_v);
    avg_max_acc = avg(tr_a);
    avg_total_time_ms = avg(total_t);
  }
};

// ============================================================================
// Scenario definition
// ============================================================================
struct Scenario
{
  std::string name;
  Eigen::Vector3d start, goal;
  GridMap3D grid_map;
  int random_seed;
  std::string description;
};

// ============================================================================
// Clearance computation (26-direction ray-march)
// ============================================================================
inline double computeClearance(const GridMap3D &grid, const Eigen::Vector3d &pos)
{
  if (grid.isOccupied(pos)) return 0.0;

  constexpr int NUM_DIRS = 26;
  constexpr double DELTA = 0.05;
  Eigen::Vector3d dirs[NUM_DIRS] = {
      Eigen::Vector3d(1,0,0), Eigen::Vector3d(-1,0,0),
      Eigen::Vector3d(0,1,0), Eigen::Vector3d(0,-1,0),
      Eigen::Vector3d(0,0,1), Eigen::Vector3d(0,0,-1),
      Eigen::Vector3d(1,1,0), Eigen::Vector3d(1,-1,0),
      Eigen::Vector3d(-1,1,0), Eigen::Vector3d(-1,-1,0),
      Eigen::Vector3d(1,0,1), Eigen::Vector3d(1,0,-1),
      Eigen::Vector3d(-1,0,1), Eigen::Vector3d(-1,0,-1),
      Eigen::Vector3d(0,1,1), Eigen::Vector3d(0,1,-1),
      Eigen::Vector3d(0,-1,1), Eigen::Vector3d(0,-1,-1),
      Eigen::Vector3d(1,1,1), Eigen::Vector3d(1,1,-1),
      Eigen::Vector3d(1,-1,1), Eigen::Vector3d(1,-1,-1),
      Eigen::Vector3d(-1,1,1), Eigen::Vector3d(-1,1,-1),
      Eigen::Vector3d(-1,-1,1), Eigen::Vector3d(-1,-1,-1)};
  for (int i = 0; i < NUM_DIRS; ++i) dirs[i].normalize();

  double min_dist = std::numeric_limits<double>::infinity();
  for (int di = 0; di < NUM_DIRS; ++di)
  {
    Eigen::Vector3d p = pos;
    for (int step = 0; step < 200; ++step)
    {
      p += dirs[di] * DELTA;
      if (p(0) < grid.min_bound(0) || p(0) > grid.max_bound(0) ||
          p(1) < grid.min_bound(1) || p(1) > grid.max_bound(1) ||
          p(2) < grid.min_bound(2) || p(2) > grid.max_bound(2))
        break;
      if (grid.isOccupied(p))
      {
        double d = (step + 1) * DELTA;
        if (d < min_dist) min_dist = d;
        break;
      }
    }
  }
  return min_dist;
}

// Find direction to nearest obstacle (used for optimization gradient)
inline Eigen::Vector3d clearanceGradient(const GridMap3D &grid, const Eigen::Vector3d &pos)
{
  constexpr int NUM_DIRS = 26;
  constexpr double DELTA = 0.05;
  Eigen::Vector3d dirs[NUM_DIRS] = {
      Eigen::Vector3d(1,0,0), Eigen::Vector3d(-1,0,0),
      Eigen::Vector3d(0,1,0), Eigen::Vector3d(0,-1,0),
      Eigen::Vector3d(0,0,1), Eigen::Vector3d(0,0,-1),
      Eigen::Vector3d(1,1,0), Eigen::Vector3d(1,-1,0),
      Eigen::Vector3d(-1,1,0), Eigen::Vector3d(-1,-1,0),
      Eigen::Vector3d(1,0,1), Eigen::Vector3d(1,0,-1),
      Eigen::Vector3d(-1,0,1), Eigen::Vector3d(-1,0,-1),
      Eigen::Vector3d(0,1,1), Eigen::Vector3d(0,1,-1),
      Eigen::Vector3d(0,-1,1), Eigen::Vector3d(0,-1,-1),
      Eigen::Vector3d(1,1,1), Eigen::Vector3d(1,1,-1),
      Eigen::Vector3d(1,-1,1), Eigen::Vector3d(1,-1,-1),
      Eigen::Vector3d(-1,1,1), Eigen::Vector3d(-1,1,-1),
      Eigen::Vector3d(-1,-1,1), Eigen::Vector3d(-1,-1,-1)};
  for (int i = 0; i < NUM_DIRS; ++i) dirs[i].normalize();

  double min_dist = std::numeric_limits<double>::infinity();
  int best_di = -1;
  for (int di = 0; di < NUM_DIRS; ++di)
  {
    Eigen::Vector3d p = pos;
    for (int step = 0; step < 200; ++step)
    {
      p += dirs[di] * DELTA;
      if (p(0) < grid.min_bound(0) || p(0) > grid.max_bound(0) ||
          p(1) < grid.min_bound(1) || p(1) > grid.max_bound(1) ||
          p(2) < grid.min_bound(2) || p(2) > grid.max_bound(2))
        break;
      if (grid.isOccupied(p))
      {
        double d = (step + 1) * DELTA;
        if (d < min_dist) { min_dist = d; best_di = di; }
        break;
      }
    }
  }
  if (best_di < 0) return Eigen::Vector3d(0, 0, 1);
  return -dirs[best_di]; // direction AWAY from nearest obstacle
}

inline double computePathClearance(const GridMap3D &grid, const Eigen::Vector3d &p1,
                                    const Eigen::Vector3d &p2)
{
  double seg_len = (p2 - p1).norm();
  if (seg_len < 1e-6) return computeClearance(grid, p1);
  int checks = std::min(100, std::max(2, (int)std::ceil(seg_len / 0.05)));
  double min_c = std::numeric_limits<double>::infinity();
  for (int i = 0; i <= checks; ++i)
  {
    double t = (double)i / checks;
    min_c = std::min(min_c, computeClearance(grid, p1 + t * (p2 - p1)));
  }
  return min_c;
}

inline void computePathMetrics(const GridMap3D &grid,
                                const std::vector<Eigen::Vector3d> &path,
                                double &total_length, double &min_clearance,
                                double &avg_clearance, double &smoothness)
{
  total_length = 0;
  min_clearance = std::numeric_limits<double>::infinity();
  avg_clearance = 0;
  smoothness = 0;
  if (path.size() < 2) return;

  for (size_t i = 1; i < path.size(); ++i)
  {
    double seg = (path[i] - path[i - 1]).norm();
    total_length += seg;
    min_clearance = std::min(min_clearance, computePathClearance(grid, path[i - 1], path[i]));
    avg_clearance += computePathClearance(grid, path[i - 1], path[i]);
  }
  avg_clearance /= (path.size() - 1);

  for (size_t i = 1; i < path.size() - 1; ++i)
  {
    Eigen::Vector3d v1 = path[i] - path[i - 1];
    Eigen::Vector3d v2 = path[i + 1] - path[i];
    double n1 = v1.norm(), n2 = v2.norm();
    if (n1 > 1e-6 && n2 > 1e-6)
    {
      double cos_angle = std::max(-1.0, std::min(1.0, v1.dot(v2) / (n1 * n2)));
      smoothness += std::acos(cos_angle);
    }
  }
}

// ============================================================================
// Scenario generators — active benchmark scenes
// ============================================================================

// 1. Sparse — baseline easy case
inline Scenario makeSparseObstacles(int seed)
{
  Scenario s;
  s.name = "1_Sparse_Pillars";
  s.description = "10 randomly placed pillars, 30x20x3m workspace";
  s.random_seed = seed;
  s.grid_map.init(Eigen::Vector3d(0, -10, 0), Eigen::Vector3d(30, 10, 3), 0.2);
  s.start = Eigen::Vector3d(1, 0, 1.5);
  s.goal = Eigen::Vector3d(28, 0, 1.5);
  s.grid_map.blockDirectPath(s.start, s.goal, 4, 0.8);
  s.grid_map.addRandomPillars(10, 0.3, 0.8, 3.0, 3.0, seed);
  return s;
}

// 2. Dense — high obstacle density
inline Scenario makeDenseObstacles(int seed)
{
  Scenario s;
  s.name = "2_Dense_Clutter";
  s.description = "40 pillars in 30x20x3m, only narrow gaps remain";
  s.random_seed = seed;
  s.grid_map.init(Eigen::Vector3d(0, -10, 0), Eigen::Vector3d(30, 10, 3), 0.2);
  s.start = Eigen::Vector3d(1, 0, 1.5);
  s.goal = Eigen::Vector3d(28, 0, 1.5);
  s.grid_map.blockDirectPath(s.start, s.goal, 5, 0.9);
  s.grid_map.addRandomPillars(40, 0.3, 0.7, 3.0, 3.0, seed);
  return s;
}

// 3. Maze — multiple turns, tests global reasoning
inline Scenario makeMaze(int seed)
{
  Scenario s;
  s.name = "3_Maze";
  s.description = "5 walls with staggered wide gaps; guaranteed zigzag passage";
  s.random_seed = seed;
  s.grid_map.init(Eigen::Vector3d(0, -10, 0), Eigen::Vector3d(30, 10, 3), 0.2);

  // Wider wall gaps keep the scene challenging but avoid accidental full blockage
  // after point-cloud reconstruction + inflation.
  double gaps[5] = {2.0, -2.8, 3.2, -2.0, 2.6};
  for (int i = 0; i < 5; ++i)
  {
    double x = 4 + i * 5.0;
    s.grid_map.addWallWithGap(x, gaps[i], 2.4, 3.0);
  }
  s.start = Eigen::Vector3d(1, 0, 1.5);
  s.goal = Eigen::Vector3d(28, 0, 1.5);

  // Explicitly carve a zigzag corridor crossing all wall gaps.
  std::vector<Eigen::Vector3d> guide = {
      s.start,
      Eigen::Vector3d(4.0,  2.0, 1.5),
      Eigen::Vector3d(9.0, -2.8, 1.5),
      Eigen::Vector3d(14.0, 3.2, 1.5),
      Eigen::Vector3d(19.0, -2.0, 1.5),
      Eigen::Vector3d(24.0, 2.6, 1.5),
      s.goal};
  s.grid_map.carveSafePassage(guide, 0.70, 0.70);

  return s;
}

// 4. Long Range — tests scaling
inline Scenario makeLongRange(int seed)
{
  Scenario s;
  s.name = "4_Long_Range";
  s.description = "50m planning distance with 25 pillars";
  s.random_seed = seed;
  s.grid_map.init(Eigen::Vector3d(0, -10, 0), Eigen::Vector3d(50, 10, 3), 0.2);
  s.start = Eigen::Vector3d(1, 0, 1.5);
  s.goal = Eigen::Vector3d(48, 0, 1.5);
  s.grid_map.blockDirectPath(s.start, s.goal, 7, 0.9);
  s.grid_map.addRandomPillars(25, 0.3, 0.8, 3.0, 3.0, seed);
  return s;
}

// 5. 3D Multi-Level — vertical maneuvering required
inline Scenario makeMultiLevel(int seed)
{
  Scenario s;
  s.name = "5_Multi_Level";
  s.description = "Platforms at different heights requiring 3D path (30x20x6m)";
  s.random_seed = seed;
  s.grid_map.init(Eigen::Vector3d(0, -10, 0), Eigen::Vector3d(30, 10, 6), 0.2);
  s.start = Eigen::Vector3d(1, -8, 0.5);
  s.goal = Eigen::Vector3d(28, 8, 5.0);

  // Low platform
  s.grid_map.addBox(Eigen::Vector3d(8, -4, 1.0), Eigen::Vector3d(3, 3, 0.15));
  // Mid platform
  s.grid_map.addBox(Eigen::Vector3d(15, 4, 2.5), Eigen::Vector3d(3, 3, 0.15));
  // High platform
  s.grid_map.addBox(Eigen::Vector3d(22, -3, 4.0), Eigen::Vector3d(2, 4, 0.15));
  // Blocking pillars between levels
  s.grid_map.addCylinder(Eigen::Vector3d(10, 0, 3), 0.4, 6.0);
  s.grid_map.addCylinder(Eigen::Vector3d(18, 0, 3), 0.4, 6.0);
  s.grid_map.addRandomPillars(8, 0.2, 0.5, 6.0, 6.0, seed);
  return s;
}

// 6. Forest — many thin obstacles at varying heights
inline Scenario makeForest(int seed)
{
  Scenario s;
  s.name = "6_Forest";
  s.description = "50 thin tree-like pillars at varying heights, 40x20x5m";
  s.random_seed = seed;
  s.grid_map.init(Eigen::Vector3d(0, -10, 0), Eigen::Vector3d(40, 10, 5), 0.2);
  s.start = Eigen::Vector3d(1, 0, 2.0);
  s.goal = Eigen::Vector3d(38, 0, 2.0);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> rx(2, 38);
  std::uniform_real_distribution<double> ry(-9, 9);
  std::uniform_real_distribution<double> rh(1.0, 5.0);
  std::uniform_real_distribution<double> rr(0.08, 0.2);

  for (int i = 0; i < 50; ++i)
    s.grid_map.addCylinder(Eigen::Vector3d(rx(rng), ry(rng), rh(rng) / 2), rr(rng), rh(rng));

  s.grid_map.blockDirectPath(s.start, s.goal, 6, 0.7);
  return s;
}

// 7. Urban Canyon — building-like blocks with street passages
inline Scenario makeUrbanCanyon(int seed)
{
  Scenario s;
  s.name = "7_Urban_Canyon";
  s.description = "Building blocks forming canyon-like passages, 40x20x8m";
  s.random_seed = seed;
  s.grid_map.init(Eigen::Vector3d(0, -10, 0), Eigen::Vector3d(40, 10, 8), 0.2);
  s.start = Eigen::Vector3d(1, 0, 3.0);
  s.goal = Eigen::Vector3d(38, 0, 3.0);

  std::mt19937 rng(seed);
  // Two rows of "buildings" with a street in between
  for (int i = 0; i < 6; ++i)
  {
    double x = 3 + i * 6.0;
    double h = 3.0 + 4.0 * std::abs(std::sin(i * 1.2));
    // North buildings
    s.grid_map.addBox(Eigen::Vector3d(x, 5.5, h / 2), Eigen::Vector3d(2.0, 3.5, h / 2));
    // South buildings
    s.grid_map.addBox(Eigen::Vector3d(x + 1.5, -5.5, h / 2 + 0.5), Eigen::Vector3d(2.0, 3.5, h / 2 + 0.5));
    // Some cross-building connections (bridges)
    if (i == 2 || i == 4)
      s.grid_map.addBox(Eigen::Vector3d(x, 0, 5.0), Eigen::Vector3d(0.4, 2.0, 1.5));
  }
  s.grid_map.addRandomPillars(10, 0.2, 0.4, 8.0, 8.0, seed);
  return s;
}

// 8. Random Field — high-entropy unstructured environment
inline Scenario makeRandomField(int seed)
{
  Scenario s;
  s.name = "8_Random_Field";
  s.description = "44 random obstacles with carved S-curve safe passage, 35x20x5m";
  s.random_seed = seed;
  s.grid_map.init(Eigen::Vector3d(0, -10, 0), Eigen::Vector3d(35, 10, 5), 0.2);
  s.start = Eigen::Vector3d(1, 0, 2.5);
  s.goal = Eigen::Vector3d(33, 0, 2.5);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> rx(2, 33);
  std::uniform_real_distribution<double> ry(-9, 9);
  std::uniform_real_distribution<double> rz(0, 5);
  std::uniform_real_distribution<double> rs(0.2, 1.0);

  // Mix of boxes and cylinders (reduced density to avoid fully disconnected free space)
  for (int i = 0; i < 22; ++i)
    s.grid_map.addBox(Eigen::Vector3d(rx(rng), ry(rng), rz(rng)),
                      Eigen::Vector3d(rs(rng), rs(rng), rs(rng) * 2));
  for (int i = 0; i < 22; ++i)
    s.grid_map.addCylinder(Eigen::Vector3d(rx(rng), ry(rng), rz(rng)),
                           rs(rng) * 0.6, rs(rng) * 3);

  s.grid_map.blockDirectPath(s.start, s.goal, 5, 0.9);

  // Keep randomness, but guarantee one non-trivial S-curve route for planner convergence.
  std::vector<Eigen::Vector3d> guide = {
      s.start,
      Eigen::Vector3d(8.0, -2.2, 2.4),
      Eigen::Vector3d(15.0, 2.0, 2.3),
      Eigen::Vector3d(23.0, -1.8, 2.7),
      s.goal};
  s.grid_map.carveSafePassage(guide, 0.85, 0.90);

  return s;
}

// ============================================================================
// CSV output for full pipeline results
// ============================================================================
inline void writePipelineCSV(const std::string &filename,
                              const std::vector<std::string> &scenario_names,
                              const std::vector<AggregatePipelineMetrics> &classical,
                              const std::vector<AggregatePipelineMetrics> &enhanced)
{
  std::ofstream f(filename);
  f << "Scenario,Algorithm,SuccessRate(%),RRT_FirstSolutionTime(ms),RRT_StdTime(ms),"
    << "RRT_PathLen(m),RRT_StdPathLen(m),RRT_Nodes,RRT_Clearance(m),"
    << "BSpline_CtrlPts,BSpline_ConvTime(ms),"
    << "LBFGS_Time(ms),LBFGS_StdTime(ms),LBFGS_Iters,LBFGS_CostReduction(%),LBFGS_Rebounds,"
    << "Traj_Length(m),Traj_StdLength(m),Traj_Clearance(m),Traj_Smoothness(rad),"
    << "Traj_MaxVel(m/s),Traj_MaxAcc(m/s2),"
    << "Total_Time(ms)\n";

  for (size_t i = 0; i < scenario_names.size(); ++i)
  {
    auto writeRow = [&f](const std::string &s, const std::string &a,
                          const AggregatePipelineMetrics &m)
    {
      f << s << "," << a << ","
        << std::fixed << std::setprecision(1) << m.success_rate * 100 << ","
        << std::setprecision(2) << m.avg_rrt_time_ms << "," << m.std_rrt_time_ms << ","
        << std::setprecision(3) << m.avg_rrt_path_len << "," << m.std_rrt_path_len << ","
        << std::setprecision(1) << m.avg_rrt_nodes << ","
        << std::setprecision(3) << m.avg_rrt_clearance << ","
        << std::setprecision(1) << m.avg_bspline_ctrl_pts << ","
        << std::setprecision(2) << m.avg_bspline_conv_time_ms << ","
        << std::setprecision(2) << m.avg_lbfgs_time_ms << "," << m.std_lbfgs_time_ms << ","
        << std::setprecision(1) << m.avg_lbfgs_iters << ","
        << std::setprecision(1) << m.avg_lbfgs_cost_reduction << ","
        << std::setprecision(1) << m.avg_rebounds << ","
        << std::setprecision(3) << m.avg_traj_length << "," << m.std_traj_length << ","
        << std::setprecision(3) << m.avg_traj_clearance << ","
        << std::setprecision(3) << m.avg_traj_smoothness << ","
        << std::setprecision(2) << m.avg_max_vel << ","
        << std::setprecision(2) << m.avg_max_acc << ","
        << std::setprecision(2) << m.avg_total_time_ms << "\n";
    };
    writeRow(scenario_names[i], "Classical", classical[i]);
    writeRow(scenario_names[i], "Enhanced", enhanced[i]);
  }
  f.close();
  std::cout << "\nResults written to " << filename << "\n";
}

// Also write per-stage detailed CSV for trajectory visualization
inline void writeTrajectoryCSV(const std::string &filename,
                                const std::vector<PipelineResult> &results,
                                int sample_every = 1)
{
  std::ofstream f(filename);
  f << "trial,algorithm,stage,x,y,z\n";
  for (size_t t = 0; t < results.size(); t += sample_every)
  {
    const auto &r = results[t];
    // RRT path
    for (const auto &pt : r.rrt_path)
      f << t << "," << r.algorithm_name << ",rrt," << pt(0) << "," << pt(1) << "," << pt(2) << "\n";
    // B-spline control points
    for (const auto &pt : r.bspline_control_points)
      f << t << "," << r.algorithm_name << ",bspline_ctrl," << pt(0) << "," << pt(1) << "," << pt(2) << "\n";
    // Optimized trajectory
    for (const auto &pt : r.optimized_trajectory)
      f << t << "," << r.algorithm_name << ",optimized," << pt(0) << "," << pt(1) << "," << pt(2) << "\n";
  }
  f.close();
}

// Write per-trial raw metrics for statistical analysis and boxplots
inline void writeTrialCSV(const std::string &filename,
                          const std::string &scenario_name,
                          const std::vector<PipelineResult> &classical_results,
                          const std::vector<PipelineResult> &enhanced_results)
{
  std::ofstream f(filename);
  f << "Scenario,Trial,Algorithm,Success,"
    << "RRT_FirstSolutionTime(ms),RRT_PathLen(m),RRT_Nodes,RRT_CollisionChecks,RRT_Clearance(m),"
    << "BSpline_CtrlPts,BSpline_ConvTime(ms),"
    << "LBFGS_Time(ms),LBFGS_Iters,LBFGS_InitialCost,LBFGS_FinalCost,LBFGS_CostReduction(%),"
    << "Traj_Length(m),Traj_Clearance(m),Traj_Smoothness(rad),Traj_MaxVel(m/s),Traj_MaxAcc(m/s2),"
    << "Total_Time(ms)\n";

  auto write_rows = [&f, &scenario_name](const std::vector<PipelineResult> &results,
                                         const std::string &algo_name)
  {
    for (size_t i = 0; i < results.size(); ++i)
    {
      const auto &r = results[i];
      double cost_reduction = 0.0;
      if (r.lbfgs.initial_cost > 1e-9)
        cost_reduction = (r.lbfgs.initial_cost - r.lbfgs.final_cost) /
                         r.lbfgs.initial_cost * 100.0;

      f << scenario_name << "," << i << "," << algo_name << ","
        << (r.rrt.success ? 1 : 0) << ","
        << std::fixed << std::setprecision(3)
        << r.rrt.time_ms << ","
        << r.rrt.path_length << ","
        << r.rrt.tree_nodes << ","
        << r.rrt.collision_checks << ","
        << r.rrt.min_clearance << ","
        << r.bspline.num_control_points << ","
        << r.bspline.conversion_time_ms << ","
        << r.lbfgs.opt_time_ms << ","
        << r.lbfgs.iterations << ","
        << r.lbfgs.initial_cost << ","
        << r.lbfgs.final_cost << ","
        << cost_reduction << ","
        << r.traj.length << ","
        << r.traj.min_clearance << ","
        << r.traj.smoothness_rad << ","
        << r.traj.max_vel << ","
        << r.traj.max_acc << ","
        << r.total_time_ms << "\n";
    }
  };

  write_rows(classical_results, "Classical");
  write_rows(enhanced_results, "Enhanced");
  f.close();
}

#endif // BENCHMARK_ALL_CORE_H_

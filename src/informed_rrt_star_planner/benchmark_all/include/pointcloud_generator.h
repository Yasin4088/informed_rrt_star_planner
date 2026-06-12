#ifndef POINTCLOUD_GENERATOR_H_
#define POINTCLOUD_GENERATOR_H_

#include <Eigen/Eigen>
#include <vector>
#include <random>
#include <cmath>
#include <functional>

// ============================================================================
// Simulated depth camera for generating point clouds from a known 3D grid
// ============================================================================
// Models a forward-facing depth camera (e.g. Intel RealSense) mounted on a
// drone. Casts rays into the occupancy grid and returns 3D points where rays
// hit occupied voxels, with optional Gaussian noise to simulate real sensors.

struct CameraParams
{
  int width = 640;
  int height = 480;
  double fx = 320.0;
  double fy = 320.0;
  double cx = 320.0;
  double cy = 240.0;
  double min_depth = 0.3;
  double max_depth = 15.0;
  double noise_stddev = 0.02;
  int subsample = 4;
};

struct PointCloudGenerator
{
  CameraParams params;

  // Use std::function to support lambdas with captures
  using OccupancyFunc = std::function<bool(const Eigen::Vector3d &)>;

  void generate(const Eigen::Vector3d &camera_pos,
                const Eigen::Matrix3d &camera_rot,
                const OccupancyFunc &is_occupied,
                std::vector<Eigen::Vector3d> &out_points,
                std::vector<Eigen::Vector3d> &out_origins)
  {
    out_points.clear();
    out_origins.clear();

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, params.noise_stddev);

    int est_points = (params.width / params.subsample) * (params.height / params.subsample);
    out_points.reserve(est_points);
    out_origins.reserve(est_points);

    for (int v = 0; v < params.height; v += params.subsample)
    {
      for (int u = 0; u < params.width; u += params.subsample)
      {
        Eigen::Vector3d ray_dir((u - params.cx) / params.fx,
                                (v - params.cy) / params.fy,
                                1.0);
        ray_dir.normalize();

        Eigen::Vector3d world_dir = camera_rot * ray_dir;

        double hit_depth = -1.0;
        for (double d = params.min_depth; d <= params.max_depth; d += 0.05)
        {
          Eigen::Vector3d pt = camera_pos + world_dir * d;
          if (is_occupied(pt))
          {
            hit_depth = d;
            break;
          }
        }

        if (hit_depth > 0)
        {
          double noisy_depth = hit_depth + noise(rng);
          noisy_depth = std::max(params.min_depth, std::min(params.max_depth, noisy_depth));
          Eigen::Vector3d pt = camera_pos + world_dir * noisy_depth;
          out_points.push_back(pt);
          out_origins.push_back(camera_pos);
        }
      }
    }
  }

  void generateMultiView(const std::vector<Eigen::Vector3d> &view_positions,
                         const Eigen::Vector3d &look_at,
                         const OccupancyFunc &is_occupied,
                         std::vector<Eigen::Vector3d> &out_points)
  {
    out_points.clear();
    std::vector<Eigen::Vector3d> pts, origins;
    for (const auto &pos : view_positions)
    {
      Eigen::Vector3d forward = (look_at - pos).normalized();
      Eigen::Vector3d world_up(0, 0, 1);
      Eigen::Vector3d right = forward.cross(world_up);
      if (right.norm() < 1e-6) right = Eigen::Vector3d(1, 0, 0);
      else right.normalize();
      Eigen::Vector3d up = right.cross(forward).normalized();
      Eigen::Matrix3d R;
      R.col(0) = right; R.col(1) = up; R.col(2) = forward;
      generate(pos, R, is_occupied, pts, origins);
      for (const auto &p : pts) out_points.push_back(p);
    }
  }

  void generateTopDown(const Eigen::Vector3d &camera_pos,
                       const OccupancyFunc &is_occupied,
                       std::vector<Eigen::Vector3d> &out_points)
  {
    Eigen::Matrix3d R;
    R.col(0) = Eigen::Vector3d(1, 0, 0);
    R.col(1) = Eigen::Vector3d(0, -1, 0);
    R.col(2) = Eigen::Vector3d(0, 0, -1);
    std::vector<Eigen::Vector3d> origins;
    generate(camera_pos, R, is_occupied, out_points, origins);
  }

  void generateLidarScan(const Eigen::Vector3d &sensor_pos,
                         int num_rings,
                         int points_per_ring,
                         const OccupancyFunc &is_occupied,
                         std::vector<Eigen::Vector3d> &out_points)
  {
    out_points.clear();
    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, params.noise_stddev);

    for (int ring = 0; ring < num_rings; ++ring)
    {
      double v_angle = M_PI * 0.35 - M_PI * 0.7 * ring / (num_rings - 1);
      double cos_v = std::cos(v_angle), sin_v = std::sin(v_angle);
      for (int i = 0; i < points_per_ring; ++i)
      {
        double h_angle = 2.0 * M_PI * i / points_per_ring;
        Eigen::Vector3d dir(cos_v * std::cos(h_angle),
                            cos_v * std::sin(h_angle),
                            sin_v);
        double hit_dist = -1.0;
        for (double d = params.min_depth; d <= params.max_depth; d += 0.05)
        {
          Eigen::Vector3d pt = sensor_pos + dir * d;
          if (is_occupied(pt)) { hit_dist = d; break; }
        }
        if (hit_dist > 0)
        {
          double noisy_d = hit_dist + noise(rng);
          noisy_d = std::max(params.min_depth, std::min(params.max_depth, noisy_d));
          out_points.push_back(sensor_pos + dir * noisy_d);
        }
      }
    }
  }
};

#endif // POINTCLOUD_GENERATOR_H_

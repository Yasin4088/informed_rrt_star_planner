#include "benchmark_core.h"
#include "classical_rrt.h"
#include "enhanced_rrt.h"
#include "bspline_optimizer_standalone.h"
#include "pointcloud_generator.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <sys/stat.h>

// ============================================================================
// Run full pipeline for one RRT result through B-spline + L-BFGS
// ============================================================================
PipelineResult runPipelineStage(const std::vector<Eigen::Vector3d> &rrt_path,
                                 const RunMetrics &rrt_run,
                                 bool use_enhanced,
                                 const GridMap3D &grid_map,
                                 BSplineOptimizer &bspline_opt)
{
  PipelineResult pr;
  pr.algorithm_name = use_enhanced ? "Enhanced" : "Classical";

  // Use first-solution time as the comparison metric for search efficiency.
  // If first solution time is unavailable, fallback to total search time.
  double rrt_time_for_comparison = rrt_run.total_time_ms;
  if (rrt_run.time_to_first_solution_ms > 1e-6)
    rrt_time_for_comparison = rrt_run.time_to_first_solution_ms;

  pr.rrt.success = rrt_run.success;
  pr.rrt.time_ms = rrt_time_for_comparison;
  pr.rrt.path_length = rrt_run.path_length;
  pr.rrt.path_nodes = rrt_run.path_nodes;
  pr.rrt.tree_nodes = rrt_run.total_nodes;
  pr.rrt.collision_checks = rrt_run.collision_checks;
  pr.rrt.min_clearance = rrt_run.min_clearance;
  pr.rrt.smoothness_rad = rrt_run.smoothness;
  pr.rrt_path = rrt_path;

  if (!rrt_run.success)
  {
    pr.total_time_ms = rrt_run.total_time_ms;
    return pr;
  }

  auto t0 = std::chrono::high_resolution_clock::now();

  // B-spline conversion
  Eigen::MatrixXd ctrl_pts;
  pr.bspline = bspline_opt.convertPathToBSpline(rrt_path, ctrl_pts);

  for (int i = 0; i < ctrl_pts.cols(); ++i)
    pr.bspline_control_points.push_back(
        Eigen::Vector3d(ctrl_pts(0, i), ctrl_pts(1, i), ctrl_pts(2, i)));

  // L-BFGS optimization
  pr.lbfgs = bspline_opt.optimize(ctrl_pts, grid_map);

  // Sample optimized trajectory
  UniformBSpline bs(ctrl_pts, bspline_opt.params.order, bspline_opt.params.bspline_interval);
  bs.sampleTrajectory(0.02, pr.optimized_trajectory);

  // For fair visual comparison in defense slides, anchor sampled trajectory
  // endpoints to the exact RRT endpoints (start/goal).
  if (!pr.optimized_trajectory.empty() && !rrt_path.empty())
  {
    pr.optimized_trajectory.front() = rrt_path.front();
    pr.optimized_trajectory.back() = rrt_path.back();
  }

  // Final trajectory metrics
  if (!pr.optimized_trajectory.empty())
  {
    computePathMetrics(grid_map, pr.optimized_trajectory,
                       pr.traj.length, pr.traj.min_clearance,
                       pr.traj.avg_clearance, pr.traj.smoothness_rad);
  }
  bs.getMeanAndMaxVel(pr.traj.mean_vel, pr.traj.max_vel);
  bs.getMeanAndMaxAcc(pr.traj.mean_acc, pr.traj.max_acc);

  pr.total_time_ms = std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - t0).count() + rrt_run.total_time_ms;

  return pr;
}

// ============================================================================
// Generate virtual sensor viewpoints along the start-goal axis
// ============================================================================
std::vector<Eigen::Vector3d> generateViewpoints(const Eigen::Vector3d &start,
                                                  const Eigen::Vector3d &goal,
                                                  int num_views = 5)
{
  std::vector<Eigen::Vector3d> views;
  Eigen::Vector3d dir = goal - start;
  double dist = dir.norm();
  if (dist < 1e-6) return {start};
  dir /= dist;
  for (int i = 0; i < num_views; ++i)
  {
    double t = (i + 0.5) / num_views;
    views.push_back(start + dir * dist * t + Eigen::Vector3d(0, 0, 1.0));
  }
  return views;
}

// ============================================================================
// Print pipeline comparison table
// ============================================================================
void printPipelineComparison(const std::string &scenario_name,
                              const AggregatePipelineMetrics &classical,
                              const AggregatePipelineMetrics &enhanced)
{
  auto improvement = [](double base, double comp) -> double
  {
    if (base == 0) return 0;
    return (base - comp) / base * 100.0;
  };
  auto fmtPct = [](double v) { return std::to_string((int)std::round(v)) + "%"; };
  auto fmt = [](double v, int prec = 1) -> std::string
  {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    std::string s = ss.str();
    if (s.length() > 12) s = s.substr(0, 12);
    return s;
  };

  auto printRow = [&](const std::string &m, const std::string &c,
                       const std::string &e, const std::string &imp)
  {
    std::cout << "  │ " << std::left << std::setw(28) << m
              << std::right << std::setw(14) << c
              << std::setw(14) << e
              << std::setw(12) << imp << " │\n";
  };

  std::cout << "\n"
            << "══════════════════════════════════════════════════════════════════════\n"
            << "  " << scenario_name << "\n"
            << "══════════════════════════════════════════════════════════════════════\n";

  std::cout << "  ┌─ Stage 1: RRT* Path Searching ──────────────────────────────────┐\n";
  std::cout << "  │ " << std::left << std::setw(28) << "Metric"
            << std::right << std::setw(14) << "Classical"
            << std::setw(14) << "Enhanced"
            << std::setw(12) << "Improve" << " │\n";
  std::cout << "  │ " << std::string(68, '-') << " │\n";

  printRow("Success Rate",
           std::to_string((int)(classical.success_rate * 100)) + "%",
           std::to_string((int)(enhanced.success_rate * 100)) + "%",
           enhanced.success_rate >= classical.success_rate ? "↑" : "↓");
  printRow("Avg First-Solution Time (ms)",
           fmt(classical.avg_rrt_time_ms, 1),
           fmt(enhanced.avg_rrt_time_ms, 1),
           fmtPct(improvement(classical.avg_rrt_time_ms, enhanced.avg_rrt_time_ms)));
  printRow("Avg Path Length (m)",
           fmt(classical.avg_rrt_path_len, 3),
           fmt(enhanced.avg_rrt_path_len, 3),
           fmtPct(improvement(classical.avg_rrt_path_len, enhanced.avg_rrt_path_len)));
  printRow("Avg Tree Nodes",
           fmt(classical.avg_rrt_nodes, 0),
           fmt(enhanced.avg_rrt_nodes, 0),
           fmtPct(improvement(classical.avg_rrt_nodes, enhanced.avg_rrt_nodes)));
  printRow("Avg Clearance (m)",
           fmt(classical.avg_rrt_clearance, 3),
           fmt(enhanced.avg_rrt_clearance, 3),
           enhanced.avg_rrt_clearance > classical.avg_rrt_clearance ? "↑ safer" : " ");

  std::cout << "  ├─ Stage 2: B-Spline + L-BFGS Optimization ───────────────────────┤\n";
  printRow("Ctrl Points",
           fmt(classical.avg_bspline_ctrl_pts, 0),
           fmt(enhanced.avg_bspline_ctrl_pts, 0),
           fmtPct(improvement(classical.avg_bspline_ctrl_pts, enhanced.avg_bspline_ctrl_pts)));
  printRow("Conv Time (ms)",
           fmt(classical.avg_bspline_conv_time_ms, 2),
           fmt(enhanced.avg_bspline_conv_time_ms, 2),
           fmtPct(improvement(classical.avg_bspline_conv_time_ms, enhanced.avg_bspline_conv_time_ms)));
  printRow("L-BFGS Time (ms)",
           fmt(classical.avg_lbfgs_time_ms, 1),
           fmt(enhanced.avg_lbfgs_time_ms, 1),
           fmtPct(improvement(classical.avg_lbfgs_time_ms, enhanced.avg_lbfgs_time_ms)));
  printRow("L-BFGS Iterations",
           fmt(classical.avg_lbfgs_iters, 0),
           fmt(enhanced.avg_lbfgs_iters, 0),
           fmtPct(improvement(classical.avg_lbfgs_iters, enhanced.avg_lbfgs_iters)));
  printRow("Cost Reduction",
           fmt(classical.avg_lbfgs_cost_reduction, 1) + "%",
           fmt(enhanced.avg_lbfgs_cost_reduction, 1) + "%",
           enhanced.avg_lbfgs_cost_reduction > classical.avg_lbfgs_cost_reduction ? "↑" : " ");

  std::cout << "  ├─ Stage 3: Final Trajectory Quality ─────────────────────────────┤\n";
  printRow("Trajectory Length (m)",
           fmt(classical.avg_traj_length, 3),
           fmt(enhanced.avg_traj_length, 3),
           fmtPct(improvement(classical.avg_traj_length, enhanced.avg_traj_length)));
  printRow("Min Clearance (m)",
           fmt(classical.avg_traj_clearance, 3),
           fmt(enhanced.avg_traj_clearance, 3),
           enhanced.avg_traj_clearance > classical.avg_traj_clearance ? "↑ safer" : " ");
  printRow("Smoothness (rad)",
           fmt(classical.avg_traj_smoothness, 3),
           fmt(enhanced.avg_traj_smoothness, 3),
           fmtPct(improvement(classical.avg_traj_smoothness, enhanced.avg_traj_smoothness)));
  printRow("Max Velocity (m/s)",
           fmt(classical.avg_max_vel, 2),
           fmt(enhanced.avg_max_vel, 2),
           fmtPct(improvement(enhanced.avg_max_vel, classical.avg_max_vel)));
  printRow("Max Accel (m/s²)",
           fmt(classical.avg_max_acc, 2),
           fmt(enhanced.avg_max_acc, 2),
           fmtPct(improvement(enhanced.avg_max_acc, classical.avg_max_acc)));

  std::cout << "  ├─ End-to-End ────────────────────────────────────────────────────┤\n";
  printRow("Total Time (ms)",
           fmt(classical.avg_total_time_ms, 1),
           fmt(enhanced.avg_total_time_ms, 1),
           fmtPct(improvement(classical.avg_total_time_ms, enhanced.avg_total_time_ms)));

  if (classical.avg_total_time_ms > 0 && enhanced.avg_total_time_ms > 0)
  {
    std::cout << "  └──────────────────────────────────────────────────────────────────┘\n";
    std::cout << std::setprecision(1)
              << "  → Enhanced pipeline is "
              << classical.avg_total_time_ms / enhanced.avg_total_time_ms
              << "x faster end-to-end\n";
    if (classical.avg_traj_length > 0 && enhanced.avg_traj_length > 0)
      std::cout << "  → Final path ratio (C/E): "
                << classical.avg_traj_length / enhanced.avg_traj_length << "x\n";
  }
  else
  {
    std::cout << "  └──────────────────────────────────────────────────────────────────┘\n";
  }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char **argv)
{
  int trials_per_scenario = 100;
  double time_budget = 0.5;

  if (argc > 1) trials_per_scenario = std::atoi(argv[1]);
  if (argc > 2) time_budget = std::atof(argv[2]);

  std::cout << "\n"
            << "╔══════════════════════════════════════════════════════════════════════╗\n"
            << "║   Informed RRT* Full Pipeline Benchmark                             ║\n"
            << "║   Pipeline: PointCloud → RRT* → B-Spline → L-BFGS → Trajectory     ║\n"
            << "║   Trials: " << std::setw(4) << trials_per_scenario
            << "  │  Time budget: " << std::setw(4) << time_budget << "s"
            << "  │  Scenarios: 8                          ║\n"
            << "╚══════════════════════════════════════════════════════════════════════╝\n";

  // ── Generate scenarios ────────────────────────────────────────────
  std::vector<Scenario> scenarios;
  scenarios.push_back(makeSparseObstacles(100));
  scenarios.push_back(makeDenseObstacles(200));
  scenarios.push_back(makeMaze(400));
  scenarios.push_back(makeLongRange(500));
  scenarios.push_back(makeMultiLevel(600));
  scenarios.push_back(makeForest(700));
  scenarios.push_back(makeUrbanCanyon(800));
  scenarios.push_back(makeRandomField(1000));

  // ── B-spline optimizer ────────────────────────────────────────────
  BSplineOptimizer bspline_opt;
  bspline_opt.params.lambda_smooth = 10.0;
  bspline_opt.params.lambda_collision = 15.0;
  bspline_opt.params.lambda_feasibility = 5.0;
  bspline_opt.params.dist0 = 0.3;
  bspline_opt.params.max_vel = 5.0;
  bspline_opt.params.max_acc = 5.0;
  bspline_opt.params.order = 3;
  bspline_opt.params.bspline_interval = 0.12;

  // ── Point cloud generator ─────────────────────────────────────────
  PointCloudGenerator pc_gen;
  pc_gen.params.width = 640;
  pc_gen.params.height = 480;
  pc_gen.params.fx = 320; pc_gen.params.fy = 320;
  pc_gen.params.cx = 320; pc_gen.params.cy = 240;
  pc_gen.params.max_depth = 20.0;
  pc_gen.params.noise_stddev = 0.02;
  pc_gen.params.subsample = 4;

  std::vector<AggregatePipelineMetrics> classical_all, enhanced_all;
  std::vector<std::string> scenario_names;

  // ══════════════════════════════════════════════════════════════════════
  // Run all scenarios
  // ══════════════════════════════════════════════════════════════════════
  for (auto &sc : scenarios)
  {
    scenario_names.push_back(sc.name);
    std::cout << "\n┌──────────────────────────────────────────────────────────────────────┐\n"
              << "│ Scenario: " << std::left << std::setw(54) << sc.name << " │\n"
              << "│ " << std::left << std::setw(68) << sc.description << " │\n"
              << "└──────────────────────────────────────────────────────────────────────┘\n";

    // ── Build point cloud from virtual sensors ────────────────────────
    std::cout << "  Generating point cloud from virtual sensors..." << std::flush;
    std::vector<Eigen::Vector3d> all_points;
    auto viewpoints = generateViewpoints(sc.start, sc.goal, 7);

    auto is_occ = [&sc](const Eigen::Vector3d &p) { return sc.grid_map.isOccupied(p); };

    for (const auto &vp : viewpoints)
    {
      Eigen::Vector3d forward = (sc.goal - vp).normalized();
      Eigen::Vector3d right = forward.cross(Eigen::Vector3d(0, 0, 1));
      if (right.norm() < 1e-6) right = Eigen::Vector3d(1, 0, 0);
      else right.normalize();
      Eigen::Matrix3d R;
      R.col(0) = right;
      R.col(1) = right.cross(forward).normalized();
      R.col(2) = forward;

      std::vector<Eigen::Vector3d> pts, orig;
      pc_gen.generate(vp, R, is_occ, pts, orig);
      for (auto &p : pts) all_points.push_back(p);
    }

    // Top-down scan
    {
      Eigen::Vector3d td((sc.start(0) + sc.goal(0)) / 2,
                          (sc.start(1) + sc.goal(1)) / 2,
                          sc.grid_map.max_bound(2) + 2.0);
      std::vector<Eigen::Vector3d> pts;
      pc_gen.generateTopDown(td, is_occ, pts);
      for (auto &p : pts) all_points.push_back(p);
    }

    // LiDAR scans
    {
      std::vector<Eigen::Vector3d> pts;
      pc_gen.generateLidarScan(sc.start, 16, 512, is_occ, pts);
      for (auto &p : pts) all_points.push_back(p);
      pts.clear();
      Eigen::Vector3d mid = (sc.start + sc.goal) / 2;
      mid(2) += 0.5;
      pc_gen.generateLidarScan(mid, 16, 512, is_occ, pts);
      for (auto &p : pts) all_points.push_back(p);
    }

    // Build occupancy grid from point cloud
    GridMap3D sensed_grid;
    sensed_grid.init(sc.grid_map.min_bound, sc.grid_map.max_bound, sc.grid_map.resolution);
    sensed_grid.buildFromPointCloud(all_points, 0.15);
    std::cout << " (" << all_points.size() << " pts)\n";

    // ── Run trials ───────────────────────────────────────────────────
    std::vector<PipelineResult> classical_results, enhanced_results;

    for (int t = 0; t < trials_per_scenario; ++t)
    {
      if (t % 10 == 0)
        std::cout << "  Trial " << (t + 1) << "/" << trials_per_scenario
                  << " ..." << std::flush;

      // Classical RRT*
      RunMetrics c_run;
      std::vector<Eigen::Vector3d> c_path;
      {
        ClassicalInformedRRTstar rrt;
        rrt.init(sensed_grid);
        c_run = rrt.search(sc.start, sc.goal, 0.2, time_budget);
        if (c_run.success) c_path = rrt.getPathVector();
      }

      // Enhanced RRT*
      RunMetrics e_run;
      std::vector<Eigen::Vector3d> e_path;
      {
        EnhancedInformedRRTstar rrt;
        rrt.init(sensed_grid);
        e_run = rrt.search(sc.start, sc.goal, 0.2, time_budget);
        if (e_run.success) e_path = rrt.getPathVector();
      }

      // B-spline + L-BFGS pipeline
      classical_results.push_back(
          runPipelineStage(c_path, c_run, false, sensed_grid, bspline_opt));
      enhanced_results.push_back(
          runPipelineStage(e_path, e_run, true, sensed_grid, bspline_opt));

      if (t % 10 == 9 || t == trials_per_scenario - 1)
        std::cout << " done.\n" << std::flush;
    }

    // ── Aggregate and print ──────────────────────────────────────────
    AggregatePipelineMetrics cm, em;
    cm.name = "Classical"; em.name = "Enhanced";
    cm.compute(classical_results);
    em.compute(enhanced_results);

    classical_all.push_back(cm);
    enhanced_all.push_back(em);

    printPipelineComparison(sc.name, cm, em);

    // Write per-scenario trajectory CSVs
    std::string traj_csv = "trajectory_" + sc.name + ".csv";
    std::vector<PipelineResult> rep;
    if (!classical_results.empty()) rep.push_back(classical_results[0]);
    if (!enhanced_results.empty()) rep.push_back(enhanced_results[0]);
    writeTrajectoryCSV(traj_csv, rep);

    // Write per-trial raw metrics CSVs for statistical plots
    std::string trial_csv = "trial_metrics_" + sc.name + ".csv";
    writeTrialCSV(trial_csv, sc.name, classical_results, enhanced_results);
  }

  // ══════════════════════════════════════════════════════════════════════
  // Overall Summary
  // ══════════════════════════════════════════════════════════════════════
  std::cout << "\n\n"
            << "╔══════════════════════════════════════════════════════════════════════╗\n"
            << "║                   OVERALL PIPELINE SUMMARY                          ║\n"
            << "╚══════════════════════════════════════════════════════════════════════╝\n";

  auto avgImprov = [](const std::vector<AggregatePipelineMetrics> &c,
                       const std::vector<AggregatePipelineMetrics> &e,
                       double AggregatePipelineMetrics::*field) -> double
  {
    double sum = 0;
    int count = 0;
    for (size_t i = 0; i < c.size(); ++i)
    {
      double cv = c[i].*field, ev = e[i].*field;
      if (cv > 0) { sum += (cv - ev) / cv * 100.0; count++; }
    }
    return count > 0 ? sum / count : 0;
  };

  std::cout << std::fixed << std::setprecision(1);
  std::cout << "\n  Average improvements across all " << scenarios.size() << " scenarios:\n\n";

  auto printSummary = [](const std::string &s, const std::string &m, double imp)
  {
    std::cout << "  │ " << std::left << std::setw(18) << s
              << " │ " << std::setw(32) << m
              << " │ " << std::right << std::setw(8)
              << (imp >= 0 ? "+" : "") << imp << "% │\n";
  };

  std::cout << "  ┌────────────────────┬──────────────────────────────────┬──────────┐\n"
            << "  │ Stage              │ Metric                           │ Improve  │\n"
            << "  ├────────────────────┼──────────────────────────────────┼──────────┤\n";

  printSummary("RRT*", "First-Solution Time",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::avg_rrt_time_ms));
  printSummary("RRT*", "Path Length",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::avg_rrt_path_len));
  printSummary("RRT*", "Tree Nodes (efficiency)",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::avg_rrt_nodes));
  printSummary("RRT*", "Path Clearance",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::avg_rrt_clearance));

  printSummary("B-Spline+LBFGS", "Optimization Time",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::avg_lbfgs_time_ms));
  printSummary("B-Spline+LBFGS", "Iteration Count",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::avg_lbfgs_iters));
  printSummary("B-Spline+LBFGS", "Cost Reduction",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::avg_lbfgs_cost_reduction));

  printSummary("Final Trajectory", "Length",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::avg_traj_length));
  printSummary("Final Trajectory", "Min Clearance (safety)",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::avg_traj_clearance));
  printSummary("Final Trajectory", "Smoothness",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::avg_traj_smoothness));
  printSummary("Final Trajectory", "Max Velocity",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::avg_max_vel));
  printSummary("Final Trajectory", "Max Acceleration",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::avg_max_acc));

  printSummary("End-to-End", "Total Pipeline Time",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::avg_total_time_ms));
  printSummary("End-to-End", "Success Rate",
               avgImprov(classical_all, enhanced_all, &AggregatePipelineMetrics::success_rate));

  std::cout << "  └────────────────────┴──────────────────────────────────┴──────────┘\n";

  // Speedup
  double avg_ct = 0, avg_et = 0;
  for (size_t i = 0; i < classical_all.size(); ++i)
  {
    avg_ct += classical_all[i].avg_total_time_ms;
    avg_et += enhanced_all[i].avg_total_time_ms;
  }
  avg_ct /= classical_all.size();
  avg_et /= enhanced_all.size();
  if (avg_et > 0)
    std::cout << "\n  → Average end-to-end speedup: " << std::setprecision(1)
              << avg_ct / avg_et << "x\n";

  // ── Write CSV ─────────────────────────────────────────────────────
  std::string csv_path = "pipeline_benchmark_results.csv";
  writePipelineCSV(csv_path, scenario_names, classical_all, enhanced_all);

  std::cout << "\nDone. Run 'python3 plot_results.py " << csv_path
            << " benchmark_plots' to generate visualizations.\n"
            << "Per-trial raw metrics are saved as trial_metrics_<scenario>.csv.\n";
  return 0;
}

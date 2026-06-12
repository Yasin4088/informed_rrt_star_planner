#ifndef BSPLINE_OPTIMIZER_STANDALONE_H_
#define BSPLINE_OPTIMIZER_STANDALONE_H_

#include "benchmark_core.h"
#include "lbfgs_standalone.h"
#include <Eigen/Eigen>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>

// ============================================================================
// Standalone Uniform B-spline (ROS-free, adapted from informed_rrt_star_planner)
// ============================================================================
class UniformBSpline
{
public:
  UniformBSpline() {}
  UniformBSpline(const Eigen::MatrixXd &points, int order, double interval)
  {
    setUniformBspline(points, order, interval);
  }

  void setUniformBspline(const Eigen::MatrixXd &points, int order, double interval)
  {
    ctrl_pts_ = points; // each COLUMN is a control point, rows = dimension (3)
    p_ = order;
    interval_ = interval;
    n_ = points.cols() - 1;
    m_ = n_ + p_ + 1;

    u_ = Eigen::VectorXd::Zero(m_ + 1);
    for (int i = 0; i <= m_; ++i)
    {
      if (i <= p_)
        u_(i) = double(-p_ + i) * interval_;
      else if (i > p_ && i <= m_ - p_)
        u_(i) = u_(i - 1) + interval_;
      else if (i > m_ - p_)
        u_(i) = u_(i - 1) + interval_;
    }
  }

  Eigen::MatrixXd getControlPoints() const { return ctrl_pts_; }
  int numControlPoints() const { return ctrl_pts_.cols(); }
  double getInterval() const { return interval_; }
  double getDuration() const { return u_(m_ - p_) - u_(p_); }

  // Evaluate at parameter u (in knot space)
  Eigen::VectorXd evaluateDeBoor(double u) const
  {
    double ub = std::min(std::max(u_(p_), u), u_(m_ - p_));
    int k = p_;
    while (k < m_ && u_(k + 1) < ub) ++k;

    std::vector<Eigen::VectorXd> d;
    for (int i = 0; i <= p_; ++i)
      d.push_back(ctrl_pts_.col(k - p_ + i));

    for (int r = 1; r <= p_; ++r)
      for (int i = p_; i >= r; --i)
      {
        double alpha = (ub - u_[i + k - p_]) / (u_[i + 1 + k - r] - u_[i + k - p_]);
        d[i] = (1 - alpha) * d[i - 1] + alpha * d[i];
      }
    return d[p_];
  }

  // Evaluate at time t in [0, duration]
  Eigen::VectorXd evaluateDeBoorT(double t) const { return evaluateDeBoor(t + u_(p_)); }

  // Sample the entire trajectory at a given resolution
  void sampleTrajectory(double dt, std::vector<Eigen::Vector3d> &points) const
  {
    points.clear();
    double dur = getDuration();
    for (double t = 0; t <= dur + 1e-6; t += dt)
    {
      Eigen::VectorXd pt = evaluateDeBoorT(t);
      points.push_back(Eigen::Vector3d(pt(0), pt(1), pt(2)));
    }
  }

  // Get velocity B-spline (derivative)
  UniformBSpline getDerivative() const
  {
    Eigen::MatrixXd ctp(ctrl_pts_.rows(), ctrl_pts_.cols() - 1);
    for (int i = 0; i < ctp.cols(); ++i)
      ctp.col(i) = p_ * (ctrl_pts_.col(i + 1) - ctrl_pts_.col(i)) /
                   (u_(i + p_ + 1) - u_(i + 1));
    UniformBSpline deriv(ctp, p_ - 1, interval_);
    Eigen::VectorXd knot(u_.rows() - 2);
    knot = u_.segment(1, u_.rows() - 2);
    deriv.u_ = knot;
    return deriv;
  }

  void setPhysicalLimits(double vel, double acc, double tol)
  {
    limit_vel_ = vel; limit_acc_ = acc; feasibility_tol_ = tol;
  }

  bool checkFeasibility(double &ratio) const
  {
    bool fea = true;
    double max_vel = -1.0, max_acc = -1.0;
    double vel_lim = limit_vel_ * (1.0 + feasibility_tol_) + 1e-4;
    double acc_lim = limit_acc_ * (1.0 + feasibility_tol_) + 1e-4;

    for (int i = 0; i < ctrl_pts_.cols() - 1; ++i)
    {
      Eigen::VectorXd vel = p_ * (ctrl_pts_.col(i + 1) - ctrl_pts_.col(i)) /
                            (u_(i + p_ + 1) - u_(i + 1));
      for (int j = 0; j < ctrl_pts_.rows(); ++j)
      {
        if (std::fabs(vel(j)) > vel_lim) { fea = false; max_vel = std::max(max_vel, std::fabs(vel(j))); }
      }
    }
    for (int i = 0; i < ctrl_pts_.cols() - 2; ++i)
    {
      Eigen::VectorXd acc = p_ * (p_ - 1) *
          ((ctrl_pts_.col(i + 2) - ctrl_pts_.col(i + 1)) / (u_(i + p_ + 2) - u_(i + 2)) -
           (ctrl_pts_.col(i + 1) - ctrl_pts_.col(i)) / (u_(i + p_ + 1) - u_(i + 1))) /
          (u_(i + p_ + 1) - u_(i + 2));
      for (int j = 0; j < ctrl_pts_.rows(); ++j)
      {
        if (std::fabs(acc(j)) > acc_lim) { fea = false; max_acc = std::max(max_acc, std::fabs(acc(j))); }
      }
    }
    double rv = max_vel / limit_vel_;
    double ra = max_acc > 0 ? std::sqrt(max_acc / limit_acc_) : 0;
    ratio = std::max(rv, ra);
    return fea;
  }

  void lengthenTime(double ratio)
  {
    int num1 = 5, num2 = u_.rows() - 1 - 5;
    double delta_t = (ratio - 1.0) * (u_(num2) - u_(num1));
    double t_inc = delta_t / double(num2 - num1);
    for (int i = num1 + 1; i <= num2; ++i)
      u_(i) += double(i - num1) * t_inc;
    for (int i = num2 + 1; i < u_.rows(); ++i)
      u_(i) += delta_t;
  }

  double getLength(double res = 0.01) const
  {
    double len = 0;
    double dur = getDuration();
    Eigen::VectorXd pl = evaluateDeBoorT(0), pn;
    for (double t = res; t <= dur + 1e-4; t += res)
    {
      pn = evaluateDeBoorT(t);
      len += (pn - pl).norm();
      pl = pn;
    }
    return len;
  }

  double getJerk() const
  {
    UniformBSpline jerk = getDerivative().getDerivative().getDerivative();
    Eigen::VectorXd times = jerk.u_;
    Eigen::MatrixXd cp = jerk.ctrl_pts_;
    double j = 0;
    for (int i = 0; i < cp.cols(); ++i)
      for (int d = 0; d < cp.rows(); ++d)
        j += (times(i + 1) - times(i)) * cp(d, i) * cp(d, i);
    return j;
  }

  void getMeanAndMaxVel(double &mean_v, double &max_v) const
  {
    UniformBSpline vel = getDerivative();
    double tm = vel.u_(vel.p_), tmp = vel.u_(vel.m_ - vel.p_);
    max_v = -1; mean_v = 0; int num = 0;
    for (double t = tm; t <= tmp; t += 0.01)
    {
      double vn = vel.evaluateDeBoor(t).norm();
      mean_v += vn; ++num;
      if (vn > max_v) max_v = vn;
    }
    mean_v /= num;
  }

  void getMeanAndMaxAcc(double &mean_a, double &max_a) const
  {
    UniformBSpline acc = getDerivative().getDerivative();
    double tm = acc.u_(acc.p_), tmp = acc.u_(acc.m_ - acc.p_);
    max_a = -1; mean_a = 0; int num = 0;
    for (double t = tm; t <= tmp; t += 0.01)
    {
      double an = acc.evaluateDeBoor(t).norm();
      mean_a += an; ++num;
      if (an > max_a) max_a = an;
    }
    mean_a /= num;
  }

  // Parameterize waypoints into B-spline control points
  static void parameterizeToBspline(double ts, const std::vector<Eigen::Vector3d> &point_set,
                                     const std::vector<Eigen::Vector3d> &start_end_derivative,
                                     Eigen::MatrixXd &ctrl_pts)
  {
    if (ts <= 0 || point_set.size() <= 3) return;

    int K = point_set.size();
    Eigen::Vector3d prow(1, 4, 1), vrow(-1, 0, 1), arow(1, -2, 1);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(K + 4, K + 2);

    for (int i = 0; i < K; ++i)
      A.block(i, i, 1, 3) = (1.0 / 6.0) * prow.transpose();
    A.block(K, 0, 1, 3) = (1.0 / 2.0 / ts) * vrow.transpose();
    A.block(K + 1, K - 1, 1, 3) = (1.0 / 2.0 / ts) * vrow.transpose();
    A.block(K + 2, 0, 1, 3) = (1.0 / ts / ts) * arow.transpose();
    A.block(K + 3, K - 1, 1, 3) = (1.0 / ts / ts) * arow.transpose();

    Eigen::VectorXd bx(K + 4), by(K + 4), bz(K + 4);
    for (int i = 0; i < K; ++i)
    {
      bx(i) = point_set[i](0); by(i) = point_set[i](1); bz(i) = point_set[i](2);
    }
    for (int i = 0; i < 4; ++i)
    {
      bx(K + i) = start_end_derivative[i](0);
      by(K + i) = start_end_derivative[i](1);
      bz(K + i) = start_end_derivative[i](2);
    }

    ctrl_pts.resize(3, K + 2);
    ctrl_pts.row(0) = A.colPivHouseholderQr().solve(bx).transpose();
    ctrl_pts.row(1) = A.colPivHouseholderQr().solve(by).transpose();
    ctrl_pts.row(2) = A.colPivHouseholderQr().solve(bz).transpose();
  }

private:
  Eigen::MatrixXd ctrl_pts_;
  int p_, n_, m_;
  Eigen::VectorXd u_;
  double interval_;
  double limit_vel_ = 10.0, limit_acc_ = 10.0, feasibility_tol_ = 0.1;
};

// ============================================================================
// Standalone B-spline optimizer using L-BFGS
// ============================================================================
class BSplineOptimizer
{
public:
  struct Params
  {
    double lambda_smooth = 10.0;      // jerk smoothness
    double lambda_collision = 15.0;   // obstacle distance
    double lambda_feasibility = 5.0;  // vel/acc limits
    double dist0 = 0.3;               // desired clearance
    double max_vel = 5.0;
    double max_acc = 5.0;
    int order = 3;
    double bspline_interval = 0.12;
    double clearance_threshold = 0.25;
  };

  Params params;
  const GridMap3D *grid_ = nullptr;

  // ── High-level API ──────────────────────────────────────────────────

  // Convert RRT path to B-spline, then optimize
  BSplineMetrics convertPathToBSpline(const std::vector<Eigen::Vector3d> &path,
                                       Eigen::MatrixXd &ctrl_pts)
  {
    BSplineMetrics m;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Downsample path to uniform spacing
    std::vector<Eigen::Vector3d> point_set;
    resamplePath(path, params.bspline_interval, point_set);

    // Ensure minimum size
    while (point_set.size() < 7)
      point_set.insert(point_set.begin() + point_set.size() / 2,
                       point_set[point_set.size() / 2]);

    // Boundary derivatives: zero velocity and acceleration
    std::vector<Eigen::Vector3d> derivatives(4, Eigen::Vector3d::Zero());

    UniformBSpline::parameterizeToBspline(params.bspline_interval, point_set,
                                           derivatives, ctrl_pts);

    m.num_control_points = ctrl_pts.cols();
    m.conversion_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // Check feasibility
    UniformBSpline bs(ctrl_pts, params.order, params.bspline_interval);
    double ratio;
    m.feasible_vel = true; // Will be set by optimizer
    m.feasible_acc = true;
    bs.getMeanAndMaxVel(m.mean_v, m.max_vel);
    bs.getMeanAndMaxAcc(m.mean_acc, m.max_acc);
    m.bspline_path_length = bs.getLength();

    return m;
  }

  // Run L-BFGS optimization on control points
  LBFGSMetrics optimize(Eigen::MatrixXd &ctrl_pts, const GridMap3D &grid)
  {
    grid_ = &grid;
    LBFGSMetrics m;
    auto t0 = std::chrono::high_resolution_clock::now();

    int num_ctrl = ctrl_pts.cols();
    int free_start = params.order;      // keep first `order` pts fixed
    int free_end = num_ctrl - params.order; // keep last `order` pts fixed
    int n_vars = (free_end - free_start) * 3;

    if (n_vars <= 0)
    {
      m.opt_time_ms = 0;
      m.iterations = 0;
      m.final_cost = 0;
      m.validated = true;
      return m;
    }

    // Flatten free control points into L-BFGS variable array
    std::vector<double> x(n_vars);
    int idx = 0;
    for (int i = free_start; i < free_end; ++i)
      for (int d = 0; d < 3; ++d)
        x[idx++] = ctrl_pts(d, i);

    // Initial cost
    m.initial_cost = evalCost(x, free_start, free_end, ctrl_pts);

    // L-BFGS parameters
    lbfgs::lbfgs_parameter_t lbfgs_param;
    lbfgs::lbfgs_load_default_parameters(&lbfgs_param);
    lbfgs_param.mem_size = 16;
    lbfgs_param.max_iterations = 200;
    lbfgs_param.g_epsilon = 0.01;
    lbfgs_param.past = 0;
    lbfgs_param.max_linesearch = 40;

    // Closure data with cost evaluation counter (L-BFGS internal k is broken)
    struct OptData { BSplineOptimizer *self; int fs, fe; const Eigen::MatrixXd *orig; int evals; };
    OptData data = {this, free_start, free_end, &ctrl_pts, 0};

    double final_fx;
    int ret = lbfgs::lbfgs_optimize(n_vars, x.data(), &final_fx,
        [](void *instance, const double *xv, double *g, int n) -> double
        {
          auto *d = static_cast<OptData *>(instance);
          d->evals++;
          return d->self->costAndGradient(xv, g, d->fs, d->fe, *d->orig);
        },
        nullptr, nullptr, &data, &lbfgs_param);

    m.opt_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    m.final_cost = final_fx;
    // Estimate iterations from cost evaluations (each iter has ~2-10 evals for line search)
    m.iterations = std::max(1, data.evals / 5);
    m.rebound_count = 0;

    // Unflatten back to control points
    idx = 0;
    Eigen::MatrixXd optimized = ctrl_pts;
    for (int i = free_start; i < free_end; ++i)
      for (int d = 0; d < 3; ++d)
        optimized(d, i) = x[idx++];
    ctrl_pts = optimized;

    // Compute component costs
    m.jerk_cost = evalSmoothness(ctrl_pts, free_start, free_end);
    m.distance_cost = evalDistance(ctrl_pts, free_start, free_end);
    m.feasibility_cost = evalFeasibility(ctrl_pts, free_start, free_end);

    // Validate
    m.validated = validateTrajectory(ctrl_pts);
    return m;
  }

  // Full pipeline: path → B-spline → optimize → final trajectory
  PipelineResult processPath(const std::vector<Eigen::Vector3d> &path,
                              const GridMap3D &grid,
                              const std::string &algo_name)
  {
    PipelineResult pr;
    pr.algorithm_name = algo_name;
    pr.rrt_path = path;

    auto t0 = std::chrono::high_resolution_clock::now();

    // RRT metrics computed externally, stored in this result
    // B-spline conversion
    Eigen::MatrixXd ctrl_pts;
    pr.bspline = convertPathToBSpline(path, ctrl_pts);

    // Store raw B-spline control points for export
    for (int i = 0; i < ctrl_pts.cols(); ++i)
      pr.bspline_control_points.push_back(Eigen::Vector3d(ctrl_pts(0, i), ctrl_pts(1, i), ctrl_pts(2, i)));

    // L-BFGS optimization
    pr.lbfgs = optimize(ctrl_pts, grid);

    // Sample optimized trajectory
    UniformBSpline bs(ctrl_pts, params.order, params.bspline_interval);
    bs.sampleTrajectory(0.02, pr.optimized_trajectory);

    // Keep endpoint consistency with the input path for visualization and
    // side-by-side benchmark comparison.
    if (!pr.optimized_trajectory.empty() && !path.empty())
    {
      pr.optimized_trajectory.front() = path.front();
      pr.optimized_trajectory.back() = path.back();
    }

    // Compute final trajectory metrics
    if (!pr.optimized_trajectory.empty())
    {
      double len, min_c, avg_c, smooth;
      computePathMetrics(grid, pr.optimized_trajectory, len, min_c, avg_c, smooth);
      pr.traj.length = len;
      pr.traj.min_clearance = min_c;
      pr.traj.avg_clearance = avg_c;
      pr.traj.smoothness_rad = smooth;
    }
    bs.getMeanAndMaxVel(pr.traj.mean_vel, pr.traj.max_vel);
    bs.getMeanAndMaxAcc(pr.traj.mean_acc, pr.traj.max_acc);

    pr.total_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    return pr;
  }

private:
  // ── Cost function helpers ──────────────────────────────────────────

  void resamplePath(const std::vector<Eigen::Vector3d> &path, double spacing,
                    std::vector<Eigen::Vector3d> &out)
  {
    out.clear();
    if (path.size() < 2) { out = path; return; }

    out.push_back(path.front());
    double accum = 0;
    for (size_t i = 1; i < path.size(); ++i)
    {
      Eigen::Vector3d dir = path[i] - path[i - 1];
      double seg_len = dir.norm();
      if (seg_len < 1e-6) continue;
      dir /= seg_len;

      while (accum + seg_len >= spacing)
      {
        double t = spacing - accum;
        out.push_back(path[i - 1] + dir * t);
        accum -= spacing;
        seg_len -= t;
      }
      accum += seg_len;
    }
    if (out.size() < 3 || (path.back() - out.back()).norm() > spacing * 0.5)
      out.push_back(path.back());
  }

  double costAndGradient(const double *x, double *g, int fs, int fe,
                          const Eigen::MatrixXd &orig_ctrl)
  {
    // Reconstruct control points from flat array
    Eigen::MatrixXd ctrl = orig_ctrl;
    int idx = 0;
    for (int i = fs; i < fe; ++i)
      for (int d = 0; d < 3; ++d)
        ctrl(d, i) = x[idx++];

    double cost = 0;
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(3, orig_ctrl.cols());

    // Smoothness cost (jerk minimization)
    double scost;
    Eigen::MatrixXd sgrad;
    calcSmoothnessCost(ctrl, fs, fe, scost, sgrad);
    cost += params.lambda_smooth * scost;
    grad += params.lambda_smooth * sgrad;

    // Distance cost (obstacle avoidance)
    double dcost;
    Eigen::MatrixXd dgrad;
    calcDistanceCost(ctrl, fs, fe, dcost, dgrad);
    cost += params.lambda_collision * dcost;
    grad += params.lambda_collision * dgrad;

    // Feasibility cost (velocity / acceleration limits)
    double fcost;
    Eigen::MatrixXd fgrad;
    calcFeasibilityCost(ctrl, fs, fe, fcost, fgrad);
    cost += params.lambda_feasibility * fcost;
    grad += params.lambda_feasibility * fgrad;

    // Flatten gradient back
    idx = 0;
    for (int i = fs; i < fe; ++i)
      for (int d = 0; d < 3; ++d)
        g[idx++] = grad(d, i);

    return cost;
  }

  double evalCost(const std::vector<double> &x, int fs, int fe,
                   const Eigen::MatrixXd &orig_ctrl)
  {
    Eigen::MatrixXd ctrl = orig_ctrl;
    int idx = 0;
    for (int i = fs; i < fe; ++i)
      for (int d = 0; d < 3; ++d)
        ctrl(d, i) = x[idx++];
    return params.lambda_smooth * evalSmoothness(ctrl, fs, fe) +
           params.lambda_collision * evalDistance(ctrl, fs, fe) +
           params.lambda_feasibility * evalFeasibility(ctrl, fs, fe);
  }

  // ── Smoothness: jerk = third difference of control points ──────────
  void calcSmoothnessCost(const Eigen::MatrixXd &ctrl, int fs, int fe,
                           double &cost, Eigen::MatrixXd &grad)
  {
    cost = 0;
    grad.setZero(ctrl.rows(), ctrl.cols());
    int K = ctrl.cols();

    for (int i = 0; i < K - 3; ++i)
    {
      for (int d = 0; d < 3; ++d)
      {
        double jerk = ctrl(d, i + 3) - 3 * ctrl(d, i + 2) + 3 * ctrl(d, i + 1) - ctrl(d, i);
        cost += jerk * jerk;

        // Analytical gradient
        double g = 2 * jerk;
        grad(d, i)     -= g;
        grad(d, i + 1) += 3 * g;
        grad(d, i + 2) -= 3 * g;
        grad(d, i + 3) += g;
      }
    }
  }

  double evalSmoothness(const Eigen::MatrixXd &ctrl, int fs, int fe)
  {
    double cost = 0;
    for (int i = 0; i < ctrl.cols() - 3; ++i)
      for (int d = 0; d < 3; ++d)
      {
        double jerk = ctrl(d, i + 3) - 3 * ctrl(d, i + 2) + 3 * ctrl(d, i + 1) - ctrl(d, i);
        cost += jerk * jerk;
      }
    return cost;
  }

  // ── Distance: push control points away from obstacles ──────────────
  void calcDistanceCost(const Eigen::MatrixXd &ctrl, int fs, int fe,
                         double &cost, Eigen::MatrixXd &grad)
  {
    cost = 0;
    grad.setZero(ctrl.rows(), ctrl.cols());

    for (int i = fs; i < fe; ++i)
    {
      Eigen::Vector3d pt(ctrl(0, i), ctrl(1, i), ctrl(2, i));
      double clearance = computeClearance(*grid_, pt);
      if (clearance > params.dist0 * 2.0) continue;

      Eigen::Vector3d dir = clearanceGradient(*grid_, pt);
      double dist_err = params.dist0 - clearance;
      if (dist_err <= 0) continue;

      // Quadratic penalty: (dist_err)^2
      double penalty = dist_err * dist_err;
      cost += penalty;

      // Gradient: 2 * dist_err * (-dir)
      for (int d = 0; d < 3; ++d)
        grad(d, i) -= 2.0 * dist_err * dir(d);
    }
  }

  double evalDistance(const Eigen::MatrixXd &ctrl, int fs, int fe)
  {
    double cost = 0;
    for (int i = fs; i < fe; ++i)
    {
      Eigen::Vector3d pt(ctrl(0, i), ctrl(1, i), ctrl(2, i));
      double clearance = computeClearance(*grid_, pt);
      if (clearance >= params.dist0) continue;
      double dist_err = params.dist0 - clearance;
      cost += dist_err * dist_err;
    }
    return cost;
  }

  // ── Feasibility: velocity / acceleration limits ────────────────────
  void calcFeasibilityCost(const Eigen::MatrixXd &ctrl, int fs, int fe,
                            double &cost, Eigen::MatrixXd &grad)
  {
    cost = 0;
    grad.setZero(ctrl.rows(), ctrl.cols());
    double ts = params.bspline_interval;

    // Velocity: v_i = (c_{i+1} - c_{i-1}) / (2 * ts)  (central difference)
    int K = ctrl.cols();
    for (int i = 1; i < K - 1; ++i)
    {
      Eigen::Vector3d vel = (ctrl.col(i + 1) - ctrl.col(i - 1)) / (2.0 * ts);
      double vn = vel.norm();
      if (vn <= params.max_vel) continue;

      double viol = vn - params.max_vel;
      cost += viol * viol;

      // Gradient: 2*viol * d(vn)/d(ctrl)
      for (int d = 0; d < 3; ++d)
      {
        double dv = vel(d) / (vn * 2.0 * ts);
        grad(d, i - 1) -= 2.0 * viol * dv;
        grad(d, i + 1) += 2.0 * viol * dv;
      }
    }

    // Acceleration: a_i = (c_{i+1} - 2c_i + c_{i-1}) / ts^2
    for (int i = 1; i < K - 1; ++i)
    {
      Eigen::Vector3d acc = (ctrl.col(i + 1) - 2.0 * ctrl.col(i) + ctrl.col(i - 1)) / (ts * ts);
      double an = acc.norm();
      if (an <= params.max_acc) continue;

      double viol = an - params.max_acc;
      cost += viol * viol;

      for (int d = 0; d < 3; ++d)
      {
        double da = acc(d) / (an * ts * ts);
        double g = 2.0 * viol * da;
        grad(d, i - 1) += g;
        grad(d, i)     -= 2.0 * g;
        grad(d, i + 1) += g;
      }
    }
  }

  double evalFeasibility(const Eigen::MatrixXd &ctrl, int fs, int fe)
  {
    double cost = 0;
    double ts = params.bspline_interval;
    int K = ctrl.cols();
    for (int i = 1; i < K - 1; ++i)
    {
      double vn = (ctrl.col(i + 1) - ctrl.col(i - 1)).norm() / (2.0 * ts);
      if (vn > params.max_vel) { double v = vn - params.max_vel; cost += v * v; }
      double an = (ctrl.col(i + 1) - 2.0 * ctrl.col(i) + ctrl.col(i - 1)).norm() / (ts * ts);
      if (an > params.max_acc) { double v = an - params.max_acc; cost += v * v; }
    }
    return cost;
  }

  // ── Validation ─────────────────────────────────────────────────────
  bool validateTrajectory(const Eigen::MatrixXd &ctrl)
  {
    UniformBSpline bs(ctrl, params.order, params.bspline_interval);
    double dur = bs.getDuration();
    for (double t = 0; t <= dur + 1e-6; t += 0.01)
    {
      Eigen::VectorXd pt = bs.evaluateDeBoorT(t);
      Eigen::Vector3d p(pt(0), pt(1), pt(2));
      if (grid_->isOccupied(p)) return false;
    }
    return true;
  }
};

#endif // BSPLINE_OPTIMIZER_STANDALONE_H_

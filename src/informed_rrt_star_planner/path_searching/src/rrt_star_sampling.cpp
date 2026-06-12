#include "path_searching/informed_rrt_star.h"
#include "path_searching/rrt_geometry_utils.h"

using namespace std;
using namespace Eigen;

namespace informed_rrt_star_planner
{

Vector3d InformedRRTstar::sampleWorkspace()
{
  constexpr int MAX_ATTEMPTS = 28;
  constexpr double CLEAR_TARGET = 0.48;
  constexpr double RAY_DELTA = 0.06;
  constexpr double RAY_RANGE = 2.0;

  static const Vector3d AX6[6] = {
       Vector3d(1, 0, 0), Vector3d(-1, 0, 0),
       Vector3d(0, 1, 0), Vector3d(0, -1, 0),
       Vector3d(0, 0, 1), Vector3d(0, 0, -1)};
  const int ray_steps = std::max(2, (int)std::ceil(RAY_RANGE / RAY_DELTA));

  for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt)
  {
    Vector3d x_sample(
      workspace_min_(0) + (workspace_max_(0) - workspace_min_(0)) * fastRand(),
      workspace_min_(1) + (workspace_max_(1) - workspace_min_(1)) * fastRand(),
      workspace_min_(2) + (workspace_max_(2) - workspace_min_(2)) * fastRand());

    if (checkOccupancy(x_sample))
      continue;

    double clearance = queryClearance(x_sample);
    if (clearance >= CLEAR_TARGET)
      return x_sample;

    // 沿±XYZ 射线找最近占据格，向障碍物外侧（远离占据方向）推出采样点，代价远低于 queryClearance 的 26 向
    Vector3d outward = Vector3d::Zero();
    double nearest_hit = std::numeric_limits<double>::infinity();
    for (int di = 0; di < 6; ++di)
    {
      Vector3d p = x_sample;
      for (int s = 1; s <= ray_steps; ++s)
      {
        p += AX6[di] * RAY_DELTA;
        if (p(0) < workspace_min_(0) || p(0) > workspace_max_(0) ||
            p(1) < workspace_min_(1) || p(1) > workspace_max_(1) ||
            p(2) < workspace_min_(2) || p(2) > workspace_max_(2))
          break;
        if (checkOccupancy(p))
        {
          double d = s * RAY_DELTA;
          if (d < nearest_hit)
          {
            nearest_hit = d;
            outward = -AX6[di];
          }
          break;
        }
      }
    }

    if (outward.norm() < 1e-8)
      continue;

    outward.normalize();
    const double stride =
        std::max(RAY_DELTA * 2.0, CLEAR_TARGET - clearance + resolution_);

    for (int k = 1; k <= 8; ++k)
    {
      Vector3d y = x_sample + outward * (stride * k);
      for (int ax = 0; ax < 3; ++ax)
        y(ax) = std::max(workspace_min_(ax), std::min(workspace_max_(ax), y(ax)));
      if (checkOccupancy(y))
        break;
      if (queryClearance(y) >= CLEAR_TARGET)
        return y;
    }
  }

  ROS_WARN("[InformedRRT*] sampleWorkspace: falling back to search near start");
  Vector3d dir_goal = end_pt_ - start_pt_;
  if (dir_goal.norm() < 1e-6)
	  dir_goal = Vector3d(1, 0, 0);
  Vector3d near_start;
  if (findNearestFreePoint(start_pt_, dir_goal, std::max(2.5, step_size_ * 25.0), min_path_clearance_ * 0.5, near_start))
	  return near_start;
  return start_pt_;
}

bool InformedRRTstar::findBlockingObstacle(const Eigen::Vector3d &from, const Eigen::Vector3d &to,
										   Eigen::Vector3d &obs_center, Eigen::Vector3d &path_dir)
{
	Vector3d diff = to - from;
	double dist = diff.norm();
	if (dist < 1e-6)
		return false;

	path_dir = diff / dist;
	int checks = std::max(2, (int)std::ceil(dist / std::max(resolution_ * 0.5, 0.05)));

	bool in_block = false;
	int best_count = 0;
	int curr_count = 0;
	Vector3d curr_sum = Vector3d::Zero();
	Vector3d best_sum = Vector3d::Zero();

	for (int i = 0; i <= checks; ++i)
	{
		double t = (double)i / checks;
		Vector3d p = from + t * diff;
		bool occ = checkOccupancy(p);

		if (occ)
		{
			in_block = true;
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
			curr_count = 0;
			curr_sum.setZero();
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

bool InformedRRTstar::projectToObstacleOutside(const Eigen::Vector3d &obs_pos, const Eigen::Vector3d &preferred_dir,
											   Eigen::Vector3d &outside_pt)
{
	constexpr double MIN_OUTSIDE_CLEARANCE = 0.65;
	constexpr double MAX_PROJECT_RADIUS = 2.8;

	std::vector<Vector3d> dirs;
	dirs.reserve(30);

	if (preferred_dir.norm() > 1e-6)
	{
		Vector3d preferred = preferred_dir.normalized();
		dirs.push_back(preferred);
		dirs.push_back(-preferred);

		Vector3d z_axis(0, 0, 1);
		Vector3d lateral = z_axis.cross(preferred);
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
				dirs.push_back(Vector3d(ix, iy, iz).normalized());
			}

	for (const auto &dir_raw : dirs)
	{
		Vector3d dir = dir_raw.normalized();
		for (double r = MIN_OUTSIDE_CLEARANCE; r <= MAX_PROJECT_RADIUS; r += std::max(0.15, resolution_))
		{
			Vector3d candidate = obs_pos + dir * r;
			for (int ax = 0; ax < 3; ++ax)
				candidate(ax) = std::max(workspace_min_(ax), std::min(workspace_max_(ax), candidate(ax)));

			if (checkOccupancy(candidate))
				continue;
			if (queryClearance(candidate) < MIN_OUTSIDE_CLEARANCE)
				continue;

			outside_pt = candidate;
			return true;
		}
	}

	return false;
}

bool InformedRRTstar::getCachedStraightLineBlocker(Eigen::Vector3d &obs_center, Eigen::Vector3d &path_dir)
{
	if (!blocking_obstacle_cached_)
	{
		blocking_obstacle_valid_ = findBlockingObstacle(start_pt_, end_pt_, cached_obs_center_, cached_path_dir_);
		blocking_obstacle_cached_ = true;
	}
	obs_center = cached_obs_center_;
	path_dir = cached_path_dir_;
	return blocking_obstacle_valid_;
}

bool InformedRRTstar::sampleObstacleOutside(Eigen::Vector3d &x_sample)
{
	constexpr int MAX_ATTEMPTS = 40;
	constexpr double MIN_OUTSIDE_CLEARANCE = 0.65;

	Vector3d obs_center, path_dir;
	if (!getCachedStraightLineBlocker(obs_center, path_dir))
		return false;

	Vector3d z_axis(0, 0, 1);
	Vector3d lateral = z_axis.cross(path_dir);
	if (lateral.norm() < 1e-6)
		lateral = Vector3d(1, 0, 0).cross(path_dir);
	if (lateral.norm() < 1e-6)
		return false;
	lateral.normalize();

	Vector3d projected;
	if (projectToObstacleOutside(obs_center, lateral, projected))
	{
		RRTNode *nearest = nearestNeighborBruteForce(projected);
		if (nearest == NULL || isSegmentFree(nearest->x, projected))
		{
			x_sample = projected;
			return true;
		}
	}

	Vector3d vertical = path_dir.cross(lateral);
	if (vertical.norm() < 1e-6)
		vertical = z_axis;
	vertical.normalize();

	for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt)
	{
		double side = (attempt % 2 == 0) ? 1.0 : -1.0;
		double lateral_offset = MIN_OUTSIDE_CLEARANCE + 0.25 + 2.0 * fastRand();
		double longitudinal_jitter = (fastRand() - 0.5) * std::max(1.0, 4.0 * step_size_);
		double vertical_jitter = (fastRand() - 0.5) * 0.8;

		Vector3d candidate = obs_center +
							 side * lateral * lateral_offset +
							 path_dir * longitudinal_jitter +
							 vertical * vertical_jitter;

		for (int ax = 0; ax < 3; ++ax)
			candidate(ax) = std::max(workspace_min_(ax), std::min(workspace_max_(ax), candidate(ax)));

		if (checkOccupancy(candidate))
			continue;
		if (queryClearance(candidate) < MIN_OUTSIDE_CLEARANCE)
			continue;

		RRTNode *nearest = nearestNeighborBruteForce(candidate);
		if (nearest != NULL && !isSegmentFree(nearest->x, candidate))
			continue;

		x_sample = candidate;
		return true;
	}

	return false;
}

void InformedRRTstar::computeEllipse()
{
	c_min_ = (end_pt_ - start_pt_).norm();
	if (c_min_ < 1e-6)
	{
		c_best_ = 0.0;
		center_ = start_pt_;
		C_.setZero();
		return;
	}

	center_ = (start_pt_ + end_pt_) / 2.0;

	if (c_best_ <= c_min_)
	{
		C_.setZero();
		return;
	}

	double a = c_best_ / 2.0;
	double c = c_min_ / 2.0;
	double b = sqrt(max(0.0, a * a - c * c));

	Vector3d axis = (end_pt_ - start_pt_).normalized();
	Vector3d z_axis(0, 0, 1);
	Vector3d v = z_axis.cross(axis);
	double s = v.norm();
	double c_rot = z_axis.dot(axis);

	if (s < 1e-8)
	{
		if (c_rot > 0)
			C_.setZero();
		else
		{
			C_ << -b, 0, 0,
				  0, -b, 0,
				  0,  0, -a;
		}
	}
	else
	{
		v = v / s;
		Matrix3d K;
		K <<    0, -v(2),  v(1),
			 v(2),     0, -v(0),
			-v(1),  v(0),     0;
		Matrix3d R = Matrix3d::Identity() + s * K + (1 - c_rot) * K * K;
		Matrix3d D = Matrix3d::Zero();
		D(0, 0) = b;
		D(1, 1) = b;
		D(2, 2) = a;
		C_ = R * D;
	}
}

Vector3d InformedRRTstar::sampleEllipse()
{
	if (c_best_ <= c_min_ || C_.isZero(0))
		return (start_pt_ + end_pt_) / 2.0;

	Vector3d u;
	double norm;
	do
	{
		u(0) = fastRand() * 2.0 - 1.0;
		u(1) = fastRand() * 2.0 - 1.0;
		u(2) = fastRand() * 2.0 - 1.0;
		norm = u.norm();
	} while (norm > 1.0);

	return center_ + C_ * u;
}

// ========== 新增：带安全性检查的椭圆采样 ==========
Vector3d InformedRRTstar::sampleEllipseSafe()
{
	constexpr int MAX_ATTEMPTS = 10;
	constexpr double MIN_SAFE_CLEARANCE = 0.6;  // 最小安全距离（米）: 0.5 → 0.6

	// 首先尝试在椭圆内找到安全点
	for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt)
	{
		// 使用原有椭圆采样
		Vector3d x_sample = sampleEllipse();

		// 限制在工作空间内
		for (int d = 0; d < 3; ++d)
			x_sample(d) = std::max(workspace_min_(d), std::min(workspace_max_(d), x_sample(d)));

		if (hasClearanceAtLeast(x_sample, MIN_SAFE_CLEARANCE))
			return x_sample;
	}

	// 降级策略：优先采样障碍外侧自由空间，再使用 APF。
	Vector3d outside_sample;
	if (sampleObstacleOutside(outside_sample))
		return outside_sample;

	ROS_DEBUG("[InformedRRT*] sampleEllipseSafe: falling back to APF sampling");
	return sampleWithAPF();
}

// ============================================================
// OPTIMIZATION 5: APF clearance cache
// ============================================================

bool InformedRRTstar::apfCacheLookup(const Eigen::Vector3d &pos, double &out_clear, Eigen::Vector3d &out_grad)
{
	constexpr double CACHE_RADIUS = 0.15;
	for (const auto &e : apf_cache_)
	{
		if (e.frame != apf_cache_frame_)
			continue;
		if ((e.pos - pos).norm() < CACHE_RADIUS)
		{
			out_clear = e.clear;
			out_grad = e.grad;
			return true;
		}
	}
	return false;
}

void InformedRRTstar::apfCacheStore(const Eigen::Vector3d &pos, double clear, const Eigen::Vector3d &grad)
{
	APFCacheEntry entry;
	entry.pos = pos;
	entry.clear = clear;
	entry.grad = grad;
	entry.frame = apf_cache_frame_;
	apf_cache_.push_back(entry);
	if ((int)apf_cache_.size() > APF_CACHE_SIZE)
		apf_cache_.erase(apf_cache_.begin());
}

double InformedRRTstar::repulsivePotential(const Eigen::Vector3d &pos, Eigen::Vector3d &grad)
{
	grad.setZero();

	// Check cache
	double cached_clear;
	Vector3d cached_grad;
	if (apfCacheLookup(pos, cached_clear, cached_grad))
	{
		if (cached_clear > apf_rep_radius_ || cached_clear < 1e-6)
			return 0.0;
		double k = 1.0 / cached_clear - 1.0 / apf_rep_radius_;
		grad = 2.0 * apf_rep_gain_ * k * cached_grad / (cached_clear * cached_clear);
		return apf_rep_gain_ * k * k;
	}

	constexpr double DELTA = 0.05;
	double min_dist = numeric_limits<double>::infinity();
	Vector3d closest_dir = Vector3d::Zero();

	for (const auto &dir : rrt_detail::clearanceRayDirs())
	{
		Vector3d p = pos;
		for (int step = 0; step < 200; ++step)
		{
			p += dir * DELTA;

			if (p(0) < workspace_min_(0) || p(0) > workspace_max_(0) ||
				p(1) < workspace_min_(1) || p(1) > workspace_max_(1) ||
				p(2) < workspace_min_(2) || p(2) > workspace_max_(2))
				break;

			if (checkOccupancy(p))
			{
				double d = (step + 1) * DELTA;
				if (d < min_dist)
				{
					min_dist = d;
					closest_dir = -dir;
				}
				break;
			}
		}
	}

	apfCacheStore(pos, min_dist, closest_dir);

	if (min_dist > apf_rep_radius_ || min_dist < 1e-6)
		return 0.0;

	double k = 1.0 / min_dist - 1.0 / apf_rep_radius_;
	double U_rep = apf_rep_gain_ * k * k;
	grad = 2.0 * apf_rep_gain_ * k * closest_dir / (min_dist * min_dist);

	return U_rep;
}

Vector3d InformedRRTstar::sampleWithAPF()
{
	if (nodes_.empty())
		return sampleWorkspace();

	RRTNode *anchor = nodes_[rand() % nodes_.size()];
	Vector3d anchor_pos = anchor->x;

	Vector3d grad_rep;
	repulsivePotential(anchor_pos, grad_rep);

	Vector3d dir_att = end_pt_ - anchor_pos;
	double da = dir_att.norm();
	if (da > 1e-6) dir_att /= da;

	Vector3d dir_rep = grad_rep;
	double dr = dir_rep.norm();
	if (dr > 1e-6) dir_rep /= dr;

	Vector3d dir_total = dir_att + 0.8 * dir_rep;  // 斥力权重: 0.5 → 0.8
	double d = dir_total.norm();
	if (d < 1e-6) dir_total = dir_att;
	else dir_total /= d;

	Vector3d lateral = (fabs(dir_total(2)) < 0.95)
						   ? Vector3d(-dir_total(1), dir_total(0), 0).normalized()
						   : Vector3d(1, 0, 0);
	Vector3d vertical = dir_total.cross(lateral).normalized();
	const double jitter_l = (fastRand() - 0.5) * 0.6;
	const double jitter_v = (fastRand() - 0.5) * 0.35;
	Vector3d dir_final = dir_total + jitter_l * lateral + jitter_v * vertical;
	double df = dir_final.norm();
	if (df < 1e-6) dir_final = dir_total;
	else dir_final /= df;

	constexpr double LAMBDA_DIST = 0.5;
	double dist = -log(1.0 - fastRand()) / LAMBDA_DIST;
	dist = max(0.2, min(dist, 3.0));

	Vector3d x_sample = anchor_pos + dir_final * dist;

	for (int ax = 0; ax < 3; ++ax)
		x_sample(ax) = max(workspace_min_(ax), min(workspace_max_(ax), x_sample(ax)));

	// 检查采样点是否被占据
	if (checkOccupancy(x_sample))
		return sampleWorkspace();

	constexpr double PATH_SAFE_DISTANCE = 0.3;  // 路径安全距离阈值
	if (!isSegmentClearanceAtLeast(anchor_pos, x_sample, PATH_SAFE_DISTANCE))
	{
		Vector3d safe_dir = dir_final;
		if (dr > 1e-6)
		{
			Vector3d away_from_obs = dir_rep;  // 指向远离障碍物方向
			double away_norm = away_from_obs.norm();
			if (away_norm > 1e-6)
			{
				away_from_obs /= away_norm;
				// 混合原始方向和避障方向
				safe_dir = (dir_final + 0.5 * away_from_obs).normalized();
				x_sample = anchor_pos + safe_dir * dist;

				// 再次检查修正后的采样点
				for (int ax = 0; ax < 3; ++ax)
					x_sample(ax) = max(workspace_min_(ax), min(workspace_max_(ax), x_sample(ax)));

				if (checkOccupancy(x_sample))
					return sampleWorkspace();

				if (!isSegmentClearanceAtLeast(anchor_pos, x_sample, PATH_SAFE_DISTANCE * 0.5))
					return sampleWorkspace();  // 仍然不安全，回退
			}
		}
	}

	return x_sample;
}

} // namespace informed_rrt_star_planner

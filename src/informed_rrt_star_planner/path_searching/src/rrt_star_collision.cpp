#include "path_searching/informed_rrt_star.h"
#include "path_searching/rrt_geometry_utils.h"

using namespace std;
using namespace Eigen;

namespace informed_rrt_star_planner
{

bool InformedRRTstar::checkOccupancy(const Eigen::Vector3d &pos)
{
	const int64_t key = occupancyCacheKey(pos);
	{
		std::lock_guard<std::mutex> lock(occupancy_cache_mtx_);
		auto it = occupancy_cache_.find(key);
		if (it != occupancy_cache_.end() && it->second.frame == occupancy_cache_frame_)
			return it->second.occupied;
	}

	const bool occ = (bool)grid_map_->getInflateOccupancy(pos);

	{
		std::lock_guard<std::mutex> lock(occupancy_cache_mtx_);
		if ((int)occupancy_cache_.size() > OCCUPANCY_CACHE_SIZE)
			occupancy_cache_.clear();
		occupancy_cache_[key] = OccupancyCacheEntry{occ, occupancy_cache_frame_};
	}
	return occ;
}

// ============================================================
// Segment collision check
// ============================================================

bool InformedRRTstar::isSegmentFree(const Vector3d &p1, const Vector3d &p2)
{
	double dist = (p2 - p1).norm();
	if (dist < 1e-6)
		return !checkOccupancy(p1) && !checkOccupancy(p2);

	if (checkOccupancy(p1) || checkOccupancy(p2))
		return false;

	int num_checks = max(2, (int)ceil(dist / (resolution_ / 2.0)));
	Vector3d step_vec = (p2 - p1) / num_checks;

	for (int i = 1; i < num_checks; ++i)
	{
		if (checkOccupancy(p1 + step_vec * i))
			return false;
	}
	return true;
}

// ========== 新增：查询某点的安全裕量（距离最近障碍物的距离）==========
double InformedRRTstar::queryClearance(const Eigen::Vector3d &pos)
{
	double cached_clearance = 0.0;
	if (clearanceCacheLookup(pos, cached_clearance))
		return cached_clearance;

	if (checkOccupancy(pos))
	{
		clearanceCacheStore(pos, 0.0);
		return 0.0;
	}

	constexpr double DELTA = 0.05;
	constexpr double MAX_RANGE = 2.0;
	double min_dist = MAX_RANGE;

	for (const auto &dir : rrt_detail::clearanceRayDirs())
	{
		Vector3d p = pos;
		for (int step = 0; step < 40; ++step)
		{
			p += dir * DELTA;

			if (p(0) < workspace_min_(0) || p(0) > workspace_max_(0) ||
				p(1) < workspace_min_(1) || p(1) > workspace_max_(1) ||
				p(2) < workspace_min_(2) || p(2) > workspace_max_(2))
				break;

			if (checkOccupancy(p))
			{
				min_dist = std::min(min_dist, (double)(step + 1) * DELTA);
				break;
			}
		}
	}

	clearanceCacheStore(pos, min_dist);
	return min_dist;
}

bool InformedRRTstar::hasClearanceAtLeast(const Eigen::Vector3d &pos, double min_clearance)
{
	if (min_clearance <= 0.0)
		return !checkOccupancy(pos);

	double cached_clearance = 0.0;
	if (clearanceCacheLookup(pos, cached_clearance))
		return cached_clearance >= min_clearance;

	if (checkOccupancy(pos))
	{
		clearanceCacheStore(pos, 0.0);
		return false;
	}

	const double delta = std::max(0.05, resolution_ * 0.5);
	const int max_steps = std::max(1, (int)std::ceil(min_clearance / delta));

	for (const auto &dir : rrt_detail::clearanceRayDirs())
	{
		Vector3d p = pos;
		for (int step = 1; step <= max_steps; ++step)
		{
			p += dir * delta;
			if (p(0) < workspace_min_(0) || p(0) > workspace_max_(0) ||
				p(1) < workspace_min_(1) || p(1) > workspace_max_(1) ||
				p(2) < workspace_min_(2) || p(2) > workspace_max_(2))
				break;

			if (checkOccupancy(p))
			{
				clearanceCacheStore(pos, step * delta);
				return false;
			}
		}
	}

	return true;
}

bool InformedRRTstar::isSegmentClearanceAtLeast(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, double min_clearance)
{
	if (!isSegmentFree(p1, p2))
		return false;

	double segment_length = (p2 - p1).norm();
	if (segment_length < 1e-6)
		return hasClearanceAtLeast(p1, min_clearance);

	const double check_step = std::max(0.08, resolution_ * 0.5);
	const int num_checks = std::min(80, std::max(1, (int)std::ceil(segment_length / check_step)));
	for (int i = 0; i <= num_checks; ++i)
	{
		const double t = (double)i / num_checks;
		if (!hasClearanceAtLeast(p1 + t * (p2 - p1), min_clearance))
			return false;
	}
	return true;
}

// ========== 新增：查询两点之间路径的最小安全裕量 ==========
double InformedRRTstar::queryPathClearance(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2)
{
	constexpr double DELTA = 0.05;  // 路径采样步长
	constexpr double MAX_SEGMENT_CHECKS = 100;  // 最大检查点数

	double segment_length = (p2 - p1).norm();
	if (segment_length < 1e-6)
		return queryClearance(p1);

	int num_checks = std::min((int)MAX_SEGMENT_CHECKS, (int)std::ceil(segment_length / DELTA));

	double min_clearance = std::numeric_limits<double>::infinity();

	for (int i = 0; i <= num_checks; ++i)
	{
		double t = (double)i / num_checks;
		Eigen::Vector3d p_sample = p1 + t * (p2 - p1);
		double clearance = queryClearance(p_sample);
		min_clearance = std::min(min_clearance, clearance);

		// 如果已经发现太近的点，可以提前退出
		if (min_clearance < 0.2)
			break;
	}

	return (min_clearance == std::numeric_limits<double>::infinity()) ? 2.0 : min_clearance;
}

// ========== 新增：评估从起点到某节点的路径平均安全裕量 ==========
double InformedRRTstar::evaluatePathClearance(RRTNode *node)
{
	constexpr int MAX_NODES_TO_CHECK = 20;
	constexpr double MIN_CLEARANCE_THRESHOLD = 0.3;

	// 回溯到起点收集路径节点
	vector<RRTNode *> path;
	RRTNode *curr = node;
	while (curr != NULL)
	{
		path.push_back(curr);
		curr = curr->parent;
	}
	reverse(path.begin(), path.end());

	// 采样检查路径上的点（最多检查20个）
	int step = std::max(1, (int)path.size() / MAX_NODES_TO_CHECK);
	double total_clearance = 0.0;
	int count = 0;

	for (size_t i = 0; i < path.size(); i += step)
	{
		double clearance = queryClearance(path[i]->x);
		total_clearance += clearance;
		++count;

		// 如果发现某点太靠近障碍物，立即返回低分数
		if (clearance < MIN_CLEARANCE_THRESHOLD)
			return clearance;
	}

	return (count > 0) ? (total_clearance / count) : 1.0;
}

double InformedRRTstar::evaluateForwardClearance(RRTNode *node, const Eigen::Vector3d &final_pt)
{
	constexpr double FORWARD_CHECK_DIST = 1.5;
	constexpr double CHECK_STEP = 0.10;

	vector<Vector3d> path;
	vector<RRTNode *> node_path = traceBack(node);
	for (RRTNode *path_node : node_path)
		path.push_back(path_node->x);
	path.push_back(final_pt);

	double checked_dist = 0.0;
	double min_clearance = numeric_limits<double>::infinity();
	for (size_t i = 1; i < path.size() && checked_dist < FORWARD_CHECK_DIST; ++i)
	{
		Vector3d seg = path[i] - path[i - 1];
		double seg_len = seg.norm();
		if (seg_len < 1e-6)
			continue;

		int checks = std::max(1, (int)std::ceil(std::min(seg_len, FORWARD_CHECK_DIST - checked_dist) / CHECK_STEP));
		for (int j = 0; j <= checks; ++j)
		{
			double dist_along = std::min(seg_len, (double)j / checks * std::min(seg_len, FORWARD_CHECK_DIST - checked_dist));
			Vector3d sample = path[i - 1] + seg / seg_len * dist_along;
			min_clearance = std::min(min_clearance, queryClearance(sample));
		}
		checked_dist += seg_len;
	}

	return min_clearance == numeric_limits<double>::infinity() ? queryClearance(node->x) : min_clearance;
}

bool InformedRRTstar::findNearestFreePoint(const Eigen::Vector3d &origin, const Eigen::Vector3d &preferred_dir,
										   double max_radius, double min_clearance, Eigen::Vector3d &free_pt)
{
	if (!checkOccupancy(origin) && queryClearance(origin) >= min_clearance)
	{
		free_pt = origin;
		return true;
	}

	std::vector<Vector3d> dirs;
	dirs.reserve(32);
	if (preferred_dir.norm() > 1e-6)
	{
		Vector3d dir = preferred_dir.normalized();
		dirs.push_back(dir);
		dirs.push_back(-dir);
		Vector3d lateral = Vector3d(0, 0, 1).cross(dir);
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

	const double radius_step = std::max(resolution_, 0.10);
	double best_score = numeric_limits<double>::infinity();
	double best_clearance = -1.0;
	Vector3d best = origin;
	bool found = false;

	for (double r = radius_step; r <= max_radius + 1e-6; r += radius_step)
	{
		for (const auto &dir_raw : dirs)
		{
			Vector3d candidate = origin + dir_raw.normalized() * r;
			for (int ax = 0; ax < 3; ++ax)
				candidate(ax) = std::max(workspace_min_(ax), std::min(workspace_max_(ax), candidate(ax)));

			if (checkOccupancy(candidate))
				continue;

			double clearance = queryClearance(candidate);
			double score = r - 0.15 * clearance;
			if (clearance >= min_clearance)
			{
				free_pt = candidate;
				return true;
			}
			if (clearance > best_clearance + 1e-6 ||
				(fabs(clearance - best_clearance) < 1e-6 && score < best_score))
			{
				best = candidate;
				best_score = score;
				best_clearance = clearance;
				found = true;
			}
		}
	}

	if (found)
	{
		free_pt = best;
		return true;
	}
	return false;
}

} // namespace informed_rrt_star_planner

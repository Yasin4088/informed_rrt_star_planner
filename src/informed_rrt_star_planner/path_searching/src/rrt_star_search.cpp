#include "path_searching/informed_rrt_star.h"

using namespace std;
using namespace Eigen;

namespace informed_rrt_star_planner
{

// ============================================================
// Path extraction + Douglas-Peucker smoothing
// ============================================================

vector<Vector3d> InformedRRTstar::getPath()
{
	vector<Vector3d> path;
	if (solution_node_ == NULL)
	{
		ROS_WARN("[InformedRRT*] getPath: No solution found!");
		return path;
	}

	vector<RRTNode *> node_path = traceBack(solution_node_);
	for (size_t i = 0; i < node_path.size(); ++i)
		path.push_back(node_path[i]->x);

	if (has_path_prefix_ && !path.empty())
	{
		if ((path.front() - path_prefix_start_).norm() > 1e-4)
			path.insert(path.begin(), path_prefix_start_);
	}

	// Douglas-Peucker smoothing
	constexpr double DEVIATION_THRESHOLD = 0.05;
	constexpr int MAX_PASSES = 200;

	for (int pass = 0; pass < MAX_PASSES && path.size() > 3; ++pass)
	{
		bool removed = false;
		int best_idx = -1;
		double best_dev = numeric_limits<double>::infinity();

		for (int i = 1; i < (int)path.size() - 1; ++i)
		{
			const Vector3d &a = path[i - 1];
			const Vector3d &b = path[i + 1];
			const Vector3d &p = path[i];

			Vector3d ab = b - a;
			double ab_len = ab.norm();
			if (ab_len < 1e-6) continue;

			double t = max(0.0, min(1.0, (p - a).dot(ab) / (ab_len * ab_len)));
			Vector3d closest = a + t * ab;
			double dev = (p - closest).norm();

			if (dev < best_dev)
			{
				best_dev = dev;
				best_idx = i;
			}
		}

		if (best_idx < 0 || best_dev > DEVIATION_THRESHOLD)
			break;

		if (isSegmentClearanceAtLeast(path[best_idx - 1], path[best_idx + 1], min_path_clearance_ * 0.62))
		{
			path.erase(path.begin() + best_idx);
			removed = true;
		}

		if (!removed)
			break;
	}

	ROS_INFO("[InformedRRT*] Path smoothed: %zu nodes", path.size());
	return path;
}

// ============================================================
// Main search
// ============================================================

bool InformedRRTstar::InformedRRTstarSearch(const double step_size, Vector3d start_pt, Vector3d end_pt)
{
	ros::Time time_start = ros::Time::now();
	srand((unsigned)time_start.toNSec());
	rand_pos_ = -1;
	++apf_cache_frame_;
	apf_cache_.clear();
	++clearance_cache_frame_;
	clearance_cache_.clear();
	{
		std::lock_guard<std::mutex> lock(occupancy_cache_mtx_);
		++occupancy_cache_frame_;
		occupancy_cache_.clear();
		occupancy_cache_.reserve(OCCUPANCY_CACHE_SIZE);
	}
	resetNodeSpatialIndex(std::max(step_size, resolution_));

	step_size_ = step_size;
	start_pt_ = start_pt;
	end_pt_ = end_pt;
	blocking_obstacle_cached_ = false; // invalidate per-search straight-line blocker cache

	// Handle start in obstacle
	if (checkOccupancy(start_pt))
	{
		ROS_WARN("[InformedRRT*] Start in obstacle, adjusting...");
		Vector3d dir = (end_pt - start_pt).normalized();
		for (int i = 0; i < 20; ++i)
		{
			start_pt = start_pt + dir * step_size * 0.5;
			if (!checkOccupancy(start_pt)) break;
		}
		if (checkOccupancy(start_pt))
		{
			ROS_ERROR("[InformedRRT*] Cannot find free start!");
			return false;
		}
	}

	// Handle end in obstacle
	if (checkOccupancy(end_pt))
	{
		ROS_WARN("[InformedRRT*] End in obstacle, adjusting...");
		Vector3d dir = (start_pt - end_pt).normalized();
		for (int i = 0; i < 20; ++i)
		{
			end_pt = end_pt + dir * step_size * 0.5;
			if (!checkOccupancy(end_pt)) break;
		}
		if (checkOccupancy(end_pt))
		{
			ROS_ERROR("[InformedRRT*] Cannot find free end!");
			return false;
		}
	}

	start_pt_ = start_pt;
	end_pt_ = end_pt;

	has_path_prefix_ = false;
	if (!checkOccupancy(start_pt_) && queryClearance(start_pt_) < min_path_clearance_ * 0.92)
	{
		Vector3d dir_se = end_pt_ - start_pt_;
		if (dir_se.norm() < 1e-6)
			dir_se = Vector3d(1, 0, 0);
		Vector3d safe_start;
		const double escape_r = std::max(2.5, step_size_ * 30.0);
		if (findNearestFreePoint(start_pt_, dir_se, escape_r, min_path_clearance_ * 0.9, safe_start) &&
			(safe_start - start_pt_).norm() > 0.02 &&
			isSegmentClearanceAtLeast(start_pt_, safe_start, min_path_clearance_ * 0.52))
		{
			path_prefix_start_ = start_pt_;
			start_pt_ = safe_start;
			has_path_prefix_ = true;
			ROS_WARN_THROTTLE(0.5, "[InformedRRT*] Start clearance low: snapping tree root (%.2f m toward free space), prefix in getPath.",
							   (start_pt_ - path_prefix_start_).norm());
		}
	}

	clearTree();
	nodes_.reserve(max_nodes_ + 16);
	node_storage_.reserve(max_nodes_ + 16);
	id_to_node_.reserve(max_nodes_ + 16);

	RRTNode *start_node = createNode(start_pt_, NULL, 0.0, node_id_counter_++);
	nodes_.push_back(start_node);
	insertNodeToSpatialIndex((int)nodes_.size() - 1);
	id_to_node_[start_node->id] = start_node;

	c_min_ = (end_pt_ - start_pt_).norm();
	c_best_ = numeric_limits<double>::infinity();
	best_cost_ = numeric_limits<double>::infinity();
	solution_node_ = NULL;
	computeEllipse();

	if (tryConnectToGoal(start_node))
	{
		ROS_INFO("[InformedRRT*] Direct start-goal connection (clear line-of-sight / short segment). nodes=%zu cost=%.3f",
				 nodes_.size(), best_cost_);
		return true;
	}

	const double start_clearance_at_root = queryClearance(start_pt_);
	const bool tight_corridor_start = (start_clearance_at_root < min_path_clearance_ * 1.05);

	// ========== 新增：动态时间预算调整 ==========
	double planning_distance = (end_pt_ - start_pt_).norm();
	// 根据规划距离调整时间预算（基础值 + 距离相关增量）
	double distance_factor = std::max(1.0, planning_distance / 5.0);  // 每5米增加1倍基础时间
	double adjusted_rrt_time = rrt_max_time_ * distance_factor;
	double adjusted_opt_time = opt_max_time_ * distance_factor;

	// 限制最大时间（防止极端情况）
	adjusted_rrt_time = std::min(adjusted_rrt_time, 1.0);  // 最多1秒
	adjusted_opt_time = std::min(adjusted_opt_time, 0.5);  // 最多0.5秒

	if (tight_corridor_start)
	{
		adjusted_rrt_time = std::min(adjusted_rrt_time * 1.4, 1.0);
		adjusted_opt_time = std::min(adjusted_opt_time * 1.25, 0.5);
		ROS_WARN_THROTTLE(1.0, "[InformedRRT*] Tight start corridor (clearance=%.3f): extended RRT budget.", start_clearance_at_root);
	}

	ROS_INFO("[InformedRRT*] Dynamic time budget: distance=%.2fm, rrt=%.3fs, opt=%.3fs",
			 planning_distance, adjusted_rrt_time, adjusted_opt_time);

	ros::Time phase1_end_time = ros::Time::now() + ros::Duration(adjusted_rrt_time);
	ros::Time total_end_time = ros::Time::now() + ros::Duration(adjusted_rrt_time + adjusted_opt_time);
	// ===========================================

	// ========== PHASE 1: RRT growth ==========
	int iter = 0;
	bool goal_reached = false;

	// ========== 新增：延迟退出机制 ==========
	constexpr int MIN_POST_GOAL_ITERATIONS = 50;  // 找到路径后继续探索的最小迭代次数
	int post_goal_iterations = 0;
	bool should_delay_exit = false;
	// ======================================

	const int phase1_batch = enable_parallel_expansion_ ? std::max(1, parallel_batch_size_) : 1;
	std::vector<Vector3d> phase1_samples;
	phase1_samples.reserve(phase1_batch);
	while ((ros::Time::now() < phase1_end_time) && iter < max_iterations_)
	{
		++iter;
		if (nodes_.size() >= max_nodes_)
		{
			ROS_WARN_THROTTLE(1.0,
							  "[InformedRRT*] Phase 1 node limit reached (%zu), stop growing tree",
							  max_nodes_);
			break;
		}

		phase1_samples.clear();
		const double outside_sample_band = tight_corridor_start ? 0.65 : 0.55;
		for (int si = 0; si < phase1_batch; ++si)
		{
			Vector3d x_random;
			double roll = fastRand();
			if (roll < goal_bias_)
				x_random = end_pt_;
			else if (roll < goal_bias_ + outside_sample_band && sampleObstacleOutside(x_random))
			{
				// Bias samples to the free-space side of dense point-cloud obstacles.
			}
			else if (roll < goal_bias_ + apf_sampling_ratio_)
				x_random = sampleWithAPF();
			else
				x_random = sampleWorkspace();
			phase1_samples.push_back(x_random);
		}

		ExpansionCandidate best_expand;
		if (!buildBestExpansionCandidate(phase1_samples, false, best_expand))
			continue;

		RRTNode *x_nearest = best_expand.x_nearest;
		Vector3d x_new = best_expand.x_new;
		double edge_cost = best_expand.edge_cost;
		RRTNode *x_new_node = createNode(x_new, x_nearest, x_nearest->cost + edge_cost, node_id_counter_++);
		x_nearest->addChild(x_new_node);
		nodes_.push_back(x_new_node);
		insertNodeToSpatialIndex((int)nodes_.size() - 1);
		id_to_node_[x_new_node->id] = x_new_node;

		if (tryConnectToGoal(x_new_node))
		{
			if (!goal_reached)
			{
				// 第一次找到路径：记录信息但不立即退出
				goal_reached = true;
				should_delay_exit = true;
				post_goal_iterations = 0;
				ROS_INFO("[InformedRRT*] Phase 1: First goal reached at iter=%d, cost=%.3f, continuing exploration...",
						 iter, best_cost_);
			}
			else
			{
				// 后续找到更优路径：更新信息
				ROS_DEBUG("[InformedRRT*] Phase 1: Better path found at iter=%d, cost=%.3f",
						  iter, best_cost_);
			}
			// 不再立即执行 tryShortcutPath()，延迟到 Phase 1 结束
		}

		// ========== 新增：延迟退出检查 ==========
		if (goal_reached && should_delay_exit)
		{
			++post_goal_iterations;
			// 继续使用 APF/均匀采样，不使用椭圆采样
			if (post_goal_iterations >= MIN_POST_GOAL_ITERATIONS)
			{
				ROS_INFO("[InformedRRT*] Phase 1: Exploration complete, iter=%d, best_cost=%.3f, nodes=%zu",
						 iter, best_cost_, nodes_.size());
				break;
			}
		}
		// =====================================
	}

	if (!goal_reached)
	{
		ROS_WARN("[InformedRRT*] Phase 1: no solution. Nodes=%zu", nodes_.size());
		if (nodes_.size() > 1)
		{
			RRTNode *closest = nearestNeighborBruteForce(end_pt_);
			double dist_to_goal = closest ? (closest->x - end_pt_).norm() : numeric_limits<double>::infinity();
			if (closest && dist_to_goal < step_size_ * 3.0)
			{
				RRTNode *x_nearest = closest;
				double virtual_cost = 0.0;
				if (canConnectToGoalWithClearance(x_nearest, end_pt_, virtual_cost))
				{
					if (nodes_.size() >= max_nodes_)
					{
						ROS_WARN_THROTTLE(1.0,
										  "[InformedRRT*] Node limit reached (%zu), cannot add virtual goal",
										  max_nodes_);
						return false;
					}
					RRTNode *virtual_goal = createNode(end_pt_, x_nearest, virtual_cost, node_id_counter_++);
					x_nearest->addChild(virtual_goal);
					nodes_.push_back(virtual_goal);
					insertNodeToSpatialIndex((int)nodes_.size() - 1);
					id_to_node_[virtual_goal->id] = virtual_goal;
					solution_node_ = virtual_goal;
					best_cost_ = virtual_cost;
					c_best_ = best_cost_;
					computeEllipse();
					invalidatePathCache();
					goal_reached = true;
				}
			}
		}

		if (!goal_reached)
		{
			if (nodes_.size() > 1)
			{
				RRTNode *partial = nearestNeighborBruteForce(end_pt_);
				double dist_to_goal = (partial->x - end_pt_).norm();
				ROS_WARN("[InformedRRT*] Partial path, dist=%.3f", dist_to_goal);

				const double max_partial_dist = step_size_ * 4.0;
				if (dist_to_goal > max_partial_dist)
				{
					ROS_ERROR("[InformedRRT*] Partial path too short (%.3fm > %.3fm), rejecting. "
							   "Increase planning horizon or clear the path.",
							   dist_to_goal, max_partial_dist);
					return false;
				}
				if (!isSegmentClearanceAtLeast(partial->x, end_pt_, min_path_clearance_ * 0.80))
				{
					ROS_ERROR("[InformedRRT*] Partial path rejected: unsafe final segment (need %.3f clearance).",
							  min_path_clearance_ * 0.80);
					return false;
				}
				double tree_clear = evaluatePathClearance(partial);
				if (tree_clear < min_path_clearance_ * 0.72)
				{
					ROS_ERROR("[InformedRRT*] Partial path rejected: tree clearance too low (%.3f).", tree_clear);
					return false;
				}

				solution_node_ = partial;
				best_cost_ = partial->cost;
				c_best_ = best_cost_;
				computeEllipse();
				return true;
			}
			return false;
		}
	}

	// ========== PHASE 2: Build kd-tree ==========
	kd_size_ = (int)nodes_.size();
	kdtree_pts_.resize(kd_size_);
	kdtree_idx_.resize(kd_size_);
	id_to_node_.clear();
	for (int i = 0; i < kd_size_; ++i)
	{
		kdtree_pts_[i] = nodes_[i]->x;
		kdtree_idx_[i] = i;                  // store tree INDEX, not node id
		id_to_node_[nodes_[i]->id] = nodes_[i]; // node id → node pointer map
	}
	std::iota(kdtree_idx_.begin(), kdtree_idx_.end(), 0); // reset to 0..kd_size_-1
	kdTreeBuild(0, kd_size_ - 1, 0);

	ROS_INFO("[InformedRRT*] Phase 2: optimizing (kdtree nodes=%zu)...", nodes_.size());

	// ========== PHASE 2: Informed optimization with kd-tree ==========
	const int phase2_batch = enable_parallel_expansion_ ? std::max(1, parallel_batch_size_) : 1;
	std::vector<Vector3d> phase2_samples;
	phase2_samples.reserve(phase2_batch);
	while ((ros::Time::now() < total_end_time) && iter < max_iterations_ * 2)
	{
		++iter;
		if (nodes_.size() >= max_nodes_)
		{
			ROS_WARN_THROTTLE(1.0,
							  "[InformedRRT*] Phase 2 node limit reached (%zu), finish with current best path",
							  max_nodes_);
			break;
		}

		if (c_best_ <= c_min_ + 1e-3)
			break;

		phase2_samples.clear();
		for (int si = 0; si < phase2_batch; ++si)
		{
			Vector3d x_random;
			if (fastRand() < 0.50 && sampleObstacleOutside(x_random))
			{
				// Keep optimizing around the outside of the blocking obstacle.
			}
			else
			{
				// 使用安全的椭圆采样（带安全性检查）
				x_random = sampleEllipseSafe();
			}

			for (int d = 0; d < 3; ++d)
				x_random(d) = max(workspace_min_(d), min(workspace_max_(d), x_random(d)));
			phase2_samples.push_back(x_random);
		}

		ExpansionCandidate best_expand;
		if (!buildBestExpansionCandidate(phase2_samples, true, best_expand))
			continue;

		RRTNode *x_nearest = best_expand.x_nearest;
		Vector3d x_new = best_expand.x_new;
		double edge_cost = best_expand.edge_cost;
		RRTNode *x_new_node = createNode(x_new, x_nearest, x_nearest->cost + edge_cost, node_id_counter_++);
		x_nearest->addChild(x_new_node);
		nodes_.push_back(x_new_node);
		insertNodeToSpatialIndex((int)nodes_.size() - 1);
		id_to_node_[x_new_node->id] = x_new_node;

		// Rebuild kd-tree every 50 nodes to include new nodes
		if (nodes_.size() % 50 == 0)
		{
			kd_size_ = (int)nodes_.size();
			kdtree_pts_.resize(kd_size_);
			kdtree_idx_.resize(kd_size_);
			id_to_node_.clear();
			for (int i = 0; i < kd_size_; ++i)
			{
				kdtree_pts_[i] = nodes_[i]->x;
				kdtree_idx_[i] = i;                  // store tree INDEX, not node id
				id_to_node_[nodes_[i]->id] = nodes_[i]; // node id → node pointer map
			}
			std::iota(kdtree_idx_.begin(), kdtree_idx_.end(), 0); // reset to 0..kd_size_-1
			kdTreeBuild(0, kd_size_ - 1, 0);
		}

		tryConnectToGoal(x_new_node);
		rewire(x_new_node);

		if (iter % 20 == 0)
			tryShortcutPath();

		if (iter % 20 == 0 && nodes_.size() > 5)
		{
			int ri = rand() % nodes_.size();
			RRTNode *rn = nodes_[ri];
			if (rn != start_node && rn != solution_node_)
				tryConnectToGoal(rn);
		}
	}

	tryShortcutPath();

	ros::Time time_end = ros::Time::now();
	ROS_INFO("[InformedRRT*] Done. nodes=%zu, best_cost=%.3f, time=%.3fs, iters=%d",
			 nodes_.size(), best_cost_, (time_end - time_start).toSec(), iter);

	return (solution_node_ != NULL);
}

} // namespace informed_rrt_star_planner

#include "path_searching/informed_rrt_star.h"

using namespace std;
using namespace Eigen;

namespace informed_rrt_star_planner
{

Vector3d InformedRRTstar::steer(const RRTNode *from, const Vector3d &to)
{
	Vector3d diff = to - from->x;
	double dist = diff.norm();
	if (dist <= step_size_)
		return to;
	return from->x + diff / dist * step_size_;
}

double InformedRRTstar::edgeCostWithTurnPenalty(const RRTNode *from, const Eigen::Vector3d &to) const
{
	double length = (to - from->x).norm();
	if (from->parent == NULL || length < 1e-6)
		return length;

	Vector3d prev = from->x - from->parent->x;
	double prev_len = prev.norm();
	if (prev_len < 1e-6)
		return length;

	double cos_angle = std::max(-1.0, std::min(1.0, prev.dot(to - from->x) / (prev_len * length)));
	double angle_penalty = 0.8 * length * (1.0 - cos_angle);
	return length + angle_penalty;
}

InformedRRTstar::ExpansionCandidate InformedRRTstar::evaluateExpansionCandidate(const Eigen::Vector3d &x_random, bool use_kdtree)
{
	ExpansionCandidate out;
	RRTNode *x_nearest = use_kdtree ? kdTreeNearestNeighbor(x_random) : nearestNeighborBruteForce(x_random);
	if (x_nearest == NULL)
		return out;

	Eigen::Vector3d x_new = steer(x_nearest, x_random);
	if (checkOccupancy(x_new))
	{
		Eigen::Vector3d outside;
		if (!projectToObstacleOutside(x_new, x_random - x_nearest->x, outside))
			return out;
		x_new = steer(x_nearest, outside);
		if (checkOccupancy(x_new))
			return out;
	}
	if (!isSegmentFree(x_nearest->x, x_new))
	{
		Eigen::Vector3d blocking_center, path_dir, outside;
		if (!findBlockingObstacle(x_nearest->x, x_new, blocking_center, path_dir) ||
			!projectToObstacleOutside(blocking_center, path_dir, outside))
			return out;
		x_new = steer(x_nearest, outside);
		if (checkOccupancy(x_new) || !isSegmentFree(x_nearest->x, x_new))
			return out;
	}

	out.valid = true;
	out.x_nearest = x_nearest;
	out.x_new = x_new;
	out.edge_cost = edgeCostWithTurnPenalty(x_nearest, x_new);
	out.goal_heuristic = (x_new - end_pt_).squaredNorm();
	return out;
}

bool InformedRRTstar::buildBestExpansionCandidate(const std::vector<Eigen::Vector3d> &samples, bool use_kdtree, ExpansionCandidate &best)
{
	best.valid = false;
	if (samples.empty())
		return false;

	// Only parallelize when tree is large enough; for tiny trees,
	// async scheduling overhead often outweighs candidate evaluation.
	const bool use_parallel_eval =
		enable_parallel_expansion_ && samples.size() >= 3 && nodes_.size() >= 160;

	if (use_parallel_eval)
	{
		std::vector<std::future<ExpansionCandidate>> futures;
		futures.reserve(samples.size());
		for (const auto &s : samples)
		{
			futures.emplace_back(std::async(std::launch::async, [this, s, use_kdtree]() {
				return evaluateExpansionCandidate(s, use_kdtree);
			}));
		}

		for (auto &f : futures)
		{
			ExpansionCandidate cand = f.get();
			if (!cand.valid)
				continue;
			if (!best.valid || cand.goal_heuristic < best.goal_heuristic)
			{
				best = cand;
			}
		}
	}
	else
	{
		for (const auto &s : samples)
		{
			ExpansionCandidate cand = evaluateExpansionCandidate(s, use_kdtree);
			if (!cand.valid)
				continue;
			if (!best.valid || cand.goal_heuristic < best.goal_heuristic)
			{
				best = cand;
			}
		}
	}

	return best.valid;
}

bool InformedRRTstar::canConnectToGoalWithClearance(RRTNode *node, const Vector3d &goal_pt, double &out_new_cost)
{
	if (checkOccupancy(goal_pt) || !isSegmentFree(node->x, goal_pt))
		return false;

	out_new_cost = node->cost + edgeCostWithTurnPenalty(node, goal_pt);
	if (out_new_cost >= best_cost_ - 1e-9)
		return false;

	// ---- Adaptive front-end clearance gate ----
	// The RRT* path is only a *guide*; the B-spline back-end (lambda_collision with dist0_)
	// is what pushes the executed trajectory to its real safety margin. But the guide path
	// must still stay clear of obstacles, otherwise the optimizer starts from inside/grazing
	// an obstacle and the UAV drifts into it.
	//
	// So the required margin ADAPTS to what is actually achievable around the goal, but is
	// never blanket-relaxed:
	//   * In open space (goal has plenty of room) we keep the FULL min_path_clearance_, so
	//     guide paths never hug obstacles unnecessarily.
	//   * Only when the local target itself sits in a tight corridor do we relax down to the
	//     goal's own achievable clearance (never below a hard floor of ~half a voxel), so the
	//     search still returns a path instead of looping on "no solution".
	// This applies to the first solution too -- using a flat hard floor for the first
	// solution let guide paths cut into obstacles in open areas (the "drift" regression).
	const double hard_floor = std::max(0.5 * resolution_, 0.10);
	const double goal_local_clear = queryClearance(goal_pt);
	const double required_clear = std::max(hard_floor, std::min(min_path_clearance_, goal_local_clear * 0.95));

	// Whole-guide-path margin: segment to goal + short forward look-ahead + back-path scan.
	double path_clearance = std::min(queryPathClearance(node->x, goal_pt),
									 evaluateForwardClearance(node, goal_pt));
	path_clearance = std::min(path_clearance, evaluatePathClearance(node));

	if (path_clearance < required_clear)
	{
		ROS_WARN_THROTTLE(2.0,
						  "[InformedRRT*] Path clearance too low (%.3f < %.3f), rejecting goal connection",
						  path_clearance, required_clear);
		return false;
	}
	return true;
}

bool InformedRRTstar::tryConnectToGoal(RRTNode *node)
{
	double new_cost;
	if (!canConnectToGoalWithClearance(node, end_pt_, new_cost))
		return false;

	if (nodes_.size() >= max_nodes_)
	{
		ROS_WARN_THROTTLE(1.0,
						  "[InformedRRT*] Node limit reached (%zu), skip goal connection",
						  nodes_.size());
		return false;
	}

	RRTNode *goal_node = createNode(end_pt_, node, new_cost, node_id_counter_++);
	nodes_.push_back(goal_node);
	insertNodeToSpatialIndex((int)nodes_.size() - 1);
	id_to_node_[goal_node->id] = goal_node;
	node->addChild(goal_node);
	solution_node_ = goal_node;
	best_cost_ = new_cost;
	c_best_ = new_cost;
	computeEllipse();
	invalidatePathCache();
	return true;
}

// ============================================================
// OPTIMIZATION 2: child pointers + efficient rewire
// ============================================================

void InformedRRTstar::rewire(RRTNode *new_node)
{
	double base_radius = 1.5 * step_size_;
	double radius_growth = sqrt((double)nodes_.size()) * step_size_;
	double rewiring_radius = min(base_radius + radius_growth, 5.0 * step_size_);
	std::vector<RRTNode *> nearby_nodes;
	queryNearbyNodes(new_node->x, rewiring_radius, nearby_nodes);

	for (RRTNode *neighbor : nearby_nodes)
	{
		if (neighbor == new_node)
			continue;

		double dist = (neighbor->x - new_node->x).norm();
		if (dist > rewiring_radius)
			continue;

		double new_cost_through = new_node->cost + edgeCostWithTurnPenalty(new_node, neighbor->x);

		if (new_cost_through < neighbor->cost - 1e-6)
		{
			if (isSegmentFree(new_node->x, neighbor->x))
			{
				// Update child list of old parent
				if (neighbor->parent != NULL)
					neighbor->parent->removeChild(neighbor);

				neighbor->parent = new_node;
				neighbor->cost = new_cost_through;
				new_node->addChild(neighbor);

			// BFS cost propagation through children
			invalidatePathCache();

			bfs_queue_.clear();
			visited_ids_.clear();
			bfs_queue_.push_back(neighbor->id);

			while (!bfs_queue_.empty())
			{
				int curr_id = bfs_queue_.back();
				bfs_queue_.pop_back();

				RRTNode *curr = id_to_node_.count(curr_id) ? id_to_node_[curr_id] : NULL;
				if (curr == NULL || visited_ids_.count(curr_id))
					continue;
				visited_ids_.insert(curr_id);

					for (RRTNode *child : curr->children)
					{
						double new_child_cost = curr->cost + edgeCostWithTurnPenalty(curr, child->x);
						if (new_child_cost < child->cost - 1e-6)
						{
							child->cost = new_child_cost;
							bfs_queue_.push_back(child->id);
						}
					}
				}
			}
		}
	}
}

// ============================================================
// OPTIMIZATION 3: path cache for traceBack
// ============================================================

vector<RRTNode *> InformedRRTstar::traceBack(RRTNode *node)
{
	if (path_cache_valid_ && node == solution_node_ && !path_cache_.empty())
		return path_cache_;

	vector<RRTNode *> path;
	RRTNode *curr = node;
	while (curr != NULL)
	{
		path.push_back(curr);
		curr = curr->parent;
	}
	reverse(path.begin(), path.end());

	if (node == solution_node_)
	{
		path_cache_ = path;
		path_cache_valid_ = true;
	}

	return path;
}

// ============================================================
// Shortcut optimization
// ============================================================

void InformedRRTstar::tryShortcutPath()
{
	if (solution_node_ == NULL)
		return;

	bool improved = true;
	int shortcut_attempts = 0;
	const int MAX_ATTEMPTS = 100;

	while (improved && shortcut_attempts < MAX_ATTEMPTS)
	{
		++shortcut_attempts;
		improved = false;

		vector<RRTNode *> current_path = traceBack(solution_node_);
		if (current_path.size() < 3)
			break;

		for (size_t i = 0; i < current_path.size() - 2 && !improved; ++i)
		{
			for (size_t j = i + 2; j < current_path.size() && !improved; ++j)
			{
				if (isSegmentClearanceAtLeast(current_path[i]->x, current_path[j]->x, min_path_clearance_ * 0.72))
				{
					double new_cost = current_path[i]->cost + edgeCostWithTurnPenalty(current_path[i], current_path[j]->x);

					if (new_cost < current_path[j]->cost - 1e-4)
					{
						RRTNode *node_j = current_path[j];

						if (node_j->parent != NULL)
							node_j->parent->removeChild(node_j);

						node_j->parent = current_path[i];
						node_j->cost = new_cost;
						current_path[i]->addChild(node_j);

					invalidatePathCache();

					bfs_queue_.clear();
					visited_ids_.clear();
					bfs_queue_.push_back(node_j->id);

					while (!bfs_queue_.empty())
					{
						int curr_id = bfs_queue_.back();
						bfs_queue_.pop_back();

						RRTNode *curr = id_to_node_.count(curr_id) ? id_to_node_[curr_id] : NULL;
						if (curr == NULL || visited_ids_.count(curr_id))
							continue;
						visited_ids_.insert(curr_id);

						for (RRTNode *child : curr->children)
						{
							double new_child_cost = curr->cost + edgeCostWithTurnPenalty(curr, child->x);
							if (new_child_cost < child->cost - 1e-6)
							{
								child->cost = new_child_cost;
								bfs_queue_.push_back(child->id);
							}
						}
						}

						if (node_j == solution_node_ || new_cost < best_cost_)
						{
							solution_node_ = node_j;
							best_cost_ = new_cost;
							c_best_ = new_cost;
							computeEllipse();
						}

						improved = true;
					}
				}
			}
		}
	}
}

} // namespace informed_rrt_star_planner

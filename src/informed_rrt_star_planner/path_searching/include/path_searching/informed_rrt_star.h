#ifndef _INFORMED_RRT_STAR_H_
#define _INFORMED_RRT_STAR_H_

#include <iostream>
#include <ros/ros.h>
#include <ros/console.h>
#include <Eigen/Eigen>
#include <plan_env/grid_map.h>
#include <queue>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace ego_planner
{

struct RRTNode
{
	Eigen::Vector3d x;
	RRTNode *parent;
	double cost;
	int id;
	std::vector<RRTNode *> children;

	RRTNode(const Eigen::Vector3d &_x, RRTNode *_parent, double _cost, int _id)
		: x(_x), parent(_parent), cost(_cost), id(_id) {}

	inline void addChild(RRTNode *child) { children.push_back(child); }
	inline void removeChild(RRTNode *child)
	{
		for (auto it = children.begin(); it != children.end(); ++it)
			if (*it == child) { children.erase(it); break; }
	}
};

class InformedRRTstar
{
public:
	typedef std::shared_ptr<InformedRRTstar> Ptr;

	InformedRRTstar();
	~InformedRRTstar();

	void initGridMap(GridMap::Ptr occ_map, const Eigen::Vector3i pool_size);
	bool InformedRRTstarSearch(const double step_size, Eigen::Vector3d start_pt, Eigen::Vector3d end_pt);
	std::vector<Eigen::Vector3d> getPath();

private:
	GridMap::Ptr grid_map_;
	inline bool checkOccupancy(const Eigen::Vector3d &pos)
	{
		return (bool)grid_map_->getInflateOccupancy(pos);
	}

	Eigen::Vector3d workspace_min_;
	Eigen::Vector3d workspace_max_;
	double resolution_;

	double step_size_;
	Eigen::Vector3d start_pt_;
	Eigen::Vector3d end_pt_;
	double rrt_max_time_;
	double opt_max_time_;
	int max_iterations_;
	size_t max_nodes_;
	double goal_bias_;
	double apf_sampling_ratio_;
	double apf_attr_gain_;
	double apf_rep_gain_;
	double apf_rep_radius_;
	double min_path_clearance_;

	// If the tree root was snapped away from the caller's start for clearance, prepend this in getPath().
	Eigen::Vector3d path_prefix_start_;
	bool has_path_prefix_;

	// Tree data
	std::vector<RRTNode *> nodes_;
	int node_id_counter_;
	RRTNode *solution_node_;
	double best_cost_;

	// Informed ellipse
	double c_best_;
	double c_min_;
	Eigen::Vector3d center_;
	Eigen::Matrix3d C_;

	// kd-tree for fast nearest neighbor in Phase 2
	// Uses an id-indexed lookup to avoid expensive pointer chasing during search
	std::vector<Eigen::Vector3d> kdtree_pts_;     // kdtree_pts_[i] = position of node at tree index i
	std::vector<int> kdtree_idx_;                  // median-partitioned node indices for kd-tree search
	std::unordered_map<int, RRTNode *> id_to_node_; // id_to_node_[id] = node pointer for O(1) lookup
	int kd_size_;
	int kdTreeBuild(int left, int right, int depth);
	void kdTreeNNRecursive(int left, int right, int depth, const Eigen::Vector3d &query,
						   int &best_node_id, double &best_dist) const;
	inline RRTNode *kdTreeNearestNeighbor(const Eigen::Vector3d &x);

	// ===== Optimization 3: path cache + reuse buffers =====
	bool path_cache_valid_;
	std::vector<RRTNode *> path_cache_;
	std::unordered_set<int> visited_ids_; // tracks visited node IDs (not array indices)
	std::vector<int> bfs_queue_;
	inline void invalidatePathCache()
	{
		path_cache_valid_ = false;
	}

	// ===== Optimization 5: APF clearance cache =====
	struct APFCacheEntry
	{
		Eigen::Vector3d pos;
		double clear;
		Eigen::Vector3d grad;
		uint64_t frame;
	};
	std::vector<APFCacheEntry> apf_cache_;
	uint64_t apf_cache_frame_;
	static constexpr int APF_CACHE_SIZE = 256;
	bool apfCacheLookup(const Eigen::Vector3d &pos, double &out_clear, Eigen::Vector3d &out_grad);
	void apfCacheStore(const Eigen::Vector3d &pos, double clear, const Eigen::Vector3d &grad);

	// ===== Fast PRNG =====
	std::vector<double> rand_buf_;
	int rand_pos_;
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

	// ===== Core methods =====
	void computeEllipse();
	Eigen::Vector3d sampleEllipse();
	Eigen::Vector3d sampleWorkspace();
	Eigen::Vector3d sampleEllipseSafe();  // 新增：带安全性检查的椭圆采样
	bool findBlockingObstacle(const Eigen::Vector3d &from, const Eigen::Vector3d &to,
							  Eigen::Vector3d &obs_center, Eigen::Vector3d &path_dir);
	bool projectToObstacleOutside(const Eigen::Vector3d &obs_pos, const Eigen::Vector3d &preferred_dir,
								  Eigen::Vector3d &outside_pt);
	bool sampleObstacleOutside(Eigen::Vector3d &x_sample);
	double queryClearance(const Eigen::Vector3d &pos);  // 新增：查询某点安全裕量
	double queryPathClearance(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2);  // 新增：查询路径安全裕量
	double evaluatePathClearance(RRTNode *node);  // 新增：评估路径安全性
	double evaluateForwardClearance(RRTNode *node, const Eigen::Vector3d &final_pt);
	bool findNearestFreePoint(const Eigen::Vector3d &origin, const Eigen::Vector3d &preferred_dir,
							  double max_radius, double min_clearance, Eigen::Vector3d &free_pt);
	double edgeCostWithTurnPenalty(const RRTNode *from, const Eigen::Vector3d &to) const;
	RRTNode *nearestNeighborBruteForce(const Eigen::Vector3d &x);
	Eigen::Vector3d steer(const RRTNode *from, const Eigen::Vector3d &to);
	bool isSegmentFree(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2);
	bool tryConnectToGoal(RRTNode *node);
	bool canConnectToGoalWithClearance(RRTNode *node, const Eigen::Vector3d &goal_pt, double &out_new_cost);
	void rewire(RRTNode *new_node);
	std::vector<RRTNode *> traceBack(RRTNode *node);
	void clearTree();
	double attractivePotential(const Eigen::Vector3d &pos, Eigen::Vector3d &grad);
	double repulsivePotential(const Eigen::Vector3d &pos, Eigen::Vector3d &grad);
	Eigen::Vector3d sampleWithAPF();
	void tryShortcutPath();
};

} // namespace ego_planner

#endif

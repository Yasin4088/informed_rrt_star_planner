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
#include <cstdint>
#include <future>
#include <mutex>

#include <path_searching/rrt_node.h>

namespace informed_rrt_star_planner
{

// Informed RRT* with APF-biased sampling, kd-tree nearest neighbor and a
// clearance-aware cost. The implementation is split across several .cpp files
// (grouped by concern) for maintainability:
//   - informed_rrt_star.cpp     : lifecycle / params / clearTree
//   - rrt_star_search.cpp       : main search loop + getPath (smoothing)
//   - rrt_star_nearest.cpp      : kd-tree build/query + brute force + spatial hash
//   - rrt_star_collision.cpp    : occupancy + clearance queries
//   - rrt_star_sampling.cpp     : workspace/ellipse/APF sampling
//   - rrt_star_tree_ops.cpp     : steer/expand/connect/rewire/shortcut
// Small hot helpers are defined inline in informed_rrt_star_inl.h.
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
	bool checkOccupancy(const Eigen::Vector3d &pos);

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
	bool enable_parallel_expansion_;
	int parallel_batch_size_;
	double apf_sampling_ratio_;
	double apf_rep_gain_;
	double apf_rep_radius_;
	double min_path_clearance_;

	// If the tree root was snapped away from the caller's start for clearance, prepend this in getPath().
	Eigen::Vector3d path_prefix_start_;
	bool has_path_prefix_;

	// Cached straight-line blocking obstacle for start_pt_ -> end_pt_. The blocking obstacle
	// on the straight line does not change within a single search, so we ray-march for it once
	// instead of on (potentially) every sampleObstacleOutside() call.
	bool blocking_obstacle_cached_;
	bool blocking_obstacle_valid_;
	Eigen::Vector3d cached_obs_center_;
	Eigen::Vector3d cached_path_dir_;
	bool getCachedStraightLineBlocker(Eigen::Vector3d &obs_center, Eigen::Vector3d &path_dir);

	// Tree data
	std::vector<RRTNode> node_storage_;
	std::vector<RRTNode *> nodes_;
	int node_id_counter_;
	RRTNode *solution_node_;
	double best_cost_;
	inline RRTNode *createNode(const Eigen::Vector3d &x, RRTNode *parent, double cost, int id);
	struct ExpansionCandidate
	{
		bool valid{false};
		Eigen::Vector3d x_new;
		RRTNode *x_nearest{nullptr};
		double edge_cost{0.0};
		double goal_heuristic{0.0};
	};
	ExpansionCandidate evaluateExpansionCandidate(const Eigen::Vector3d &x_random, bool use_kdtree);
	bool buildBestExpansionCandidate(const std::vector<Eigen::Vector3d> &samples, bool use_kdtree, ExpansionCandidate &best);
	struct OccupancyCacheEntry
	{
		bool occupied;
		uint64_t frame;
	};
	std::unordered_map<int64_t, OccupancyCacheEntry> occupancy_cache_;
	std::mutex occupancy_cache_mtx_;
	uint64_t occupancy_cache_frame_;
	static constexpr int OCCUPANCY_CACHE_SIZE = 18000;
	inline int64_t occupancyCacheKey(const Eigen::Vector3d &pos) const;

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

	// ===== Optimization 2.5: spatial hash for fast rewire neighborhood =====
	std::unordered_map<int64_t, std::vector<int>> node_spatial_index_;
	double node_spatial_cell_size_;
	inline int64_t nodeSpatialKeyFromCoord(int ix, int iy, int iz) const;
	inline void nodeSpatialCoord(const Eigen::Vector3d &pos, int &ix, int &iy, int &iz) const;
	inline void resetNodeSpatialIndex(double cell_size);
	inline void insertNodeToSpatialIndex(int node_idx);
	void queryNearbyNodes(const Eigen::Vector3d &center, double radius, std::vector<RRTNode *> &out_nodes) const;

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

	// ===== Optimization 6: per-search voxel clearance cache =====
	struct ClearanceCacheEntry
	{
		double clearance;
		uint64_t frame;
	};
	std::unordered_map<int64_t, ClearanceCacheEntry> clearance_cache_;
	uint64_t clearance_cache_frame_;
	static constexpr int CLEARANCE_CACHE_SIZE = 12000;
	inline int64_t clearanceCacheKey(const Eigen::Vector3d &pos) const;
	inline bool clearanceCacheLookup(const Eigen::Vector3d &pos, double &clearance_out);
	inline void clearanceCacheStore(const Eigen::Vector3d &pos, double clearance);

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
	Eigen::Vector3d sampleEllipseSafe();  //带安全性检查的椭圆采样
	bool findBlockingObstacle(const Eigen::Vector3d &from, const Eigen::Vector3d &to,
							  Eigen::Vector3d &obs_center, Eigen::Vector3d &path_dir);
	bool projectToObstacleOutside(const Eigen::Vector3d &obs_pos, const Eigen::Vector3d &preferred_dir,
								  Eigen::Vector3d &outside_pt);
	bool sampleObstacleOutside(Eigen::Vector3d &x_sample);
	double queryClearance(const Eigen::Vector3d &pos);  // 查询某点安全裕量
	bool hasClearanceAtLeast(const Eigen::Vector3d &pos, double min_clearance);
	bool isSegmentClearanceAtLeast(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, double min_clearance);
	double queryPathClearance(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2);  // 查询路径安全裕量
	double evaluatePathClearance(RRTNode *node);  // 评估路径安全性
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
	double repulsivePotential(const Eigen::Vector3d &pos, Eigen::Vector3d &grad);
	Eigen::Vector3d sampleWithAPF();
	void tryShortcutPath();
};

} // namespace informed_rrt_star_planner

// Inline definitions of small hot helpers (must follow the class declaration).
#include <path_searching/informed_rrt_star_inl.h>

#endif // _INFORMED_RRT_STAR_H_

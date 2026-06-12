#ifndef _INFORMED_RRT_STAR_INL_H_
#define _INFORMED_RRT_STAR_INL_H_

// Inline definitions of the small, hot helper members of InformedRRTstar.
// They live here (instead of a .cpp) so they stay inline across all the
// translation units that make up the planner, avoiding cross-TU call overhead
// while keeping the declaration header (informed_rrt_star.h) easy to read.
// This file is included at the bottom of informed_rrt_star.h.

#include <algorithm>
#include <cmath>
#include <limits>

namespace informed_rrt_star_planner
{

inline RRTNode *InformedRRTstar::createNode(const Eigen::Vector3d &x, RRTNode *parent, double cost, int id)
{
	if (node_storage_.capacity() == 0)
		node_storage_.reserve(max_nodes_ + 16);
	node_storage_.emplace_back(x, parent, cost, id);
	return &node_storage_.back();
}

inline int64_t InformedRRTstar::occupancyCacheKey(const Eigen::Vector3d &pos) const
{
	const double cell = std::max(0.04, resolution_ * 0.5);
	const int64_t ix = static_cast<int64_t>(std::floor((pos(0) - workspace_min_(0)) / cell));
	const int64_t iy = static_cast<int64_t>(std::floor((pos(1) - workspace_min_(1)) / cell));
	const int64_t iz = static_cast<int64_t>(std::floor((pos(2) - workspace_min_(2)) / cell));
	return (ix * 73856093LL) ^ (iy * 19349663LL) ^ (iz * 83492791LL);
}

inline int64_t InformedRRTstar::nodeSpatialKeyFromCoord(int ix, int iy, int iz) const
{
	return ((int64_t)ix * 73856093LL) ^ ((int64_t)iy * 19349663LL) ^ ((int64_t)iz * 83492791LL);
}

inline void InformedRRTstar::nodeSpatialCoord(const Eigen::Vector3d &pos, int &ix, int &iy, int &iz) const
{
	const double cell = std::max(0.05, node_spatial_cell_size_);
	ix = static_cast<int>(std::floor((pos(0) - workspace_min_(0)) / cell));
	iy = static_cast<int>(std::floor((pos(1) - workspace_min_(1)) / cell));
	iz = static_cast<int>(std::floor((pos(2) - workspace_min_(2)) / cell));
}

inline void InformedRRTstar::resetNodeSpatialIndex(double cell_size)
{
	node_spatial_cell_size_ = std::max(0.05, cell_size);
	node_spatial_index_.clear();
	node_spatial_index_.reserve(std::max<size_t>(256, max_nodes_ / 2));
}

inline void InformedRRTstar::insertNodeToSpatialIndex(int node_idx)
{
	if (node_idx < 0 || node_idx >= (int)nodes_.size())
		return;

	int ix, iy, iz;
	nodeSpatialCoord(nodes_[node_idx]->x, ix, iy, iz);
	node_spatial_index_[nodeSpatialKeyFromCoord(ix, iy, iz)].push_back(node_idx);
}

inline int64_t InformedRRTstar::clearanceCacheKey(const Eigen::Vector3d &pos) const
{
	const double cell = std::max(0.02, resolution_ * 0.5);
	const int64_t ix = static_cast<int64_t>(std::floor((pos(0) - workspace_min_(0)) / cell));
	const int64_t iy = static_cast<int64_t>(std::floor((pos(1) - workspace_min_(1)) / cell));
	const int64_t iz = static_cast<int64_t>(std::floor((pos(2) - workspace_min_(2)) / cell));

	// 64-bit spatial hash for voxel index triple.
	return (ix * 73856093LL) ^ (iy * 19349663LL) ^ (iz * 83492791LL);
}

inline bool InformedRRTstar::clearanceCacheLookup(const Eigen::Vector3d &pos, double &clearance_out)
{
	const int64_t key = clearanceCacheKey(pos);
	auto it = clearance_cache_.find(key);
	if (it == clearance_cache_.end())
		return false;
	if (it->second.frame != clearance_cache_frame_)
		return false;
	clearance_out = it->second.clearance;
	return true;
}

inline void InformedRRTstar::clearanceCacheStore(const Eigen::Vector3d &pos, double clearance)
{
	if ((int)clearance_cache_.size() > CLEARANCE_CACHE_SIZE)
		clearance_cache_.clear();

	ClearanceCacheEntry entry;
	entry.clearance = clearance;
	entry.frame = clearance_cache_frame_;
	clearance_cache_[clearanceCacheKey(pos)] = entry;
}

inline RRTNode *InformedRRTstar::kdTreeNearestNeighbor(const Eigen::Vector3d &x)
{
	if (kdtree_pts_.empty())
		return NULL;

	int best_node_id = -1;
	double best_dist = std::numeric_limits<double>::infinity();
	kdTreeNNRecursive(0, kd_size_ - 1, 0, x, best_node_id, best_dist);

	if (best_node_id < 0 || best_node_id >= (int)nodes_.size())
		return NULL;
	return nodes_[best_node_id];
}

} // namespace informed_rrt_star_planner

#endif // _INFORMED_RRT_STAR_INL_H_

#include "path_searching/informed_rrt_star.h"

using namespace std;
using namespace Eigen;

namespace informed_rrt_star_planner
{

// ============================================================
// Spatial hash neighborhood query (Optimization 2.5)
// ============================================================

void InformedRRTstar::queryNearbyNodes(const Eigen::Vector3d &center, double radius, std::vector<RRTNode *> &out_nodes) const
{
	out_nodes.clear();
	if (nodes_.empty())
		return;
	if (node_spatial_index_.empty())
	{
		out_nodes.reserve(nodes_.size());
		for (RRTNode *n : nodes_)
			out_nodes.push_back(n);
		return;
	}

	const double cell = std::max(0.05, node_spatial_cell_size_);
	int cix, ciy, ciz;
	nodeSpatialCoord(center, cix, ciy, ciz);
	const int reach = std::max(1, (int)std::ceil(radius / cell));
	const double radius2 = radius * radius;

	for (int dx = -reach; dx <= reach; ++dx)
	{
		for (int dy = -reach; dy <= reach; ++dy)
		{
			for (int dz = -reach; dz <= reach; ++dz)
			{
				const int64_t key = nodeSpatialKeyFromCoord(cix + dx, ciy + dy, ciz + dz);
				auto it = node_spatial_index_.find(key);
				if (it == node_spatial_index_.end())
					continue;

				for (int idx : it->second)
				{
					if (idx < 0 || idx >= (int)nodes_.size())
						continue;
					RRTNode *cand = nodes_[idx];
					if ((cand->x - center).squaredNorm() <= radius2)
						out_nodes.push_back(cand);
				}
			}
		}
	}
}

// ============================================================
// OPTIMIZATION 1: kd-tree for O(log n) nearest neighbor
// ============================================================

int InformedRRTstar::kdTreeBuild(int left, int right, int depth)
{
	if (left > right)
		return -1;

	int mid = (left + right) / 2;
	int dim = depth % 3;

	std::nth_element(kdtree_idx_.begin() + left,
					 kdtree_idx_.begin() + mid,
					 kdtree_idx_.begin() + right + 1,
					 [dim, this](int a, int b) {
						 return kdtree_pts_[a](dim) < kdtree_pts_[b](dim);
					 });

	kdTreeBuild(left, mid - 1, depth + 1);
	kdTreeBuild(mid + 1, right, depth + 1);

	return kdtree_idx_[mid];
}

void InformedRRTstar::kdTreeNNRecursive(int left, int right, int depth,
									   const Eigen::Vector3d &query,
									   int &best_node_id, double &best_dist) const
{
	if (left > right)
		return;

	int mid = (left + right) / 2;
	int node_id = kdtree_idx_[mid];
	const Vector3d &node_pos = kdtree_pts_[node_id];
	double dist = (node_pos - query).norm();
	if (dist < best_dist)
	{
		best_dist = dist;
		best_node_id = node_id;
	}

	int dim = depth % 3;
	double diff = query(dim) - node_pos(dim);

	// kdtree_idx_ stores a median-partitioned array, not a heap layout.
	// Recurse by subarray bounds so the nearest-neighbor pruning matches kdTreeBuild().
	if (diff <= 0)
	{
		kdTreeNNRecursive(left, mid - 1, depth + 1, query, best_node_id, best_dist);
		if (fabs(diff) < best_dist)
			kdTreeNNRecursive(mid + 1, right, depth + 1, query, best_node_id, best_dist);
	}
	else
	{
		kdTreeNNRecursive(mid + 1, right, depth + 1, query, best_node_id, best_dist);
		if (fabs(diff) < best_dist)
			kdTreeNNRecursive(left, mid - 1, depth + 1, query, best_node_id, best_dist);
	}
}

RRTNode *InformedRRTstar::nearestNeighborBruteForce(const Vector3d &x)
{
	if (nodes_.empty())
		return NULL;

	double best_dist = numeric_limits<double>::infinity();
	RRTNode *best = NULL;
	for (size_t i = 0; i < nodes_.size(); ++i)
	{
		double d = (nodes_[i]->x - x).norm();
		if (d < best_dist)
		{
			best_dist = d;
			best = nodes_[i];
		}
	}
	return best;
}

} // namespace informed_rrt_star_planner

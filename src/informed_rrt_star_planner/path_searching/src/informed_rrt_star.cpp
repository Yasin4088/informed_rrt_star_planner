#include "path_searching/informed_rrt_star.h"

using namespace std;
using namespace Eigen;

namespace ego_planner
{

InformedRRTstar::InformedRRTstar()
	: step_size_(0.25),
	  node_id_counter_(0),
	  solution_node_(NULL),
	  best_cost_(numeric_limits<double>::infinity()),
	  c_best_(numeric_limits<double>::infinity()),
	  c_min_(0.0),
	  rrt_max_time_(0.5),
	  opt_max_time_(0.10),
	  max_iterations_(10000),
	  max_nodes_(6000),
	  goal_bias_(0.15),
	  apf_sampling_ratio_(0.35),    // 斥力采样比例
	  apf_attr_gain_(1.0),          // 吸引力增益
	  apf_rep_gain_(2.0),           // 斥力增益
	  apf_rep_radius_(1.5),         // 斥力范围
	  min_path_clearance_(0.25),    
	  has_path_prefix_(false),
	  path_cache_valid_(false),
	  apf_cache_frame_(0),
	  rand_pos_(-1)
{
	rand_buf_.resize(4096);
	for (auto &v : rand_buf_)
		v = (double)rand() / RAND_MAX;
}

InformedRRTstar::~InformedRRTstar()
{
	clearTree();
}

void InformedRRTstar::initGridMap(GridMap::Ptr occ_map, const Eigen::Vector3i pool_size)
{
	grid_map_ = occ_map;
	resolution_ = grid_map_->getResolution();
	workspace_min_ = grid_map_->getMinBoundary();
	workspace_max_ = grid_map_->getMaxBoundary();

	ROS_INFO("[InformedRRT*] initGridMap: resolution=%.3f, workspace_min=(%.2f,%.2f,%.2f), workspace_max=(%.2f,%.2f,%.2f)",
			 resolution_, workspace_min_(0), workspace_min_(1), workspace_min_(2),
			 workspace_max_(0), workspace_max_(1), workspace_max_(2));
}

void InformedRRTstar::clearTree()
{
	for (size_t i = 0; i < nodes_.size(); ++i)
		delete nodes_[i];
	nodes_.clear();
	node_id_counter_ = 0;
	solution_node_ = NULL;
	best_cost_ = numeric_limits<double>::infinity();
	kdtree_pts_.clear();
	kdtree_idx_.clear();
	id_to_node_.clear();
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

inline RRTNode *InformedRRTstar::kdTreeNearestNeighbor(const Vector3d &x)
{
	if (kdtree_pts_.empty())
		return NULL;

	int best_node_id = -1;
	double best_dist = numeric_limits<double>::infinity();
	kdTreeNNRecursive(0, kd_size_ - 1, 0, x, best_node_id, best_dist);

	if (best_node_id < 0 || best_node_id >= (int)nodes_.size())
		return NULL;
	return nodes_[best_node_id];
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

Vector3d InformedRRTstar::steer(const RRTNode *from, const Vector3d &to)
{
	Vector3d diff = to - from->x;
	double dist = diff.norm();
	if (dist <= step_size_)
		return to;
	return from->x + diff / dist * step_size_;
}

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

bool InformedRRTstar::sampleObstacleOutside(Eigen::Vector3d &x_sample)
{
	constexpr int MAX_ATTEMPTS = 40;
	constexpr double MIN_OUTSIDE_CLEARANCE = 0.65;

	Vector3d obs_center, path_dir;
	if (!findBlockingObstacle(start_pt_, end_pt_, obs_center, path_dir))
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

// ========== 新增：查询某点的安全裕量（距离最近障碍物的距离）==========
double InformedRRTstar::queryClearance(const Eigen::Vector3d &pos)
{
	if (checkOccupancy(pos))
		return 0.0;

	// 扩展为26方向检测（立方体6面 + 12边 + 8角）
	constexpr int NUM_DIRS = 26;
	constexpr double DELTA = 0.05;
	constexpr double MAX_RANGE = 2.0;

	Vector3d ray_dirs[NUM_DIRS] = {
		// 6面
		Vector3d(1, 0, 0), Vector3d(-1, 0, 0),
		Vector3d(0, 1, 0), Vector3d(0, -1, 0),
		Vector3d(0, 0, 1), Vector3d(0, 0, -1),
		// 12边
		Vector3d(1, 1, 0), Vector3d(1, -1, 0), Vector3d(-1, 1, 0), Vector3d(-1, -1, 0),
		Vector3d(1, 0, 1), Vector3d(1, 0, -1), Vector3d(-1, 0, 1), Vector3d(-1, 0, -1),
		Vector3d(0, 1, 1), Vector3d(0, 1, -1), Vector3d(0, -1, 1), Vector3d(0, -1, -1),
		// 8角
		Vector3d(1, 1, 1), Vector3d(1, 1, -1), Vector3d(1, -1, 1), Vector3d(1, -1, -1),
		Vector3d(-1, 1, 1), Vector3d(-1, 1, -1), Vector3d(-1, -1, 1), Vector3d(-1, -1, -1)
	};
	// 归一化所有射线方向
	for (int i = 0; i < NUM_DIRS; ++i)
		ray_dirs[i].normalize();

	double min_dist = MAX_RANGE;

	for (int di = 0; di < NUM_DIRS; ++di)
	{
		Vector3d p = pos;
		for (int step = 0; step < 40; ++step)
		{
			p += ray_dirs[di] * DELTA;

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

	return min_dist;
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

		// 检查是否被障碍物占据
		if (checkOccupancy(x_sample))
			continue;

		// 检查安全裕量
		double clearance = queryClearance(x_sample);
		if (clearance >= MIN_SAFE_CLEARANCE)
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

double InformedRRTstar::attractivePotential(const Eigen::Vector3d &pos, Eigen::Vector3d &grad)
{
	constexpr double D_STAR = 2.0;
	Vector3d d = pos - end_pt_;
	double dist = d.norm();

	if (dist < D_STAR)
	{
		grad = 2.0 * apf_attr_gain_ * d;
		return apf_attr_gain_ * dist * dist;
	}
	else
	{
		grad = 2.0 * apf_attr_gain_ * D_STAR * d.normalized();
		return apf_attr_gain_ * D_STAR * (2.0 * dist - D_STAR);
	}
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

	// 斥力场射线检测：从6方向扩展到26方向（立方体6面 + 12边 + 8角）
	constexpr int NUM_DIRS = 26;
	constexpr double DELTA = 0.05;
	Vector3d ray_dirs[NUM_DIRS] = {
		// 6面
		Vector3d(1, 0, 0), Vector3d(-1, 0, 0),
		Vector3d(0, 1, 0), Vector3d(0, -1, 0),
		Vector3d(0, 0, 1), Vector3d(0, 0, -1),
		// 12边
		Vector3d(1, 1, 0), Vector3d(1, -1, 0), Vector3d(-1, 1, 0), Vector3d(-1, -1, 0),
		Vector3d(1, 0, 1), Vector3d(1, 0, -1), Vector3d(-1, 0, 1), Vector3d(-1, 0, -1),
		Vector3d(0, 1, 1), Vector3d(0, 1, -1), Vector3d(0, -1, 1), Vector3d(0, -1, -1),
		// 8角
		Vector3d(1, 1, 1), Vector3d(1, 1, -1), Vector3d(1, -1, 1), Vector3d(1, -1, -1),
		Vector3d(-1, 1, 1), Vector3d(-1, 1, -1), Vector3d(-1, -1, 1), Vector3d(-1, -1, -1)
	};
	// 归一化所有射线方向
	for (int i = 0; i < NUM_DIRS; ++i)
		ray_dirs[i].normalize();

	double min_dist = numeric_limits<double>::infinity();
	Vector3d closest_dir = Vector3d::Zero();

	for (int di = 0; di < NUM_DIRS; ++di)
	{
		Vector3d p = pos;
		for (int step = 0; step < 200; ++step)
		{
			p += ray_dirs[di] * DELTA;

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
					closest_dir = -ray_dirs[di];
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

	Vector3d grad_att, grad_rep;
	attractivePotential(anchor_pos, grad_att);
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

	// Rodrigues rotation noise
	Vector3d z_axis(0, 0, 1);
	Vector3d axis_yaw = z_axis.cross(dir_total);
	double s_yaw = axis_yaw.norm();
	double c_yaw = z_axis.dot(dir_total);
	if (s_yaw > 1e-8) axis_yaw /= s_yaw;
	Matrix3d R_yaw = Matrix3d::Identity();
	if (s_yaw > 1e-8)
	{
		Matrix3d K;
		K << 0, -axis_yaw(2), axis_yaw(1),
			 axis_yaw(2), 0, -axis_yaw(0),
			-axis_yaw(1), axis_yaw(0), 0;
		R_yaw = Matrix3d::Identity() + s_yaw * K + (1 - c_yaw) * K * K;
	}

	Vector3d dir_noisy = R_yaw * dir_total;

	Vector3d perp = (fabs(dir_noisy(2)) < 0.99)
						? Vector3d(-dir_noisy(1), dir_noisy(0), 0).normalized()
						: Vector3d(1, 0, 0);
	Vector3d axis_pitch = dir_noisy.cross(perp);
	double s_pitch = axis_pitch.norm();
	double c_pitch = dir_noisy.dot(perp);
	if (s_pitch > 1e-8) axis_pitch /= s_pitch;
	Matrix3d R_pitch = Matrix3d::Identity();
	if (s_pitch > 1e-8)
	{
		Matrix3d K;
		K << 0, -axis_pitch(2), axis_pitch(1),
			 axis_pitch(2), 0, -axis_pitch(0),
			-axis_pitch(1), axis_pitch(0), 0;
		R_pitch = Matrix3d::Identity() + s_pitch * K + (1 - c_pitch) * K * K;
	}
	dir_noisy = R_pitch * dir_noisy;

	double blend = 0.7 + 0.3 * fastRand();
	Vector3d dir_final = blend * dir_total + (1.0 - blend) * dir_noisy;
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

	// 新增：检查从锚点到采样点的路径是否安全
	constexpr double PATH_SAFE_DISTANCE = 0.3;  // 路径安全距离阈值
	double path_clearance = queryPathClearance(anchor_pos, x_sample);
	if (path_clearance < PATH_SAFE_DISTANCE)
	{
		// 路径不安全，尝试修正采样方向使其更远离障碍物
		Vector3d safe_dir = dir_final;
		double clearance = queryClearance(anchor_pos);

		// 沿斥力方向后退一个安全距离
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

				path_clearance = queryPathClearance(anchor_pos, x_sample);
				if (path_clearance < PATH_SAFE_DISTANCE * 0.5)
					return sampleWorkspace();  // 仍然不安全，回退
			}
		}
	}

	return x_sample;
}

bool InformedRRTstar::canConnectToGoalWithClearance(RRTNode *node, const Vector3d &goal_pt, double &out_new_cost)
{
	if (checkOccupancy(goal_pt) || !isSegmentFree(node->x, goal_pt))
		return false;

	out_new_cost = node->cost + edgeCostWithTurnPenalty(node, goal_pt);
	if (out_new_cost >= best_cost_ - 1e-9)
		return false;

	double path_clearance = std::min(evaluatePathClearance(node),
									 queryPathClearance(node->x, goal_pt));
	path_clearance = std::min(path_clearance, evaluateForwardClearance(node, goal_pt));
	if (path_clearance < min_path_clearance_)
	{
		ROS_WARN_THROTTLE(1.0,
						  "[InformedRRT*] Path clearance too low (%.3f < %.3f), rejecting goal connection",
						  path_clearance, min_path_clearance_);
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

	RRTNode *goal_node = new RRTNode(end_pt_, node, new_cost, node_id_counter_++);
	nodes_.push_back(goal_node);
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

	for (size_t i = 0; i < nodes_.size(); ++i)
	{
		RRTNode *neighbor = nodes_[i];
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
// Path smoothing
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

		if (isSegmentFree(path[best_idx - 1], path[best_idx + 1]) &&
			queryPathClearance(path[best_idx - 1], path[best_idx + 1]) >= min_path_clearance_ * 0.62)
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

	step_size_ = step_size;
	start_pt_ = start_pt;
	end_pt_ = end_pt;

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
			isSegmentFree(start_pt_, safe_start) &&
			queryPathClearance(start_pt_, safe_start) >= min_path_clearance_ * 0.52)
		{
			path_prefix_start_ = start_pt_;
			start_pt_ = safe_start;
			has_path_prefix_ = true;
			ROS_WARN_THROTTLE(0.5, "[InformedRRT*] Start clearance low: snapping tree root (%.2f m toward free space), prefix in getPath.",
							   (start_pt_ - path_prefix_start_).norm());
		}
	}

	clearTree();

	RRTNode *start_node = new RRTNode(start_pt_, NULL, 0.0, node_id_counter_++);
	nodes_.push_back(start_node);
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

		Vector3d x_random;
		double roll = fastRand();
		const double outside_sample_band = tight_corridor_start ? 0.65 : 0.55;

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

		// Phase 1: brute-force nearest (tree is small, O(n) is fine)
		RRTNode *x_nearest = nearestNeighborBruteForce(x_random);
		if (x_nearest == NULL)
			continue;

		Vector3d x_new = steer(x_nearest, x_random);
		if (checkOccupancy(x_new))
		{
			Vector3d outside;
			if (!projectToObstacleOutside(x_new, x_random - x_nearest->x, outside))
				continue;
			x_new = steer(x_nearest, outside);
			if (checkOccupancy(x_new))
				continue;
		}
		if (!isSegmentFree(x_nearest->x, x_new))
		{
			Vector3d blocking_center, path_dir, outside;
			if (!findBlockingObstacle(x_nearest->x, x_new, blocking_center, path_dir) ||
				!projectToObstacleOutside(blocking_center, path_dir, outside))
				continue;
			x_new = steer(x_nearest, outside);
			if (checkOccupancy(x_new) || !isSegmentFree(x_nearest->x, x_new))
				continue;
		}

		double edge_cost = edgeCostWithTurnPenalty(x_nearest, x_new);
		RRTNode *x_new_node = new RRTNode(x_new, x_nearest, x_nearest->cost + edge_cost, node_id_counter_++);
		x_nearest->addChild(x_new_node);
		nodes_.push_back(x_new_node);
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
					RRTNode *virtual_goal = new RRTNode(end_pt_, x_nearest, virtual_cost, node_id_counter_++);
					x_nearest->addChild(virtual_goal);
					nodes_.push_back(virtual_goal);
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
				double last_leg_clear = queryPathClearance(partial->x, end_pt_);
				if (!isSegmentFree(partial->x, end_pt_) || last_leg_clear < min_path_clearance_ * 0.80)
				{
					ROS_ERROR("[InformedRRT*] Partial path rejected: unsafe final segment (clearance=%.3f, need %.3f).",
							  last_leg_clear, min_path_clearance_ * 0.80);
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

		if (checkOccupancy(x_random))
			continue;

		RRTNode *x_nearest = kdTreeNearestNeighbor(x_random);
		if (x_nearest == NULL)
			continue;

		Vector3d x_new = steer(x_nearest, x_random);
		if (checkOccupancy(x_new))
		{
			Vector3d outside;
			if (!projectToObstacleOutside(x_new, x_random - x_nearest->x, outside))
				continue;
			x_new = steer(x_nearest, outside);
			if (checkOccupancy(x_new))
				continue;
		}
		if (!isSegmentFree(x_nearest->x, x_new))
		{
			Vector3d blocking_center, path_dir, outside;
			if (!findBlockingObstacle(x_nearest->x, x_new, blocking_center, path_dir) ||
				!projectToObstacleOutside(blocking_center, path_dir, outside))
				continue;
			x_new = steer(x_nearest, outside);
			if (checkOccupancy(x_new) || !isSegmentFree(x_nearest->x, x_new))
				continue;
		}

		double edge_cost = edgeCostWithTurnPenalty(x_nearest, x_new);
		RRTNode *x_new_node = new RRTNode(x_new, x_nearest, x_nearest->cost + edge_cost, node_id_counter_++);
		x_nearest->addChild(x_new_node);
		nodes_.push_back(x_new_node);
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
				if (isSegmentFree(current_path[i]->x, current_path[j]->x) &&
					queryPathClearance(current_path[i]->x, current_path[j]->x) >= min_path_clearance_ * 0.72)
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

} // namespace ego_planner

#include "path_searching/informed_rrt_star.h"

using namespace std;
using namespace Eigen;

namespace informed_rrt_star_planner
{

InformedRRTstar::InformedRRTstar()
	: step_size_(0.25),
	  node_id_counter_(0),
	  solution_node_(NULL),
	  best_cost_(numeric_limits<double>::infinity()),
	  c_best_(numeric_limits<double>::infinity()),
	  c_min_(0.0),
	  // Runtime tunables (loaded from ~informed_rrt_star/* in initGridMap).
	  rrt_max_time_(0.0),
	  opt_max_time_(0.0),
	  max_iterations_(0),
	  max_nodes_(0),
	  goal_bias_(0.0),
	  enable_parallel_expansion_(false),
	  parallel_batch_size_(0),
	  apf_sampling_ratio_(0.0),
	  apf_rep_gain_(0.0),
	  apf_rep_radius_(0.0),
	  min_path_clearance_(0.0),
	  has_path_prefix_(false),
	  blocking_obstacle_cached_(false),
	  blocking_obstacle_valid_(false),
	  path_cache_valid_(false),
	  apf_cache_frame_(0),
	  clearance_cache_frame_(0),
	  occupancy_cache_frame_(0),
	  node_spatial_cell_size_(0.20),
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

	ros::NodeHandle pnh("~");
	pnh.param("informed_rrt_star/rrt_max_time", rrt_max_time_, 0.5);
	pnh.param("informed_rrt_star/opt_max_time", opt_max_time_, 0.10);
	pnh.param("informed_rrt_star/max_iterations", max_iterations_, 10000);
	int max_nodes_int = 6000;
	pnh.param("informed_rrt_star/max_nodes", max_nodes_int, 6000);
	max_nodes_ = std::max<size_t>(1000, static_cast<size_t>(std::max(1000, max_nodes_int)));
	pnh.param("informed_rrt_star/goal_bias", goal_bias_, 0.15);
	pnh.param("informed_rrt_star/enable_parallel_expansion", enable_parallel_expansion_, true);
	pnh.param("informed_rrt_star/parallel_batch_size", parallel_batch_size_, 2);
	pnh.param("informed_rrt_star/apf_sampling_ratio", apf_sampling_ratio_, 0.35);
	pnh.param("informed_rrt_star/apf_rep_gain", apf_rep_gain_, 2.0);
	pnh.param("informed_rrt_star/apf_rep_radius", apf_rep_radius_, 1.5);
	pnh.param("informed_rrt_star/min_path_clearance", min_path_clearance_, 0.25);

	rrt_max_time_ = std::max(0.05, rrt_max_time_);
	opt_max_time_ = std::max(0.01, opt_max_time_);
	max_iterations_ = std::max(1000, max_iterations_);
	goal_bias_ = std::min(0.9, std::max(0.0, goal_bias_));
	parallel_batch_size_ = std::max(1, parallel_batch_size_);
	apf_sampling_ratio_ = std::min(0.95, std::max(0.0, apf_sampling_ratio_));
	apf_rep_gain_ = std::max(0.0, apf_rep_gain_);
	apf_rep_radius_ = std::max(0.1, apf_rep_radius_);
	min_path_clearance_ = std::max(0.05, min_path_clearance_);

	ROS_INFO("[InformedRRT*] initGridMap: resolution=%.3f, workspace_min=(%.2f,%.2f,%.2f), workspace_max=(%.2f,%.2f,%.2f)",
			 resolution_, workspace_min_(0), workspace_min_(1), workspace_min_(2),
			 workspace_max_(0), workspace_max_(1), workspace_max_(2));
	ROS_INFO("[InformedRRT*] Params: rrt_max_time=%.3f opt_max_time=%.3f max_iterations=%d max_nodes=%zu "
			 "goal_bias=%.2f parallel=%d batch=%d apf_ratio=%.2f rep_gain=%.2f rep_radius=%.2f min_clear=%.2f",
			 rrt_max_time_, opt_max_time_, max_iterations_, max_nodes_, goal_bias_,
			 enable_parallel_expansion_ ? 1 : 0, parallel_batch_size_, apf_sampling_ratio_,
			 apf_rep_gain_, apf_rep_radius_, min_path_clearance_);
}

void InformedRRTstar::clearTree()
{
	node_storage_.clear();
	nodes_.clear();
	node_id_counter_ = 0;
	solution_node_ = NULL;
	best_cost_ = numeric_limits<double>::infinity();
	kdtree_pts_.clear();
	kdtree_idx_.clear();
	id_to_node_.clear();
	node_spatial_index_.clear();
}

} // namespace informed_rrt_star_planner

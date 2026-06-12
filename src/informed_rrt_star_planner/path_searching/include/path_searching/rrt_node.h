#ifndef _RRT_NODE_H_
#define _RRT_NODE_H_

#include <Eigen/Eigen>
#include <vector>

namespace informed_rrt_star_planner
{

// Tree node used by InformedRRTstar. Kept in its own header so the tree data
// structure can be reused/inspected independently of the planner internals.
struct RRTNode
{
	Eigen::Vector3d x;
	RRTNode *parent;
	double cost;
	int id;
	std::vector<RRTNode *> children;

	RRTNode(const Eigen::Vector3d &_x, RRTNode *_parent, double _cost, int _id)
		: x(_x), parent(_parent), cost(_cost), id(_id)
	{
		children.reserve(4);
	}

	inline void addChild(RRTNode *child) { children.push_back(child); }
	inline void removeChild(RRTNode *child)
	{
		for (auto it = children.begin(); it != children.end(); ++it)
			if (*it == child)
			{
				children.erase(it);
				break;
			}
	}
};

} // namespace informed_rrt_star_planner

#endif // _RRT_NODE_H_

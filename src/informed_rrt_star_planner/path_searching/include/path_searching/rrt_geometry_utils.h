#ifndef _RRT_GEOMETRY_UTILS_H_
#define _RRT_GEOMETRY_UTILS_H_

#include <Eigen/Eigen>
#include <array>

namespace informed_rrt_star_planner
{
namespace rrt_detail
{

// 26 normalized ray directions (axis + face-diagonal + corner) shared by the
// clearance queries and the APF potential. Defined inline so every translation
// unit that needs it gets the same singleton without an ODR violation.
inline const std::array<Eigen::Vector3d, 26> &clearanceRayDirs()
{
	static const std::array<Eigen::Vector3d, 26> dirs = [] {
		std::array<Eigen::Vector3d, 26> d = {
			Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(-1, 0, 0),
			Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, -1, 0),
			Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, -1),
			Eigen::Vector3d(1, 1, 0), Eigen::Vector3d(1, -1, 0), Eigen::Vector3d(-1, 1, 0), Eigen::Vector3d(-1, -1, 0),
			Eigen::Vector3d(1, 0, 1), Eigen::Vector3d(1, 0, -1), Eigen::Vector3d(-1, 0, 1), Eigen::Vector3d(-1, 0, -1),
			Eigen::Vector3d(0, 1, 1), Eigen::Vector3d(0, 1, -1), Eigen::Vector3d(0, -1, 1), Eigen::Vector3d(0, -1, -1),
			Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(1, 1, -1), Eigen::Vector3d(1, -1, 1), Eigen::Vector3d(1, -1, -1),
			Eigen::Vector3d(-1, 1, 1), Eigen::Vector3d(-1, 1, -1), Eigen::Vector3d(-1, -1, 1), Eigen::Vector3d(-1, -1, -1)};
		for (auto &v : d)
			v.normalize();
		return d;
	}();
	return dirs;
}

} // namespace rrt_detail
} // namespace informed_rrt_star_planner

#endif // _RRT_GEOMETRY_UTILS_H_

# catkin_ws 工作空间说明

本仓库为**基于 EGO-Planner 改进的局部路径规划**（前端 **Informed RRT\*** + 后端 **均匀 B 样条优化**），并与 **XTDrone + PX4（仿真或实机）**、可选 **VINS-Fusion** 组成完整闭环。

---

## 工作流程

**PX4/MAVROS 状态估计** → **本工作空间局部地图与重规划** → **B 样条轨迹** → **`traj_server` 发布位置指令** → **XTDrone 桥接到控制器**。

>## 持续更新中.....
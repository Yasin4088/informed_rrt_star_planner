# Informed RRT* Planner

本工作空间用于无人机局部路径规划实验，核心为 `informed_rrt_star_planner`：

- 前端搜索：Informed RRT*
- 后端优化：Uniform B-spline + L-BFGS
- 执行链路：XTDrone + PX4 + MAVROS

## 1. 运行逻辑总览

当前默认运行入口是：

- 飞控与通信侧：`uav.sh`
- 规划侧：`run_planner.sh`
- 规划主 launch：`src/informed_rrt_star_planner/plan_manage/launch/single_uav.launch`

数据链路如下：

1. PX4 / Gazebo 产生机体状态，MAVROS输出里程计；
2. `single_uav.launch` 将里程计作为规划输入（当前默认：`/iris_0/mavros/odometry/in`）；
3. `informed_rrt_star_planner_node` 完成局部地图构建、重规划、B样条优化；
4. `traj_server` 将B样条轨迹转换为时序位置指令；
5. XTDrone通信节点将指令桥接到控制侧执行；
6. RViz显示搜索树、轨迹、机体姿态和飞行历史。

简化流程：

`MAVROS odom -> 局部地图 -> Informed RRT* -> B-spline/L-BFGS -> traj_server -> XTDrone桥接 -> PX4执行`

## 2. 当前定位源说明（重点）

`single_uav.launch` 中当前配置为：

- 使用 `odom_topic=/iris_0/mavros/odometry/in`
- VINS里程计入口已保留但默认注释

这意味着当前主流程是 **MAVROS里程计驱动规划**，而不是默认依赖 VIO。

## 3. 快速启动（双终端）

终端A（飞控与通信）：

```bash
cd ~/catkin_ws
bash uav.sh
```

终端B（规划与可视化）：

```bash
cd ~/catkin_ws
bash run_planner.sh
```

也可直接启动规划 launch：

```bash
cd ~/catkin_ws
roslaunch informed_rrt_star_planner single_uav.launch
```

## 4. 算法核心思路

### 4.1 前端：Informed RRT*

前端目标是快速产出一条可行、可优化的粗路径。运行时一般经历：

1. 在局部自由空间采样；
2. 最近邻扩展生成新节点；
3. rewiring更新树结构成本；
4. 找到可行解后切换到informed采样（椭球域）加速收敛；
5. 将粗路径交给后端优化器。

相对经典RRT*，当前实现引入了更强的引导机制（如APF偏置、障碍外侧采样、清障约束等），在复杂障碍中可更快找到“可优化解”。

### 4.2 后端：B样条 + L-BFGS

后端目标是把折线路径变为可执行轨迹：

1. 路径离散点转换为均匀B样条控制点；
2. 以平滑性、碰撞代价、动力学可行性为目标函数；
3. 使用L-BFGS迭代优化控制点；
4. 输出连续轨迹并由 `traj_server` 按时间前瞻发布命令。

该两阶段方案相比“仅搜索不优化”，可显著改善轨迹平滑性与控制可执行性。

## 5. 关键参数入口

关键运行参数集中在：

- `src/informed_rrt_star_planner/plan_manage/launch/single_uav.launch`
- `src/informed_rrt_star_planner/plan_manage/launch/run_in_xtdrone.launch`

常调参数包含：

- 规划前瞻：`manager/planning_horizon`、`fsm/planning_horizen_time`
- RRT时间预算：`informed_rrt_star/rrt_max_time`、`informed_rrt_star/opt_max_time`
- 速度加速度约束：`manager/max_vel`、`manager/max_acc`
- 安全裕量：`informed_rrt_star/min_path_clearance`、`optimization/dist0`

## 6. 目录索引

- `src/informed_rrt_star_planner/path_searching`：Informed RRT*搜索
- `src/informed_rrt_star_planner/plan_env`：局部地图与感知融合
- `src/informed_rrt_star_planner/bspline_opt`：B样条与优化
- `src/informed_rrt_star_planner/plan_manage`：规划FSM与轨迹管理
- `src/informed_rrt_star_planner/traj_utils`：轨迹消息与工具
- `src/informed_rrt_star_planner/Utils`：可视化、工具包与消息定义

## 7. 备注

- 本工作空间默认通过 `catkin_make` 编译。
- 建议将“飞控链路”和“规划链路”分终端独立运行，便于定位问题。

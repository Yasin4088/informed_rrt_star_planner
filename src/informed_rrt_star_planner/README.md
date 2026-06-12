# informed_rrt_star_planner

该目录是项目的规划核心，整体采用“前端搜索 + 后端优化”的局部重规划架构：

- 前端：Informed RRT* 负责快速生成可行路径；
- 后端：Uniform B-spline + L-BFGS 负责轨迹平滑与可执行约束；
- 执行：`traj_server` 将轨迹转换为位置指令并发布。

## 1. 运行入口与主逻辑

重点入口：

- `plan_manage/launch/single_uav.launch`

该 launch 的职责：

1. 设定地图尺寸、目标点、odometry输入；
2. 构建必要TF关系（`map/world/odom/base_link`）；
3. 启动 `informed_rrt_star_planner_node`（规划主节点）；
4. 启动 `traj_server`（轨迹下发）；
5. 启动 `odom_visualization`（机体可视化）。

当前默认里程计源：

- `/iris_0/mavros/odometry/in`

保留了 VINS 入口但默认不启用（注释状态）。

## 2. 模块划分

- `path_searching/`  
  Informed RRT* 搜索器实现，负责采样、扩展、碰撞检测、rewire、路径提取。

- `plan_env/`  
  局部栅格地图构建与环境查询，提供占据判断和安全距离查询。

- `bspline_opt/`  
  B样条轨迹参数化与L-BFGS优化，目标包含平滑、碰撞代价、动力学可行性。

- `plan_manage/`  
  任务状态机（FSM）、重规划触发、前后端衔接、轨迹管理。

- `traj_utils/`  
  轨迹相关数据结构和消息工具。

- `Utils/`  
  通用工具、可视化、消息定义和辅助包。

- `benchmark_all/`  
  独立基准测试（无ROS强依赖），用于比较经典与增强版算法性能。
  - 头文件：`benchmark_all/include/`
  - 源码：`benchmark_all/src/`
  - 脚本：`benchmark_all/build_and_run.sh`

## 3. 算法流程（按一次重规划）

1. **状态输入**：接收 odom 和局部感知数据；
2. **局部地图更新**：维护当前可用栅格空间；
3. **前端搜索**：Informed RRT* 生成可行路径；
4. **后端优化**：B样条 + L-BFGS 提升平滑性与可执行性；
5. **轨迹发布**：`traj_server` 按时间前瞻输出控制命令；
6. **循环重规划**：在时间阈值或安全触发下再次规划。

## 4. 相比经典RRT*的增强点（概要）

典型增强策略包括：

- informed采样与APF偏置；
- 障碍外侧采样与安全裕量约束；
- 更高效的最近邻与重连策略；
- 搜索与优化的预算分配（先可行、再提质）。

### 改造收益（工程视角）

- **收敛速度**：更快拿到首条可行路径；
- **路径质量**：更短、更平滑、避障冗余更合理；
- **可执行性**：后端优化后更符合速度/加速度约束；
- **稳定性**：在密集障碍和狭窄通道下鲁棒性更好。

## 5. 常用参数建议

主要调参文件：

- `plan_manage/launch/run_in_xtdrone.launch`

调参优先级建议：

1. 先调运动学约束：`manager/max_vel`、`manager/max_acc`
2. 再调重规划频率：`fsm/thresh_replan_time`
3. 再调搜索预算：`informed_rrt_star/rrt_max_time`、`opt_max_time`
4. 最后调安全裕量：`informed_rrt_star/min_path_clearance`、`optimization/dist0`

这样做的好处是：先保证“能飞、稳飞”，再提升“飞得好”。

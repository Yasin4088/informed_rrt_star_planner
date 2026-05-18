#!/bin/bash

# --- 用于存储后台进程PID的变量 ---
RVIZ_PID=""
LAUNCH_PID=""

# --- 退出处理函数 ---
cleanup() {
    echo "接收到退出信号，正在终止后台进程..."

    if [ ! -z "$RVIZ_PID" ] && kill -0 "$RVIZ_PID" 2>/dev/null; then
        echo "正在终止 RViz 进程 (PID: $RVIZ_PID)..."
        kill $RVIZ_PID
        wait $RVIZ_PID 2>/dev/null
    fi

    if [ ! -z "$LAUNCH_PID" ] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
        echo "正在终止 ROS Launch 进程 (PID: $LAUNCH_PID)..."
        kill $LAUNCH_PID
        wait $LAUNCH_PID 2>/dev/null
    fi

    echo "所有后台进程已终止。"
    exit 0
}

# --- 捕获退出信号 ---
trap cleanup SIGINT SIGTERM

# --- 启动逻辑 ---
echo "=== 正在并行启动 RViz ==="
# RViz：静默运行，不输出日志
(cd ~/catkin_ws && nohup rviz -d informed_rrt_star_rviz.rviz > /dev/null 2>&1) &
RVIZ_PID=$!
echo "RViz 已在后台启动 (PID: $RVIZ_PID)"

echo "=== 正在启动 ROS Launch (日志输出到终端) ==="
# roslaunch：日志直接输出到当前终端，不加 nohup、不重定向
(cd ~/catkin_ws && roslaunch informed_rrt_star_planner single_uav.launch) &
LAUNCH_PID=$!
echo "ROS Launch 已启动，日志将打印在终端 (PID: $LAUNCH_PID)"

echo "按 Ctrl+C 可以退出并终止所有进程。"

# 等待所有后台进程
wait


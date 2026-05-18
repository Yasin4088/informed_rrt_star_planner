#!/bin/bash

# --- 用于存储后台进程PID的变量 ---
PX4_PID=""
GET_POSE_PID=""
COMM_PID=""

# --- 退出处理函数 ---
cleanup() {
    echo "接收到退出信号，正在终止后台进程..."

    if [ ! -z "$PX4_PID" ] && kill -0 "$PX4_PID" 2>/dev/null; then
        echo "正在终止 PX4 进程 (PID: $PX4_PID)..."
        kill $PX4_PID
        wait $PX4_PID 2>/dev/null
    fi

    if [ ! -z "$GET_POSE_PID" ] && kill -0 "$GET_POSE_PID" 2>/dev/null; then
        echo "正在终止 位置获取 进程 (PID: $GET_POSE_PID)..."
        kill $GET_POSE_PID
        wait $GET_POSE_PID 2>/dev/null
    fi

    if [ ! -z "$COMM_PID" ] && kill -0 "$COMM_PID" 2>/dev/null; then
        echo "正在终止 通信 进程 (PID: $COMM_PID)..."
        kill $COMM_PID
        wait $COMM_PID 2>/dev/null
    fi

    echo "所有后台进程已终止。"
    exit 0
}

# --- 捕获退出信号 ---
trap cleanup SIGINT SIGTERM

# --- 设置一个变量，存储 XTDrone 的根目录 ---
XTDRONE_ROOT="$HOME/XTDrone"

# --- 启动逻辑 ---
echo "=== 启动PX4仿真 ==="
# 不生成日志
nohup roslaunch px4 test.launch > /dev/null 2>&1 &
PX4_PID=$!
echo "PX4仿真已在后台启动 (PID: $PX4_PID)"

echo "=== 等待仿真启动 (sleep 4s)... ==="
sleep 4

echo "=== 启动位置获取节点 ==="
cd "$XTDRONE_ROOT/sensing/pose_ground_truth"
# 不生成日志
nohup python get_local_pose.py iris 1 > /dev/null 2>&1 &
GET_POSE_PID=$!
echo "位置获取节点已在后台启动 (PID: $GET_POSE_PID)"

echo "=== 等待位置节点初始化 (sleep 2s)... ==="
sleep 2

echo "=== 启动通信节点 ==="
cd "$XTDRONE_ROOT/communication"
# 不生成日志
nohup python multirotor_communication.py iris 0 > /dev/null 2>&1 &
COMM_PID=$!
echo "通信节点已在后台启动 (PID: $COMM_PID)"

echo "=== 启动键盘控制节点 ==="
echo "按 Ctrl+C 可以退出并终止所有进程。"
cd "$XTDRONE_ROOT/control/keyboard"
python multirotor_keyboard_control.py iris 1 vel

# 当键盘控制结束后，执行清理函数
cleanup


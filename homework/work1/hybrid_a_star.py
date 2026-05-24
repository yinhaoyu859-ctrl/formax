
import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

# 动态规划启发式：预计算从任意网格点到目标的最短距离（不考虑运动学约束）
from dynamic_programming_heuristic import calc_distance_heuristic

# Reeds-Shepp 曲线库：48种最优曲线，用于解析扩展，连接两姿态点
from ReedsSheppPath import reeds_shepp_path_planning as rs
from car import move, check_car_collision, MAX_STEER, WB, plot_car, BUBBLE_R

# ======================== 离散化分辨率 ========================
XY_GRID_RESOLUTION = 2.0        # 位置网格分辨率 [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # 航向角分辨率 [rad]
MOTION_RESOLUTION = 0.1         # 路径插值分辨率 [m]，用于生成平滑路径点
N_STEER = 20                    # 转向角离散采样数量

# ======================== 代价权重 ========================
SB_COST = 100.0             # 换向惩罚（前进↔后退切换），值大则尽量避免换向
BACK_COST = 5.0             # 倒车惩罚，使算法优先选择前进
STEER_CHANGE_COST = 5.0     # 转向角变化惩罚，鼓励平滑转向
STEER_COST = 1.0            # 转向角幅值惩罚，鼓励小角度转向
H_COST = 5.0                # 启发式代价权重

show_animation = True


class Node:
    """
    Hybrid A* 搜索节点

    存储离散化后的状态索引（用于快速查找）以及连续的路径信息（用于运动学约束）。
    """

    def __init__(self, x_ind, y_ind, yaw_ind, direction,
                 x_list, y_list, yaw_list, directions,
                 steer=0.0, parent_index=None, cost=None):
        # 离散状态索引 
        self.x_index = x_ind          # x 方向网格索引
        self.y_index = y_ind          # y 方向网格索引
        self.yaw_index = yaw_ind      # 航向角网格索引
        self.direction = direction    # 行驶方向（True=前进, False=后退）

        # ---------- 连续路径序列 ----------
        # 从父节点到当前节点之间的所有路径点（含插值）
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.directions = directions  # 每个路径点的方向序列

        # ---------- 控制量 ----------
        self.steer = steer            # 到达该节点所采用的转向角

        # ---------- 重构信息 ----------
        self.parent_index = parent_index  # 父节点在 closedList 中的索引，用于回溯路径
        self.cost = cost                  # 从起点到当前节点的累积代价


class Path:
    """
    最终规划结果的路径对象

    包含完整的连续路径点序列、方向序列及总代价。
    """

    def __init__(self, x_list, y_list, yaw_list, direction_list, cost):
        self.x_list = x_list              # 路径 x 坐标序列
        self.y_list = y_list              # 路径 y 坐标序列
        self.yaw_list = yaw_list          # 路径航向角序列
        self.direction_list = direction_list  # 路径方向序列（True=前进, False=后退）
        self.cost = cost                  # 路径总代价


class Config:
    """
    搜索空间的离散化配置

    将连续的 (x, y, yaw) 空间映射到三维网格索引，
    同时存储障碍物范围用于边界检查。
    """

    def __init__(self, ox, oy, xy_resolution, yaw_resolution):
        # 获取障碍物占据的连续范围
        min_x_m = min(ox)
        min_y_m = min(oy)
        max_x_m = max(ox)
        max_y_m = max(oy)

        # 确保边界点被包含在障碍物集合中
        ox.append(min_x_m)
        oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        # 将连续范围映射到离散网格索引范围
        self.min_x = round(min_x_m / xy_resolution)
        self.min_y = round(min_y_m / xy_resolution)
        self.max_x = round(max_x_m / xy_resolution)
        self.max_y = round(max_y_m / xy_resolution)

        # 各维度的网格宽度（用于将三维索引压缩为一维）
        self.x_w = round(self.max_x - self.min_x)
        self.y_w = round(self.max_y - self.min_y)

        # 航向角范围：[-π, π) 映射到整数索引
        self.min_yaw = round(- math.pi / yaw_resolution) - 1
        self.max_yaw = round(math.pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw - self.min_yaw)


def calc_motion_inputs():
    """
    生成所有可能的控制输入组合
  
    """
    # 均匀采样转向角，并额外加入 0.0（直行）
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER,
                                             N_STEER), [0.0])):
        for d in [1, -1]:           # 1=前进, -1=后退
            yield [steer, d]


def get_neighbors(current, config, ox, oy, kd_tree):
    """
    获取当前节点的所有有效邻居节点

    遍历所有控制输入，模拟运动并做碰撞检测，
    只返回发生碰撞且在地图范围内的邻居。

    """
    for steer, d in calc_motion_inputs():
        node = calc_next_node(current, steer, d, config, ox, oy, kd_tree)
        if node and verify_index(node, config):
            yield node


def calc_next_node(current, steer, direction, config, ox, oy, kd_tree):
    """
    基于车辆运动学模型，从当前节点出发，模拟一个控制输入并生成下一个候选节点

    Returns:
        新 Node 对象，若路径有碰撞则返回 None
    """
    # 从当前节点的路径末端状态开始模拟
    x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]

    # 路径模拟长度 = 1.5 倍网格分辨率，保证相邻网格间可到达
    arc_l = XY_GRID_RESOLUTION * 1.5
    x_list, y_list, yaw_list, direction_list = [], [], [], []

    # 按运动分辨率逐步模拟，生成连续路径点
    for _ in np.arange(0, arc_l, MOTION_RESOLUTION):
        x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)
        x_list.append(x)
        y_list.append(y)
        yaw_list.append(yaw)
        direction_list.append(direction == 1)

    # 碰撞检测：路径中任一点与障碍物重合则丢弃此节点
    if not check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
        return None

    # 将连续状态离散化为网格索引
    d = direction == 1
    x_ind = round(x / XY_GRID_RESOLUTION)
    y_ind = round(y / XY_GRID_RESOLUTION)
    yaw_ind = round(yaw / YAW_GRID_RESOLUTION)

    # ---------- 代价计算 ----------
    added_cost = 0.0

    # 换向惩罚：方向发生变化时施加
    if d != current.direction:
        added_cost += SB_COST

    # 转向角幅值惩罚：鼓励小角度转向
    added_cost += STEER_COST * abs(steer)

    # 转向角变化惩罚：鼓励平滑转向（相邻段间转向角变化小）
    added_cost += STEER_CHANGE_COST * abs(current.steer - steer)

    # 累积代价 = 当前代价 + 新增惩罚 + 路径长度
    cost = current.cost + added_cost + arc_l

    node = Node(x_ind, y_ind, yaw_ind, d, x_list,
                y_list, yaw_list, direction_list,
                parent_index=calc_index(current, config),
                cost=cost, steer=steer)

    return node


def is_same_grid(n1, n2):
    """
    判断两个节点是否在同一离散网格中

    用于重复状态检测，避免在搜索中扩展已访问的状态。
    """
    if n1.x_index == n2.x_index \
            and n1.y_index == n2.y_index \
            and n1.yaw_index == n2.yaw_index:
        return True
    return False


def analytic_expansion(current, goal, ox, oy, kd_tree):
    """
    解析扩展：尝试用 Reeds-Shepp 曲线直接连接当前节点到目标

    遍历 48 种 Reed-Shepp 曲线，选出无碰撞且代价最小的一条。
    这是 Hybrid A* 区别于标准 A* 的关键：一旦解析扩展成功，
    无需继续搜索即可找到到达目标的路径。

    Returns:
        最优的 Reeds-Shepp 路径对象，若无可行路径则返回 None
    """
    start_x = current.x_list[-1]
    start_y = current.y_list[-1]
    start_yaw = current.yaw_list[-1]

    goal_x = goal.x_list[-1]
    goal_y = goal.y_list[-1]
    goal_yaw = goal.yaw_list[-1]

    # 最大曲率 = tan(最大转向角) / 轴距
    max_curvature = math.tan(MAX_STEER) / WB
    paths = rs.calc_paths(start_x, start_y, start_yaw,
                          goal_x, goal_y, goal_yaw,
                          max_curvature, step_size=MOTION_RESOLUTION)

    if not paths:
        return None

    best_path, best = None, None

    # 在所有 RS 曲线中选择无碰撞且代价最小的
    for path in paths:
        if check_car_collision(path.x, path.y, path.yaw, ox, oy, kd_tree):
            cost = calc_rs_path_cost(path)
            if not best or best > cost:
                best = cost
                best_path = path

    return best_path


def update_node_with_analytic_expansion(current, goal,
                                        c, ox, oy, kd_tree):
    """
    尝试从当前节点做解析扩展

    若成功，将 RS 曲线的路径包装为一个 Node，作为 "直达" 目标邻居节
    Returns:
        (is_success, node): 是否成功 及 解析扩展产生的目标节点
    """
    path = analytic_expansion(current, goal, ox, oy, kd_tree)

    if path:
        if show_animation:
            plt.plot(path.x, path.y)

        # 去掉 RS 路径的第一个点（与当前节点末端重合）
        f_x = path.x[1:]
        f_y = path.y[1:]
        f_yaw = path.yaw[1:]

        f_cost = current.cost + calc_rs_path_cost(path)
        f_parent_index = calc_index(current, c)

        # 转换方向标志
        fd = []
        for d in path.directions[1:]:
            fd.append(d >= 0)

        f_steer = 0.0
        f_path = Node(current.x_index, current.y_index, current.yaw_index,
                      current.direction, f_x, f_y, f_yaw, fd,
                      cost=f_cost, parent_index=f_parent_index, steer=f_steer)
        return True, f_path

    return False, None


def calc_rs_path_cost(reed_shepp_path):
    """
    计算 Reed-Shepp 路径的代价值

    Returns:
        float: 路径总代价
    """
    cost = 0.0

    # 路径长度代价（后退有额外惩罚）
    for length in reed_shepp_path.lengths:
        if length >= 0:  # 前进段
            cost += length
        else:            # 后退段
            cost += abs(length) * BACK_COST

    # 换向惩罚：检测相邻段的符号乘积是否为负
    for i in range(len(reed_shepp_path.lengths) - 1):
        if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:
            cost += SB_COST

    # 曲线段转向惩罚
    for course_type in reed_shepp_path.ctypes:
        if course_type != "S":  # S=直线, R=右转, L=左转
            cost += STEER_COST * abs(MAX_STEER)

    # 转向变化惩罚：计算相邻曲线段的转向角差异
    n_ctypes = len(reed_shepp_path.ctypes)
    u_list = [0.0] * n_ctypes
    for i in range(n_ctypes):
        if reed_shepp_path.ctypes[i] == "R":
            u_list[i] = - MAX_STEER
        elif reed_shepp_path.ctypes[i] == "L":
            u_list[i] = MAX_STEER

    for i in range(len(reed_shepp_path.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(u_list[i + 1] - u_list[i])

    return cost


def hybrid_a_star_planning(start, goal, ox, oy, xy_resolution, yaw_resolution):
    """
    Hybrid A* 主搜索算法

    整体流程：
      1. 初始化起止节点、网格配置、KD-Tree、DP 启发式表
      2. 将起点加入 openList（优先队列）
      3. 循环：
         a. 弹出 openList 中代价最小的节点
         b. 尝试解析扩展（Reed-Shepp 曲线）直达目标
         c. 若失败，按运动学模型扩展邻居节点
         d. 新邻居入队，已访问节点加入 closedList
      4. 找到路径后回溯重构完整路径
    Returns:
        Path 对象，包含路径点序列及代价
    """
    # 将角度规范化到 [-π, π)
    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])
    tox, toy = ox[:], oy[:]

    # 构建障碍物 KD-Tree，用于快速最近邻查询（碰撞检测加速）
    obstacle_kd_tree = cKDTree(np.vstack((tox, toy)).T)

    # 网格配置：确定搜索范围的离散化参数
    config = Config(tox, toy, xy_resolution, yaw_resolution)

    # 创建起点节点和目标节点
    start_node = Node(round(start[0] / xy_resolution),
                      round(start[1] / xy_resolution),
                      round(start[2] / yaw_resolution), True,
                      [start[0]], [start[1]], [start[2]], [True], cost=0)
    goal_node = Node(round(goal[0] / xy_resolution),
                     round(goal[1] / xy_resolution),
                     round(goal[2] / yaw_resolution), True,
                     [goal[0]], [goal[1]], [goal[2]], [True])

    # openList: 待扩展节点; closedList: 已扩展节点
    openList, closedList = {}, {}

    # 预计算 DP 启发式表：h_dp[grid_index] = 从该网格到目标的最短距离
    h_dp = calc_distance_heuristic(
        goal_node.x_list[-1], goal_node.y_list[-1],
        ox, oy, xy_resolution, BUBBLE_R)

    # 优先队列 pq: 元素为 (f_cost, node_index)
    # f = g + h，其中 g=累计代价, h=启发式估计
    pq = []
    openList[calc_index(start_node, config)] = start_node
    heapq.heappush(pq, (calc_cost(start_node, h_dp, config),
                        calc_index(start_node, config)))
    final_path = None

    while True:
        if not openList:
            print("Error: Cannot find path, No open set")
            return Path([], [], [], [], 0)

        # 弹出 f 值最小的节点
        cost, c_id = heapq.heappop(pq)
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            continue  # 该节点已被更好的路径处理过，跳过

        if show_animation:  # pragma: no cover
            plt.plot(current.x_list[-1], current.y_list[-1], "xc")
            # 按 Esc 键停止可视化
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if len(closedList.keys()) % 10 == 0:
                plt.pause(0.001)

        # 每次扩展前先尝试 RS 解析扩展：若能无碰撞直达目标，搜索终止
        is_updated, final_path = update_node_with_analytic_expansion(
            current, goal_node, config, ox, oy, obstacle_kd_tree)

        if is_updated:
            print("path found")
            break

        # 按运动学模型扩展邻居节点
        for neighbor in get_neighbors(current, config, ox, oy,
                                      obstacle_kd_tree):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            # 若邻居不在 openList 中，或找到更优路径，则更新
            if neighbor_index not in openList \
                    or openList[neighbor_index].cost > neighbor.cost:
                heapq.heappush(
                    pq, (calc_cost(neighbor, h_dp, config),
                         neighbor_index))
                openList[neighbor_index] = neighbor

    # 从 closedList 中回溯，重构完整路径
    path = get_final_path(closedList, final_path)
    return path


def calc_cost(n, h_dp, c):
    """
    计算节点的总估价函数值：f = g + h

    Returns:
        float: f = g + h 代价
    """
    # 将二维网格索引 (y, x) 压缩为一维索引
    ind = (n.y_index - c.min_y) * c.x_w + (n.x_index - c.min_x)
    if ind not in h_dp:
        # 该网格不可达（在障碍物内），返回极大值
        return n.cost + 999999999
    return n.cost + H_COST * h_dp[ind].cost


def get_final_path(closed, goal_node):
    """
    从 closedList 中回溯重构最终路径

    Returns:
        Path 对象
    """
    # 从目标节点开始，先收集其路径信息
    reversed_x, reversed_y, reversed_yaw = \
        list(reversed(goal_node.x_list)), list(reversed(goal_node.y_list)), \
        list(reversed(goal_node.yaw_list))
    direction = list(reversed(goal_node.directions))
    nid = goal_node.parent_index
    final_cost = goal_node.cost

    # 沿父节点链回溯到起点
    while nid:
        n = closed[nid]
        reversed_x.extend(list(reversed(n.x_list)))
        reversed_y.extend(list(reversed(n.y_list)))
        reversed_yaw.extend(list(reversed(n.yaw_list)))
        direction.extend(list(reversed(n.directions)))
        nid = n.parent_index

    # 反转得到起点→终点的正序路径
    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    direction = list(reversed(direction))

    # 修正第一个方向标记，使其与第二个一致
    direction[0] = direction[1]

    path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

    return path


def verify_index(node, c):
    """
    验证节点的离散索引是否在有效范围内

    Returns:
        bool: 索引是否合法
    """
    x_ind, y_ind = node.x_index, node.y_index
    if c.min_x <= x_ind <= c.max_x and c.min_y <= y_ind <= c.max_y:
        return True
    return False


def calc_index(node, c):
    """
    将三维离散索引 (yaw, y, x) 压缩为一维整数索引

    Returns:
        int: 一维索引
    """
    ind = (node.yaw_index - c.min_yaw) * c.x_w * c.y_w + \
          (node.y_index - c.min_y) * c.x_w + (node.x_index - c.min_x)

    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind


def main():
    """
    主函数：构造测试地图，执行 Hybrid A* 规划并可视化结果

    地图为 60×60 的矩形场地，内部有两个垂直隔墙形成 S 形通道：
      - 左隔墙：x=20, y=0~39
      - 右隔墙：x=40, y=20~60

    起点 (10, 10, 90°)，终点 (50, 50, -90°)
    """
    print("Start Hybrid A* planning")

    ox, oy = [], []

    # 构造地图边界：底边
    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    # 右边
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    # 顶边
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    # 左边
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    # 内部左隔墙 (x=20, y=0~39)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    # 内部右隔墙 (x=40, y=20~60)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    # 起点：靠近左下角，朝右
    start = [10.0, 10.0, np.deg2rad(90.0)]
    # 终点：靠近右上角，朝左
    goal = [50.0, 50.0, np.deg2rad(-90.0)]

    print("start : ", start)
    print("goal : ", goal)

    if show_animation:
        plt.plot(ox, oy, ".k")
        rs.plot_arrow(start[0], start[1], start[2], fc='g')
        rs.plot_arrow(goal[0], goal[1], goal[2])
        plt.grid(True)
        plt.axis("equal")

    # 执行 Hybrid A* 规划
    path = hybrid_a_star_planning(
        start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)

    x = path.x_list
    y = path.y_list
    yaw = path.yaw_list

    # 动画演示车辆沿路径行驶
    if show_animation:
        for i_x, i_y, i_yaw in zip(x, y, yaw):
            plt.cla()
            plt.plot(ox, oy, ".k")
            plt.plot(x, y, "-r", label="Hybrid A* path")
            plt.grid(True)
            plt.axis("equal")
            plot_car(i_x, i_y, i_yaw)
            plt.pause(0.0001)

    print(__file__ + " done!!")


if __name__ == '__main__':
    main()

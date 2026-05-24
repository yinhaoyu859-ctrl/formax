"""

RRT* (Rapidly-exploring Random Tree Star) 路径规划算法

RRT* 是 RRT 的渐进最优改进版本。相比原始 RRT，核心区别在于两点：
1. choose_parent: 新节点在邻域内选择代价最小的父节点（而非直接选最近节点）
2. rewire: 检查邻域内节点是否能通过新节点获得更低代价，即"重布线"
这两点保证随着采样数趋于无穷，路径概率收敛到全局最优解。

author: Atsushi Sakai(@Atsushi_twi)

"""

import math
import sys
import matplotlib.pyplot as plt
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from RRT.rrt import RRT

show_animation = True


class RRTStar(RRT):
    """
    继承自 RRT 基类，增加了 choose_parent 和 rewire 两个核心机制实现渐进最优性
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0  # 从起点到该节点的累积路径代价

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=30.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=300,
                 connect_circle_dist=50.0,
                 search_until_max_iter=False,
                 robot_radius=0.0):
        """
        参数说明：
            start, goal:          起点和终点坐标 [x, y]
            obstacle_list:        障碍物列表 [[x, y, radius], ...]
            rand_area:            随机采样范围 [min, max]
            expand_dis:           单步最大扩展距离
            path_resolution:      路径插值分辨率
            goal_sample_rate:     目标点采样概率 (百分比)
            max_iter:             最大迭代次数
            connect_circle_dist:  rewire/choose_parent 的邻域搜索半径系数
            search_until_max_iter: True 则搜索到最大迭代次数才返回，寻找更优路径
            robot_radius:         半径（碰撞检测用）
        """
        super().__init__(start, goal, obstacle_list, rand_area, expand_dis,
                         path_resolution, goal_sample_rate, max_iter,
                         robot_radius=robot_radius)
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.search_until_max_iter = search_until_max_iter
        self.node_list = []

    def planning(self, animation=True):
        """
        RRT* 主搜索循环

        与 RRT 的关键区别：找到新节点后不是直接加入树中，而是：
        1. 在邻域内找到代价最小的父节点 (choose_parent)
        2. 尝试通过新节点降低邻域内其他节点的代价 (rewire)
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            # 1. 随机采样
            rnd = self.get_random_node()
            # 2. 找到最近节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            # 3. 从最近节点向采样点扩展一步
            new_node = self.steer(self.node_list[nearest_ind], rnd,
                                  self.expand_dis)
            near_node = self.node_list[nearest_ind]
            # 4. 暂设为最近节点的代价 + 欧氏距离（后续 choose_parent 会优化）
            new_node.cost = near_node.cost + \
                math.hypot(new_node.x-near_node.x,
                           new_node.y-near_node.y)

            if self.check_collision(
                    new_node, self.obstacle_list, self.robot_radius):
                # 5. RRT* 核心：在邻域内选最优父节点
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_inds)
                if node_with_updated_parent:
                    # 6. RRT* 核心：重布线，尝试降低邻域节点代价
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            if animation:
                self.draw_graph(rnd)

            # 不等到最大迭代次数，一旦找到可达目标路径就返回
            if ((not self.search_until_max_iter)
                    and new_node):
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        # 达到最大迭代次数后，在候选路径中选最优的
        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds):
        """
        RRT* 核心步骤一：在邻域节点中选择使 new_node 代价最小的父节点

        与原始 RRT 的区别：RRT 直接用最近节点作为父节点，
        而 RRT* 在邻域内遍历所有候选父节点，选择使 new_node 总代价最小的那个。

        参数:
            new_node:  候选新节点
            near_inds: 邻域节点索引列表
        返回:
            重新指定父节点后的 new_node 副本，若邻域内无可达节点则返回 None
        """
        if not near_inds:
            return None

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                # 从 near_node 经由无碰撞路径到 new_node 的总代价
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # 碰撞节点不可达
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        # 用代价最小的邻域节点作为 new_node 的父节点
        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        """
        在所有树节点中，找到能无碰撞连接到终点且总代价最小的节点

        分两步：
        1. 筛选距离终点在 expand_dis 范围内的节点
        2. 在候选节点中进一步筛选可无碰撞连接到终点的
        3. 返回代价最低的那个节点在 self.node_list 中的索引
        """
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.x, n.y) for n in self.node_list
        ]
        # 距离终点在扩展步长内的节点
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        # 进一步过滤：与终点的连接路径无碰撞
        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        # 总代价 = 该节点累积代价 + 到终点的距离
        safe_goal_costs = [self.node_list[i].cost +
                           self.calc_dist_to_goal(self.node_list[i].x, self.node_list[i].y)
                           for i in safe_goal_inds]

        min_cost = min(safe_goal_costs)
        for i, cost in zip(safe_goal_inds, safe_goal_costs):
            if cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        以 new_node 为中心，返回搜索半径 r 内的所有树节点索引

        半径 r 采用 RRT* 论文中的经典公式：
            r = min(γ·√(log n / n), expand_dis)
        其中 γ = connect_circle_dist，n 为当前节点数。
        该公式保证搜索半径随节点数增加而缩小，兼顾收敛性与局部优化效率。

        返回: 邻域内节点在 self.node_list 中的索引列表
        """
        nnode = len(self.node_list) + 1
        # RRT* 的渐进最优半径公式
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        # 半径不能超过最大扩展步长，避免在大范围空旷区域搜索过多节点
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
        RRT* 核心步骤二：遍历邻域节点，若通过 new_node 到达它们的代价更低则重新连接

        原理：新节点加入后，邻域内某些节点通过 new_node 到起点的代价可能比
        它们当前的代价更小。此时将这些节点的父节点改为 new_node。

        参数:
            new_node:  刚加入树的新节点
            near_inds: 邻域节点索引列表
        """
        for i in near_inds:
            near_node = self.node_list[i]
            # 尝试从 new_node 连到 near_node
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(
                edge_node, self.obstacle_list, self.robot_radius)
            improved_cost = near_node.cost > edge_node.cost

            # 若新路径无碰撞且代价更低，执行重布线
            if no_collision and improved_cost:
                # 将原 near_node 的子节点也指向新的父节点
                for node in self.node_list:
                    if node.parent == self.node_list[i]:
                        node.parent = edge_node
                self.node_list[i] = edge_node
                # 重布线后，需要向下传播更新所有后代节点的代价
                self.propagate_cost_to_leaves(self.node_list[i])

    def calc_new_cost(self, from_node, to_node):
        """计算从 from_node 到 to_node 的总代价 = from_node 的累积代价 + 两点距离"""
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):
        """
        rewire 后递归更新所有后代节点的累积代价

        当某个节点的父节点被重新连接后，该节点及其所有子节点的 cost 都需要
        从新的父节点重新计算，保证整棵子树代价的一致性。
        """
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


def main():
    """测试用例：在 8 个圆形障碍物环境中规划从 (0,0) 到 (6,10) 的路径"""
    print("Start " + __file__)

    # 障碍物定义 [x, y, radius]
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1),
        (6, 12, 1),
    ]

    rrt_star = RRTStar(
        start=[0, 0],
        goal=[6, 10],
        rand_area=[-2, 15],
        obstacle_list=obstacle_list,
        expand_dis=1,
        robot_radius=0.8)
    path = rrt_star.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        if show_animation:
            rrt_star.draw_graph()
            # 绘制最终路径，红色虚线
            plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
            plt.grid(True)
            plt.show()


if __name__ == '__main__':
    main()

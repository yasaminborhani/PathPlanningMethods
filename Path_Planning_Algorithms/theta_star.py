import math
import matplotlib.pyplot as plt
import numpy as np

class ThetaStar:
    def __init__(self, start, goal, grid):
        self.start = start
        self.goal = goal
        self.grid = grid

        self.movements = [
            (0, -1),  # up
            (0, 1),   # down
            (-1, 0),  # left
            (1, 0),   # right
            (-1, -1), # diagonal: up-left
            (-1, 1),  # diagonal: up-right
            (1, -1),  # diagonal: down-left
            (1, 1)    # diagonal: down-right
        ]

    def euclidean_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def is_valid_point(self, point):
        x, y = point
        return 0 <= x < len(self.grid) and 0 <= y < len(self.grid[0])

    def is_feasible_point(self, point):
        x, y = point
        return self.grid[x][y] == 255

    def heuristic_cost(self, point):
        return self.euclidean_distance(point, self.goal)

    def get_neighbors(self, point):
        neighbors = []
        for move in self.movements:
            dx, dy = move
            neighbor = (point[0] + dx, point[1] + dy)
            if self.is_valid_point(neighbor) and self.is_feasible_point(neighbor):
                neighbors.append(neighbor)
        return neighbors

    def obstacle_penalty(self, point):
        x, y = point
        distances = []
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] == 0:
                    distances.append(self.euclidean_distance((x, y), (i, j)))
        if distances:
            min_distance = min(distances)
            return 1.0 / (min_distance + 1.0)
        return 0.0

    def run(self):
        open_set = set()
        closed_set = set()
        parent = {}
        g_cost = {}
        f_cost = {}

        open_set.add(self.start)
        g_cost[self.start] = 0
        f_cost[self.start] = self.heuristic_cost(self.start)

        while open_set:
            current = min(open_set, key=lambda point: f_cost[point])

            if current == self.goal:
                path = []
                while current in parent:
                    path.insert(0, current)
                    current = parent[current]
                path.insert(0, self.start)
                return path

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                tentative_g_cost = g_cost[current] + self.euclidean_distance(current, neighbor) + self.obstacle_penalty(neighbor)

                if neighbor not in closed_set or tentative_g_cost < g_cost[neighbor]:
                    parent[neighbor] = current
                    g_cost[neighbor] = tentative_g_cost
                    f_cost[neighbor] = g_cost[neighbor] + self.heuristic_cost(neighbor)

                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return []

    def plot_path(self, path=None):
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(np.array(self.grid), cmap='gray')

        ax.plot(self.start[1], self.start[0], 'ro', markersize=8, label='Start')
        ax.plot(self.goal[1], self.goal[0], 'go', markersize=8, label='Goal')

        if path:
            path_x = [point[1] for point in path]
            path_y = [point[0] for point in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')

        ax.legend()
        plt.show()
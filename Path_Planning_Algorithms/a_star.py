import heapq
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y, g_cost, h_cost):
        self.x = x
        self.y = y
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = None
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

def calculate_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def is_valid_location(x, y, image):
    rows, cols = image.shape
    return 0 <= x < rows and 0 <= y < cols and image[x, y] != 0

def get_neighboring_nodes(node, image, goal_x, goal_y):
    x, y = node.x, node.y
    neighbors = []
    possible_moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Four possible moves (down, up, right, left)

    for dx, dy in possible_moves:
        new_x, new_y = x + dx, y + dy
        if is_valid_location(new_x, new_y, image):
            g_cost = node.g_cost + 1  # Assuming each step has a uniform cost of 1
            h_cost = calculate_distance(new_x, new_y, goal_x, goal_y)
            new_node = Node(new_x, new_y, g_cost, h_cost)
            new_node.parent = node  # Set the parent node
            neighbors.append(new_node)
    
    return neighbors

def reconstruct_path(node):
    path = []
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    return list(reversed(path))

def A_star(image, start_x, start_y, goal_x, goal_y):
    rows, cols = image.shape
    open_list = []
    closed_set = set()

    start_node = Node(start_x, start_y, 0, calculate_distance(start_x, start_y, goal_x, goal_y))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add((current_node.x, current_node.y))

        if current_node.x == goal_x and current_node.y == goal_y:
            return reconstruct_path(current_node)

        neighbors = get_neighboring_nodes(current_node, image, goal_x, goal_y)
        for neighbor in neighbors:
            if (neighbor.x, neighbor.y) in closed_set:
                continue
            
            existing_node = next((n for n in open_list if n.x == neighbor.x and n.y == neighbor.y), None)
            if existing_node is None or neighbor.g_cost < existing_node.g_cost:
                if existing_node is not None:
                    open_list.remove(existing_node)
                heapq.heappush(open_list, neighbor)
    
    return None
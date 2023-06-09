import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf

class Node:
    def __init__(self, position):
        self.position = position
        self.parent = None
        self.cost = float('inf')

def get_neighbors(position, image):
    height, width = image.shape[:2]
    row, col = position
    neighbors = []
    
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < height and 0 <= new_col < width:
                pixel_value = image[new_row, new_col]
                if pixel_value == 255:
                    neighbors.append((new_row, new_col))
    
    return neighbors

def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def rrt_smart(image, start_pixel, goal_pixel, max_iterations=10000, step_size=10, sample_density=0.3):
    height, width = image.shape[:2]
    start_node = Node(start_pixel)
    start_node.cost = 0
    
    nodes = [start_node]
    goal_node = None
    
    for _ in range(max_iterations):
        if np.random.uniform() < sample_density:
            random_point = goal_pixel
        else:
            random_point = (np.random.randint(0, height), np.random.randint(0, width))
            
        nearest_node = min(nodes, key=lambda node: euclidean_distance(node.position, random_point))
        
        # Skip if the random point is the same as the nearest node's position
        if np.array_equal(nearest_node.position, random_point):
            continue
        
        new_point = tuple(int(nearest_node.position[i] + step_size * (random_point[i] - nearest_node.position[i]) / euclidean_distance(nearest_node.position, random_point)) for i in range(2))
        
        if image[new_point] == 0:
            continue
        
        new_node = Node(new_point)
        new_node.parent = nearest_node
        new_node.cost = nearest_node.cost + euclidean_distance(nearest_node.position, new_node.position)
        
        near_nodes = [node for node in nodes if euclidean_distance(node.position, new_node.position) <= step_size]
        min_cost = nearest_node.cost
        min_node = nearest_node
        for near_node in near_nodes:
            cost = near_node.cost + euclidean_distance(near_node.position, new_node.position)
            if cost < min_cost and image[line_of_sight(near_node.position, new_node.position)]:
                min_cost = cost
                min_node = near_node
        
        new_node.parent = min_node
        new_node.cost = min_cost
        nodes.append(new_node)
        
        if euclidean_distance(new_node.position, goal_pixel) <= step_size:
            goal_node = Node(goal_pixel)
            goal_node.parent = new_node
            goal_node.cost = new_node.cost + euclidean_distance(new_node.position, goal_pixel)
            nodes.append(goal_node)
            break
    
    if goal_node is not None:
        path = []
        current_node = goal_node
        while current_node.parent:
            path.append(tf.constant(current_node.position, dtype=tf.int32))
            current_node = current_node.parent
        
        path.reverse()
        return path
    
    return None
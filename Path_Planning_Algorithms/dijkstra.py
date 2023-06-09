import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf

def dijkstra(map, start, target, flag=True):
    rows, cols = map.shape
    distances = np.full((rows, cols), np.inf)
    distances[start[0], start[1]] = 0
    visited = np.zeros((rows, cols), dtype=bool)
    prev = np.full((rows, cols), None)
    pq = []

    pq.append((0, start))

    while pq:
        pq.sort()  # Sort the priority queue based on distance
        curr_dist, curr_pos = pq.pop(0)
        curr_row, curr_col = curr_pos

        if curr_pos == target:
            break

        if visited[curr_row, curr_col]:
            continue

        visited[curr_row, curr_col] = True

        neighbors = get_neighbors(curr_pos, rows, cols, flag)

        for neighbor in neighbors:
            neighbor_row, neighbor_col = neighbor

            if visited[neighbor_row, neighbor_col]:
                continue

            if map[neighbor_row, neighbor_col] == 0:  # Check for obstacles
                continue

            new_dist = distances[curr_row, curr_col] + distance(curr_pos, neighbor)

            if new_dist < distances[neighbor_row, neighbor_col]:
                distances[neighbor_row, neighbor_col] = new_dist
                prev[neighbor_row, neighbor_col] = curr_pos
                pq.append((new_dist, neighbor))

    path = []
    curr = target

    while curr is not None:
        path.append(curr)
        curr = prev[curr[0], curr[1]]

    path.reverse()
    return path

def get_neighbors(pos, rows, cols, flag=False):
    row, col = pos
    neighbors = []
    if flag:
        if row > 0:
            neighbors.append((row - 1, col))
        if row < rows - 1:
            neighbors.append((row + 1, col))
        if col > 0:
            neighbors.append((row, col - 1))
        if col < cols - 1:
            neighbors.append((row, col + 1))


    else:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                new_row = row + dr
                new_col = col + dc

                if 0 <= new_row < rows and 0 <= new_col < cols:
                    neighbors.append((new_row, new_col))

    return neighbors

def distance(pos1, pos2):
    return 1  # Assuming all adjacent cells have the same cost

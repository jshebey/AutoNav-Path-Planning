import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import heapq


def createOccupancyGrid(image_name):
  """ Builds an occupancy grid based on an image

  Args:
      image_name: Image location/name

  Returns:
      A matrix filled with 0s and 1s (occupancy grid)
  """
  #Reads in the input RGB image and converts it to grayscale
  image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
  #Converts all pixels less with a value < 127 to 0 and > 127 to 1
  threshold, occupancy_grid = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY_INV)

  return occupancy_grid

def calculateCostMap(block_size, occupancy_grid):

  """ Calculates an initial cost map

  Args:
      block_size: Grid cell pixel size for taking the average
      occupancy_grid: Occupancy grid full of 0s and 1s

  Returns:
      A cost map with values ranging from 1 (low cost) to 5 (high cost)
  """

  #Creates a zero filled map based on the specified block size
  costMap = np.zeros((occupancy_grid.shape[0] // block_size, occupancy_grid.shape[1] // block_size))

  #Loops through the occupancy grid and breaks it up into blocks
  for i in range(0, occupancy_grid.shape[0], block_size):
    for j in range(0, occupancy_grid.shape[1], block_size):
      #Doesnt calculate if the current block isn't divisible by the height/width of the grid
      if occupancy_grid.shape[0] >= i + block_size and occupancy_grid.shape[1] >= j + block_size:
        #Grabs a block from the occupancy grid
        block = occupancy_grid[i:(i + block_size), j:(j + block_size)]
        #Finds the average value of the current block (0 = low average, 1 = high average)
        average = np.mean(block)
        #Scales the cost from 1 to 5 using the equation: cost = -4x + 5 (Used the points (0, 5) and (1, 1))
        cost = round(-4 * average + 5)
        costMap[i // block_size, j // block_size] = cost

  return costMap


def adjustedCostMap(cost_map, left_line, right_line):

  """ Calculates an adjusted cost map

    Args:
        cost_map: Initial cost map
        left_line: Flag saying if the left line is in the image
        right_line: Flag saying if the right line is in the image

    Returns:
        An adjusted cost map with close objects morphed together
        and the area not within the lines as a high cost
  """

  #Converts values to the left of the left line to 5
  if left_line:
    #Loops through each grid cell
    for i in range(0, cost_map.shape[0]):
      for j in range(0, cost_map.shape[1]):
        #Keeps converting until a value other than 1 is hit
        if(cost_map[i, j] == 1):
          cost_map[i, j] = 5
        else:
          #Ends loop and goes to the next row
          break
  
  #Converts values to the right of the right line to 5
  if right_line:
    #Loops through each grid cell
    for i in range(0, cost_map.shape[0]):
      for j in range(cost_map.shape[1] - 1, 0, -1):
        #Keeps converting until a value other than 1 is hit
        if(cost_map[i, j] == 1):
          cost_map[i, j] = 5
        else:
          #Ends loop and goes to the next row
          break
  
  #Converts gaps in between objects (within 3 blocks away) to 5
  #Really sloppy, need to update
  for i in range(0, cost_map.shape[0]):
      for j in range(0, cost_map.shape[1]):
        #Checks if the current cell has a weight of 4 or 5
        if cost_map[i, j] == 4 or cost_map[i, j] == 5:
          
          #Checks if there is another object in the y direction
          if i < cost_map.shape[0] - 4:
            if cost_map[i + 2, j] == 4 or cost_map[i + 2, j] == 5:
              cost_map[i + 1, j] = 5
            if cost_map[i + 3, j] == 4 or cost_map[i + 3, j] == 5:
              cost_map[i + 2, j] = 5
            if cost_map[i + 4, j] == 4 or cost_map[i + 4, j] == 5:
              cost_map[i + 3, j] = 5

          #Checks if there is another object in the -y direction
          if i > 3:
            if cost_map[i - 2, j] == 4 or cost_map[i - 2, j] == 5:
              cost_map[i - 1, j] = 5
            if cost_map[i - 3, j] == 4 or cost_map[i - 3, j] == 5:
              cost_map[i - 2, j] = 5
            if cost_map[i - 4, j] == 4 or cost_map[i - 4, j] == 5:
              cost_map[i - 3, j] = 5

          #Checks if there is another object in the y direction
          if j < cost_map.shape[1] - 4:
              if cost_map[i, j + 2] == 4 or cost_map[i, j + 2] == 5:
                cost_map[i, j + 1] = 5
              if cost_map[i, j + 3] == 4 or cost_map[i, j + 3] == 5:
                cost_map[i, j + 2] = 5
              if cost_map[i, j + 4] == 4 or cost_map[i, j + 4] == 5:
                cost_map[i, j + 3] = 5

          #Checks if there is another object in the -y direction
          if j > 3:
              if cost_map[i, j - 2] == 4 or cost_map[i, j - 2] == 5:
                cost_map[i - 1, j] = 5
              if cost_map[i, j - 3] == 4 or cost_map[i, j - 3] == 5:
                cost_map[i, j - 2] = 5
              if cost_map[i, j - 4] == 4 or cost_map[i, j - 4] == 5:
                cost_map[i, j - 3] = 5
  
  return cost_map

def heuristic(curr, end):
  """ Calculates the heuristic of the current location using Manhattan distance

    Args:
        curr: Current grid cell
        end: Goal grid cell

    Returns:
        A heuristic based on the row and column differences
  """
  #Takes the absolute value of the difference of the current and goal row location
  row_diff = abs(curr[0] - end[0])
  #Takes the absolute value of the difference of the current and goal column location
  col_diff = abs(curr[1] - end[1])

  return row_diff + col_diff

def findNeighbors(curr_loc, grid_dimensions):
  """ Finds the valid neighbor grid cells to the current node

    Args:
        curr_loc: Current grid cell location
        grid_dimensions: Grid row and column count

    Returns:
        A list of neighbors
  """
  neighbors = []

  #Finds neighbor in the row above
  if curr_loc[0] != 0:
    neighbors.append((curr_loc[0] - 1, curr_loc[1]))

  #Finds neighbor in the row below
  if curr_loc[0] != grid_dimensions[0] - 1:
    neighbors.append((curr_loc[0] + 1, curr_loc[1]))

  #Finds neighbor in the column to the left
  if curr_loc[1] != 0:
    neighbors.append((curr_loc[0], curr_loc[1] - 1))

  #Finds neighbor in the column to the right
  if curr_loc[1] != grid_dimensions[1] - 1:
    neighbors.append((curr_loc[0], curr_loc[1] + 1))

  return neighbors

def a_star(adjusted_cost_map, start, end):
  """ Finds the best path using an A* search algorithm

    Args:
        adjusted_cost_map: cost map with close objects morphed together
        and the area not within the lines as a high cost

        start: Starting location

        end: Goal location

    Returns:
        The cheapest and most optimal path
  """

  #Creates a priority queue
  p_queue = []
  #Adds the starting node to the queue
  heapq.heappush(p_queue, (0, start))

  #Holds each node's parent
  came_from = {}

  #Dictionaries to map points to heuristic values

  #g is the cost from the start to current node (current sum)
  g = {start: 0}
  #f is the cost from the current node's path to the goal
  f = {start: heuristic(start, end)}

  #Main loop runs while the queue isn't empty
  while p_queue:
    #Current node is the node with the lowest f value in the queue
    curr_f, curr = heapq.heappop(p_queue)
    #Checks if the goal has been reached
    if curr == end:
      #List to keep track of path from start to end
      path = []

      #Loops through each node and its parent until the end is reached
      while curr in came_from:
        path.append(curr)
        #Grabs parent
        curr = came_from[curr]
      #Adds the starting node and reverse the list
      path.append(start)
      return path[::-1]

    #Finds the neighbors of the current node
    neighbors = findNeighbors(curr, adjusted_cost_map.shape) 

    #Loops through each neighbor 
    for neighbor in neighbors:
      #Gets the cost of the neighboring node
      neighbor_cost = adjusted_cost_map[neighbor[0], neighbor[1]]
      #Calculates the new g cost from the start to the neighbor
      new_g = g[curr] + neighbor_cost

      #Checks if the neighbor hasn't been visited or if the new path is cheaper
      if neighbor not in g or new_g < g[neighbor]:
        #Adds the neighbor and current node value to the dictionary
        came_from[neighbor] = curr
        #The neighbor and updated g value are added to the g list
        g[neighbor] = new_g
        #The neighbor and updated f value are added to the f list
        f[neighbor] = new_g + heuristic(neighbor, end)
        #Adds neighbor node to the explore node list
        heapq.heappush(p_queue, (f[neighbor], neighbor))
  
  #Returns no path if none is found
  return None

def drawPath(adjusted_cost_map, path):
  """ Draws the A* found path on the cost map
s
    Args:
        adjusted_cost_map: cost map with close objects morphed together
        and the area not within the lines as a high cost

        path: List of locations to go from the start to end node

    Returns:
        A 
  """
  for loc in path:
    adjusted_cost_map[loc[0], loc[1]] = 0
  
  return adjusted_cost_map


occupancy_grid = createOccupancyGrid("maps/map2.JPG")
cost_map = calculateCostMap(40, occupancy_grid)
adjusted_cost_map = adjustedCostMap(cost_map, True, True)


start = (adjusted_cost_map.shape[0] - 1, adjusted_cost_map.shape[1] // 2)
end = (0, adjusted_cost_map.shape[1] // 2)

path = a_star(adjusted_cost_map, start, end)
print(adjusted_cost_map)


drawPath(adjusted_cost_map, path)

cost_colors = ['black', 'green', 'lightgreen', 'yellow', 'orange', 'red']
color_map = ListedColormap(cost_colors)

plt.imshow(adjusted_cost_map, cmap = color_map, interpolation='nearest')  # 'nearest' keeps each value as a block of pixels

# Add a color bar to show the mapping of values to colors
plt.colorbar(label="Cost Value")

rows = cost_map.shape[0]
columns = cost_map.shape[1]

plt.xticks(np.arange(-0.5, columns, 1), [])  # Create ticks at cell boundaries on x-axis
plt.yticks(np.arange(-0.5, rows, 1), [])  # Create ticks at cell boundaries on y-axis


# Set the aspect ratio so that grid cells are square
plt.gca().set_aspect('equal', adjustable='box')

# Optional: Add gridlines to show the grid
# These gridlines will match the exact size of the matrix
plt.grid(which='both', color='black', linestyle='-', linewidth=1)

# Show the plot
plt.show()
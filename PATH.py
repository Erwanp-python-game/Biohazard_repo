class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    global authorized_map
    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)
    count0=0
    # Loop until you find the end
    while len(open_list) > 0 and count0<100:
        count0=count0+1
        #print(count0)
        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        #print(current_node.f,current_node.position,len(closed_list))

        # Pop current off open list, add to closed list
        #print([i.position for i in open_list])
        #print([i.f for i in open_list])
        open_list.pop(current_index)
        #print(current_node.position in [i.position for i in open_list])
        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])


            # Make sure walkable terrain
            if maze[node_position[0],node_position[1]] != 1:
               new_node = Node(current_node, node_position)
               children.append(new_node)

        # Loop through children
        for child in children:
            A1=False
            A2=False
            # Child is on the closed list
            for closed_child in closed_list:
                
                if child.position == closed_child.position :
                    A1=True
            for open_node in open_list:
                if child.position == open_node.position and child.g < open_node.g:
                    A2=True
            # Create the f, g, and h values
            if not(A1 or A2):
                #print(child.position)
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + 1*child.h
                open_list.append(child)


        closed_list.append(current_node)
    return []


import pickle
#import matplotlib.pyplot as plt
def main():
    global authorized_map
    f = open("level/"+str(2), "rb")
    level_w=pickle.load(f)
    level_h=pickle.load(f)
    level_map=pickle.load(f)
    level_map.T
    pickle.load(f)
    pickle.load(f)
    zmap=pickle.load(f)
    light_wall=pickle.load(f)
    pickle.load(f)
    hmap=pickle.load(f)
    pickle.load(f)
    authorized_map=pickle.load(f)
    authorized_map.T
    f.close()

    start = (0, 0)
    end = (7, 6)


    path = astar(authorized_map, (22,65), (30,73))
    for i in path:
         #print(authorized_map[i[0]][i[1]])
         authorized_map[i[0]][i[1]]=3
    # plt.imshow(authorized_map)
    # plt.show()

    print(path)


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

class Node:
    def __init__(self, value, number, connections=None):
        self.index = number
        self.connections = connections
        self.value = value

class Network:
    def __init__(self, nodes=None):
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes 

    def get_mean_degree(self):
            print("")
        #Your code  for task 3 goes here

    def get_mean_clustering(self):
            print("")
        #Your code for task 3 goes here

    def get_mean_path_length(self):
            print("")
        #Your code for task 3 goes here

    def make_random_network(self, N, connection_probability=0.5):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1
            print(node.connections)

    def make_ring_network(self, N, neighbour_range=1):
        self.nodes = []
        for node_number in range(N):
            neighbours = np.zeros(N)
            for n in range(node_number-neighbour_range, (node_number+neighbour_range+1)):
                neighbour = (n+N)%N
                if neighbour != node_number:
                    neighbours[neighbour] = 1
            self.nodes.append(Node(0, node_number, neighbours))

    def make_small_world_network(self, N, re_wire_prob=0.2):
        self.make_ring_network(N, 2)
        for node in self.nodes:
            node_number = node.index
            connections = node.connections
            edges = []
            for i in range(node_number, N):
                 if connections[i] == 1:
                      edges.append(i)
            randomIndexes = np.arange(node_number, N).tolist()
            for edge in edges:
                 randomIndexes.remove(edge)
            randomIndexes.remove(node_number)
            for edge in edges:
                randomNum = random.random()
                if randomNum <= re_wire_prob:
                    if randomIndexes != []:
                        randomIndex = randomIndexes[random.randint(0, len(randomIndexes)-1)]
                        randomIndexes.remove(randomIndex)
                        connections[randomIndex] = 1
                        connections[edge] = 0

            node.connections = connections


    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])
        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)
            for neighbour_index in range(i+1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)
                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

example = Network()
example.make_small_world_network(10, 0.2)
example.plot()
plt.show()
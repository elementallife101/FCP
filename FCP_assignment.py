import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

"""
Modules
-------
numpy - Useful for data structures
        Documentation: https://numpy.org/doc/stable/

matplotlib.pyplot - Useful for plotting
                    Documentation: https://matplotlib.org/stable/api/pyplot_summary.html

matplotlib.cm - Useful for plotting coloured maps
                Documentation: https://matplotlib.org/stable/api/cm_api.html

argparse - Useful for handling arguments
           Documentation: https://docs.python.org/3/library/argparse.html
"""

## Created by Tycho for Task 3.
def mean(number_list):
    total = 0
    for number in number_list:
        total += number
    result = total / len(number_list)
    return result

## Preexisting Class - Used in Block 4.
class Queue:
    def __init__(self):
        self.queue = []
    def push(self, item):
        self.queue.insert(0, item)
        return self.queue
    def pop(self):
        return self.queue.pop()
    def is_empty(self):
        return len(self.queue)==0
	
class Node:
	def __init__(self, value, number, connections=None):
		self.index = number
		self.connections = connections
		self.value = value

        
		## Created by Tycho for Task 3.
		self.new_connections = 0

	def get_neighbours(self):
		return np.where(np.array(self.connections)==1)[0]
           
class Network:
	def __init__(self, nodes=None):
		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes 

	def new_connections_marker(self):
			"""
			Ensures that the new_connections variable contains the actual nodes, rather than 0 or 1 to ensure iteration within new_connections.
			Must be run after nodes are added to network.
			Called via network_name.new_connections_marker().

			Arguments:
				self.nodes (Type: List) - Represents a list of the nodes and connections that exist within a network.
			"""
			for node in self.nodes: 
				if node.new_connections == 0:
					new_connections_list = []
					for index in range(len(node.connections)):
						if node.connections[index] == 1:
							new_connections_list.append(self.nodes[index])
							
					node.new_connections = new_connections_list

	def get_mean_degree(self):
		"""
		Returns the mean degree of the all the nodes in a network. (The mean of all nodes mean degrees).
		Called via network_name.get_mean_degree().

		Arguments:
			self.nodes (Type: List) - Represents a list of the nodes and connections that exist within a network.

		Returns:
			mean_degree (Type: Float) - Represents the mean degree of all the nodes in the network.
		"""
		self.new_connections_marker()
		list_of_lengths = []
		for examplenode in self.nodes:
				list_of_lengths.append(len(examplenode.new_connections))

		mean_degree = float(mean(list_of_lengths))
		print("Mean Degree:", mean_degree)
		return mean_degree

	def get_clustering(self):
		"""
		Returns the mean clustering coefficient for a given network.
		Called via network_name.get_clustering().

		Arguments:
			self.nodes (Type: list) - Represents a list of the nodes and connections that exist within a network.
		
		Returns:
			mean_coefficient (Type: float) - Represents the mean clustering coefficient for the given number.
		"""
		self.new_connections_marker()
		clustering_coefficient_list = []

		for node in self.nodes:
			connections_list = node.new_connections
			possible_triangles = (len(connections_list)) * (len(connections_list) - 1) / 2
			connection_number = 0

			## Cycles through the neighbours of every neighbour to test for connections.
			for neighbour in connections_list:
				for neighbour_of_neighbour in neighbour.new_connections:
					if neighbour_of_neighbour in connections_list:
						connection_number += 1
			## We divide by 2 as the algorithm goes over every path twice (start -> end, end -> start)
			try:
				clustering_coefficient = (connection_number / possible_triangles) / 2
			## This is due to the fact that if possible_triangles = 0,
			## The "fraction of a node's neighbours that connect to each other forming a triangle that includes the original node" must be equal to 0 as none do.
			except ZeroDivisionError:
				clustering_coefficient = 0
			clustering_coefficient_list.append(clustering_coefficient)

		# Calculates and returns the answer
		mean_coefficient = float(mean(clustering_coefficient_list))
		print("Clustering Coefficient:", mean_coefficient)
		return mean_coefficient

	def get_path_length(self):
			"""
			Returns the mean path length for a given network.
			Called via network_name.get_path_length().

			Arguments:
				self.nodes (Type: List) - Represents a list of the nodes and connections that exist within a network.
			
			Returns:
				mean(average_path_length) (Type: Float) - Represents the mean path length for the given network.
			"""
			self.new_connections_marker()

			# Method below is imported from Block 4.
			def breadth_first_search(network,start_node,goal):
				"""
				Returns the shortest route between 2 given nodes in a network.

				Arguments:
					network (Type: list) - Represents a list of nodes and connections which are present in the network.
					start_node (Type: node) - Represents the starting node in the network.
					goal (Type: node) - Represents the goal node in the network.

				Returns:
					len(route) (Type: int) - Represents the length of the shortest route between the starting node and the goal node.
				"""
				
				search_queue = Queue()
				search_queue.push(start_node)
				visited = []


				#We search until the queue is empty. If it's empty, that means there's no path from the start to the goal
				while not search_queue.is_empty():
					#Pop the next node from the Queue
					node_to_check = search_queue.pop()
			
					#If we are at the goal, then we are finished.
					if node_to_check == goal:
						break
			
					#If not, we need to add all the neighbours of the current node to the search queue. 
					#Start by looping through all the neighbours
					for neighbour_index in node_to_check.get_neighbours():
						#Get a node based on the index
						neighbour = network[neighbour_index]
						#Check we haven't already visited the neighbour.
						if neighbour_index not in visited:
							#if not, add it to the search queue and store in visited.
							search_queue.push(neighbour)
							visited.append(neighbour_index)
							#Set the parent property to allow for backtracking.
							neighbour.parent = node_to_check
			
				#Now we've found the goal node, we need to backtrace to get the path. 
				#We start at the goal.
				node_to_check = goal
				#We make sure the start node has no parent.
				start_node.parent = None
				route = []
			
				#Loop over node parents until we reach the start.
				while node_to_check.parent:
					#Add node to our route
					route.append(node_to_check)
					#Update node to be the parent of our current node
					node_to_check = node_to_check.parent

				route = [node.value for node in route[::-1]]
				return len(route)
			
			average_path_length = []

			## Cycles through every ending node, and adds the path lengths to path_length.
			## It then takes the mean of the path_length, and then adds the mean to average_path_length.
			## It then cycles to the next starting node, and repeats. It then returns the mean of average_path_length.
			for node in self.nodes:
				path_length = []
				for neighbour in self.nodes:
					if neighbour != node:
						path_length.append(breadth_first_search(self.nodes,node, neighbour))

				average_path_length.append(mean(path_length))

			print("Average Path Length:",mean(average_path_length),"\n")
			return mean(average_path_length)    

	def analysis(self):
		"""
		Useful for visualisation of the model. Returns each node and the connections it has for a random network.
		Arguments:
			self.nodes (Type: List) - Represents a list of nodes and connections which are present in the network.
		"""
		for nodal in self.nodes:
				print("Node",nodal)
				print("Node.value",str(nodal.value))
				print("Node.number",str(nodal.index))
				print("Node.connections",str(nodal.connections)+"\n") 

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
		

	def make_ring_network(self, N, neighbour_range=1):
			print("")
		#Your code  for task 4 goes here

	def make_small_world_network(self, N, re_wire_prob=0.2):
			print("")
		#Your code for task 4 goes here

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

def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0):
	'''
	This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
	Inputs: population (numpy array)
			row (int)
			col (int)
			external (float)
	Returns:
			change_in_agreement (float)
	'''

	#Your code for task 1 goes here

	return np.random * population

def ising_step(population, external=0.0):
	'''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''
	
	n_rows, n_cols = population.shape
	row = np.random.randint(0, n_rows)
	col  = np.random.randint(0, n_cols)

	agreement = calculate_agreement(population, row, col, external=0.0)

	if agreement < 0:
		population[row, col] *= -1

	#Your code for task 1 goes here

def plot_ising(im, population):
	'''
	This function will display a plot of the Ising model
	'''
	new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
	im.set_data(new_im)
	plt.pause(0.1)

def test_ising():
	'''
	This function will test the calculate_agreement function in the Ising model
	'''
	print("Testing ising model calculations")
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1)==4), "Test 1"
	population[1, 1] = 1.
	assert(calculate_agreement(population,1,1)==-4), "Test 2"
	population[0, 1] = 1.
	assert(calculate_agreement(population,1,1)==-2), "Test 3"
	population[1, 0] = 1.
	assert(calculate_agreement(population,1,1)==0), "Test 4"
	population[2, 1] = 1.
	assert(calculate_agreement(population,1,1)==2), "Test 5"
	population[1, 2] = 1.
	assert(calculate_agreement(population,1,1)==4), "Test 6"
	"Testing external pull"
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1,1)==3), "Test 7"
	assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
	assert(calculate_agreement(population,1,1,10)==14), "Test 9"
	assert(calculate_agreement(population,1,1,-10)==-6), "Test 10"
	print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

def defuant_main():
	print("")
	#Your code for task 2 goes here

def test_defuant():
	print("")
	#Your code for task 2 goes here


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
	# Handles flags using the module "Argparse".
    ## Optional arguments - Used for a random network and tests correspondingly.
    parser = argparse.ArgumentParser()
    parser.add_argument("--network")
    parser.add_argument("--test_network",action="store_true")

    ## Optional argument - Used to help the user with understanding the structure of the network
    parser.add_argument("--analysis",action="store_true")
    args = parser.parse_args()

    if args.network:
        testing_network = Network()
        testing_network.make_random_network(int(args.network),0.50)

        testing_network.get_mean_degree()
        testing_network.get_clustering()
        testing_network.get_path_length()

    if args.test_network:
        test_networks()
    
    if args.analysis:
        testing_network = Network()
        testing_network.make_random_network(10,0.50)

        testing_network.analysis()

if __name__=="__main__":
	main()
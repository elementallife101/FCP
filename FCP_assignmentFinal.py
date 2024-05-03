import numpy as np
import random
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
					#print("Index",node_to_check.index)
					#print("get_neighbours()",node_to_check.get_neighbours(),"\n")
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
				#for nodal in network:
					#print("Node",nodal)
					#print("Node.value",str(nodal.value))
					#print("Node.number",str(nodal.index))
					#print("Node.connections",str(nodal.connections)+"\n") 			
				#Now we've found the goal node, we need to backtrace to get the path. 
				#We start at the goal.
				node_to_check = goal
				#print("Starting Index", start_node.index)
				#print("Ending Index",node_to_check.index)
				#We make sure the start node has no parent.
				start_node.parent = None
				route = []
				#print("node_to_check.parent",node_to_check.parent)
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
					## If there is no connections, continue.
					try:
						if neighbour != node:
							path_length.append(breadth_first_search(self.nodes,node, neighbour))

					except AttributeError:
						continue

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
		'''
		Creates a ring network of size N where neighbours are connected to each other with range neighbour_range. 
		'''
		self.nodes = []
		#cycles through each node to be created, creates an empty list of neighbours
		for node_number in range(N):
			neighbours = np.zeros(N)
			#adds the each neighbour rto the list using a forloop
			for n in range(node_number-neighbour_range, (node_number+neighbour_range+1)):
				neighbour = (n+N)%N
				#excludes self-connections
				if neighbour != node_number:
					neighbours[neighbour] = 1
			self.nodes.append(Node(0, node_number, neighbours))

	def make_small_world_network(self, N, re_wire_prob=0.2):
		'''
		Creates a small world network of size N where each connection is re-wired with a probability re_wire_prob
		'''
		self.make_ring_network(N, 2)
		#creates a ring network with a range of 2 as a base

		for node in self.nodes:
			node_number = node.index
			connections = node.connections
			edges = []
			#iterates through each node, initialises a list of the edges it connects to
			for i in range(node_number, N):
					if connections[i] == 1:
						edges.append(i)
					#the edges list contains all nodes which connect to the target node, excluding those with lower indexes as these would have already been checked on previous iterations
			randomIndexes = np.arange(node_number, N).tolist()
			for edge in edges:
					randomIndexes.remove(edge)
			randomIndexes.remove(node_number)
			#randomIndexes is the list of edges from the target node which an edge can be moved to when it is re-wired. It contains all of the nodes in the network excluding those which the target node already connects to and the target node itself.
			for edge in edges:
				randomNum = random.random()
				if randomNum <= re_wire_prob:
					#iterates through each of the edges, generates a random number which is checked against the re-wire probability to see if it should be re-wired.
					if randomIndexes != []:
						randomIndex = randomIndexes[random.randint(0, len(randomIndexes)-1)]
						randomIndexes.remove(randomIndex)
						randomIndexes.append(edge)
						connections[randomIndex] = 1
						connections[edge] = 0
						#if there are nodes for the target edge to be re-wired to, picks a random index from RandomIndexes, sets that connection to 1, sets the original edge to 0, adds the target edge back to randomIndexes as other edges can now be re-wired to it.
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

	def ising_plot(self, fig, ax):
		'''
		Version of the plot method used in task 5 for plotting the ising model on the network as it develops. The same as the plot method except that the figure and axis are the same for each plot so are passed as arguments. The node's colour also changes differently depending on its value.
		'''
		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])
		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)
			if node.value == 1: 
				colour = "blue"
			else:
				colour = "red"
			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=colour)
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

def random_pop(row, col, agree_prob):
	'''
	This function creates a random array of a population using an agree probability.
	Inputs: 
		row (int)
		col (int)
		agree_prob(float)
	Returns: 
		grid(numpy array)
	'''
	gridTemp = np.random.rand(col, row)
	grid = []
	for row in gridTemp:
		embedGrid = []
		for item in row:
			#creates random array, iterates through each item in the array
			if item > agree_prob:
					embedGrid.append(-1)
			else:
				embedGrid.append(1)
			#if each item in the list is greater than the agree probability, it is set to disagree, otherwise it is set to agree.
		grid.append(embedGrid)
	return np.array(grid)

def agreement_calc(old_value, neighbours, external):
	agreement = 0
	for item in neighbours:
		agreement += (old_value*item)
	agreement += external*old_value
	#calculates the agreement using each of the neighbouts and the cell's previous value.
	return agreement

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
	old_value = population[row][col]
	neighbours = []
	if row > 0:
		neighbours.append(population[row-1][col])
	else:
		neighbours.append(population[row][col])
	if row < len(population)-1:
		neighbours.append(population[row+1][col])
	else:
		neighbours.append(population[0][col])
	if col > 0:
		neighbours.append(population[row][col-1])
	else:
		neighbours.append(population[row][col])
	if col < len(population[row])-1:
		neighbours.append(population[row][col+1])
	else:
		neighbours.append(population[row][0])
	#finds the neighbours for the target cell. If the target cell is on any of the edges of the population grid, its neighbour will be on the opposite edge.
	agreement = agreement_calc(old_value, neighbours, external)
	return agreement

def network_agreement(network, node, external=0.0):
	'''
	Calculates the change in agreement for a single node in a network using the opinions and its neighbours. Takes the node and network as well as the external pull on opinions as parameters, returns the change in agreement.
	'''
	neighbour_indexes = [i for i in range(0, len(node.connections)) if node.connections[i] == 1]
	old_value = node.value
	neighbours = [network.nodes[neighbour].value for neighbour in neighbour_indexes]
	agreement = agreement_calc(old_value, neighbours, external)
	return agreement

def ising_step(population, external=0.0, tolerance=1.0):
	'''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
			tolerance (float) -optional - the tolerance of the society to those who have different opinions to their neighbours
	'''
	n_rows, n_cols = population.shape
	row = np.random.randint(0, n_rows)
	col  = np.random.randint(0, n_cols)
	agreement = calculate_agreement(population, row, col, external)
	#picks a random cell in the population grid, calculates its agreement
	
	if agreement < 0:
		population[row, col] *= -1
	#if the agreement is negative, flips the value of the cell
	else:
		if tolerance > 0:
			prob = np.e**(-agreement/tolerance)
			event = np.random.rand()
			if prob > event:
				population[row][col] *= -1
	#if the agreement is positive or 0, flips the value at a probability calculated using the agreement and tolerance
 
def network_ising_step(network, external=0.0, tolerance=1.0):
	'''
	Performs a single step of the ising model on a network. Randomly selects a node and updates its value based on its agreement with its neighbours. Returns the mean opinion of the network so this can be trakced over time. 
	'''
	node = network.nodes[random.randint(0, len(network.nodes)-1)]
	agreement = network_agreement(network, node, external)
	#randomly selects a node and calculates its agreement

	if agreement < 0:
		node.value *=-1
	else:
		if tolerance > 0:
			prob = np.e**(-agreement/tolerance)
			event = np.random.rand()
			if prob > event:
				node.value *= -1
	#uses identical logic to the ising_step() function but updates a node's value instead of an array item's value
	opinions = [node.value for node in network.nodes]
	mean_opinion = mean(opinions)
	#calculates the mean opinion of the network
	return mean_opinion

def add_agreement(population):
	'''
	Assigns agree (1) or disagree (-1) values to each node in a network randomly.
	'''
	for node in population.nodes:
		random_num = random.randint(0, 1)
		if random_num == 0:
			random_num = -1
		node.value = random_num
		

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
	assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
	assert(calculate_agreement(population,1,1,-10)==14), "Test 10"
	print("Tests passed")


def ising_main(population, alpha=1.0, external=0.0, is_network=False):

	# Iterating an update 100 times
	if is_network == False:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()
		im = ax.imshow(population, interpolation='none', cmap='RdPu_r')
		for frame in range(100):
			for step in range(1000):
					ising_step(population, external, alpha)
			print('Step:', frame, end='\r')
			plot_ising(im, population)
	
	if is_network == True:
		add_agreement(population)
		#adds agreement values to the network
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()
		#creates a figure to display the network on
		mean_opinions = []
		for frame in range(100):
			mean_opinion = network_ising_step(population, external, alpha)
			mean_opinions.append(mean_opinion)
			print('Step:', frame, end='\r')
			population.ising_plot(fig, ax)
			plt.pause(0.1)
			#iterates though 100 steps of the network, performs steps whcih both update the network and return the mean opinion to be graphed over time. Plots the current state of the network.
		plt.pause(2.0)
		plt.close()
		fig2 = plt.figure()
		ax2 = fig2.add_subplot(111)
		#ends the first plot and creates a second
		nums = np.arange(len(mean_opinions))
		plt.plot(nums, mean_opinions)
		plt.xlabel("Time")
		plt.ylabel("Mean opinion")
		plt.show()
		#plots the change in mean opinion over time

'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

def defuant_main(pop, beta, threshold, timestep):
	# Generate initial population with random opinions
	population = np.random.rand(pop)
	
	# Create subplots for visualization
	fig, (ax1, ax2) = plt.subplots(1, 2)
	plt.ion()  # Turn on interactive mode for dynamic plotting
	
	# Iterate over timesteps
	for i in range(timestep):
		# Plot histogram of opinions
		plot_opinions(population, i+1, ax1)
		
		# Plot individual opinions over time
		plot_opinions1(population, i+1, ax2, beta, threshold)

		# Update opinions of the population
		for j in range(timestep):
			update_opinions(population, beta, threshold)
		
		# This section can be uncommented to see the full animation of the plot
		

		# Clear the histogram plot for the next timestep, except for the last timestep
		if i != timestep-1:
			ax1.clear()
		
	# Turn off interactive mode and display plots
		plt.ioff()
	plt.show()

# Function to update opinions based on Deffuant model dynamics
def update_opinions(population, beta, threshold):
	# Select a random individual
	initial = random.randint(0, len(population)-1)
	
	# Determine the index of the neighbor, handling boundary conditions
	neighbour = (initial + random.choice([-1, 1])) % len(population)

	# Calculate the difference in opinions
	difference = abs(population[initial] - population[neighbour])
	
	# Update opinions if the difference is below the threshold
	if difference < threshold:
		# Update opinions based on the difference and beta parameter
		update_value = beta * (population[neighbour] - population[initial])
		update_value1 = beta * (population[initial] - population[neighbour])
		population[initial] += update_value
		population[neighbour] += update_value1 
	return population

# Function to plot histogram of opinions at each timestep
def plot_opinions(population, timestep, ax):
	bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	ax.hist(population, bins=bins)
	ax.set_title(f"Timestep = {timestep}")
	ax.set_xlabel('Opinion')

# Function to plot individual opinions over time
def plot_opinions1(population, timestep, ax, beta, threshold):
	x = [timestep] * len(population)
	ax.scatter(x, population, color="red")
	ax.set_title(f"Coupling: {beta}, Threshold: {threshold}")

# Function to test the Deffuant model
def test_defuant():
	for i in range(10):
		assert update_opinions([0.2,0.4,0.8], 0.5, 0.5)==[0.2,0.4,0.8] or [0.3,0.3,0.8] or [0.2,0.6,0.6]
		assert update_opinions([0.1,0.3,0.6], 0.1, 0.5)==[0.12,0.28,0.6] or [0.1,0.33,0.57] or [0.1,0.3,0.6]
		assert update_opinions([0.4,0.45,0.6], 0.5, 0.1)==[0.425,0.425,0.6] or [0.3,0.3,0.6]
		assert update_opinions([0.3,0.4,0.9], 0.1, 0.2)==[0.31,0.39,0.9] or [0.3,0.4,0.9] 
		assert update_opinions([0.3,0.4,0.9], 0.5, 0.5)==[0.35,0.35,0.9] or [0.3,0.4,0.9] 
	print("Tests passed")


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
	# Handles flags using the module "Argparse".
	## Optional arguments - Used for a random network and tests correspondingly.
	parser = argparse.ArgumentParser()
	parser.add_argument("-network")
	parser.add_argument("-test_network",action="store_true")

	parser.add_argument("-ising_model", action="store_true")
	parser.add_argument("-external", default=0.0, type=float)
	parser.add_argument("-alpha", default=1.0, type=float)
	parser.add_argument("-test_ising", action="store_true")

	parser.add_argument("-use_network", default=0, type=int)

	parser.add_argument("-defuant", action="store_true")
	parser.add_argument("-beta", default=0.2, type=float)
	parser.add_argument("-threshold", default=0.2, type=float)
	parser.add_argument("-test_defuant", action="store_true")

	## Optional argument - Used to help the user with understanding the structure of the network
	parser.add_argument("-analysis",action="store_true")

	parser.add_argument("-ring_network", default=0, type=int)
	parser.add_argument("-small_world", default=0, type=int)
	parser.add_argument("-re_wire", default=0.2, type=float)

	args = parser.parse_args()
	
	if args.use_network == 0:
		ising_pop = random_pop(100, 100, 0.5)
		is_network = False
	else:
		network = Network()
		network.make_small_world_network(args.use_network)
		ising_pop = network
		is_network = True

	alpha = args.alpha
	external = args.external

	if args.ising_model:
		ising_main(ising_pop, alpha, external, is_network)
	
	if args.test_ising:
		test_ising()
	
	# Set default values for population size and timestep
	pop = 100
	timestep = 100

	# Assign values of beta and threshold from command-line arguments
	beta = args.beta
	threshold = args.threshold
	
	# Check if the '-defuant' flag is provided
	if args.defuant:
		# Call the 'defuant_main' function with specified parameters
		defuant_main(pop, beta, threshold, timestep)
	if args.test_defuant:
		test_defuant()

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


	ring_N = args.ring_network
	if ring_N > 0:
		ring_network = Network()
		ring_network.make_ring_network(ring_N)
		ring_network.plot()
		plt.show()
	
	re_wire = args.re_wire
	small_N = args.small_world
	if small_N > 0:
		small_network = Network()
		small_network.make_small_world_network(small_N, re_wire)
		small_network.plot()
		plt.show()


if __name__=="__main__":
	main()

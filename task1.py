import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#READ: Currently this only contains the code for task 1, please copy paste the code from the other tasks over to complete them. 

def random_pop(row, col, agree_prob):
    gridTemp = np.random.rand(col, row)
    grid = []
    for row in gridTemp:
        embedGrid = []
        for item in row:
            if item > agree_prob:
                    embedGrid.append(-1)
            else:
                embedGrid.append(1)
        grid.append(embedGrid)
    return grid

def flip(int):
      if int == 1:
            return -1
      else:
            return 1

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
    if row < len(population)-1:
         neighbours.append(population[row+1][col])
    if col > 0:
         neighbours.append(population[row][col-1])
    if col < len(population[row])-1:
         neighbours.append(population[row][col+1])
    
    agreement = 0
    for item in neighbours:
          agreement += (old_value*item)
    agreement += external*old_value
    return agreement

def ising_step(population, external=0.0, tolerance=0.0):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external "pull" on opinion
    '''
    
    for row in range(0, len(population)):
        for col in range(0, len(population[row])):
            agreement = calculate_agreement(population, row, col, external=0.0)
            if agreement < 0:
                population[row][col] *= -1
            else:
                if tolerance > 0:
                    prob = np.e**(-agreement/tolerance)
                    event = np.random.rand()
                    if prob > event:
                        population[row][col] *= -1

	

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


def ising_main(population, alpha=1.0, external=0.0):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        for step in range(10):
            ising_step(population, external, alpha)
        print('Step:', frame, end='\r')
        plot_ising(im, population)

def main():
    args = sys.argv
    for i in range(0, len(args)):
        if args[i] == "-ising_model":
            if i+1 < len(args):
                i += 1
                if args[i] == "-external":
                    i+=1
                    ising_main(random_pop(100, 100, 0.5), 1.0, args[i])
                elif args[i] == "-alpha":
                     i+=1
                     ising_main(random_pop(100, 100, 0.5), args[i])
    
	#You should write some code for handling flags here

if __name__=="__main__":
	main()
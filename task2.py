import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
# Main function to run the Deffuant model simulation
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
        # Draw and pause to allow for interactive plotting
        plt.draw()
        plt.pause(0.01)
        
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

# Define the main function to handle command-line arguments and execute the program
def main():
    # Create an ArgumentParser object with a description
    parser = argparse.ArgumentParser(description="PLOT")
    
    # Define command-line arguments using add_argument method
    parser.add_argument("-defuant", action="store_true")
    parser.add_argument("-beta", default=0.2, type=float)
    parser.add_argument("-threshold", default=0.2, type=float)
    parser.add_argument("-test_defuant", action="store_true")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
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

# Call the main function if the script is executed directly
if __name__ == "__main__":
    main()
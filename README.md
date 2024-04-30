<h1> FCP Summative Assessment - Modelling Networks and Opinion Dynamics </h1>
Github Repository: https://github.com/elementallife101/FCP
<br>

Description:
In today's society, it is important to consider how opinions between different people spreads over time. For example, how a political view can shift over time or how social media's influence can vary over time. This is the key aim of opinion dynamics. Through different computational models, this allows us to more accurately predict how opinions can differ and what role social connections play in the outcomes that we observe in everyday society. Furthermore, this model as it helps with predicting future opinions, so would be useful for opinion surveys and expected results, such as the national polls, allowing for companies to react much faster to any anticipated outcomes, as these models allow us to predict larger opinion changes.

Our model attempts to predict the different opinion changes present within a smaller network. As our assigned FCP summative assessment, we have modelled the opinion dynamics of people using several different models in order to examine how each model predicts the different attitudes over time. 

Contents:
- Task 1 (Ising Model)
- Task 2 (Defuant Model)
- Task 3 (Networks)
- Task 4 (Small World Networks)
- Task 5 (Combining Networks)

How to install:

1) Install Python 3. The link for this can be found here: https://www.python.org/downloads/
2) Install the required dependencies for the module.
3) Run via bash terminal using the flags indicated below. (Structure: python FCP_assignment.py <flag> <argument>)

Task 1 (Ising Model):
- Required Dependencies
	-Argparse
	-Numpy
	-Matplotlib
- Flags
	- -ising_model
		This argument generates a random population and runs the ising model, plotting the results.
	- -alpha (Type:int)
		This argument gives a value for the "alpha" variable used in the ising_model to represent the tolerance of society to those who disagree with their neighbours. If left out, the value will default to 1.0.
	- -external (Type:int)
		This argument gives a value for the external pull on opinions. If left out, the value will default to 0.0.
	- -test_ising
		This argument runs through the tests provided to show whether the ising model is working as intended.

Task 2 (Defuant Model):
- Required Dependencies
- Flags

Task 3 (Networks):
- Required Dependencies
	- Argparse (Install Link: https://pypi.org/project/argparse/)
	- Numpy (Install Link: https://numpy.org/install/)
- Flags
	- -network <size(Type:int)>
		This argument generates a random network of the specified size. It then calculates and returns the mean degree, the mean clustering coefficient and mean path length of the 		randomly generated network.
	- -test_network 
		This argument runs through the test assertions which were provided within the specification to ensure that the module is completing the intended operation.
	- -analysis
		This argument is designed to help users visualise the data structures and formats present within the code. When called, it returns each node and the connections it has for 		a random network of size 10.

Task 4 (Small World Networks):
- Required Dependencies
	-Argparse
	-Numpy
	-Matplotlib
- Flags
	- -ring_network <size(Type:int)>
		This argument creates a ring network of the specified size. It then plots this network.
	- -small_world <size(Type:int)>
		This argument creates a small world network of the specified size and re-wire probability, it then plots this network.
	- -re_wire (Type:int)
		This argument gives a value for the re-wire probability of the small world network. If left out, the value will default to 0.2.

Task 5 (Combining Models):
- Required Dependencies
- Flags

Output Examples:


Credits:
Chenghe Tang - 
Flavio Vela - 
Rory Sutherland - 
Tycho Twohig - https://github.com/elementallife101/FCP

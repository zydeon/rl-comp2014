=================================
RL-Glue Python Invasive-Species README
=================================
----------------------------
Introduction
----------------------------
This is Invasive Species domain with RL-Glue Python codec. According to RL-Glue codec, there are three main files:
1) InvasiveEnvironment.py, which has the RL-Glue environment interface
2) InvasiveAgent.py, which has the RL-Glue agent interface
3) InvasiveExperiment.py, which provides an example experimental setup using the model.

InvasiveEnvironment.py starts with a default setting. To customize the environment, some input parameters need to be provided. The InvasiveEnvironment.__init__ function takes two input parameters have following types defined in the SimulateNextState.py module:
1) SimulationParameterClass contains related parameters regarding the environment such as production rate and death rate.
2) ActionParameterClass contains the parameters which define the cost function

The above classes are contained in the SimulateNextState.py file. The main simulation method is in this file and named simulateNextState().
The file Utilities.py provides an encapsulated class which provides useful methods for calculation and creating a graph.

This example requires the Python Codec:
http://glue.rl-community.org/Home/Extensions/python-codec

Required Packages
----------------------------
Other than pyhton basic packages, you need to have numpy,rlglue, and networkx
For java agent, you need to have apache commons-lang3-3.1.jar

Running
----------------------------
- These instructions assume that you have rl_glue (or rl_glue.exe) installed on your path so that you don't have to type the full path to it.
- They also assume that the RL-Glue Python codec has been installed to your Python path.  If not, you will need to set your Python path to include them or add it at each step (one example is given below).
- Make sure that the number of reaches and the habitat size match in InvasiveEnvironment.py and InvasiveExperiment.py.

Type the following in different console/terminal windows:
#If you want to do them in the same terminal window, append an ampersand & to each line
$> python invasive_agent.py
#Alternatively, if you don't have the Python codec on your Python path
#$> PYTHONPATH=/path/to/python/codec/src python InvasiveAgent.py

$> python InvasiveEnvironment.py
$> python InvasiveExperiment.py
$> rl_glue #(maybe rl_glue.exe)


----------------------------
More Information
----------------------------
You can find more information by looking into docs folder.

Majid Alkaee Taleghan
alkaee@gmail.com

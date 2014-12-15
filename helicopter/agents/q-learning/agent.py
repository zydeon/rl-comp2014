import random
import sys
import copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation

from random import Random

class skeleton_agent(Agent):
	randGenerator=Random()
	lastAction=Action()
	lastObservation=Observation()
	states_diff_list = []
	
	def agent_init(self, taskSpecString):
		TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString) 
		if(TaskSpec.valid):
			self.discountFactor = TaskSpec.getDiscountFactor()
			self.rangeObservation = TaskSpec.getDoubleObservations()
			self.rangeAction = TaskSpec.getDoubleActions()
		else:
			print "Task Spec could not be parsed: "+taskSpecString;
		
	def agent_start(self, observation):
		thisIntAction=self.randGenerator.randint(0,1)
		returnAction=Action()
		returnAction.intArray=[thisIntAction]
		
		lastAction=copy.deepcopy(returnAction)
		lastObservation=copy.deepcopy(observation)

		return returnAction
	
	def agent_step(self,reward, observation):
		#Generate random action, 0 or 1
		thisIntAction=self.randGenerator.randint(0,1)
		returnAction=Action()
		returnAction.intArray=[thisIntAction]
		
		lastAction=copy.deepcopy(returnAction)
		lastObservation=copy.deepcopy(observation)

		return returnAction
	
	def agent_end(self,reward):
		pass
	
	def agent_cleanup(self):
		pass
	
	def agent_message(self,inMessage):
		if inMessage=="what is your name?":
			return "my name is skeleton_agent, Python edition!";
		else:
			return "I don't know how to respond to your message";


if __name__=="__main__":
	AgentLoader.loadAgent(skeleton_agent())
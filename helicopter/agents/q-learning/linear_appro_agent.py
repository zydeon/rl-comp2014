import sys
import copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random
import numpy

class helicopter_agent(Agent):
	randGenerator=Random()
	lastAction= None
	lastObservation= None
	rangeAction = None
	rangeObservation = None
	discountFactor = 1

      	#for approximate reward function 
	reward_list = []
	observation_list = []
	reward_weight = []
        # end

	#for approximate value function
	states_diff_list = []
	action_list = []
	value_function_weight = []
	#end

	def agent_init(self,taskSpecString):
		"""
			obtain range of observation , range of aciont and discount factor

		"""
		TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString) 
		if(TaskSpec.valid):
			self.discountFactor = TaskSpec.getDiscountFactor()
			self.rangeObservation = TaskSpec.getDoubleObservations()
			self.rangeAction = TaskSpec.getDoubleActions()
		else:
			print "Task Spec could not be parsed: "+taskSpecString;
		
		 

	def agent_start(self,observation):
		#Generate random action, 0 or 1  

		print " " 
		thisDoubleAction=self.approximateAction()
 		
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
		#test for approximate value function		
			
 		# self.last_observation_list.append(observation.doubleArray)
		self.action_list.append(thisDoubleAction)

		#end of test

		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		 
		return returnAction
	
	def agent_step(self,reward, observation):
		self.states_diff_list.append([a - b for (a, b) in zip (observation.doubleArray, self.lastObservation.doubleArray)])

		self.lastObservation=copy.deepcopy(observation)
		 
		self.approximateValueFunction()		
		#end of test
		print reward
		#test how reward approximation works
		self.approximateRewardFunction(reward,observation)
		 
		#end of test

		thisDoubleAction=self.approximateAction()  
		  
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
		 
		 
		#approximate value function 
		self.action_list.append(thisDoubleAction)
		#end of test


		
		
		self.lastAction=copy.deepcopy(returnAction)
		
		 
		return returnAction
	
	def agent_end(self,reward): 
		self.action_list = self.action_list[:-1]
		# self.reward_list = []
		# self.observation_list = []
		
		# self.last_observation_list = []
		# self.action_list = []
		# self.next_observation_list = []
		# self.value_function_weight = []		
		# self.reward_weight = []
		# self.lastAction = None
		# self.lastObservation = None
		pass


	def agent_cleanup(self):
		pass
	
	def agent_message(self,inMessage):
		pass
 	
	def approximateValueFunction(self):
		"""
			try to approximate value function: V(s) = A*s+B*a


		"""
		if(len(self.action_list)<2):
			return 
		
		import numpy
		import math
		from operator import sub
		 
		Coff = numpy.linalg.lstsq(self.action_list, self.states_diff_list)[0]
		
		self.value_function_weight = numpy.matrix.transpose(Coff)

		#get the error
		# total_error = 0
		# for i in range(0,len(self.action_list)):
		# 	value = numpy.add(numpy.inner(self.value_function_weight,self.action_list[i]),self.last_observation_list[i])
		# 	total_error += numpy.linalg.norm(numpy.subtract(value,self.next_observation_list[i]))
		 
				
	def approximateRewardFunction(self,reward,observation):
		"""	
			try to approximate reward function r = A*s  

		"""
		self.reward_list.append(reward)
		self.observation_list.append(observation.doubleArray)

		if(len(self.reward_list)<2):
			return
		
		import numpy
		import math
		A = numpy.linalg.lstsq(self.observation_list,self.reward_list)[0]
		
		 
		self.reward_weight = A
		 
 		#get the error
		total_error =0
		for i in range(0,len(self.reward_list)):
			value = numpy.inner(A,self.observation_list[i])
			 
			total_error +=math.pow(value-self.reward_list[i],2)  

		  

	def approximateAction(self):
		"""
			choose action according to approximate reward function and approximate value function


		"""
		#generate several action and compair the action with highest reward with previous reward , if it is lower than previous one
 		# regenerate 
		if(len(self.reward_list) < 2):
			return self.randomSmoothAction()

		 
		action = []
		action_reward = -10000 
		while(True):
			for i in range(0,100):
				randAction = self.randomSmoothAction() 
				reward  = numpy.inner(self.reward_weight,numpy.add(numpy.inner				 		(self.value_function_weight,randAction),self.lastObservation.doubleArray))
				if(reward > action_reward):
					action = randAction
					action_reward = reward
			if(action_reward > numpy.inner(self.reward_weight,self.lastObservation.doubleArray)):				
				# print "predicted reward===",action_reward 	
				#print "predicted state====    ",numpy.add(numpy.inner				 		(self.value_function_weight,randAction),self.lastObservation.doubleArray)			
				 				
				return action
			 
						

	def randomAction(self):
		"""
			generate random action.--- test purpose

		"""
		 
		action = []
		action_length = len(self.rangeAction) 
		for i in range(0,action_length):
			action.append(self.randGenerator.uniform(self.rangeAction[i][0],self.rangeAction[i][1]))		
		
		 
		return action
		 		
	def randomSmoothAction(self):
		"""
			generate random action smoothly.--- test purpose

		"""
		 
		import math
		if(self.lastAction==None):
			return [0.0,0.0,0.0,0.0]
		action = []
		action_length = len(self.rangeAction) 
		for i in range(0,action_length):
			sign  = self.randGenerator.uniform(0,1)
			if(sign < 0.5):
				action.append(math.pow(self.randGenerator.uniform(0,0.15)+self.lastAction.doubleArray[i],2)+self.randGenerator.uniform(0,0.1))		
			else:
				action.append(math.pow(self.lastAction.doubleArray[i]-self.randGenerator.uniform(0,0.15),2)-self.randGenerator.uniform(0,0.1))		
		 		  
		return action
if __name__=="__main__":
	AgentLoader.loadAgent(helicopter_agent())
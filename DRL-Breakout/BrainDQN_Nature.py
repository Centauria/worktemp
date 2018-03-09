# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import tensorflow as tf 
import numpy as np 
import random
from collections import deque 

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 500. # timesteps to observe before training
EXPLORE = 50000000. #200000 #frames over which to anneal epsilon
FINAL_EPSILON = 0#0.001 # final value of epsilon
INITIAL_EPSILON = 0.1#0.01 # starting value of epsilon
DELTA_EPSILON=(INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 5000 #100
SAVE_TIME=10000
LEARNING_RATE=0.01

class BrainDQN:

	def __init__(self,actions):
		# init replay memory
		self.replayMemory = deque()
		# init some parameters
		self.timeStep = 0
		self.observe_time=0
		self.epsilon = INITIAL_EPSILON
		self.actions = actions
		
		self._build_net()
		e_params=tf.get_collection('eval-net params')
		t_params=tf.get_collection('target-net params')
		self.copyTargetQNetworkOperation=[tf.assign(t,e) for t,e in zip(t_params,e_params)]

		# saving and loading networks
		self.saver = tf.train.Saver()
		self.merge=tf.summary.merge_all()
		self.session = tf.InteractiveSession()
		
		self.writer=tf.summary.FileWriter("logs/", self.session.graph)
		
		self.session.run(tf.global_variables_initializer())
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.session, checkpoint.model_checkpoint_path)
			print ("Successfully loaded:", checkpoint.model_checkpoint_path)
			self.timeStep=int(checkpoint.model_checkpoint_path.split('-')[-1])
			self.epsilon=INITIAL_EPSILON-DELTA_EPSILON*self.timeStep
		else:
			print ("Could not find old network weights")

	def _build_net(self):
		# build evaluate network
		self.stateInput=tf.placeholder(tf.float32,[None,80,80,4],name='input')
		self.actionInput=tf.placeholder(tf.float32,[None,self.actions],name='Q-target')
		with tf.variable_scope('eval-net'):
			# c_names(collections_names) are the collections to store variables
			c_names,w_initializer,b_initializer=\
				['eval-net params',tf.GraphKeys.GLOBAL_VARIABLES],\
				tf.truncated_normal_initializer(stddev = 0.01),\
				tf.constant_initializer(0.1)
			
			# first layer. collections is used later when assign to target net
			with tf.variable_scope('conv1'):
				w_conv1=tf.get_variable('w_conv1',[8,8,4,32],\
					initializer=w_initializer,collections=c_names)
				b_conv1=tf.get_variable('b_conv1',[32],\
					initializer=b_initializer,collections=c_names)
				h_conv1=tf.nn.relu(self.conv2d(self.stateInput,w_conv1,4) + b_conv1)
				h_pool1=self.max_pool_2x2(h_conv1)
			
			# second layer. 
			with tf.variable_scope('conv2'):
				w_conv2=tf.get_variable('w_conv2',[4,4,32,64],\
					initializer=w_initializer,collections=c_names)
				b_conv2=tf.get_variable('b_conv2',[64],\
					initializer=b_initializer,collections=c_names)
				h_conv2=tf.nn.relu(self.conv2d(h_pool1,w_conv2,2) + b_conv2)
			
			# third layer.
			with tf.variable_scope('conv3'):
				w_conv3=tf.get_variable('w_conv3',[3,3,64,64],\
					initializer=w_initializer,collections=c_names)
				b_conv3=tf.get_variable('b_conv3',[64],\
					initializer=b_initializer,collections=c_names)
				h_conv3=tf.nn.relu(self.conv2d(h_conv2,w_conv3,1)+b_conv3)
				h_conv3_flat=tf.reshape(h_conv3,[-1,1600])
			
			# first full link layer.
			with tf.variable_scope('fc1'):
				w_fc1=tf.get_variable('w_fc1',[1600,512],\
					initializer=w_initializer,collections=c_names)
				b_fc1=tf.get_variable('b_fc1',[512],\
					initializer=b_initializer,collections=c_names)
				h_fc1=tf.nn.relu(tf.matmul(h_conv3_flat,w_fc1)+b_fc1)
			
			# Q value layer.
			with tf.variable_scope('fc2'):
				w_fc2=tf.get_variable('w_fc2',[512,self.actions],\
					initializer=w_initializer,collections=c_names)
				b_fc2=tf.get_variable('b_fc2',[self.actions],\
					initializer=b_initializer,collections=c_names)
				self.QValue=tf.matmul(h_fc1,w_fc2)+b_fc2
		
		with tf.variable_scope('loss'):
			loss=tf.reduce_mean(tf.squared_difference(self.actionInput,self.QValue))
		
		with tf.variable_scope('train'):
			# self.trainStep=tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)
			self.trainStep=tf.train.AdamOptimizer(1e-6).minimize(loss)
		
		# build target net
		self.stateInputT=tf.placeholder(tf.float32,[None,80,80,4],name='input_')
		self.yInput = tf.placeholder(tf.float32, [None],name='yinput_') 
		with tf.variable_scope('target-net'):
			c_names=['target-net params',tf.GraphKeys.GLOBAL_VARIABLES]
			
			# first layer. collections is used later when assign to target net
			with tf.variable_scope('conv1'):
				w_conv1=tf.get_variable('w_conv1',[8,8,4,32],\
					initializer=w_initializer,collections=c_names)
				b_conv1=tf.get_variable('b_conv1',[32],\
					initializer=b_initializer,collections=c_names)
				h_conv1=tf.nn.relu(self.conv2d(self.stateInputT,w_conv1,4) + b_conv1)
				h_pool1=self.max_pool_2x2(h_conv1)
			
			# second layer. 
			with tf.variable_scope('conv2'):
				w_conv2=tf.get_variable('w_conv2',[4,4,32,64],\
					initializer=w_initializer,collections=c_names)
				b_conv2=tf.get_variable('b_conv2',[64],\
					initializer=b_initializer,collections=c_names)
				h_conv2=tf.nn.relu(self.conv2d(h_pool1,w_conv2,2) + b_conv2)
			
			# third layer.
			with tf.variable_scope('conv3'):
				w_conv3=tf.get_variable('w_conv3',[3,3,64,64],\
					initializer=w_initializer,collections=c_names)
				b_conv3=tf.get_variable('b_conv3',[64],\
					initializer=b_initializer,collections=c_names)
				h_conv3=tf.nn.relu(self.conv2d(h_conv2,w_conv3,1)+b_conv3)
				h_conv3_flat=tf.reshape(h_conv3,[-1,1600])
			
			# first full link layer.
			with tf.variable_scope('fc1'):
				w_fc1=tf.get_variable('w_fc1',[1600,512],\
					initializer=w_initializer,collections=c_names)
				b_fc1=tf.get_variable('b_fc1',[512],\
					initializer=b_initializer,collections=c_names)
				h_fc1=tf.nn.relu(tf.matmul(h_conv3_flat,w_fc1)+b_fc1)
			
			# Q value layer.
			with tf.variable_scope('fc2'):
				w_fc2=tf.get_variable('w_fc2',[512,self.actions],\
					initializer=w_initializer,collections=c_names)
				b_fc2=tf.get_variable('b_fc2',[self.actions],\
					initializer=b_initializer,collections=c_names)
				self.QValueT=tf.matmul(h_fc1,w_fc2)+b_fc2
		
		with tf.name_scope('summaries'):
			tf.summary.image('input_layer',self.stateInput)
			tf.summary.scalar('loss',loss)
			tf.summary.histogram('Q_values',self.QValue)

	def trainQNetwork(self):
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replayMemory,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]

		# Step 2: calculate y 
		y_batch = np.zeros((BATCH_SIZE))
		QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
		for i in range(0,BATCH_SIZE):
			terminal = minibatch[i][4]
			if terminal:
				y_batch[i]=reward_batch[i]
			else:
				y_batch[i]=reward_batch[i] + GAMMA * np.max(QValue_batch[i])

		self.trainStep.run(feed_dict={
			self.yInput : y_batch,
			self.actionInput : action_batch,
			self.stateInput : state_batch
		})

		# save network every 100000 iteration
		if self.timeStep % SAVE_TIME == 0:
			self.saver.save(self.session,'saved_networks/network-dqn', global_step = self.timeStep)

		if self.timeStep % UPDATE_TIME == 0:
			self.session.run(self.copyTargetQNetworkOperation)

		
	def setPerception(self,nextObservation,action_index,reward,terminal,episode):
		#newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
		action=np.zeros(self.actions)
		action[action_index]=1
		newState = np.append(self.currentState[:,:,1:],nextObservation,axis = 2)
		self.replayMemory.append((self.currentState,action,reward,newState,terminal))
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()
		if self.observe_time<OBSERVE:
			self.observe_time+=1
			state = "observe"
			# print info
			print ("TIMESTEP", self.timeStep, "/ STATE", state, \
	            "/ EPSILON", self.epsilon, "/ EPISODE", episode, "/ REWARD", reward)
		else:
			# Train the network
			self.trainQNetwork()
			
			if self.timeStep <= EXPLORE:
				state = "explore"
			else:
				state = "train"
				
			# print info
			print ("TIMESTEP", self.timeStep, "/ STATE", state, \
	            "/ EPSILON", self.epsilon, "/ EPISODE", episode, "/ REWARD", reward)
			
			rs=self.merge.eval(feed_dict={
				self.stateInput:[self.currentState],
				self.actionInput:[action]
			})
			self.writer.add_summary(rs,global_step=self.timeStep)
			
			self.timeStep += 1
		
		self.currentState = newState

	def getAction(self):
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
		action_index = 0
		if self.timeStep % FRAME_PER_ACTION == 0:
			if random.random() <= self.epsilon:
				action_index = random.randrange(self.actions)
			else:
				action_index = np.argmax(QValue)

		# change episilon
		if self.epsilon > FINAL_EPSILON and not self.observe_time < OBSERVE:
			self.epsilon -= DELTA_EPSILON

		return action_index

	def setInitState(self,observation):
		self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

	def conv2d(self,x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
		

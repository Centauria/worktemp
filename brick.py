import sys
import cv2


import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque



CNN_INPUT_WIDTH = 80
CNN_INPUT_HEIGHT = 80
SERIES_LENGTH = 4
CNN_INPUT_DEPTH = SERIES_LENGTH

REWARD_COFF = 3.0

INITIAL_EPSILON=1.0
FINAL_EPSILON=0.0001
REPLAY_SIZE=50000
BATCH_SIZE=32
GAMMA=0.99
OBSERVE_TIME=500
ENV_NAME='Breakout-v4'
EPISODE=100000
STEP=1500
TEST=10

class ImageProcess:
	# this is the ColorMat2B function
	@staticmethod
	def reshapesize(state):
		state_gray=cv2.cvtColor(cv2.resize(state,(CNN_INPUT_HEIGHT,CNN_INPUT_WIDTH),cv2.COLOR_BGR2GRAY))
		_,state_binary=cv2.threshold(state_gray,5,255,cv2.THRESH_BINARY)
		state_binary_small=cv2.resize(state_binary,(CNN_INPUT_WIDTH,CNN_INPUT_HEIGHT))
		cnn_input_image=state_binary_small.reshape((CNN_INPUT_HEIGHT,CNN_INPUT_WIDTH))
		return cnn_input_image
	@staticmethod
	def reshapeBin(state):
		height=state.shape[0]
		width=state.shape[1]
		nchannel=state.shape[2]
		
		sHeight=int(height*0.5)
		sWidth=CNN_INPUT_WIDTH
		
		state_gray=cv2.cvtColor(state,cv2.COLOR_BGR2GRAY)
		_,state_binary=cv2.threshold(state_gray,5,255,cv2.THRESH_BINARY)
		state_binary_small=cv2.resize(state_binary,(sWidth,sHeight),interpolation=cv2.INTER_AREA)
		
		cnn_input_image=state_binary_small[25:,:]
		cnn_input_image=cnn_input_image.reshape((CNN_INPUT_WIDTH,CNN_INPUT_HEIGHT))
		return cnn_input_image

class Tools:
	@staticmethod
	def gen_weights(shape):
		weight=tf.truncated_normal(shape,stddev=0.01)
		return tf.Variable(weight)
	@staticmethod
	def gen_bias(shape):
		bias=tf.constant(0.01,shape=shape)
		return tf.Variable(bias)
class DQN:
	def __init__(self,env,session,writer):
		self.epsilon=INITIAL_EPSILON
		self.replay_buffer=deque()
		self.recent_history_queue=deque()
		self.action_dim=env.action_space.n
		self.state_dim=CNN_INPUT_HEIGHT * CNN_INPUT_WIDTH
		self.time_step=0
		
		self.session=session
		self.create_network()
		self.observe_time=0
		self.writer=writer
		
		self.session.run(tf.global_variables_initializer())
		
	def create_network(self):
		self.input_layer=tf.placeholder(tf.float32,[None,CNN_INPUT_WIDTH,CNN_INPUT_HEIGHT,CNN_INPUT_DEPTH],name='status-input')
		self.input_action=tf.placeholder(tf.float32,[None,self.action_dim])
		self.input_y=tf.placeholder(tf.float32,[None])
		
		W1=Tools.gen_weights([8,8,4,32])
		b1=Tools.gen_bias([32])
		
		h_conv1=tf.nn.relu(
			tf.nn.conv2d(self.input_layer,W1,
				strides=[1,4,4,1],padding='SAME')
			+b1)
		conv1=tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
		
		W2=Tools.gen_weights([4,4,32,64])
		b2=Tools.gen_bias([64])
		
		h_conv2=tf.nn.relu(
			tf.nn.conv2d(conv1,W2,
				strides=[1,2,2,1],padding='SAME')
			+b2)
		# conv2=tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
		conv2=h_conv2
		
		W3=Tools.gen_weights([3,3,64,64])
		b3=Tools.gen_bias([64])
		
		h_conv3=tf.nn.relu(
			tf.nn.conv2d(conv2,W3,
				strides=[1,1,1,1],padding='SAME')
			+b3)
		# conv3=tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
		conv3=h_conv3
		
		W_fc1=Tools.gen_weights([1600,512])
		b_fc1=Tools.gen_bias([512])
		
		conv3_flat=tf.reshape(conv3,[-1,1600])
		
		h_fc1=tf.nn.relu(tf.matmul(conv3_flat,W_fc1)+b_fc1)
		
		W_fc2=Tools.gen_weights([512,self.action_dim])
		b_fc2=Tools.gen_bias([self.action_dim])
		
		self.Q_value=tf.matmul(h_fc1,W_fc2)+b_fc2
		Q_action=tf.reduce_sum(tf.multiply(self.Q_value,self.input_action),reduction_indices=1)
		self.cost=tf.reduce_mean(tf.square(self.input_y-Q_action))
		
		self.optimizer=tf.train.AdamOptimizer(1e-6).minimize(self.cost)
		
		with tf.name_scope('summaries'):
			tf.summary.image('input_layer',self.input_layer)
			tf.summary.histogram('reward',tf.reduce_max(self.input_y))
			tf.summary.histogram('Q_value',tf.reduce_max(self.Q_value))
			tf.summary.histogram('rewards',self.input_y)
			tf.summary.histogram('Q_values',self.Q_value)
			tf.summary.scalar('reward',tf.reduce_max(self.input_y))
			tf.summary.scalar('Q_value',tf.reduce_max(self.Q_value))
			tf.summary.scalar('cost',self.cost)
	
	def train_network(self,episode):
		self.time_step+=1
		
		minibatch=random.sample(self.replay_buffer,BATCH_SIZE)
		state_batch=[data[0] for data in minibatch]
		action_batch=[data[1] for data in minibatch]
		reward_batch=[data[2] for data in minibatch]
		next_state_batch=[data[3] for data in minibatch]
		done_batch=[data[4] for data in minibatch]
		
		y_batch=[]
		Q_value_batch=self.Q_value.eval(feed_dict={
			self.input_layer: next_state_batch
		})
		
		for i in range(BATCH_SIZE):
			if done_batch[i]:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i]+GAMMA*np.max(Q_value_batch[i]))
		rs,_=self.session.run([tf.summary.merge_all(),self.optimizer],feed_dict={
			self.input_layer: state_batch,
			self.input_action: action_batch,
			self.input_y: y_batch
		})
		self.writer.add_summary(rs,episode)
	
	def percieve(self,state_shadow,action_index,reward,next_state_shadow,done,episode):
		action=np.zeros(self.action_dim)
		action[action_index]=1
		
		self.replay_buffer.append([state_shadow,action,reward,next_state_shadow,done])
		
		self.observe_time+=1
		
		if self.observe_time % 1000 and self.observe_time <= OBSERVE_TIME == 0:
			print('observe_time:',self.observe_time,end='\r')
		
		if len(self.replay_buffer)>REPLAY_SIZE:
			self.replay_buffer.popleft()
		
		if len(self.replay_buffer)>BATCH_SIZE and self.observe_time > OBSERVE_TIME:
			self.train_network(episode)
	
	def get_greedy_action(self,state_shadow):
		rst=self.Q_value.eval(feed_dict={
			self.input_layer: [state_shadow]
		})[0]
		print(np.argmax(rst),np.max(rst),end='\r')
		return np.argmax(rst)
	
	def get_action(self,state_shadow):
		if self.epsilon >= FINAL_EPSILON and self.observe_time > OBSERVE_TIME:
			self.epsilon -=(INITIAL_EPSILON-FINAL_EPSILON)/10000
		
		action=np.zeros(self.action_dim)
		action_index=None
		if random.random()<self.epsilon:
			action_index=random.randint(0,self.action_dim-1)
		else:
			action_index=self.get_greedy_action(state_shadow)
		
		return action_index

if __name__=='__main__':
	env=gym.make(ENV_NAME)
	state_shadow=None
	next_state_shadow=None
	
	session=tf.InteractiveSession()
	writer=tf.summary.FileWriter('./logs',session.graph)
	agent=DQN(env,session,writer)
	saver=tf.train.Saver(max_to_keep=0)
	
	try:
		for episode in range(EPISODE):
			
			if episode % 10 == 0:
				saver.save(session,'checkpoint/brick-ckpt',global_step=episode)
			
			state=env.reset()
			state=ImageProcess.reshapeBin(state)
			
			state_shadow=np.stack((state,state,state,state),axis=2)
			
			for step in range(STEP):
				# env.render()
				action=agent.get_action(state_shadow)
				next_state,reward,done,_=env.step(action)
				next_state=np.reshape(ImageProcess.reshapeBin(next_state),(80,80,1))
				next_state_shadow=np.append(next_state,state_shadow[:,:,:3],axis=2)
				
				agent.percieve(state_shadow,action,reward,next_state_shadow,done,episode)
				state_shadow=next_state_shadow
				print('Episode:',episode,'Step:',step,end='\r')
				
				if done:
					print()
					break
	finally:
		writer.close()
		session.close()

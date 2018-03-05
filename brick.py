import sys
import cv2


import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
import gc
import tracemalloc


CNN_INPUT_WIDTH = 80
CNN_INPUT_HEIGHT = 80
SERIES_LENGTH = 4
CNN_INPUT_DEPTH = SERIES_LENGTH

REWARD_COFF = 3.0

INITIAL_EPSILON=1.0
FINAL_EPSILON=0.0001
REPLAY_SIZE=100
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
		self.replay_buffer=np.zeros((REPLAY_SIZE,3,CNN_INPUT_WIDTH,CNN_INPUT_HEIGHT,CNN_INPUT_DEPTH))
		self.active_replay_index=0
		# self.recent_history_queue=deque()
		self.action_dim=env.action_space.n
		self.state_dim=CNN_INPUT_HEIGHT * CNN_INPUT_WIDTH
		self.time_step=0
		
		self.session=session
		self.create_network()
		self.observe_time=0
		self.writer=writer
		
		self.temp=None
		
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
	
	def train_network(self,step):
		self.time_step+=1
		
		if self.temp==None:
			self.temp={}
			
			if self.active_replay_index<REPLAY_SIZE:
				self.temp['minibatch']=self.replay_buffer[:self.active_replay_index][random.sample(range(self.active_replay_index),BATCH_SIZE)]
			else:
				self.temp['minibatch']=self.replay_buffer[:self.active_replay_index][random.sample(range(REPLAY_SIZE),BATCH_SIZE)]
			self.temp['state_batch']=self.temp['minibatch'][:,0,:,:,:]
			self.temp['next_state_batch']=self.temp['minibatch'][:,1,:,:,:]
			self.temp['action_index_batch']=self.temp['minibatch'][:,2,0,0,0].astype(int)
			self.temp['reward_batch']=self.temp['minibatch'][:,2,1,0,0]
			self.temp['done_batch']=self.temp['minibatch'][:,2,2,0,0]
			
			self.temp['action_batch']=np.zeros((BATCH_SIZE,self.action_dim))
			for i in range(BATCH_SIZE):
				self.temp['action_batch'][i][self.temp['action_index_batch'][i]]=1
				self.temp['done_batch'][i]=True if self.temp['done_batch'][i]==1 else False
			
			self.temp['y_batch']=np.zeros(BATCH_SIZE)
			self.temp['Q_value_batch']=self.Q_value.eval(feed_dict={
				self.input_layer: self.temp['next_state_batch']
			})
		else:
			if self.active_replay_index<REPLAY_SIZE:
				self.temp['minibatch'][:]=self.replay_buffer[:self.active_replay_index][random.sample(range(self.active_replay_index),BATCH_SIZE)]
			else:
				self.temp['minibatch'][:]=self.replay_buffer[:self.active_replay_index][random.sample(range(REPLAY_SIZE),BATCH_SIZE)]
			self.temp['state_batch'][:]=self.temp['minibatch'][:,0,:,:,:]
			self.temp['next_state_batch'][:]=self.temp['minibatch'][:,1,:,:,:]
			self.temp['action_index_batch'][:]=self.temp['minibatch'][:,2,0,0,0].astype(int)
			self.temp['reward_batch'][:]=self.temp['minibatch'][:,2,1,0,0]
			self.temp['done_batch'][:]=self.temp['minibatch'][:,2,2,0,0]
			
			self.temp['action_batch'][:]=0
			for i in range(BATCH_SIZE):
				self.temp['action_batch'][i][self.temp['action_index_batch'][i]]=1
				self.temp['done_batch'][i]=True if self.temp['done_batch'][i]==1 else False
			
			self.temp['y_batch'][:]=0
			self.temp['Q_value_batch'][:]=self.Q_value.eval(feed_dict={
				self.input_layer: self.temp['next_state_batch']
			})
		
		for i in range(BATCH_SIZE):
			if self.temp['done_batch'][i]:
				self.temp['y_batch'][i]=self.temp['reward_batch'][i]
			else:
				self.temp['y_batch'][i]=self.temp['reward_batch'][i]+GAMMA*np.max(self.temp['Q_value_batch'][i])
		rs,_=self.session.run([tf.summary.merge_all(),self.optimizer],feed_dict={
			self.input_layer: self.temp['state_batch'],
			self.input_action: self.temp['action_batch'],
			self.input_y: self.temp['y_batch']
		})
#		del y_batch
#		gc.collect()
		self.writer.add_summary(rs,step)
	
	def percieve(self,state_shadow,action_index,reward,next_state_shadow,done,step):
#		action=np.zeros(self.action_dim)
#		action[action_index]=1
		
		current_index=self.active_replay_index%REPLAY_SIZE
		self.replay_buffer[current_index][0][:]=state_shadow
		self.replay_buffer[current_index][1][:]=next_state_shadow
		self.replay_buffer[current_index][2][0][0][0]=action_index
		self.replay_buffer[current_index][2][1][0][0]=reward
		self.replay_buffer[current_index][2][2][0][0]=done
		
		self.active_replay_index+=1
		
		self.observe_time+=1
		
		if self.observe_time % 1000 and self.observe_time <= OBSERVE_TIME == 0:
			print('observe_time:',self.observe_time,end='\r')
		
		# if len(self.replay_buffer)>REPLAY_SIZE:
			# self.replay_buffer.popleft()
		
		if self.active_replay_index>BATCH_SIZE and self.observe_time > OBSERVE_TIME:
			self.train_network(step)
	
	def get_greedy_action(self,state_shadow):
		rst=self.Q_value.eval(feed_dict={
			self.input_layer: [state_shadow]
		})[0]
		print(np.argmax(rst),np.max(rst),end='\r')
		return np.argmax(rst)
	
	def get_action(self,state_shadow):
		if self.epsilon >= FINAL_EPSILON and self.observe_time > OBSERVE_TIME:
			self.epsilon -=(INITIAL_EPSILON-FINAL_EPSILON)/10000
		
#		action=np.zeros(self.action_dim)
		action_index=None
		if random.random()<self.epsilon:
			action_index=random.randint(0,self.action_dim-1)
		else:
			action_index=self.get_greedy_action(state_shadow)
		
		return action_index

def main():
	tracemalloc.start()
	env=gym.make(ENV_NAME)
	state_shadow=None
	next_state_shadow=None
	
	session=tf.InteractiveSession()
	writer=tf.summary.FileWriter('./logs',session.graph)
	agent=DQN(env,session,writer)
	saver=tf.train.Saver(max_to_keep=0)
#	agent.session.graph.finalize()
	
	full_step=0
	
	try:
		for episode in range(EPISODE):
			
			if episode % 10 == 0:
				saver.save(session,'checkpoint/brick-ckpt',global_step=episode)
			
			state=env.reset()
			state=ImageProcess.reshapeBin(state)
			
			state_shadow=np.stack((state,state,state,state),axis=2)
			
			for step in range(STEP):
				env.render()
				action=agent.get_action(state_shadow)
				next_state,reward,done,_=env.step(action)
				next_state=np.reshape(ImageProcess.reshapeBin(next_state),(80,80,1))
				next_state_shadow=np.append(next_state,state_shadow[:,:,:3],axis=2)
				
				agent.percieve(state_shadow,action,reward,next_state_shadow,done,full_step)
				state_shadow=next_state_shadow
				print('Episode:',episode,'Step:',step,'Full step:',full_step,end='\r')
				full_step+=1
				if done:
					print()
					break
			
	finally:
		snapshot = tracemalloc.take_snapshot()
		top_stats = snapshot.statistics('lineno')
		print("")
		for stat in top_stats[:10]:
			print(stat)
		writer.close()
		session.close()

if __name__=='__main__':
	main()

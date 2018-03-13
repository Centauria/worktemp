# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:46:18 2018

@author: qxb-810
"""

import os
import cv2
import gym
import tensorflow as tf
import numpy as np
import shutil
import moviepy
from BrainDQN_Nature import BrainDQN
from moviepy.editor import VideoClip

CNN_INPUT_WIDTH = 80
CNN_INPUT_HEIGHT = 80
SERIES_LENGTH = 4
CNN_INPUT_DEPTH = SERIES_LENGTH

REWARD_COFF = 3.0

ENV_NAME='Breakout-v4'
EPISODE=100000
MAX_STEP=1500

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
	def reshapeHalf(state):
		height=state.shape[0]
#		width=state.shape[1]
#		nchannel=state.shape[2]
		
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

def make_frame_best(t):
	frame=cv2.imread('temp/%s-%s.png'%(0,t))
	return frame

class Evaluator:
	
	def __init__(self,env,agent,tag):
		self.env=env
		self.agent=agent
		self.tag=tag
		self.n_episode=10
		self.rewards=[]
		self.frames=[]
		self.fps=25
		
		agent.e_greedy=False
	
	def __del__(self):
		self.agent.e_greedy=True
	
	def run(self):
		for episode in range(self.n_episode):
			done=False
			frame=0
			state=self.env.reset()
			self.agent.setInitState(ImageProcess.reshapeHalf(state))
			self.rewards.append(0)
			# self.frames.append([])
			# self.frames[-1].append(state)
			cv2.imwrite('temp/%s-%s.png'%(episode,frame),state)
			while not done:
				action=self.agent.getAction()
				state,reward,done,_=self.env.step(action)
				self.agent.setPerception(np.reshape(ImageProcess.reshapeHalf(state),(80,80,1)),action,reward,done,episode)
				self.rewards[-1]+=reward
				# self.frames[-1].append(state)
				frame+=1
				cv2.imwrite('temp/%s-%s.png'%(episode,frame),state)
			self.frames.append(frame)
		
		# self.allframes=np.concatenate(self.frames)
		Evaluator.best=np.argmax(self.rewards)
		print('All scores:',self.rewards)
		print('Best score:',np.max(self.rewards))
	
	
	
	def export_best(self):
		global make_frame_best
		print('Eval-best=',0)
		print('Eval-make-best=',make_frame_best)
		animation=VideoClip(make_frame_best,duration=self.frames[0]/self.fps)
		animation.write_videofile('best-%s.mp4'%self.tag,fps=self.fps,codec='mpeg4')

def main():
	env=gym.make(ENV_NAME)
	
	agent=BrainDQN(env.action_space.n)
	
	try:
		for episode in range(EPISODE):
			state=env.reset()
			state=ImageProcess.reshapeHalf(state)
			
			agent.setInitState(state)
			
			for step in range(MAX_STEP):
				# env.render()
				action=agent.getAction()
				next_state,reward,done,_=env.step(action)
				next_state=np.reshape(ImageProcess.reshapeHalf(next_state),(80,80,1))
				agent.setPerception(next_state,action,reward,done,episode)
				if done:
					break
			
			if episode%10==3:
				print('Generate evaluation video')
				if not os.path.exists('temp'):
					os.mkdir('temp')
				evaluator=Evaluator(env,agent,episode)
				evaluator.run()
				evaluator.export_best()
				# evaluator.export_all()
				del evaluator
				shutil.rmtree('temp')
			
	finally:
		pass

if __name__=='__main__':
	main()

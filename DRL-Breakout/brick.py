# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:46:18 2018

@author: qxb-810
"""
from __future__ import print_function
import cv2
import gym
import numpy as np
from BrainDQN_Nature import BrainDQN

CNN_INPUT_WIDTH = 80
CNN_INPUT_HEIGHT = 80
SERIES_LENGTH = 4
CNN_INPUT_DEPTH = SERIES_LENGTH

REWARD_COFF = 3.0

ENV_NAME='Breakout-v4'
EPISODE=100000
MAX_STEP=3000

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

def main():
	env=gym.make(ENV_NAME)
	
	agent=BrainDQN(env.action_space.n)
	
	try:
		for episode in range(EPISODE):
			state=env.reset()
			state=ImageProcess.reshapeHalf(state)
			
			agent.setInitState(state)
			life=5
			for step in range(MAX_STEP):
				# env.render()
				action=agent.getAction()
				next_state,reward,done,info=env.step(action)
				next_state=np.reshape(ImageProcess.reshapeHalf(next_state),(80,80,1))
				if info['ale.lives']<life:
					#reward-=(life-info['ale.lives'])
					life=info['ale.lives']
				agent.setPerception(next_state,action,reward,done,episode)
				if done:
					break
			if episode%500==100:
				agent.e_greedy=False
				for e in range(10):
					state=env.reset()
					state=ImageProcess.reshapeHalf(state)
					
					agent.setInitState(state)
					life=5
					for step in range(MAX_STEP):
						# env.render()
						action=agent.getAction()
						next_state,reward,done,info=env.step(action)
						next_state=np.reshape(ImageProcess.reshapeHalf(next_state),(80,80,1))
						if info['ale.lives']<life:
							#reward-=(life-info['ale.lives'])
							life=info['ale.lives']
						agent.setPerception(next_state,action,reward,done,episode)
						if done:
							break
				agent.e_greedy=True
			
	finally:
		pass

if __name__=='__main__':
	main()

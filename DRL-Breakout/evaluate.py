# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:14:22 2018

@author: qxb-810
"""

import gym
import cv2
import numpy as np
from moviepy.editor import VideoClip
from BrainDQN_Nature import BrainDQN
from brick import ImageProcess,ENV_NAME

env=gym.make(ENV_NAME)
agent=BrainDQN(env.action_space.n,print_log=False)

n_episode=10
best=0
done=True
rewards=[]
frames=[]
buffer=[]
fps=25

def make_frame(t):
	global done,env,agent
	if done:
		state=env.reset()
		agent.setInitState(ImageProcess.reshapeHalf(state))
		done=False
	else:
		action=agent.getAction()
		state,reward,done,_=env.step(action)
		agent.setPerception(np.reshape(ImageProcess.reshapeHalf(state),(80,80,1)),action,reward,done,0)
	return state

def make_frame_best(t):
	return buffer.pop(0)

def load_frames():
	for frame in range(frames[best]):
		print("Loading from episode %s : frame %s"%(best,frame))
		buffer.append(cv2.imread('temp/%s-%s.png'%(best,frame)))

def run():
	global best
	for episode in range(n_episode):
		frame=0
		done=False
		state=env.reset()
		agent.setInitState(ImageProcess.reshapeHalf(state))
		rewards.append(0)
		cv2.imwrite('temp/%s-%s.png'%(episode,frame),state)
		while not done:
			action=agent.getAction()
			state,reward,done,_=env.step(action)
			agent.setPerception(np.reshape(ImageProcess.reshapeHalf(state),(80,80,1)),action,reward,done,episode)
			rewards[-1]+=reward
			frame+=1
			cv2.imwrite('temp/%s-%s.png'%(episode,frame),state)
		frames.append(frame)
	best=np.argmax(rewards)
	print('All scores:',rewards)
	print('Best score:',np.max(rewards))

def make_movie_best():
	animation=VideoClip(make_frame_best,duration=len(buffer)/fps)
	animation.write_videofile('evaluation-%s.mp4'%agent.timeStep,fps=fps,codec='mpeg4')

def make_movie():
	animation=VideoClip(make_frame,duration=180)
	animation.write_videofile('evaluation-%s.mp4'%agent.timeStep,fps=fps,codec='mpeg4')

if __name__=='__main__':
#	run()
#	load_frames()
#	make_movie_best()
	make_movie()
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:44:25 2018

@author: qxb-810
"""
from __future__ import print_function
import os
import re
from tqdm import tqdm
from array import array
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

filename_pattern='slurm-[0-9]*.out'
content_pattern='TIMESTEP ([0-9]*) / STATE ([a-z]*) / EPSILON (-?\d+\.?\d*e?-?\d+?) / EPISODE ([0-9]*) / ACTION [0-9] / REWARD (-?[.0-9]*)'

step=array('i')
epsilon=array('d')
reward=array('d')

eval_episode=array('i')
eval_reward=array('d')

print('Scanning directory for log file...')
for f in os.listdir('.'):
	if re.match(filename_pattern,f):
		print('Find:',f)
		with open(f) as file:
			try:
				for line in tqdm(file):
					if re.match(content_pattern,line):
						m=re.match(content_pattern,line)
						episode=int(m.group(4))
						state=m.group(2)
						r=float(m.group(5))
						if state=='explore' or state=='observe' or state=='train':
							if episode==len(step):
								step.append(int(m.group(1)))
								epsilon.append(float(m.group(3)))
								reward.append(0 if r<0 else r)
							else:
								reward[episode]+=(0 if r<0 else r)
						elif state=='evaluate':
							if len(eval_episode)==0 or eval_episode[-1]!=episode:
								eval_episode.append(episode)
								eval_reward.append(0 if r<0 else r)
							else:
								eval_reward[-1]+=(0 if r<0 else r)
							
			except KeyboardInterrupt:
				print('Interrupted, ignoring the following data...')
				pass
			finally:
				with open("episode-reward-%i.txt"%episode,'w') as result_file:
					for e in zip(eval_episode,np.array(eval_reward)/10):
						print('Episode: %i, Average reward: %d'%e,file=result_file)
				plt.figure(1)
				x=range(len(step))
				plt.plot(x,step)
				plt.savefig("episode-step-%i.jpg"%episode)
				plt.figure(2)
				x=range(len(epsilon))
				plt.plot(x,epsilon)
				plt.savefig("episode-epsilon-%i.jpg"%episode)
				plt.figure(3)
				x=range(len(reward))
				plt.scatter(x,reward,marker='.')
				plt.savefig("episode-reward-%i.jpg"%episode)
				plt.figure(4)
				plt.plot(eval_episode,np.array(eval_reward)/10)
				plt.savefig("episode-reward-eval-%i.jpg"%episode)
		
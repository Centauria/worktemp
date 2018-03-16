# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:44:25 2018

@author: qxb-810
"""

import os
import re
from tqdm import tqdm
from array import array
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

filename_pattern='slurm-[0-9]*.out'
content_pattern='TIMESTEP ([0-9]*) / STATE [a-z]* / EPSILON (-?\d+\.?\d*e?-?\d?) / EPISODE ([0-9]*) / REWARD (-?[.0-9]*)'

step=array('i')
epsilon=array('d')
reward=array('d')

print('Scanning directory for log file...')
for f in os.listdir('.'):
	if re.match(filename_pattern,f):
		print('Find:',f)
		with open(f) as file:
			try:
				for line in tqdm(file):
					if re.match(content_pattern,line):
						m=re.match(content_pattern,line)
						episode=int(m.group(3))
						if episode==len(step):
							step.append(int(m.group(1)))
							epsilon.append(float(m.group(2)))
							reward.append(float(m.group(4)))
						else:
							reward[episode]+=float(m.group(4))
			except KeyboardInterrupt:
				print('Interrupted, ignoring the following data...')
				pass
			finally:
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
		
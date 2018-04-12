# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:00:22 2018

@author: qxb-810
"""

import tensorflow as tf

class BrainLSTM:
	
	def __init__(self):
#		init replay memory
		
#		init parameters
		
#		init network
		self._build_net()
#		saving and loading network
		
	def _build_net(self):
		self.sentenceInput=tf.placeholder(tf.string,[None],name='sentence')
		self.reward=tf.placeholder(tf.float32,[1],name='reward')
		with tf.variable_scope('eval-net'):
			# c_names(collections_names) are the collections to store variables
			c_names,w_initializer,b_initializer=\
				['eval-net params',tf.GraphKeys.GLOBAL_VARIABLES],\
				tf.truncated_normal_initializer(stddev = 0.01),\
				tf.constant_initializer(0.1)
			
			# first layer. collections is used later when assign to target net
			
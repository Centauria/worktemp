# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:32:41 2018

@author: qxb-810
"""

import moviepy
from moviepy.editor import VideoClip
import numpy as np
import pygame as pg
import threading

HEIGHT=60
WIDTH=200

pg.init()
screen = pg.display.set_mode((WIDTH,HEIGHT),0,32)
pg.display.set_caption('23333')
running=True
p_tiny=pg.image.load('pygame_tiny.png').convert()
clock=pg.time.Clock()

def make_frame(t):
	res=pg.surfarray.array3d(pg.display.get_surface())
	return res.transpose((1,0,2))

def make_movie():
	animation=VideoClip(make_frame,duration=1)
	animation.write_videofile('233.mp4',fps=30,codec='mpeg4')
	pass

x,y=0.0,0.0
r=20
while running:
	for event in pg.event.get():
		if event.type==pg.QUIT:
			running=False
			break
		screen.blit(p_tiny,(0,0))
		pg.draw.circle(screen,(255,255,0),\
				 (int(x),int(y)),r)
		x+=2*clock.tick()
		x%=WIDTH
		y+=3*clock.tick()
		y%=HEIGHT
		pg.display.update()
		pass
	pass

pg.quit()
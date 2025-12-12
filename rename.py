from math import *
import pygame# python 3.7.1, pygame 1.9.4
from pygame.locals import *
import pickle
pygame.init()
import os


for i in os.listdir('image/boss'):
	name=i
	if i[5:8]=='wlk':

	
		name=i
		print(name,i[:-6]+i[-5:])
		

		im=pygame.image.load('image/boss/'+i)
		pygame.image.save(im,'image/test_blend/'+i[:-6]+i[-5:])
		
		

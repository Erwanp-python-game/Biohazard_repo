from math import *
import pygame# python 3.7.1, pygame 1.9.4
from pygame.locals import *
import pickle
pygame.init()
import os

M=[]
for j,i in enumerate(os.listdir('image/gun')):

	if j%4==0 :
		DICT={}
	DICT[j%4]=i
	if j%4==3:
		M.append(DICT)

f = open("image/gunD", "wb")
pickle.dump(M, f)
f.close()	

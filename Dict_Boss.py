from math import *
import pygame# python 3.7.1, pygame 1.9.4
from pygame.locals import *
import pickle
pygame.init()
import os

M=[]
DICT={}
ind=-1
A=[]
for j,i in enumerate(os.listdir('image/Boss')):
	if i[4]!=ind:
		ind=i[4]
		if len(A)>0:
			M.append(A)
		A=[]
	
	if i[5:8]=='wlk':
		if i[10]==str(0):
			if DICT!={}:
				A.append(DICT.copy())
			DICT={}
		DICT[int(i[10])*45.]=i
A.append(DICT.copy())
M.append(A)
print(M)

f = open("image/BossD", "wb")
pickle.dump(M, f)
f.close()	

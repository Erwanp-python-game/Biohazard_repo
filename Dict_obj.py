from math import *
import pygame# python 3.7.1, pygame 1.9.4
from pygame.locals import *
import pickle
pygame.init()
import os

M=[]
K=os.listdir('image/objets')
Y=[int(i[1:-6]) for i in K]
K=[x for _, x in sorted(zip(Y, K))]
print(K)
for j,i in enumerate(K):
	print(i[1:-6])
	if j%8==0:
		DICT={}
	DICT[45*(j%8)]=i
	if j%8==7:
		M.append(DICT)

f = open("image/obj_D", "wb")
pickle.dump(M, f)
f.close()	

DICT={}
for j,i in enumerate(os.listdir('image/obj_destr')):
	if int(i[1:-10])-1 not in DICT.keys():
		DICT[int(i[1:-10])-1]=[]
	DICT[int(i[1:-10])-1].append(i)
f = open("image/obj_destr_D", "wb")
pickle.dump(DICT, f)
f.close()	

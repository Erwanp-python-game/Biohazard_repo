from math import *
import pygame# python 3.7.1, pygame 1.9.4
from pygame.locals import *
import pickle
pygame.init()
import os

M=[]
Mat_hit=[]
for j,i in enumerate(os.listdir('image/monsters')):
	print(1+j//40,j%40,i)
	if j%40==0:
		A=[]
		DICT_at_hit={}
	if j%40==8 :
		DICT={}
	if j%40>=8:
		angle=(j%8)*45
		DICT[angle]=i
	else:
		DICT_at_hit[j%40]=i
	if j%8==7 and j%40>=8:
		print(DICT)
		A.append(DICT.copy())

	if j%40==39:
		print(A)
		M.append(A)
		Mat_hit.append(DICT_at_hit)
print(M,Mat_hit)

f = open("image/monstD", "wb")
pickle.dump(M, f)
pickle.dump(Mat_hit, f)
f.close()	

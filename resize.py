from math import *
import pygame# python 3.7.1, pygame 1.9.4
from pygame.locals import *
import pickle
pygame.init()
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data


for j,i in enumerate(os.listdir('image/test_blend')):

	print(i)
	image=pygame.image.load('image/test_blend/'+i)
	#if i[2]=='h' or i[2]=='a':#image.get_size()[0]!=160:
	print('done')
	imageT=pygame.transform.scale(image,((160*image.get_size()[0])//image.get_size()[1],160))
	imageT2=imageT.subsurface(((imageT.get_size()[0]-160)//2,0,160,160))
	#imageT2=pygame.Surface((160, 160))
	#imageT2.blit(image,(0,20))
	pygame.image.save(imageT2,'image/test_blend/'+i)
	
for j,i in enumerate(os.listdir('image/test_blend')):
	Im=load_image('image/test_blend/'+i)
	Im=np.where(np.expand_dims(np.sum(Im[:,:,:-1],axis=-1),-1)==3,np.array([0,0,0,255]),Im).astype(np.uint8)
	Im = Image.fromarray(Im, 'RGBA')
	Im.save('image/test_blend/'+i)

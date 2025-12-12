import numpy as np
from math import *
import matplotlib.pyplot as plt
import time
import pygame# python 3.7.1, pygame 1.9.4
from pygame.locals import *
import pickle
pygame.init()
import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from random import random,randint
from PATH import astar,Node
import ast
import datetime as dt
from Boss import BOSS,LIZARD,nearest_valid,rot_z,rot_y,rot_plan,light_modif


# def rot_z(theta):
	# return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
# def rot_y(theta):
	# return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])

# def rot_plan(theta):
	# return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
activatedT=[]
lifts=[]
scrnL=np.array([80,40])
screen=np.full((*scrnL,6),0.0)
depth=np.full((2*scrnL[0],2*scrnL[1],1),100.0)
POS=np.full((2*scrnL[0],2*scrnL[1]),1000.0)
horizon=np.full((scrnL[0],1),100.0)
Ang=np.full((*scrnL,2),0.0)
I=np.indices(scrnL)
I_n=np.divide(np.moveaxis((np.indices(scrnL)),0,2)-0.5*scrnL,scrnL[1])
screen[:,:,1:3]=I_n
Ang=I_n
Ang[:,:,0]=Ang[:,:,0]*pi/4
Ang[:,:,1]=Ang[:,:,1]*atan(1/2)*2

screen[:,:,0]=screen[:,:,0]

screen[:,:,4]=np.sin(Ang[:,:,0])
screen[:,:,5]=np.cos(-Ang[:,:,1]+pi*0.5)
screen[:,:,3]=1

TAN1=np.amax(screen[:,:,5])
TAN2=np.amax(screen[:,:,4])

screenV=screen[:,:,3:]
screenP=screen[:,:,:3]
CENTER=np.expand_dims(np.linalg.norm(screen[:,:,:3]-[0,0,0],axis=-1).repeat(2,axis=0).repeat(2,axis=1),-1)
TORCHE=np.expand_dims((np.maximum(np.cos(I_n[:,:,0]*pi/2)*np.cos(I_n[:,:,1]*2*pi/2),0)).repeat(2,axis=0).repeat(2,axis=1),-1)
torch_on=0
	
R_c=np.array([-1,0,0])
Vg=np.array([1,-sqrt(2)/2])/sqrt(3/2)
Vd=np.array([1,sqrt(2)/2])/sqrt(3/2)

levelD={}
#example : levelD[n]={'deco':[2],'obj':[0,0,0,0,0,0,0,0,0,2,0,0,5],'RA':[5,5,5,5,5,5,5,5,5,5,5,5,5],'RAmon':[5,5,5,5,5,5,5,5,5,5],'mon':[0,0,0,0,0,0,0,0,0,0],'wall':[0,0,0,0,0,0,0,0,0,0],'flat':[0,0,0,0,0,0,0,0,0,9],'door':[0,0,0,0,0,0,0,0,0,0]}
levelD[0]={'deco':[2],'obj':[6,7,2,3,4,5,0,0,0,2,0,0,5],'RA':[7,6,5,5,5,5,5,5,5,5,5,5,5],'RAmon':[5,5,5,5,5,5,5,5,5,5],'mon':[0,1,0,0,0,0,0,0,0,0],'wall':[8,9,0,0,0,0,0,0,0,0],'flat':[0,2,0,0,0,0,0,0,0,9],'door':[0,2,0,0,0,0,0,0,0,0]}
levelD[1]={'deco':[3,4,5],'obj':[4,1,0,8,9,10,11,12,13,2,0,0,5],'RA':[4,5,5,5,5,5,5,5,5,5,5,5,5],'RAmon':[5,8,5,5,5,5,5,5,5,5],'mon':[0,2,0,0,0,0,0,0,0,0],'wall':[10,11,13,12,14,17,0,0,15,16],'flat':[0,3,4,4,0,0,0,0,0,9],'door':[3,3,3,3,3,3,3,3,6,5]}
levelD[2]={'deco':[6,3,4,7,4,8,9,10],'obj':[1,11,13,14,15,0,0,0,0,2,0,0,5],'RA':[5,5,5,5,5,5,5,5,5,5,5,5,5],'RAmon':[5,8,5,5,5,5,5,5,5,5],'mon':[0,2,1,0,0,0,0,0,0,0],'wall':[0,1,2,8,12,16,15,20,19,0,17],'flat':[5,0,0,0,0,0,0,0,0,9],'door':[7,7,7,7,7,7,7,7,9,8]}
levelD[3]={'deco':[11,12,13,14,15,16,17],'obj':[1,16,0,0,0,0,0,0,0,2,0,0,5],'RA':[5,7,5,5,5,5,5,5,5,5,5,5,5],'RAmon':[5,8,5,5,5,5,5,5,5,5],'mon':[0,2,1,0,0,0,0,0,0,0],'wall':[21,24,25,26,27,28,22,23,29,29],'flat':[11,6,7,0,0,0,0,0,0,9],'door':[3,4,4,4,4,4,4,4,4,4]}
levelD[99]={'deco':[1],'obj':[0,1,2,3,4,5,0,0,0,2,0,0,5],'RA':[5,5,5,5,5,5,5,5,5,5,5,5,5],'RAmon':[5,5,5,5,5,5,5,5,5,5],'mon':[0,1,0,0,0,0,0,0,0,0],'wall':[0,2,3,1,0,5,7,0,0,0],'flat':[0,1,0,0,0,0,0,0,0,9],'door':[0,1,0,0,0,0,0,0,0,0]}
destr=[0,4,6]
level=0
level_nameL=['Level 0: Training','Level 1: The Lab','Level 2: The Storage','Level 0: Training']
level_arme=[1,2,2,2]
level_end=[5,7,8,10]
level_start=[1,2,1,1]
for i in range(100-len(level_arme)):
	level_arme.append(4)
	level_end.append(0)
	level_start.append(100)
	level_nameL.append('no')
level_end[99]=10
level_start[99]=4


def source_pos(code):
	x=(int(code.split(',')[0])-50)*2
	y=(int(code.split(',')[1])-50)*2
	z=zmap[int(code.split(',')[1])][int(code.split(',')[0])]-hmap[int(code.split(',')[0])][int(code.split(',')[1])]
	return np.array([x,y,z-5])



# def nearest_valid(autho,x):
	# d=1000
	# xq=[x[0]/2+50,x[1]/2+50]
	# xR=[int(xq[0]),int(xq[1])]
	# xfin=xR
	# #print(autho[xR[1]-2:xR[1]-2+2][xR[0]-2:xR[0]+2])
	# for i in [-2,-1,0,1,2]:
		# for j in [-2,-1,0,1,2]:
			# if (i!=0 or j!=0) and autho[xR[1]+j][xR[0]+i]==2:
				# D=(xR[0]+i-xq[0])**2+(xR[1]+j-xq[1])**2
				# if D<=d:
					# d=D
					# xfin=[xR[0]+i,xR[1]+j]
					# #print(autho[xR[1]+j][xR[0]+i])
	# return xfin




class Wall():
	def __init__(self,u,v,w,text,door,deco,freq,phase):
		self.a=np.full((scrnL[0],scrnL[1],3),u)
		self.b=np.full((scrnL[0],scrnL[1],3),v)
		self.X=np.full((scrnL[0],scrnL[1],3),w)
		self.door=door
		self.closed=1
		self.n=np.cross(self.a[0][0],self.b[0][0])
		N=np.stack((self.a[0][0],self.b[0][0],-self.n),axis=-1)
		V=np.maximum(np.minimum(np.linalg.solve(N,-self.X[0][0]+R_c)[:-1],1),0)
		self.norm=np.linalg.norm(self.X[0][0][:-1]+V[0]*self.a[0][0][:-1]+V[1]*self.b[0][0][:-1]-R_c[:-1])
		self.normf=self.norm
		self.norm3=np.linalg.norm(self.X[0][0]+V[0]*self.a[0][0]+V[1]*self.b[0][0]-R_c)
		self.text=text
		Imdeco=pygame.image.load(self.text)
		if deco!=0:
			Imdeco=pygame.transform.scale(Imdeco,(120*freq,120))
			for i in range(freq):
				Imdeco.blit(pygame.image.load(text),(120*i,0))
			Imdeco.blit(pygame.image.load('image/deco/'+str(levelD[level]['deco'][deco-1])+'.png'),(120*(freq-1-phase),0))
		if door>1 and door<5:
			verrou=pygame.image.load('image/deco/verrou.png')
			if door==2:
				colo=(255,50,50)
			if door==3:
				colo=(50,255,50)
			if door==4:
				colo=(50,50,255)
			verrou.fill(colo,special_flags=BLEND_RGB_MULT)
			Imdeco.blit(verrou,(0,0))
		
		IM=np.transpose(pygame.surfarray.pixels3d(Imdeco),(1,0,2))
		#print(np.sum(((IM-np.array([0,255,255]))==0).all(-1)),np.sum(((IM-np.array([255,0,255]))==0).all(-1)),text,((IM-np.array([255,0,255]))==0).all(-1).shape)
		IM=np.where(np.expand_dims(((IM-np.array([255,0,255]))==0).all(-1),-1),-1,IM)
		IM=np.where(np.expand_dims(((IM-np.array([0,255,255]))==0).all(-1),-1),-2,IM)
		self.window=np.sum(IM<=-1)
		self.wall_im=[np.flip(np.minimum(IM+1,255),(0,1))]
		self.phase=0
		#print(self.text[11:-3])
		if self.text[11:-3]in['floor7.','floor10.']:
			self.phase=1
		files=[filename for filename in os.listdir(self.text[:11]) if filename.startswith(self.text[11:-3])]
		files_d=[filename for filename in os.listdir('image/deco') if filename.startswith(str(levelD[level]['deco'][deco-1])+'.')]
		if (len(files)>1) or (len(files_d)>1):
			for k in range(max(len(files)-1,len(files_d)-1)):
				if (len(files)>1):
					Im_t=pygame.image.load(self.text[:-4]+'.'+str(min(k,len(files)-2)+1)+'.png')
				else:
					Im_t=pygame.image.load(text)
				Imdeco=Im_t
				if deco!=0:
					Imdeco=pygame.transform.scale(Imdeco,(120*freq,120))
					for i in range(freq):
						Imdeco.blit(Im_t,(120*i,0))
					deco_name='image/deco/'+str(levelD[level]['deco'][deco-1])+'.png'
					if (len(files_d)>1):
						deco_name='image/deco/'+str(levelD[level]['deco'][deco-1])+'.'+str(min(k,len(files_d)-2)+1)+'.png'
					Imdeco.blit(pygame.image.load(deco_name),(120*(freq-1-phase),0))
				if door>1 and door<5:
					verrou=pygame.image.load('image/deco/verrou.png')
					if door==2:
						colo=(255,50,50)
					if door==3:
						colo=(50,255,50)
					if door==4:
						colo=(50,50,255)
					verrou.fill(colo,special_flags=BLEND_RGB_MULT)
					Imdeco.blit(verrou,(0,0))
				
				IM=np.transpose(pygame.surfarray.pixels3d(Imdeco),(1,0,2))
				#print(np.sum(((IM-np.array([0,255,255]))==0).all(-1)),np.sum(((IM-np.array([255,0,255]))==0).all(-1)),text,((IM-np.array([255,0,255]))==0).all(-1).shape)
				IM=np.where(np.expand_dims(((IM-np.array([255,0,255]))==0).all(-1),-1),-1,IM)
				IM=np.where(np.expand_dims(((IM-np.array([0,255,255]))==0).all(-1),-1),-2,IM)
				self.window=np.sum(IM<=-1)
				self.wall_im.append(np.flip(np.minimum(IM+1,255),(0,1)))
		
		
		
		self.U=np.array([0])
		self.z=w[-1]
		self.ID=str(int((self.X[0][0][0]+0.5*self.b[0][0][0])//2+50))+','+str(int((self.X[0][0][1]+0.5*self.b[0][0][1])//2+50))
		self.transp=0
		if self.a[0][0][-1]==0 or self.b[0][0][-1]==0:
			self.ID=str(int((self.X[0][0][0]+0.5*self.b[0][0][0]+0.5*self.a[0][0][0])//2+50))+','+str(int((self.X[0][0][1]+0.5*self.b[0][0][1]+0.5*self.a[0][0][1])//2+50))
		self.TTT0=[]
		self.TTT1=[]
		self.TTT2=[]
		self.to_device()
	def opendoor(self,door):
		Imdeco=pygame.image.load(self.text)
		verrou=pygame.image.load('image/deco/verrouO.png')
		if door==2:
			colo=(255,50,50)
		if door==3:
			colo=(50,255,50)
		if door==4:
			colo=(50,50,255)
		verrou.fill(colo,special_flags=BLEND_RGB_MULT)
		Imdeco.blit(verrou,(0,0))
		IM=np.transpose(pygame.surfarray.pixels3d(Imdeco),(1,0,2))
		IM=np.where(np.expand_dims(((IM-np.array([255,0,255]))==0).all(-1),-1),-1,IM)
		IM=np.where(np.expand_dims(((IM-np.array([0,255,255]))==0).all(-1),-1),-2,IM)
		self.window=np.sum(IM<=-1)
		self.wall_im[0]=np.flip(np.minimum(IM+1,255),(0,1))
		return 0
	
	def texture(self,sc1,sc2):
		
		self.borne=[self.wall_im[0].shape[0]-1,self.wall_im[0].shape[1]-1]
		if self.door!=0 :
			self.format=120*np.array([1,1])
		else:
			self.format=120*np.array([np.linalg.norm(self.a[0][0])/10,np.linalg.norm(self.b[0][0])/10])
		if self.text=='image/wall/wall29.png':
			self.format=120*np.array([1,np.linalg.norm(self.b[0][0])/30])
		

	def calc_norm(self):
		N=np.stack((self.a[0][0],self.b[0][0],-self.n),axis=-1)
		V=np.maximum(np.minimum(np.linalg.solve(N,-self.X[0][0]+R_c)[:-1],1),0)
		self.norm=np.linalg.norm(self.X[0][0][:-1]+V[0]*self.a[0][0][:-1]+V[1]*self.b[0][0][:-1]-R_c[:-1])-min(self.window,1)*0.001
		self.norm3=np.linalg.norm(self.X[0][0]+V[0]*self.a[0][0]+V[1]*self.b[0][0]-R_c)
	def calc_normfast(self):
		self.normf=np.linalg.norm(self.X[0][0][:-1]+0.5*self.a[0][0][:-1]+0.5*self.b[0][0][:-1]-R_c[:-1])
		
	def normal(self,trans):
		
		t=-trans
		No=self.n[:-1]
		No=No/np.linalg.norm(No)
		return (-t-abs(np.dot(-t,No))*No)
	
	def test_behind(self):
		
		if self.door==0:
			if abs(np.arctan2(self.n[1],self.n[0])+ang[0])<np.arctan(TAN2):
				return False
		
		Mg=np.stack((self.b[0][0][0:-1],-Vg@Rp),axis=-1)
		self.Ig=np.linalg.solve(Mg,x-self.X[0][0][0:-1])
		
		Md=np.stack((self.b[0][0][0:-1],-Vd@Rp),axis=-1)
		self.Id=np.linalg.solve(Md,x-self.X[0][0][0:-1])
		
		Rp0=rot_plan(-ang[0])
		Xa=(self.X[0][0][0:-1]-x)@Rp0
		Xb=(self.X[0][0][0:-1]+self.b[0][0][0:-1]-x)@Rp0
		

		

		if ((Xa[0]>1 and Xb[0]>1) or(self.Id[0]<1 and self.Id[0]>0 and self.Id[1]>1) or (self.Ig[0]<1 and self.Ig[0]>0 and self.Ig[1]>1))or self.door!=0:

			if self.door!=0 and self.closed==0 or self.window>0:
				return True
			else:
				
				global horizon

				B0=self.b[:,0,:-1]
				X0=self.X[:,0,:-1]#+5*np.random.random(self.X[:,0,:-1].shape)
				self.M0=np.stack((B0,-screen[:,24,3:-1]@Rp),axis=-1)# -------------- PB à résoudre pièce haute
				self.A0=-X0+(screen[:,24,:2]-x)@Rp+x
				self.S0=(np.linalg.solve(self.M0,self.A0))
				self.U0=np.where(np.all(self.S0<=horizon,axis=-1)&np.all(self.S0>0,axis=-1)&np.all(self.S0[:,:-1]<1,axis=-1),1,0)
				horizon=self.S0[:,-1:]*np.expand_dims(self.U0,-1)+horizon*(1-np.expand_dims(self.U0,-1))
				
				if np.sum(self.U0)>0:
					return True
				else:
					return False
					
		else:
			#print('oh')
			return False
		
	def to_device(self):
		self.a0=torch.tensor(self.a*1.01).to('cuda')
		self.b0=torch.tensor(self.b*1.01).to('cuda')
		self.X0=torch.tensor(self.X).to('cuda')

	def render(self):
		global depth,POS,Light,Sky_view

		self.M=torch.stack((self.a0,self.b0,-torch.tensor(screenV).to('cuda')),axis=-1)
		self.B=-self.X0+torch.tensor(screenP).to('cuda')
		
		self.S=(torch.linalg.solve(self.M,self.B))
		
		# self.M=torch.tensor(np.stack((self.a[:,:,:]*1.01,self.b[:,:,:]*1.01,-screenV[:,:,:]),dims=-1))
		# self.B=torch.tensor(-self.X[:,:,:]+screenP[:,:,:])
		# self.S=(torch.linalg.solve(self.M.to('cuda'),self.B.to('cuda'))).to('cpu').numpy()
		
		for i in range(1):
			self.S=self.S.repeat_interleave(2,dim=0).repeat_interleave(2,dim=1)
			self.S[:,:,:-1]=(torch.roll(self.S[:,:,:-1],1,dims=0)+torch.roll(self.S[:,:,:-1],1,dims=1)+torch.roll(self.S[:,:,:-1],-1,dims=1)+torch.roll(self.S[:,:,:-1],-1,dims=0))/4
			AbsS=torch.absolute(self.S[:,:,-1])
			self.S[:,:,-1]=(torch.sign(self.S[:,:,-1])*(torch.roll(AbsS,1,dims=0)+torch.roll(AbsS,1,dims=1)+torch.roll(AbsS,-1,dims=1)+torch.roll(AbsS,-1,dims=0))/4)

			
		self.U=torch.all(self.S<=torch.tensor(depth).to('cuda'),dim=-1)&torch.all(self.S>0,dim=-1)&torch.all(self.S[:,:,:-1]<1,dim=-1)
		self.U=self.U.to('cpu')
		self.S=self.S.to('cpu')
		
		
		if torch.sum(self.U)<5:
			return np.full((2*scrnL[0],2*scrnL[1],1),0)
		else:
			Ue=torch.unsqueeze(self.U,-1)
			if self.transp:
				Sky_view=1
				return -1*torch.dstack((Ue,Ue,Ue)).numpy()
				
			
			

			
			self.G=torch.remainder(torch.maximum(((1-self.S[:,:,:-1])*self.format),torch.tensor(0))+int(self.phase*c3*0.5),torch.tensor(self.borne)).long()
			colorL=[1,1,1]
			if self.ID in light_color.keys():
				colorL=np.round(np.maximum(np.array(light_color[self.ID]),0.1),2)
			

			colorL=light_modif(colorL,level,c3)
			
			
			ind=c//(12//len(self.wall_im))
			#print(self.G.T.shape,self.wall_im[ind].shape)
			
			Ar=torch.moveaxis(torch.tensor(self.wall_im[ind].copy())[self.G.T[0,:,:],self.G.T[1,:,:],:],1,0)*colorL*torch.dstack((self.U,self.U,self.U))
			Ar=Ar.numpy()
			
			filt=1-np.expand_dims((Ar==0).all(2),-1)
			#if self.text[11:-3]!='floor7.':
			depth=(self.S[:,:,-1:]*Ue*(filt)).numpy()+depth*(1-Ue*(filt)).numpy()
			

			Xl=((np.expand_dims(self.S[::2,::2,0],-1)*self.a+np.expand_dims(self.S[::2,::2,1],-1)*self.b+self.X).repeat(2,axis=0).repeat(2,axis=1))*Ue.numpy()
			self.U=self.U.numpy()
			POS=POS*(1-self.U)
			D=np.full((2*scrnL[0],2*scrnL[1]),1000.0)
			if self.ID in light_wall.keys():
				Y0=[np.linalg.norm(source_pos(i)-R_c) for i in light_wall[self.ID]]
				X0=[x for _,x in sorted(zip(Y0,light_wall[self.ID]))]
				for i in X0[:min(len(X0),4)]:
					Xsource=source_pos(i)
					D=np.minimum(D,np.linalg.norm(Xl-Xsource,axis=-1)*self.U)
				POS=POS+D
				Ar=Ar*level_light
			else:
				D=np.minimum(D,(np.linalg.norm(Xl-R_c,axis=-1)**0.5)*self.U)
				POS=POS+D
				Ar=Ar*torch_on*TORCHE**3

			return Ar

Xthing,Ything=np.indices((2*scrnL[0],2*scrnL[1]))
RA=5

Boule=[]
	

class Thing():
	def __init__(self,x0,type_M,vivant,group):
		self.x0=x0
		self.norm=np.linalg.norm(self.x0-x+[1,0])
		self.width=2*scrnL[0]/self.norm
		self.widthY=2*scrnL[0]/self.norm
		self.DX=0
		self.DY=0
		self.type_M=levelD[level]['mon'][type_M]
		self.angle=0
		self.orient=2*pi*random()
		self.im=np.minimum(pygame.surfarray.pixels3d(MD[self.type_M][1][45*(((-ang[0]+pi/8+self.orient)//(pi/4))%8)]),255)
		self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)
		self.borne=[self.im.shape[0]-1,self.im.shape[1]-1]
		self.z=zmap[int(self.x0[1]+100)//2][int(self.x0[0]+100)//2]
		self.active=0
		self.inline=0
		self.vie=M_charac['life'][self.type_M]
		self.track=[]
		self.mort=0
		self.vivant=vivant
		self.range=M_charac['dist'][self.type_M]
		self.attack_range=False
		self.group=group
		self.f0=(self.x0-x)@rot_plan(-ang[0])
		self.U=0
		self.RA=levelD[level]['RAmon'][type_M]*(1+(random()*0.2-0.1))
	def calc_norm(self):

		self.f0=(self.x0-x)@rot_plan(-ang[0])
		self.norm=np.linalg.norm(self.x0-x)
		self.width=self.RA*2*scrnL[0]/self.f0[0]
		self.widthY=self.RA*2*scrnL[0]/self.f0[0]
		self.DX=-self.RA*scrnL[0]/self.f0[0]+scrnL[0]*(self.f0[1]/self.f0[0])/TAN2
		self.DY=-(self.RA*scrnL[0]*1.7-5*scrnL[0])/self.f0[0]-scrnL[1]*tan(ang[1])/TAN1-1*scrnL[1]*self.RA*(z-self.z)/self.f0[0]
		A=45*(((atan((self.f0[1]/self.f0[0]))+self.orient-pi/2-ang[0]+pi/8)//(pi/4))%8)
		if A!=self.angle:
			self.angle=A
			self.im=np.minimum(pygame.surfarray.pixels3d(MD[self.type_M][c//3][self.angle]),255)
			self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)
	def walk(self):
		self.im=np.minimum(pygame.surfarray.pixels3d(MD[self.type_M][c//3][self.angle]),255)
		self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)

				
		
		
	def test_behind(self):
		if self.f0[0]>0 and abs(self.f0[1]/self.f0[0])<TAN2+0.5 :
			if self.norm<=horizon[min(max(int(self.width//2+self.DX+scrnL[0])//2,0),scrnL[0]-1)]:
				return True
			else:
				return False
		else:
			return False		
		
		
	def move(self):
		global VIE,HIT
		self.attack_range=self.norm<=5
		if self.range==1:
			self.attack_range=(self.test_behind())&(self.norm<=20)
		
		
		if not(self.attack_range) and self.active and self.vie>0:
			
			X0=nearest_valid(authorized_map,x)
			if len(self.track)>0:
				if ((X0[0]-self.track[-1][1])**2+(X0[1]-self.track[-1][0])**2)**0.5>0:
					self.track=astar(authorized_map,(int(self.x0[1]+101)//2,int(self.x0[0]+101)//2),(X0[1],X0[0]))
			if len(self.track)>2:
				dest=[self.track[2][1],self.track[2][0]]
				self.orient=-np.angle((dest[1]-int(self.x0[1]+101)//2)*1j+(1e-3+dest[0]-int(self.x0[0]+101)//2))-pi/2+(random()-0.5)*pi/5
				if np.linalg.norm(np.array(dest)-np.array([(self.x0[0]+101)/2,(self.x0[1]+101)/2]))<sqrt(2)+0.2:
					self.track.pop(0)
				V=[+0.1+M_charac['speed'][self.type_M]*self.active,0]@rot_plan(self.orient+pi/2)
				self.x0=self.x0+V
			else:
				self.active=0
		if not(self.attack_range) and self.active==0 and self.vie>0:
			V=[-0.1,0]@rot_plan(self.orient-pi/2)
			if level_map[int((self.x0+7*V)[1]+101)//2][int((self.x0+7*V)[0]+101)//2] and authorized_map[int((self.x0)[1]+101)//2][int((self.x0)[0]+101)//2]==2 and self.active==0:
				self.orient=self.orient+pi
			self.x0=self.x0+V
		
		if self.norm>50  and self.active and self.vie>0:
			self.active=0
			self.orient=random()*pi
		self.z=zmap[int(self.x0[1]+100)//2][int(self.x0[0]+100)//2]	
		
		if self.norm<20+min(shoot,1)*20*min(1,arme) and self.active==0 and self.vie>0:
			self.active=1
			s=pygame.mixer.Sound("son/grognespot%s.ogg"%(self.type_M+1))
			s.play()
			X0=nearest_valid(authorized_map,x)
			self.track=astar(authorized_map,(int(self.x0[1]+101)//2,int(self.x0[0]+101)//2),(X0[1],X0[0]))
			
		if self.attack_range and self.active and self.vie>0:	
			if self.range==0:
				if c//4==2 and c%3==0:
					HIT=1
					if arme!=0:
						Accuracy[0]=Accuracy[0]+1
					VIE-=M_charac['degat'][self.type_M]
					s=pygame.mixer.Sound("son/aie.ogg")
					s.play()
			else:
				if c2//8==2 and c2%6==0:
					A00=atan((self.x0[1]-x[1])/(self.x0[0]-x[0]))
					if self.x0[0]-x[0]>0:
						A00+=pi
					A11=atan((self.z-z)/abs(self.x0[0]-x[0]))
					#print(self.z,z)
					Boule.append(boule(self.x0[0],self.x0[1],self.z,A11+pi/2,A00,0.6,0,'image/effects/hurt.png'))
			
		

		
	def render(self):
		global Killed_E
		if self.attack_range and self.active and self.vie>0:
			if self.range==0:
				self.im=np.minimum(pygame.surfarray.pixels3d(MDa[self.type_M][c//4]),255)
			else:
				self.im=np.minimum(pygame.surfarray.pixels3d(MDa[self.type_M][c2//8]),255)
			
			self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)


		if ((self.inline and shoot==1 and (self.attack_range or arme!=0)) or(np.linalg.norm(self.x0-explo_pt)<20 and explo==1)) and self.vie>0:
			self.im=np.minimum(pygame.surfarray.pixels3d(MDa[self.type_M][4]),255)
			self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)
			if shoot:
				self.vie-=DEGAT[arme]
			if explo:
				self.vie-=50
			for i in range(randint(5,10)):
				Boule.append(boule(self.x0[0],self.x0[1],self.z,pi*random(),2*pi*random(),0.1*random(),2,'image/effects/vert.png'))
			if self.active==0:
				self.active=1
				X0=nearest_valid(authorized_map,x)
				self.track=astar(authorized_map,(int(self.x0[1]+101)//2,int(self.x0[0]+101)//2),(X0[1],X0[0]))
				s=pygame.mixer.Sound("son/grognespot%s.ogg"%(self.type_M+1))
				s.play()


		
		if self.vie<=0:
			self.im=np.minimum(pygame.surfarray.pixels3d(MDa[self.type_M][4+int(self.mort)]),255)
			self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)
			if self.mort<4:
				self.mort=min(self.mort+0.5,3)
			if self.mort==0.5:
				s=pygame.mixer.Sound("son/grognemort%s.ogg"%(self.type_M+1))
				Killed_E[0]=Killed_E[0]+1
				s.play()
		
		
		
		colorT=light_array[int(self.x0[0]+101)//2][int(self.x0[1]+101)//2]
		if light_array[int(self.x0[0]+101)//2][int(self.x0[1]+101)//2].sum()==0:
			colorT=np.array([1,1,1.])
		colorT=light_modif(colorT,level,c3)	
		
		SQUARE=np.all(self.norm<=depth,axis=-1) & (Xthing<=self.width+self.DX+scrnL[0]) & (Xthing>=self.DX+scrnL[0])& (Ything<=self.widthY+self.DY+scrnL[1]) & (Ything>=self.DY+scrnL[1])
		
		self.U=np.stack(((Xthing-self.DX-scrnL[0])/self.width,(Ything-self.DY-scrnL[1])/self.widthY),-1)*np.expand_dims(SQUARE,-1)
		self.G=np.maximum((self.U*160).astype(int),0)
		Ar=np.moveaxis(self.im[tuple(map(tuple,self.G.T))]*colorT,1,0)
		if light_array[int(self.x0[0]+101)//2][int(self.x0[1]+101)//2].sum()==0:
			Ar=Ar*torch_on*TORCHE**3
			Ar=np.minimum(np.divide(Ar,0.1*self.norm**0.5),255)
		else:
			Ar=Ar*level_light
		self.Ut=np.moveaxis(self.vis[tuple(map(tuple,self.G.T))],1,0)
		self.inline=min(np.sum(self.Ut[scrnL[0]-10:scrnL[0]+10,3*scrnL[1]//2-10:3*scrnL[1]//2+10]),1)
		return Ar
		
class Object():
	def __init__(self,x0,type_M,vivant,group):
		self.RA=levelD[level]['RA'][type_M]#*(1+0.2*(random()-0.5))
		self.x0=x0
		self.norm=np.linalg.norm(self.x0-x+[1,0])
		self.width=2*scrnL[0]/self.norm
		self.widthY=2*scrnL[0]/self.norm
		self.DX=0
		self.DY=0
		self.type_M=levelD[level]['obj'][type_M]
		self.angle=0
		self.orient=2*pi*random()
		if self.type_M==4:
			self.orient=pi/2
			self.angle=pi/2
		if self.type_M==10:
			R=1
			self.orient=R*pi/2
			self.angle=R*pi/2
		self.im=np.minimum(pygame.surfarray.pixels3d(MO[self.type_M][45*(((-ang[0]+pi/8+self.orient)//(pi/4))%8)]),255)
		self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)
		self.borne=[self.im.shape[0]-1,self.im.shape[1]-1]
		self.z=zmap[int(self.x0[1]+100)//2][int(self.x0[0]+100)//2]
		self.active=0
		self.inline=0
		self.vie=1#M_charac['life'][self.type_M]
		self.track=[]
		self.mort=0
		self.vivant=vivant
		self.color=0
		self.group=group
		self.U=0
	def calc_norm(self):
		global VIE,AMMO,Picked_O
		self.f0=(self.x0-x)@rot_plan(-ang[0])
		self.norm=np.linalg.norm(self.x0-x)
		self.width=self.RA*2*scrnL[0]/self.f0[0]
		self.widthY=self.RA*2*scrnL[0]/self.f0[0]
		self.DX=-self.RA*scrnL[0]/self.f0[0]+scrnL[0]*(self.f0[1]/self.f0[0])/TAN2
		self.DY=-(self.RA*scrnL[0]*1.7-5*scrnL[0])/self.f0[0]-scrnL[1]*tan(ang[1])/TAN1-1*scrnL[1]*self.RA*(z-self.z)/self.f0[0]
		A=45*(((atan((self.f0[1]/self.f0[0]))+self.orient-pi/2-ang[0]+pi/8)//(pi/4))%8)
		if A!=self.angle:
			self.angle=A
			self.im=np.minimum(pygame.surfarray.pixels3d(MO[self.type_M][self.angle]),255)
			if self.color!=0:
				u=[0,0,0]
				u[self.color-1]=1
				self.im=u*self.im
			self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)
		
		if self.norm<5 :
			if self.type_M==1:
				VIE=min(100,VIE+20)
				draw_vie()
				self.x0=np.array([-49.,-49])
				s=pygame.mixer.Sound("son/plop.ogg")
				s.play()
				Picked_O[0]=Picked_O[0]+1
			if self.type_M==2:
				self.x0=np.array([-49.,-49])
				AMMO[self.color-1]+=10
				draw_AMMO()
				s=pygame.mixer.Sound("son/plop2.ogg")
				s.play()
				Picked_O[0]=Picked_O[0]+1
			if self.type_M==5:
				self.x0=np.array([-49.,-49])
				CARTE[self.color-1]+=1
				draw_cards()
				s=pygame.mixer.Sound("son/plop3.ogg")
				s.play()
				Picked_O[0]=Picked_O[0]+1
		
	def test_behind(self):
		if self.f0[0]>0 and abs(self.f0[1]/self.f0[0])<TAN2+0.5 :
			if self.norm<=horizon[min(max(int(self.width//2+self.DX+scrnL[0])//2,0),scrnL[0]-1)]+20:
				return True
		else:
			return False
		
	def render(self):
		global Boule,explo,VIE,explo_pt,Killed_O
		if self.inline and shoot==1 and (self.norm<=5 or arme!=0) and self.vie>0 and self.type_M in destr:
			self.vie=0
			Killed_O[0]=Killed_O[0]+1
			if arme!=0:
				Accuracy[0]=Accuracy[0]+1
			if self.type_M!=6:
				s=pygame.mixer.Sound("son/barril.ogg")
				for i in range(randint(5,10)):
					Boule.append(boule(self.x0[0],self.x0[1],self.z,pi*random(),2*pi*random(),2*random()+0.2,1,'image/effects/spark.png'))
				s.play()
				explo=5
				if self.norm<15:
					VIE=VIE-40
					explo_pt=self.x0
					s=pygame.mixer.Sound("son/aie.ogg")
					s.play()

			
		if self.mort<4 and self.vie<=0:
			self.im=np.minimum(pygame.surfarray.pixels3d(Mod[self.type_M][int(self.mort)]),255)
			self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)
			self.mort=min(self.mort+1,3)
			
		
		colorT=light_array[int(self.x0[0]+101)//2][int(self.x0[1]+101)//2]
		if light_array[int(self.x0[0]+101)//2][int(self.x0[1]+101)//2].sum()==0:
			colorT=np.array([1,1,1])
		colorT=light_modif(colorT,level,c3)
		
		SQUARE=np.all(self.norm<=depth,axis=-1) & (Xthing<=self.width+self.DX+scrnL[0]) & (Xthing>=self.DX+scrnL[0])& (Ything<=self.widthY+self.DY+scrnL[1]) & (Ything>=self.DY+scrnL[1])
		
		self.U=np.stack(((Xthing-self.DX-scrnL[0])/self.width,(Ything-self.DY-scrnL[1])/self.widthY),-1)*np.expand_dims(SQUARE,-1)
		self.G=np.maximum((self.U*160).astype(int),0)
		Ar=np.moveaxis(self.im[tuple(map(tuple,self.G.T))]*colorT,1,0)
		if light_array[int(self.x0[0]+101)//2][int(self.x0[1]+101)//2].sum()==0:
			Ar=Ar*torch_on*TORCHE**3
			Ar=np.minimum(np.divide(Ar,0.1*self.norm**0.5),255)
		else:
			Ar=Ar*level_light
		self.Ut=np.moveaxis(self.vis[tuple(map(tuple,self.G.T))],1,0)
		self.inline=min(np.sum(self.Ut[scrnL[0]-10:scrnL[0]+10,3*scrnL[1]//2-10:3*scrnL[1]//2+10]),1)
		return Ar

class boule(pygame.sprite.Sprite):
	def __init__(self,x,y,z,ang1,ang2,v,masse,name):
		self.p=np.array([x,y,z])
		self.ang1=ang1
		self.vx=v*sin(ang1)*cos(ang2)
		self.vy=v*sin(ang1)*sin(ang2)
		self.vz=v*cos(ang1)
		self.v=np.array([self.vx,self.vy,self.vz])
		self.im=pygame.image.load(name)
		self.D=np.linalg.norm(x-self.p[:-1])
		self.f0=(self.p[:-1]-x)@rot_plan(-ang[0])
		self.masse=masse
		self.hit=0
		self.lifetime=0
		if 'image/effects/hurt.png'==name or 'image/effects/hurt2.png'==name:
			self.hit=1
		
		self.X=1*(self.f0[1]/self.f0[0])/TAN2+1
		self.Y=1*(self.p[-1]/self.f0[0])/TAN1+1-1*tan(ang[1])/TAN1
		
	def update(self):
		self.lifetime+=1
		self.v+=np.array([0,0,0.01])*self.masse
		self.p+=self.v
		self.D=np.linalg.norm(x-self.p[:-1])
		self.f0=(self.p[:-1]-x)@rot_plan(-ang[0])
		
		self.X=1*(self.f0[1]/self.f0[0])/TAN2+1
		self.Y=1*((self.p[-1]-z)/self.f0[0])/TAN1+1-1*tan(ang[1])/TAN1

		global VIE,HIT
		if np.linalg.norm((self.p[:-1]-x))<2 and self.hit==1:
			VIE-=10
			HIT=1
			s=pygame.mixer.Sound("son/aie.ogg")
			s.play()
			return True
		
		if (level_map[int(self.p[1]+101)//2][int(self.p[0]+101)//2] and self.hit==0) or (self.p[-1]-z)>5 or (self.p[-1]-z)<-5+hmap[int(self.p[1]+101)//2][int(self.p[0]+101)//2] or self.lifetime>500:
			#print('kill')
			return True
		else:
			return False

	
	def affiche(self):
		if self.f0[0]>0 and abs(self.f0[1]/self.f0[0])<TAN2+0.5 and self.D<=depth[int(self.X*scrnL[0])%(2*scrnL[0])][int(self.Y*scrnL[1]%(2*scrnL[1]))]:
			self.imA=pygame.transform.scale(self.im,(min(int(300/self.f0[0]),window[1]//1),min(int(300/self.f0[0]),window[1]//1)))
			fond.blit(self.imA,(int((window[0]//2)*self.X),int((window[1]//2)*self.Y)))


font = pygame.font.Font('freesansbold.ttf', 13)
def draw_vie():
	pygame.draw.line(fenetre, (255,0,0),(int(0.11*window[0]),int(1.05*window[1])),(int(0.11*window[0]+0.1*window[0]),int(1.05*window[1])),10)
	pygame.draw.line(fenetre, (0,255,0),(int(0.11*window[0]),int(1.05*window[1])),(int(0.11*window[0]+0.1*window[0]*max(VIE,0)/100),int(1.05*window[1])),10)
def draw_hud():
	global back,font,code1,code2,fontC
	fontC = pygame.font.Font('freesansbold.ttf', int(32*window[0]/(12*scrnL[0])))
	code1=pygame.transform.scale(code01,(int(0.8*window[1]),int(0.16*window[1])))
	code2=pygame.transform.scale(code02,(int(0.8*window[1]),int(0.16*window[1])))
	font = pygame.font.Font('freesansbold.ttf', int(13*window[0]/(12*scrnL[0])))
	back=pygame.transform.scale(back,((int(0.8*window[1]),int(10*window[1]))))
	fenetre.blit(pygame.transform.scale(pygame.image.load('image/Interface/fond.png'),(window[0],int(0.2*window[1]))),(0,window[1]))
	fenetre.blit(pygame.transform.scale(pygame.image.load('image/Interface/persom.png'),(int(0.2*window[1]),int(0.2*window[1]))),(0,window[1]))
	fenetre.blit(pygame.transform.scale(pygame.image.load('image/Interface/text.png'),(int(0.8*window[1]),int(0.18*window[1]))),(int(0.45*window[1]),int(1.01*window[1])))
def draw_AMMO():
	for i in range(3):
		u=[100,100,100]
		u[i]=250
		pygame.draw.line(fenetre, (50,50,50),(int(0.8*window[0]),window[1]+int((0.05*i+0.04)*window[1])),(int(0.95*window[0]),window[1]+int((0.05*i+0.04)*window[1])),20)
		text = font.render('AMMO type '+str(i)+'   '+str(AMMO[i]), True, tuple(u))
		textRect = text.get_rect()
		textRect.topleft = (int(0.8*window[0]),window[1]+int((0.05*i+0.025)*window[1]))
		fenetre.blit(text,textRect)
		
def draw_cards():
	fenetre.blit(pygame.transform.scale(pygame.image.load('image/Interface/cards.png'),(int(0.12*window[0]),int(0.08*window[0]))),(int(0.65*window[0]),int(1.01*window[1])))	
	text = font.render('CARDS', True, (200,200,200))
	textRect = text.get_rect()
	textRect.topleft = (int((0.69)*window[0]),int(1.135*window[1]))
	fenetre.blit(text,textRect)
		
	for i in range(3):
		text = font.render(str(CARTE[i]), True, (200,200,200))
		textRect = text.get_rect()
		textRect.topleft = (int((0.67+0.04*i)*window[0]),int(1.11*window[1]))
		fenetre.blit(text,textRect)

def show_message():
	Shift_5=int(5*window[0]/(12*scrnL[0]))
	Shift_13=int(13*window[0]/(12*scrnL[0]))
	fenetre.blit(pygame.transform.scale(pygame.image.load('image/Interface/text.png'),(int(0.8*window[1]),int(0.18*window[1]))),(int(0.45*window[1]),int(1.01*window[1])))
	fenetre.blit(back.subsurface((0,max(linenumber+Sline-5,0)*Shift_13,int(0.8*window[1]),int(0.18*window[1])-2*Shift_5)),(int(0.45*window[1]),Shift_5+int(1.01*window[1])))


def write_message(msg,I):
	global linenumber,indk
	Shift_5=int(5*window[0]/(12*scrnL[0]))
	Shift_13=int(13*window[0]/(12*scrnL[0]))
	if msg[I]=='#':
		linenumber+=1
		indk=I
		return 0
	text = font.render(msg[indk:I], True, (255,255,255))
	textRect = text.get_rect()
	text2 = font.render(msg[I], True, (255,255,255))
	back.blit(text2,(Shift_5+textRect[2],Shift_13*linenumber))
	if textRect[2]>int(0.7*window[1]) or I==len(msg)-1:
		indk=I
		linenumber+=1
	if I==len(msg)-1:
		linenumber+=1
	show_message()

window=12*scrnL
back=pygame.Surface((int(0.8*window[1]),int(10*window[1])),SRCALPHA)
fenetre = pygame.display.set_mode((window[0],int(window[1]*1.2)))
running=1
mouse_c=1
m_stock=window//2
pygame.mouse.set_pos([window[0]//2-10,window[1]//2+10])
pygame.time.wait(100)
ang=[0.0,0.0]
pygame.mouse.set_visible(0)
v=1.2
x=np.array([-1,0])
c=0
c2=0
z=0
Sline=0

M_charac={}
M_charac['life']=[50,100,25]
M_charac['speed']=[0.35,0.35,0.45]
M_charac['degat']=[10,15,15]
M_charac['dist']=[0,0,1]

f = open("image/monstD", "rb")
M1=pickle.load(f)
MD=[]
for i in M1[:]:
	MD0=[]
	for j in i[:]:
		M0={}
		for k in j.keys():
			M0[k]=pygame.image.load('image/monsters/'+j[k])
		MD0.append(M0)
	MD.append(MD0)

Ma0=pickle.load(f)
MDa=[]
for j in Ma0:
	Ma={}
	for k in j.keys():
		Ma[k]=pygame.image.load('image/monsters/'+j[k])
	MDa.append(Ma)
	
f.close()


f = open("image/obj_D", "rb")
MO1=pickle.load(f)
MO=[]
for i in MO1[:]:
	M0={}
	for k in i.keys():
		M0[k]=pygame.image.load('image/objets/'+i[k])
	MO.append(M0)
f.close()

f = open("image/obj_destr_D", "rb")
Mod=pickle.load(f)
Mod2=Mod.copy()
for i in Mod2.keys():
	for k,j in enumerate(Mod2[i]):
		Mod[i][k]=pygame.image.load('image/obj_destr/'+j)
f.close()

f = open("image/gunD", "rb")
MG0=pickle.load(f)
MG1=[]
for i in MG0:
	MG2={}
	for j in i.keys():
		MG2[j]=pygame.transform.scale(pygame.transform.scale(pygame.image.load('image/gun/'+i[j]),(2*scrnL[0],2*scrnL[1])),window)
	MG1.append(MG2)
f.close()
VIE=100
GUN_im=MG1[0][0]
attack=0
shoot=0
coolD=0
arme=0
TotAr=1
COOLDOWN=[10,10,20,0]
DEGAT=[35,25,70,10]
clock = pygame.time.Clock()
colorGUN=(1.,1.,1.)
AMMO=[0,0,0]
CARTE=[0,0,0]
Trig_liste=[]
dicoTEXT={}
groupD=[]
code=0

Killed_E=np.array([0,0,0])
Killed_O=np.array([0,0,0])
Picked_O=np.array([0,0,0])
Accuracy=np.array([0,0,0.])
Explored=np.array([0.,0.])
Play_Time=[0,0]

porteson=pygame.mixer.Sound("son/SF-fermport.ogg")
pygame.mixer.music.load("son/mu.ogg")
pygame.mixer.music.set_volume(0.4)
#----------------------LEVEL

def return_num(key):
	if key[K_KP0]:
		return 0
	if key[K_KP1]:
		return 1
	if key[K_KP2]:
		return 2
	if key[K_KP3]:
		return 3
	if key[K_KP4]:
		return 4
	if key[K_KP5]:
		return 5
	if key[K_KP6]:
		return 6
	if key[K_KP7]:
		return 7
	if key[K_KP8]:
		return 8
	if key[K_KP9]:
		return 9
	return -1

def rescale_Gun():
	global MG1,BLOOD
	for k,i in enumerate(MG0.copy()):
		for j in i.keys():
			MG1[k][j]=pygame.transform.scale(pygame.transform.scale(pygame.image.load('image/gun/'+i[j]),(2*scrnL[0],2*scrnL[1])),window)
	BLOOD=pygame.transform.scale(pygame.image.load('image/effects/blood.png'),window)

def reset_all():
	global VIE,attack,shoot,coolD,arme,TotAr,colorGUN,mouse_c,ang,x,z,R_c,screen
	VIE=100
	attack=0
	shoot=0
	coolD=0
	arme=0
	colorGUN=(1.,1.,1.)
	mouse_c=1
	ang=[1e-5,0.0]
	x=np.array([-1,0])
	z=0
	R_c=np.array([-1,0,0])
	screen=np.full((*scrnL,6),0.0)
	screen[:,:,1:3]=I_n
	screen[:,:,4]=np.sin(Ang[:,:,0])
	screen[:,:,5]=np.cos(-Ang[:,:,1]+pi*0.5)
	screen[:,:,3]=1


def write_text_msg(text):
	font3 = pygame.font.Font('freesansbold.ttf', int(26*window[0]/(12*scrnL[0])))
	text = font3.render(text, True, (255,255,255))
	textRect = text.get_rect()
	fenetre.blit(text,(int(window[0]*0.01),int(window[1]*0.03)))


def stats():
	fenetre.fill((0,0,0))
	pygame.display.flip()
	pygame.time.wait(100)
	IMLOAD=pygame.transform.scale(pygame.image.load('image/interface/back0.png'),(window[0],int(window[1]*1.2)))
	fenetre.blit(IMLOAD,(0,0))
	pygame.display.flip()
	fontH = pygame.font.Font('freesansbold.ttf', int(32*window[0]/(12*scrnL[0])))
	text0 = fontH.render(level_nameL[level], True, (255,255,255))
	textRect0=text0.get_rect()
	for ci in range(len(level_nameL[level])):
		text = fontH.render(level_nameL[level][ci], True, (255,255,255))
		text2 = fontH.render(level_nameL[level][0:ci], True, (255,255,255))
		textRect = text2.get_rect()
		fenetre.blit(text,(int(window[0]*0.5+textRect[2]-textRect0[2]/2),int(window[1]*0.05)))
	
		pygame.event.get()
		pygame.time.wait(50)
		pygame.display.flip()
	
	
	
	fontH = pygame.font.Font('freesansbold.ttf', int(28*window[0]/(12*scrnL[0])))
	stats=[]
	gap='                   '
	stats.append('Ennemies killed :'+str(Killed_E[0])+'/'+str(Killed_E[1]))
	stats.append('Total ennemies killed :'+str(Killed_E[-1]))
	stats.append('Objects destroyed :'+str(Killed_O[0])+'/'+str(Killed_O[1]))
	stats.append('Total objects destroyed :'+str(Killed_O[-1]))
	stats.append('Objects picked :'+str(Picked_O[0])+'/'+str(Picked_O[1]))
	stats.append('Total objects picked :'+str(Picked_O[-1]))
	stats.append('Accuracy :'+str(int(100*Accuracy[0]/max(1,Accuracy[1])))+'%')
	stats.append('Total accuracy :'+str(int(100*Accuracy[-1]))+'%')
	stats.append('Map Explored :'+str(int(Explored[0]*100))+'%')
	stats.append('Total Map Explored :'+str(int(Explored[1]*100))+'%')
	K=str(int(Play_Time[0]//3600))+'h'+str(int((Play_Time[0]%3600)//60))+'m'+str(int(Play_Time[0]%60))+'s'
	Kt=str(int(Play_Time[1]//3600))+'h'+str(int((Play_Time[1]%3600)//60))+'m'+str(int(Play_Time[1]%60))+'s'
	stats.append('Level time :'+K)
	stats.append('Total time :'+Kt)
	linenumber=2
	for c,anim in enumerate(stats):
		if c%2==0:
			linenumber+=2
			shift=0
		else:
			shift=int(window[0]*0.55)
		indK=0
		for ci in range(len(anim)):
			text = fontH.render(anim[ci], True, (255,255,255))
			text2 = fontH.render(anim[indK:ci], True, (255,255,255))
			textRect = text2.get_rect()
			fenetre.blit(text,(int(window[0]*0.05+textRect[2]+shift),int(window[1]*0.05+28*window[0]/(12*scrnL[0])*linenumber)))
			if textRect[2]>int(0.95*window[0]):
				indK=ci
				linenumber+=2
		
			
			pygame.event.get()
			pygame.time.wait(50)
			pygame.display.flip()
	stat_run=1
	text0 = fontH.render('Press space to continue', True, (255,255,255))
	textRect0=text0.get_rect()
	linenumber+=2
	fenetre.blit(text0,(int(window[0]*0.5-textRect0[2]/2),int(window[1]*0.05+28*window[0]/(12*scrnL[0])*linenumber)))
	pygame.display.flip()
	while stat_run==1:
		for event in pygame.event.get():
			if event.type == QUIT:
				running=0
		key=pygame.key.get_pressed()
		if key[K_SPACE]:
			stat_run=0
	return 1

def animation(N):
	if N==0:
		fenetre.fill((0,0,0))
		pygame.display.flip()
		pygame.time.wait(1000)
		anim='Phone:There has been ... an incident, at a biotechnology and genetics lab. We need you in the end. Come now !'
		fontH = pygame.font.Font('freesansbold.ttf', int(32*window[0]/(12*scrnL[0])))
		s=pygame.mixer.Sound("son/rain.ogg")
		s.play()
		for i in range(50):
			pygame.event.get()
			IMLOAD=pygame.transform.scale(pygame.image.load('image/animation/a0_0.png'),(window[0],int(window[1]*1.2)))
			if i%2==0:
				IMLOAD=pygame.transform.scale(pygame.image.load('image/animation/a0_0bis.png'),(window[0],int(window[1]*1.2)))
			fenetre.blit(IMLOAD,(0,0))
			pygame.display.flip()
			pygame.time.wait(100)
				
		timeL=[10,5000,250,4000,150,150,150]
		for i in range(7):
			IMLOAD=pygame.transform.scale(pygame.image.load('image/animation/a'+str(N)+'_'+str(i)+'.png'),(window[0],int(window[1]*1.2)))
			fenetre.blit(IMLOAD,(0,0))
			for j in range(timeL[i]//10):
				pygame.event.get()
				pygame.time.wait(10)
				pygame.display.flip()
			if i==2:
				s=pygame.mixer.Sound("son/phone.ogg")
				s.play()
		indK=0
		linenumber=0
		for ci in range(len(anim)):
			text = fontH.render(anim[ci], True, (255,255,255))
			text2 = fontH.render(anim[indK:ci], True, (255,255,255))
			textRect = text2.get_rect()
			fenetre.blit(text,(int(window[0]*0.05+textRect[2]),int(window[1]*0.05+32*window[0]/(12*scrnL[0])*linenumber)))
			if textRect[2]>int(0.6*window[0]):
				indK=ci
				linenumber+=1

			
			pygame.event.get()
			pygame.time.wait(50)
			pygame.display.flip()
		
		return 1
	return 0

modif_game=['0_1','0_V2','1_2','1_6','2_1','2_6','3_2']

tutotxt=[]
tutotxt.append('Use the arrow keys to move and leftclick to attack')
tutotxt.append('Scroll with the mouse to change weapon')

def change_game(num):
	global TotAr,tuto,activatedT,torch_on,AMMO
	if num=='0_1':
		tuto=1
	if num=='0_V2':
		tuto=2
		TotAr=2
	if num=='1_2':
		activatedT.append(Trig_liste[-1][1])
	if num=='1_6':
		activatedT.remove(Trig_liste[-1][1])
	if num=='2_1':
		activatedT.append(Trig_liste[-2][1])
	if num=='2_6':
		activatedT.remove(Trig_liste[-2][1])
		torch_on=1
		for i in thing:
			if i.type_M==14 and  (i not in ennemies):
				i.x0=np.array([-100,-100])
	if num=='3_2':
		TotAr=3
		for i in thing:
			if i.type_M==16 and  (i not in ennemies):
				i.x0=np.array([-100,-100])
		AMMO[1]+=10
		draw_AMMO()



Xmap_,Ymap_=np.indices((500,500))
def update_map(level_map,x):
	global MAP
	MAP=np.minimum(MAP+np.where(((x[1]/2+50-Xmap_)**2)+((x[0]/2+50-Ymap_)**2)<20**2,level_map,0),1)

mapback=pygame.image.load('image/interface/map.png')
def show_MAP(MAP):
	global mapback,shiftM
	mapback=pygame.transform.scale(mapback,(int(0.75*window[1]),int(0.75*window[1])))
	X=min(x[1]/2+50-50-int(shiftM[1]),0)*(0.7*window[1])/100
	Y=min(x[0]/2+50-50-int(shiftM[0]),0)*(0.7*window[1])/100
	X0=max(x[1]/2+50-50-399-int(shiftM[1]),0)
	Y0=max(x[0]/2+50-50-399-int(shiftM[0]),0)
	
	if X<0:
		shiftM[1]=x[1]/2
		X=0
	if Y<0:
		shiftM[0]=x[0]/2
		Y=0
	if X0>499:
		shiftM[1]=399-x[1]/2
		X0=0
	if Y0>499:
		shiftM[1]=399-x[0]/2
		Y0=0
	MAP2=100*MAP+np.minimum(np.where(((x[1]/2+50-Xmap_)**2)+((x[0]/2+50-Ymap_)**2)<20**2,50+10000*MAP/(((x[1]/2+50-Xmap_)**2)+((x[0]/2+50-Ymap_)**2)),0),150)
	# plt.imshow(np.minimum(np.where(((x[1]/2+50-Xmap_)**2)+((x[0]/2+50-Ymap_)**2)<20**2,100*MAP/(((x[1]/2+50-Xmap_)**2)+((x[0]/2+50-Ymap_)**2)),0),100))
	# plt.show()
	MAP2=np.dstack((MAP2,MAP2,MAP2))
	for i in thing:
		pos=(i.x0+100)/2
		if MAP2[int(pos[1]),int(pos[0]),:].any()!=0:
			if i in ennemies:
				if i.vie>0:
					MAP2[int(pos[1])][int(pos[0])]=[255,0,0]
			else:
				MAP2[int(pos[1])][int(pos[0])]=[0,200,200]
		
	mapim=pygame.surfarray.make_surface(MAP2)
	mapim=mapim.subsurface((int(min(max(x[1]/2+50-50-int(shiftM[1]),0),399)),int(min(max(x[0]/2+50-50-int(shiftM[0]),0),399)),100,100))
	mapim=pygame.transform.flip(pygame.transform.scale(mapim,(int(0.7*window[1]),int(0.7*window[1]))),0,1)
	fenetre.blit(mapim,(int(window[0]/2-0.35*window[1]),int(0.15*window[1])))

	pygame.draw.circle(fenetre,(255,0,0),(int(window[0]/2+min(max(X+X0+shiftM[1]*(0.7*window[1])/100,-0.36*window[1]),0.36*window[1])),int(0.5*window[1]-min(max(Y+Y0+shiftM[0]*(0.7*window[1])/100,-0.36*window[1]),0.36*window[1]))),int(0.01*window[1]))
	fenetre.blit(mapback,(int(window[0]/2-0.375*window[1]),int(0.125*window[1])))

folder_im=pygame.image.load('image/interface/folder.png')
folder_im2=pygame.image.load('image/interface/folder2.png')
fleche=pygame.image.load('image/interface/fleche.png')
fleche2=pygame.image.load('image/interface/fleche2.png')
docs=['no_doc.png']
doc_on=0
INDDOC=0
def show_doc():
	global doc_on,INDDOC

	
	
	if doc_on:

		if abs(mouse[0]-int(1.3*window[1]))<int(0.05*window[1]) and abs(mouse[1]-int(0.8*window[1]))<int(0.05*window[1]):
			fl=pygame.transform.scale(fleche2,(int(0.1*window[1]),int(0.1*window[1])))
			if clic[0]==1:
				INDDOC=(INDDOC+1)%len(docs)
				pygame.time.wait(300)
		else:
			fl=pygame.transform.scale(fleche,(int(0.1*window[1]),int(0.1*window[1])))
		docim=pygame.image.load('image/doc/'+docs[INDDOC])
		docim=pygame.transform.scale(docim,(int(0.75*window[1]),int(0.75*window[1])))
		fenetre.blit(docim,(int(window[0]/2-0.375*window[1]),int(0.125*window[1])))
		fenetre.blit(fl,(int(1.25*window[1]),int(0.75*window[1])))
		fenetre.blit(fl,(int(1.25*window[1]),int(0.75*window[1])))
		if abs(mouse[0]-int(0.3*window[1]))<int(0.15*window[1]) and abs(mouse[1]-int(0.3*window[1]))<int(0.15*window[1]):
			if clic[0]==1:
				doc_on=0
				pygame.time.wait(300)
				show_MAP(MAP)
	else:
		if abs(mouse[0]-int(0.3*window[1]))<int(0.15*window[1]) and abs(mouse[1]-int(0.3*window[1]))<int(0.15*window[1]):
			fim=pygame.transform.scale(folder_im2,(int(0.3*window[1]),int(0.3*window[1])))
			if clic[0]==1:
				doc_on=1
				pygame.time.wait(300)
		else:
			fim=pygame.transform.scale(folder_im,(int(0.3*window[1]),int(0.3*window[1])))
		fenetre.blit(fim,(int(0.15*window[1]),int(0.15*window[1])))
	
		
def check_trigger():
	global startmsg,indk,Sline,groupD,v,activatedT
	#print(queueT,startmsg)
	for i in Trig_liste:
		if np.linalg.norm(2*(i[0]-50)-x)<10 and (i[1] not in activatedT):
			
			Sline=0
			activatedT.append(i[1])
			queueT.append(i[1])
		if len(queueT)>0:
			if startmsg==0 and queueT[0]==i[1]:
				startmsg=len(dicoTEXT[i[1]])
				if "lvl%s_%s.ogg"%(level,i[1]) in os.listdir('son'):
					s=pygame.mixer.Sound("son/lvl%s_%s.ogg"%(level,i[1]))
					s.play()
	
				ref=str(level)+'_'+str(i[1])
				if ref in modif_game:
					change_game(ref)
					
			if startmsg!=0 and queueT[0]==i[1]:
				write_message(dicoTEXT[i[1]],len(dicoTEXT[i[1]])-startmsg)
				startmsg=max(startmsg-1,0)
				if startmsg==0:
					queueT.remove(i[1])
				
				if startmsg==0 and i[1]==level_end[level]:
					Killed_E[-1]=Killed_E[-1]+Killed_E[0]
					Killed_O[-1]=Killed_O[-1]+Killed_O[0]
					Picked_O[-1]=Picked_O[-1]+Picked_O[0]
					Explored[0]=np.sum(MAP)/np.sum(level_map)
					if level!=0:
						Accuracy[-1]=(Accuracy[-1]+Accuracy[0]/max(Accuracy[1],1))/2
						Explored[1]=(Explored[1]+Explored[0])/2
					else:
						Accuracy[-1]=Accuracy[0]/max(Accuracy[1],1)
						Explored[1]=Explored[0]
					Play_Time[0]=((dt.datetime.now()-Play_Time[0])).total_seconds()
					Play_Time[1]=Play_Time[1]+Play_Time[0]
	
					pygame.time.wait(1000)
					stat=stats()
					if stat:
						load_level(str(level+1))
				if startmsg==0 and i[1]==level_start[level]:
					v=1.2
				
			
		if startmsg==0:
			indk=0
	for i in groupD:
		if np.sum([max(j.vie,0) for j in thing if j.group==i])==0 and ('G'+str(i) not in activatedT) :
			Sline=0
			activatedT.append('G'+str(i))
			queueT.append('G'+str(i))
		if len(queueT)>0:
			if startmsg==0 and queueT[0]==('G'+str(i)):
				startmsg=len(dicoTEXT['G'+str(i)])
				if "lvl%s_G%s.ogg"%(level,i) in os.listdir('son'):
					s=pygame.mixer.Sound("son/lvl%s_G%s.ogg"%(level,i))
					s.play()
					
				ref=str(level)+'_G'+str(i)
				if ref in modif_game:
					change_game(ref)	
			
			
			if startmsg!=0 and queueT[0]==('G'+str(i)):
				write_message(dicoTEXT['G'+str(i)],len(dicoTEXT['G'+str(i)])-startmsg)
				startmsg=max(startmsg-1,0)
				if startmsg==0:
					queueT.remove('G'+str(i))
			
			
			
		if np.sum([np.sum(j.U) for j in thing if j.group==i])>0 and ('V'+str(i) not in activatedT):
			Sline=0
			activatedT.append('V'+str(i))
			queueT.append('V'+str(i))
		if len(queueT)>0:
			if startmsg==0 and queueT[0]=='V'+str(i):
				startmsg=len(dicoTEXT['V'+str(i)])
				if "lvl%s_V%s.ogg"%(level,i) in os.listdir('son'):
					s=pygame.mixer.Sound("son/lvl%s_V%s.ogg"%(level,i))
					s.play()
				
				ref=str(level)+'_V'+str(i)
				if ref in modif_game:
					change_game(ref)
	
				
			if startmsg!=0 and queueT[0]=='V'+str(i):
				write_message(dicoTEXT['V'+str(i)],len(dicoTEXT['V'+str(i)])-startmsg)
				startmsg=max(startmsg-1,0)
				if startmsg==0:
					queueT.remove('V'+str(i))
		
		if startmsg==0:
			indk=0
			if 'G'+str(i) in activatedT and 'G'+str(i) not in queueT:
				groupD.remove(i)
			
		

def load_level(level_name):
	
	global lifts,activatedT,TotAr,MAP,v,tuto,level, groupD,indk,startmsg,activatedT,queueT,linenumber,back,dicoTEXT,Trig_liste,AMMO,level_w,level_h,level_map,zmap,light_wall,hmap,authorized_map,M_liste,light_color,light_array,ratio,level_light,wall,doors,h_wall,thing,ennemies
	level=int(level_name)
	TotAr=level_arme[level]
	v=0
	skip=False
	activatedT=[]
	if not skip:
		b=animation(level-1)
		if b:
			pygame.time.wait(1000)
	IMLOAD=pygame.transform.scale(pygame.image.load('image/level/'+level_name+'.png'),window)
	fontH = pygame.font.Font('freesansbold.ttf', int(32*window[0]/(12*scrnL[0])))
	for ci in range(len(level_nameL[level])+1):
		fenetre.fill((0,0,0))
		fenetre.blit(IMLOAD,(0,int(0.2*window[1])))
		text = fontH.render(level_nameL[level][0:ci]+'_', True, (255,255,255))
		textRect = text.get_rect()
		fenetre.blit(text,(int(window[0]*0.05),int(window[1]*0.05)))
		pygame.time.wait(50)
		pygame.display.flip()
	fenetre.fill((0,0,0))
	fenetre.blit(IMLOAD,(0,int(0.2*window[1])))
	text = fontH.render(level_nameL[level], True, (255,255,255))
	textRect = text.get_rect()
	fenetre.blit(text,(int(window[0]*0.05),int(window[1]*0.05)))
	pygame.time.wait(50)
	pygame.display.flip()
	pygame.time.wait(3000)
	fenetre.fill((0,0,0))
	pygame.display.flip()
	pygame.time.wait(500)
	pygame.mouse.set_pos([window[0]//2-10,window[1]//2+10])
	draw_hud()
	draw_cards()
	draw_vie()
	tuto=0
	reset_all()
	groupD=[]
	activatedT=[]
	queueT=[]
	indk=0
	startmsg=0
	linenumber=0
	global Killed_E,Killed_O,Picked_O,Accuracy,Explored,L0
	Killed_E[0:-1]=0
	Killed_O[0:-1]=0
	Picked_O[0:-1]=0
	Accuracy[0:-1]=0
	Explored[0]=0
	Play_Time[0]=dt.datetime.now()
	L0=[]

	
	AMMO=[0,0,0]
	AMMO[0]=min(TotAr-1,1)*20
	draw_AMMO()
	
	wall=[]
	doors=[]#'2'#'Titounet'
	f = open("level/"+level_name, "rb")
	level_w=pickle.load(f)
	level_h=pickle.load(f)
	level_map=pickle.load(f)
	level_map.T
	MAP=np.full((500,500),0)
	pickle.load(f)
	pickle.load(f)
	zmap=pickle.load(f)
	light_wall=pickle.load(f)
	pickle.load(f)
	hmap=pickle.load(f)
	pickle.load(f)
	authorized_map=pickle.load(f)
	authorized_map.T
	M_liste=pickle.load(f)
	light_color=pickle.load(f)
	light_array=pickle.load(f)
	Trig_liste=pickle.load(f)
	pickle.load(f)
	lifts=pickle.load(f)

	f.close()
	ratio=2
	level_light=[1,1,1]
	text_file = open("level/texts/"+level_name+".txt", "r")
	dicoTEXT = ast.literal_eval(text_file.read())
	text_file.close()
	back=pygame.Surface((int(0.8*window[1]),int(10*window[1])),SRCALPHA)
	
	for i in level_w:
		xw=list(2*i[0])
		b=list(2*i[1])
		H=i[-4]
		xw.append(i[4]*2-H)
		b.append(i[5]*2)
		im='image/wall/wall'+str(levelD[level]['wall'][i[2]])+'.png'
		if i[3]!=0:
			im='image/door/'+str(levelD[level]['door'][i[2]])+'.png'

		wall.append(Wall([0.,0,5*2+H],b,xw,im,i[3],i[-3],i[-2],i[-1]))
		if i[3]!=0:
			doors.append(wall[-1])
	[i.texture(5,5) for i in wall]
	
	lenH=0
	for i in level_h:
		a=2*i[0]
		b=2*i[1]
		x1=2*i[2]
		x2=2*i[2]+[0,0,10]
		H=i[3]
		#print(a,b,x1)
		wall.append(Wall(list(a),list(b),list(x1+[0,0,-H]),'image/flat/roof'+str(levelD[level]['flat'][i[4]])+'.png',0,0,1,0))
		if i[4]==9:
			wall[-1].transp=1
		
		if str(levelD[level]['flat'][i[4]])in['7','10']:
			lenH+=1
			wall.append(Wall(list(-a),list(b),list(a+x2),'image/flat/floor'+str(6)+'.png',0,0,1,0))
			wall.append(Wall(list(-a),list(b),list(a+x2+[0,0,-2]),'image/flat/floor'+str(levelD[level]['flat'][i[4]])+'.png',0,0,1,0))
		else:	
			wall.append(Wall(list(-a),list(b),list(a+x2),'image/flat/floor'+str(levelD[level]['flat'][i[4]])+'.png',0,0,1,0))
		
		
	h_wall=[]
	[i.texture(2+int(np.linalg.norm(i.a)/500),2+int(np.linalg.norm(i.b)/500)) for i in wall[-2*len(level_h)-lenH:]]
	h_wall=wall[-2*len(level_h)-lenH:]
	zmap=zmap.T
	
	if level==3:
		
		L0.append(LIZARD(np.array([65,65]),0,x,scrnL,ang,zmap,level))
		L0[0].Activate(x,authorized_map)
		L0[0].start_pattern()
		
	
	thing=[]
	ennemies=[]
	
	for i in M_liste:
		if i[-1] not in groupD:
			groupD.append(i[-1])
		
		if i[-2]:
			monst=Thing(2*(i[0]-50),int(i[1]),1,i[-1])
			thing.append(monst)
			ennemies.append(monst)
			Killed_E[1]=Killed_E[1]+1
		else:
			objet=Object(2*(i[0]-50),int(i[1]),1,i[-1])
			thing.append(objet)
			if thing[-1].type_M in destr:
				Killed_O[1]=Killed_O[1]+1
			if thing[-1].type_M in [1,2,5]:	
				Picked_O[1]=Picked_O[1]+1
			if thing[-1].type_M==2 or thing[-1].type_M==5:
				
				thing[-1].color=(i[-3])+1
	if 0 in groupD:
		groupD.remove(0)
	


SKY0=pygame.surfarray.pixels3d(pygame.image.load('image/ciel/ciel0.png'))
LAND0=pygame.surfarray.pixels3d(pygame.image.load('image/ciel/lanscape0.png'))
LAND0=np.where(np.expand_dims(((LAND0-np.array([0,255,255]))==0).all(-1),-1),-2,LAND0)
BLOOD=pygame.transform.scale(pygame.image.load('image/effects/blood.png'),window)
code01=pygame.image.load('image/Interface/code.png')
code02=pygame.image.load('image/Interface/code2.png')
ttt=[]
bleed=0
explo=0
explo_pt=np.array([0.,0.])
Sky_view=0
ang=(1e-5,0)
draw_hud()
L0=[]
load_level(str(level))
pygame.mixer.music.play(-1)
Im=np.full((2*scrnL[0],2*scrnL[1],3),0)
depth=np.full((2*scrnL[0],2*scrnL[1],1),100.0)
horizon=np.full((scrnL[0],1),10000.0)
POS=np.full((2*scrnL[0],2*scrnL[1]),1000.0)
code_show=0
code_show2=0
code_num=0
code_cool=0
c3=0





while running==1:
	key=pygame.key.get_pressed()
	if key[K_0]:
		load_level('99')
		v=1.2
	milliseconds=[time.time()*1000]
	Rp=rot_plan(ang[0])
	c=(c+1)%(4*3)
	c2=(c2+1)%24
	c3=(c3+1)%10000
	render_w=0
	render_w2=0
	render_C=0
	wall.sort(key=lambda s: s.norm)
	doors.sort(key=lambda s: s.norm)
	h_wall.sort(key=lambda s: s.norm)
	thing.sort(key=lambda s: s.norm)
	ennemies.sort(key=lambda s: s.norm)
	#print(len(wall),len(doors),len(h_wall),len(thing),len(ennemies))
	check_trigger()
	Im=Im*0
	depth=depth*0+100.
	horizon=horizon*0+10000.
	trans=np.array([0.0,0.0])
	POS=POS*0+1000.
	HIT=0
	rset_L=0
	for j,i in enumerate(lifts):
		if abs(2*(i[0]-50)-x[0])<5 and abs(2*(i[1]-50)-x[1])<5:
			rset_L+=1
			if j%2==0:
				if nextL[0]!=i[0] or nextL[1]!=i[1]:
					nextL=lifts[j+1]
					trans=2*(-i+nextL)
			if j%2==1:
				if nextL[0]!=i[0] or nextL[1]!=i[1]:
					nextL=lifts[j-1]
					trans=2*(-i+nextL)
			
			x=x+trans
			z=zmap[int(x[1]+100)//2][int(x[0]+100)//2]
			screen[:,:,:3]=screen[:,:,:3]+np.hstack((trans,[0]))
			screen[:,:,2]=z*2
			zprev=z
			R_c=np.hstack((x,[2*z]))
			trans=np.array([0,0])
			[i.calc_norm() for i in wall]
	if rset_L==0:
		nextL=[0,0]

	
	for event in pygame.event.get():
		if event.type == QUIT:
			running=0
		if event.type==MOUSEBUTTONUP :
			if event.button==4:
				if mouse_c==0:
					Sline=max(Sline-1,-linenumber)
					show_message()
					pygame.display.flip()
				else:
					arme=(arme+1)%TotAr
					GUN_im=MG1[arme][0].copy()
					colorGUN=(1.,1.,1.)
			if event.button==5:
				if mouse_c==0:
					Sline=min(Sline+1,0)
					show_message()
					pygame.display.flip()
				else:
					arme=(arme-1)%TotAr
					GUN_im=MG1[arme][0].copy()
					colorGUN=(1.,1.,1.)
	clic=pygame.mouse.get_pressed()
	mouse=pygame.mouse.get_pos()	


	if key[K_KP_PLUS]==1:
		window=[int(window[0]*1.2),int(window[1]*1.2)]
		m_stock=np.array([window[0]//2,window[1]//2])
		fenetre = pygame.display.set_mode((window[0],int(window[1]*1.2)))
		rescale_Gun()
		draw_hud()
	if key[K_KP_MINUS]==1:
		window=[int(window[0]*0.8),int(window[1]*0.8)]
		m_stock=np.array([window[0]//2,window[1]//2])
		fenetre = pygame.display.set_mode((window[0],int(window[1]*1.2)))
		rescale_Gun()
		draw_hud()

	if clic[2]==1:
		mouse_c=0
		shiftM=np.array([0.,0.])
		pygame.mouse.set_visible(1)
		update_map(level_map,x)
		doc_on=0
		show_MAP(MAP)
	if clic[0]==1:
		if not(abs(mouse[0]-int(0.3*window[1]))<int(0.15*window[1]) and abs(mouse[1]-int(0.3*window[1]))<int(0.15*window[1])) and doc_on==0:
			mouse_c=1
			pygame.mouse.set_visible(0)
		

		if coolD==0:
			attack=1
	if mouse_c==0:
		if key[K_UP] or key[K_w]:
			shiftM=shiftM+[-1,0.]
		if key[K_DOWN] or key[K_s]:
			shiftM=shiftM+[1,0.]
		if key[K_RIGHT] or key[K_d]:
			shiftM=shiftM+[0.,-1.]
		if key[K_LEFT] or key[K_a]:
			shiftM=shiftM+[0,1.]
		
		if doc_on==0:
			update_map(level_map,x)
			show_MAP(MAP)

		show_doc()
		pygame.display.flip()
		continue


		
	if attack and (AMMO[arme-1]>0 or arme==0):
		shoot=(shoot+1)%3
		GUN_im=MG1[arme][shoot//2+1].copy()
		GUN_im.fill(colorGUN,special_flags=BLEND_RGB_MULT)
		if shoot==0:
			attack=0
			GUN_im=MG1[arme][0].copy()
			colorGUN=(1.,1.,1.)
		if shoot==1 :
			s = pygame.mixer.Sound("son/gun%s.ogg"%(arme))
			s.play()
			if arme!=0:
				Accuracy[1]=Accuracy[1]+1
				AMMO[arme-1]=max(AMMO[arme-1]-1,0)
				draw_AMMO()
		coolD=COOLDOWN[arme]


		
	if AMMO[arme-1]<=0 and arme!=0:
		shoot=0
	coolD=max(coolD-1,0)
	
	if key[K_UP] or key[K_w]:
		trans=trans+[-v,0.]
	if key[K_DOWN] or key[K_s]:
		trans=trans+[v,0.]
	if key[K_RIGHT] or key[K_d]:
		trans=trans+[0.,-v]
	if key[K_LEFT] or key[K_a]:
		trans=trans+[0.,v]
	
	if trans.any()!=np.array([0.0,0.0]).any():
		No=np.array([0.,0.])
		for i in wall:
			if i not in h_wall:
				if i.norm3>3:
					break
				trans=i.normal(trans@Rp)@rot_plan(-ang[0])
		x=x-trans@Rp
		z=zmap[int(x[1]+100)//2][int(x[0]+100)//2]
		screen[:,:,:3]=screen[:,:,:3]-np.hstack((trans@Rp,[0]))
		screen[:,:,2]=z*2
		zprev=z
		R_c=np.hstack((x,[2*z]))
		

	if mouse_c==1:
		rot_a=-(2*pi*(mouse-m_stock)/1000)*[1,-0.2]
		if abs((ang+rot_a)[1])<pi/8:
			ang=ang+rot_a
		screenV=screen[:,:,3:]@rot_y(ang[1])@rot_z(ang[0])
		screenP=(screen[:,:,:3]-R_c)@rot_y(ang[1])@rot_z(ang[0])+R_c
		
		pygame.mouse.set_pos([window[0]//2,window[1]//2])
	Rp=rot_plan(ang[0])
	
	
	#print(time.time()*1000-milliseconds[0])------until there same time small and big
	
	[i.calc_norm() for i in wall[0:10]]
	if c2==0:
		[i.calc_norm() for i in wall]
	[i.calc_norm() for i in thing[0:20]]
	if c2==23:
		[i.calc_norm() for i in thing]
	
	
	
	
	CLOSED=0
	code_show=0
	for i in doors[0:5]:
		Ry0=rot_y(-ang[1])
		Ry1=rot_y(ang[1])
		Ri=i.X
		shift=i.z+5
		if i.norm<6 and i.door>=10000:
			code_show=1

			
			
		if i.norm<6 and (i.door==1 or (CARTE[min(i.door-2,2)]>0 and i.door-2<3) or(10000+code==i.door)):
			if CARTE[min(i.door-2,2)]>0 and i.door-2<3 and i.door!=1:
				CARTE[i.door-2]-=1
				i.opendoor(i.door)
				i.door=1
				s=pygame.mixer.Sound("son/unlock.ogg")
				s.play()
				
			if (10000+code==i.door):
				code_show2+=1
				if code_show2>10:
					i.door=1
					code_show2=0
					code_show=0
				
			i.X=np.maximum(Ri-[0.,0.,3],[-10000,-10000,-14.5+shift])
			if i.closed:
				porteson.play()
			i.closed=0
		else:
			if (Ri)[0][0][-1]<-5.+shift:
				i.X=np.minimum(Ri+[0.,0.,3],[10000,10000,-5+shift])
			if (Ri)[0][0][-1]<-4.5+shift:
				i.closed=1
		for j in ennemies[0:5]:
			if j.active and j.vie>0 and level_map[int(j.x0[1]+101)//2][int(j.x0[0]+101)//2]==1 :#np.sum(level_map[int(j.x0[1]+101)//2-1:int(j.x0[1]+101)//2+1,int(j.x0[0]+100)//2-1:int(j.x0[0]+100)//2+1]==1 )>0:
				Ri=i.X
				shift=i.z+5
				if ((j.x0[0]-(0.5*i.b[0][0][0]+i.X[0][0][0]))**2+(j.x0[1]-(0.5*i.b[0][0][1]+i.X[0][0][1]))**2)**0.5<10:# LIMITATION PORTE TAILLE 2 FULL CASES EDITEUR ET DISTANCE D'UNE FULL CASE ENTRE DEUX PORTES
					i.X=np.maximum(Ri-[0.,0.,9],[-10000,-10000,-14.5+shift])
					
					# if i.closed:
						# porteson.play()
					i.closed=0

		CLOSED+=1-i.closed
	
	
	
	D_piece=0
	Sky_view=0
	IS=[]
	for i in wall[0:50]:
		
		devant=True
		if i not in h_wall:
			devant=i.test_behind()
			if i.window>0 and devant:
				CLOSED=1
				Sky_view=1
		else:
			if i.norm>6 :
				devant=False
				if CLOSED!=0 :#and h_wall.index(i)<=6: # INSTEAD CHECK IF ASSOCIATED DOOR WITH THIS FLOOR IS OPEN AND VISIBLE---COMPLICATED
					devant=True
			
		render_w2+=1
		D_piece=max(i.norm,D_piece)
		
		
		if devant:
			render_w+=1
			if i.text[11:-3]not in['floor7.','floor10.']:
				Im=i.render()+Im*(1-np.expand_dims(i.U,-1))
			else:
				IS.append(i)
				
		#print(CLOSED,i.window,(i  in h_wall),i.norm)
		
		if np.count_nonzero(np.sum(Im[3:-3,3:-3],axis=-1)==0)<50 or render_w>50 or i.norm>150:

			break

	
	
	
	if level in [2]:
		if randint(0,100)==0:
			level_light=np.array([1,1,1.])*random()
		else:
			level_light=np.minimum(level_light+0.1*np.array([1,1,1.]),np.array([1,1,1.]))
	
	if Sky_view:
		LAND=np.roll(LAND0,int(ang[0]*12*scrnL[0]/(2*pi)),axis=0)
		LAND=LAND[:2*scrnL[0],:,:]
		LAND=np.roll(LAND,-int(tan(ang[1])*scrnL[1]/TAN1+scrnL[1]),axis=1)
		LAND=LAND[:,:2*scrnL[1],:]

		
		SKY=np.roll(SKY0,int(ang[0]*6*scrnL[0]/(2*pi)),axis=0)
		SKY=SKY[:2*scrnL[0],:,:]
		SKY=np.roll(SKY,-int(tan(ang[1])*scrnL[1]/TAN1+scrnL[1]),axis=1)
		SKY=SKY[:,:2*scrnL[1],:]
		POS=POS*((Im>-1).all(-1))+((Im<=-1).all(-1))*10
		Im=Im*np.expand_dims(((Im>-1).all(-1)),-1)+LAND*np.expand_dims(((Im<=-1).all(-1)),-1)
		Im=Im*np.expand_dims(((Im>-1).all(-1)),-1)+SKY*np.expand_dims(((Im<=-1).all(-1)),-1)
	
	Im=np.minimum((0.8*(np.divide(Im,0.01*np.expand_dims((4*POS)**2,-1))+np.divide(Im,0.1*np.expand_dims((POS),-1)))+0.2*Im),255)#,100)+np.divide(200,np.maximum((depth/5)**2,1))

	horizon=horizon*np.expand_dims(np.linalg.norm(screen[:,20,3:],axis=-1),-1)
	
	#ttt0=time.time()
	[i.move() for i in ennemies[0:10]]
	
	
	
	if c2==23:
		update_map(level_map,x)
	if c%3==0:
		[i.walk() for i in ennemies[0:20]]
	
	if len(L0)>0:
		Deg,Boul=L0[0].pattern(x,scrnL,c,ang,TAN1,TAN2,z,authorized_map,horizon,zmap)
		if Deg!=0:
			VIE=VIE-Deg
			HIT=1
			s=pygame.mixer.Sound("son/aie.ogg")
			s.play()
		if len(Boul)>0:
			for j in Boul:
				Boule.append(boule(*j))
	
	
	for i in thing:
		if i.norm>np.percentile(depth,95):
			break
		if i.test_behind():
			render_C+=1
			Im=i.render()*np.expand_dims(i.Ut,-1)+Im*(1-np.expand_dims(i.Ut,-1))
			depth=depth*(1-np.expand_dims(i.Ut,-1))+np.expand_dims(i.Ut,-1)*i.norm
	if len(L0)>0:
		if L0[0].test_behind(TAN2,scrnL,horizon):
			Im=L0[0].render(depth,Xthing,Ything,scrnL,light_array,level_light,TORCHE,torch_on,arme,shoot,explo,explo_pt,DEGAT,c3)*np.expand_dims(L0[0].Ut,-1)+Im*(1-np.expand_dims(L0[0].Ut,-1))
			depth=depth*(1-np.expand_dims(L0[0].Ut,-1))+np.expand_dims(L0[0].Ut,-1)*L0[0].norm


	for i in IS:
		Im=i.render()*0.5+Im*(1-0.5*np.expand_dims(i.U,-1))

	
	if colorGUN!=tuple(255*light_array[int(x[0]+100)//2][int(x[1]+100)//2]):
		colorGUN=tuple(255*light_array[int(x[0]+100)//2][int(x[1]+100)//2])
		GUN_im=MG1[arme][0].copy()
		GUN_im.fill(colorGUN,special_flags=BLEND_RGB_MULT)
	


	fond=pygame.surfarray.make_surface(Im[3:-3,3:-3])
	fond=pygame.transform.scale(fond,window)

	
	for i in Boule:
		KILL=i.update()
		if KILL:
			Boule.remove(i)
	[i.affiche() for i in Boule]
	
	
	fond.blit(GUN_im,(0,0))
	
	if HIT :
		bleed=20
		draw_vie()
	if bleed!=0:
		BLOOD0=BLOOD.copy()
		BLOOD0.fill((255, 255, 255, 255*bleed/20), special_flags=BLEND_RGBA_MULT)
		fond.blit(BLOOD0,(0,0))
		bleed=max(bleed-1,0)
	if explo!=0:
		fond.fill((40*explo, 40*explo, 40*explo), special_flags=BLEND_RGB_ADD)
		explo=max(explo-1,0)
	
	fenetre.blit(fond,(0,0))
	
	if code_show:
		if code_show2==0:
			if code_num==0:
				code=0
			Number=return_num(key)
			code_cool=max(code_cool-1,0)
			if Number!=-1 and code_cool==0:
				code=code+Number*(10**(3-code_num))
				code_num=(code_num+1)%4
				code_cool=15

		fenetre.blit(code1,(int(0.5*window[0]-0.4*window[1]),int(0.84*window[1])))
		col_code=(255,255,255)
		for i in range(4):
			if i>=code_num:
				col_code=(100,100,100)
			text = fontC.render(str((code//(10**(3-i)))%10), True, col_code)
			fenetre.blit(text,(int(0.5*window[0]+(-2.5+i)*0.16*window[1]+0.17*window[1]),int(0.86*window[1])))

			
	if code_show2>0:
		fenetre.blit(code2,(int(0.5*window[0]-0.4*window[1]),int(0.84*window[1])))
		for i in range(4):
			text = fontC.render(str((code//(10**(3-i)))%10), True, (255,255,255))
			fenetre.blit(text,(int(0.5*window[0]+(-2.5+i)*0.16*window[1]+0.17*window[1]),int(0.86*window[1])))
	
	
	
	
	
	if tuto!=0:
		write_text_msg(tutotxt[tuto-1])
	pygame.display.flip()
	
	if v!=0:
		v=1.2*max(24*(-milliseconds[0]+ time.time()*1000)/1000,1)
	
	print(1000/(-milliseconds[0]+ time.time()*1000),-milliseconds[0]+ time.time()*1000,render_w,render_w2)
	#ttt.append(1000/(-milliseconds[0]+ time.time()*1000))
	#print(np.mean(ttt))
	
	clock.tick_busy_loop(24)

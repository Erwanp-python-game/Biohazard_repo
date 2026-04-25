import numpy as np
from math import *
import pygame# python 3.7.1, pygame 1.9.4
from pygame.locals import *
from random import random,randint
import pickle
from PATH import astar,Node
# import matplotlib.pyplot as plt


window=(500,500)

X=np.moveaxis(np.indices((window[0],window[1])),0,-1)

def light_modif(colorL,level,c):
	colorL2=np.round(colorL,2)
	if level==3 and (colorL2==np.array([1.,0.1,0.1])).all() or (colorL2==np.array([0.9,0.1,0.1])).all():
		colorL2=colorL2*(0.25*cos(c/5)+0.75)
		pass
	return colorL2

def nearest_valid(autho,x):
	d=1000
	xq=[x[0]/2+50,x[1]/2+50]
	xR=[int(xq[0]),int(xq[1])]
	xfin=xR
	for i in [-2,-1,0,1,2]:
		for j in [-2,-1,0,1,2]:
			if (i!=0 or j!=0) and autho[xR[1]+j][xR[0]+i]==2:
				D=(xR[0]+i-xq[0])**2+(xR[1]+j-xq[1])**2
				if D<=d:
					d=D
					xfin=[xR[0]+i,xR[1]+j]
	return xfin

def rot_z(theta):
	return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
def rot_y(theta):
	return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])

def rot_x(theta):
	return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def rot_plan(theta):
	return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

class BOSS():
	def __init__(self,x0,num,x,scrnL,ang,zmap,level,vie):
		self.num = 0
		self.thing_t = 2
		self.x0=2*(x0-50)
		self.group=99
		self.level=level
		self.norm=np.linalg.norm(self.x0-x+[1,0])
		self.width=2*scrnL[0]/self.norm
		self.widthY=2*scrnL[0]/self.norm
		self.DX=0
		self.DY=0
		self.type_M='BOSS'
		self.num=num
		self.angle=0
		self.orient=2*pi*random()
		self.loadim()
		self.im=np.minimum(pygame.surfarray.pixels3d(self.MD[1][45*(((-ang[0]+pi/8+self.orient)//(pi/4))%8)]),255)
		self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)
		self.borne=[self.im.shape[0]-1,self.im.shape[1]-1]
		self.z=zmap[int(self.x0[1]+100)//2][int(self.x0[0]+100)//2]
		self.active=0
		self.inline=0
		self.vie=vie#M_charac['life'][self.type_M]
		self.track=[]
		self.mort=0
		self.range=1#M_charac['dist'][self.type_M]
		self.attack_range=False
		#self.group=group
		self.V=0.25
		self.f0=(self.x0-x)@rot_plan(-ang[0])
		self.U=0
		self.RA=9#levelD[level]['RAmon'][type_M]*(1+(random()*0.2-0.1))
		self.BL=[]
		self.light=np.array([1, 1, 1.])

	def loadim(self):
		f = open("image/BossD", "rb")
		M1=pickle.load(f)
		i=M1[self.num]
		MD0=[]
		for j in i[:]:
			M0={}
			for k in j.keys():
				M0[k]=pygame.image.load('image/Boss/'+j[k])
			MD0.append(M0)
		self.MD=MD0
			
		f.close()
		
		self.MDhit=[]
		for i in range(5):
			self.MDhit.append(pygame.image.load('image/Boss/boss%dhit%d.png'%(self.num,i)))
	
	
	def calc_norm(self,x,scrnL,c,ang,TAN1,TAN2,z):

		self.f0=(self.x0-x)@rot_plan(-ang[0])
		self.norm=np.linalg.norm(self.x0-x)
		self.width=self.RA*2*scrnL[0]/self.f0[0]
		self.widthY=self.RA*2*scrnL[0]/self.f0[0]
		self.DX=-self.RA*scrnL[0]/self.f0[0]+scrnL[0]*(self.f0[1]/self.f0[0])/TAN2
		self.DY=-(self.RA*scrnL[0]*1.7-5*scrnL[0])/self.f0[0]-scrnL[1]*tan(ang[1])/TAN1-(2*scrnL[1]*(z-self.z)/self.f0[0])/TAN1
		A=45*(((atan((self.f0[1]/self.f0[0]))+self.orient-pi/2-ang[0]+pi/8)//(pi/4))%8)
		if A!=self.angle:
			self.angle=A
			self.im=np.minimum(pygame.surfarray.pixels3d(self.MD[c//3][self.angle]),255)
			self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)



	def walk(self,c):
		
		self.im=np.minimum(pygame.surfarray.pixels3d(self.MD[c//3][self.angle]),255)
		self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)

				
		
		
	def test_behind(self,TAN2,scrnL,horizon):
		if self.f0[0]>0 and abs(self.f0[1]/self.f0[0])<TAN2+0.5 :
			if self.norm<=horizon[min(max(int(self.width//2+self.DX+scrnL[0])//2,0),scrnL[0]-1)]:
				return True
			else:
				return False
		else:
			return False	
	
	def check_inline(self,x,authorized_map):

		X1=np.array([int(x[1]+100)//2,int(x[0]+100)//2])
		X2=np.array([int(self.x0[1]+100)//2,int(self.x0[0]+100)//2])
		b=(X1[0]-X2[0])*X1[1]-X1[0]*(X1[1]-X2[1])
		a=(X1[1]-X2[1])
		c=(X1[0]-X2[0])
		
		A=np.where((np.absolute(c*X[:,:,1]-a*X[:,:,0]-b)<0.5*(abs(a)+abs(c)))&(X[:,:,0]<=max(X1[0],X2[0])+0)&(X[:,:,1]<=max(X1[1],X2[1])+0)&(X[:,:,1]>=min(X1[1],X2[1])-0)&(X[:,:,0]>=min(X1[0],X2[0])-0)&(authorized_map==2))
		A1=A[0]-np.roll(A[0],1)
		A2=A[1]-np.roll(A[1],1)
		
		if (A1[1:]>1).any() or (A2[1:]>1).any():
			return False
		else:
			return True
		

		
	def move_to_x(self,x,authorized_map,TAN2,scrnL,horizon,zmap):
		self.attack_range=self.norm<=5
		if self.range==1:
			self.attack_range=(self.check_inline(x,authorized_map))&(self.norm<=20)
			
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
				V=[+0.1+self.V*self.active,0]@rot_plan(self.orient+pi/2)
				self.x0=self.x0+V


			# else:
				# self.active=0
		
		self.z=zmap[int(self.x0[1]+100)//2][int(self.x0[0]+100)//2]
		
	def Activate(self,x,authorized_map):

		self.active=1
		s=pygame.mixer.Sound("son/grognespot%s.ogg"%(1))
		s.play()
		X0=nearest_valid(authorized_map,x)
		self.track=astar(authorized_map,(int(self.x0[1]+101)//2,int(self.x0[0]+101)//2),(X0[1],X0[0]))

			

	def hitten(self,arme,shoot,explo,explo_pt,DEGAT):
		
		if ((self.inline and shoot==1 and (self.attack_range or arme!=0)) or(np.linalg.norm(self.x0-explo_pt[:-1])<20 and explo==1)) and self.vie>0:

			self.im=np.minimum(pygame.surfarray.pixels3d(self.MDhit[0]),255)
			self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)
			if shoot:
				self.vie-=DEGAT[arme]
				
			if explo:
				self.vie-=50
			self.BL=[]
			for i in range(randint(5,10)):
				self.BL.append((self.x0[0],self.x0[1],self.z,pi*random(),2*pi*random(),0.1*random(),2,'image/effects/rouge.png',0))

	def die(self):

		self.im=np.minimum(pygame.surfarray.pixels3d(self.MDhit[int(self.mort)+1]),255)
		self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)
		if self.mort<4:
			self.mort=min(self.mort+0.5,3)
		if self.mort==0.5:
			s=pygame.mixer.Sound("son/grognemort%s.ogg"%(1))
			s.play()
		
		
	def render(self,depth,Xthing,Ything,scrnL,light_array,level_light,TORCHE,torch_on,arme,shoot,explo,explo_pt,DEGAT,c):
		
		self.hitten(arme,shoot,explo,explo_pt,DEGAT)
		if self.vie<=0:
			self.die()
		
		colorT=light_array[int(self.x0[0]+101)//2][int(self.x0[1]+101)//2]
		if light_array[int(self.x0[0]+101)//2][int(self.x0[1]+101)//2].sum()==0:
			colorT=np.array([1,1,1.])
		
		colorT=light_modif(colorT,self.level,c)
		
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


class LIZARD(BOSS):
	def start_pattern(self):
		self.V=0.75
		self.patternX=0
		self.step=0
		self.att_im=[]
		for i in range(2):
			self.att_im.append([])
			for j in range(3):
				self.att_im[-1].append(pygame.image.load('image/Boss/boss0att%d_%d.png'%(i,(j+1))))


	def pattern(self,x,scrnL,c,ang,TAN1,TAN2,z,authorized_map,horizon,zmap):
		DEGAT=0
		B=None

		if self.norm<30 and self.active==0:
			self.Activate(x,authorized_map)

		#print(self.step, self.patternX,self.active,self.vie,self.range,self.attack_range)
		if self.vie>0 and self.active:
			self.move_to_x(x,authorized_map,TAN2,scrnL,horizon,zmap)
			
			if self.patternX==3 :
				self.patternX=0
				self.range=0
				self.V=1.5
				self.attack_range=False
				s = pygame.mixer.Sound("son/grognespot%s.ogg" % (1))
				s.play()
			if self.range==0:
				self.V=max(self.V-0.01,0.75)
				self.patternX-=1
				if self.norm>5 and self.patternX<-48 :
					self.range=1
					self.patternX=0
					self.step=-12
			
			if self.attack_range:
				if self.range==1:
					if self.step<6 and self.step>=0:
						self.im=np.minimum(pygame.surfarray.pixels3d(self.att_im[1][int(self.step//2)]),255)
						self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)
					self.step+=1
					if self.step==2:
						A00=atan((self.x0[1]-x[1])/(self.x0[0]-x[0]))
						if self.x0[0]-x[0]>0:
							A00+=pi

						A11=atan((self.z-z)/((self.x0[0]-x[0])**2+(self.x0[1]-x[1])**2)**0.5)
						self.BL.append((self.x0[0],self.x0[1],self.z,A11+pi/2,A00,1.6,0,'image/effects/hurt2.png',15))
					if self.step==12:
						self.patternX+=1
						self.step=0
				else:
					if self.step<6:
						self.im=np.minimum(pygame.surfarray.pixels3d(self.att_im[0][int(self.step//2)]),255)
						self.vis=np.where(np.sum(self.im,axis=-1)!=0,1,0)
					if self.step==5:
						DEGAT=25
					
					self.step+=1
					if self.step==24:
						self.step=0
	
	
	
			else:
				if c%3==0:
					self.walk(c)

					if self.range == 1:
						self.step+=1

						if self.step==12:
							self.patternX+=1
							self.step=0
		self.calc_norm(x,scrnL,c,ang,TAN1,TAN2,z)
		B=self.BL.copy()
		self.BL=[]
		return DEGAT,B


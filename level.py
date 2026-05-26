import numpy as np
import os
from math import *
import pygame# python 3.7.1, pygame 1.9.4
from pygame.locals import *
pygame.init()
import pickle
import matplotlib.pyplot as plt
from level_data import *
print('level name ?')
name=input()
font = pygame.font.Font('freesansbold.ttf', 13)

import math
import numpy as np

import numpy as np

def point_in_parallelogram(P, A, B, D, eps=1e-9):
    """
    Vectorized check for points inside a parallelogram.

    Parameters
    ----------
    P : array (..., 2)
        Array of points, e.g. shape (l1, l2, 2)
    A, B, D : array-like shape (2,)
        Parallelogram vertices.
        A is the origin corner,
        AB and AD are the edge vectors.
    eps : float
        Numerical tolerance.

    Returns
    -------
    mask : ndarray (...)
        Boolean array with same leading shape as P.
    """
    P = np.asarray(P, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    D = np.asarray(D, dtype=float)

    AB = B - A
    AD = D - A

    # Matrix whose columns are the parallelogram edges
    M = np.stack([AB, AD], axis=1)   # shape (2,2)
    M_inv = np.linalg.inv(M)

    # Coordinates relative to A
    AP = P - A                       # shape (...,2)

    # Solve AP = u*AB + v*AD
    uv = AP @ M_inv.T                # shape (...,2)

    u = uv[..., 0]
    v = uv[..., 1]

    return (
        (-eps <= u) & (u <= 1 + eps) &
        (-eps <= v) & (v <= 1 + eps)
    )
def rectangle_fixed_A_C(A1, C1, theta_deg):
    x1, y1 = A1
    x2, y2 = C1

    # Vector AC
    dx = x2 - x1
    dy = y2 - y1

    # Unit vector u (direction of rectangle side)
    theta = math.radians(theta_deg)
    ux = math.cos(theta)
    uy = math.sin(theta)

    # Perpendicular vector v
    vx = -uy
    vy = ux

    # Solve: u * a + v * b = AC
    # => [ux vx][a] = [dx]
    #    [uy vy][b]   [dy]

    det = ux * vy - uy * vx  # should be 1 for orthonormal basis

    a = (dx * vy - dy * vx) / det
    b = (ux * dy - uy * dx) / det

    # Build points
    B1 = (x1 + a * ux, y1 + a * uy)
    D1 = (x1 + b * vx, y1 + b * vy)

    return np.array(A1).astype(int), np.array(B1).astype(int), np.array(C1).astype(int), np.array(D1).astype(int)




panel=pygame.Surface((350, 500))
col_t=[[155,155,0],[0,155,155],[100,0,155],[80,80,150],[80,150,80],[150,0,0],[20,20,200],[200,20,100],[10,100,200],[100,50,20],[20,100,50],[50,20,100],[190,0,0],[0,190,0],[0,0,190],[0,190,0],[0,0,190],[0,190,0],[0,0,190],[0,190,0],[0,0,190],[0,190,0],[0,0,190]]

for c_0, i in enumerate(levelD[int(name)]['wall']):
	textim = 'image/wall/wall' + str(i) + '.png'
	im = pygame.image.load(textim)
	im = pygame.transform.scale(im, (50, 50))
	panel.blit(im, (0, 50 * c_0))
	pygame.draw.rect(panel, col_t[c_0], (0, 50 * c_0, 10, 10))

for c_0, i in enumerate(levelD[int(name)]['flat']):
	textim = 'image/flat/roof' + str(i) + '.png'
	print(textim)
	im = pygame.image.load(textim)
	im = pygame.transform.scale(im, (50, 50))
	panel.blit(im, (50, 50 * c_0))
	pygame.draw.rect(panel, col_t[c_0], (50, 50 * c_0, 10, 10))

	textim = 'image/flat/floor' + str(i) + '.png'
	im = pygame.image.load(textim)
	im = pygame.transform.scale(im, (50, 50))
	panel.blit(im, (100, 50 * c_0))
	pygame.draw.rect(panel, col_t[c_0], (100, 50 * c_0, 10, 10))

for c_0, i in enumerate(levelD[int(name)]['door']):
	textim = 'image/door/' + str(i) + '.png'
	im = pygame.image.load(textim)
	im = pygame.transform.scale(im, (50, 50))
	panel.blit(im, (150, 50 * c_0))
	pygame.draw.rect(panel, col_t[c_0], (150, 50 * c_0, 10, 10))

for c_0, i in enumerate(levelD[int(name)]['mon']):
	textim = 'image/monsters/m' + str(i + 1) + 'wlk1a1.png'
	im = pygame.image.load(textim)
	im = pygame.transform.scale(im, (50, 50))
	panel.blit(im, (200, 50 * c_0))
	pygame.draw.rect(panel, col_t[c_0], (200, 50 * c_0, 10, 10))

for c_0, i in enumerate(levelD[int(name)]['obj']):
	textim = 'image/objets/O' + str(i + 1) + 'a1.png'
	im = pygame.image.load(textim)
	im = pygame.transform.scale(im, (50, 50))
	panel.blit(im, (250, 50 * c_0))
	pygame.draw.rect(panel, col_t[c_0], (250, 50 * c_0, 10, 10))

for c_0, i in enumerate(levelD[int(name)]['deco']):
	textim = 'image/deco/' + str(i) + '.png'
	im = pygame.image.load(textim)
	im = pygame.transform.scale(im, (25, 25))
	panel.blit(im, (300, 25 * c_0))
	pygame.draw.rect(panel, col_t[c_0], (300, 25 * c_0, 10, 10))


window=(500,500)
fenetre = pygame.display.set_mode((window[0]+350+500,window[1]))
fenetre2 = pygame.Surface((window[0],window[1]))
fenetre2.set_colorkey((0,0,0))
running=1
level_w=np.full((window[0],window[1]),0.0)
level_w_transp=np.full((window[0],window[1]),0.0)
authorized=np.full((window[0],window[1]),0.0)
col=np.full((window[0],window[1],3),0)
seg=0
X=np.moveaxis(np.indices((window[0],window[1])),0,-1)

wall_liste=[]
h_liste=[]

F=pygame.image.load('image/icons/fleche.png').convert_alpha()
F_=pygame.image.load('image/icons/fleche2.png').convert_alpha()
Mo=pygame.image.load('image/icons/monster.png').convert()
B=pygame.image.load('image/icons/black.png').convert()
Amp=pygame.image.load('image/icons/bulb.png').convert()
B.fill((0,0,0))
X1=np.array([0,0])
X2=np.array([0,0])

surf=pygame.Surface((2500, 2500))
surf.fill((0,0,0))
transColor = surf.get_at((0,0))
surf.set_colorkey(transColor)
#print(transColor,B.get_at((0,0)))

editor_controls=pygame.Surface((window[0]+350+500,window[1]))



with open("editor controls.txt") as f:
  for c,x in enumerate(f):
	  text = font.render(x, True, (255, 255, 255))
	  textRect = text.get_rect()
	  editor_controls.blit(text, (window[0] + 350, c*13))


zmap=np.full((window[0],window[1]),0.0)
hmap=np.full((window[0],window[1]),0.0)
light_wall={}
light_color={}
light_L=[]
M_liste=[]
Trig_liste=[]
TrigN=0
lifts=[]
stairs=[]
sphere=[]
face=0
face_d=['AB','A','B']
orientation=0.
slant=0
if name in os.listdir("level/"):
	f = open("level/"+str(name), "rb")
	wall_liste=pickle.load(f)
	h_liste=pickle.load(f)
	level_w=pickle.load(f).T
	surf=pygame.surfarray.make_surface(pickle.load(f))
	surf.set_colorkey((0,0,0))
	col=pickle.load(f)
	zmap=pickle.load(f)
	light_wall=pickle.load(f)
	light_L=pickle.load(f)
	hmap=pickle.load(f)
	authorized=pickle.load(f)
	pickle.load(f)
	M_liste=pickle.load(f)
	pickle.load(f)
	pickle.load(f)
	Trig_liste=pickle.load(f)
	TrigN=pickle.load(f)
	lifts=pickle.load(f)
	stairs = pickle.load(f)
	level_w_transp= pickle.load(f)
	sphere = pickle.load(f)
	f.close()
#h_liste=[]
# Wl=[]
# for i in wall_liste:
# 	Wl.append(tuple(list(i)+[0]))
# wall_liste=Wl
	
# Wl=[]
# for i in M_liste:
# 	i_=list(i)
# 	i_.insert(3,pi/2)
# 	Wl.append(tuple(i_))
# M_liste=Wl

active=['ON','OFF']

x=0
y=0

texture=0
texture2=0
faceA=1
door=0
select=0
selected=0
sel_wall=[]
plafond=0
height=0
Monstre=0
h1=0
h2=0
type_M=0
light_array=np.full((window[0],window[1],3),0.)
light=0
Clight=(1,1,1)
add_roof=1
def flood_fill_authorized(arr,xl):

	
	L_coord=[]
	for i in wall_liste:
		xp=i[0]+50
		yp=i[1]+xp
		L_coord.append(str(xp[0])+','+str(xp[1]))
	A=arr.copy()
	P=[xl]
	v=0

	while len(P)>0 and v<20000:
		x=P[0]
		v+=1
		A[x[0]][x[1]]=2
		P.remove(x)
		
		if A[x[0]+1][x[1]]==0 and (x[0]+1,x[1]) not in P:
			P.append((x[0]+1,x[1]))
		if A[x[0]-1][x[1]]==0 and (x[0]-1,x[1]) not in P:
			P.append((x[0]-1,x[1]))
		if A[x[0]][x[1]+1]==0 and (x[0],x[1]+1) not in P:
			P.append((x[0],x[1]+1))
		if A[x[0]][x[1]-1]==0 and (x[0],x[1]-1) not in P:
			P.append((x[0],x[1]-1))

	return A


def flood_fill(arr,xl,wall_liste,col):
	global light_wall,light_color,light_array
	
	L_coord=[]
	L_coord2=[]
	for i in wall_liste:
		xp=i[0]+50
		yp=i[1]
		L_coord.append(str(int(xp[0]+0.5*yp[0]))+','+str(int(xp[1]+0.5*yp[1])))
		L_coord2.append(str(yp[0])+','+str(yp[1]))
	A=arr.copy()
	P=[xl]
	v=0

	while len(P)>0 and v<20000:
		x=P[0]
		v+=1
		A[x[0]][x[1]]=2
		P.remove(x)
		for i in [-1,0,1]:
			for j in [-1,0,1]:
				light_array[x[0]+i][x[1]+j]=col
				clef=str(x[0]+i)+','+str(x[1]+j)
				if clef in L_coord:
					if clef not in light_wall.keys():
						light_wall[clef]=[str(xl[0])+','+str(xl[1])]
						light_color[clef]=col
					else:
						if str(xl[0])+','+str(xl[1]) not in light_wall[clef]:
							light_wall[clef].append(str(xl[0])+','+str(xl[1]))# ------------------- sytème de clef par le milieu du mur
							light_color[clef]=tuple(0.5*np.array(light_color[clef])+0.5*np.array(col))
		
		if A[x[0]+1][x[1]]==0 and (x[0]+1,x[1]) not in P:
			P.append((x[0]+1,x[1]))
		if A[x[0]-1][x[1]]==0 and (x[0]-1,x[1]) not in P:
			P.append((x[0]-1,x[1]))
		if A[x[0]][x[1]+1]==0 and (x[0],x[1]+1) not in P:
			P.append((x[0],x[1]+1))
		if A[x[0]][x[1]-1]==0 and (x[0],x[1]-1) not in P:
			P.append((x[0],x[1]-1))
	
	# for i in h_liste:# here for rotation
	# 	X1=i[2]+50
	# 	X2=i[0]+i[1]+X1
	# 	V=0.5*(i[0]+i[1])
	# 	if xl[0]<max(X1[0],X2[0]) and xl[1]<max(X1[1],X2[1]) and xl[0]>min(X1[0],X2[0]) and xl[1]>min(X1[1],X2[1]):
	# 		clef=str(int(X1[0]+V[0]))+','+str(int(X1[1]+V[1]))
	#
	# 		if clef not in light_wall.keys():
	# 			light_wall[clef]=[str(xl[0])+','+str(xl[1])]
	# 			light_color[clef]=col
	# 		else:
	# 			if str(xl[0])+','+str(xl[1]) not in light_wall[clef]:
	# 				light_wall[clef].append(str(xl[0])+','+str(xl[1]))
	# 				light_color[clef]=tuple(0.5*np.array(light_color[clef])+0.5*np.array(col))
	for i in h_liste:# here for rotation
		X1=i[2]+50
		X3 = i[0] + X1
		X2=i[0]+i[1]+X1
		X4 = i[1] + X1
		V=0.5*(i[0]+i[1])
		#if xl[0]<max(X1[0],X2[0]) and xl[1]<max(X1[1],X2[1]) and xl[0]>min(X1[0],X2[0]) and xl[1]>min(X1[1],X2[1]):

		if point_in_parallelogram(np.array(xl),X1[:-1],X3[:-1],X4[:-1]):
			clef=str(int(X1[0]+V[0]))+','+str(int(X1[1]+V[1]))

			if clef not in light_wall.keys():
				light_wall[clef]=[str(xl[0])+','+str(xl[1])]
				light_color[clef]=col
			else:
				if str(xl[0])+','+str(xl[1]) not in light_wall[clef]:
					light_wall[clef].append(str(xl[0])+','+str(xl[1]))
					light_color[clef]=tuple(0.5*np.array(light_color[clef])+0.5*np.array(col))


sel_light=[]
sel_plaf=[]
axis=['X','Y']
sel_Mo=[]
M_O=1

Type_t=['object ','monster ']
Type_im=['image/icons/obj.png','image/icons/monster.png']

def write_options():
	step=0
	text = font.render('wall', True, (100+155*(1-door)*(1-light)*(1-plafond)*(1-select)*(1-Monstre),100+155*(1-door)*(1-light)*(1-plafond)*(1-select)*(1-Monstre),100+155*(1-door)*(1-light)*(1-plafond)*(1-select)*(1-Monstre)))
	textRect = text.get_rect()
	textRect.topleft = (5, step)
	fenetre.blit(text,textRect)
	text = font.render('door', True, (100+155*(door)*(1-light)*(1-plafond)*(1-select)*(1-Monstre),100+155*(door)*(1-light)*(1-plafond)*(1-select)*(1-Monstre),100+155*(door)*(1-light)*(1-plafond)*(1-select)*(1-Monstre)))
	textRect = text.get_rect()
	textRect.topleft = (50, step)
	fenetre.blit(text,textRect)
	step+=15
	text = font.render('texture  '+str(texture)+' value key num pad 0->9 Key +/- for deco'+str(deco)+' '+face_d[face]+' '+' 4/5 freq '+str(freq)+' 6/7 phase '+str(phase), True, col_t[texture])
	textRect = text.get_rect()
	textRect.topleft = (5, step)
	fenetre.blit(text,textRect)
	step+=15
	text = font.render('floor mode  '+active[1-plafond]+' heigh '+str(-2*height)+' on/off key return value key +/-', True, (100+155*(plafond),100+155*(plafond),100+155*(plafond)))
	textRect = text.get_rect()
	textRect.topleft = (5, step)
	fenetre.blit(text,textRect)
	step+=15
	text = font.render('roof height '+str(H)+' value key p/o roof height at mouse '+str(Hcurr), True, (100+155*(plafond),100+155*(plafond),100+155*(plafond)))
	textRect = text.get_rect()
	textRect.topleft = (5, step)
	fenetre.blit(text,textRect)
	step+=15
	text = font.render('slope floor along axis'+axis[rot], True, (100+155*(plafond),100+155*(plafond),100+155*(plafond)))
	textRect = text.get_rect()
	textRect.topleft = (5, step)
	fenetre.blit(text,textRect)
	step+=15
	text = font.render('light  '+active[1-light]+' on/off key l value with key 1 2 3 up '+str([round(i,1) for i in Clight]), True, np.array(Clight)*(100+155*(light)))
	textRect = text.get_rect()
	textRect.topleft = (5, step)
	fenetre.blit(text,textRect)
	step+=15
	text = font.render('select mode  '+active[1-select]+' on/off key e + suppr to discard group (validate right enter) '+str(group), True, (100+155*(select),100+155*(select),100+155*(select)))
	textRect = text.get_rect()
	textRect.topleft = (5, step)
	fenetre.blit(text,textRect)
	step+=15
	text = font.render(Type_t[M_O]+active[1-Monstre]+' on/off key 0, switch monster/object key shift', True, (col_t[type_M+ammoT][0]//2+105*(Monstre),col_t[type_M+ammoT][1]//2+105*(Monstre),col_t[type_M+ammoT][2]//2+105*(Monstre)))
	textRect = text.get_rect()
	textRect.topleft = (5, step)
	fenetre.blit(text,textRect)
	step+=15
	if M_O==0:
		text = font.render('ammo type'+str(ammoT), True, (200*(Monstre),200*(Monstre),200*(Monstre)))
		textRect = text.get_rect()
		textRect.topleft = (5, step)
		fenetre.blit(text,textRect)

rot=1
group=0
phase=0
freq=1
H=0
H_curr=0
trigger=0
deco=0
ammoT=0
sel_Trig=[]
sel_lift=[]
sel_stairs=[]
font2 = pygame.font.Font('freesansbold.ttf', 10)
lift=0
stair=0
slope=0
erase=1
angle_flat=0
sphere_on=0
Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
Amp.fill(255*np.array(Clight),special_flags=BLEND_RGB_MULT)
while running==1:
	fenetre.fill((0,0,0))
	fenetre2.fill((0, 0, 0))
	fenetre.blit(editor_controls, (0, 0))
	for event in pygame.event.get():
		if event.type == QUIT:
			running=0
	clic=pygame.mouse.get_pressed()
	key=pygame.key.get_pressed()
	
	if key[K_UP] or key[K_w]:
		y=max(y-1,0)
	if key[K_DOWN] or key[K_s]:
		y=min(y+1,399)
	if key[K_RIGHT] or key[K_d]:
		x=min(x+1,399)
	if key[K_LEFT] or key[K_a]:
		x=max(x-1,0)
	
	
	
	if (type_M==9 or type_M==12) and M_O==0:
		if key[K_t]:
			ammoT=(ammoT+1)%3
			pygame.time.wait(300)
			Mo=pygame.image.load(Type_im[M_O]).convert()
			Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
	else:
		ammoT=0


	if key[K_u]:
		sphere_on=(sphere_on+1)%2
		print('sphere_on',sphere_on)
		pygame.time.wait(300)

	if key[K_i]:
		angle_flat=(angle_flat+5)%360
		print('angle_flat',angle_flat)
		pygame.time.wait(300)

	if key[K_h]:
		add_roof=(add_roof+1)%2
		print('adding roof',add_roof)
		pygame.time.wait(300)
	if key[K_j]:
		slope=(slope+1)%2
		print('slope',slope)
		pygame.time.wait(300)

	if key[K_c]:
		faceA=(faceA+1)%2
		if faceA:
			texture2=texture
		pygame.time.wait(300)
	if key[K_v]:
		face=(face+1)%3
		print(face_d[face])
		pygame.time.wait(300)
	if key[K_x]:
		select=0
		light=0
		plafond=0
		Monstre=0
		trigger=(trigger+1)%2
		lift=0
		pygame.time.wait(300)
	if key[K_q]:
		select=0
		light=0
		plafond=0
		Monstre=0
		trigger=0
		lift=(lift+1)%2
		pygame.time.wait(300)
	if key[K_g]:
		select=0
		light=0
		plafond=0
		Monstre=0
		trigger=0
		lift=0
		stair=(stair+1)%2
		pygame.time.wait(300)
	if key[K_b]:
		orientation=(orientation+pi/4)%(2*pi)
		pygame.time.wait(500)
	if key[K_n]:
		slant=(slant+1)%3
		pygame.time.wait(300)
		print('slant',slant)

	if key[K_LSHIFT]:
		if Monstre:
			M_O=(M_O+1)%2
			Mo=pygame.image.load(Type_im[M_O]).convert()
			Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
			pygame.time.wait(300)
	if key[K_KP0]:
		if Monstre:
			type_M=0
			Mo=pygame.image.load(Type_im[M_O]).convert()
			Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
		else:
			if faceA:
				texture=0
				texture2=0
			else:
				texture2 = 0
	if key[K_KP1]:
		if Monstre:
			type_M=1
			Mo=pygame.image.load(Type_im[M_O]).convert()
			Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
		else:
			if faceA:
				texture=1
				texture2=1
			else:
				texture2 = 1
	if key[K_KP2]:
		if Monstre:
			type_M=2
			Mo=pygame.image.load(Type_im[M_O]).convert()
			Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
		else:
			if faceA:
				texture=2
				texture2=2
			else:
				texture2 = 2
	if key[K_KP3]:
		if Monstre:
			type_M=3
			Mo=pygame.image.load(Type_im[M_O]).convert()
			Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
		else:
			if faceA:
				texture=3
				texture2=3
			else:
				texture2 = 3
	if key[K_KP4]:
		if Monstre:
			type_M=4
			Mo=pygame.image.load(Type_im[M_O]).convert()
			Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
		else:
			if faceA:
				texture=4
				texture2=4
			else:
				texture2 = 4
	if key[K_KP5]:
		if Monstre:
			type_M=5
			Mo=pygame.image.load(Type_im[M_O]).convert()
			Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
		else:
			if faceA:
				texture=5
				texture2=5
			else:
				texture2 = 5
	if key[K_KP6]:
		if Monstre:
			type_M=6
			Mo=pygame.image.load(Type_im[M_O]).convert()
			Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
		else:
			if faceA:
				texture=6
				texture2=6
			else:
				texture2 = 6
	if key[K_KP7]:
		if Monstre:
			type_M=7
			Mo=pygame.image.load(Type_im[M_O]).convert()
			Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
		else:
			if faceA:
				texture=7
				texture2=7
			else:
				texture2 = 7
	if key[K_KP8]:
		if Monstre:
			type_M=8
			Mo=pygame.image.load(Type_im[M_O]).convert()
			Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
		else:
			if faceA:
				texture=8
				texture2=8
			else:
				texture2 = 8
	if key[K_KP9]:
		if Monstre:
			type_M=9
			Mo=pygame.image.load(Type_im[M_O]).convert()
			Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
		else:
			if faceA:
				texture=9
				texture2=9
			else:
				texture2 = 9
	if key[K_RETURN]:
		plafond=(plafond+1)%2
		select=0
		light=0
		lift=0
		pygame.time.wait(300)
	if key[K_e]:
		select=(select+1)%2
		light=0
		plafond=0
		Monstre=0
		lift=0
		erase=1
		pygame.time.wait(300)
	if key[K_k] and select:
		erase = (erase + 1) % 2
		print('erase',erase)
		pygame.time.wait(300)
	if key[K_r]:
		if plafond:
			rot=(rot+1)%2
			pygame.time.wait(300)
	if key[K_l]:
		light=(light+1)%2
		select=0
		plafond=0
		lift=0
		pygame.time.wait(300)
	if key[K_SPACE]:
		door=(door+1)%2
		pygame.time.wait(300)
	if key[K_0]:
		Monstre=(Monstre+1)%2
		select=0
		pygame.time.wait(300)
	
	if key[K_1]:
		Clight=((Clight[0]+0.1)%1,Clight[1],Clight[2])
		Amp=pygame.image.load('image/icons/bulb.png').convert()
		Amp.fill(255*np.array(Clight),special_flags=BLEND_RGB_MULT)
		pygame.time.wait(300)
	if key[K_2]:
		Clight=(Clight[0],(Clight[1]+0.1)%1,Clight[2])
		pygame.time.wait(300)
		Amp=pygame.image.load('image/icons/bulb.png').convert()
		Amp.fill(255*np.array(Clight),special_flags=BLEND_RGB_MULT)
	if key[K_3]:
		Clight=(Clight[0],Clight[1],(Clight[2]+0.1)%1)
		pygame.time.wait(300)
		Amp=pygame.image.load('image/icons/bulb.png').convert()
		Amp.fill(255*np.array(Clight),special_flags=BLEND_RGB_MULT)


	if key[K_LCTRL] and len(sel_wall)==1 and wall_liste[sel_wall[0]][3]!=0:
		select=0
		I=wall_liste[sel_wall[0]]
		x0=I[0]+50
		y0=I[1]+x0
		col[x0[0],x0[1]]=[0,255,0]
		col[y0[0],y0[1]]=[0,255,0]
		Monstre=1
		M_O=0
		type_M=12
		Mo=pygame.image.load(Type_im[M_O]).convert()
		Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
		if key[K_LSHIFT]:
			wall_liste[sel_wall[0]]=wall_liste[sel_wall[0]][0:3]+tuple([1])+wall_liste[sel_wall[0]][4:]
			sel_wall=[]
			type_M=0
			
	if key[K_KP_MULTIPLY] and len(sel_wall)==1 and wall_liste[sel_wall[0]][3]!=0:
		select=0
		I=wall_liste[sel_wall[0]]
		x0=I[0]+50
		y0=I[1]+x0
		col[x0[0],x0[1]]=[0,255,0]
		col[y0[0],y0[1]]=[0,255,0]
		print('Enter code 4 digits:')
		code=10000+int(input())
		print(code)
		wall_liste[sel_wall[0]]=wall_liste[sel_wall[0]][0:3]+tuple([code])+wall_liste[sel_wall[0]][4:]
		print(wall_liste[sel_wall[0]])
		sel_wall=[]
		

	if type_M==12:
		if select==1 or plafond==1 or M_O==1 or light==1:
			wall_liste[sel_wall[0]]=wall_liste[sel_wall[0]][0:3]+tuple([1])+wall_liste[sel_wall[0]][4:]
			sel_wall=[]
			type_M=0
		
	
	if key[K_DELETE]:
		# if plafond:
			# h_liste=[]
			# col=np.where(col[:,:,-1:]==100,[0,0,0],col)

		
		svg=wall_liste[:]
		#print(sel_wall)
		wall_liste=[]
		for i in range(len(svg)):
			if i in sel_wall:
			
				xs=svg[i][0].astype(int)+50
				ys=svg[i][1].astype(int)+xs
				b=(xs[0]-ys[0])*xs[1]-xs[0]*(xs[1]-ys[1])
				a=(xs[1]-ys[1])
				c=(xs[0]-ys[0])
				col=np.where(np.expand_dims((np.absolute(c*X[:,:,1]-a*X[:,:,0]-b)<0.5*(abs(a)+abs(c)))&(X[:,:,0]<=max(xs[0],ys[0])+0)&(X[:,:,1]<=max(xs[1],ys[1])+0)&(X[:,:,1]>=min(xs[1],ys[1])-0)&(X[:,:,0]>=min(xs[0],ys[0])-0),-1) ,[0,0,0],col)
				if erase:
					level_w=np.where((np.absolute(c*X[:,:,1]-a*X[:,:,0]-b)<0.5*(abs(a)+abs(c)))&(X[:,:,0]<=max(xs[0],ys[0])+0)&(X[:,:,1]<=max(xs[1],ys[1])+0)&(X[:,:,1]>=min(xs[1],ys[1])-0)&(X[:,:,0]>=min(xs[0],ys[0])-0) ,0,level_w)
					level_w_transp = np.where((np.absolute(c * X[:, :, 1] - a * X[:, :, 0] - b) < 0.5 * (abs(a) + abs(c))) & (
								X[:, :, 0] <= max(xs[0], ys[0]) + 0) & (X[:, :, 1] <= max(xs[1], ys[1]) + 0) & (
												   X[:, :, 1] >= min(xs[1], ys[1]) - 0) & (
												   X[:, :, 0] >= min(xs[0], ys[0]) - 0), 0, level_w_transp)
					if door==0:
						authorized=np.where((np.absolute(c*X[:,:,1]-a*X[:,:,0]-b)<0.5*(abs(a)+abs(c)))&(X[:,:,0]<=max(xs[0],ys[0])+0)&(X[:,:,1]<=max(xs[1],ys[1])+0)&(X[:,:,1]>=min(xs[1],ys[1])-0)&(X[:,:,0]>=min(xs[0],ys[0])-0) ,0,authorized)

				if np.sum(level_w[xs[0]-1:xs[0]+2,xs[1]-1:xs[1]+2])>0:
					col[xs[0],xs[1]]=[0,255,0]
				else:
					col[xs[0],xs[1]]=[0,0,0]
				if np.sum(level_w[ys[0]-1:ys[0]+2,ys[1]-1:ys[1]+2])>0:
					col[ys[0],ys[1]]=[0,255,0]
				else:
					col[ys[0],ys[1]]=[0,0,0]
				A=(np.angle(ys[0]-xs[0]+x+(ys[1]-xs[1]+y)*1j))
				surf.blit(pygame.transform.rotate(B,180-A*180/pi),(5*xs+5*ys)//2+[-5,-5])
			else:
				wall_liste.append(svg[i])
		sel_wall=[]
		for j in (sel_light):
			x0=j[0]
			col[x0[0],x0[1]]=(col[x0[0]+1,x0[1]+1]+col[x0[0]-1,x0[1]-1])//2
		Ltemp=[]
		for j in (light_L):
			add=True
			for k in (sel_light):
				#print(j[0],k[0] ,j[1],k[1])
				if j[0][0]==k[0][0] and j[0][1]==k[0][1]:
					add=False
					surf.blit(B,(5*j[0][0]-5,5*j[0][1]-5))
			if add:
				Ltemp.append(j)
		light_L=Ltemp
		sel_light=[]
		
		Ltemp=[]
		for j in (M_liste):
			add=True
			for k in (sel_Mo):
				if j[0][0]==k[0][0] and j[0][1]==k[0][1]:
					add=False
					surf.blit(B,(5*j[0][0]-5,5*j[0][1]-5))
			if add:
				Ltemp.append(j)
		M_liste=Ltemp
		sel_Mo=[]
		
		
		Ltemp=[]
		for j in (Trig_liste):
			add=True
			for k in (sel_Trig):
				if j[0][0]==k[0][0] and j[0][1]==k[0][1]:
					add=False
					surf.blit(B,(5*j[0][0]-5,5*j[0][1]-5))
			if add:
				Ltemp.append(j)
		Trig_liste=Ltemp
		sel_Trig=[]
		
		Ltemp=[]
		for j in (lifts):
			add=True
			for k in (sel_lift):
				if j[0]==k[0] and j[1]==k[1]:
					add=False
					surf.blit(B,(5*j[0]-5,5*j[1]-5))
			if add:
				Ltemp.append(j)
		lifts=Ltemp
		sel_lift=[]
		
		Ltemp=[]
		for j in (stairs):
			add=True
			for k in (sel_stairs):
				if j[0]==k[0] and j[1]==k[1]:
					add=False
					surf.blit(B,(5*j[0]-5,5*j[1]-5))
			if add:
				Ltemp.append(j)
		stairs=Ltemp
		sel_lift=[]
		
		L_temp=[]
		L_mid=[]
		for j in sel_plaf:
			L_mid.append(j[1])
		for i in range(len(h_liste)):
			if i not in L_mid:
				L_temp.append(h_liste[i])
			else:
				U=h_liste[i]
				X1=(U[2]+50)
				X2=(U[0]+X1+U[1])
				X1=X1[:-1]
				X2=X2[:-1]
				#print(X1,X2)
				col=np.where(np.expand_dims((X[:,:,0]>min(X1[0],X2[0]))&(X[:,:,0]<max(X1[0],X2[0]))&(X[:,:,1]>min(X1[1],X2[1]))&(X[:,:,1]<max(X1[1],X2[1]))&(level_w==0),-1)&(col!=[200,200,200]),[0,0,0],col)
		h_liste=L_temp
		sel_plaf=[]
			

	if door:
		CC=[255,255,255]
	else:
		CC=col_t[texture]
	mouse=pygame.mouse.get_pos()#+5*np.array([x,y])
	if select==0 and plafond==0 and light==0 and Monstre==0 and trigger==0 and lift==0 and stair==0:
		if seg==1 and sphere_on:
			X_m = np.array([mouse[0] // 5 + x, mouse[1] // 5 + y])
			Radius = np.linalg.norm(X_m - X2) * 5
			pygame.draw.circle(fenetre2, [50, 50, 200],5 * X2 - 5 * np.array([x, y]), Radius)

		if key[K_KP_PLUS]:
			deco=deco+1
			pygame.time.wait(300)
		if key[K_KP_MINUS]:
			deco=max(deco-1,0)
			pygame.time.wait(300)
			
		if key[K_4]:
			freq=freq+1
			pygame.time.wait(300)
		if key[K_5]:
			freq=max(freq-1,1)
			pygame.time.wait(300)
		
		if key[K_6]:
			phase=phase+1
			pygame.time.wait(300)
		if key[K_7]:
			phase=max(phase-1,0)
			pygame.time.wait(300)
		
		
		if clic[0]==1:
			if sphere_on==0:
				level_w[mouse[0]//5+x,mouse[1]//5+y]=1
				if (levelD[int(name)]['deco'][deco-1] in deco_door):
					level_w_transp[mouse[0]//5+x,mouse[1]//5+y]=1
				if door==0:
					authorized[mouse[0]//5+x,mouse[1]//5+y]=1
				col[mouse[0]//5+x,mouse[1]//5+y]=[0,255,0]
			else:
				if seg==0:
					col[mouse[0] // 5 + x, mouse[1] // 5 + y] = [0, 255, 0]
			pygame.time.wait(200)
			if seg==1 and sphere_on==0:
				A=(np.angle(mouse[0]//5-X2[0]+x+(mouse[1]//5-X2[1]+y)*1j))
				surf.blit(pygame.transform.rotate(F,180-A*180/pi),(5*X2+mouse+[5*x,5*y])//2+[-5,-5])
			X1=X2
			X2=np.array([mouse[0]//5+x,mouse[1]//5+y])

			
			if seg==1 and sphere_on==0:
				
				col[mouse[0]//5+x,mouse[1]//5+y]=[0,255,0]
				b=(X1[0]-X2[0])*X1[1]-X1[0]*(X1[1]-X2[1])
				a=(X1[1]-X2[1])
				c=(X1[0]-X2[0])
				#print(X1-50,X2-X1)
				wall_liste.append((X1-50,X2-X1,[texture,texture2,face_d[face]],door,-2.5,0,0,deco,freq,phase,slant))#-2.5h,0pente vecteur b
				
				col=np.where(np.expand_dims((np.absolute(c*X[:,:,1]-a*X[:,:,0]-b)<0.5*(abs(a)+abs(c)))&(X[:,:,0]<=max(X1[0],X2[0])+0)&(X[:,:,1]<=max(X1[1],X2[1])+0)&(X[:,:,1]>=min(X1[1],X2[1])-0)&(X[:,:,0]>=min(X1[0],X2[0])-0)&(level_w==0),-1) ,CC,col)
				level_w=np.where((np.absolute(c*X[:,:,1]-a*X[:,:,0]-b)<0.5*(abs(a)+abs(c)))&(X[:,:,0]<=max(X1[0],X2[0])+0)&(X[:,:,1]<=max(X1[1],X2[1])+0)&(X[:,:,1]>=min(X1[1],X2[1])-0)&(X[:,:,0]>=min(X1[0],X2[0])-0)&(level_w==0) ,1,level_w)
				if (levelD[int(name)]['deco'][deco - 1] in deco_door):
					level_w_transp=np.where((np.absolute(c*X[:,:,1]-a*X[:,:,0]-b)<0.5*(abs(a)+abs(c)))&(X[:,:,0]<=max(X1[0],X2[0])+0)&(X[:,:,1]<=max(X1[1],X2[1])+0)&(X[:,:,1]>=min(X1[1],X2[1])-0)&(X[:,:,0]>=min(X1[0],X2[0])-0)&(level_w_transp==0) ,1,level_w_transp)
				print(levelD[int(name)]['deco'][deco-1],deco_destruc,deco)
				if door==0 and (levelD[int(name)]['deco'][deco-1] not in deco_destruc or deco==0) and (levelD[int(name)]['deco'][deco-1] not in deco_door or deco==0):
					authorized=np.where((np.absolute(c*X[:,:,1]-a*X[:,:,0]-b)<0.5*(abs(a)+abs(c)))&(X[:,:,0]<=max(X1[0],X2[0])+0)&(X[:,:,1]<=max(X1[1],X2[1])+0)&(X[:,:,1]>=min(X1[1],X2[1])-0)&(X[:,:,0]>=min(X1[0],X2[0])-0)&(authorized==0) ,1,authorized)
			seg=1
			#print(level_w)
		if clic[2]==1:
			
			# if seg:
				# col[X2[0],X2[1]]=col[X2[0]+1,X2[1]+1]
			seg=0
			
	if select==1 and plafond==0 and light==0 and Monstre==0 and trigger==0 and lift==0 and stair==0:
		if key[K_KP_PLUS]:
			group=group+1
			pygame.time.wait(300)
		if key[K_KP_MINUS]:
			group=max(group-1,0)
			pygame.time.wait(300)
		if key[K_KP_ENTER]:

			for i in sel_Mo:
				M_liste[i[1]]=tuple(list(M_liste[i[1]][:-1])+[group])

		 
		if clic[0]==1:
			
			pygame.time.wait(200)
			
			X1=X2
			X2=np.array([mouse[0]//5+x,mouse[1]//5+y])
			seg=(seg+1)%2
			if seg==0:
				for j,i in enumerate(wall_liste[:]):
					x0=i[0]+50
					y0=i[1]+x0
					if x0[0]<max(X1[0],X2[0]) and x0[1]<max(X1[1],X2[1]) and x0[0]>min(X1[0],X2[0]) and x0[1]>min(X1[1],X2[1]) and y0[0]<max(X1[0],X2[0]) and y0[1]<max(X1[1],X2[1]) and y0[0]>min(X1[0],X2[0]) and y0[1]>min(X1[1],X2[1]):
						print(y0, x0)
						col[int(x0[0]),int(x0[1])]=[255,255,0]

						col[int(y0[0]),int(y0[1])]=[255,255,0]
						selected=1
						sel_wall.append(j)
						#wall_liste.remove(i)
					
						
					
				for j,i in enumerate(light_L):
					x0=i[0]
					if x0[0]<max(X1[0],X2[0]) and x0[1]<max(X1[1],X2[1]) and x0[0]>min(X1[0],X2[0]) and x0[1]>min(X1[1],X2[1]):
						sel_light.append(i)
						col[x0[0],x0[1]]=[255,255,0]
						surf.blit(B,(5*x0[0]-5,5*x0[1]-5))
						Amp=pygame.image.load('image/icons/bulb.png').convert()
						Amp.fill((255,255,0),special_flags=BLEND_RGB_MULT)
						surf.blit(Amp,(5*x0[0]-5,5*x0[1]-5))
						Amp=pygame.image.load('image/icons/bulb.png').convert()
						Amp.fill(255*np.array(Clight),special_flags=BLEND_RGB_MULT)
						
				
				for j,i in enumerate(h_liste):
					x0=i[2]+50+i[0]//2+i[1]//2
					if x0[0]<max(X1[0],X2[0]) and x0[1]<max(X1[1],X2[1]) and x0[0]>min(X1[0],X2[0]) and x0[1]>min(X1[1],X2[1]):
						col[int(x0[0]),int(x0[1])]=[255,255,0]
						sel_plaf.append((x0,j,i))
				

				for j,i in enumerate(M_liste):
					x0=i[0]
					#print(x0,'a')
					if x0[0]<max(X1[0],X2[0]) and x0[1]<max(X1[1],X2[1]) and x0[0]>min(X1[0],X2[0]) and x0[1]>min(X1[1],X2[1]):

						sel_Mo.append((x0,j,i))
						surf.blit(B,(5*x0[0]-5,5*x0[1]-5))
						Mo=pygame.image.load(Type_im[M_O]).convert()
						Mo.fill((255,255,0),special_flags=BLEND_RGB_MULT)
						surf.blit(Mo,(5*x0[0]-5,5*x0[1]-5))
						Mo=pygame.image.load(Type_im[M_O]).convert()
						Mo.fill(col_t[type_M+ammoT],special_flags=BLEND_RGB_MULT)
				
				for j,i in enumerate(Trig_liste):
					x0=i[0]
					if x0[0]<max(X1[0],X2[0]) and x0[1]<max(X1[1],X2[1]) and x0[0]>min(X1[0],X2[0]) and x0[1]>min(X1[1],X2[1]):
						sel_Trig.append((x0,j,i))
						text = font2.render(str(i[-1]), True, (255,255,0))
						surf.blit(B,(5*x0[0]-5,5*x0[1]-5))
						surf.blit(text,(5*x0[0]-5,5*x0[1]-5))
				for j,i in enumerate(lifts):
					x0=i
					if x0[0]<max(X1[0],X2[0]) and x0[1]<max(X1[1],X2[1]) and x0[0]>min(X1[0],X2[0]) and x0[1]>min(X1[1],X2[1]):
						sel_lift.append(x0)
						text = font2.render('L'+str(j), True, (255,255,0))
						surf.blit(B,(5*x0[0]-5,5*x0[1]-5))
						surf.blit(text,(5*x0[0]-5,5*x0[1]-5))

				for j,i in enumerate(stairs):
					x0=i
					if x0[0]<max(X1[0],X2[0]) and x0[1]<max(X1[1],X2[1]) and x0[0]>min(X1[0],X2[0]) and x0[1]>min(X1[1],X2[1]):
						sel_stairs.append(x0)
						text = font2.render('S'+str(j), True, (255,255,0))
						surf.blit(B,(5*x0[0]-5,5*x0[1]-5))
						surf.blit(text,(5*x0[0]-5,5*x0[1]-5))
					

		if clic[2]==1:
			seg=0
			selected=0
			for j in (sel_wall):
				I=wall_liste[j]
				x0=I[0]+50
				y0=I[1]+x0
				col[x0[0],x0[1]]=[0,255,0]
				col[y0[0],y0[1]]=[0,255,0]
			for j in (sel_light):
				x0=j[0]
				col[x0[0],x0[1]]=[200,200,200]
				surf.blit(B,(5*x0[0]-5,5*x0[1]-5))
				surf.blit(Amp,(5*x0[0]-5,5*x0[1]-5))
			for j in (sel_plaf):
				x0=j[1]
				col[int(x0[0]),int(x0[1])]=col_t[j[-1]]
			
			for j in sel_Mo:
				x0=j[0]
				surf.blit(B,(5*x0[0]-5,5*x0[1]-5))
				surf.blit(Mo,(5*x0[0]-5,5*x0[1]-5))
			for j in sel_Trig:
				x0=j[0]
				text = font2.render(str(j[-1][-1]), True, (255,255,255))
				surf.blit(B,(5*x0[0]-5,5*x0[1]-5))
				surf.blit(text,(5*x0[0]-5,5*x0[1]-5))
				

			
			sel_light=[]
			sel_wall=[]
			sel_plaf=[]
			sel_Mo=[]
			sel_Trig=[]
	
	if select==0 and plafond and light==0 and Monstre==0 and trigger==0 and lift==0 and stair==0:
		if key[K_KP_PLUS]:
			height=height-0.5
			pygame.time.wait(300)
		if key[K_KP_MINUS]:
			height=height+0.5
			pygame.time.wait(300)
			
		if key[K_p]:
			H=H+1
			pygame.time.wait(300)
		if key[K_o]:
			H=H-1
			pygame.time.wait(300)
		
		
		
		if clic[0]==1:
			#print(height)
			pygame.time.wait(200)
			X1=X2
			X2=np.array([mouse[0]//5+x,mouse[1]//5+y])
			h1=h2
			h2=height
			seg=(seg+1)%2
			A_, B_, C_, D_ = rectangle_fixed_A_C(X1, X2, angle_flat)
			C_ = A_ + (B_ - A_) + (D_ - A_)
			if seg==0:
				hmap=np.where(point_in_parallelogram(X,A_,B_,D_),H,hmap)
				#hmap=np.where((X[:,:,0]>=min(X1[0],X2[0]))&(X[:,:,0]<=max(X1[0],X2[0]))&(X[:,:,1]>=min(X1[1],X2[1]))&(X[:,:,1]<=max(X1[1],X2[1])),H,hmap)

				if rot:
					pente=((h2-h1)/(X2[1]-X1[1]))
					alt=np.expand_dims(pente*(X[:,:,1]-X1[1]),-1)
					if add_roof:
						# col=np.where(np.expand_dims((X[:,:,0]>min(X1[0],X2[0]))&(X[:,:,0]<max(X1[0],X2[0]))&(X[:,:,1]>min(X1[1],X2[1]))&(X[:,:,1]<max(X1[1],X2[1]))&(level_w==0),-1)&(col!=[200,200,200]),[0,100,100]+50*(alt+h1)*[1,0,0],col)
						# print(point_in_parallelogram(X,A_,B_,D_).shape)
						col=np.where(np.expand_dims((point_in_parallelogram(X,A_,B_,D_))&(level_w==0),-1)&(col!=[200,200,200]),[0,100,100]+50*(alt+h1)*[1,0,0],col)

					zmap=np.where((point_in_parallelogram(X,A_,B_,D_)),alt[:,:,0]+h1,zmap)
					# zmap = np.where((X[:, :, 0] >= min(X1[0], X2[0])) & (X[:, :, 0] <= max(X1[0], X2[0])) & (
					# 			X[:, :, 1] >= min(X1[1], X2[1])) & (X[:, :, 1] <= max(X1[1], X2[1])), alt[:, :, 0] + h1,
					# 				zmap)
					if add_roof:
						if slope==0:
							h_liste.append((np.array([B_[0]-A_[0],B_[1]-A_[1] , 0]),np.array([D_[0]-A_[0],D_[1]-A_[1] , h2 - h1]),np.array([X1[0]-50,X1[1]-50,-2.5+h1]),H,texture))#0 était -2.5+h1
						else:
							print('angle_flat',angle_flat)
							# format is a,b,x0
							# X1 and X2 not changing but a and b yes --> needs rotation
							# ax=(X2[0] - X1[0])*cos(angle_flat*2*pi/360)
							# ay = -(X2[0] - X1[0]) * sin(angle_flat*2*pi/360)
							# bx = (X2[1] - X1[1]) * sin(angle_flat*2*pi/360)
							# by=(X2[1] - X1[1])*cos(angle_flat*2*pi/360)



							h_liste.append((np.array([B_[0]-A_[0],B_[1]-A_[1] , 0]), np.array([D_[0]-A_[0],D_[1]-A_[1] , h2 - h1]),
											np.array([X1[0] - 50, X1[1] - 50, -2.5 + h1]), H, texture,1))


							# h_liste.append((np.array([ax, 0, 0]), np.array([bx,by , h2 - h1]),
							# 				np.array([X1[0] - 50, X1[1] - 50, -2.5 + h1]), H, texture,1))
							if pente==0:
								print('platform')
								X1p=[]
								X2p=[]
								X1p.append(A_)
								X2p.append(B_)

								X1p.append(B_)
								X2p.append(C_)

								X1p.append(C_)
								X2p.append(D_)

								X1p.append(D_)
								X2p.append(A_)

								# X1p.append(np.array([X1[0],X1[1]]))
								# X2p.append(np.array([X2[0],X1[1]]))
								#
								# X1p.append(np.array([X2[0],X1[1]]))
								# X2p.append(np.array([X2[0],X2[1]]))
								#
								# X1p.append(np.array([X2[0],X2[1]]))
								# X2p.append(np.array([X1[0],X2[1]]))
								#
								# X1p.append(np.array([X1[0],X2[1]]))
								# X2p.append(np.array([X1[0],X1[1]]))
								
								for xx in range(len(X1p)):
									b = (X1p[xx][0] - X2p[xx][0]) * X1p[xx][1] - X1p[xx][0] * (X1p[xx][1] - X2p[xx][1])
									a = (X1p[xx][1] - X2p[xx][1])
									c = (X1p[xx][0] - X2p[xx][0])
									# b = (X1p[xx][0] - X2p[xx][0]) * X1p[xx][1] - X1p[xx][0] * (X1p[xx][1] - X2p[xx][1])
									# a = (X1p[xx][1] - X2p[xx][1])
									# c = (X1p[xx][0] - X2p[xx][0])

									wall_liste.append((
													  X1p[xx] - 50, X2p[xx] - X1p[xx], [texture, texture2, face_d[face]], door, h1+2.5, 0, 0,
													  deco, freq, phase, slant))

									col = np.where(np.expand_dims(
										(np.absolute(c * X[:, :, 1] - a * X[:, :, 0] - b) < 0.5 * (abs(a) + abs(c))) & (
													X[:, :, 0] <= max(X1p[xx][0], X2p[xx][0]) + 0) & (
													X[:, :, 1] <= max(X1p[xx][1], X2p[xx][1]) + 0) & (
													X[:, :, 1] >= min(X1p[xx][1], X2p[xx][1]) - 0) & (
													X[:, :, 0] >= min(X1p[xx][0], X2p[xx][0]) - 0) & (level_w == 0), -1), CC, col)
					if slope == 0: # below this finish parallelogram stuff
						for j,i in enumerate(wall_liste[:]):
							x0=i[0]+50
							y0=i[1]+x0
							#if x0[0]<=max(X1[0],X2[0]) and x0[1]<=max(X1[1],X2[1]) and x0[0]>=min(X1[0],X2[0]) and x0[1]>=min(X1[1],X2[1]) and y0[0]<=max(X1[0],X2[0]) and y0[1]<=max(X1[1],X2[1]) and y0[0]>=min(X1[0],X2[0]) and y0[1]>=min(X1[1],X2[1]):
							if point_in_parallelogram(x0,A_,B_,D_) and point_in_parallelogram(y0,A_,B_,D_):
								if -np.sign((i[1])[1]*(X2[1]-X1[1]))>0:# HERE #tuple(list(i[:-4])+[(alt[:,:,0])[x0[0],x0[1]]-2.5+h1]+[-np.sign(-(i[1])[0]*(X2[0]-X1[0]))*((alt[:,:,0])[y0[0],y0[1]])]+[H]+i[-1])
									wall_liste[j]=tuple(list(i[:4])+[(alt[:,:,0])[x0[0],x0[1]]-2.5+h1]+[-np.sign(-(i[1])[1]*(X2[1]-X1[1]))*((alt[:,:,0])[x0[0],x0[1]])]+[H]+list(i[7:]))
								else:
									wall_liste[j]=tuple(list(i[:4])+[(alt[:,:,0])[x0[0],x0[1]]-2.5+h1]+[-np.sign(-(i[1])[1]*(X2[1]-X1[1]))*((alt[:,:,0])[y0[0],y0[1]])]+[H]+list(i[7:]))
				else:
					pente=((h2-h1)/(X2[0]-X1[0]))
					alt=np.expand_dims(pente*(X[:,:,0]-X1[0]),-1)
					if add_roof:
						col=np.where(np.expand_dims((point_in_parallelogram(X,A_,B_,D_))&(level_w==0),-1)&(col!=[200,200,200]),[0,100,100]+50*(alt+h1)*[1,0,0],col)
						#col=np.where(np.expand_dims((X[:,:,0]>min(X1[0],X2[0]))&(X[:,:,0]<max(X1[0],X2[0]))&(X[:,:,1]>min(X1[1],X2[1]))&(X[:,:,1]<max(X1[1],X2[1]))&(level_w==0),-1)&(col!=[200,200,200]),[0,100,100]+50*(alt+h1)*[1,0,0],col)
					zmap=np.where(point_in_parallelogram(X,A_,B_,D_),alt[:,:,0]+h1,zmap)
					#zmap=np.where((X[:,:,0]>=min(X1[0],X2[0]))&(X[:,:,0]<=max(X1[0],X2[0]))&(X[:,:,1]>=min(X1[1],X2[1]))&(X[:,:,1]<=max(X1[1],X2[1])),alt[:,:,0]+h1,zmap)

					if add_roof:
						if slope==0:
							#h_liste.append((np.array([X2[0]-X1[0],0,h2-h1]),np.array([0,X2[1]-X1[1],0]),np.array([X1[0]-50,X1[1]-50,-2.5+h1]),H,texture))
							h_liste.append((np.array([B_[0]-A_[0],B_[1]-A_[1],h2-h1]),np.array([D_[0]-A_[0],D_[1]-A_[1],0]),np.array([X1[0]-50,X1[1]-50,-2.5+h1]),H,texture))
						else:
							# h_liste.append((np.array([X2[0] - X1[0], 0, h2 - h1]), np.array([0, X2[1] - X1[1], 0]),
							# 				np.array([X1[0] - 50, X1[1] - 50, -2.5 + h1]), H, texture,1))
							h_liste.append((np.array([B_[0]-A_[0],B_[1]-A_[1], h2 - h1]), np.array([D_[0]-A_[0],D_[1]-A_[1], 0]),
											np.array([X1[0] - 50, X1[1] - 50, -2.5 + h1]), H, texture,1))
							if pente==0:
								print('platform')
					if slope==0:
						for j,i in enumerate(wall_liste[:]):
							x0=i[0]+50
							y0=i[1]+x0
							if point_in_parallelogram(x0,A_,B_,D_) and point_in_parallelogram(y0,A_,B_,D_):
							#if x0[0]<=max(X1[0],X2[0]) and x0[1]<=max(X1[1],X2[1]) and x0[0]>=min(X1[0],X2[0]) and x0[1]>=min(X1[1],X2[1]) and y0[0]<=max(X1[0],X2[0]) and y0[1]<=max(X1[1],X2[1]) and y0[0]>=min(X1[0],X2[0]) and y0[1]>=min(X1[1],X2[1]):
								#HERE
								if -np.sign((i[1])[0]*(X2[0]-X1[0]))>0:
									wall_liste[j]=tuple(list(i[:4])+[(alt[:,:,0])[x0[0],x0[1]]-2.5+h1]+[-np.sign(-(i[1])[0]*(X2[0]-X1[0]))*((alt[:,:,0])[x0[0],x0[1]])]+[H]+list(i[7:]))
								else:
									wall_liste[j]=tuple(list(i[:4])+[(alt[:,:,0])[x0[0],x0[1]]-2.5+h1]+[-np.sign(-(i[1])[0]*(X2[0]-X1[0]))*((alt[:,:,0])[y0[0],y0[1]])]+[H]+list(i[7:]))
				if add_roof:
					col[(X1[0]+X2[0])//2,(X1[1]+X2[1])//2]=col_t[texture]
			
		if clic[2]==1:
			#if seg:
			#col[(X1[0]+X2[0])//2,(X1[1]+X2[1])//2]=col[1+(X1[0]+X2[0])//2,1+(X1[1]+X2[1])//2]
			seg=0
			
		if seg:
			if add_roof:
				# pygame.draw.rect(fenetre,[(50*(height))%255,100,100],(5*X2-5*np.array([x,y]),mouse-5*X2+5*np.array([x,y])))

				A_0, B_0, C_0, D_0 = rectangle_fixed_A_C(X2, np.array([mouse[0]//5+x,mouse[1]//5+y]), angle_flat)
				C_0 = A_0 + (B_0 - A_0) + (D_0 - A_0)
				pygame.draw.polygon(fenetre2, [(50 * (height)) % 255, 100, 200],
								 (5 * A_0 - 5 * np.array([x, y]), 5*B_0 - 5 * np.array([x, y]),5*C_0 - 5 * np.array([x, y]),5*D_0 - 5 * np.array([x, y])))

			else:
				pygame.draw.rect(fenetre2, [(50 * (height)) % 255, 255, 200],
								 (5 * X2 - 5 * np.array([x, y]), mouse - 5 * X2 + 5 * np.array([x, y])))
		
	if select==0 and plafond==0 and light==1 and Monstre==0 and trigger==0 and lift==0 and stair==0:
		if clic[0]==1:
			XL=np.array([mouse[0]//5+x,mouse[1]//5+y])
			if col[XL[0],XL[1],:].all()!=200:
				light_L.append([XL,Clight])
			col[XL[0],XL[1],:]=[200,200,200]
			pygame.time.wait(100)
			print((5*(mouse[0]//5)-5+5*x,5*(mouse[1]//5)-5+5*y))
			surf.blit(B,(5*(mouse[0]//5)-5+5*x,5*(mouse[1]//5)-5+5*y))
			surf.blit(Amp,(5*(mouse[0]//5)-5+5*x,5*(mouse[1]//5)-5+5*y))
			pygame.time.wait(100)
	if select==0 and plafond==0 and light==0 and Monstre and trigger==0 and lift==0 and stair==0:
		if clic[0]==1:

			XL=np.array([mouse[0]//5+x,mouse[1]//5+y])
			M_liste.append((XL,type_M,ammoT,pi-orientation,M_O,group))#M_liste.append((XL,type_M,ammoT,M_O,group))
			surf.blit(B,(5*(mouse[0]//5)-5+5*x,5*(mouse[1]//5)-5+5*y))
			surf.blit(Mo,(5*(mouse[0]//5)-5+5*x,5*(mouse[1]//5)-5+5*y))
			surf.blit(pygame.transform.rotate(F_, 180 - orientation * 180 / pi),
					  (5 * (mouse[0] // 5) - 5 + 5 * x, 5 * (mouse[1] // 5) - 5 + 5 * y))

			if type_M==12:

				wall_liste[sel_wall[0]]=wall_liste[sel_wall[0]][0:3]+tuple([ammoT+2])+wall_liste[sel_wall[0]][4:]
				sel_wall=[]
				type_M=0
			pygame.time.wait(100)
	
	if clic[1]==1:
		print('pos',np.array([mouse[0]//5+x,mouse[1]//5+y]))
			
	
	Hcurr=hmap[mouse[0]//5+x][mouse[1]//5+y]
	
	C=pygame.surfarray.make_surface(col[x:100+x,y:100+y,:])
	C.set_colorkey((0,0,0))
	fenetre.blit(pygame.transform.scale(C,window),(0,0))
	fenetre.blit(fenetre2,(0,0))
	
	if seg and select and plafond==0 and light==0 and trigger==0 and lift==0 and stair==0:
		pygame.draw.rect(fenetre,[155,0,55],(5*X2-5*np.array([x,y]),mouse-5*X2+5*np.array([x,y])))
	
	if select==0 and plafond==0 and light==0 and Monstre==0 and trigger and lift==0 and stair==0:
		pygame.draw.circle(fenetre,[200,200,200],mouse,25,2)
		if clic[0]==1:
			XL=np.array([mouse[0]//5+x,mouse[1]//5+y])
			TrigN+=1
			Trig_liste.append((XL,TrigN))
			text = font2.render(str(TrigN), True, (255,255,255))
			surf.blit(B,(5*(mouse[0]//5)-5+5*x,5*(mouse[1]//5)-5+5*y))
			surf.blit(text,(5*(mouse[0]//5)-5+5*x,5*(mouse[1]//5)-5+5*y))
			pygame.time.wait(300)
	if select==0 and plafond==0 and light==0 and Monstre and trigger==0 and lift==0 and stair==0:
		if levelD[int(name)]['obj'][type_M]==28 and M_O==0:
			pygame.draw.circle(fenetre, [200, 200, 200], mouse+20*np.array([np.cos(orientation-pi/2),np.sin(orientation-pi/2)]), 20, 2)
	if select==0 and plafond==0 and light==0 and Monstre==0 and trigger==0 and lift==1:
		pygame.draw.rect(fenetre,[200,200,200],(mouse[0]-12,mouse[1]-12,24,24),2)
		if clic[0]==1:
			XL=np.array([mouse[0]//5+x,mouse[1]//5+y])
			LN=len(lifts)
			lifts.append(XL)
			text = font2.render('L'+str(LN), True, (255,255,255))
			surf.blit(B,(5*(mouse[0]//5)-5+5*x,5*(mouse[1]//5)-5+5*y))
			surf.blit(text,(5*(mouse[0]//5)-5+5*x,5*(mouse[1]//5)-5+5*y))
			pygame.time.wait(300)
	if select==0 and plafond==0 and light==0 and Monstre==0 and trigger==0 and lift==0 and stair==1:
		pygame.draw.rect(fenetre,[200,200,200],(mouse[0]-12,mouse[1]-12,24,24),2)
		if clic[0]==1:
			XL=np.array([mouse[0]//5+x,mouse[1]//5+y])
			SN=len(stairs)
			stairs.append(XL)
			text = font2.render('S'+str(SN), True, (255,255,255))
			surf.blit(B,(5*(mouse[0]//5)-5+5*x,5*(mouse[1]//5)-5+5*y))
			surf.blit(text,(5*(mouse[0]//5)-5+5*x,5*(mouse[1]//5)-5+5*y))
			pygame.time.wait(300)
	
	# if 1 in key:
		# print('-------------')
		# print('Door ',active[1-door],'key space')
		# print('Texture ',texture,'key pav num')
		# print('select mode ',active[1-select],'key e')
		# print('floor mode ',active[1-plafond],'key enter')
		# print('height ',height,'key +/-')
		# print('light ',active[1-light],'key l')
		# pygame.time.wait(50)
	
	
	for i in range(100):
		pygame.draw.line(fenetre,(50,50,50),(i*5,window[1]),(i*5,0))
		pygame.draw.line(fenetre,(50,50,50),(window[1],i*5),(0,i*5))
	for i in range(20):
		pygame.draw.line(fenetre,(100,100,100),(i*25,window[1]),(i*25,0))
		pygame.draw.line(fenetre,(100,100,100),(window[1],i*25),(0,i*25))
		
	pygame.draw.line(fenetre,(0,0,250),(5*window[1]-5*x,250-5*y),(0-5*x,250-y*5))
	pygame.draw.line(fenetre,(0,0,250),(250-5*x,5*window[1]-5*y),(250-5*x,0-5*y))
	
	pygame.draw.line(fenetre,(0,255,255),(mouse[0],window[1]),(mouse[0],0))
	pygame.draw.line(fenetre,(0,255,255),(window[0],mouse[1]),(0,mouse[1]))
	if seg and select==0 and plafond==0:
		A=(np.angle(mouse[0]//5-X2[0]+x+(mouse[1]//5-X2[1]+y)*1j))
		fenetre.blit(pygame.transform.rotate(F,180-A*180/pi),5*X2+[-2-5*x,-10-5*y])

	

	fenetre.blit(surf,(-5*x,-5*y))
	write_options()
	text = font.render('Y', True, (255,255,255))
	textRect = text.get_rect()
	textRect.topleft = (250, 0)
	fenetre.blit(text,textRect)
	text = font.render('X', True, (255,255,255))
	textRect = text.get_rect()
	textRect.topleft = (0, 250)
	fenetre.blit(text,textRect)


	fenetre.blit(panel,(500,0))

	xsquare=500+door*150
	ysquare=texture*50
	ysquare2 = texture2 * 50
	if deco!=0:
		pygame.draw.rect(fenetre, (255, 255, 0), (500+300, (deco-1)*25, 25, 25), 3)
	if plafond:
		xsquare=500+50
		pygame.draw.rect(fenetre, (255, 255, 0), (xsquare+50, ysquare, 50, 50), 3)

	elif Monstre:
		xsquare = 500 + 200+50-M_O*50
		ysquare=type_M*50
	pygame.draw.rect(fenetre,(255,255,0),(xsquare,ysquare,50,50),3)
	pygame.draw.circle(fenetre,(255-100*faceA,100*faceA,255-100*faceA), (xsquare+25,ysquare2+25), 25, 2)

	fenetre.blit(pygame.transform.rotate(F, 180 - orientation * 180 / pi),
				 (10, 200))
	pygame.display.flip()

light_wall={}

for i in light_L:
	flood_fill(level_w,(i[0][0],i[0][1]),wall_liste,i[1])

#print(light_wall)

authorized_map=flood_fill_authorized(authorized,(50,50))
for i in lifts[1::2]:
	authorized_map=flood_fill_authorized(authorized_map,i)
for i in stairs[1::2]:
	authorized_map=flood_fill_authorized(authorized_map,i)


fig,ax=plt.subplots(1,3)
authorized_map=authorized_map.T
ax[0].imshow(authorized_map)
ax[1].imshow(level_w.T)
ax[2].imshow(level_w_transp.T)
plt.show()
print(np.sum(authorized_map))

# wall_liste_replace=[]
# for i in wall_liste:
# 	print(i)
# 	j=list(i).copy()
# 	print(j[2])
# 	j[2]=list(j[2])
# 	j[2].append('AB')
# 	print(j)
# 	wall_liste_replace.append(j)
# 	print('-----------')
# wall_liste=wall_liste_replace

level_w=level_w.T

print('do you want to save (y/n) ?')
saving=input()
if saving!='n':
	if saving=='y':
		f = open("level/"+str(name), "wb")
	else:
		f = open("level/autosave", "wb")
	pickle.dump(wall_liste, f)
	pickle.dump(h_liste, f)
	pickle.dump(level_w, f)
	pickle.dump(pygame.surfarray.pixels3d(surf), f)
	pickle.dump(col, f)
	pickle.dump(zmap, f)
	pickle.dump(light_wall, f)
	pickle.dump(light_L, f)
	pickle.dump(hmap, f)
	pickle.dump(authorized, f)
	pickle.dump(authorized_map, f)
	pickle.dump(M_liste, f)
	pickle.dump(light_color, f)
	pickle.dump(light_array, f)
	pickle.dump(Trig_liste,f)
	pickle.dump(TrigN,f)
	pickle.dump(lifts,f)
	pickle.dump(stairs, f)
	pickle.dump(level_w_transp, f)
	pickle.dump(sphere, f)
	f.close()

import matplotlib.pyplot as plt
import os
# Prevent BLAS from taking over threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from level_data import *
from math import *
import time
import pygame  #current python 3.8.8 numpy 1.20.1 pygame 2.0 attention pour numpy à avoir la bonne version blas/lapack                   python 3.7.1, pygame 1.9.4
from pygame.locals import *
import pickle

pygame.init()
import os
# import torch
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from random import random, randint
from PATH import astar, Node
import ast
import datetime as dt
from Boss import BOSS, LIZARD, nearest_valid, rot_z, rot_y, rot_plan, light_modif

# def rot_z(theta):
# return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
# def rot_y(theta):
# return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])

# def rot_plan(theta):
# return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
activatedT = []
lifts = []
scrnL = np.array([80, 40])
screen = np.full((*scrnL, 6), 0.0)
depth = np.full((2 * scrnL[0], 2 * scrnL[1], 1), 100.0)
POS = np.full((2 * scrnL[0], 2 * scrnL[1]), 1000.0)
horizon = np.full((scrnL[0], 1), 100.0)
gun_width=1
shift_a=[0,25,0,10,0]


Ang = np.full((*scrnL, 2), 0.0)
I = np.indices(scrnL)
I_n = np.divide(np.moveaxis((np.indices(scrnL)), 0, 2) - 0.5 * scrnL, scrnL[1])
screen[:, :, 1:3] = I_n
Ang = I_n
Ang[:, :, 0] = Ang[:, :, 0] * pi / 4
Ang[:, :, 1] = Ang[:, :, 1] * atan(1 / 2) * 2



screen[:, :, 0] = screen[:, :, 0]

screen[:, :, 4] = np.sin(Ang[:, :, 0])
screen[:, :, 5] = np.cos(-Ang[:, :, 1] + pi * 0.5)
screen[:, :, 3] = 1

TAN1 = np.amax(screen[:, :, 5])
TAN2 = np.amax(screen[:, :, 4])

screenV = screen[:, :, 3:]
screenP = screen[:, :, :3]


#----------------
screen0 = np.full((2*scrnL[0],2*scrnL[1], 6), 0.0)
Ang0 = np.full((2*scrnL[0],2*scrnL[1], 2), 0.0)
I0 = np.indices((2*scrnL[0],2*scrnL[1]))
I_n0 = np.divide(np.moveaxis((np.indices((2*scrnL[0],2*scrnL[1]))), 0, 2) -  scrnL, 2*scrnL[1])
screen0[:, :, 1:3] = I_n0
Ang0 = I_n0
Ang0[:, :, 0] = Ang0[:, :, 0] * pi / 4
Ang0[:, :, 1] = Ang0[:, :, 1] * atan(1 / 2) * 2
screen0[:, :, 0] = screen0[:, :, 0]

screen0[:, :, 4] = np.sin(Ang0[:, :, 0])
screen0[:, :, 5] = np.cos(-Ang0[:, :, 1] + pi * 0.5)
screen0[:, :, 3] = 1

screenV0 = screen0[:, :, 3:]
screenP0 = screen0[:, :, :3]

#----------------


CENTER = np.expand_dims(np.linalg.norm(screen[:, :, :3] - [0, 0, 0], axis=-1).repeat(2, axis=0).repeat(2, axis=1), -1)
TORCHE = np.expand_dims(
    (np.maximum(np.cos(I_n[:, :, 0] * pi / 2) * np.cos(I_n[:, :, 1] * 2 * pi / 2), 0)).repeat(2, axis=0).repeat(2,
                                                                                                                axis=1),
    -1)
torch_on = 0

R_c = np.array([-1, 0, 0])
Vg = np.array([1, -sqrt(2) / 2]) / sqrt(3 / 2)
Vd = np.array([1, sqrt(2) / 2]) / sqrt(3 / 2)

setting = {}
setting['smooth'] = False
destr = [0, 4, 6,11]
level = 2
level_nameL = ['Level 0: Training', 'Level 1: The Lab', 'Level 2: The Storage', 'Level 3: The Basement',
               'Level 4: The Manor','Level 5: The Caves']
level_arme = [1, 2, 2, 2, 3,5]  # last 3
level_end = [5, 7, 8, 6, 6,10]
level_start = [1, 2, 1, 1, 1,1]
for i in range(100 - len(level_arme)):
    level_arme.append(5)
    level_end.append(0)
    level_start.append(100)
    level_nameL.append('no')
level_end[99] = 10
level_start[99] = 4

z_tileable_deco = [26, 28, 31]


def source_pos(code):
    x = (int(code.split(',')[0]) - 50) * 2
    y = (int(code.split(',')[1]) - 50) * 2
    z = zmap[int(code.split(',')[1])][int(code.split(',')[0])] - hmap[int(code.split(',')[0])][int(code.split(',')[1])]
    return np.array([x, y, z - 5])


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
def plane(a,b,X):
    M = np.stack((a * 1.1, b * 1.1, -screenV), axis=-1)
    B = -X + screenP
    inv_M = np.linalg.inv(M.astype(np.float32))
    S = np.einsum('...ij,...j->...i', inv_M, B.astype(np.float32))
    return S

def explo_zone(R,dist):
    white = np.array([[[1, 1, 1]]])
    yellow=np.array([[[1,1,0]]])
    red=np.array([[[1,0,0]]])
    #color=yellow*(1-(dist/R))+red*(dist/R)
    color = np.where(dist<R/2,white*(1-(2*dist/R))+yellow*(2*dist/R),yellow*(1-(2*(dist-0.5*R)/R))+red*(2*(dist-0.5*R)/R))
    return color

class Wall():
    def __init__(self, u, v, w, text, door, deco, freq, phase):
        self.a = np.full((scrnL[0], scrnL[1], 3), u)
        self.b = np.full((scrnL[0], scrnL[1], 3), v)
        self.X = np.full((scrnL[0], scrnL[1], 3), w)
        self.door = door
        self.closed = 1
        self.n = np.cross(self.a[0][0], self.b[0][0])
        N = np.stack((self.a[0][0], self.b[0][0], -self.n), axis=-1)
        V = np.maximum(np.minimum(np.linalg.solve(N, -self.X[0][0] + R_c)[:-1], 1), 0)
        self.norm = np.linalg.norm(self.X[0][0][:-1] + V[0] * self.a[0][0][:-1] + V[1] * self.b[0][0][:-1] - R_c[:-1])
        self.normf = self.norm
        self.norm3 = np.linalg.norm(self.X[0][0] + V[0] * self.a[0][0] + V[1] * self.b[0][0] - R_c)
        self.text = text[0]
        self.text2 = text[1]
        self.verrou = pygame.image.load('image/deco/verrouO.png')
        self.deco=deco
        self.vie=0
        self.explo=0
        Imdeco = pygame.image.load(self.text)
        Imdeco2 = pygame.image.load(self.text2)
        if deco != 0:
            freq2 = int(0.5 + np.linalg.norm(self.a[0][0]) / 10)
            if levelD[level]['deco'][deco - 1] in z_tileable_deco:
                freq2 = 1
            Imdeco = pygame.transform.scale(Imdeco, (120 * freq, 120 * freq2))
            Imdeco2 = pygame.transform.scale(Imdeco2, (120 * freq, 120 * freq2))
            for i in range(freq):
                for j in range(freq2):
                    Imdeco.blit(pygame.image.load(text[0]), (120 * i, 120 * (j)))
                    Imdeco2.blit(pygame.image.load(text[1]), (120 * i, 120 * (j)))
            if text[2] == 'A' or text[2] == 'AB':
                Imdeco.blit(pygame.image.load('image/deco/' + str(levelD[level]['deco'][deco - 1]) + '.png'),
                            (120 * (freq - 1 - phase), 120 * (freq2 - 1)))
            if text[2] == 'B' or text[2] == 'AB':
                Imdeco2.blit(pygame.image.load('image/deco/' + str(levelD[level]['deco'][deco - 1]) + '.png'),
                             (120 * (freq - 1 - phase), 120 * (freq2 - 1)))
        if door > 1 and door < 5:
            verrou = pygame.image.load('image/deco/verrou.png')
            if door == 2:
                colo = (255, 50, 50)
            if door == 3:
                colo = (50, 255, 50)
            if door == 4:
                colo = (50, 50, 255)
            verrou.fill(colo, special_flags=BLEND_RGB_MULT)
            Imdeco.blit(verrou, (0, 0))
            Imdeco2.blit(verrou, (0, 0))

        IM = np.transpose(pygame.surfarray.pixels3d(Imdeco), (1, 0, 2))
        IM = np.where(np.expand_dims(((IM - np.array([255, 0, 255])) == 0).all(-1), -1), -1, IM)
        IM = np.where(np.expand_dims(((IM - np.array([0, 255, 255])) == 0).all(-1), -1), -2, IM)
        self.window = np.sum(IM <= -1)
        self.wall_im = [np.flip(np.minimum(IM + 1, 255), (0, 1))]

        IM2 = np.transpose(pygame.surfarray.pixels3d(Imdeco2), (1, 0, 2))
        IM2 = np.where(np.expand_dims(((IM2 - np.array([255, 0, 255])) == 0).all(-1), -1), -1, IM2)
        IM2 = np.where(np.expand_dims(((IM2 - np.array([0, 255, 255])) == 0).all(-1), -1), -2, IM2)
        self.wall_im2 = [np.flip(np.minimum(IM2 + 1, 255), (0, 1))]

        self.phase = 0
        if self.text[11:-3] in liquid_floor:
            self.phase = 1

        files = [filename for filename in os.listdir(self.text[:11]) if filename.startswith(self.text[11:-3])]
        files_d = [filename for filename in os.listdir('image/deco') if
                   filename.startswith(str(levelD[level]['deco'][deco - 1]) + '.')]
        if (len(files) > 1) or (len(files_d) > 1):
            for k in range(max(len(files) - 1, len(files_d) - 1)):
                if (len(files) > 1):
                    Im_t = pygame.image.load(self.text[:-4] + '.' + str(min(k, len(files) - 2) + 1) + '.png')
                else:
                    Im_t = pygame.image.load(text[0])
                Imdeco = Im_t
                if deco != 0:
                    freq2 = int(0.5 + np.linalg.norm(self.a[0][0]) / 10)
                    if levelD[level]['deco'][deco - 1] in z_tileable_deco:
                        freq2 = 1
                    Imdeco = pygame.transform.scale(Imdeco, (120 * freq, 120 * freq2))
                    for i in range(freq):
                        for j in range(freq2):
                            Imdeco.blit(Im_t, (120 * i, 120 * j))
                    deco_name = 'image/deco/' + str(levelD[level]['deco'][deco - 1]) + '.png'
                    if (len(files_d) > 1):
                        deco_name = 'image/deco/' + str(levelD[level]['deco'][deco - 1]) + '.' + str(
                            min(k, len(files_d) - 2) + 1) + '.png'
                    Imdeco.blit(pygame.image.load(deco_name), (120 * (freq - 1 - phase), 120 * (freq2 - 1)))
                if door > 1 and door < 5:
                    verrou = pygame.image.load('image/deco/verrou.png')
                    if door == 2:
                        colo = (255, 50, 50)
                    if door == 3:
                        colo = (50, 255, 50)
                    if door == 4:
                        colo = (50, 50, 255)
                    verrou.fill(colo, special_flags=BLEND_RGB_MULT)
                    Imdeco.blit(verrou, (0, 0))

                IM = np.transpose(pygame.surfarray.pixels3d(Imdeco), (1, 0, 2))
                IM = np.where(np.expand_dims(((IM - np.array([255, 0, 255])) == 0).all(-1), -1), -1, IM)
                IM = np.where(np.expand_dims(((IM - np.array([0, 255, 255])) == 0).all(-1), -1), -2, IM)
                self.window = np.sum(IM <= -1)
                self.wall_im.append(np.flip(np.minimum(IM + 1, 255), (0, 1)))

        files2 = [filename for filename in os.listdir(self.text2[:11]) if filename.startswith(self.text2[11:-3])]
        if (len(files2) > 1) or (len(files_d) > 1):
            for k in range(max(len(files2) - 1, len(files_d) - 1)):
                if (len(files2) > 1):
                    Im_t = pygame.image.load(self.text2[:-4] + '.' + str(min(k, len(files) - 2) + 1) + '.png')
                else:
                    Im_t = pygame.image.load(text[1])
                Imdeco2 = Im_t
                if deco != 0:
                    freq2 = int(0.5 + np.linalg.norm(self.a[0][0]) / 10)
                    if levelD[level]['deco'][deco - 1] in z_tileable_deco:
                        freq2 = 1
                    Imdeco2 = pygame.transform.scale(Imdeco2, (120 * freq, 120 * freq2))
                    for i in range(freq):
                        for j in range(freq2):
                            Imdeco2.blit(Im_t, (120 * i, 120 * j))
                    deco_name = 'image/deco/' + str(levelD[level]['deco'][deco - 1]) + '.png'
                    if (len(files_d) > 1):
                        deco_name = 'image/deco/' + str(levelD[level]['deco'][deco - 1]) + '.' + str(
                            min(k, len(files_d) - 2) + 1) + '.png'
                    Imdeco2.blit(pygame.image.load(deco_name), (120 * (freq - 1 - phase), 120 * (freq2 - 1)))
                if door > 1 and door < 5:
                    verrou = pygame.image.load('image/deco/verrou.png')
                    if door == 2:
                        colo = (255, 50, 50)
                    if door == 3:
                        colo = (50, 255, 50)
                    if door == 4:
                        colo = (50, 50, 255)
                    verrou.fill(colo, special_flags=BLEND_RGB_MULT)
                    Imdeco2.blit(verrou, (0, 0))

                IM2 = np.transpose(pygame.surfarray.pixels3d(Imdeco2), (1, 0, 2))
                IM2 = np.where(np.expand_dims(((IM2 - np.array([255, 0, 255])) == 0).all(-1), -1), -1, IM2)
                IM2 = np.where(np.expand_dims(((IM2 - np.array([0, 255, 255])) == 0).all(-1), -1), -2, IM2)
                self.wall_im2.append(np.flip(np.minimum(IM2 + 1, 255), (0, 1)))

        self.U = np.array([0])
        self.z = w[-1]
        self.ID = str(int((self.X[0][0][0] + 0.5 * self.b[0][0][0]) // 2 + 50)) + ',' + str(
            int((self.X[0][0][1] + 0.5 * self.b[0][0][1]) // 2 + 50))
        self.transp = 0
        if self.a[0][0][-1] == 0 or self.b[0][0][-1] == 0:
            self.ID = str(int((self.X[0][0][0] + 0.5 * self.b[0][0][0] + 0.5 * self.a[0][0][0]) // 2 + 50)) + ',' + str(
                int((self.X[0][0][1] + 0.5 * self.b[0][0][1] + 0.5 * self.a[0][0][1]) // 2 + 50))
        self.TTT0 = []
        self.TTT1 = []
        self.TTT2 = []
        if self.text[11:-3] in liquid_floor:
            self.a = self.a * 1.5
            self.b = self.b * 1.5
            self.X = self.X - 0.25 * self.a - 0.25 * self.b

        if self.text[11:-3] in transp_floor:
            self.transp=1

    def opendoor(self, door):
        Imdeco = pygame.image.load(self.text)
        verrou = self.verrou
        if door == 2:
            colo = (255, 50, 50)
        if door == 3:
            colo = (50, 255, 50)
        if door == 4:
            colo = (50, 50, 255)
        verrou.fill(colo, special_flags=BLEND_RGB_MULT)
        Imdeco.blit(verrou, (0, 0))
        IM = np.transpose(pygame.surfarray.pixels3d(Imdeco), (1, 0, 2))
        IM = np.where(np.expand_dims(((IM - np.array([255, 0, 255])) == 0).all(-1), -1), -1, IM)
        IM = np.where(np.expand_dims(((IM - np.array([0, 255, 255])) == 0).all(-1), -1), -2, IM)
        self.window = np.sum(IM <= -1)
        self.wall_im[0] = np.flip(np.minimum(IM + 1, 255), (0, 1))
        return 0

    def texture(self, sc1, sc2):

        self.borne = [self.wall_im[0].shape[0] - 1, self.wall_im[0].shape[1] - 1]
        if self.door != 0:
            self.format = 120 * np.array([1, 1])
        else:
            self.format = 120 * np.array([np.linalg.norm(self.a[0][0]) / 10, np.linalg.norm(self.b[0][0]) / 10])
        if self.text == 'image/wall/wall29.png':
            self.format = 120 * np.array([1, np.linalg.norm(self.b[0][0]) / 30])

    def calc_norm(self):
        global moving_cam
        if len(self.wall_im)>1 or len(self.wall_im2)>1:
            moving_cam=True
        N = np.stack((self.a[0][0], self.b[0][0], -self.n), axis=-1)
        V = np.maximum(np.minimum(np.linalg.solve(N, -self.X[0][0] + R_c)[:-1], 1), 0)
        self.norm = np.linalg.norm(
            self.X[0][0][:-1] + V[0] * self.a[0][0][:-1] + V[1] * self.b[0][0][:-1] - R_c[:-1]) - min(self.window,
                                                                                                      1) * 0.001
        self.norm3 = np.linalg.norm(self.X[0][0] + V[0] * self.a[0][0] + V[1] * self.b[0][0] - R_c)

    def calc_normfast(self):
        self.normf = np.linalg.norm(self.X[0][0][:-1] + 0.5 * self.a[0][0][:-1] + 0.5 * self.b[0][0][:-1] - R_c[:-1])

    def normal(self, trans):

        t = -trans
        No = self.n[:-1]
        No = No / np.linalg.norm(No)

        # rap = [-t[0] * No[1] + t[1] * No[0], -t[0] * No[0] - t[1] * No[1]]
        # rap0 = [-(cos(-ang[0]) * No[1] + sin(-ang[0]) * No[0]), (-cos(-ang[0]) * No[0] - sin(-ang[0]) * No[1])]
        #
        # Sang=np.arctan2(rap[0], rap[1])
        # Sang0 = np.arctan2(rap0[0], rap0[1])
        # print(Sang,Sang0,t,No)

        x_ = self.X[0][0][0]
        y_ = self.X[0][0][1]
        a_ = self.b[0][0][0]
        b_ = self.b[0][0][1]
        self.side = b_ * R_c[0] - a_ * R_c[1] + a_ * y_ - b_ * x_
        if self.side < 0:
            return (-t - abs(np.dot(-t, No)) * No)
        else:
            return (-t - abs(np.dot(-t, -No)) * -No)

    def test_behind(self):

        # if self.door==0 and self.window<=0:# ou bien transparent
        # 	# if abs(np.arctan2(self.n[1],self.n[0])+ang[0]%(2*pi))<np.arctan(TAN2):# truc bizarre ici mettre des angles mieux
        # 	# 	return Falseve
        # 	No = self.n[:-1]
        # 	No = -No / np.linalg.norm(No)
        # 	rap=[(cos(-ang[0])*No[1]-sin(-ang[0])*No[0]),(cos(-ang[0])*No[0]+sin(-ang[0])*No[1])]
        # 	if abs(np.arctan2(rap[0],rap[1]))>0.5*pi+np.arctan(TAN2):# truc bizarre ici mettre des angles mieux
        # 		return False

        Mg = np.stack((self.b[0][0][0:-1], -Vg @ Rp), axis=-1)
        self.Ig = np.linalg.solve(Mg, x - self.X[0][0][0:-1])

        Md = np.stack((self.b[0][0][0:-1], -Vd @ Rp), axis=-1)
        self.Id = np.linalg.solve(Md, x - self.X[0][0][0:-1])

        Rp0 = rot_plan(-ang[0])
        Xa = (self.X[0][0][0:-1] - x) @ Rp0
        Xb = (self.X[0][0][0:-1] + self.b[0][0][0:-1] - x) @ Rp0

        if ((Xa[0] > 1 and Xb[0] > 1) or (self.Id[0] < 1 and self.Id[0] > 0 and self.Id[1] > 1) or (
                self.Ig[0] < 1 and self.Ig[0] > 0 and self.Ig[1] > 1)) or self.door != 0:

            if self.door != 0 and self.closed == 0:
                return True
            else:

                global horizon

                B0 = self.b[:, 0, :-1]
                X0 = self.X[:, 0, :-1]  # +5*np.random.random(self.X[:,0,:-1].shape)
                self.M0 = np.stack((B0, -screen[:, 24, 3:-1] @ Rp), axis=-1)  # -------------- PB à résoudre pièce haute
                self.A0 = -X0 + (screen[:, 24, :2] - x) @ Rp + x
                self.S0 = (np.linalg.solve(self.M0, self.A0))
                self.U0 = np.where(
                    np.all(self.S0 <= horizon, axis=-1) & np.all(self.S0 > 0, axis=-1) & np.all(self.S0[:, :-1] < 1,
                                                                                                axis=-1), 1, 0)

                if self.window == 0:
                    horizon = self.S0[:, -1:] * np.expand_dims(self.U0, -1) + horizon * (
                                1 - np.expand_dims(self.U0, -1))

                if np.sum(self.U0) > 0:
                    return True
                else:
                    return False

        else:
            # print('oh')
            return False

    def breakable(self):
        global x_d
        self.inline=0
        self.shot = []
        self.d_shot=100
        x_d0=[]
        for i in range(len(x_d)):
            inline = min(np.sum(self.U[int(scrnL[0]*(1+2*x_d[i][0]) - gun_width):int(scrnL[0]*(1+2*x_d[i][0]) + gun_width), int(2 * scrnL[1]*(1+2*x_d[i][1]) // 2 - gun_width):int(2 * scrnL[1]*(1+2*x_d[i][1]) // 2 + gun_width)]),
                            1)
            self.inline=(self.inline or inline)
            self.shot.append(i)
            if not inline:
                x_d0.append(x_d[i])
            else:
                depth_ = depth.shape
                self.d_shot=min(self.d_shot,depth_cached[int(depth_[0] * (x_d[0][0] + 0.5))][int(depth_[1] * (x_d[0][1] + 0.5))])
        x_d=x_d0
        return self.inline

    def render(self):
        global depth, POS, Sky_view,explo_R
        self.time=(0,0)
        milliseconds=[time.time() * 1000]
        label_m=[]
        self.M = np.stack((self.a[::2, ::2, :] * 1.01, self.b[::2, ::2, :] * 1.01, -screenV[::2, ::2, :]), axis=-1)
        self.B = -self.X[::2, ::2, :] + screenP[::2, ::2, :]

        # self.S=(np.linalg.solve(self.M.astype(np.float32),self.B.astype(np.float32)))
        inv_M = np.linalg.inv(self.M.astype(np.float32))
        self.S = np.einsum('...ij,...j->...i', inv_M, self.B.astype(np.float32))
        milliseconds.append(time.time() * 1000)
        label_m.append('solving')

        # self.M=torch.tensor(np.stack((self.a[:,:,:]*1.01,self.b[:,:,:]*1.01,-screenV[:,:,:]),axis=-1))
        # self.B=torch.tensor(-self.X[:,:,:]+screenP[:,:,:])
        # self.S=(torch.linalg.solve(self.M.to('cuda'),self.B.to('cuda'))).to('cpu').numpy()

        for i in range(2):
            self.S = self.S.repeat(2, axis=0).repeat(2, axis=1)
            self.S[:, :, :-1] = (np.roll(self.S[:, :, :-1], 1, axis=0) + np.roll(self.S[:, :, :-1], 1,
                                                                                 axis=1) + np.roll(self.S[:, :, :-1],
                                                                                                   -1,
                                                                                                   axis=1) + np.roll(
                self.S[:, :, :-1], -1, axis=0)) / 4
            AbsS = np.absolute(self.S[:, :, -1])
            self.S[:, :, -1] = (np.sign(self.S[:, :, -1]) * (
                        np.roll(AbsS, 1, axis=0) + np.roll(AbsS, 1, axis=1) + np.roll(AbsS, -1, axis=1) + np.roll(AbsS,
                                                                                                                  -1,
                                                                                                                  axis=0)) / 4)
        milliseconds.append(time.time() * 1000)
        label_m.append('upscale')
        self.U = np.all(self.S <= depth, axis=-1) & np.all(self.S > 0, axis=-1) & np.all(self.S[:, :, :-1] < 1, axis=-1)
        if (shoot == 1 or self.explo) and levelD[level]['deco'][self.deco - 1] in deco_destruc and self.deco!=0:
            if self.breakable() or self.explo:
                if ((self.d_shot<5 or arme!=0) and arme!=4) or self.explo:
                    self.vie+=1
                    s = pygame.mixer.Sound("son/barril.ogg")
                    s.play()
                if self.vie==3:
                    self.X=self.X*0
            #print(self.breakable(),self.text,self.deco,levelD[level]['deco'][self.deco - 1])
        milliseconds.append(time.time() * 1000)
        label_m.append('contour')

        if np.sum(self.U) < 5:
            return np.full((2 * scrnL[0], 2 * scrnL[1], 1), 0)
        else:
            Ue = np.expand_dims(self.U, -1)
            if self.transp:
                Sky_view = 1
                return -1 * np.dstack((Ue, Ue, Ue))

            self.G = np.mod(
                np.maximum(((1 - self.S[:, :, :-1]) * self.format).astype(int), 0) + int(self.phase * c3 * 0.5),
                self.borne)

            milliseconds.append(time.time() * 1000)
            label_m.append('indexing')

            colorL = [1, 1, 1]
            if self.ID in light_color.keys():
                colorL = np.round(np.maximum(np.array(light_color[self.ID]), 0.1), 2)

            colorL = light_modif(colorL, level, c3)

            x_ = self.X[0][0][0]
            y_ = self.X[0][0][1]
            a_ = self.b[0][0][0]
            b_ = self.b[0][0][1]
            side = b_ * R_c[0] - a_ * R_c[1] + a_ * y_ - b_ * x_
            if levelD[level]['deco'][self.deco - 1] not in deco_destruc:

                if side < 0:
                    ind = c // (12 // len(self.wall_im))
                    # Ar=np.moveaxis(self.wall_im[ind][tuple(map(tuple,self.G.T))]*colorL,1,0)*np.dstack((self.U,self.U,self.U))
                    Ar = np.where(np.stack([self.U] * 3, axis=-1),
                                  np.moveaxis(self.wall_im[ind][tuple(map(tuple, self.G.T))] * colorL, 1, 0), 0)
                else:
                    ind2 = c // (12 // len(self.wall_im2))  # 2
                    # Ar=np.moveaxis(self.wall_im2[ind2][tuple(map(tuple,self.G.T))]*colorL,1,0)*np.dstack((self.U,self.U,self.U))
                    Ar = np.where(np.stack([self.U] * 3, axis=-1),
                                  np.moveaxis(self.wall_im2[ind2][tuple(map(tuple, self.G.T))] * colorL, 1, 0), 0)
            else:
                Ar = np.where(np.stack([self.U] * 3, axis=-1),
                              np.moveaxis(self.wall_im[min(self.vie,2)][tuple(map(tuple, self.G.T))] * colorL, 1, 0), 0)
            milliseconds.append(time.time() * 1000)
            label_m.append('render')

            filt = 1 - np.expand_dims((Ar == 0).all(2), -1)
            if self.text[11:-3] not in liquid_floor:
                depth = np.where(Ue * (filt), self.S[:, :, -1:], depth)
            self.U=self.U*filt[:,:,0]
            Xl = ((np.expand_dims(self.S[::2, ::2, 0], -1) * self.a*1.01 + np.expand_dims(self.S[::2, ::2, 1],
                                                                                     -1) * self.b*1.01 + self.X).repeat(2,
                                                                                                                   axis=0).repeat(
                2, axis=1)) * Ue

            POS = (POS * (1 - self.U)).astype(np.float32)
            D = np.full((2 * scrnL[0], 2 * scrnL[1]), 1000.0).astype(np.float32)
            if self.ID in light_wall.keys():
                Y0 = [np.linalg.norm(source_pos(i) - R_c) for i in light_wall[self.ID]]
                X0 = [x for _, x in sorted(zip(Y0, light_wall[self.ID]))]
                for i in X0[:min(len(X0), 4)]:
                    Xsource = source_pos(i)#+np.array([10*sin(c3/100),10*sin(c3/100),0.])
                    D = np.minimum(D, np.linalg.norm(Xl - Xsource, axis=-1).astype(np.float32) * self.U)
                POS = POS + D
                Ar = Ar * level_light
            else:
                D = np.minimum(D, (np.linalg.norm(Xl - R_c, axis=-1) ** 0.5).astype(np.float32) * self.U)
                POS = POS + D
                Ar = Ar * torch_on * TORCHE ** 3
            if explo!=0:
                explo_R=np.minimum(explo_R,np.expand_dims(np.linalg.norm(Xl - np.array([explo_pt[0],explo_pt[1],0.]), axis=-1)*self.U+100*(1-self.U),-1))
            if explo==4:
                self.explo=np.sum(explo_R<20)
            else:
                self.explo=0


            milliseconds.append(time.time() * 1000)
            label_m.append('distance and position light')
            self.time=((np.array(milliseconds) - np.roll(np.array(milliseconds), 1))[1:],label_m)

            return Ar

from numba import njit,prange
from numba.typed import List
@njit(inline='always')
def solve3x3(M, B):
    # determinant of M
    det = (
        M[0,0]*(M[1,1]*M[2,2] - M[1,2]*M[2,1])
      - M[0,1]*(M[1,0]*M[2,2] - M[1,2]*M[2,0])
      + M[0,2]*(M[1,0]*M[2,1] - M[1,1]*M[2,0])
    )

    if abs(det) < 1e-12:
        return np.zeros(3)  # nearly singular

    inv_det = 1.0 / det

    # Cramer's rule — replace columns with B
    dx = (
        B[0]*(M[1,1]*M[2,2] - M[1,2]*M[2,1])
      - M[0,1]*(B[1]*M[2,2] - M[1,2]*B[2])
      + M[0,2]*(B[1]*M[2,1] - M[1,1]*B[2])
    )

    dy = (
        M[0,0]*(B[1]*M[2,2] - M[1,2]*B[2])
      - B[0]*(M[1,0]*M[2,2] - M[1,2]*M[2,0])
      + M[0,2]*(M[1,0]*B[2] - B[1]*M[2,0])
    )

    dz = (
        M[0,0]*(M[1,1]*B[2] - B[1]*M[2,1])
      - M[0,1]*(M[1,0]*B[2] - B[1]*M[2,0])
      + B[0]*(M[1,0]*M[2,1] - M[1,1]*M[2,0])
    )

    return np.array([dx*inv_det, dy*inv_det, dz*inv_det])
@njit(fastmath=True)
def norm_last_axis(diff):
    n = np.empty(diff.shape[0], dtype=np.float64)
    for i in range(diff.shape[0]):
            v = diff[i]
            n[i] = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) ** 0.5
    return n


from numba import set_num_threads, get_num_threads



# Control Numba's threading
set_num_threads(4)  # or whatever number fits your CPU
print("Using", get_num_threads(), "threads for Numba parallel loops.")
@njit(fastmath=True, cache=True,parallel=True)
def render_numba(screenX, screenY, wallN, wall_a, wall_b, wall_x, screenV, screenP,formatL,phaseL,borneL,c3,wall_imL,lightL):
    S_out = np.zeros((2*screenX, 2*screenY,3))
    depth0=np.zeros((2*screenX, 2*screenY))+300
    D = np.zeros((2 * screenX, 2 * screenY))

    for i in prange(2*screenX):  # Only outer loop is parallel
        for j in range(2*screenY):  # Sequential inner loop
            M = np.empty((3, 3))
            for k in range(wallN):
                #S=(S_in[k][(i+1)//2][(j+1)//2]+S_in[k][(i)//2][(j)//2]+S_in[k][(i)//2][(j+1)//2]+S_in[k][(i+1)//2][(j)//2])/4
                #if i%2==0 and j%2==0:

                M[:, 0] = wall_a[k] * 1.01
                M[:, 1] = wall_b[k] * 1.01
                M[:, 2] = -screenV[i][j]

                B = (-wall_x[k] + screenP[i][j])
                #S=np.linalg.solve(M,B)
                S = solve3x3(M,B)
                U =  np.all(S>0) & (S[0]<1) & (S[1]<1)

                if U and S[2]<depth0[i][j]:
                    depth0[i][j]=min(depth0[i][j],S[2])

                    G = np.mod(np.maximum(((1 - S[:-1]) * formatL[k]).astype(np.int32), 0) + int(phaseL[k] * c3 * 0.5),
                        borneL[k])
                    Ar = wall_imL[k][0][G[0],G[1]]
                    xl=S[0]*wall_a[k] * 1.01+S[1]*wall_b[k] * 1.01+wall_x[k]
                    diff=lightL[k]-xl

                    D[i][j]=min(norm_last_axis(diff))



            S_out[i, j] = Ar

    return S_out,D,depth0

@njit(inline='always')
def intersect_ray_plane(O, D, P0, A, B):
    """Return (t,u,v) for ray-plane intersection, or (-1,-1,-1) if no hit."""
    # normal = A × B
    Nx = A[1]*B[2] - A[2]*B[1]
    Ny = A[2]*B[0] - A[0]*B[2]
    Nz = A[0]*B[1] - A[1]*B[0]

    denom = Nx*D[0] + Ny*D[1] + Nz*D[2]
    if abs(denom) < 1e-9:
        return -1.0, -1.0, -1.0  # parallel

    # distance along ray
    t = ((P0[0]-O[0])*Nx + (P0[1]-O[1])*Ny + (P0[2]-O[2])*Nz) / denom
    if t <= 0:
        return -1.0, -1.0, -1.0  # behind camera

    # hit point
    Px = O[0] + t*D[0] - P0[0]
    Py = O[1] + t*D[1] - P0[1]
    Pz = O[2] + t*D[2] - P0[2]

    # project to wall local coords (u,v)
    detAB = A[0]*B[1] - A[1]*B[0]
    if abs(detAB) < 1e-9:
        return -1.0, -1.0, -1.0

    # Approximate u,v projection (2D in wall plane)
    u = (Px*B[1] - Py*B[0]) / detAB
    v = (Py*A[0] - Px*A[1]) / (-detAB)

    return t, u, v
@njit(parallel=True, fastmath=True)
def render_numba_ray(screenX, screenY, wallN,
                 wall_a, wall_b, wall_x,
                 screenV, screenP,
                 formatL, phaseL, borneL, c3, wall_imL):

    S_out = np.zeros((2*screenX, 2*screenY, 3))
    depth0 = np.full((2*screenX, 2*screenY), 300.0)

    cam = np.zeros(3)  # assuming camera at origin (adjust if not)

    for i in prange(2*screenX):
        for j in range(2*screenY):
            Ar = np.zeros(3)
            D = screenV[i//2, j//2]    # ray direction
            O = screenP[i//2, j//2]    # ray origin (camera or eye point)

            for k in range(wallN):
                A = wall_a[k]
                B = wall_b[k]
                P0 = wall_x[k]

                t, u, v = intersect_ray_plane(O, D, P0, A, B)

                if (t > 0 and 0 < u < 1 and 0 < v < 1 and t < depth0[i, j]):
                    depth0[i, j] = t

                    G0 = int((1 - u) * formatL[k][0])
                    G1 = int((1 - v) * formatL[k][1])
                    if G0 < 0: G0 = 0
                    if G1 < 0: G1 = 0
                    G0 = (G0 + int(phaseL[k] * c3 * 0.5)) % borneL[k][0]
                    G1 = (G1 + int(phaseL[k] * c3 * 0.5)) % borneL[k][1]

                    Ar = wall_imL[k][0][G0, G1].astype(np.float64)

            S_out[i, j] = Ar

    return S_out





Xthing, Ything = np.indices((2 * scrnL[0], 2 * scrnL[1]))
RA = 5

Boule = []


class Thing():
    def __init__(self, x0, type_M, vivant, group):
        self.x0 = x0
        self.norm = np.linalg.norm(self.x0 - x + [1, 0])
        self.width = 2 * scrnL[0] / self.norm
        self.widthY = 2 * scrnL[0] / self.norm
        self.DX = 0
        self.DY = 0
        self.type_M = levelD[level]['mon'][type_M]
        self.angle = 0
        self.orient = 2 * pi * random()
        self.im = np.minimum(
            pygame.surfarray.pixels3d(MD[self.type_M][1][45 * (((-ang[0] + pi / 8 + self.orient) // (pi / 4)) % 8)]),
            255)
        self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)
        self.borne = [self.im.shape[0] - 1, self.im.shape[1] - 1]
        self.z = zmap[int(self.x0[1] + 100) // 2][int(self.x0[0] + 100) // 2]
        self.active = 0
        self.inline = 0
        self.vie = M_charac['life'][self.type_M]
        self.track = []
        self.mort = 0
        self.vivant = vivant
        self.range = M_charac['dist'][self.type_M]
        self.attack_range = False
        self.group = group
        self.f0 = (self.x0 - x) @ rot_plan(-ang[0])
        self.U = 0
        self.RA = levelD[level]['RAmon'][type_M] * (1 + (random() * 0.2 - 0.1))
        if self.type_M==6:
            self.shield = 0
        else:
            self.shield = 1

    def calc_norm(self):

        self.f0 = (self.x0 - x) @ rot_plan(-ang[0])
        self.norm = np.linalg.norm(self.x0 - x)
        self.width = self.RA * 2 * scrnL[0] / self.f0[0]
        self.widthY = self.RA * 2 * scrnL[0] / self.f0[0]
        self.DX = -self.RA * scrnL[0] / self.f0[0] + scrnL[0] * (self.f0[1] / self.f0[0]) / TAN2
        self.DY = -(self.RA * scrnL[0] * 1.7 - 5 * scrnL[0]) / self.f0[0] - scrnL[1] * tan(ang[1]) / TAN1 - (
                    2 * scrnL[1] * (z - self.z) / self.f0[
                0]) / TAN1  # WARNING DO THE SAME FOR OBJECT AND BOSS + BOULE FOR BOSS
        A = 45 * (((atan((self.f0[1] / self.f0[0])) + self.orient - pi / 2 - ang[0] + pi / 8) // (pi / 4)) % 8)
        if A != self.angle:
            self.angle = A
            self.im = np.minimum(pygame.surfarray.pixels3d(MD[self.type_M][c // 3][self.angle]), 255)
            self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)

    def walk(self):
        self.im = np.minimum(pygame.surfarray.pixels3d(MD[self.type_M][c // 3][self.angle]), 255)
        self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)
        if self.type_M == 6 and self.shield == 0:
            im=MD[self.type_M][c // 3][self.angle].copy()
            im.blit(shield,(0,0))
            self.im = np.minimum(pygame.surfarray.pixels3d(im), 255)
            self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)

    def test_behind(self):
        if self.f0[0] > 0 and abs(self.f0[1] / self.f0[0]) < TAN2 + 0.5:
            if self.norm <= horizon[min(max(int(self.width // 2 + self.DX + scrnL[0]) // 2, 0), scrnL[0] - 1)]:
                return True
            else:
                return False
        else:
            return False

    def move(self):
        global VIE, HIT
        self.explo_zone()
        self.attack_range = self.norm <= 5
        if self.range == 1:
            self.attack_range = (self.test_behind()) & (self.norm <= 20)

        if not (self.attack_range) and self.active and self.vie > 0:

            X0 = nearest_valid(authorized_map, x)
            if len(self.track) > 0:
                if ((X0[0] - self.track[-1][1]) ** 2 + (X0[1] - self.track[-1][0]) ** 2) ** 0.5 > 0:
                    self.track = astar(authorized_map, (int(self.x0[1] + 101) // 2, int(self.x0[0] + 101) // 2),
                                       (X0[1], X0[0]))
            if len(self.track) > 2:
                dest = [self.track[2][1], self.track[2][0]]
                self.orient = -np.angle((dest[1] - int(self.x0[1] + 101) // 2) * 1j + (
                            1e-3 + dest[0] - int(self.x0[0] + 101) // 2)) - pi / 2 + (random() - 0.5) * pi / 5
                if np.linalg.norm(np.array(dest) - np.array([(self.x0[0] + 101) / 2, (self.x0[1] + 101) / 2])) < sqrt(
                        2) + 0.2:
                    self.track.pop(0)
                V = [+0.1 + M_charac['speed'][self.type_M] * self.active, 0] @ rot_plan(self.orient + pi / 2)
                self.x0 = self.x0 + V
            else:
                self.active = 0
        if not (self.attack_range) and self.active == 0 and self.vie > 0:
            V = [-0.1, 0] @ rot_plan(self.orient - pi / 2)
            if level_map[int((self.x0 + 7 * V)[1] + 101) // 2][int((self.x0 + 7 * V)[0] + 101) // 2] and \
                    authorized_map[int((self.x0)[1] + 101) // 2][
                        int((self.x0)[0] + 101) // 2] == 2 and self.active == 0:
                self.orient = self.orient + pi
            self.x0 = self.x0 + V

        if self.norm > 50 and self.active and self.vie > 0:
            self.active = 0
            self.orient = random() * pi
        self.z = zmap[int(self.x0[1] + 100) // 2][int(self.x0[0] + 100) // 2]
        spot = 1
        if arme in [0, 4]:
            spot = 0
        if self.norm < 20 + min(shoot, 1) * 20 * spot and self.active == 0 and self.vie > 0:
            self.active = 1
            s = pygame.mixer.Sound("son/grognespot%s.ogg" % (self.type_M + 1))
            s.play()
            X0 = nearest_valid(authorized_map, x)
            self.track = astar(authorized_map, (int(self.x0[1] + 101) // 2, int(self.x0[0] + 101) // 2), (X0[1], X0[0]))

        if self.attack_range and self.active and self.vie > 0:
            if self.range == 0:
                if c // 4 == 2 and c % 3 == 0:
                    HIT = 1

                    VIE -= M_charac['degat'][self.type_M]
                    s = pygame.mixer.Sound("son/aie.ogg")
                    s.play()
            else:
                if c2 // 8 == 2 and c2 % 6 == 0:
                    A00 = atan((self.x0[1] - x[1]) / (self.x0[0] - x[0]))
                    if self.x0[0] - x[0] > 0:
                        A00 += pi
                    A11 = atan((self.z - z) / ((self.x0[0] - x[0]) ** 2 + (self.x0[1] - x[1]) ** 2) ** 0.5)
                    if self.type_M!=6:
                        Boule.append(
                            boule(self.x0[0], self.x0[1], self.z, A11 + pi / 2, A00, 0.6, 0, 'image/effects/hurt.png',
                                  M_charac['degat'][self.type_M]))
                    else:
                        Boule.append(
                            boule(self.x0[0], self.x0[1], self.z, A11 + pi / 2, A00, 1., 0, 'image/effects/hurt3.png',
                                  M_charac['degat'][self.type_M]))

    def explo_zone(self):
        if (np.linalg.norm(self.x0 - explo_pt) < 20 and explo == 1) and self.vie > 0:
            self.vie -= DEGAT[4]*self.shield
            for i in range(randint(5, 10)):
                if self.type_M<=4:
                    Boule.append(boule(self.x0[0], self.x0[1], self.z, pi * random(), 2 * pi * random(), 0.1 * random(), 2,
                                       'image/effects/vert.png', 0))
                else:
                    Boule.append(
                        boule(self.x0[0], self.x0[1], self.z, pi * random(), 2 * pi * random(), 0.1 * random(), 2,
                              'image/effects/rouge.png', 0))
            if self.active == 0:
                self.active = 1
                X0 = nearest_valid(authorized_map, x)
                self.track = astar(authorized_map, (int(self.x0[1] + 101) // 2, int(self.x0[0] + 101) // 2),
                                   (X0[1], X0[0]))
                s = pygame.mixer.Sound("son/grognespot%s.ogg" % (self.type_M + 1))
                s.play()

    def render(self):
        global Killed_E,x_d

        if self.attack_range and self.active and self.vie > 0:
            if self.range == 0:
                self.im = np.minimum(pygame.surfarray.pixels3d(MDa[self.type_M][c // 4]), 255)
            else:
                self.im = np.minimum(pygame.surfarray.pixels3d(MDa[self.type_M][c2 // 8]), 255)

            self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)

            if self.type_M==6 and self.shield==0:
                if self.range == 0:
                    im=MDa[self.type_M][c // 4].copy()
                else:
                    im = MDa[self.type_M][c2 // 8].copy()
                im.blit(shield,(0,0))
                self.im = np.minimum(pygame.surfarray.pixels3d(im), 255)
                self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)

        if (self.inline and shoot == 2 and (self.attack_range or arme != 0)) and self.vie > 0 and arme != 4:
            self.im = np.minimum(pygame.surfarray.pixels3d(MDa[self.type_M][4]), 255)
            if self.type_M==6 and self.shield==0:
                im=MDa[self.type_M][4].copy()
                im.blit(shield,(0,0))
                self.im = np.minimum(pygame.surfarray.pixels3d(im), 255)

            self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)
            if shoot:
                if arme==2:
                    nb_bullets=5
                else:
                    nb_bullets = 1
                self.vie -= self.shield*DEGAT[arme]*len(self.shot)/nb_bullets
                if self.type_M == 6 and self.shield==0:
                    self.shield=1
                    s = pygame.mixer.Sound("son/shield.ogg")
                    s.play()
                else:
                    for i in range(randint(5, 10)):
                        if self.type_M <= 4:
                            Boule.append(
                                boule(self.x0[0], self.x0[1], self.z, pi * random(), 2 * pi * random(), 0.1 * random(), 2,
                                      'image/effects/vert.png', 0))
                        else:
                            Boule.append(
                                boule(self.x0[0], self.x0[1], self.z, pi * random(), 2 * pi * random(), 0.1 * random(), 2,
                                      'image/effects/rouge.png', 0))
                if arme != 0:
                    Accuracy[0] = Accuracy[0] + 1

            if self.active == 0:
                self.active = 1
                X0 = nearest_valid(authorized_map, x)
                self.track = astar(authorized_map, (int(self.x0[1] + 101) // 2, int(self.x0[0] + 101) // 2),
                                   (X0[1], X0[0]))
                s = pygame.mixer.Sound("son/grognespot%s.ogg" % (self.type_M + 1))
                s.play()

        if self.vie <= 0:
            self.im = np.minimum(pygame.surfarray.pixels3d(MDa[self.type_M][4 + int(self.mort)]), 255)
            self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)
            if self.mort < 4:
                self.mort = min(self.mort + 0.5, 3)
            if self.mort == 0.5:
                s = pygame.mixer.Sound("son/grognemort%s.ogg" % (self.type_M + 1))
                Killed_E[0] = Killed_E[0] + 1
                s.play()

            if self.mort == 2.5 and self.type_M == 4:
                global thing, ennemies
                for i in range(randint(1, 3)):
                    monst = Thing(self.x0 + 5 * (np.random.random(2) - 0.5), 0, 1, 0)
                    monst.RA = monst.RA * 0.75
                    thing.append(monst)
                    ennemies.append(monst)
                    Killed_E[1] = Killed_E[1] + 1
                for i in range(randint(50, 60)):
                    Boule.append(boule(self.x0[0], self.x0[1], self.z - 1, -pi * random(), 2 * pi * random(), 1, 5,
                                       'image/effects/vert.png', 1))

        colorT = light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2]
        if light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2].sum() == 0:
            colorT = np.array([1, 1, 1.])
        colorT = light_modif(colorT, level, c3)
        if explo!=0 and np.linalg.norm(self.x0 - explo_pt) < 20:
            colorT*=np.array([1,0,0])
        SQUARE = np.all(self.norm <= depth, axis=-1) & (Xthing <= self.width + self.DX + scrnL[0]) & (
                    Xthing >= self.DX + scrnL[0]) & (Ything <= self.widthY + self.DY + scrnL[1]) & (
                             Ything >= self.DY + scrnL[1])

        self.U = np.stack(((Xthing - self.DX - scrnL[0]) / self.width, (Ything - self.DY - scrnL[1]) / self.widthY),
                          -1) * np.expand_dims(SQUARE, -1)
        self.G = np.maximum((self.U * 160).astype(int), 0)
        Ar = np.moveaxis(self.im[tuple(map(tuple, self.G.T))] * colorT, 1, 0)
        if light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2].sum() == 0:
            Ar = Ar * torch_on * TORCHE ** 3
            Ar = np.minimum(np.divide(Ar, 0.1 * self.norm ** 0.5), 255)
        else:
            Ar = Ar * level_light
        self.Ut = np.moveaxis(self.vis[tuple(map(tuple, self.G.T))], 1, 0)

        if shoot==1:
            self.inline=False
            self.shot = []
            x_d0=[]
            for i in range(len(x_d)):
                inline = min(np.sum(self.Ut[int(scrnL[0]*(1+2*x_d[i][0]) - gun_width):int(scrnL[0]*(1+2*x_d[i][0]) + gun_width), int(2 * scrnL[1]*(1+2*x_d[i][1]) // 2 - gun_width):int(2 * scrnL[1]*(1+2*x_d[i][1]) // 2 + gun_width)]),
                                1)
                self.inline=(self.inline or inline)
                self.shot.append(i)
                if not inline:
                    x_d0.append(x_d[i])
            x_d=x_d0
        #print(x_d,y_d,np.sum(self.Ut[int(scrnL[0]*(1+2*x_d) - gun_width):int(scrnL[0]*(1+2*x_d) + gun_width), int(2 * scrnL[1]*(1+2*y_d) // 2 - gun_width):int(2 * scrnL[1]*(1+2*y_d) // 2 + gun_width)]))
        return Ar

import matplotlib.pyplot as plt
class Object():
    def __init__(self, x0, type_M, vivant, group):
        self.RA = levelD[level]['RA'][type_M]  # *(1+0.2*(random()-0.5))
        self.x0 = x0
        self.norm = np.linalg.norm(self.x0 - x + [1, 0])
        self.width = 2 * scrnL[0] / self.norm
        self.widthY = 2 * scrnL[0] / self.norm
        self.DX = 0
        self.DY = 0
        self.type_M = levelD[level]['obj'][type_M]
        self.orient = 2 * pi * random()
        self.angle=self.orient
        if self.type_M == 4:
            self.orient = pi / 2
            self.angle = pi / 2
        if self.type_M == 10:
            R = 1
            self.orient = R * pi / 2
            self.angle = R * pi / 2
        self.im = np.minimum(
            pygame.surfarray.pixels3d(MO[self.type_M][45 * (((-ang[0] + pi / 8 + self.orient) // (pi / 4)) % 8)]), 255)
        self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)
        self.borne = [self.im.shape[0] - 1, self.im.shape[1] - 1]
        self.z = zmap[int(self.x0[1] + 100) // 2][int(self.x0[0] + 100) // 2]
        self.active = 0
        self.inline = 0
        self.vie = 1  # M_charac['life'][self.type_M]
        self.track = []
        self.mort = 0
        self.vivant = vivant
        self.color = 0
        self.group = group
        self.U = 0

    def calc_norm(self):
        global VIE, AMMO, Picked_O, CARTE, logL
        self.f0 = (self.x0 - x) @ rot_plan(-ang[0])
        self.norm = np.linalg.norm(self.x0 - x)
        self.width = self.RA * 2 * scrnL[0] / self.f0[0]
        self.widthY = self.RA * 2 * scrnL[0] / self.f0[0]
        self.DX = -self.RA * scrnL[0] / self.f0[0] + scrnL[0] * (self.f0[1] / self.f0[0]) / TAN2
        self.DY = -(self.RA * scrnL[0] * 1.7 - 5 * scrnL[0]) / self.f0[0] - scrnL[1] * tan(ang[1]) / TAN1 - (
                    2 * scrnL[1] * (z - self.z) / self.f0[0]) / TAN1
        A = 45 * (((atan((self.f0[1] / self.f0[0])) + self.orient - pi / 2 - ang[0] + pi / 8) // (pi / 4)) % 8)
        if A != self.angle:
            self.angle = A
            self.im = np.minimum(pygame.surfarray.pixels3d(MO[self.type_M][self.angle]), 255)
            if self.color != 0:
                u = [0, 0, 0]
                u[self.color - 1] = 1
                self.im = u * self.im
            self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)

        if self.norm < 5:
            if self.type_M == 1:
                VIE = min(100, VIE + 20)
                draw_vie()
                self.x0 = np.array([-49., -49])
                s = pygame.mixer.Sound("son/plop.ogg")
                logL.append('healing picked')
                s.play()
                Picked_O[0] = Picked_O[0] + 1
            if self.type_M == 2:
                self.x0 = np.array([-49., -49])
                AMMO[self.color - 1] += 10
                draw_AMMO()
                s = pygame.mixer.Sound("son/plop2.ogg")
                logL.append('ammo picked')
                s.play()
                Picked_O[0] = Picked_O[0] + 1
            if self.type_M == 5:
                self.x0 = np.array([-49., -49])
                CARTE[self.color - 1] += 1
                draw_cards()
                s = pygame.mixer.Sound("son/plop3.ogg")
                s.play()
                logL.append('card picked')
                Picked_O[0] = Picked_O[0] + 1

    def test_behind(self):
        if self.f0[0] > 0 and abs(self.f0[1] / self.f0[0]) < TAN2 + 0.5:
            if self.norm <= horizon[min(max(int(self.width // 2 + self.DX + scrnL[0]) // 2, 0), scrnL[0] - 1)] + 20:
                return True
        else:
            return False

    def render(self):
        global Boule, explo, VIE, explo_pt, Killed_O, HIT,x_d
        if ((self.inline and shoot == 2 and (self.norm <= 5 or arme != 0) and arme != 4) or (
                np.linalg.norm(self.x0 - explo_pt) < 20 and explo == 1)) and self.vie > 0 and self.type_M in destr:
            self.vie = 0
            Killed_O[0] = Killed_O[0] + 1
            if arme != 0:
                Accuracy[0] = Accuracy[0] + 1
            if self.type_M not in [6,11]:
                s = pygame.mixer.Sound("son/barril.ogg")
                for i in range(randint(5, 10)):
                    Boule.append(
                        boule(self.x0[0], self.x0[1], self.z, pi * random(), 2 * pi * random(), 2 * random() + 0.2, 1,
                              'image/effects/spark.png', 1))
                s.play()
                explo = 5
                explo_pt = self.x0
                if self.norm < 15:
                    VIE = VIE - DEGAT[4]
                    HIT = 1
                    s = pygame.mixer.Sound("son/aie.ogg")
                    s.play()

        if self.mort < 4 and self.vie <= 0:
            self.im = np.minimum(pygame.surfarray.pixels3d(Mod[self.type_M][int(self.mort)]), 255)
            self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)
            self.mort = min(self.mort + 1, 3)

        colorT = light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2]
        if light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2].sum() == 0:
            colorT = np.array([1, 1, 1])
        colorT = light_modif(colorT, level, c3)
        if explo!=0 and np.linalg.norm(self.x0 - explo_pt) < 20:
            colorT*=np.array([1,0,0])


        SQUARE = np.all(self.norm <= depth, axis=-1) & (Xthing <= self.width + self.DX + scrnL[0]) & (
                    Xthing >= self.DX + scrnL[0]) & (Ything <= self.widthY + self.DY + scrnL[1]) & (
                             Ything >= self.DY + scrnL[1])

        self.U = np.stack(((Xthing - self.DX - scrnL[0]) / self.width, (Ything - self.DY - scrnL[1]) / self.widthY),
                          -1) * np.expand_dims(SQUARE, -1)
        self.G = np.maximum((self.U * 160).astype(int), 0)
        Ar = np.moveaxis(self.im[tuple(map(tuple, self.G.T))] * colorT, 1, 0)
        if light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2].sum() == 0:
            Ar = Ar * torch_on * TORCHE ** 3
            Ar = np.minimum(np.divide(Ar, 0.1 * self.norm ** 0.5), 255)
        else:
            Ar = Ar * level_light
        self.Ut = np.moveaxis(self.vis[tuple(map(tuple, self.G.T))], 1, 0)
        self.shot=[]
        if shoot==1:
            x_d0=[]
            self.inline=False
            for i in range(len(x_d)):
                inline = min(np.sum(self.Ut[int(scrnL[0]*(1+2*x_d[i][0]) - gun_width):int(scrnL[0]*(1+2*x_d[i][0]) + gun_width), int(2 * scrnL[1]*(1+2*x_d[i][1]) // 2 - gun_width):int(2 * scrnL[1]*(1+2*x_d[i][1]) // 2 + gun_width)]),
                                1)
                self.inline=(self.inline or inline)
                self.shot.append(i)
                if not inline:
                    x_d0.append(x_d[i])
            x_d=x_d0

        return Ar




class boule(pygame.sprite.Sprite):
    def __init__(self, x, y, z, ang1, ang2, v, masse, name, deg):
        self.p = np.array([x, y, z])
        self.ang1 = ang1
        self.vx = v * sin(ang1) * cos(ang2)
        self.vy = v * sin(ang1) * sin(ang2)
        self.vz = v * cos(ang1)
        self.v = np.array([self.vx, self.vy, self.vz])
        self.im = pygame.image.load(name)
        self.D = np.linalg.norm(x - self.p[:-1])
        self.f0 = (self.p[:-1] - x) @ rot_plan(-ang[0])
        self.masse = masse
        self.hit = 0
        self.lifetime = 0
        if 'image/effects/hurt.png' == name or 'image/effects/hurt2.png' == name or 'image/effects/hurt3.png' == name:
            self.hit = 1

        self.X = 1 * (self.f0[1] / self.f0[0]) / TAN2 + 1
        self.Y = 1 * (self.p[-1] / self.f0[0]) / TAN1 + 1 - 1 * tan(ang[1]) / TAN1
        self.deg = deg

    def update(self):
        self.lifetime += 1
        self.v += np.array([0, 0, 0.01]) * self.masse
        self.p += self.v
        self.D = np.linalg.norm(x - self.p[:-1])
        self.f0 = (self.p[:-1] - x) @ rot_plan(-ang[0])

        self.X = 1 * (self.f0[1] / self.f0[0]) / TAN2 + 1
        self.Y = 2 * ((self.p[-1] - z) / self.f0[0]) / TAN1 + 1 - tan(ang[1]) / TAN1

        global VIE, HIT
        if np.linalg.norm((self.p[:-1] - x)) < 2 and self.hit == 1:
            VIE -= self.deg
            HIT = 1
            s = pygame.mixer.Sound("son/aie.ogg")
            s.play()
            return True

        if (authorized_map[int(self.p[1] + 101) // 2][int(self.p[0] + 101) // 2] == 0 or (
                level_map[int(self.p[1] + 101) // 2][int(self.p[0] + 101) // 2] == 1 and
                authorized_map[int(self.p[1] + 101) // 2][int(self.p[0] + 101) // 2] == 2)) or (self.p[-1] - z) > 5 + \
                hmap[int(self.p[1] + 101) // 2][int(self.p[0] + 101) // 2] or (
                self.p[-1] - z) < -5 or self.lifetime > 500:

            return True
        else:
            return False

    def affiche(self):
        if self.f0[0] > 0 and abs(self.f0[1] / self.f0[0]) < TAN2 + 0.5 and self.D <= \
                depth[int(self.X * scrnL[0]) % (2 * scrnL[0])][int(self.Y * scrnL[1] % (2 * scrnL[1]))]:
            self.imA = pygame.transform.scale(self.im, (
            min(int(Ratio*300 / self.f0[0]), window[1] // 1), min(int(Ratio*300 / self.f0[0]), window[1] // 1)))
            fond.blit(self.imA, (int((window[0] // 2) * self.X), int((window[1] // 2) * self.Y)))


class grenade(pygame.sprite.Sprite):
    def __init__(self, x, y, z, ang1, ang2, v, masse, deg):
        self.p = np.array([x, y, z])
        self.ang1 = ang1
        self.vx = v * sin(ang1) * cos(ang2)
        self.vy = v * sin(ang1) * sin(ang2)
        self.vz = v * cos(ang1)
        self.v = np.array([self.vx, self.vy, self.vz])
        self.p += 2 * np.array([cos(ang2), sin(ang2), 0])
        self.im = []
        for i in range(4):
            self.im.append(pygame.image.load('image/effects/grenade%s.png' % str(i)))
        self.D = np.linalg.norm(x - self.p[:-1])
        self.f0 = (self.p[:-1] - x) @ rot_plan(-ang[0])
        self.masse = masse
        self.hit = 0
        self.lifetime = 0

        self.X = 1 * (self.f0[1] / self.f0[0]) / TAN2 + 1
        self.Y = 1 * (self.p[-1] / self.f0[0]) / TAN1 + 1 - 1 * tan(ang[1]) / TAN1
        self.deg = deg
        self.cool = 0
        self.size = 1000

    def update(self):
        self.lifetime += 1
        self.v += np.array([0, 0, 0.01]) * self.masse
        self.p += self.v
        self.D = np.linalg.norm(x - self.p[:-1])
        self.f0 = (self.p[:-1] - x) @ rot_plan(-ang[0])

        self.X = 1 * (self.f0[1] / self.f0[0]) / TAN2 + 1
        self.Y = 2 * ((self.p[-1] - z) / self.f0[0]) / TAN1 + 1 - tan(ang[1]) / TAN1

        # global VIE, HIT
        # if np.linalg.norm((self.p[:-1] - x)) < 2 and self.lifetime >= 100:
        # 	VIE -= self.deg
        # 	HIT = 1
        # 	s = pygame.mixer.Sound("son/aie.ogg")
        # 	s.play()
        # 	return True

        self.cool = max(0, self.cool - 1)

        if self.p[-1] >= zmap[int(self.p[1] + 101) // 2][int(self.p[0] + 101) // 2] + 2.5:
            self.v[-1] *= -1
            self.v *= 0.5
            self.p[-1] = zmap[int(self.p[1] + 101) // 2][int(self.p[0] + 101) // 2] + 2.4
        if (authorized_map[int(self.p[1] + 101) // 2][int(self.p[0] + 101) // 2] == 1) or (
                level_map[int(self.p[1] + 101) // 2][int(self.p[0] + 101) // 2] == 1) and self.cool == 0:
            for i in wall[:20]:
                if i not in h_wall:
                    N = np.stack((i.a[0][0], i.b[0][0], -i.n), axis=-1)
                    V = np.maximum(np.minimum(np.linalg.solve(N, -i.X[0][0] + self.p)[:-1], 1), 0)
                    norm = np.linalg.norm(i.X[0][0][:-1] + V[0] * i.a[0][0][:-1] + V[1] * i.b[0][0][:-1] - self.p[:-1])
                    if norm < 4:
                        # self.v[:-1] *= -1

                        No = i.n[:-1]
                        No = No / np.linalg.norm(No)

                        x_ = i.X[0][0][0]
                        y_ = i.X[0][0][1]
                        a_ = i.b[0][0][0]
                        b_ = i.b[0][0][1]
                        side = b_ * self.p[0] - a_ * self.p[1] + a_ * y_ - b_ * x_

                        if side < 0:
                            self.v[:-1] += +2 * abs(np.dot(self.v[:-1], No)) * No
                        else:
                            self.v[:-1] += +2 * abs(np.dot(self.v[:-1], -No)) * -No
                        self.v *= 0.5
                        self.cool = 20
                        break

        if self.lifetime == 100:
            global explo, Boule, explo_pt, VIE, HIT
            s = pygame.mixer.Sound("son/barril.ogg")
            for i in range(randint(5, 10)):
                Boule.append(
                    boule(self.p[0], self.p[1], self.p[2], pi * random(), 2 * pi * random(), 2 * random() + 0.2, 1,
                          'image/effects/spark.png', 1))
            s.play()
            explo = 5
            explo_pt = self.p[:-1]
            if self.D < 15:
                VIE = VIE - DEGAT[4]
                HIT = 1
                s = pygame.mixer.Sound("son/aie.ogg")
                s.play()
        if self.lifetime >= 100 and self.lifetime <= 106:
            self.im = []
            self.size = 5000
            for i in range(4):
                self.im.append(pygame.image.load('image/effects/explo%s.png' % str(self.lifetime - 100)))

        if self.lifetime == 107:
            return True
        return False

    def affiche(self):
        if self.f0[0] > 0 and abs(self.f0[1] / self.f0[0]) < TAN2 + 0.5 and self.D <= \
                depth[int(self.X * scrnL[0]) % (2 * scrnL[0])][int(self.Y * scrnL[1] % (2 * scrnL[1]))]:

            self.imA = pygame.transform.scale(self.im[(c) % 4], (
                min(int(Ratio*self.size / self.f0[0]), window[1] // 1), min(int(Ratio*self.size / self.f0[0]), window[1] // 1)))

            if len(IS)>0:
                for j,i in enumerate(IS):
                    if i.U[int(self.X * scrnL[0]) % (2 * scrnL[0])][int(self.Y * scrnL[1] % (2 * scrnL[1]))] and i.X[0,0,-1]/2<self.p[-1]:
                        colorliquid=IS_rendered[j][int(self.X * scrnL[0]) % (2 * scrnL[0])][int(self.Y * scrnL[1] % (2 * scrnL[1]))]
                        self.imA.fill(colorliquid, special_flags=BLEND_RGB_MULT)


            shift = min(int(self.size / self.f0[0]), window[1] // 1) // 2
            fond.blit(self.imA, (int((window[0] // 2) * self.X) - shift, int((window[1] // 2) * self.Y) - shift))


font = pygame.font.Font('freesansbold.ttf', 13)


def draw_vie():
    pygame.draw.line(fenetre, (255, 0, 0), (int(0.11 * window[0]), int(1.05 * window[1])),
                     (int(0.11 * window[0] + 0.1 * window[0]), int(1.05 * window[1])), int(10*window[0]/960))
    pygame.draw.line(fenetre, (0, 255, 0), (int(0.11 * window[0]), int(1.05 * window[1])),
                     (int(0.11 * window[0] + 0.1 * window[0] * max(VIE, 0) / 100), int(1.05 * window[1])), int(10*window[0]/960))


def draw_hud():
    global back, font, code1, code2, fontC
    fontC = pygame.font.Font('freesansbold.ttf', int(32 * window[0] / (12 * scrnL[0])))
    code1 = pygame.transform.scale(code01, (int(0.8 * window[1]), int(0.16 * window[1])))
    code2 = pygame.transform.scale(code02, (int(0.8 * window[1]), int(0.16 * window[1])))
    font = pygame.font.Font('freesansbold.ttf', int(13 * window[0] / (12 * scrnL[0])))
    back = pygame.transform.scale(back, ((int(0.8 * window[1]), int(10 * window[1]))))
    fenetre.blit(
        pygame.transform.scale(pygame.image.load('image/Interface/fond.png'), (window[0], int(0.2 * window[1]))),
        (0, window[1]))
    fenetre.blit(pygame.transform.scale(pygame.image.load('image/Interface/persom.png'),
                                        (int(0.2 * window[1]), int(0.2 * window[1]))), (0, window[1]))
    fenetre.blit(pygame.transform.scale(pygame.image.load('image/Interface/text.png'),
                                        (int(0.8 * window[1]), int(0.18 * window[1]))),
                 (int(0.45 * window[1]), int(1.01 * window[1])))


def draw_AMMO():
    for i in range(3):
        u = [100, 100, 100]
        u[i] = 250
        pygame.draw.line(fenetre, (50, 50, 50), (int(0.8 * window[0]), window[1] + int((0.05 * i + 0.04) * window[1])),
                         (int(0.95 * window[0]), window[1] + int((0.05 * i + 0.04) * window[1])), 20)
        text = font.render('AMMO type ' + str(i) + '   ' + str(AMMO[i]), True, tuple(u))
        textRect = text.get_rect()
        textRect.topleft = (int(0.8 * window[0]), window[1] + int((0.05 * i + 0.025) * window[1]))
        fenetre.blit(text, textRect)


def draw_cards():
    fenetre.blit(pygame.transform.scale(pygame.image.load('image/Interface/cards.png'),
                                        (int(0.12 * window[0]), int(0.08 * window[0]))),
                 (int(0.65 * window[0]), int(1.01 * window[1])))
    text = font.render('CARDS', True, (200, 200, 200))
    textRect = text.get_rect()
    textRect.topleft = (int((0.69) * window[0]), int(1.135 * window[1]))
    fenetre.blit(text, textRect)

    for i in range(3):
        text = font.render(str(CARTE[i]), True, (200, 200, 200))
        textRect = text.get_rect()
        textRect.topleft = (int((0.67 + 0.04 * i) * window[0]), int(1.11 * window[1]))
        fenetre.blit(text, textRect)


def show_message():
    Shift_5 = int(5 * window[0] / (12 * scrnL[0]))
    Shift_13 = int(13 * window[0] / (12 * scrnL[0]))
    fenetre.blit(pygame.transform.scale(pygame.image.load('image/Interface/text.png'),
                                        (int(0.8 * window[1]), int(0.18 * window[1]))),
                 (int(0.45 * window[1]), int(1.01 * window[1])))
    fenetre.blit(back.subsurface(
        (0, max(linenumber + Sline - 5, 0) * Shift_13, int(0.8 * window[1]), int(0.18 * window[1]) - 2 * Shift_5)),
                 (int(0.45 * window[1]), Shift_5 + int(1.01 * window[1])))


def write_message(msg, I):
    global linenumber, indk
    Shift_5 = int(5 * window[0] / (12 * scrnL[0]))
    Shift_13 = int(13 * window[0] / (12 * scrnL[0]))
    if msg[I] == '#':
        linenumber += 1
        indk = I
        return 0
    text = font.render(msg[indk:I], True, (255, 255, 255))
    textRect = text.get_rect()
    text2 = font.render(msg[I], True, (255, 255, 255))
    back.blit(text2, (Shift_5 + textRect[2], Shift_13 * linenumber))
    if textRect[2] > int(0.7 * window[1]) or I == len(msg) - 1:
        indk = I
        linenumber += 1
    if I == len(msg) - 1:
        linenumber += 1
    show_message()


window = 12 * scrnL
back = pygame.Surface((int(0.8 * window[1]), int(10 * window[1])), SRCALPHA)
fenetre = pygame.display.set_mode((window[0], int(window[1] * 1.2)))
running = 1
mouse_c = 1
m_stock = window // 2
pygame.mouse.set_pos([window[0] // 2 - 10, window[1] // 2 + 10])
pygame.time.wait(100)
ang = [0.0, 0.0]
pygame.mouse.set_visible(0)
v = 1.2
x = np.array([-1, 0])
c = 0
c2 = 0
z = 0
Sline = 0

f = open("image/monstD", "rb")
M1 = pickle.load(f)
MD = []
for i in M1[:]:
    MD0 = []
    for j in i[:]:
        M0 = {}
        for k in j.keys():
            M0[k] = pygame.image.load('image/monsters/' + j[k])
        MD0.append(M0)
    MD.append(MD0)

Ma0 = pickle.load(f)
MDa = []
for j in Ma0:
    Ma = {}
    for k in j.keys():
        Ma[k] = pygame.image.load('image/monsters/' + j[k])
    MDa.append(Ma)

f.close()

f = open("image/obj_D", "rb")
MO1 = pickle.load(f)
MO = []
for i in MO1[:]:
    M0 = {}
    for k in i.keys():
        M0[k] = pygame.image.load('image/objets/' + i[k])
    MO.append(M0)
f.close()

f = open("image/obj_destr_D", "rb")
Mod = pickle.load(f)
Mod2 = Mod.copy()
for i in Mod2.keys():
    for k, j in enumerate(Mod2[i]):
        Mod[i][k] = pygame.image.load('image/obj_destr/' + j)
f.close()

f = open("image/gunD", "rb")
MG0 = pickle.load(f)
MG1 = []
for i in MG0:
    MG2 = {}
    for j in i.keys():
        if setting['smooth'] == False:
            MG2[j] = pygame.transform.scale(
                pygame.transform.scale(pygame.image.load('image/gun/' + i[j]), (2 * scrnL[0], 2 * scrnL[1])), window)
        else:
            MG2[j] = pygame.transform.smoothscale(
                pygame.transform.scale(pygame.image.load('image/gun/' + i[j]), (2 * scrnL[0], 2 * scrnL[1])), window)
    MG1.append(MG2)
f.close()
VIE = 100
GUN_im = MG1[0][0]
attack = 0
shoot = 0
coolD = 0
arme = 0
TotAr = 1
COOLDOWN = [10, 10, 17, 0, 10]
REC = [0, 1, 3, 0, 0]
DEGAT = [35, 25, 70, 10, 50]
PREC=[0,pi/200,pi/100,pi/100]
BULLETS=[1,1,5,1,1]
clock = pygame.time.Clock()
colorGUN = (1., 1., 1.)
AMMO = [0, 0, 0, 0]
CARTE = [0, 0, 0]
Trig_liste = []
dicoTEXT = {}
groupD = []
code = 0

Killed_E = np.array([0, 0, 0])
Killed_O = np.array([0, 0, 0])
Picked_O = np.array([0, 0, 0])
Accuracy = np.array([0, 0, 0.])
Explored = np.array([0., 0.])
Play_Time = [0, 0]

porteson = pygame.mixer.Sound("son/SF-fermport.ogg")
pygame.mixer.music.load("son/mu.ogg")
pygame.mixer.music.set_volume(0.4)


# ----------------------LEVEL

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
    global MG1, BLOOD
    for k, i in enumerate(MG0.copy()):
        for j in i.keys():
            if setting['smooth'] == False:
                MG1[k][j] = pygame.transform.scale(
                    pygame.transform.scale(pygame.image.load('image/gun/' + i[j]), (2 * scrnL[0], 2 * scrnL[1])),
                    window)
            else:
                MG1[k][j] = pygame.transform.smoothscale(
                    pygame.transform.scale(pygame.image.load('image/gun/' + i[j]), (2 * scrnL[0], 2 * scrnL[1])),
                    window)
    BLOOD = pygame.transform.scale(pygame.image.load('image/effects/blood.png'), window)


def reset_all():
    global VIE, attack, shoot, coolD, arme, TotAr, colorGUN, mouse_c, ang, x, z, R_c, screen
    VIE = 100
    attack = 0
    shoot = 0
    coolD = 0
    arme = 0
    colorGUN = (1., 1., 1.)
    mouse_c = 1
    ang = [1e-5, 0.0]
    x = np.array([-1, 0])
    z = 0
    R_c = np.array([-1, 0, 0])
    screen = np.full((*scrnL, 6), 0.0)
    screen[:, :, 1:3] = I_n
    screen[:, :, 4] = np.sin(Ang[:, :, 0])
    screen[:, :, 5] = np.cos(-Ang[:, :, 1] + pi * 0.5)
    screen[:, :, 3] = 1


def write_text_msg(text0):
    font3 = pygame.font.Font('freesansbold.ttf', int(26 * window[0] / (12 * scrnL[0])))
    text = font3.render(text0, True, (255, 255, 255), (0, 0, 0))
    textRect = text.get_rect()
    fenetre.blit(text, (int(window[0] * 0.01), int(window[1] * 0.03)))


def write_log(text0, d):
    font3 = pygame.font.Font('freesansbold.ttf', int(14 * window[0] / (12 * scrnL[0])))
    text = font3.render(text0, True, (150, 0, 0))
    textRect = text.get_rect()
    fenetre.blit(text, (int(window[0] * 0.99) - textRect[2], int(window[1] * 0.02 * (d + 1))))


def stats():
    fenetre.fill((0, 0, 0))
    pygame.display.flip()
    pygame.time.wait(100)
    IMLOAD = pygame.transform.scale(pygame.image.load('image/interface/back0.png'), (window[0], int(window[1] * 1.2)))
    fenetre.blit(IMLOAD, (0, 0))
    pygame.display.flip()
    fontH = pygame.font.Font('freesansbold.ttf', int(32 * window[0] / (12 * scrnL[0])))
    text0 = fontH.render(level_nameL[level], True, (255, 255, 255))
    textRect0 = text0.get_rect()
    for ci in range(len(level_nameL[level])):
        text = fontH.render(level_nameL[level][ci], True, (255, 255, 255))
        text2 = fontH.render(level_nameL[level][0:ci], True, (255, 255, 255))
        textRect = text2.get_rect()
        fenetre.blit(text, (int(window[0] * 0.5 + textRect[2] - textRect0[2] / 2), int(window[1] * 0.05)))

        pygame.event.get()
        pygame.time.wait(50)
        pygame.display.flip()

    fontH = pygame.font.Font('freesansbold.ttf', int(28 * window[0] / (12 * scrnL[0])))
    stats = []
    gap = '                   '
    stats.append('Ennemies killed :' + str(Killed_E[0]) + '/' + str(Killed_E[1]))
    stats.append('Total ennemies killed :' + str(Killed_E[-1]))
    stats.append('Objects destroyed :' + str(Killed_O[0]) + '/' + str(Killed_O[1]))
    stats.append('Total objects destroyed :' + str(Killed_O[-1]))
    stats.append('Objects picked :' + str(Picked_O[0]) + '/' + str(Picked_O[1]))
    stats.append('Total objects picked :' + str(Picked_O[-1]))
    stats.append('Accuracy :' + str(int(100 * Accuracy[0] / max(1, Accuracy[1]))) + '%')
    stats.append('Total accuracy :' + str(int(100 * Accuracy[-1])) + '%')
    stats.append('Map Explored :' + str(int(Explored[0] * 100)) + '%')
    stats.append('Total Map Explored :' + str(int(Explored[1] * 100)) + '%')
    K = str(int(Play_Time[0] // 3600)) + 'h' + str(int((Play_Time[0] % 3600) // 60)) + 'm' + str(
        int(Play_Time[0] % 60)) + 's'
    Kt = str(int(Play_Time[1] // 3600)) + 'h' + str(int((Play_Time[1] % 3600) // 60)) + 'm' + str(
        int(Play_Time[1] % 60)) + 's'
    stats.append('Level time :' + K)
    stats.append('Total time :' + Kt)
    linenumber = 2
    for c, anim in enumerate(stats):
        if c % 2 == 0:
            linenumber += 2
            shift = 0
        else:
            shift = int(window[0] * 0.55)
        indK = 0
        for ci in range(len(anim)):
            text = fontH.render(anim[ci], True, (255, 255, 255))
            text2 = fontH.render(anim[indK:ci], True, (255, 255, 255))
            textRect = text2.get_rect()
            fenetre.blit(text, (int(window[0] * 0.05 + textRect[2] + shift),
                                int(window[1] * 0.05 + 28 * window[0] / (12 * scrnL[0]) * linenumber)))
            if textRect[2] > int(0.95 * window[0]):
                indK = ci
                linenumber += 2

            pygame.event.get()
            pygame.time.wait(50)
            pygame.display.flip()
    stat_run = 1
    text0 = fontH.render('Press space to continue', True, (255, 255, 255))
    textRect0 = text0.get_rect()
    linenumber += 2
    fenetre.blit(text0, (
    int(window[0] * 0.5 - textRect0[2] / 2), int(window[1] * 0.05 + 28 * window[0] / (12 * scrnL[0]) * linenumber)))
    pygame.display.flip()
    while stat_run == 1:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = 0
        key = pygame.key.get_pressed()
        if key[K_SPACE]:
            stat_run = 0
    return 1


def animation(N):
    if N == 0:
        fenetre.fill((0, 0, 0))
        pygame.display.flip()
        pygame.time.wait(1000)
        anim = 'Phone:There has been ... an incident, at a biotechnology and genetics lab. We need you in the end. Come now !'
        fontH = pygame.font.Font('freesansbold.ttf', int(32 * window[0] / (12 * scrnL[0])))
        s = pygame.mixer.Sound("son/rain.ogg")
        s.play()
        for i in range(50):
            pygame.event.get()
            IMLOAD = pygame.transform.scale(pygame.image.load('image/animation/a0_0.png'),
                                            (window[0], int(window[1] * 1.2)))
            if i % 2 == 0:
                IMLOAD = pygame.transform.scale(pygame.image.load('image/animation/a0_0bis.png'),
                                                (window[0], int(window[1] * 1.2)))
            fenetre.blit(IMLOAD, (0, 0))
            pygame.display.flip()
            pygame.time.wait(100)

        timeL = [10, 5000, 250, 4000, 150, 150, 150]
        for i in range(7):
            IMLOAD = pygame.transform.scale(pygame.image.load('image/animation/a' + str(N) + '_' + str(i) + '.png'),
                                            (window[0], int(window[1] * 1.2)))
            fenetre.blit(IMLOAD, (0, 0))
            for j in range(timeL[i] // 10):
                pygame.event.get()
                pygame.time.wait(10)
                pygame.display.flip()
            if i == 2:
                s = pygame.mixer.Sound("son/phone.ogg")
                s.play()
        indK = 0
        linenumber = 0
        for ci in range(len(anim)):
            text = fontH.render(anim[ci], True, (255, 255, 255))
            text2 = fontH.render(anim[indK:ci], True, (255, 255, 255))
            textRect = text2.get_rect()
            fenetre.blit(text, (
            int(window[0] * 0.05 + textRect[2]), int(window[1] * 0.05 + 32 * window[0] / (12 * scrnL[0]) * linenumber)))
            if textRect[2] > int(0.6 * window[0]):
                indK = ci
                linenumber += 1

            pygame.event.get()
            pygame.time.wait(50)
            pygame.display.flip()

        return 1
    return 0


modif_game = ['0_1', '0_V2', '1_2', '1_6', '2_1', '2_6', '3_2', '3_3', '3_4', '3_G99', '4_2', '4_3', '4_4']

tutotxt = []
tutotxt.append('Use the arrow keys to move and leftclick to attack')
tutotxt.append('Scroll with the mouse to change weapon')


def change_game(num):
    global TotAr, tuto, activatedT, torch_on, AMMO, docs, logL
    if num == '0_1':
        tuto = 1
    if num == '0_V2':
        tuto = 2
        TotAr = 2
        logL.append('pistol picked')
        s = pygame.mixer.Sound("son/plop_special.ogg")
        s.play()
    if num == '1_2':
        activatedT.append(Trig_liste[-1][1])
    if num == '1_6':
        activatedT.remove(Trig_liste[-1][1])
    if num == '2_1':
        activatedT.append(Trig_liste[-2][1])
    if num == '2_6':
        activatedT.remove(Trig_liste[-2][1])
        torch_on = 1
        logL.append('torch picked')
        s = pygame.mixer.Sound("son/plop_special.ogg")
        s.play()
        for i in thing:
            if i.type_M == 14 and (i not in ennemies):
                i.x0 = np.array([-100, -100])
    if num == '3_2':
        TotAr = 3
        for i in thing:
            if i.type_M == 16 and (i not in ennemies):
                i.x0 = np.array([-100, -100])
        AMMO[1] += 10
        draw_AMMO()
        logL.append('shotgun picked')
        s = pygame.mixer.Sound("son/plop_special.ogg")
        s.play()

    if num == '3_3':
        logL.append('doc picked')
        docs.insert(0, 'code_g.png')
        s = pygame.mixer.Sound("son/plop_special.ogg")
        s.play()

        for i in thing:
            if i.type_M == 17 and (i not in ennemies):
                i.x0 = np.array([-100, -100])
                break

    if num == '3_4':
        logL.append('doc picked')
        docs.insert(0, 'code_d.png')
        s = pygame.mixer.Sound("son/plop_special.ogg")
        s.play()
        for i in thing:
            if i.type_M == 17 and (i not in ennemies):
                i.x0 = np.array([-100, -100])
                break
    if num == '3_G99':
        logL.append('doc picked')
        docs.insert(0, 'ID.png')
        s = pygame.mixer.Sound("son/plop_special.ogg")
        s.play()

    if num == '4_2':
        logL.append('doc picked')
        docs.insert(0, 'code_color.png')
        s = pygame.mixer.Sound("son/plop_special.ogg")
        s.play()
        for i in thing:
            if i.type_M == 17 and (i not in ennemies):
                i.x0 = np.array([-100, -100])
                break
    if num == '4_3':
        logL.append('doc picked')
        docs.insert(0, 'code_color_ordre.png')
        s = pygame.mixer.Sound("son/plop_special.ogg")
        s.play()
        for i in thing:
            if i.type_M == 17 and (i not in ennemies):
                i.x0 = np.array([-100, -100])
                break
    if num == '4_4':
        logL.append('doc picked')
        docs.insert(0, 'DNA.png')
        s = pygame.mixer.Sound("son/plop_special.ogg")
        s.play()
        counterspecial = 0
        for i in thing:
            if i.type_M == 17 and (i not in ennemies):
                i.x0 = np.array([-100, -100])
                counterspecial += 1
            if counterspecial == 2:
                break


Xmap_, Ymap_ = np.indices((500, 500))


def update_map(level_map, x):  # WARNING correct
    global MAP
    base_map = (-level_map + 2 * authorized_map) / 4

    MAP = np.minimum(
        MAP + np.where(((x[1] / 2 + 50 - Xmap_) ** 2) + ((x[0] / 2 + 50 - Ymap_) ** 2) < 15 ** 2, base_map, 0),
        base_map)


mapback = pygame.image.load('image/interface/map.png')


def show_MAP(MAP):
    global mapback, shiftM

    mapback = pygame.transform.scale(mapback, (int(0.75 * window[1]), int(0.75 * window[1])))
    X = int(min(x[1] / 2 + 50 - 50 - (shiftM[1]), 0)) * (0.7 * window[1]) / 100
    Y = int(min(0.5 + x[0] / 2 + 50 - 50 - (shiftM[0]), 0)) * (0.7 * window[1]) / 100
    X0 = max(x[1] / 2 + 50 - 50 - 399 - (shiftM[1]), 0)
    Y0 = max(0.5 + x[0] / 2 + 50 - 50 - 399 - (shiftM[0]), 0)

    if X < 0:
        shiftM[1] = int(x[1] / 2)
        X = 0
    if Y < 0:
        shiftM[0] = int(0.5 + x[0] / 2)
        Y = 0
    if X0 > 499:
        shiftM[1] = 399 - int(x[1] / 2)
        X0 = 0
    if Y0 > 499:
        shiftM[1] = 399 - int(0.5 + x[0] / 2)
        Y0 = 0

    MAP2 = 100 * MAP + np.minimum(np.where(((x[1] / 2 + 50 - Xmap_) ** 2) + ((x[0] / 2 + 50 - Ymap_) ** 2) < 15 ** 2,
                                           50 + 10000 * MAP / (
                                                       ((x[1] / 2 + 50 - Xmap_) ** 2) + ((x[0] / 2 + 50 - Ymap_) ** 2)),
                                           0), 150)

    MAP2 = np.dstack((MAP2, MAP2, MAP2))
    for i in thing:
        pos = (i.x0 + 100) / 2
        if MAP2[int(pos[1]), int(pos[0]), :].any() != 0:
            if i in ennemies:
                if i.vie > 0:
                    MAP2[int(pos[1])][int(pos[0])] = [255, 0, 0]
            else:
                MAP2[int(pos[1])][int(pos[0])] = [0, 200, 200]

    mapim = pygame.surfarray.make_surface(MAP2)
    mapim = mapim.subsurface((int(min(max(x[1] / 2 + 50 - 50 - (shiftM[1]), 0), 399)),
                              int(min(max(0.5 + x[0] / 2 + 50 - 50 - (shiftM[0]), 0), 399)), 100, 100))
    mapim = pygame.transform.flip(pygame.transform.scale(mapim, (int(0.7 * window[1]), int(0.7 * window[1]))), 0, 1)
    fenetre.blit(mapim, (int(window[0] / 2 - 0.35 * window[1]), int(0.15 * window[1])))

    zz = np.array([int(min(max(x[1] / 2 + 50 - 50 - (shiftM[1]), 0), 399)),
                   int(min(max(0.5 + x[0] / 2 + 50 - 50 - (shiftM[0]), 0), 399))])

    for i in lifts:
        pos = i - np.array([zz[1], zz[0]]) - 50

        pos = pos * (0.007 * window[1])
        x1 = max(min(int(pos[1] + window[0] * 0.5), int(window[0] / 2 + 0.35 * window[1])),
                 int(window[0] / 2 - 0.35 * window[1]))
        y1 = max(min(int(-pos[0] + window[1] * 0.5), int(0.85 * window[1])), int(0.15 * window[1]))
        if x1 not in [int(window[0] / 2 + 0.35 * window[1]), int(window[0] / 2 - 0.35 * window[1])] and y1 not in [
            int(0.85 * window[1]), int(0.15 * window[1])]:
            fontH = pygame.font.Font('freesansbold.ttf', int(16 * window[0] / (12 * scrnL[0])))
            text0 = fontH.render('lift', True, (255, 255, 0))
            fenetre.blit(text0, (x1 + 10, y1))

            pygame.draw.circle(fenetre, (255, 255, 0), (x1, y1), int(0.01 * window[1]))
    for i in doors:
        if i.door > 1:
            pos0 = (i.X[0, 0, :-1] + np.array([0, 1])) / 2 - np.array([zz[1], zz[0]])
            pos1 = pos0 + i.b[0, 0, :-1] / 2
            pos0 = pos0 * (0.007 * window[1])
            pos1 = pos1 * (0.007 * window[1])
            XX = 50 + (i.X[0, 0, :-1] + np.array([0, 1])) / 2

            if i.door == 2:
                colo = np.array([255, 50, 50]) * min(MAP2[int(XX[1])][int(XX[0])][0] / 100, 1)
            if i.door == 3:
                colo = np.array([50, 255, 50]) * min(MAP2[int(XX[1])][int(XX[0])][0] / 100, 1)
            if i.door == 4:
                colo = np.array([50, 50, 255]) * min(MAP2[int(XX[1])][int(XX[0])][0] / 100, 1)

            if i.door > 4:
                colo = np.array([100, 0, 100]) * min(MAP2[int(XX[1])][int(XX[0])][0] / 100, 1)

            x1 = max(min(int(pos0[1] + window[0] * 0.5), int(window[0] / 2 + 0.35 * window[1])),
                     int(window[0] / 2 - 0.35 * window[1]))
            y1 = max(min(int(-pos0[0] + window[1] * 0.5), int(0.85 * window[1])), int(0.15 * window[1]))
            x2 = max(min(int(pos1[1] + window[0] * 0.5), int(window[0] / 2 + 0.35 * window[1])),
                     int(window[0] / 2 - 0.35 * window[1]))
            y2 = max(min(int(-pos1[0] + window[1] * 0.5), int(0.85 * window[1])), int(0.15 * window[1]))
            pygame.draw.line(fenetre, colo.astype(int), (x1, y1), (x2, y2), 3)

    pos = 0.5 * x - np.array([zz[1], zz[0]])
    pos = pos * (0.007 * window[1])
    xm = max(min(int(pos[1] + window[0] * 0.5), int(window[0] / 2 + 0.35 * window[1])),
             int(window[0] / 2 - 0.35 * window[1]))
    ym = max(min(int(-pos[0] + window[1] * 0.5), int(0.85 * window[1])), int(0.15 * window[1]))
    if xm not in [int(window[0] / 2 + 0.35 * window[1]), int(window[0] / 2 - 0.35 * window[1])] and ym not in [
        int(0.85 * window[1]), int(0.15 * window[1])]:
        pygame.draw.polygon(fenetre, (200, 100, 100), (
        (xm, ym), (xm + 15 * cos(ang[0] + pi / 8 + pi / 2), ym - 15 * sin(ang[0] + pi / 8 + pi / 2)),
        (xm + 15 * cos(ang[0] - pi / 8 + pi / 2), ym - 15 * sin(ang[0] - pi / 8 + pi / 2))), 0)
        pygame.draw.circle(fenetre, (255, 0, 0), (xm, ym), int(0.01 * window[1]))

    fenetre.blit(mapback, (int(window[0] / 2 - 0.375 * window[1]), int(0.125 * window[1])))


folder_im = pygame.image.load('image/interface/folder.png')
folder_im2 = pygame.image.load('image/interface/folder2.png')
fleche = pygame.image.load('image/interface/fleche.png')
fleche2 = pygame.image.load('image/interface/fleche2.png')
docs = ['no_doc.png']
doc_on = 0
INDDOC = 0


def show_doc():
    global doc_on, INDDOC

    if doc_on:

        if abs(mouse[0] - int(1.3 * window[1])) < int(0.05 * window[1]) and abs(mouse[1] - int(0.8 * window[1])) < int(
                0.05 * window[1]):
            fl = pygame.transform.scale(fleche2, (int(0.1 * window[1]), int(0.1 * window[1])))
            if clic[0] == 1:
                INDDOC = (INDDOC + 1) % len(docs)
                pygame.time.wait(300)
        else:
            fl = pygame.transform.scale(fleche, (int(0.1 * window[1]), int(0.1 * window[1])))
        docim = pygame.image.load('image/doc/' + docs[INDDOC])
        docim = pygame.transform.scale(docim, (int(0.75 * window[1]), int(0.75 * window[1])))
        fenetre.blit(docim, (int(window[0] / 2 - 0.375 * window[1]), int(0.125 * window[1])))
        fenetre.blit(fl, (int(1.25 * window[1]), int(0.75 * window[1])))
        fenetre.blit(fl, (int(1.25 * window[1]), int(0.75 * window[1])))
        if abs(mouse[0] - int(0.3 * window[1])) < int(0.15 * window[1]) and abs(mouse[1] - int(0.3 * window[1])) < int(
                0.15 * window[1]):
            if clic[0] == 1:
                doc_on = 0
                pygame.time.wait(300)
                show_MAP(MAP)
    else:
        if abs(mouse[0] - int(0.3 * window[1])) < int(0.15 * window[1]) and abs(mouse[1] - int(0.3 * window[1])) < int(
                0.15 * window[1]):
            fim = pygame.transform.scale(folder_im2, (int(0.3 * window[1]), int(0.3 * window[1])))
            if clic[0] == 1:
                doc_on = 1
                pygame.time.wait(300)
        else:
            fim = pygame.transform.scale(folder_im, (int(0.3 * window[1]), int(0.3 * window[1])))
        fenetre.blit(fim, (int(0.15 * window[1]), int(0.15 * window[1])))


def check_trigger():
    global startmsg, indk, Sline, groupD, v, activatedT
    # print(queueT,startmsg)
    for i in Trig_liste:
        if np.linalg.norm(2 * (i[0] - 50) - x) < 10 and (i[1] not in activatedT):
            Sline = 0
            activatedT.append(i[1])
            queueT.append(i[1])
        if len(queueT) > 0:
            if startmsg == 0 and queueT[0] == i[1]:
                startmsg = len(dicoTEXT[i[1]])
                if "lvl%s_%s.ogg" % (level, i[1]) in os.listdir('son'):
                    s = pygame.mixer.Sound("son/lvl%s_%s.ogg" % (level, i[1]))
                    s.play()

                ref = str(level) + '_' + str(i[1])
                if ref in modif_game:
                    change_game(ref)

            if startmsg != 0 and queueT[0] == i[1]:
                write_message(dicoTEXT[i[1]], len(dicoTEXT[i[1]]) - startmsg)
                startmsg = max(startmsg - 1, 0)
                if startmsg == 0:
                    queueT.remove(i[1])

                if startmsg == 0 and i[1] == level_end[level]:
                    Killed_E[-1] = Killed_E[-1] + Killed_E[0]
                    Killed_O[-1] = Killed_O[-1] + Killed_O[0]
                    Picked_O[-1] = Picked_O[-1] + Picked_O[0]

                    base_map = (-level_map + 2 * authorized_map) / 4

                    Explored[0] = np.sum(MAP) / np.sum(base_map)
                    if level != 0:
                        Accuracy[-1] = (Accuracy[-1] + Accuracy[0] / max(Accuracy[1], 1)) / 2
                        Explored[1] = (Explored[1] + Explored[0]) / 2
                    else:
                        Accuracy[-1] = Accuracy[0] / max(Accuracy[1], 1)
                        Explored[1] = Explored[0]
                    Play_Time[0] = ((dt.datetime.now() - Play_Time[0])).total_seconds()
                    Play_Time[1] = Play_Time[1] + Play_Time[0]

                    pygame.time.wait(1000)
                    stat = stats()
                    if stat:
                        load_level(str(level + 1))
                if startmsg == 0 and i[1] == level_start[level]:
                    v = 1.2

        if startmsg == 0:
            indk = 0
    for i in groupD:
        if np.sum([max(j.vie, 0) for j in thing if j.group == i]) == 0 and ('G' + str(i) not in activatedT):
            Sline = 0
            activatedT.append('G' + str(i))
            queueT.append('G' + str(i))
        if len(queueT) > 0:
            if startmsg == 0 and queueT[0] == ('G' + str(i)):
                startmsg = len(dicoTEXT['G' + str(i)])
                if "lvl%s_G%s.ogg" % (level, i) in os.listdir('son'):
                    s = pygame.mixer.Sound("son/lvl%s_G%s.ogg" % (level, i))
                    s.play()

                ref = str(level) + '_G' + str(i)
                if ref in modif_game:
                    change_game(ref)

            if startmsg != 0 and queueT[0] == ('G' + str(i)):
                write_message(dicoTEXT['G' + str(i)], len(dicoTEXT['G' + str(i)]) - startmsg)
                startmsg = max(startmsg - 1, 0)
                if startmsg == 0:
                    queueT.remove('G' + str(i))

        if np.sum([np.sum(j.U) for j in thing if j.group == i]) > 0 and ('V' + str(i) not in activatedT):
            Sline = 0
            activatedT.append('V' + str(i))
            queueT.append('V' + str(i))
        if len(queueT) > 0:
            if startmsg == 0 and queueT[0] == 'V' + str(i):
                startmsg = len(dicoTEXT['V' + str(i)])
                if "lvl%s_V%s.ogg" % (level, i) in os.listdir('son'):
                    s = pygame.mixer.Sound("son/lvl%s_V%s.ogg" % (level, i))
                    s.play()

                ref = str(level) + '_V' + str(i)
                if ref in modif_game:
                    change_game(ref)

            if startmsg != 0 and queueT[0] == 'V' + str(i):
                write_message(dicoTEXT['V' + str(i)], len(dicoTEXT['V' + str(i)]) - startmsg)
                startmsg = max(startmsg - 1, 0)
                if startmsg == 0:
                    queueT.remove('V' + str(i))

        if startmsg == 0:
            indk = 0
            if 'G' + str(i) in activatedT and 'G' + str(i) not in queueT:
                groupD.remove(i)


def load_level(level_name):
    global SKY0,LAND0,stairs, torch_on, lifts, activatedT, TotAr, MAP, v, tuto, level, groupD, indk, startmsg, activatedT, queueT, linenumber, back, dicoTEXT, Trig_liste, AMMO, level_w, level_h, level_map, zmap, light_wall, hmap, authorized_map, M_liste, light_color, light_array, ratio, level_light, wall, doors, h_wall, thing, ennemies
    level = int(level_name)
    if level>=5:
        SKY0 = pygame.surfarray.pixels3d(pygame.image.load('image/ciel/ciel1.png'))
        LAND0 = pygame.surfarray.pixels3d(pygame.image.load('image/ciel/lanscape1.png'))
        LAND0 = np.where(np.expand_dims(((LAND0 - np.array([0, 255, 255])) == 0).all(-1), -1), -2, LAND0)
    TotAr = level_arme[level]
    v = 0
    skip = False
    activatedT = []
    if not skip:
        b = animation(level - 1)
        if b:
            pygame.time.wait(1000)
    IMLOAD = pygame.transform.scale(pygame.image.load('image/level/' + level_name + '.png'), window)
    fontH = pygame.font.Font('freesansbold.ttf', int(32 * window[0] / (12 * scrnL[0])))
    for ci in range(len(level_nameL[level]) + 1):
        fenetre.fill((0, 0, 0))
        fenetre.blit(IMLOAD, (0, int(0.2 * window[1])))
        text = fontH.render(level_nameL[level][0:ci] + '_', True, (255, 255, 255))
        textRect = text.get_rect()
        fenetre.blit(text, (int(window[0] * 0.05), int(window[1] * 0.05)))
        pygame.time.wait(50)
        pygame.display.flip()
    fenetre.fill((0, 0, 0))
    fenetre.blit(IMLOAD, (0, int(0.2 * window[1])))
    text = fontH.render(level_nameL[level], True, (255, 255, 255))
    textRect = text.get_rect()
    fenetre.blit(text, (int(window[0] * 0.05), int(window[1] * 0.05)))
    pygame.time.wait(50)
    pygame.display.flip()
    pygame.time.wait(3000)
    fenetre.fill((0, 0, 0))
    pygame.display.flip()
    pygame.time.wait(500)
    pygame.mouse.set_pos([window[0] // 2 - 10, window[1] // 2 + 10])
    draw_hud()
    draw_cards()
    draw_vie()
    tuto = 0
    reset_all()
    groupD = []
    activatedT = []
    queueT = []
    indk = 0
    startmsg = 0
    linenumber = 0
    global Killed_E, Killed_O, Picked_O, Accuracy, Explored, L0
    Killed_E[0:-1] = 0
    Killed_O[0:-1] = 0
    Picked_O[0:-1] = 0
    Accuracy[0:-1] = 0
    Explored[0] = 0
    Play_Time[0] = dt.datetime.now()
    L0 = []

    if level > 2:
        torch_on = 1

    AMMO = [0, 0, 0, 0]
    AMMO[0] = max(min(TotAr - 1, 1) * 20, 0)
    AMMO[1] = max(min(TotAr - 2, 1) * 20, 0)
    AMMO[2] = max(min(TotAr - 3, 1) * 20, 0)
    AMMO[3] = max(min(TotAr - 4, 1) * 20, 0)
    draw_AMMO()

    wall = []
    doors = []  # '2'#'Titounet'
    f = open("level/" + level_name, "rb")
    level_w = pickle.load(f)
    level_h = pickle.load(f)
    level_map = pickle.load(f)
    level_map.T
    MAP = np.full((500, 500), 0)
    pickle.load(f)
    pickle.load(f)
    zmap = pickle.load(f)
    light_wall = pickle.load(f)
    pickle.load(f)
    hmap = pickle.load(f)
    pickle.load(f)
    authorized_map = pickle.load(f)
    authorized_map.T
    M_liste = pickle.load(f)
    light_color = pickle.load(f)
    light_array = pickle.load(f)
    Trig_liste = pickle.load(f)
    pickle.load(f)
    lifts = pickle.load(f)
    stairs = pickle.load(f)

    f.close()
    ratio = 2
    level_light = [1, 1, 1]
    text_file = open("level/texts/" + level_name + ".txt", "r")
    dicoTEXT = ast.literal_eval(text_file.read())
    text_file.close()
    back = pygame.Surface((int(0.8 * window[1]), int(10 * window[1])), SRCALPHA)

    for i in level_w:
        xw = list(2 * i[0])
        b = list(2 * i[1])
        H = i[-4]
        xw.append(i[4] * 2 - H)
        b.append(i[5] * 2)
        im = 'image/wall/wall' + str(levelD[level]['wall'][i[2][0]]) + '.png'
        im2 = 'image/wall/wall' + str(levelD[level]['wall'][i[2][1]]) + '.png'
        if i[3] != 0:
            im = 'image/door/' + str(levelD[level]['door'][i[2][0]]) + '.png'
            im2 = 'image/door/' + str(levelD[level]['door'][i[2][1]]) + '.png'

        wall.append(Wall([0., 0, 5 * 2 + H], b, xw, [im, im2, i[2][2]], i[3], i[-3], i[-2], i[-1]))
        if i[3] != 0:
            doors.append(wall[-1])
    [i.texture(5, 5) for i in wall]

    lenH = 0
    for i in level_h:
        a = 2 * i[0]
        b = 2 * i[1]
        x1 = 2 * i[2]
        x2 = 2 * i[2] + [0, 0, 10]
        H = i[3]
        im = 'image/flat/roof' + str(levelD[level]['flat'][i[4]]) + '.png'
        wall.append(Wall(list(a), list(b), list(x1 + [0, 0, -H]), [im, im, i[2][2]], 0, 0, 1, 0))
        if i[4] == 9:
            wall[-1].transp = 1

        if str(levelD[level]['flat'][i[4]]) in ['7', '10']:  # WARNING TO FINISH
            lenH += 1
            im = 'image/flat/floor' + str(6) + '.png'
            wall.append(Wall(list(-a), list(b), list(a + x2), [im, im, i[2][2]], 0, 0, 1, 0))
            im = 'image/flat/floor' + str(levelD[level]['flat'][i[4]]) + '.png'
            wall.append(Wall(list(-a), list(b), list(a + x2 + [0, 0, -2]), [im, im, i[2][2]], 0, 0, 1, 0))
        else:
            im = 'image/flat/floor' + str(levelD[level]['flat'][i[4]]) + '.png'
            wall.append(Wall(list(-a), list(b), list(a + x2), [im, im, i[2][2]], 0, 0, 1, 0))

    h_wall = []
    [i.texture(2 + int(np.linalg.norm(i.a) / 500), 2 + int(np.linalg.norm(i.b) / 500)) for i in
     wall[-2 * len(level_h) - lenH:]]
    h_wall = wall[-2 * len(level_h) - lenH:]
    zmap = zmap.T
    thing = []
    if level == 3:
        L0.append(LIZARD(np.array([144, 144]), 0, x, scrnL, ang, zmap, level, 700))
        thing.append(L0[-1])
        L0[0].start_pattern()
        groupD.append(99)

    ennemies = []
    difficulty_var=[0.,0.,0.,0.]
    for i in M_liste:
        if i[-1] not in groupD:
            groupD.append(i[-1])
        if i[-2]:
            monst = Thing(2 * (i[0] - 50), int(i[1]), 1, i[-1])
            thing.append(monst)
            ennemies.append(monst)
            Killed_E[1] = Killed_E[1] + 1
            difficulty_var[0]+=monst.vie
            if monst.attack_range:
                difficulty_var[1]+=M_charac['degat'][monst.type_M]/24
            else:
                difficulty_var[1] += M_charac['degat'][monst.type_M]/12
        else:
            objet = Object(2 * (i[0] - 50), int(i[1]), 1, i[-1])
            objet.orient=i[3]
            objet.angle = i[3]
            thing.append(objet)
            if thing[-1].type_M in destr:
                Killed_O[1] = Killed_O[1] + 1
            if thing[-1].type_M in [1, 2, 5]:
                Picked_O[1] = Picked_O[1] + 1
            if thing[-1].type_M == 2 or thing[-1].type_M == 5:
                thing[-1].color = (i[-4]) + 1
            if thing[-1].type_M==2:
                difficulty_var[2]+=10*DEGAT[objet.color]/COOLDOWN[objet.color]
            if thing[-1].type_M == 1:
                difficulty_var[3] +=20

    if 0 in groupD:
        groupD.remove(0)

SKY0 = pygame.surfarray.pixels3d(pygame.image.load('image/ciel/ciel0.png'))
LAND0 = pygame.surfarray.pixels3d(pygame.image.load('image/ciel/lanscape0.png'))
LAND0 = np.where(np.expand_dims(((LAND0 - np.array([0, 255, 255])) == 0).all(-1), -1), -2, LAND0)
BLOOD = pygame.transform.scale(pygame.image.load('image/effects/blood.png'), window)
code01 = pygame.image.load('image/Interface/code.png')
code02 = pygame.image.load('image/Interface/code2.png')
shield=pygame.image.load('image/effects/shield.png').convert_alpha()
ttt = []
bleed = 0
explo = 0
explo_pt = np.array([0., 0.])
Sky_view = 0
ang = (1e-5, 0)

draw_hud()
L0 = []
load_level(str(level))
pygame.mixer.music.play(-1)
Im = np.full((2 * scrnL[0], 2 * scrnL[1], 3), 0).astype(np.float32)
depth = np.full((2 * scrnL[0], 2 * scrnL[1], 1), 100.0).astype(np.float32)
explo_R=np.full((2 * scrnL[0], 2 * scrnL[1], 1), 100.0).astype(np.float32)
horizon = np.full((scrnL[0], 1), 10000.0).astype(np.float32)
POS = np.full((2 * scrnL[0], 2 * scrnL[1]), 1000.0).astype(np.float32)
code_show = 0
code_show2 = 0
code_num = 0
code_cool = 0
c3 = 0
lift_msg = False
liftC = 0
yg = 0
xg = 0
logL = []
C_log = 0
impact = pygame.image.load('./image/effects/impact0.png')
averaged_time = np.full((21), 0.)
elastic_count=0
Ratio=window[0]/960
x_d=[(0.,0.)]
nb_wall = []
time_wall = []
while running == 1:
    moving_cam = False
    milliseconds = [time.time() * 1000]
    fire = 0
    key = pygame.key.get_pressed()
    if key[K_0]:
        load_level('99')
        v = 1.2
    Rp = rot_plan(ang[0])
    c = (c + 1) % (4 * 3)
    c2 = (c2 + 1) % 24
    c3 = (c3 + 1) % 10000
    render_w = 0
    render_w2 = 0
    render_C = 0
    check_trigger()
    Im = Im * 0
    depth = depth * 0 + 100.
    horizon = horizon * 0 + 10000.
    POS = POS * 0 + 1000.
    explo_R=explo_R*0+1000
    trans = np.array([0.0, 0.0])
    HIT = 0
    rset_L = 0
    recoil = 0.
    milliseconds.append(time.time() * 1000)
    label_deltat = ['reset']
    for j, i in enumerate(stairs):
        if abs(2 * (i[0] - 50) - x[0]) < 5 and abs(2 * (i[1] - 50) - x[1]) < 5:
            if j % 2 == 0:
                nextS = stairs[j + 1]
                trans = 2 * (-i + nextS)
                x = x + trans
                z = zmap[int(x[1] + 100) // 2][int(x[0] + 100) // 2]
                screen[:, :, :3] = screen[:, :, :3] + np.hstack((trans, [0]))
                screen[:, :, 2] = z * 2
                screen0[:, :, :3] = screen0[:, :, :3] + np.hstack((trans, [0]))
                screen0[:, :, 2] = z * 2
                zprev = z
                R_c = np.hstack((x, [2 * z]))
                trans = np.array([0, 0])
                moving_cam = True
                [i.calc_norm() for i in wall]

    for j, i in enumerate(lifts):
        if abs(2 * (i[0] - 50) - x[0]) < 5 and abs(2 * (i[1] - 50) - x[1]) < 5:
            lift_msg = True

            rset_L += 1
            if j % 2 == 0 and key[K_SPACE] and liftC == 0:
                s = pygame.mixer.Sound("son/lift.ogg")
                s.play()
                nextL = lifts[j + 1]
                trans = 2 * (-i + nextL)
                liftC = 20

            if j % 2 == 1 and key[K_SPACE] and liftC == 0:
                s = pygame.mixer.Sound("son/lift.ogg")
                s.play()
                nextL = lifts[j - 1]
                trans = 2 * (-i + nextL)
                liftC = 20

            if key[K_SPACE] and liftC == 20:
                x = x + trans
                z = zmap[int(x[1] + 100) // 2][int(x[0] + 100) // 2]
                screen[:, :, :3] = screen[:, :, :3] + np.hstack((trans, [0]))
                screen[:, :, 2] = z * 2
                zprev = z
                R_c = np.hstack((x, [2 * z]))
                trans = np.array([0, 0])
                moving_cam = True
                [i.calc_norm() for i in wall]
            liftC = max(0, liftC - 1)

    if rset_L == 0:
        nextL = [0, 0]
    milliseconds.append(time.time() * 1000)
    label_deltat.append('lifts')
    wall.sort(key=lambda s: s.norm)
    doors.sort(key=lambda s: s.norm)
    h_wall.sort(key=lambda s: s.norm)
    thing.sort(key=lambda s: s.norm)
    ennemies.sort(key=lambda s: s.norm)
    milliseconds.append(time.time() * 1000)
    label_deltat.append('sorting')

    for event in pygame.event.get():
        if event.type == QUIT:
            running = 0
        if event.type == MOUSEBUTTONUP:
            if event.button == 4:
                if mouse_c == 0:
                    Sline = max(Sline - 1, -linenumber)
                    show_message()
                    pygame.display.flip()
                else:
                    arme = (arme + 1) % TotAr
                    GUN_im = MG1[arme][0].copy()
                    colorGUN = (1., 1., 1.)
            if event.button == 5:
                if mouse_c == 0:
                    Sline = min(Sline + 1, 0)
                    show_message()
                    pygame.display.flip()
                else:
                    arme = (arme - 1) % TotAr
                    GUN_im = MG1[arme][0].copy()
                    colorGUN = (1., 1., 1.)
    clic = pygame.mouse.get_pressed()
    mouse = pygame.mouse.get_pos()
    if not ((mouse == m_stock).all()):
        moving_cam = True

    milliseconds.append(time.time() * 1000)
    label_deltat.append('events')

    if key[K_KP_PLUS] == 1:
        window = [int(window[0] * 1.2), int(window[1] * 1.2)]
        m_stock = np.array([window[0] // 2, window[1] // 2])
        fenetre = pygame.display.set_mode((window[0], int(window[1] * 1.2)))
        rescale_Gun()
        draw_hud()
        draw_cards()
        show_message()
        draw_vie()
        draw_AMMO()
        Ratio = window[0] / 960
    if key[K_KP_MINUS] == 1:
        window = [int(window[0] * 0.8), int(window[1] * 0.8)]
        m_stock = np.array([window[0] // 2, window[1] // 2])
        fenetre = pygame.display.set_mode((window[0], int(window[1] * 1.2)))
        rescale_Gun()
        draw_hud()
        draw_cards()
        show_message()
        draw_vie()
        draw_AMMO()
        Ratio = window[0] / 960

    if clic[2] == 1:
        mouse_c = 0
        shiftM = np.array([0., 0.])
        pygame.mouse.set_visible(1)
        update_map(level_map, x)
        doc_on = 0
        show_MAP(MAP)
    if clic[0] == 1:
        if coolD == 0 and mouse_c == 1:
            attack = 1
        if mouse_c == 0:
            coolD = 10

        if not (abs(mouse[0] - int(0.3 * window[1])) < int(0.15 * window[1]) and abs(
                mouse[1] - int(0.3 * window[1])) < int(0.15 * window[1])) and doc_on == 0:
            if mouse_c == 0:
                m_stock = np.array(mouse)
            mouse_c = 1
            pygame.mouse.set_visible(0)

    if mouse_c == 0:
        if key[K_UP] or key[K_z]:
            shiftM = shiftM + [-1, 0.]
        if key[K_DOWN] or key[K_s]:
            shiftM = shiftM + [1, 0.]
        if key[K_RIGHT] or key[K_d]:
            shiftM = shiftM + [0., -1.]
        if key[K_LEFT] or key[K_q]:
            shiftM = shiftM + [0, 1.]

        if doc_on == 0:
            update_map(level_map, x)
            show_MAP(MAP)

        show_doc()
        pygame.display.flip()
        continue

    milliseconds.append(time.time() * 1000)
    label_deltat.append('clicks_and_keys')
    if attack == 0:
        GUN_im = MG1[arme][0].copy()
        colorGUN = (1., 1., 1.)
    if attack and (AMMO[arme - 1] > 0 or arme == 0):
        moving_cam = True
        if arme != 0:
            shoot = (shoot + 1) % 3
        else:
            shoot = (shoot + 0.5) % 3
        GUN_im = MG1[arme][shoot // 2 + 1].copy()
        if not (np.sum(colorGUN) == 0 and torch_on == 1):
            GUN_im.fill(colorGUN, special_flags=BLEND_RGB_MULT)
        if shoot == 0:
            attack = 0

        if shoot == 1:
            s = pygame.mixer.Sound("son/gun%s.ogg" % (arme))
            s.play()
            if arme != 0 and arme != 4:
                Accuracy[1] = Accuracy[1] + 1
                AMMO[arme - 1] = max(AMMO[arme - 1] - 1, 0)
                draw_AMMO()
                fire = 1
                recoil = REC[arme]
            if arme == 4:
                Boule.append(grenade(x[0], x[1], z, -ang[1] + pi / 2, -ang[0], 1.5,1., 50))
                AMMO[arme - 1] = max(AMMO[arme - 1] - 1, 0)

        coolD = COOLDOWN[arme]

    if AMMO[arme - 1] <= 0 and arme != 0:
        shoot = 0
    coolD = max(coolD - 1, 0)

    if key[K_UP] or key[K_z]:
        moving_cam = True
        yg = (yg + 1) % 100
        trans = trans + [-v, 0.]
    if key[K_DOWN] or key[K_s]:
        moving_cam = True
        yg = (yg + 1) % 100
        trans = trans + [v, 0.]
    if key[K_RIGHT] or key[K_d]:
        moving_cam = True
        xg = (xg + 1) % 100
        trans = trans + [0., -v]
    if key[K_LEFT] or key[K_q]:
        moving_cam = True
        xg = (xg + 1) % 100
        trans = trans + [0., v]
    trans = trans + [recoil, 0.]
    recoil = 0
    if trans.any() != np.array([0.0, 0.0]).any():
        No = np.array([0., 0.])
        for i in wall:
            if i not in h_wall:
                if i.norm3 > 3:
                    break
                if (i.door and i.closed) or not i.door:
                    trans = i.normal(trans @ Rp) @ rot_plan(-ang[0])

        x = x - trans @ Rp
        if authorized_map[int(x[1] + 100) // 2][int(x[0] + 100) // 2]!=0:
            last_ok_pos = x
        else:
            x=last_ok_pos
            trans=trans*0.

        z = zmap[int(x[1] + 100) // 2][int(x[0] + 100) // 2]
        screen[:, :, :3] = screen[:, :, :3] - np.hstack((trans @ Rp, [0]))
        screen[:, :, 2] = z * 2
        screen0[:, :, :3] = screen0[:, :, :3] - np.hstack((trans @ Rp, [0]))
        screen0[:, :, 2] = z * 2
        zprev = z
        R_c = np.hstack((x, [2 * z]))

    if mouse_c == 1:
        rot_a = -(2 * pi * (mouse - m_stock) / 500) * [1, -0.2]  # 1000
        if abs((ang + rot_a)[1]) < pi / 8:
            ang = ang + rot_a
        screenV = screen[:, :, 3:] @ rot_y(ang[1]) @ rot_z(ang[0])
        screenP = (screen[:, :, :3] - R_c) @ rot_y(ang[1]) @ rot_z(ang[0]) + R_c

        screenV0 = screen0[:, :, 3:] @ rot_y(ang[1]) @ rot_z(ang[0])
        screenP0 = (screen0[:, :, :3] - R_c) @ rot_y(ang[1]) @ rot_z(ang[0]) + R_c

        pygame.mouse.set_pos([window[0] // 2, window[1] // 2])
    m_stock = np.array([window[0] // 2, window[1] // 2])
    Rp = rot_plan(ang[0])

    milliseconds.append(time.time() * 1000)
    label_deltat.append('gun_clicks_and_keys')

    # print(time.time()*1000-milliseconds[0])------until there same time small and big

    [i.calc_norm() for i in wall[0:10]]
    if c2 == 0:
        [i.calc_norm() for i in wall]
    [i.calc_norm() for i in thing[0:20] if i.type_M != 'BOSS']
    if c2 == 23:
        [i.calc_norm() for i in thing if i.type_M != 'BOSS']

    milliseconds.append(time.time() * 1000)
    label_deltat.append('calc_norm')

    CLOSED = 0
    code_show = 0
    for i in doors[0:5]:
        Ry0 = rot_y(-ang[1])
        Ry1 = rot_y(ang[1])
        Ri = i.X
        shift = i.z + 5
        if i.norm < 6 and i.door >= 10000:
            code_show = 1

        if i.norm < 6 and (
                i.door == 1 or (CARTE[min(i.door - 2, 2)] > 0 and i.door - 2 < 3) or (10000 + code == i.door)):
            if CARTE[min(i.door - 2, 2)] > 0 and i.door - 2 < 3 and i.door != 1:
                CARTE[i.door - 2] -= 1
                draw_cards()
                i.opendoor(i.door)
                i.door = 1
                s = pygame.mixer.Sound("son/unlock.ogg")
                s.play()
                logL.append('door unlocked')

            if (10000 + code == i.door):
                code_show2 += 1
                if code_show2 > 10:
                    i.door = 1
                    logL.append('door unlocked')
                    code_show2 = 0
                    code_show = 0

            i.X = np.maximum(Ri - [0., 0., 3], [-10000, -10000, -14.5 + shift])
            if i.closed:
                porteson.play()
            i.closed = 0
        else:
            if (Ri)[0][0][-1] < -5. + shift:
                i.X = np.minimum(Ri + [0., 0., 3], [10000, 10000, -5 + shift])
            if (Ri)[0][0][-1] < -4.5 + shift:
                i.closed = 1
        for j in ennemies[0:5]:
            if j.active and j.vie > 0 and level_map[int(j.x0[1] + 101) // 2][int(j.x0[
                                                                                     0] + 101) // 2] == 1:  # np.sum(level_map[int(j.x0[1]+101)//2-1:int(j.x0[1]+101)//2+1,int(j.x0[0]+100)//2-1:int(j.x0[0]+100)//2+1]==1 )>0:
                Ri = i.X
                shift = i.z + 5
                if ((j.x0[0] - (0.5 * i.b[0][0][0] + i.X[0][0][0])) ** 2 + (j.x0[1] - (0.5 * i.b[0][0][1] + i.X[0][0][
                    1])) ** 2) ** 0.5 < 10:  # LIMITATION PORTE TAILLE 2 FULL CASES EDITEUR ET DISTANCE D'UNE FULL CASE ENTRE DEUX PORTES
                    i.X = np.maximum(Ri - [0., 0., 9], [-10000, -10000, -14.5 + shift])

                    # if i.closed:
                    # porteson.play()
                    i.closed = 0

        CLOSED += 1 - i.closed

    milliseconds.append(time.time() * 1000)
    label_deltat.append('doors')
    wall_count=min(40+elastic_count,70)
    if explo!=0:
        moving_cam=True

    render_arr=[]
    if moving_cam == True:

        Sky_view = 0
        IS = []
        empty_pixel_count=128000
        for i in wall[0:wall_count]:
            devant = True
            if i not in h_wall:
                devant = i.test_behind()
                if i.window > 0 and devant:
                    CLOSED = 1
                    Sky_view = 1
            else:
                if i.norm > 6 and i.text[11:-3] not in liquid_floor:
                    devant = False
                    if CLOSED != 0:  # and h_wall.index(i)<=6: # INSTEAD CHECK IF ASSOCIATED DOOR WITH THIS FLOOR IS OPEN AND VISIBLE---COMPLICATED
                        devant = True
            render_w2 += 1
            render_arr.append(devant)
            if devant:
                render_w += 1
                # if i.text[11:-3] not in liquid_floor:
                #     Im = i.render() + Im * (1 - np.expand_dims(i.U, -1))
                #     # plt.imshow(Im/255)
                #     # plt.show()
                #     empty_pixel_count = np.sum((np.sum(Im[3:-3, 3:-3], axis=-1) == 0).astype(int))
                #     if render_w==1:
                #         time_in_render=i.time[0]
                #         label_t_render=i.time[1]
                #     else:
                #         time_in_render+=i.time[0]
                #
                #
                # else:
                #     IS.append(i)
                #
                #
                # if empty_pixel_count < 50 or i.norm > 150:
                #     break
        if render_w2>wall_count-10:
            elastic_count+=10
        else:
            elastic_count = max(elastic_count-10,0)



    if moving_cam == True:
        Im_cached = Im
        depth_cached = depth
        POS_cached = POS
        horizon_cached = horizon
    else:
        Im = Im_cached
        depth = depth_cached
        POS = POS_cached
        horizon = horizon_cached

    milliseconds.append(time.time() * 1000)
    label_deltat.append('walls')

    #NUMBA ATTEMPT
    wall_a=[]
    wall_b = []
    wall_X = []
    formatL=[]
    phaseL=[]
    borneL=[]
    wall_imL=List()
    lightL=List()
    WN=0
    for ci,i in enumerate(wall[0:min(wall_count,len(render_arr))]):
        if render_arr[ci]:
            wall_a.append(i.a[0,0,:])
            wall_b.append(i.b[0, 0, :])
            wall_X.append(i.X[0, 0, :])
            formatL.append(i.format)
            phaseL.append(i.phase)
            borneL.append(i.borne)
            wall_imL.append(i.wall_im)
            WN+=1
            Y0 = [np.linalg.norm(source_pos(i) - R_c) for i in light_wall[i.ID]]
            X0 = [x for _, x in sorted(zip(Y0, light_wall[i.ID]))]
            lightL.append(np.array([source_pos(i) for i in X0][:4]))

    # xx=render_ray_step_sign(scrnL[0],scrnL[1],10,np.array(wall_a),np.array(wall_b),np.array(wall_X),screenV0, screenP0, R_c0)
    # milliseconds.append(time.time() * 1000)
    # label_deltat.append('numba attempt ray tracing deactivated')
    # print(render_arr)

    if moving_cam == True:

        xx,pos_,depth_=render_numba(scrnL[0], scrnL[1], WN, np.array(wall_a), np.array(wall_b), np.array(wall_X), screenV0,screenP0, np.array(formatL), np.array(phaseL), np.array(borneL),c3,wall_imL,lightL)
        Im=xx
        POS=pos_
        depth=np.expand_dims(depth_,axis=-1)
    # if c3%100==1:
    #
    #     plt.imshow(np.minimum(np.maximum(xx,0),255)/255)
    #     plt.show()
    #     plt.imshow(pos_)
    #     plt.show()
    #     plt.imshow(depth_)
    #     plt.show()
    time_in_render = time.time() * 1000-milliseconds[-1]
    milliseconds.append(time.time() * 1000)
    label_deltat.append('numba attempt solving')
    #NUMBA ATTEMPT



    if level in [2]:
        if randint(0, 100) == 0:
            level_light = np.array([1, 1, 1.]) * random()
        else:
            level_light = np.minimum(level_light + 0.1 * np.array([1, 1, 1.]), np.array([1, 1, 1.]))

    if Sky_view:
        LAND = np.roll(LAND0, int(ang[0] * 12 * scrnL[0] / (2 * pi)), axis=0)
        LAND = LAND[:2 * scrnL[0], :, :]
        LAND = np.roll(LAND, -int(tan(ang[1]) * scrnL[1] / TAN1 + scrnL[1]), axis=1)
        LAND = LAND[:, :2 * scrnL[1], :]

        SKY = np.roll(SKY0, int(ang[0] * 6 * scrnL[0] / (2 * pi)), axis=0)
        SKY = SKY[:2 * scrnL[0], :, :]
        SKY = np.roll(SKY, -int(tan(ang[1]) * scrnL[1] / TAN1 + scrnL[1]), axis=1)
        SKY = SKY[:, :2 * scrnL[1], :]
        POS = POS * ((Im > -1).all(-1)) + ((Im <= -1).all(-1)) * 10

        Im = np.where(np.expand_dims(((Im > 0).all(-1)), -1), Im, LAND)
        Im = np.where(np.expand_dims(((Im > 0).all(-1)), -1), Im, SKY)

    milliseconds.append(time.time() * 1000)
    label_deltat.append('sky')
    Im = np.minimum((0.8 * (np.divide(Im, 0.01 * np.expand_dims((4 * POS) ** 2, -1)) + np.divide(Im,0.1 * np.expand_dims((POS),-1))) + 0.2 * Im),255)  # ,100)+np.divide(200,np.maximum((depth/5)**2,1))
    if explo!=0:
        a_e = np.full((scrnL[0], scrnL[1], 3), np.array([0,0,1.]))
        b_e = np.full((scrnL[0], scrnL[1], 3), np.array([sin(ang[0]),cos(ang[0]),0.]))
        X_e1 = np.full((scrnL[0], scrnL[1], 3), np.array([explo_pt[0],explo_pt[1],0.]))
        S=plane(a_e,b_e,X_e1)
        S=S.repeat(2,axis=0)
        S = S.repeat(2, axis=1)
        D_e=np.expand_dims(((S[:, :, 0]) ** 2 + (S[:, :, 1]) ** 2)**0.5,-1)
        Im = np.where((explo_R < 4 * explo+np.random.randint(-1,1,explo_R.shape)), (0.5*255+Im *0.5)* explo_zone(4*explo,explo_R), Im)
        Im = np.where((np.expand_dims(S[:, :, 2],-1)<depth)& (D_e<(4*explo)+np.random.randint(-1,1,explo_R.shape)), (0.5*255+Im *0.5)* explo_zone(4*explo,D_e), Im)

        # if explo==4:
        #     plt.imshow(Im/255)
        #     plt.show()



    horizon = horizon * np.expand_dims(np.linalg.norm(screen[:, 20, 3:], axis=-1), -1)

    milliseconds.append(time.time() * 1000)
    label_deltat.append('lights')

    # ttt0=time.time()
    [i.move() for i in ennemies[0:10]]

    milliseconds.append(time.time() * 1000)
    label_deltat.append('movethings')

    if c2 == 23:
        update_map(level_map, x)
    if c % 3 == 0:
        [i.walk() for i in ennemies[0:20]]
    milliseconds.append(time.time() * 1000)
    label_deltat.append('walkthings')
    if len(L0) > 0:
        Deg, Boul = L0[0].pattern(x, scrnL, c, ang, TAN1, TAN2, z, authorized_map, horizon, zmap)
        if Deg != 0:
            VIE = VIE - Deg
            HIT = 1
            s = pygame.mixer.Sound("son/aie.ogg")
            s.play()
        if len(Boul) > 0:
            for j in Boul:
                Boule.append(boule(*j))

    milliseconds.append(time.time() * 1000)
    label_deltat.append('boss')
    if shoot==1 and arme!=0 and arme!=4:
        a_shift=np.random.normal(0,PREC[arme],2)
        x_d=[(a_shift[0]/(pi / 4),a_shift[1]/(atan(1 / 2) * 2))]
        if arme==2:
            for i in range(4):
                a_shift = np.random.normal(0, PREC[arme], 2)
                x_d.append((a_shift[0] / (pi / 4), a_shift[1] / (atan(1 / 2) * 2)))
        depth_=depth.shape
        depth1=depth[int(depth_[0]*(x_d[0][0]+0.5))][int(depth_[1]*(x_d[0][1]+0.5))]
    if shoot == 1 and arme == 0:
        x_d=[(0,0)]
    y_d=x_d.copy()
    for i in thing:
        if i.type_M != 'BOSS':
            if i.norm > np.percentile(depth, 95):
                break
            if i.test_behind():
                render_C += 1
                Im = i.render() * np.expand_dims(i.Ut, -1) + Im * (1 - np.expand_dims(i.Ut, -1))
                depth = depth * (1 - np.expand_dims(i.Ut, -1)) + np.expand_dims(i.Ut, -1) * i.norm
    milliseconds.append(time.time() * 1000)
    label_deltat.append('things')
    if len(L0) > 0:
        if L0[0].test_behind(TAN2, scrnL, horizon):
            Im = L0[0].render(depth, Xthing, Ything, scrnL, light_array, level_light, TORCHE, torch_on, arme, shoot,
                              explo, explo_pt, DEGAT, c3) * np.expand_dims(L0[0].Ut, -1) + Im * (
                             1 - np.expand_dims(L0[0].Ut, -1))
            depth = depth * (1 - np.expand_dims(L0[0].Ut, -1)) + np.expand_dims(L0[0].Ut, -1) * L0[0].norm


    if colorGUN != tuple(255 * light_array[int(x[0] + 100) // 2][int(x[1] + 100) // 2]):
        colorGUN = tuple(255 * light_array[int(x[0] + 100) // 2][int(x[1] + 100) // 2])
        GUN_im = MG1[arme][0].copy()
        if not (np.sum(colorGUN) == 0 and torch_on == 1):
            GUN_im.fill(colorGUN, special_flags=BLEND_RGB_MULT)
    IS_rendered=[]
    for i in IS:
        IS_rendered.append(i.render())
        Im = IS_rendered[-1] * 0.5 + Im * (1 - 0.5 * np.expand_dims(i.U, -1))
    if fire:
        Im = np.minimum(Im + 100 * TORCHE, 255)

    Im=np.maximum(Im,0)

    fond = pygame.surfarray.make_surface(Im[3:-3, 3:-3])
    if setting['smooth'] == True:
        fond = pygame.transform.smoothscale(fond, window)
    else:
        fond = pygame.transform.scale(fond, window)

    milliseconds.append(time.time() * 1000)
    label_deltat.append('rendering')

    for i in Boule:
        KILL = i.update()
        if KILL:
            Boule.remove(i)
    [i.affiche() for i in Boule]
    milliseconds.append(time.time() * 1000)
    label_deltat.append('boule')


    if shoot==1 and arme!=0 and arme!=4:
        for i in range(len(y_d)):
            depth1=depth[int(depth_[0]*(y_d[i][0]+0.5))][int(depth_[1]*(y_d[i][1]+0.5))]
            impact2=pygame.transform.scale(impact,(int(Ratio*400/depth1),int(Ratio*400/depth1)))
            impact2=pygame.transform.rotate(impact2,360*random())
            fond.blit(impact2,(int(window[0]//2-Ratio*200/depth1+y_d[i][0]*window[0]),int(window[1]//2-Ratio*200/depth1+y_d[i][1]*window[1])))
    fond.blit(GUN_im, ((int(Ratio*shift_a[arme]+-10*Ratio * (cos(2 * pi * xg / 20)) ** 2), int(10*Ratio * (cos(2 * pi * yg / 20)) ** 2))))

    if HIT:
        bleed = 20
        draw_vie()
    if bleed != 0:
        BLOOD0 = BLOOD.copy()
        BLOOD0.fill((255, 255, 255, 255 * bleed / 20), special_flags=BLEND_RGBA_MULT)
        fond.blit(BLOOD0, (0, 0))
        bleed = max(bleed - 1, 0)
    if explo != 0:
        fond.fill((40 * explo, 40 * explo, 40 * explo), special_flags=BLEND_RGB_ADD)
        explo = max(explo - 1, 0)

    milliseconds.append(time.time() * 1000)
    label_deltat.append('bleed, explo, effects')

    fenetre.blit(fond, (0, 0))

    milliseconds.append(time.time() * 1000)
    label_deltat.append('blit image')

    if code_show:
        if code_show2 == 0:
            if code_num == 0:
                code = 0
            Number = return_num(key)
            code_cool = max(code_cool - 1, 0)
            if Number != -1 and code_cool == 0:
                code = code + Number * (10 ** (3 - code_num))
                code_num = (code_num + 1) % 4
                code_cool = 15

        fenetre.blit(code1, (int(0.5 * window[0] - 0.4 * window[1]), int(0.84 * window[1])))
        col_code = (255, 255, 255)
        for i in range(4):
            if i >= code_num:
                col_code = (100, 100, 100)
            text = fontC.render(str((code // (10 ** (3 - i))) % 10), True, col_code)
            fenetre.blit(text, (
            int(0.5 * window[0] + (-2.5 + i) * 0.16 * window[1] + 0.17 * window[1]), int(0.86 * window[1])))

    if code_show2 > 0:
        fenetre.blit(code2, (int(0.5 * window[0] - 0.4 * window[1]), int(0.84 * window[1])))
        for i in range(4):
            text = fontC.render(str((code // (10 ** (3 - i))) % 10), True, (255, 255, 255))
            fenetre.blit(text, (
            int(0.5 * window[0] + (-2.5 + i) * 0.16 * window[1] + 0.17 * window[1]), int(0.86 * window[1])))

    if tuto != 0:
        write_text_msg(tutotxt[tuto - 1])
    if lift_msg:
        write_text_msg('press space to take the lift')
        lift_msg = False

    if len(logL) > 0:
        for c1 in range(len(logL)):
            write_log(logL[c1], c1)
        C_log = (C_log + 1) % 100
        if C_log == 0:
            logL.pop(0)

    pygame.display.flip()

    if v != 0:
        v = 1.2 * max(24 * (-milliseconds[0] + time.time() * 1000) / 1000, 1)

    milliseconds.append(time.time() * 1000)
    label_deltat.append('end')

    if c3 % 1000 == 2:
        milliT = np.expand_dims(milliseconds, -1)
    else:
        if c3!=1:
            milliT = np.concatenate((milliT, np.expand_dims(milliseconds, -1)), axis=1)

    averaged_time = averaged_time + np.round((np.array(milliseconds) - np.roll(np.array(milliseconds), 1))[1:], 1)
    if render_w!=0 and time_in_render<100:
        nb_wall.append(render_w)
        time_wall.append(time_in_render)
    if c3 % 1001 == 1000:
        averaged_time = np.round(averaged_time / 1000, 1)
        milliseconds = np.mean(milliT, axis=-1)
        timelist = averaged_time#np.round((np.array(milliseconds) - np.roll(np.array(milliseconds), 1))[1:], 1)
        # for i in range(len(label_deltat)):
        # 	print(timelist[i],'ms | ',round(100*timelist[i]/np.sum(timelist),1),'%',label_deltat[i])
        sortingtime = [(x, str(y) + ' ms', str(round(100 * y / np.sum(timelist), 1)) + ' %') for y, x in
                       sorted(zip(timelist, label_deltat))]
        sortingtime.reverse()
        print('Top contribution', sortingtime[:3])
        # print('framerate', round(1000 / (milliseconds[-1] - milliseconds[0]), 1), '/s | total_time ',
        #       round(np.sum(timelist), 1), 'ms/ (goal is 42 ms) | vie', VIE, '%')
        print('average,framerate', round(1000 / np.sum(averaged_time), 1), '/s | average time',
              round(np.sum(averaged_time), 1), 'ms')
        print(sortingtime)
        # print('detail on wall rendering',np.sum(time_in_render),time_in_render, label_t_render)
        print('*-----------*')
        ttt = []
        averaged_time = np.full((21), 0.)
        plt.scatter(nb_wall,time_wall,color='blue')
        plt.show()
        nb_wall=[]
        time_wall=[]

    clock.tick_busy_loop(24)

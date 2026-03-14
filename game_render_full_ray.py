import matplotlib.pyplot as plt
import numpy as np
from level_data import *
from math import *
import time
import pygame  #current python 3.8.8 numpy 1.20.1 pygame 2.0 attention pour numpy à avoir la bonne version blas/lapack                   python 3.7.1, pygame 1.9.4
from pygame.locals import *
import pickle
import cv2
pygame.init()
import os
# import torch
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from random import random, randint
from PATH import astar, Node
import ast
import datetime as dt
from Boss import BOSS, LIZARD, nearest_valid, rot_z, rot_y, rot_plan, light_modif
from numba.typed import List
from numba import types
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
height_list=[]
horizon2 = np.full((scrnL[0], len(height_list)), 10000.0)
Ang = np.full((*scrnL, 2), 0.0)
I = np.indices(scrnL)
I_n = np.divide(np.moveaxis((np.indices(scrnL)), 0, 2) - 0.5 * scrnL, scrnL[1])
screen[:, :, 1:3] = I_n
Ang = I_n
Ang[:, :, 0] = Ang[:, :, 0] * pi / 4
Ang[:, :, 1] = Ang[:, :, 1] * atan(1 / 2) * 2
gun_width=1
shift_a=[0,25,0,10,0]


screen[:, :, 0] = screen[:, :, 0]

screen[:, :, 4] = np.sin(Ang[:, :, 0])
screen[:, :, 5] = np.cos(-Ang[:, :, 1] + pi * 0.5)
screen[:, :, 3] = 1

TAN1 = np.amax(screen[:, :, 5])
TAN2 = np.amax(screen[:, :, 4])

screenV = screen[:, :, 3:]
screenP = screen[:, :, :3]



CENTER = np.expand_dims(np.linalg.norm(screen[:, :, :3] - [0, 0, 0], axis=-1).repeat(2, axis=0).repeat(2, axis=1), -1)
TORCHE = np.expand_dims(
    (np.maximum(np.cos(I_n[:, :, 0] * pi / 2) * np.cos(I_n[:, :, 1] * 2 * pi / 2), 0)).repeat(4, axis=0).repeat(4,
                                                                                                                axis=1),
    -1)
torch_on = 0
TORCHE3=TORCHE ** 3

R_c = np.array([-1, 0, 0])
Vg = np.array([1, -sqrt(2) / 2]) / sqrt(3 / 2)
Vd = np.array([1, sqrt(2) / 2]) / sqrt(3 / 2)

height_list=[]

setting = {}
setting['smooth'] = False
destr = [0, 4, 6,11]
level = 6
level_nameL = ['Level 0: Training', 'Level 1: The Lab', 'Level 2: The Storage', 'Level 3: The Basement',
               'Level 4: The Manor','Level 5: The Caves','Level 6: The Floating Boat']
level_arme = [1, 2, 2, 2, 3,4,5]  # last 3
level_end = [5, 7, 8, 6, 6,10,99]
level_start = [1, 2, 1, 1, 1,1,1]
for i in range(100 - len(level_arme)):
    level_arme.append(5)
    level_end.append(0)
    level_start.append(100)
    level_nameL.append('no')
level_end[99] = 10
level_start[99] = 4

z_tileable_deco = [26, 28, 31]
Im = np.full((2 * scrnL[0], 2 * scrnL[1], 3), 0)
def create_cell_array(cell_size):
    n = 500 // cell_size

    cell_array = List()  # outer list

    for i in range(n):
        row = List()  # second level
        for j in range(n):
            row.append( List.empty_list(types.int64))  # inner empty list
        cell_array.append(row)

    return cell_array
from numba import njit, prange
import numpy as np

def build_cell_csr(cell_array):

    nx = len(cell_array)
    ny = len(cell_array[0])

    n_cells = nx * ny

    # count total objects
    total = 0
    for cx in range(nx):
        for cy in range(ny):
            total += len(cell_array[cx][cy])

    cell_objects = np.empty(total, dtype=np.int32)
    cell_start   = np.empty(n_cells, dtype=np.int32)
    cell_count   = np.empty(n_cells, dtype=np.int32)

    index = 0

    for cy in range(ny):
        for cx in range(nx):

            cell_id = cx + cy * nx
            cell = cell_array[cx][cy]

            cell_start[cell_id] = index
            cell_count[cell_id] = len(cell)

            for obj in cell:
                cell_objects[index] = obj
                index += 1

    return cell_start, cell_count, cell_objects

@njit(parallel=True, fastmath=True)
def intersect(screenV, screenP, cell_start, cell_count, cell_objects, cell_size,
              all_a, all_b, all_X,
              all_aa, all_bb, all_n,all_ab,all_inv_det,all_opening,all_freq,all_phase,all_tile_z,all_trans_im,all_format,all_wall_im,all_light,all_light_w):

    X0 = screenP[0, 0]

    origin_x = 0.5 * (X0[0] + 100.0)
    origin_y = 0.5 * (X0[1] + 100.0)

    I0x = int(origin_x) // cell_size
    I0y = int(origin_y) // cell_size

    w=160*2
    h=80*2
    
    Im = np.full((w, h, 3), 0, dtype=np.float32)
    S = np.full((w, h, 3), 1e6, dtype=np.float32)
    Xl = np.full((w, h, 3), 1e6, dtype=np.float32)
    POS_l = np.full((w, h), 1e6, dtype=np.float32)
    wall_ind = np.zeros((w, h), np.int32)

    n_obj = len(all_a)

    for i in prange(w):

        # Thread-local arrays
        visited = np.empty((h, n_obj), np.uint8)
        visited[:] = 0

        t_int = np.empty(h, np.float32)
        for jj in range(h):
            t_int[jj] = 1e9

        # Precompute rays for this row once
        rays = np.empty((h, 3), np.float32)
        i0 = i // 4

        i1 = (i + 1) // 4
        if i1 > w//4-1:
            i1 = w//4-1

        i2 = (i + 2) // 4
        if i2 > w//4-1:
            i2 = w//4-1

        i3 = (i + 3) // 4
        if i3 > w//4-1:
            i3 = w//4-1



        for j in range(h):
            j0 = j // 4
            j1 = (j + 1) // 4
            j2 = (j + 2) // 4
            j3 = (j + 3) // 4
            if j1 > h//4-1:
                j1 = h//4-1
            if j2 > h//4-1:
                j2 = h//4-1
            if j3 > h//4-1:
                j3 = h//4-1


            r0 = screenV[i0, j0]
            r1 = screenV[i1, j1]
            r2 = screenV[i2, j2]
            r3 = screenV[i3, j3]
            rays[j, 0] = 0.25 * (r0[0] + r1[0]+r2[0] + r3[0])
            rays[j, 1] = 0.25 * (r0[1] + r1[1]+r2[1] + r3[1])
            rays[j, 2] = 0.25 * (r0[2] + r1[2]+r2[2] + r3[2])

        # DDA setup
        ix = I0x
        iy = I0y

        ray0 = rays[h//4]  # representative ray for stepping
        dx = ray0[0]
        dy = ray0[1]

        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        step_y = 1 if dy > 0 else -1 if dy < 0 else 0

        if dx != 0.0:
            next_x = (ix + (step_x > 0)) * cell_size
            t_max_x = (next_x - origin_x) / dx
            t_delta_x = cell_size / abs(dx)
        else:
            t_max_x = 1e9
            t_delta_x = 1e9

        if dy != 0.0:
            next_y = (iy + (step_y > 0)) * cell_size
            t_max_y = (next_y - origin_y) / dy
            t_delta_y = cell_size / abs(dy)
        else:
            t_max_y = 1e9
            t_delta_y = 1e9

        if t_max_x < t_max_y:
            ix -= step_x
            t_max_x -= t_delta_x
        else:
            iy -= step_y
            t_max_y -= t_delta_y

        # Grid traversal
        for g in range(100):

            if t_max_x < t_max_y:
                ix += step_x
                t = t_max_x
                t_max_x += t_delta_x
            else:
                iy += step_y
                t = t_max_y
                t_max_y += t_delta_y

            # Clamp cell indices
            cx = ix
            cy = iy
            if cx < 0:
                cx = 0
            elif cx >= 500 // cell_size:
                cx = 500 // cell_size - 1

            if cy < 0:
                cy = 0
            elif cy >= 500 // cell_size:
                cy = 500 // cell_size - 1

            cell_id = cx + cy * (500 // cell_size)

            start = cell_start[cell_id]
            count = cell_count[cell_id]

            # For each subpixel
            for j in range(h):

                if t_int[j] < t:
                    continue

                ray = rays[j]



                for k in range(start, start + count):
                    obj = cell_objects[k]


                    if visited[j, obj] == 1:
                        continue

                    visited[j, obj] = 1

                    a = all_a[obj]
                    b = all_b[obj]
                    n = all_n[obj]
                    X = all_X[obj]

                    aa = all_aa[obj]
                    bb = all_bb[obj]
                    ab = all_ab[obj]
                    inv_det = all_inv_det[obj]


                    # ---- INLINE INTERSECTION ----

                    denom = ray[0]*n[0] + ray[1]*n[1] + ray[2]*n[2]

                    if denom > 1e-9 or denom < -1e-9:

                        dx0 = X[0] - X0[0]
                        dy0 = X[1] - X0[1]
                        dz0 = X[2] - X0[2]

                        t_ = (dx0*n[0] + dy0*n[1] + dz0*n[2]) / denom

                        if 0.0 < t_ < t_int[j]:

                            px = X0[0] + t_ * ray[0]
                            py = X0[1] + t_ * ray[1]
                            pz = X0[2] + t_ * ray[2]

                            # if n[0] == 0 and n[1] == 0:
                            #
                            #     if pz < cell_array_z[cx][cy][0]  and X[2] > 0:
                            #         visited[j, obj] = 0
                            #         continue
                            #     if pz > cell_array_z[cx][cy][1]  and X[2] < 0:
                            #         visited[j, obj] = 0
                            #         continue

                            ax = px - X[0]
                            ay = py - X[1]
                            az = pz - X[2]

                            if ab==0:
                                u = (ax*a[0] + ay*a[1] + az*a[2]) / aa
                                v = (ax*b[0] + ay*b[1] + az*b[2]) / bb
                            else:
                                da = ax * a[0] + ay * a[1] + az * a[2]
                                db = ax * b[0] + ay * b[1] + az * b[2]
                                u = (da * bb - db * ab) * inv_det
                                v = (db * aa - da * ab) * inv_det

                            if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:

                                open=True
                                if all_opening[obj]:
                                    f = all_format[obj]
                                    iu = int((1 - u) * f[0])
                                    iv = int((1 - v) * f[1])
                                    tile_z = all_tile_z[obj]
                                    gu = iu % 120
                                    gv = iv % 120
                                    freq = all_freq[obj]
                                    if ((-iv // 120 + all_phase[obj] - freq + 1) % freq) == 0:
                                        shift = 120
                                        if iu // 120 > 0 and not (tile_z):
                                            shift = 0

                                    else:
                                        shift = 0
                                    trans=all_trans_im[obj][0]
                                    open=trans[gu,gv+shift]
                                if open:
                                    t_int[j] = t_
                                    S[i, j, 0] = u
                                    S[i, j, 1] = v
                                    S[i, j, 2] = t_

                                    wall_ind[i, j] = obj
                                    Xl[i, j, 0]=px
                                    Xl[i, j, 1] = py
                                    Xl[i, j, 2] = pz
                                    # Im[i, j, 0] =im[gu, gv + shift,0]
                                    # Im[i, j, 1] = im[gu, gv + shift, 1]
                                    # Im[i, j, 2] = im[gu, gv + shift, 2]


            # Early exit check
            done = True
            for jj in range(h):
                if t_int[jj] >= t:
                    done = False
                    break

            if done or g==99:
                for jj in range(h):
                    obj1=wall_ind[i, jj]
                    tile_z = all_tile_z[obj1]
                    u=S[i, jj,0]
                    v=S[i, jj,1]
                    f = all_format[obj1]
                    iu = int((1 - u) * f[0])
                    iv = int((1 - v) * f[1])

                    gu = iu % 120
                    gv = iv % 120
                    freq = all_freq[obj1]
                    if ((-iv // 120 + all_phase[obj1] - freq + 1) % freq) == 0:
                        shift = 120
                        if iu // 120 > 0 and not (tile_z):
                            shift = 0
                    else:
                        shift = 0
                    im = all_wall_im[obj1][0]
                    Cl=all_light_w[obj1]
                    r=im[gu, gv + shift,0]
                    Im[i, jj, 0] =r*Cl[0]
                    Im[i, jj, 1] = im[gu, gv + shift, 1]*Cl[1]
                    Im[i, jj, 2] = im[gu, gv + shift, 2]*Cl[2]


                    light_x=all_light[obj1]
                    dm=1e6
                    for k in (light_x):
                        dx2=k[0]-Xl[i,jj,0]
                        dy2=k[1]-Xl[i,jj,1]
                        dz2=k[2]-Xl[i,jj,2]
                        d=dx2*dx2+dy2*dy2+dz2*dz2
                        if dm>d:
                            dm=d
                    POS_l[i, jj]=np.sqrt(dm)
                    if r==-1:
                        Im[i, jj, 0] = 0
                        Im[i, jj, 1] = 255
                        Im[i, jj, 2] = 255
                        POS_l[i, jj]=1e-9
                break

    return S, wall_ind,Xl,Im,POS_l

@njit( fastmath=True)
def thing_render(counter,counter2,a0,a1,x_perso,all_x_e,Im,S,all_RA,all_im_m,all_im_o,all_obj_mon,all_types_e,all_angle,all_ima_m,all_mort,all_attack_range,all_range,all_light_e):
    c0 = np.cos(a0)
    s0 = np.sin(-a0)

    c1 = np.cos(a1)
    s1 = np.sin(a1)

    W=scrnL[0]*2*2
    H=scrnL[1]*2*2

    f1=scrnL[0]*2/TAN2
    f2 = scrnL[1]*2 / TAN1

    index_e = np.full((W, H), -1, dtype=np.int64)
    depth_e = np.full((W, H), 1e6, dtype=np.float64)
    print(len(all_x_e))
    for i in range(len(all_x_e)):
        x_e=all_x_e[i]

        if all_obj_mon[i]:
            mort=int(all_mort[i])
            if mort==0:
                if all_attack_range[i]:
                    if all_range[i]:
                        f=counter2//8
                    else:
                        f=counter//4
                    im = all_ima_m[all_types_e[i],f , :, :, :]
                else:
                    im=all_im_m[all_types_e[i],counter//3,int(all_angle[i]//45),:,:,:]
            else:
                im=all_ima_m[all_types_e[i],4+mort,:,:,:]


        else:
            im = all_im_o[all_types_e[i], int(all_angle[i]//45), :, :, :]
        d = x_e - x_perso
        dx, dy, dz = d

        RA=all_RA[i]
        x1 = c0 * dx + s0 * dy
        if x1>0:
            y1 = -s0 * dx + c0 * dy
            z1 = -dz+(0.75*RA-5)

            x2 = c1 * x1 - s1 * z1
            y2 = y1
            z2 = s1 * x1 + c1 * z1



            sx = int(W * 0.5 + f1 * y2 / x2)
            sy = int(H * 0.5 - f2 * z2 / x2)
            width = int(RA *  W / x1)
            if sx+width//2>0 and sx-width//2<W and sy+width//2>0 and sy-width//2<H: #and S[sx,sy,2]>x1:

                for gx in range(0,width):
                    ix=sx-width//2+gx
                    ix_r = int(160 * gx / width)
                    if ix >= 0 and ix<W and ix_r<160:
                        for gy in range(0, width):
                            iy=sy-width//2+gy
                            iy_r = int(160 * gy / width)
                            if iy >= 0 and iy<H and S[ix,iy,2]>x1 and depth_e[ix,iy]>x1 and iy_r<160:
                                r=im[ix_r,iy_r,0]
                                g=im[ix_r, iy_r, 1]
                                b=im[ix_r, iy_r, 2]
                                if r+g+b>0:
                                    l=all_light_e[i]
                                    Im[ix,iy,0] = r*l[0]
                                    Im[ix, iy, 1] = g*l[1]
                                    Im[ix, iy, 2] = b*l[2]
                                    index_e[ix,iy]=i
                                    depth_e[ix, iy]=x1
    return Im,index_e

def source_pos(code):
    x = (int(code.split(',')[0]) - 50) * 2
    y = (int(code.split(',')[1]) - 50) * 2
    z = zmap[int(code.split(',')[1])][int(code.split(',')[0])] - hmap[int(code.split(',')[0])][int(code.split(',')[1])]
    return np.array([x, y, z - 5])


def cells_crossed_by_segment(X, A, cell_size=1.0, eps=1e-9):
    x0, y0 = X
    ax, ay = A

    dx, dy = ax, ay

    ix = np.floor(x0 / cell_size)
    iy = np.floor(y0 / cell_size)

    cells = [(ix, iy)]

    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_y = 1 if dy > 0 else -1 if dy < 0 else 0

    if dx != 0:
        next_x = (ix + (step_x > 0)) * cell_size
        t_max_x = (next_x - x0) / dx
        t_delta_x = cell_size / abs(dx)
    else:
        t_max_x = float("inf")
        t_delta_x = float("inf")

    if dy != 0:
        next_y = (iy + (step_y > 0)) * cell_size
        t_max_y = (next_y - y0) / dy
        t_delta_y = cell_size / abs(dy)
    else:
        t_max_y = float("inf")
        t_delta_y = float("inf")

    t = 0.0
    while t <= 1.0 + eps:
        if t_max_x < t_max_y:
            ix += step_x
            t = t_max_x
            t_max_x += t_delta_x
        else:
            iy += step_y
            t = t_max_y
            t_max_y += t_delta_y

        if t > 1.0 + eps:
            break

        cells.append((ix, iy))

    return cells

def cells_covered_by_plane(X, A,B, cell_size=1.0):
    x0, y0 = X
    ax, ay = A
    bx, by = B


    ix = np.floor(x0 / cell_size)
    iy = np.floor(y0 / cell_size)

    ix1 = np.floor((x0+ax) / cell_size)
    iy1 = np.floor((y0+by) / cell_size)


    xmin = int(min(ix, ix1))
    xmax = int(max(ix, ix1))
    ymin = int(min(iy, iy1))
    ymax = int(max(iy, iy1))
    cells = []
    for ix in range(xmin, xmax + 1):
        for iy in range(ymin, ymax + 1):
            cells.append((ix, iy))

    return cells
def plane(a,b,X):
    M = np.stack((a * 1.1, b * 1.1, -screenV), axis=-1)
    B = -X + screenP
    inv_M = np.linalg.inv(M)
    S = np.einsum('...ij,...j->...i', inv_M, B)
    return S

def explo_zone(R,dist):
    if explo_type!=2:
        white = np.array([[[1, 1, 1]]])
        yellow=np.array([[[1,1,0]]])
        red=np.array([[[1,0,0]]])
    else:
        white = np.array([[[1, 1, 1]]])
        yellow=np.array([[[1,1,1]]])
        red=np.array([[[0.5,0.5,1]]])
    #color=yellow*(1-(dist/R))+red*(dist/R)
    color = np.where(dist<R/2,white*(1-(2*dist/R))+yellow*(2*dist/R),yellow*(1-(2*(dist-0.5*R)/R))+red*(2*(dist-0.5*R)/R))
    return color

class Wall():
    def __init__(self, u, v, w, text, door, deco, freq, phase,slant):
        self.num=len(wall)
        self.freq=freq
        self.phase_=phase
        if levelD[level]['deco'][deco - 1] in z_tileable_deco:
            self.tile_z=1
        else:
            self.tile_z = 0
        #(freq - 1 - phase)
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

        self.door_deco=(levelD[level]['deco'][self.deco - 1] in deco_door and self.deco != 0)

        Imdeco = pygame.image.load(self.text)
        Imdeco2 = pygame.image.load(self.text2)


        Imdeco = pygame.transform.scale(Imdeco, (120 * 2, 120 ))
        Imdeco2 = pygame.transform.scale(Imdeco2, (120 * 2, 120 ))
        for i in range(2):
            Imdeco.blit(pygame.image.load(text[0]), (120 * i, 0))
            Imdeco2.blit(pygame.image.load(text[1]), (120 * i, 0))

        if deco != 0:
            if text[2] == 'A' or text[2] == 'AB':
                Imdeco.blit(pygame.image.load('image/deco/' + str(levelD[level]['deco'][deco - 1]) + '.png'),
                            (0 , 0))
            if text[2] == 'B' or text[2] == 'AB':
                Imdeco2.blit(pygame.image.load('image/deco/' + str(levelD[level]['deco'][deco - 1]) + '.png'),
                             (0 , 0))
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
        self.sky = 0
        self.window=0
        IM = np.transpose(pygame.surfarray.pixels3d(Imdeco), (1, 0, 2))
        IM = np.where(np.expand_dims(((IM - np.array([255, 0, 255])) == 0).all(-1), -1), -1, IM)
        IM = np.where(np.expand_dims(((IM - np.array([0, 255, 255])) == 0).all(-1), -1), -2, IM)
        self.window = np.sum(IM <= -1)
        self.sky = self.sky or np.sum(IM <= -2)
        self.wall_im = [np.flip(np.minimum(IM + 1, 255), (0, 1))]
        if self.window and not(self.sky):
            self.trans_im=[np.flip((IM!=-1).all(axis=-1), (0, 1))]
        else:
            self.trans_im=np.array([0.])

        IM2 = np.transpose(pygame.surfarray.pixels3d(Imdeco2), (1, 0, 2))
        IM2 = np.where(np.expand_dims(((IM2 - np.array([255, 0, 255])) == 0).all(-1), -1), -1, IM2)
        IM2 = np.where(np.expand_dims(((IM2 - np.array([0, 255, 255])) == 0).all(-1), -1), -2, IM2)
        self.wall_im2 = [np.flip(np.minimum(IM2 + 1, 255), (0, 1))]
        self.sky = self.sky or np.sum(IM2 <= -2)
        self.window =self.window or np.sum(IM <= -1)
        if self.window and not(self.sky):
            self.trans_im2=[np.flip((IM2!=-1).all(axis=-1), (0, 1))]

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

                Imdeco = Im_t.copy()
                Imdeco = pygame.transform.scale(Imdeco, (120 * 2, 120))
                for i in range(2):
                    Imdeco.blit(pygame.image.load(text[0]), (120 * i, 0))
                if deco != 0:


                    deco_name = 'image/deco/' + str(levelD[level]['deco'][deco - 1]) + '.png'
                    if (len(files_d) > 1):
                        deco_name = 'image/deco/' + str(levelD[level]['deco'][deco - 1]) + '.' + str(
                            min(k, len(files_d) - 2) + 1) + '.png'
                    Imdeco.blit(pygame.image.load(deco_name), (0 , 0))
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
                self.window =self.window or np.sum(IM <= -1)
                self.sky = self.sky or np.sum(IM <= -2)
                self.wall_im.append(np.flip(np.minimum(IM + 1, 255), (0, 1)))
                if self.window and not (self.sky):
                    self.trans_im .append(np.flip((IM != -1).all(axis=-1), (0, 1)))

        files2 = [filename for filename in os.listdir(self.text2[:11]) if filename.startswith(self.text2[11:-3])]
        if (len(files2) > 1) or (len(files_d) > 1):
            for k in range(max(len(files2) - 1, len(files_d) - 1)):
                if (len(files2) > 1):
                    Im_t = pygame.image.load(self.text2[:-4] + '.' + str(min(k, len(files) - 2) + 1) + '.png')
                else:
                    Im_t = pygame.image.load(text[1])

                Imdeco2 = Im_t.copy()
                Imdeco2 = pygame.transform.scale(Imdeco2, (120 * 2, 120))
                for i in range(2):
                    Imdeco2.blit(pygame.image.load(self.text2), (120 * i, 0))
                if deco != 0:



                    deco_name = 'image/deco/' + str(levelD[level]['deco'][deco - 1]) + '.png'
                    if (len(files_d) > 1):
                        deco_name = 'image/deco/' + str(levelD[level]['deco'][deco - 1]) + '.' + str(
                            min(k, len(files_d) - 2) + 1) + '.png'
                    Imdeco2.blit(pygame.image.load(deco_name), (0 , 0))
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
                self.sky = self.sky or np.sum(IM2 <= -2)
                self.window = self.window or np.sum(IM2 <= -1)
                self.wall_im2.append(np.flip(np.minimum(IM2 + 1, 255), (0, 1)))
                if self.window and not (self.sky):
                    self.trans_im2 .append(np.flip((IM2 != -1).all(axis=-1), (0, 1)))


        self.U = np.array([0])
        self.z = w[-1]
        self.ID = str(int((self.X[0][0][0] + 0.5 * self.b[0][0][0]) // 2 + 50)) + ',' + str(
            int((self.X[0][0][1] + 0.5 * self.b[0][0][1]) // 2 + 50))
        self.transp = 0

        if (self.a[0][0][-1] == 0 and self.b[0][0][-1] == 0) or (self.text.split('/')[1]=='flat'):
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

        self.slanted=slant!=0
        self.X_old=self.X.copy()
        self.b_old=self.b.copy()
        self.a_old=self.a.copy()
        self.X_middle=self.X.copy()
        self.b_middle=self.b.copy()
        self.a_middle=self.a.copy()
        self.angle0=0
        if  self.slanted:# SLANTED WALLS
            self.angle0=-(pi/8)*(slant-1.5)*2
            angle1=pi-np.arctan2(self.b[:,:,1],self.b[:,:,0])
            self.a[:,:,0]=self.a[:,:,0]+np.sin(self.angle0)*np.sin(angle1)*self.a[:,:,2]
            self.a[:, :, 1] = self.a[:, :, 1] + np.sin(self.angle0) * np.cos(angle1) * self.a[:, :, 2]
            self.X[:,:,0]=self.X[:,:,0]-np.sin(self.angle0)*np.sin(angle1)*self.a[:,:,2]
            self.X[:, :, 1] = self.X[:, :, 1] - np.sin(self.angle0) * np.cos(angle1) * self.a[:, :, 2]

            if self.angle0>0:

                self.a_middle[:,:,0]=self.a_old[:,:,0]+np.sin(self.angle0)*np.sin(angle1)*(2.5)
                self.a_middle[:, :, 1] = self.a_old[:, :, 1] + np.sin(self.angle0) * np.cos(angle1) * (2.5)
                self.X_middle[:,:,0]=self.X[:,:,0]-np.sin(self.angle0)*np.sin(angle1)*(2.5)
                self.X_middle[:, :, 1] = self.X[:, :, 1] - np.sin(self.angle0) * np.cos(angle1) * (2.5)

        self.linked=[]
        self.overlap=1.01
        if self.a[0][0][-1] == 0 and self.b[0][0][-1] == 0:
            self.overlap=1.0

        self.a_sliced = self.a[::2, ::2]* self.overlap
        self.b_sliced = self.b[::2, ::2]* self.overlap
        self.X_sliced = self.X[::2, ::2]
        h, w, _ = self.a_sliced.shape
        if not hasattr(self, "_M"):
            h, w, _ = self.a_sliced.shape
            self._M = np.empty((h, w, 3, 3), dtype=np.float64)
            self._B = np.empty((h, w, 3), dtype=np.float64)

        self.Ub = np.empty_like(depth[:,:,0]).astype(bool)

        self._tmp_bool = np.empty(self.Ub.shape, dtype=bool)
        self._tmp_bool2 = np.empty(self.Ub.shape, dtype=bool)
        # ensure self.U and self.Ub exist with correct shapes/dtypes
        self.U = np.empty(self.Ub.shape, dtype=bool)
        self.Ar = np.full((Im.shape), 0.)
        self.Gx=np.full((self.Ub.shape), 0)
        self.Gy = np.full((self.Ub.shape), 0)
        self.Xl= np.full((Im.shape), 0.)
        self.Xl_small = np.full((Im[::2,::2,:].shape), 0.)
        self.Da=np.full((2 * scrnL[0], 2 * scrnL[1]), 1000.0, dtype=float)
        self.S0_ = np.full((Im[::2, ::2, 0].shape), 0.)
        self.S1_=np.full((Im[::2,::2,0].shape), 0.)
        self.filt=np.empty(self.Ub.shape, dtype=bool)
        self.rendered=False

        self.height_rel=[]
        self.inside = False

        self.colorL = np.array([1., 1, 1])
        if self.ID in light_color.keys():
            self.colorL = np.round(np.maximum(np.array(light_color[self.ID]), 0.1), 2)
        self.side=1.
        if self.window and not(self.sky):
            self.opening=True
        else:
            self.opening=False

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
        # IM = np.transpose(pygame.surfarray.pixels3d(Imdeco), (1, 0, 2))
        # IM = np.where(np.expand_dims(((IM - np.array([255, 0, 255])) == 0).all(-1), -1), -1, IM)
        # IM = np.where(np.expand_dims(((IM - np.array([0, 255, 255])) == 0).all(-1), -1), -2, IM)
        # self.window = np.sum(IM <= -1)
        #
        # self.wall_im[0] = np.flip(np.minimum(IM + 1, 255), (0, 1))
        IM = np.transpose(pygame.surfarray.pixels3d(Imdeco), (1, 0, 2))

        self.wall_im[0] = np.flip(np.minimum(IM, 255), (0, 1))
        return 0

    def texture(self, sc1, sc2):

        self.borne = [self.wall_im[0].shape[0] - 1, self.wall_im[0].shape[1] - 1]
        if self.door != 0:
            self.format = 120 * np.array([1, 1])
        else:
            self.format = 120 * np.array([np.linalg.norm(self.a[0][0]) / 10, np.linalg.norm(self.b[0][0]) / 10])
        if self.text == 'image/wall/wall29.png':
            self.format = 120 * np.array([1, np.linalg.norm(self.b[0][0]) / 30])
    def reset_rend(self):
        self.rendered = False
    def calc_norm(self):# IMPROVE FOR SLANTED
        global moving_cam,all_light_w

        if len(self.wall_im)>1 or len(self.wall_im2)>1:
            moving_cam=True
        N = np.stack((self.a_old[0][0], self.b_old[0][0], -self.n), axis=-1)
        V = np.maximum(np.minimum(np.linalg.solve(N, -self.X_old[0][0] + R_c)[:-1], 1), 0)
        self.norm = np.linalg.norm(
            self.X_old[0][0][:-1] + V[0] * self.a_old[0][0][:-1] + V[1] * self.b_old[0][0][:-1] - R_c[:-1]) - min(self.window,
                                                                                                      1) * 0.001
        self.norm3 = np.linalg.norm(
            self.X_middle[0][0] + V[0] * self.a_middle[0][0] + V[1] * self.b_middle[0][0] - (R_c))
        if self.angle0>0:
            N = np.stack((self.a_old[0][0], self.b_old[0][0], -self.n), axis=-1)
            V = np.maximum(np.minimum(np.linalg.solve(N, -self.X_middle[0][0] + R_c)[:-1], 1), 0)
            self.norm3 = np.linalg.norm(
                (self.X_middle[0][0] + V[0] * self.a_old[0][0] + V[1] * self.b_old[0][0])[:-1] - (
                            R_c[:-1] ))

        # if self.text.split('/')[1]=='flat':
        #     add0=np.linalg.norm(self.X[0][0][:-1]-R_c[:-1])
        #     add1 = np.linalg.norm(self.X[0][0][:-1] - R_c[:-1])
        #     add2 = np.linalg.norm(self.X[0][0][:-1] - R_c[:-1])
        #     add3 = np.linalg.norm(self.X[0][0][:-1] - R_c[:-1])
        #     self.add=(add0+add1+add2+add3)/4
        #     self.norm+=self.add*1e-3
        self.inter=V
        self.reset_rend()
        if (shoot == 1 or self.explo) and levelD[level]['deco'][self.deco - 1] in deco_destruc and self.deco != 0:
            self.breakable()

        colorL = [1, 1, 1]
        if self.ID in light_color.keys():
            colorL = np.round(np.maximum(np.array(light_color[self.ID]), 0.1), 2)

        colorL = light_modif(colorL, level, c3)
        self.colorL=colorL
        all_light_w[self.num]=self.colorL

    def calc_normfast(self):
        self.normf = np.linalg.norm(self.X_middle[0][0][:-1] + 0.5 * self.a_middle[0][0][:-1] + 0.5 * self.b_middle[0][0][:-1] - R_c[:-1])

    def normal(self):


        No = self.n[:-1]
        No = No / np.linalg.norm(No)

        # rap = [-t[0] * No[1] + t[1] * No[0], -t[0] * No[0] - t[1] * No[1]]
        # rap0 = [-(cos(-ang[0]) * No[1] + sin(-ang[0]) * No[0]), (-cos(-ang[0]) * No[0] - sin(-ang[0]) * No[1])]
        #
        # Sang=np.arctan2(rap[0], rap[1])
        # Sang0 = np.arctan2(rap0[0], rap0[1])



        x_ = self.X_middle[0][0][0]
        y_ = self.X_middle[0][0][1]
        a_ = self.b_old[0][0][0]
        b_ = self.b_old[0][0][1]



        self.side = b_ * R_c[0] - a_ * R_c[1] + a_ * y_ - b_ * x_
        if self.side < 0:
            return No
        else:
            return -No
        # if self.side < 0:
        #     return (-t - abs(np.dot(-t, No)) * No)
        # else:
        #     return (-t - abs(np.dot(-t, -No)) * -No)

    def test_behind(self):
        milliseconds = [time.perf_counter()*1000]
        x_ = self.X[0][0][0]
        y_ = self.X[0][0][1]
        a_ = self.b[0][0][0]
        b_ = self.b[0][0][1]
        self.side = b_ * R_c[0] - a_ * R_c[1] + a_ * y_ - b_ * x_

        if self.angle0 < 0:
            Mg = np.stack((self.b_old[0][0][0:-1], -Vg @ Rp), axis=-1)
            self.Ig = np.linalg.solve(Mg, x - self.X_old[0][0][0:-1])

            Md = np.stack((self.b_old[0][0][0:-1], -Vd @ Rp), axis=-1)
            self.Id = np.linalg.solve(Md, x - self.X_old[0][0][0:-1])

            Rp0 = rot_plan(-ang[0])
            Xa = (self.X_old[0][0][0:-1] - x) @ Rp0
            Xb = (self.X_old[0][0][0:-1] + self.b_old[0][0][0:-1] - x) @ Rp0
        else:
            Mg = np.stack((self.b_middle[0][0][0:-1], -Vg @ Rp), axis=-1)
            self.Ig = np.linalg.solve(Mg, x - self.X[0][0][0:-1])

            Md = np.stack((self.b_middle[0][0][0:-1], -Vd @ Rp), axis=-1)
            self.Id = np.linalg.solve(Md, x - self.X[0][0][0:-1])

            Rp0 = rot_plan(-ang[0])
            Xa = (self.X_middle[0][0][0:-1] - x) @ Rp0
            Xb = (self.X_middle[0][0][0:-1] + self.b_middle[0][0][0:-1] - x) @ Rp0


        if ((Xa[0] > 1 and Xb[0] > 1) or (self.Id[0] < 1 and self.Id[0] > 0 and self.Id[1] > 1) or (
                self.Ig[0] < 1 and self.Ig[0] > 0 and self.Ig[1] > 1)) or self.door != 0:

            if self.door != 0 and self.closed == 0:

                milliseconds.append(time.perf_counter()*1000)
                i.time_behind = milliseconds[1] - milliseconds[0]

                return True
            else:

                global horizon,horizon2

                B0 = self.b_old[:, 0, :-1].copy()
                X0 = self.X_old[:, 0, :-1].copy()

                if self.inside:
                    h=0
                    if self.X[0,0,2] in height_list:
                        h=height_list.index(self.X[0,0,2])
                    if h>0:

                        horizon0=np.minimum(horizon,np.min(horizon2[:,:h+1],axis=1)[:,None])
                    else:
                        horizon0 = np.minimum(horizon, horizon2[:, 0, None])
                else:
                    horizon0=horizon
                self.M0 = np.stack((B0, -screen[:, 24, 3:-1] @ Rp), axis=-1)  # -------------- PB à résoudre pièce haute
                self.A0 = -X0 + (screen[:, 24, :2] - x) @ Rp + x
                self.S0 = (np.linalg.solve(self.M0, self.A0))
                self.U0 = np.where(
                    np.all(self.S0 <= horizon0, axis=-1) & np.all(self.S0 > 0, axis=-1) & np.all(self.S0[:, :-1] < 1, axis=-1), 1, 0)


                if self.angle0<0:

                    angle1=pi-np.arctan2(self.b_old[:,0,1],self.b_old[:,0,0])
                    X0[:, 0] = self.X_old[:, 0, 0] - np.sin(self.angle0) * np.sin(angle1) *self.a_old[:, 0, 2]
                    X0[:, 1] = self.X_old[:, 0, 1] - np.sin(self.angle0) * np.cos(angle1) * self.a_old[:, 0, 2]
                    self.M0 = np.stack((B0, -screen[:, 24, 3:-1] @ Rp), axis=-1)  # -------------- PB à résoudre pièce haute
                    self.A0 = -X0 + (screen[:, 24, :2] - x) @ Rp + x
                    self.S1 = (np.linalg.solve(self.M0, self.A0))
                    self.U1 = np.where(
                        np.all(self.S1 <= horizon, axis=-1) & np.all(self.S1 > 0, axis=-1) & np.all(self.S1[:, :-1] < 1, axis=-1), 1, 0)

                if self.angle0>0:

                    self.U1=self.U0.copy()
                    X0[:, 0] = self.X[:, 0, 0]
                    X0[:, 1] = self.X[:, 0, 1]
                    self.M0 = np.stack((B0, -screen[:, 24, 3:-1] @ Rp), axis=-1)  # -------------- PB à résoudre pièce haute
                    self.A0 = -X0 + (screen[:, 24, :2] - x) @ Rp + x
                    self.S1 = (np.linalg.solve(self.M0, self.A0))
                    self.U0 = np.where(
                        np.all(self.S1 <= horizon, axis=-1) & np.all(self.S1 > 0, axis=-1) & np.all(self.S1[:, :-1] < 1, axis=-1), 1, 0)

                if self.window and not(self.sky):
                    if self.side<0:
                        t_im=self.trans_im
                    else:
                        t_im=self.trans_im2
                    ind = c // (12 // len(t_im))
                    u = ((1 - self.S0[ :, 0]) * self.format[1]).astype(int)
                    G_g = np.mod(np.maximum(u, 0), 120)

                    blocked = 1-np.any(1-t_im[ind][:, G_g + 120 * ((((-u // 120 + self.phase_ - self.freq + 1) % self.freq)) == 0)],axis=0)
                    blocked=blocked&self.U0.astype(bool)
                    if not(self.inside):
                        horizon[blocked]=self.S0[blocked,1,None]
                    else:
                        horizon2[blocked, h]=self.S0[blocked,1]



                if (self.window == 0 or self.sky!=0) and self.inside==False:
                    if self.angle0!=0:

                        horizon = self.S1[:, -1:] * np.expand_dims(self.U1, -1) + horizon * (
                                    1 - np.expand_dims(self.U1, -1))
                    else:
                        horizon = self.S0[:, -1:] * np.expand_dims(self.U0, -1) + horizon * (
                                    1 - np.expand_dims(self.U0, -1))

                if self.inside and (self.window == 0 or self.sky!=0):

                    horizon2[:,h] = (self.S0[:, -1:] * np.expand_dims(self.U0, -1) + horizon2[:,h,None] * (
                            1 - np.expand_dims(self.U0, -1)))[:,0]


                if np.sum(self.U0) > 0:
                    milliseconds.append(time.perf_counter()*1000)
                    i.time_behind=milliseconds[1]-milliseconds[0]
                    return True
                else:
                    milliseconds.append(time.perf_counter()*1000)
                    i.time_behind = milliseconds[1] - milliseconds[0]
                    return False

        else:

            milliseconds.append(time.perf_counter()*1000)
            i.time_behind = milliseconds[1] - milliseconds[0]
            return False

    def breakable(self):
        global x_d
        self.inline=0
        self.shot = []
        self.d_shot=100
        x_d0=[]
        for i in range(len(x_d)):
            # inline = min(np.sum(self.U[int(scrnL[0]*(1+2*x_d[i][0]) - gun_width):int(scrnL[0]*(1+2*x_d[i][0]) + gun_width), int(2 * scrnL[1]*(1+2*x_d[i][1]) // 2 - gun_width):int(2 * scrnL[1]*(1+2*x_d[i][1]) // 2 + gun_width)]),
            #                 1)
            inline_e = wall_ind_i[int(2 * scrnL[0] * (1 + 2 * x_d[i][0]) - gun_width):int(
                2 * scrnL[0] * (1 + 2 * x_d[i][0]) + gun_width),
                       int(2 * 2 * scrnL[1] * (1 + 2 * x_d[i][1]) // 2 - gun_width):int(
                           2 * 2 * scrnL[1] * (1 + 2 * x_d[i][1]) // 2 + gun_width)]
            inline = (inline_e == self.num).any()
            self.inline=(self.inline or inline)
            self.shot.append(i)
            if not inline:
                x_d0.append(x_d[i])
            else:
                depth_ = depth.shape
                self.d_shot=min(self.d_shot,depth_cached[int(depth_[0] * (x_d[0][0] + 0.5))][int(depth_[1] * (x_d[0][1] + 0.5))])
        x_d=x_d0


        if inline or self.explo:
            if ((self.d_shot<5 or arme!=0) and arme!=4) or self.explo:
                self.vie+=1
                s = pygame.mixer.Sound("son/barril.ogg")
                s.play()
            if self.vie==3:
                self.X=self.X*0
                all_X[self.num]=self.X[0,0,:]
                self.X_sliced = self.X[::2, ::2]
                self.X_middle = self.X_middle * 0



    def cross_wall(self,trans):
        Mg = np.stack((self.b_middle[0][0][0:-1], trans @ Rp), axis=-1)
        self.Ig = np.linalg.solve(Mg, x - self.X[0][0][0:-1])



        if self.Ig[0]<1 and self.Ig[0]>0:
            if self.side < 0:
                w_im = self.wall_im
                ind = c // (12 // len(self.wall_im))
            else:
                w_im = self.wall_im2
                ind = c // (12 // len(self.wall_im2))
            return np.sum(w_im[ind][60,int((1-self.Ig[0])*self.format[1])%120+120*(int(-(1-self.Ig[0])*self.format[1]//120+(self.phase_-self.freq+1))%self.freq==0),:])!=0
        else:
            return False

    def compute_mask_fast(self):

        S = self.S  # local bind
        d = depth[:,:,0]  # local bind

        a0 = S[..., 0]
        a1 = S[..., 1]
        a2 = S[..., 2]

        # tmp_bool = ((a0 <= d) & (a0 > 0)) combined with channels 1 and 2, in-place
        np.less_equal(a0, d, out=self._tmp_bool)  # tmp = a0 <= d
        np.logical_and(self._tmp_bool, a0 > 0, out=self._tmp_bool)  # tmp &= (a0 > 0)
        # combine channel 1
        np.logical_and(self._tmp_bool, a1 <= d, out=self._tmp_bool)
        np.logical_and(self._tmp_bool, a1 > 0, out=self._tmp_bool)
        # combine channel 2
        np.logical_and(self._tmp_bool, a2 <= d, out=self._tmp_bool)
        np.logical_and(self._tmp_bool, a2 > 0, out=self._tmp_bool)
        # now tmp_bool is mask1

        # tmp_bool2 = (a0 < 1) & (a1 < 1)
        np.less(a0, 1, out=self._tmp_bool2)
        np.logical_and(self._tmp_bool2, a1 < 1, out=self._tmp_bool2)
        # final U = tmp_bool & tmp_bool2  (reuse self.U as output)
        np.logical_and(self._tmp_bool, self._tmp_bool2, out=self.U)

        # fill Ub without allocation
        self.Ub[:] = self.U

    def render(self):
        global depth, POS, Sky_view,explo_R,wall_index

        self.rendered=True
        self.time=(0,0)
        milliseconds=[time.perf_counter()*1000]
        label_m=[]
        # self.M = np.stack((self.a[::2, ::2, :] * 1.01, self.b[::2, ::2, :] * 1.01, -screenV[::2, ::2, :]), axis=-1)
        # self.B = -self.X[::2, ::2, :] + screenP[::2, ::2, :]
        #
        # # self.S=(np.linalg.solve(self.M.astype(np.float32),self.B.astype(np.float32)))
        # inv_M = np.linalg.inv(self.M)
        # self.S = np.einsum('...ij,...j->...i', inv_M, self.B)


        # M = self._M
        # B = self._B
        # M[..., 0] = self.a_sliced
        # M[..., 1 ] = self.b_sliced
        # M[..., 2] = -v_
        #
        # # --- fill B in-place ---
        # B[...] = p_ - self.X_sliced
        #
        # # --- solve batched 3x3 systems ---
        # self.S = np.linalg.solve(M, B)

        milliseconds.append(time.perf_counter()*1000)
        label_m.append('solving')

        # self.M=torch.tensor(np.stack((self.a[:,:,:]*1.01,self.b[:,:,:]*1.01,-screenV[:,:,:]),axis=-1))
        # self.B=torch.tensor(-self.X[:,:,:]+screenP[:,:,:])
        # self.S=(torch.linalg.solve(self.M.to('cuda'),self.B.to('cuda'))).to('cpu').numpy()

        # for i in range(2):
        #     self.S = self.S.repeat(2, axis=0).repeat(2, axis=1)
        #     self.S[:, :, :-1] = (np.roll(self.S[:, :, :-1], 1, axis=0) + np.roll(self.S[:, :, :-1], 1,
        #                                                                          axis=1) + np.roll(self.S[:, :, :-1],
        #                                                                                            -1,
        #                                                                                            axis=1) + np.roll(
        #         self.S[:, :, :-1], -1, axis=0)) / 4
        #     AbsS = np.absolute(self.S[:, :, -1])
        #     self.S[:, :, -1] = (np.sign(self.S[:, :, -1]) * (
        #                 np.roll(AbsS, 1, axis=0) + np.roll(AbsS, 1, axis=1) + np.roll(AbsS, -1, axis=1) + np.roll(AbsS,
        #                                                                                                           -1,
        
        # S00=np.sign(self.S[:,:,-1])
        # S00=cv2.resize(S00,None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST)
        # self.S[:, :, -1]=np.absolute(self.S[:,:,-1])
        # self.S=cv2.resize(self.S,None,fx=4,fy=4,interpolation=cv2.INTER_LINEAR)
        # self.S[:, :, -1]=self.S[:, :, -1]*S00

        milliseconds.append(time.perf_counter()*1000)
        label_m.append('upscale')

        #self.compute_mask_fast()

        # if not(self.sky) and self.window:
        #
        #     u = ((1 - self.S[ :,:, :-1]) * self.format).astype(int)
        #     G_g = np.mod(np.maximum(u, 0), 120)
        #     if self.side < 0:
        #         w_im = self.wall_im
        #         t_im = self.trans_im
        #     else:
        #         w_im = self.wall_im2
        #         t_im = self.trans_im2
        #     if levelD[level]['deco'][self.deco - 1] not in deco_destruc:
        #         ind = c // (12 // len(t_im))
        #     else:
        #         ind=min(self.vie,2)
        #     texture = t_im[ind][G_g[:, :, 0], G_g[:, :, 1] + 120 * ((((-u[:, :, 1] // 120 + self.phase_-self.freq+1) % self.freq)) == 0) * (u[:, :, 0] < 120 + 1000 * self.tile_z)]
        #     self.Ub=self.Ub&(texture)



        # wall_index[self.Ub] = len(wall_rend)
        # depth[..., 0][self.Ub] = self.S[:, :, -1][self.Ub]



        milliseconds.append(time.perf_counter()*1000)
        label_m.append('contour')


        if np.sum(self.U) < 25:
            return np.full((2 * scrnL[0], 2 * scrnL[1], 1), 0)
        else:
            Ue = np.expand_dims(self.U, -1)
            if self.transp:
                Sky_view = 1
                return -1 * np.dstack((Ue, Ue, Ue))
            # self.G=np.full((self.U.shape[0],self.U.shape[1],2),0)
            # self.G[self.Ub]=np.mod(
            #     np.maximum(((1 - self.S[:, :, :-1][self.Ub]) * self.format).astype(int), 0) + int(self.phase * c3 * 0.5),
            #     self.borne)


            milliseconds.append(time.perf_counter()*1000)
            label_m.append('indexing')

            # colorL = [1, 1, 1]
            # if self.ID in light_color.keys():
            #     colorL = np.round(np.maximum(np.array(light_color[self.ID]), 0.1), 2)
            #
            # colorL = light_modif(colorL, level, c3)
            #
            # x_ = self.X[0][0][0]
            # y_ = self.X[0][0][1]
            # a_ = self.b[0][0][0]
            # b_ = self.b[0][0][1]
            # side = b_ * R_c[0] - a_ * R_c[1] + a_ * y_ - b_ * x_
            # self.Ar[:]=self.Ar*0.
            # self.Gx[:] = self.G[..., 0]
            # self.Gy[:] = self.G[..., 1]
            #
            # if levelD[level]['deco'][self.deco - 1] not in deco_destruc:
            #     if side < 0:
            #         ind = c // (12 // len(self.wall_im))
            #
            #         # boolean mask
            #         Ub = self.Ub
            #
            #         # fast coordinate extraction (no python loops)
            #         gx = self.Gx[Ub]
            #         gy = self.Gy[Ub]
            #
            #         # fast vectorized texture lookup
            #         wall_vals = self.wall_im[ind][gx, gy]
            #
            #         # assign colored pixels
            #         self.Ar[Ub] = wall_vals * colorL
            #
            #     else:
            #         ind2 = c // (12 // len(self.wall_im))
            #
            #         # boolean mask
            #         Ub = self.Ub
            #
            #         # fast coordinate extraction (no python loops)
            #         gx = self.Gx[Ub]
            #         gy = self.Gy[Ub]
            #
            #         # fast vectorized texture lookup
            #         wall_vals = self.wall_im2[ind2][gx, gy]
            #
            #         # assign colored pixels
            #         self.Ar[Ub] = wall_vals * colorL
            # else:
            #     ind = min(self.vie,2)
            #
            #     # boolean mask
            #     Ub = self.Ub
            #
            #     # fast coordinate extraction (no python loops)
            #     gx = self.Gx[Ub]
            #     gy = self.Gy[Ub]
            #
            #     # fast vectorized texture lookup
            #     wall_vals = self.wall_im2[ind][gx, gy]
            #
            #     # assign colored pixels
            #     self.Ar[Ub] = wall_vals * colorL

            milliseconds.append(time.perf_counter()*1000)
            label_m.append('render')
            # # 1) Build filt quickly (no expand_dims, no all(2))
            # self.filt[:] = ~(self.Ar == 0).all(axis=2)
            #
            # # filt is (H, W) boolean
            #
            # # 2) If needed, write depth in masked locations
            # if self.text[11:-3] not in liquid_floor:
            #     depth[..., 0][self.filt] = self.S[:, :, -1][self.filt]
            #
            # # 3) Update U
            # self.U &= self.filt  # faster than multiply and no temp arrays
            # self.Ub[:] = self.U  # already boolean so no .astype(bool)
            #
            # # 4) Vectorized Xl construction (remove expand_dims, use broadcasting)
            # self.Xl_small[:] = (
            #         self.S[::2, ::2, 0, None] * (self.a * self.overlap) +
            #         self.S[::2, ::2, 1, None] * (self.b * self.overlap) +
            #         self.X
            # )
            #
            # # Upscale 2× using np.repeat once (repeat twice is slow)
            #
            # self.Xl = cv2.resize(self.Xl_small,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)

            # 5) Make D fast
            # self.Da[:] =1000.#
            milliseconds.append(time.perf_counter() * 1000)
            label_m.append('global clipping')


            # if self.ID in light_wall.keys():
            #     Y0 = [np.linalg.norm(source_pos(i) - R_c) for i in light_wall[self.ID]]
            #     X0 = [x for _, x in sorted(zip(Y0, light_wall[self.ID]))]
            #     for i in X0[:min(len(X0), 4)]:
            #         Xsource = source_pos(i)#+np.array([10*sin(c3/100),10*sin(c3/100),0.])
            #         self.Da[self.Ub] = np.minimum(self.Da[self.Ub], np.linalg.norm(self.Xl[self.Ub] - Xsource, axis=-1))
            #     POS[self.Ub] = self.Da[self.Ub]
            #     self.Ar = self.Ar * level_light
            # else:
            #     self.Da[self.Ub] = np.minimum(self.Da[self.Ub], (np.linalg.norm(self.Xl[self.Ub] - R_c, axis=-1) ** 0.5))
            #     POS[self.Ub] = self.Da[self.Ub]
            #     self.Ar = self.Ar * torch_on * TORCHE ** 3
            # if explo!=0:
            #     explo_R=np.minimum(explo_R,np.expand_dims(np.linalg.norm(self.Xl - np.array([explo_pt[0],explo_pt[1],0.]), axis=-1)*self.U+100*(1-self.U),-1))
            # if explo==4:
            #     self.explo=np.sum(explo_R<20)
            # else:
            #     self.explo=0


            milliseconds.append(time.perf_counter()*1000)
            label_m.append('distance and position light')

            self.time=((np.array(milliseconds) - np.roll(np.array(milliseconds), 1))[1:],label_m)

            return self.Ar

from numba import njit,prange


@njit(parallel=True, fastmath=True)
def render_numba(screenX, screenY, wallN, wall_a, wall_b, wall_x, screenV, screenP, R_c):
    S_out = np.zeros((screenX, screenY, 3))

    for i in prange(screenX):       # Only outer loop is parallel
        for j in range(screenY):    # Sequential inner loop
            for k in range(wallN):
                M = np.empty((3, 3))
                M[:, 0] = wall_a[k] * 1.01
                M[:, 1] = wall_b[k] * 1.01
                M[:, 2] = -screenV[i][j]

                B = (-wall_x[k] + screenP[i][j]).ravel()
                S = min(np.linalg.solve(M, B).ravel())

            S_out[i, j] = S

    return S_out

#
# @njit(parallel=True, fastmath=True)
# def render_ray_step_sign(screenX, screenY, wallN, wall_a, wall_b, wall_x, screenV, screenP,
#                          step_size=10., max_steps=50):
#     """
#     Ray-stepping renderer that correctly handles wall side orientation.
#
#     screenX, screenY: screen resolution
#     wallN: number of walls
#     wall_a, wall_b: wall direction vectors (3D)
#     wall_x: wall positions (3D)
#     screenV: ray direction vector (3D)
#     screenP: ray origin (3D)
#     step_size: distance increment along ray
#     max_steps: maximum number of steps per pixel
#     """
#     S_out = np.zeros((screenX, screenY, 3))  # final intersection points or last position
#
#     for i in prange(screenX):  # only outer loop parallel
#         for j in range(screenY):
#             ray_pos = screenP.copy()
#             intersect_found = False
#
#             for k in range(wallN):
#                 # precompute wall normal and initial side
#                 a = wall_a[k]
#                 b = wall_b[k]
#                 x0 = wall_x[k]
#                 normal = np.cross(a, b)
#                 initial_side = np.dot(ray_pos - x0, normal)
#
#                 # store for comparison
#                 wall_a[k] = normal  # optional if needed
#
#             for s in range(max_steps):
#                 for k in range(wallN):
#                     a = wall_a[k]
#                     b = wall_b[k]  # unused here; just keep loop structure
#                     x0 = wall_x[k]
#                     normal = np.cross(a, b)
#                     side = np.dot(ray_pos - x0, normal)
#
#                     # detect sign flip relative to initial side
#                     if side * initial_side <= 0.0:
#                         S_out[i, j] = ray_pos
#                         intersect_found = True
#                         break  # exit wall loop
#
#                 if intersect_found:
#                     break  # exit stepping loop
#
#                 ray_pos += np.array([1.,0.,0.])
#
#             if not intersect_found:
#                 S_out[i, j] = ray_pos
#
#     return S_out





Xthing, Ything = np.indices((2 * scrnL[0], 2 * scrnL[1]))
RA = 5

Boule = []


class Thing():
    def __init__(self, x0, type_M, vivant, group):
        self.num=len(thing)
        self.thing_t = 1
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
        self.preprocess_walk()
        self.preprocess_attack()
        self.Ar=np.empty_like(Im,dtype=float)
        self.U=np.empty_like(Im[:,:,:-1],dtype=float)
        self.U*=0
        self.SQUARE = np.empty((160, 80), dtype=np.bool_)
        colorT = light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2]
        if light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2].sum() == 0:
            colorT = np.array([1, 1, 1.])
        colorT = light_modif(colorT, level, 0)
        if explo!=0 and np.linalg.norm(self.x0 - explo_pt) < 20:
            colorT*=np.array([1,0,0])
        self.light=colorT

    def calc_norm(self):
        global all_angle
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
            all_angle[self.num]=self.angle

            self.im = self.im_dict[c // 3][self.angle]  # np.minimum(pygame.surfarray.pixels3d(MD[self.type_M][c // 3][self.angle]), 255)
            self.vis = self.vis_dict[c // 3][self.angle]  # np.where(np.sum(self.im, axis=-1) != 0, 1, 0)
            # self.im = np.minimum(pygame.surfarray.pixels3d(MD[self.type_M][c // 3][self.angle]), 255)
            # self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)

    def preprocess_walk(self):
        dict0 = {}
        for c_ in range(4):
            dict1={}
            for angle_ in range(8):
                dict1[angle_*45]=np.minimum(pygame.surfarray.pixels3d(MD[self.type_M][c_][45*angle_]), 255)
            dict0[c_]=dict1.copy()
        self.im_dict=dict0.copy()

        dict0 = {}
        for c_ in range(4):
            dict1={}
            for angle_ in range(8):
                dict1[angle_*45]=np.where(np.sum(self.im_dict[c_][45*angle_], axis=-1) != 0, 1, 0)
            dict0[c_]=dict1.copy()
        self.vis_dict=dict0.copy()

        if self.type_M == 6:
            dict0 = {}
            for c_ in range(4):
                dict1 = {}
                for angle_ in range(8):
                    IMs=MD[self.type_M][c_][45 * angle_].copy()
                    IMs.blit(shield, (0, 0))
                    dict1[angle_ * 45] = np.minimum(pygame.surfarray.pixels3d(IMs), 255)
                dict0[c_] = dict1.copy()
            self.im_dict_shield = dict0.copy()

            dict0 = {}
            for c_ in range(4):
                dict1 = {}
                for angle_ in range(8):
                    dict1[angle_ * 45] = np.where(np.sum(self.im_dict_shield[c_][45 * angle_], axis=-1) != 0, 1, 0)
                dict0[c_] = dict1.copy()
            self.vis_dict_shield = dict0.copy()




    def preprocess_attack(self):
        dict0 = {}
        for c_ in range(8):
            dict0[c_]=np.minimum(pygame.surfarray.pixels3d(MDa[self.type_M][c_]), 255)
        self.im_dict_attack=dict0.copy()
        dict0 = {}
        for c_ in range(8):
            dict0[c_]=np.where(np.sum(self.im_dict_attack[c_], axis=-1) != 0, 1, 0)
        self.vis_dict_attack=dict0.copy()

        if self.type_M == 6:
            dict0 = {}
            for c_ in range(8):
                IMs=MDa[self.type_M][c_].copy()
                IMs.blit(shield,(0,0))
                dict0[c_] = np.minimum(pygame.surfarray.pixels3d(IMs), 255)
            self.im_dict_attack_s = dict0.copy()
            dict0 = {}
            for c_ in range(8):
                dict0[c_] = np.where(np.sum(self.im_dict_attack_s[c_], axis=-1) != 0, 1, 0)
            self.vis_dict_attack_s = dict0.copy()

    def walk(self):

        self.im = self.im_dict[c//3][self.angle]#np.minimum(pygame.surfarray.pixels3d(MD[self.type_M][c // 3][self.angle]), 255)
        self.vis = self.vis_dict[c//3][self.angle]#np.where(np.sum(self.im, axis=-1) != 0, 1, 0)
        if self.type_M == 6 and self.shield == 0:
            self.im = self.im_dict_shield[c // 3][self.angle]
            self.vis = self.vis_dict_shield[c // 3][self.angle]
            # im=MD[self.type_M][c // 3][self.angle].copy()
            # if c%2:
            #     im.blit(shield,(0,0))
            # else:
            #     im.blit(shield2, (0, 0))
            # self.im = np.minimum(pygame.surfarray.pixels3d(im), 255)
            # self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)

    def test_behind(self):
        if self.f0[0] > 0 and abs(self.f0[1] / self.f0[0]) < TAN2 + 0.5:
            if self.norm <= horizon[min(max(int(self.width // 2 + self.DX + scrnL[0]) // 2, 0), scrnL[0] - 1)]:
                return True
            else:
                return False
        else:
            return False

    def move(self):
        global VIE, HIT,all_x_e,all_attack_range,all_range
        self.explo_zone()
        self.attack_range = self.norm <= 5
        if self.range == 1:
            self.attack_range = (self.test_behind()) & (self.norm <= 20)
        all_attack_range[self.num]=self.attack_range
        all_range[self.num] = self.range
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
        all_x_e[self.num]=np.concatenate((self.x0, np.array([2 * self.z])))

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
            self.vie -= explo_deg[explo_type]*self.shield
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
        global Killed_E,x_d,all_mort,all_light_e
        milliseconds=[time.perf_counter()*1000]
        label_m=[]
        # if self.attack_range and self.active and self.vie > 0:
        #     if self.range == 0:
        #         self.im = self.im_dict_attack[c//4]
        #         self.vis=self.vis_dict_attack[c//4]
        #     else:
        #
        #         self.im = self.im_dict_attack[c2//8]
        #         self.vis=self.vis_dict_attack[c2//8]
        #
        #     if self.type_M==6 and self.shield==0:
        #         if self.range == 0:
        #             im=self.im_dict_attack_s[c // 4]
        #             vis=self.vis_dict_attack_s[c // 4]
        #         else:
        #             im = self.im_dict_attack_s[c2 // 8]
        #             vis = self.vis_dict_attack_s[c2 // 8]
        #         self.im = im
        #         self.vis =vis

        milliseconds.append(time.perf_counter()*1000)
        label_m.append('attack')
        if (self.inline and shoot == 2 and (self.attack_range or arme != 0)) and self.vie > 0 and arme != 4:
            # self.im = self.im_dict_attack[4]
            # self.vis = self.vis_dict_attack[4]
            # if self.type_M==6 and self.shield==0:
            #     self.im = self.im_dict_attack_s[4]
            #     vis = self.vis_dict_attack_s[4]

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
        milliseconds.append(time.perf_counter()*1000)
        label_m.append('hit and move')
        if self.vie <= 0:
            # self.im = self.im_dict_attack[4 + int(self.mort)]
            # self.vis = self.vis_dict_attack[4 + int(self.mort)]
            all_mort[self.num]=self.mort
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
                    global all_range,all_x_e,all_types_e,all_obj_mon,all_angle,all_attack_range,all_RA
                    all_x_e=np.concatenate((all_x_e,np.array([np.concatenate((monst.x0, np.array([2 * monst.z])))])),axis=0)
                    all_types_e=np.concatenate((all_types_e,np.array([monst.type_M])))
                    all_mort = np.concatenate((all_mort, np.array([monst.mort])))
                    all_obj_mon = np.concatenate((all_obj_mon, np.array([monst.thing_t])))
                    all_angle = np.concatenate((all_angle, np.array([monst.angle])))
                    all_attack_range = np.concatenate((all_attack_range, np.array([monst.attack_range])))
                    all_range = np.concatenate((all_range, np.array([monst.range])))
                    all_light_e = np.concatenate((all_light_e, np.array([monst.light])))
                    all_RA= np.concatenate((all_RA, np.array([monst.RA])))
                for i in range(randint(50, 60)):
                    Boule.append(boule(self.x0[0], self.x0[1], self.z - 1, -pi * random(), 2 * pi * random(), 1, 5,
                                       'image/effects/vert.png', 1))
        milliseconds.append(time.perf_counter()*1000)
        label_m.append('dead')
        colorT = light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2]
        if light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2].sum() == 0:
            colorT = np.array([1, 1, 1.])
        colorT = light_modif(colorT, level, c3)
        if explo!=0 and np.linalg.norm(self.x0 - explo_pt) < 20:
            colorT*=np.array([1,0,0])
        self.light=colorT
        all_light_e[self.num]=self.light
        #
        #
        # self.SQUARE[:] = np.all(self.norm <= depth, axis=-1) & (Xthing <= self.width + self.DX + scrnL[0]) & (
        #             Xthing >= self.DX + scrnL[0]) & (Ything <= self.widthY + self.DY + scrnL[1]) & (
        #                      Ything >= self.DY + scrnL[1]).astype(bool)
        milliseconds.append(time.perf_counter()*1000)
        label_m.append('square')


        # # channel 0
        # self.U[..., 0] = (Xthing - self.DX - scrnL[0]) /self.width
        # self.U[..., 0] *= self.SQUARE
        #
        # # channel 1
        # self.U[..., 1] = (Ything - self.DY - scrnL[1]) / self.widthY
        # self.U[..., 1] *= self.SQUARE


        milliseconds.append(time.perf_counter()*1000)
        label_m.append('U')
        # self.G = np.maximum((self.U * 160).astype(int), 0)
        milliseconds.append(time.perf_counter()*1000)
        label_m.append('G')

        # self.Ar=self.im[self.G[..., 0], self.G[..., 1]]*colorT

        milliseconds.append(time.perf_counter()*1000)
        label_m.append('Ar')
        # if light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2].sum() == 0:
        #     self.Ar = self.Ar * torch_on * TORCHE3
        #     self.Ar /=  0.1 * np.sqrt(self.norm)
        #     np.minimum(self.Ar, 255, out=self.Ar)
        # else:
        #     self.Ar = self.Ar * level_light
        milliseconds.append(time.perf_counter()*1000)
        label_m.append('light')

        # self.Ut = self.vis[self.G[..., 0], self.G[..., 1]]

        milliseconds.append(time.perf_counter()*1000)
        label_m.append('Ut')
        if shoot==1:
            self.inline=False
            self.shot = []
            x_d0=[]
            for i in range(len(x_d)):
                # inline = min(np.sum(self.Ut[int(scrnL[0]*(1+2*x_d[i][0]) - gun_width):int(scrnL[0]*(1+2*x_d[i][0]) + gun_width), int(2 * scrnL[1]*(1+2*x_d[i][1]) // 2 - gun_width):int(2 * scrnL[1]*(1+2*x_d[i][1]) // 2 + gun_width)]),
                #                 1)

                inline_e = index_e[int(2*scrnL[0]*(1+2*x_d[i][0]) - gun_width):int(2*scrnL[0]*(1+2*x_d[i][0]) + gun_width), int(2*2 * scrnL[1]*(1+2*x_d[i][1]) // 2 - gun_width):int(2*2 * scrnL[1]*(1+2*x_d[i][1]) // 2 + gun_width)]
                inline=(inline_e==self.num).any()

                self.inline=(self.inline or inline)
                self.shot.append(i)
                if not inline:
                    x_d0.append(x_d[i])
            x_d=x_d0

        milliseconds.append(time.perf_counter()*1000)
        label_m.append('inline')
        self.time = ((np.array(milliseconds) - np.roll(np.array(milliseconds), 1))[1:], label_m)

        return self.Ar

import matplotlib.pyplot as plt
class Object():
    def __init__(self, x0, type_M, vivant, group):
        self.num=len(thing)
        self.thing_t = 0
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
        self.preprocess_walk()
        self.U=np.empty_like(Im[:,:,:-1],dtype=float)
        self.U*=0
        colorT = light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2]
        if light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2].sum() == 0:
            colorT = np.array([1, 1, 1])
        colorT = light_modif(colorT, level, 0)
        if explo!=0 and np.linalg.norm(self.x0 - explo_pt) < 20:
            colorT*=np.array([1,0,0])
        self.light=colorT

    def preprocess_walk(self):

        dict1={}
        for angle_ in range(8):
            dict1[angle_*45]=np.minimum(pygame.surfarray.pixels3d(MO[self.type_M][45*angle_]), 255)
        self.im_dict=dict1.copy()
        dict1={}
        for angle_ in range(8):
            dict1[angle_*45]=np.where(np.sum(self.im_dict[45*angle_], axis=-1) != 0, 1, 0)
        self.vis_dict=dict1.copy()

    def calc_norm(self):
        global VIE, AMMO, Picked_O, CARTE, logL,all_angle
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
            all_angle[self.num]=self.angle

            #self.im = np.minimum(pygame.surfarray.pixels3d(MO[self.type_M][self.angle]), 255)
            self.im=self.im_dict[self.angle]
            if self.color != 0:
                u = [0, 0, 0]
                u[self.color - 1] = 1
                self.im = u * self.im
            self.vis=self.vis_dict[self.angle]
            #self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)

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
                if self.color - 1==2:
                    AMMO[self.color - 1] += 40
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
        global Boule, explo, VIE, explo_pt, Killed_O, HIT,x_d,explo_type
        self.time=(0,0)
        milliseconds=[time.perf_counter()*1000]
        label_m=[]

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
                explo_type=0
                if self.norm < 15:
                    VIE = VIE - explo_deg[explo_type]/2
                    HIT = 1
                    s = pygame.mixer.Sound("son/aie.ogg")
                    s.play()#

        if self.mort < 4 and self.vie <= 0:# improve
            self.im = np.minimum(pygame.surfarray.pixels3d(Mod[self.type_M][int(self.mort)]), 255)
            self.vis = np.where(np.sum(self.im, axis=-1) != 0, 1, 0)
            self.mort = min(self.mort + 1, 3)

        colorT = light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2]
        if light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2].sum() == 0:
            colorT = np.array([1, 1, 1])
        colorT = light_modif(colorT, level, c3)
        if explo!=0 and np.linalg.norm(self.x0 - explo_pt) < 20:
            colorT*=np.array([1,0,0])
        self.light=colorT
        all_light_e[self.num]=self.light

        # SQUARE = np.all(self.norm <= depth, axis=-1) & (Xthing <= self.width + self.DX + scrnL[0]) & (
        #             Xthing >= self.DX + scrnL[0]) & (Ything <= self.widthY + self.DY + scrnL[1]) & (
        #                      Ything >= self.DY + scrnL[1])



        # # channel 0
        # self.U[..., 0] = (Xthing - self.DX - scrnL[0]) /self.width
        # self.U[..., 0] *= SQUARE
        #
        # # channel 1
        # self.U[..., 1] = (Ything - self.DY - scrnL[1]) / self.widthY
        # self.U[..., 1] *= SQUARE
        #
        # self.G = np.maximum((self.U * 160).astype(int), 0)

        # self.Ar = self.im[self.G[..., 0], self.G[..., 1]] * colorT
        # if light_array[int(self.x0[0] + 101) // 2][int(self.x0[1] + 101) // 2].sum() == 0:
        #     self.Ar = self.Ar * torch_on * TORCHE3
        #     self.Ar /=  0.1 * np.sqrt(self.norm)
        #     np.minimum(self.Ar, 255, out=self.Ar)
        # else:
        #     self.Ar = self.Ar * level_light
        # self.Ut = self.vis[self.G[..., 0], self.G[..., 1]]
        self.shot=[]
        if shoot==1:
            x_d0=[]
            self.inline=False
            for i in range(len(x_d)):
                # inline = min(np.sum(self.Ut[int(scrnL[0]*(1+2*x_d[i][0]) - gun_width):int(scrnL[0]*(1+2*x_d[i][0]) + gun_width), int(2 * scrnL[1]*(1+2*x_d[i][1]) // 2 - gun_width):int(2 * scrnL[1]*(1+2*x_d[i][1]) // 2 + gun_width)]),
                #                 1)
                inline=True
                self.inline=(self.inline or inline)
                self.shot.append(i)
                if not inline:
                    x_d0.append(x_d[i])
            x_d=x_d0
        milliseconds.append(time.perf_counter()*1000)
        label_m.append('end')
        self.time = ((np.array(milliseconds) - np.roll(np.array(milliseconds), 1))[1:], label_m)
        return None


class Object_parallax():
    def __init__(self, x0, type_M):
        self.RA = 500.
        self.x0 = x0[:-1]
        self.decal=0.
        self.norm = np.linalg.norm(self.x0 - x + [1, 0])
        self.width = 2 * scrnL[0] / self.norm
        self.widthY = 2 * scrnL[0] / self.norm
        self.DX = 0
        self.DY = 0
        self.type_M = levelD[level]['obj_parallax'][type_M]
        self.im=pygame.image.load('./image/obj_parallax/O_%s.png'%str(self.type_M)).convert_alpha()
        if self.type_M==1:
            self.im2 = pygame.image.load('./image/obj_parallax/O_%s.1.png' % str(self.type_M)).convert_alpha()
            self.im3 = pygame.image.load('./image/obj_parallax/O_%s.2.png' % str(self.type_M)).convert_alpha()
        self.z = x0[-1]
        self.erupt=0


    def calc_norm(self):
        if x[1]<600 and x[1]>300 and self.decal==0:
            self.x0=self.x0+x
            self.decal=1
        self.f0 = (self.x0 - x) @ rot_plan(-ang[0])
        self.norm = np.linalg.norm(self.x0 - x)
        self.width = self.RA * 2 * scrnL[0] / self.f0[0]
        self.widthY = self.RA * 2 * scrnL[0] / self.f0[0]
        self.DX = -self.RA * scrnL[0] / self.f0[0] + scrnL[0] * (self.f0[1] / self.f0[0]) / TAN2
        self.DY = -(self.RA * scrnL[0] * 1.7 - 5 * scrnL[0]) / self.f0[0] - scrnL[1] * tan(ang[1]) / TAN1 - (
                    2 * scrnL[1] * (z - self.z) / self.f0[0]) / TAN1

        self.x0+=np.array([10,0])

        if self.x0[0]>10000:
            self.x0[0]=self.x0[0]-20000


    def test_behind(self):
        if self.f0[0] > 0 and abs(self.f0[1] / self.f0[0]) < TAN2 + 0.5:
            return True
        else:
            return False

    def affiche(self):
        if self.test_behind():
            if self.type_M==1:
                if randint(0,100)==0:
                    self.erupt=2
                if self.erupt==0:
                    self.imA = pygame.transform.scale(self.im, (int(2*scrnL[0]*self.RA/self.f0[0]),int(2*scrnL[0]*self.RA/self.f0[0])))
                if self.erupt==1:
                    self.imA = pygame.transform.scale(self.im2, (
                    int(2 * scrnL[0] * self.RA / self.f0[0]), int(2 * scrnL[0] * self.RA / self.f0[0])))
                if self.erupt==2:
                    self.imA = pygame.transform.scale(self.im3, (
                    int(2 * scrnL[0] * self.RA / self.f0[0]), int(2 * scrnL[0] * self.RA / self.f0[0])))
                self.erupt=max(0,self.erupt-0.5)

            else:
                self.imA = pygame.transform.scale(self.im, (
                int(2 * scrnL[0] * self.RA / self.f0[0]), int(2 * scrnL[0] * self.RA / self.f0[0])))
            transparency = 255 * min(1, (2000/self.f0[0]))
            self.imA.fill((255,255,255,transparency), special_flags=BLEND_RGBA_MULT)
            fond.blit(self.imA,  (self.DX+ scrnL[0], 0.3*scrnL[1]+ self.DY+ scrnL[1]))


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
                authorized_map[int(self.p[1] + 101) // 2][int(self.p[0] + 101) // 2] == 2 and level_w_transp[int(self.p[1] + 101) // 2][int(self.p[0] + 101) // 2]==0)) or (self.p[-1] - z) > 5 + \
                hmap[int(self.p[1] + 101) // 2][int(self.p[0] + 101) // 2] or (
                self.p[-1] - z) < -5 or self.lifetime > 500:

            return True
        else:
            return False

    def affiche(self):
        if self.f0[0] > 0 and abs(self.f0[1] / self.f0[0]) < TAN2 + 0.5 and self.D <= \
                depth[int(self.X * scrnL[0]) % (2 * scrnL[0])][int(self.Y * scrnL[1] % (2 * scrnL[1]))]:

            colorT = light_array[int(self.p[0] + 101) // 2][int(self.p[1] + 101) // 2]
            if light_array[int(self.p[0] + 101) // 2][int(self.p[1] + 101) // 2].sum() == 0:
                colorT = np.array([1, 1, 1.])
            colorT = light_modif(colorT, level, c3)

            self.imb=self.im.copy()
            self.imb.fill(255*colorT, special_flags=BLEND_RGB_MULT)

            self.imA = pygame.transform.scale(self.imb, (
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
                level_map[int(self.p[1] + 101) // 2][int(self.p[0] + 101) // 2] == 1) and self.cool == 0 and level_w_transp[int(self.p[1] + 101) // 2][int(self.p[0] + 101) // 2]==0:
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
                        self.v *= 0.3
                        self.cool = 20
                        break

        if self.lifetime == 50:
            global explo, Boule, explo_pt, VIE, HIT,explo_type
            s = pygame.mixer.Sound("son/barril.ogg")
            for i in range(randint(5, 10)):
                Boule.append(
                    boule(self.p[0], self.p[1], self.p[2], pi * random(), 2 * pi * random(), 2 * random() + 0.2, 1,
                          'image/effects/spark.png', 1))
            s.play()
            explo = 5
            explo_pt = self.p[:-1]
            explo_type=1
            if self.D < 15:
                VIE = VIE - explo_deg[explo_type]/2
                HIT = 1
                s = pygame.mixer.Sound("son/aie.ogg")
                s.play()
        if self.lifetime >= 50 and self.lifetime <= 56:
            self.im = []
            self.size = 5000
            for i in range(4):
                self.im.append(pygame.image.load('image/effects/explo%s.png' % str(self.lifetime - 50)))

        if self.lifetime == 57:
            return True
        return False

    def affiche(self):
        if self.f0[0] > 0 and abs(self.f0[1] / self.f0[0]) < TAN2 + 0.5 and self.D <= \
                depth[int(self.X * scrnL[0]) % (2 * scrnL[0])][int(self.Y * scrnL[1] % (2 * scrnL[1]))]:

            self.imA = pygame.transform.scale(self.im[(c) % 4], (
                min(int(Ratio*self.size / self.f0[0]), window[1] // 1), min(int(Ratio*self.size / self.f0[0]), window[1] // 1)))

            colorT = light_array[int(self.p[0] + 101) // 2][int(self.p[1] + 101) // 2]
            if light_array[int(self.p[0] + 101) // 2][int(self.p[1] + 101) // 2].sum() == 0:
                colorT = np.array([1, 1, 1.])
            colorT = light_modif(colorT, level, c3)

            self.imb=self.imA.copy()
            self.imb.fill(255*colorT, special_flags=BLEND_RGB_MULT)

            if len(IS)>0:
                for j,i in enumerate(IS):
                    if i.U[int(self.X * scrnL[0]) % (2 * scrnL[0])][int(self.Y * scrnL[1] % (2 * scrnL[1]))] and i.X[0,0,-1]/2<self.p[-1]:
                        colorliquid=IS_rendered[j][int(self.X * scrnL[0]) % (2 * scrnL[0])][int(self.Y * scrnL[1] % (2 * scrnL[1]))]
                        self.imb.fill(colorliquid, special_flags=BLEND_RGB_MULT)


            shift = min(int(self.size / self.f0[0]), window[1] // 1) // 2
            fond.blit(self.imb, (int((window[0] // 2) * self.X) - shift, int((window[1] // 2) * self.Y) - shift))


font = pygame.font.Font('freesansbold.ttf', 13)


def draw_vie():
    pygame.draw.line(fenetre, (255, 0, 0), (int(0.11 * window[0]), int(1.05 * window[1])),
                     (int(0.11 * window[0] + 0.1 * window[0]), int(1.05 * window[1])), int(10*window[0]/960))
    pygame.draw.line(fenetre, (0, 255, 0), (int(0.11 * window[0]), int(1.05 * window[1])),
                     (int(0.11 * window[0] + 0.1 * window[0] * max(VIE, 0) / 100), int(1.05 * window[1])), int(10*window[0]/960))


def draw_hud():
    global back, font, code1, code2, fontC,incinerate,incinerate2
    fontC = pygame.font.Font('freesansbold.ttf', int(32 * window[0] / (12 * scrnL[0])))
    code1 = pygame.transform.scale(code01, (int(0.8 * window[1]), int(0.16 * window[1])))
    code2 = pygame.transform.scale(code02, (int(0.8 * window[1]), int(0.16 * window[1])))
    incinerate = pygame.transform.scale(incinerate, (int(0.8 * window[1]), int(0.16 * window[1])))
    incinerate2 = pygame.transform.scale(incinerate2, (int(0.8 * window[1]), int(0.16 * window[1])))
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
        if TotAr-1>i:
            u = [100, 100, 100]
            u[i] = 250
            pygame.draw.line(fenetre, (50, 50, 50), (int(0.8 * window[0]), window[1] + int((0.05 * i + 0.04) * window[1])),
                             (int(0.95 * window[0]), window[1] + int((0.05 * i + 0.04) * window[1])), 20)
            text = font.render('AMMO type ' + str(i) + '   ' + str(AMMO[i]), True, tuple(u))
            textRect = text.get_rect()
            textRect.topleft = (int(0.8 * window[0]), window[1] + int((0.05 * i + 0.025) * window[1]))
            fenetre.blit(text, textRect)
    if TotAr>4:
        nades = pygame.image.load('image/effects/grenade0.png')
        nades = pygame.transform.scale(nades, (int(0.025 * window[0]), int(0.025 * window[0])))
        pygame.draw.line(fenetre, (50, 50, 50), (int(0.97 * window[0]),window[1] + int((0.025 * 0 + 0.01) * window[1]) ),
                         (int(0.97 * window[0]), window[1] + int((0.025 * 6 + 0.01) * window[1]) ), 20)
        for i in range(AMMO[3]):
            fenetre.blit(nades,(int(0.96 * window[0]), window[1] + int((0.025 * i + 0.01) * window[1])))



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
nades=pygame.image.load('image/effects/grenade0.png')
nades=pygame.transform.scale(nades,(int(0.025*window[0]),int(0.025*window[0])))
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
                pygame.transform.scale(pygame.image.load('image/gun/' + i[j]), (2*2 * scrnL[0], 2*2 * scrnL[1])), window)
        else:
            MG2[j] = pygame.transform.smoothscale(
                pygame.transform.scale(pygame.image.load('image/gun/' + i[j]), (2*2 * scrnL[0], 2*2 * scrnL[1])), window)
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
DEGAT = [35, 25, 70, 30, 150]
explo_deg=[50,DEGAT[4],5000]
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
    MG3 = []
    for c, i in enumerate(MG1):
        temp_d = {}
        for k in i.keys():
            arr = pygame.surfarray.pixels3d(i[k])
            arr = arr.astype(float)
            arr[np.any(arr != 0, axis=-1)] = np.minimum(25 + arr[np.any(arr != 0, axis=-1)] * (230 / 255), 255)
            arr = arr.astype(int)
            temp_d[k] = pygame.surfarray.make_surface(arr)
        MG3.append(temp_d.copy())
    MG1 = MG3
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
    if N == 4:
        fenetre.fill((0,0,0))
        fontH = pygame.font.Font('freesansbold.ttf', int(18 * window[0] / (12 * scrnL[0])))
        anim = 'General Fisher: Here is your next mission, you will go to this abandonned mine and track the Viper organisation. From the archives you recovered, it appears that this unknown group infiltrated the biolab and created all these giant bugs. Understood!'

        IMLOAD = pygame.transform.scale(pygame.image.load('image/animation/a4_0.png'),
                                        (window[0], int(window[1] * 1.2)))
        IMLOAD.set_colorkey((0,0,0))
        IMLOAD2 = pygame.transform.scale(pygame.image.load('image/animation/a4_1.png'),
                                        (window[0], int(window[1] * 1.2)))
        IMLOAD2.set_colorkey((0, 0, 0))
        IMLOAD3 = pygame.transform.scale(pygame.image.load('image/animation/a4_2.png'),
                                        (window[0], int(window[1] * 1.2)))
        IMLOAD3.set_colorkey((0, 0, 0))
        linenumber = 0
        indK = 0
        for i in range(len(anim)+50):
            if i%10==0:
                fenetre.blit(IMLOAD, (0, 0))
            if i % 10 == 5 and i<len(anim):
                if i<5*10 or i>10*10:
                    fenetre.blit(IMLOAD2, (0, 0))
                else:
                    fenetre.blit(IMLOAD3, (0, 0))


            if i<len(anim):
                text = fontH.render(anim[i], True, (255, 255, 255))
                text2 = fontH.render(anim[indK:i], True, (255, 255, 255))
                textRect = text2.get_rect()
                fenetre.blit(text, (
                    int(window[0] * 0.05 + textRect[2]),
                    int(window[1] * 1.01 + 18 * window[0] / (12 * scrnL[0]) * linenumber)))

                if textRect[2] > int(0.9 * window[0]):
                    indK = i
                    linenumber += 1


            pygame.display.flip()
            pygame.time.wait(50)
        return 1
    if N == 5:
        IMLOAD = pygame.transform.scale(pygame.image.load('image/animation/a5_0.png'),
                                        (int(1.2*window[0]), int(1.2*window[1] * 1.2)))
        IMLOAD_b = pygame.transform.scale(pygame.image.load('image/animation/a5_1.png'),
                                        (int(1.2*window[0]), int(1.2*window[1] * 1.2)))
        IMLOAD.set_colorkey((0, 0, 0))
        t=np.linspace(0,1.,300)
        t0=0.5
        sig=0.15
        a_rand=np.random.random(300)
        shake0=np.sin(2*pi*t*a_rand*10)*np.exp(-(t-t0)**2/sig**2)*a_rand
        a_rand = np.random.random(300)
        shake1 = np.sin(2 * pi * t * a_rand * 10) * np.exp(-(t - t0) ** 2 / sig ** 2) * a_rand

        for i in range(300):
            fenetre.fill((0,0,0))


            if i==75:
                s = pygame.mixer.Sound("son/container.ogg")
                s.play()

            fenetre.blit(IMLOAD, (-int(.1*window[0])+int(10*shake0[i]), int(10*shake1[i])-int(.1*window[0])))

            fenetre.blit(IMLOAD_b, (-int(.1 * window[0]) + int(10 * shake0[max(i-10,0)]), int(10 * shake1[max(i-10,0)])-int(.1*window[0])))
            pygame.draw.rect(fenetre, (0, 0, 0), (0, 0.5 * window[0], window[0], 0.25 * window[1]))
            pygame.display.flip()
            pygame.time.wait(1)
        sig=0.5
        shake3 = (np.sin(2 * pi * t*3 ) * np.exp(-(t - t0) ** 2 / sig ** 2))*360/(2*pi)
        shake4 = -(np.sin(2 * pi * t*3 )**2 * np.exp(-(t - t0) ** 2 / sig ** 2))*360/(2*pi)
        # plt.plot(shake3)
        # plt.plot(shake4)
        # plt.show()
        for i in range(300):
            fenetre.fill((0,0,0))


            if i==0:
                s = pygame.mixer.Sound("son/grince.ogg")
                s.play()

            fenetre.blit(IMLOAD, (-int(.1*window[0])+int(0.3*shake3[i]), int(0.2*shake4[i])-int(.1*window[0])))
            fenetre.blit(IMLOAD_b, (-int(.1 * window[0]) + int(0.3 * shake3[max(i-10,0)]), int(0.2 * shake4[max(i-1,0)])-int(.1*window[0])))
            pygame.draw.rect(fenetre, (0, 0, 0), (0, 0.5 * window[0], window[0], 0.25 * window[1]))
            pygame.display.flip()
            pygame.time.wait(1)
        t = np.linspace(0, 1., 300)
        t0 = 0.25
        sig = 0.15
        a_rand = np.random.random(300)
        shake0 = np.sin(2 * pi * t * a_rand * 10) * np.exp(-(t - t0) ** 2 / sig ** 2) * a_rand
        a_rand = np.random.random(300)
        shake1 = np.sin(2 * pi * t * a_rand * 10) * np.exp(-(t - t0) ** 2 / sig ** 2) * a_rand

        for i in range(300):
            if i==0:
                s = pygame.mixer.Sound("son/container.ogg")
                s.play()
            fenetre.fill((0, 0, 0))

            fenetre.blit(IMLOAD, (-int(.1*window[0])+int(10 * shake0[i]), int(10 * shake1[i])-int(.1*window[0])))
            fenetre.blit(IMLOAD_b, (-int(.1 * window[0]) + int(10 * shake0[max(i-10,0)]), int(10 * shake1[max(i-10,0)])-int(.1*window[0])))
            pygame.draw.rect(fenetre, (0, 0, 0), (0, 0.5 * window[0], window[0], 0.25 * window[1]))
            pygame.display.flip()
            pygame.time.wait(1)


        fontH = pygame.font.Font('freesansbold.ttf', int(32 * window[0] / (12 * scrnL[0])))
        anim = 'Ok, it stopped moving, I must have arrived to the lizard men city ! Let s get out of here !'
        linenumber = 0
        indK = 0
        for i in range(len(anim)+50):
            if i<len(anim):
                text = fontH.render(anim[i], True, (255, 255, 255))
                text2 = fontH.render(anim[indK:i], True, (255, 255, 255))
                textRect = text2.get_rect()
                fenetre.blit(text, (
                    int(window[0] * 0.05 + textRect[2]),
                    int(window[1] * 1.01 + 32 * window[0] / (12 * scrnL[0]) * linenumber)))

                if textRect[2] > int(0.9 * window[0]):
                    indK = i
                    linenumber += 1


            pygame.display.flip()
            pygame.time.wait(50)
        return 1
    return 0


modif_game = ['0_1', '0_V2', '1_2', '1_6', '2_1', '2_6', '3_2', '3_3', '3_4', '3_G99', '4_2', '4_3', '4_4','5_2','5_1','5_8','5_7']

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
        draw_AMMO()
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
    if num == '5_2':
        TotAr = 5
        AMMO[3] += 5
        draw_AMMO()
        logL.append('grenades picked')
        s = pygame.mixer.Sound("son/plop_special.ogg")
        s.play()

    if num == '5_1':

        activatedT.append(Trig_liste[5][1])
    if num == '5_8':
        activatedT.remove(Trig_liste[5][1])
    if num == '5_7':
        logL.append('doc picked')
        s = pygame.mixer.Sound("son/plop_special.ogg")
        s.play()
        docs.insert(0, 'page28.png')
        docs.insert(0, 'page27.png')
        docs.insert(0, 'page26.png')
        docs.insert(0, 'page25.png')
        docs.insert(0, 'journal.png')

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
        ll=10*window[0]/960
        pygame.draw.polygon(fenetre, (255, 255, 255), (
        (xm, ym),
        (xm - ll * cos(ang[0] + pi / 4 + pi / 2), ym + ll * sin(ang[0] + pi / 4 + pi / 2)),
        (xm + ll * cos(ang[0] + pi / 2), ym - ll * sin(ang[0] + pi / 2)),
        (xm - ll * cos(ang[0] - pi / 4 + pi / 2), ym + ll * sin(ang[0] - pi / 4 + pi / 2))), 0)
        pygame.draw.polygon(fenetre, (255, 0, 0), (
        (xm, ym),
        (xm - ll * cos(ang[0] + pi / 4 + pi / 2), ym + ll * sin(ang[0] + pi / 4 + pi / 2)),
        (xm + ll * cos(ang[0] + pi / 2), ym - ll * sin(ang[0] + pi / 2)),
        (xm - ll * cos(ang[0] - pi / 4 + pi / 2), ym + ll * sin(ang[0] - pi / 4 + pi / 2))), 3)
        #pygame.draw.circle(fenetre, (255, 0, 0), (xm, ym), int(0.01 * window[1]))

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
    global cell_start, cell_count, cell_objects,cell_array_z,cell_array_N,all_inv_det,all_ab,all_aa,all_bb,all_n,all_a,all_b,all_X,all_walls,cell_size,cell_array,CARTE,horizon2,height_list,op,level_w_transp,SKY0_im,LAND0_im,SKY0,LAND0,stairs, torch_on, lifts, activatedT, TotAr, MAP, v, tuto, level, groupD, indk, startmsg, activatedT, queueT, linenumber, back, dicoTEXT, Trig_liste, AMMO, level_w, level_h, level_map, zmap, light_wall, hmap, authorized_map, M_liste, light_color, light_array, ratio, level_light, wall, doors, h_wall, thing, ennemies
    CARTE = [0, 0, 0]
    level = int(level_name)
    if level==5:
        SKY0 = pygame.surfarray.pixels3d(pygame.image.load('image/ciel/ciel1.png'))
        LAND0 = pygame.surfarray.pixels3d(pygame.image.load('image/ciel/lanscape1.png'))
        LAND0 = np.where(np.expand_dims(((LAND0 - np.array([0, 255, 255])) == 0).all(-1), -1), -2, LAND0)
        SKY0_im = pygame.image.load('image/ciel/ciel1.png')
        LAND0_im = pygame.image.load('image/ciel/lanscape1.png')
    if level==6:
        SKY0 = pygame.surfarray.pixels3d(pygame.image.load('image/ciel/ciel2.png'))
        LAND0 = pygame.surfarray.pixels3d(pygame.image.load('image/ciel/lanscape2.png'))
        LAND0 = np.where(np.expand_dims(((LAND0 - np.array([0, 255, 255])) == 0).all(-1), -1), -2, LAND0)
        SKY0_im = pygame.image.load('image/ciel/ciel2.png')
        LAND0_im = pygame.image.load('image/ciel/lanscape2.png')
        #LAND0_im.set_colorkey((0, 255, 255))

        op = []
        for i in range(30):
            xra = np.random.randint(-10000, 10000)
            yra = np.random.randint(200, 10000)
            signra = (-1) ** randint(0, 1)
            rr=1-min(randint(0,5),1)
            op.append(Object_parallax([ xra, yra * signra, 200], rr))

    TotAr = level_arme[level]
    v = 0
    skip = True
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
    #fenetre.fill((0, 0, 0))
    pygame.display.flip()
    pygame.time.wait(500)
    pygame.mouse.set_pos([window[0] // 2 - 10, window[1] // 2 + 10])

    tuto = 0
    reset_all()
    draw_hud()
    draw_cards()
    draw_vie()
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
    AMMO[2] = max(min(TotAr - 3, 1) * 50, 0)
    AMMO[3] = max(min(TotAr - 4, 1) * 5, 0)
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
    level_w_transp= pickle.load(f)
    level_w_transp=level_w_transp.T

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
        H = i[6]
        xw.append(i[4] * 2 - H)
        b.append(i[5] * 2)
        im = 'image/wall/wall' + str(levelD[level]['wall'][i[2][0]]) + '.png'
        im2 = 'image/wall/wall' + str(levelD[level]['wall'][i[2][1]]) + '.png'
        if i[3] != 0:
            im = 'image/door/' + str(levelD[level]['door'][i[2][0]]) + '.png'
            im2 = 'image/door/' + str(levelD[level]['door'][i[2][1]]) + '.png'

        wall.append(Wall([0., 0., 5 * 2 + H], b, xw, [im, im2, i[2][2]], i[3], i[7], i[8], i[9],i[10]))

        if i[3] != 0:
            doors.append(wall[-1])
    [i.texture(5, 5) for i in wall]

    lenH = 0
    app=0
    for i in level_h:

        a = 2 * i[0]
        b = 2 * i[1]
        x1 = 2 * i[2]

        H = i[3]
        x2 = 2 * i[2] + [0, 0, 10+H*0.001]
        im = 'image/flat/roof' + str(levelD[level]['flat'][i[4]]) + '.png'
        if len(i)==5:
            app+=1
            wall.append(Wall(list(a), list(b), list(x1 + [0, 0, -H]), [im, im, i[2][2]], 0, 0, 1, 0,0))
        if i[4] == 9:
            wall[-1].transp = 1
        liquid=[i.split('r')[1][:-1] for i in liquid_floor]
        if str(levelD[level]['flat'][i[4]]) in liquid:  # WARNING TO FINISH
            lenH += 1
            im = 'image/flat/L_floor' + str(levelD[level]['flat'][i[4]]) + '.png'
            app += 1
            wall.append(Wall(list(-a), list(b), list(a + x2), [im, im, i[2][2]], 0, 0, 1, 0,0))
            im = 'image/flat/floor' + str(levelD[level]['flat'][i[4]]) + '.png'
            app += 1
            wall.append(Wall(list(-a), list(b), list(a + x2 + [0, 0, -2]), [im, im, i[2][2]], 0, 0, 1, 0,0))
        else:
            im = 'image/flat/floor' + str(levelD[level]['flat'][i[4]]) + '.png'
            app += 1
            wall.append(Wall(list(-a), list(b), list(a + x2), [im, im, i[2][2]], 0, 0, 1, 0,0))

    h_wall = []
    [i.texture(2 + int(np.linalg.norm(i.a) / 500), 2 + int(np.linalg.norm(i.b) / 500)) for i in
     wall[-app:]]
    h_wall = wall[-app:]

    cell_size=2
    cell_array_N=np.full((500//cell_size,500//cell_size),0)
    cell_array_z = np.full((500 // cell_size, 500 // cell_size,2), 0)
    cell_array_z[:,:,0]=50
    cell_array_z[:, :, 1] = -50
    cell_array = create_cell_array(cell_size)
    fig,ax=plt.subplots(1,4)
    for cw,i in enumerate(wall):
        if i not in h_wall:
            cells=cells_crossed_by_segment(0.5*(i.X[0,0,:-1]+100),0.5*i.b[0,0,:-1],cell_size)
            for u in cells:
                cell_array_N[int(u[0])][int(u[1])]+=1
                cell_array[int(u[0])][int(u[1])].append(cw)

            ax[0].scatter([i[0] for i in cells],[i[1] for i in cells])
            count_w=0
            h_wall_temp=[]
            for j in h_wall:
                V1=i.X[0,0,:-1]-j.X[0,0,:-1]
                V2 = i.X[0, 0, :-1]+i.a[0, 0, :-1]+i.b[0, 0, :-1]-j.X[0,0,:-1]
                d1x=np.dot(V1,j.a[0,0,:-1])
                d1y=np.dot(V1,j.b[0,0,:-1])
                d2x=np.dot(V2,j.a[0,0,:-1])
                d2y=np.dot(V2,j.b[0,0,:-1])


                if np.linalg.norm(j.a[0,0,:-1])**2>=d1x>=0 and np.linalg.norm(j.b[0,0,:-1])**2>=d1y>=0 and np.linalg.norm(j.a[0,0,:-1])**2>=d2x>=0 and np.linalg.norm(j.b[0,0,:-1])**2>=d2y>=0:

                    count_w+=1
                    i.linked.append(j.ID)

                    if j.text.split('/')[-1][:4]=='roof' and j.a[0,0,2]==0 and j.b[0,0,2]==0:
                        # check when plafond is stairs and choose what to do with door, check if the two roof are one inside the other or not
                        h_wall_temp.append(j)
                        if j.X[0,0,2]<i.X[0,0,2]:# if wall smaller than roof then in inside
                            i.inside=True
                        i.height_rel.append((j.X[0,0,2],j.text,j.ID,i.X[0,0,2],i.text,i.ID,i.deco))
            if len(i.height_rel)>1:# if multiple roofs check if they are included one into the other and with a smaller one
                for k in h_wall_temp:
                    for j in h_wall_temp:
                        if j.X[0, 0, 2] < i.X[0, 0, 2]:
                            if j!=k:# checking k inside j
                                V1 = k.X[0, 0, :-1] - j.X[0, 0, :-1]
                                V2 = k.X[0, 0, :-1] + k.a[0, 0, :-1] + k.b[0, 0, :-1] - j.X[0, 0, :-1]
                                d1x = np.dot(V1, j.a[0, 0, :-1])
                                d1y = np.dot(V1, j.b[0, 0, :-1])
                                d2x = np.dot(V2, j.a[0, 0, :-1])
                                d2y = np.dot(V2, j.b[0, 0, :-1])

                                if np.linalg.norm(j.a[0, 0, :-1]) ** 2 >= d1x >= 0 and np.linalg.norm(
                                        j.b[0, 0, :-1]) ** 2 >= d1y >= 0 and np.linalg.norm(
                                        j.a[0, 0, :-1]) ** 2 >= d2x >= 0 and np.linalg.norm(
                                        j.b[0, 0, :-1]) ** 2 >= d2y >= 0 :

                                    i.inside=True
                                else:

                                    i.inside=False
        else:
            cells=cells_covered_by_plane(0.5*(i.X[0,0,:-1]+100),0.5*i.a[0,0,:-1],0.5*i.b[0,0,:-1],cell_size)
            for u in cells:
                cell_array_N[int(u[0])][int(u[1])]+=1
                cell_array[int(u[0])][int(u[1])].append(cw)
                if i.X[0,0,-1]<cell_array_z[int(u[0]),int(u[1]),0] and i.X[0,0,-1]>0:
                    cell_array_z[int(u[0]),int(u[1]),0]=i.X[0,0,-1]
                if i.X[0,0,-1]>cell_array_z[int(u[0]),int(u[1]),1] and i.X[0,0,-1]<0:
                    cell_array_z[int(u[0]),int(u[1]),1]=i.X[0,0,-1]

    ax[1].imshow(cell_array_N)
    ax[2].imshow(cell_array_z[:,:,0])
    ax[3].imshow(cell_array_z[:, :, 1])
    plt.show()
    cell_start, cell_count, cell_objects = build_cell_csr(cell_array)
    all_walls=wall.copy()
    all_a=np.array([i.a[0,0,:] for i in all_walls])
    all_b = np.array([i.b[0, 0, :] for i in all_walls])
    all_X = np.array([i.X[0, 0, :] for i in all_walls])
    all_ab=np.array([np.dot(i.a[0, 0, :],i.b[0, 0, :]) for i in all_walls])
    all_aa = np.array([np.dot(i.a[0, 0, :],i.a[0, 0, :]) for i in all_walls])
    all_bb = np.array([np.dot(i.b[0, 0, :], i.b[0, 0, :]) for i in all_walls])
    all_n = np.array([np.cross(i.a[0, 0, :], i.b[0, 0, :]) for i in all_walls])
    all_inv_det = np.array([1.0 / (all_aa[i]*all_bb[i] - all_ab[i]**2) for i in range(len(all_walls))])


    global all_opening,all_freq,all_phase,all_tile_z,all_trans_im,all_format,all_wall_im,all_light,all_light_w
    all_opening=np.array([i.opening for i in all_walls])
    all_freq=np.array([i.freq for i in all_walls])
    all_phase = np.array([i.phase_ for i in all_walls])
    all_tile_z = np.array([i.tile_z for i in all_walls])
    all_format=np.array([i.format for i in all_walls])
    all_light_w=np.array([i.colorL for i in all_walls])

    all_trans_im = List()
    all_wall_im = List()
    all_light = List()
    for element in all_walls:
        inner = List.empty_list(types.boolean[:, :])
        inner_im = List.empty_list(types.float32[:,:,:])
        inner_light= List.empty_list(types.float32[:])

        if isinstance(element.trans_im, list) and len(element.trans_im) > 0:
            for arr in element.trans_im:
                inner.append(arr.astype(np.bool_))

        if isinstance(element.wall_im, list) and len(element.wall_im) > 0:
            for arr in element.wall_im:
                inner_im.append(arr.astype(np.float32))

        if element.ID in light_wall.keys():
            for light_x in light_wall[element.ID]:
                inner_light.append(source_pos(light_x).astype(np.float32))

        # If element was 0 or empty → just append empty inner list
        all_trans_im.append(inner)
        all_wall_im.append(inner_im)
        all_light.append(inner_light)



    height_list = [i.X[0][0][2] for i in wall if i.inside ]
    height_list=list(set(height_list))
    height_list.sort()

    horizon2 = np.full((scrnL[0], len(height_list)), 10000.0)

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
                difficulty_var[2]+=10*DEGAT[objet.color]/(COOLDOWN[objet.color]+1)
            if thing[-1].type_M == 1:
                difficulty_var[3] +=20

    global all_things,all_x_e,all_RA,all_im_m,all_im_o,all_obj_mon,all_types_e,all_angle,all_ima_m,all_mort,all_attack_range,all_range,all_light_e
    all_things = thing.copy()

    all_x_e=np.array([np.concatenate((i.x0,np.array([2*i.z]))) for i in all_things])

    all_RA = np.array([i.RA for i in all_things])

    types_monst=list(set(levelD[level]['mon']))
    all_im_m=np.full((len(types_monst),4,8,160,160,3),0.)
    all_ima_m = np.full((len(types_monst), 8, 160, 160, 3), 0.)
    for c_0,t_2 in enumerate(types_monst):
        for t_1 in range(8):
            for t_0 in range(4):
                all_im_m[c_0,t_0,t_1,:,:,:]=np.minimum(pygame.surfarray.pixels3d(MD[t_2][t_0][45*t_1]),255)
            all_ima_m[c_0, t_1, :, :, :] = np.minimum(pygame.surfarray.pixels3d(MDa[t_2][t_1]), 255)


    types_obj=list(set(levelD[level]['obj']))
    all_im_o=np.full((len(types_obj),8,160,160,3),0.)
    print(Mod)
    for c_0,t_2 in enumerate(types_obj):
        for t_1 in range(8):
             all_im_o[c_0,t_1,:,:,:]=np.minimum(pygame.surfarray.pixels3d(MO[t_2][45*t_1]),255)

    all_types_e = np.array([types_monst.index(i.type_M)  if i.thing_t==1 else types_obj.index(i.type_M)  for i in all_things])
    all_mort = np.array(
        [i.mort if i.thing_t == 1 else 0 for i in all_things])
    all_obj_mon = np.array([i.thing_t for i in all_things])
    all_angle = np.array([i.angle for i in all_things])
    all_attack_range=np.array(
        [i.attack_range if i.thing_t == 1 else 0 for i in all_things])
    all_range=np.array(
        [i.range if i.thing_t == 1 else 0 for i in all_things])

    all_light_e = np.array([i.light for i in all_things])

    if 0 in groupD:
        groupD.remove(0)

MG3=[]
for c,i in enumerate(MG1):
    temp_d={}
    for k in i.keys():
        arr=pygame.surfarray.pixels3d(i[k])
        arr=arr.astype(float)
        arr[np.any(arr!=0,axis=-1)]=np.minimum(25+arr[np.any(arr!=0,axis=-1)]*(230/255),255)
        arr=arr.astype(int)
        temp_d[k]=pygame.surfarray.make_surface(arr)
    MG3.append(temp_d.copy())
MG1=MG3

SKY0 = pygame.surfarray.pixels3d(pygame.image.load('image/ciel/ciel0.png'))
LAND0 = pygame.surfarray.pixels3d(pygame.image.load('image/ciel/lanscape0.png'))
LAND0 = np.where(np.expand_dims(((LAND0 - np.array([0, 255, 255])) == 0).all(-1), -1), -2, LAND0)


SKY0_im=pygame.image.load('image/ciel/ciel0.png')
LAND0_im = pygame.image.load('image/ciel/lanscape0.png')
LAND0_im.set_colorkey((0,255,255))

BLOOD = pygame.transform.scale(pygame.image.load('image/effects/blood.png'), window)
code01 = pygame.image.load('image/Interface/code.png')
code02 = pygame.image.load('image/Interface/code2.png')
incinerate = pygame.image.load('image/Interface/incinerate.png')
incinerate2 = pygame.image.load('image/Interface/incinerate2.png')
shield=pygame.image.load('image/effects/shield.png').convert_alpha()
shield2=pygame.image.load('image/effects/shield2.png').convert_alpha()
ttt = []
bleed = 0
explo = 0
explo_pt = np.array([0., 0.])
Sky_view = 0
ang = (1e-5, 0)
explo_cool=0
draw_hud()
L0 = []
load_level(str(level))
pygame.mixer.music.play(-1)

depth = np.full((2 * scrnL[0], 2 * scrnL[1], 1), 100.0)
wall_index = np.full((2 * scrnL[0], 2 * scrnL[1]), 0)
explo_R=np.full((2 * scrnL[0], 2 * scrnL[1], 1), 100.0)
horizon = np.full((scrnL[0], 1), 10000.0)

horizon2 = np.full((scrnL[0], len(height_list)), 10000.0)
POS = np.full((2 * scrnL[0], 2 * scrnL[1]), 1000.0)
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
averaged_time = np.full((23), 0.)
elastic_count=0
Ratio=window[0]/960
x_d=[(0.,0.)]
nb_wall = []
time_wall = []
time_tot=[]
time_behind = []
plot_stats=True
sensitivity=500#500
movement=0
render_w_old=0




while running == 1:

    moving_cam = True
    milliseconds = [time.perf_counter()*1000]
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
    wall_index*=0
    horizon = horizon * 0 + 10000.
    horizon2 = horizon2 * 0 + 10000.
    POS = POS * 0 + 1000.
    explo_R=explo_R*0+1000
    trans = np.array([0.0, 0.0])
    HIT = 0
    rset_L = 0
    recoil = 0.
    milliseconds.append(time.perf_counter()*1000)
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
    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('lifts')
    wall.sort(key=lambda s: s.norm)
    doors.sort(key=lambda s: s.norm)
    h_wall.sort(key=lambda s: s.norm)
    thing.sort(key=lambda s: s.norm)
    ennemies.sort(key=lambda s: s.norm)
    if level==6:
        op.sort(key=lambda s: -s.norm)
    milliseconds.append(time.perf_counter()*1000)
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
                    colorGUN = (255., 255., 255.)
            if event.button == 5:
                if mouse_c == 0:
                    Sline = min(Sline + 1, 0)
                    show_message()
                    pygame.display.flip()
                else:
                    arme = (arme - 1) % TotAr
                    GUN_im = MG1[arme][0].copy()
                    colorGUN = (255., 255., 255.)
    clic = pygame.mouse.get_pressed()
    mouse = pygame.mouse.get_pos()
    if not ((mouse == m_stock).all()):
        moving_cam = True

    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('events')

    if key[K_KP_PLUS] == 1:
        window = [int(window[0] * 1.05), int(window[1] * 1.05)]
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
        window = [int(window[0] * 0.95), int(window[1] * 0.95)]
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

    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('clicks_and_keys')
    if attack == 0:
        GUN_im = MG1[arme][0].copy()
        colorGUN = (255., 255., 255.)
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
                draw_AMMO()

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
    No=np.array([0.,0])
    previous=0.5

    if trans.any() != np.array([0.0, 0.0]).any():
        No = np.array([0., 0.])
        for i in wall[0:20]:
            if i not in h_wall:

                if i.norm3 < 3:

                    if (i.door and i.closed) or not i.door:
                        #trans = i.normal(trans @ Rp) @ rot_plan(-ang[0])
                        if i.inter[1]<1 and i.inter[1]>0 or (previous in [0,1.] and i.inter[1]in [0,1.]):
                            if not(i.door_deco) or (i.door_deco and i.cross_wall(trans)):
                                No+=i.normal()
                        previous=i.inter[1]



        trans=trans-(abs(np.dot(trans@ Rp, No)) * No)@ rot_plan(-ang[0])

        x = x - trans @ Rp

        # if authorized_map[int(x[1] + 100) // 2][int(x[0] + 100) // 2]==1:
        #     last_ok_pos = x
        # else:
        #     x=last_ok_pos
        #     trans=trans*0.

        z = zmap[int(x[1] + 100) // 2][int(x[0] + 100) // 2]
        screen[:, :, :3] = screen[:, :, :3] - np.hstack((trans @ Rp, [0]))
        screen[:, :, 2] = z * 2
        zprev = z
        R_c = np.hstack((x, [2 * z]))

    if mouse_c == 1:
        rot_a = -(2 * pi * (mouse - m_stock) / sensitivity) * [1, -0.2]  # 1000
        if abs((ang + rot_a)[1]) < pi / 8:
            ang = ang + rot_a
        screenV = screen[:, :, 3:] @ rot_y(ang[1]) @ rot_z(ang[0])
        screenP = (screen[:, :, :3] - R_c) @ rot_y(ang[1]) @ rot_z(ang[0]) + R_c

        pygame.mouse.set_pos([window[0] // 2, window[1] // 2])
    m_stock = np.array([window[0] // 2, window[1] // 2])
    Rp = rot_plan(ang[0])

    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('gun_clicks_and_keys')



    [i.calc_norm() for i in wall[0:20]]
    [i.reset_rend() for i in h_wall[0:20]]
    if c2 == 0:
        [i.calc_norm() for i in wall]
    [i.calc_norm() for i in thing[0:20] if i.type_M != 'BOSS']
    if c2 == 23:
        [i.calc_norm() for i in thing if i.type_M != 'BOSS']

    milliseconds.append(time.perf_counter()*1000)
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
            all_X[i.num]=i.X[0,0,:]
            i.X_sliced = i.X[::2, ::2]
            if i.closed:
                porteson.play()
            i.closed = 0
        else:
            if (Ri)[0][0][-1] < -5. + shift:
                i.X = np.minimum(Ri + [0., 0., 3], [10000, 10000, -5 + shift])
                all_X[i.num] = i.X[0, 0, :]
                i.X_sliced = i.X[::2, ::2]
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
                    i.X_sliced = i.X[::2, ::2]
                    all_X[i.num] = i.X[0, 0, :]
                    # if i.closed:
                    # porteson.play()
                    i.closed = 0

        CLOSED += 1 - i.closed

    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('doors')
    wall_count=min(40+elastic_count,70)
    if explo!=0:
        moving_cam=True

    time_in_behind=0.
    add_h=[]
    link_h=[]
    ID0=''
    cw=0
    max_depth=1000
    wall_rend=[]
    render_type = 'ray'
    IS = []
    if render_type!='ray':
        if moving_cam == True:

            v_ = screenV[::2, ::2]
            p_ = screenP[::2, ::2]
            Sky_view = 0

            empty_pixel_count=128000
            for ci,i in enumerate(wall[0:wall_count]):

                cw+=1
                devant = True
                if i not in h_wall:
                    devant = i.test_behind()

                    time_in_behind+=i.time_behind
                    if i.window > 0 and devant:
                        CLOSED = 1
                        if i.sky>0:
                            Sky_view = 1
                else:

                    if i.norm > 6 and i.text[11:-3] not in liquid_floor:
                        # if len(add_h)<6:# for more floor maybe 4 if previous empty_pixel is big
                        #     add_h.append(i)
                        if i.ID in link_h:
                            add_h.append(i)


                        devant = False
                        # if CLOSED != 0:  # and h_wall.index(i)<=6: # INSTEAD CHECK IF ASSOCIATED DOOR WITH THIS FLOOR IS OPEN AND VISIBLE---COMPLICATED
                        #     devant = True
                    else:
                        ID0=i.ID
                render_w2 += 1

                if devant:
                    if i.door==0:

                        link_h+=i.linked


                    if i.text[11:-3] not in liquid_floor:

                        render_w += 1
                        rend=i.render()
                        wall_rend.append(i)
                        # Im[i.Ub]=rend[i.Ub]
                        if np.min(depth)!=100:
                            max_depth=np.max(depth[depth!=100])
                        # Im = i.render() + Im * (1 - np.expand_dims(i.U, -1))

                        #empty_pixel_count = np.sum((np.sum(Im[3:-3:3, 3:-3:3], axis=-1) == 0).astype(int))

                        empty_pixel_count = np.sum((depth == 100).astype(int))


                        if render_w==1 :#and c3==1:

                            time_in_render=i.time[0]
                            label_t_render=i.time[1]
                        else:
                            time_in_render+=i.time[0]
                            label_t_render = i.time[1]


                    else:
                        IS.append(i)


                if (empty_pixel_count < 4 ) or i.norm>150 or ci==wall_count-1:

                    render_w_add=0

                    add_h+=[k for k in h_wall if k.ID in link_h and k.rendered==False]

                    if empty_pixel_count>4 :
                        for j in add_h:
                            if j.rendered==False and j.text[11:-3] not in liquid_floor:
                                rend = j.render()
                                wall_rend.append(j)
                                # Im[j.Ub] = rend[j.Ub]
                                render_w_add+=1
                                time_in_render += j.time[0]
                                label_t_render = j.time[1]
                    #empty_pixel_count = np.sum((np.sum(Im[3:-3:3, 3:-3:3], axis=-1) == 0).astype(int))
                    empty_pixel_count = np.sum((depth == 100).astype(int))
                    break

            if render_w2>wall_count-10:
                elastic_count += 10
                if render_w_add>0 and empty_pixel_count<4:
                    elastic_count -= 10
            else:
                elastic_count = max(elastic_count-10,0)
        render_w_old=render_w

        if moving_cam == True:
            render_w_add2=0
            if empty_pixel_count<4 and (horizon!=10000).any():
                [i.calc_normfast() for i in add_h if i.rendered == False]
                add_h2=[i for i in add_h if i.normf<max(horizon[horizon!=10000.]) and i.rendered==False and i.text[11:-3] not in liquid_floor and i in h_wall]
                for j in add_h2:

                    rend = j.render()
                    wall_rend.append(j)
                    # Im[j.Ub] = rend[j.Ub]
                    render_w_add2 += 1
                        #empty_pixel_count = np.sum((np.sum(Im[3:-3:3, 3:-3:3], axis=-1) == 0).astype(int))

            render_sup_wall=0
            for ci,i in enumerate( wall[cw:cw+10]):
                if i.rendered == False and i.test_behind() and i not in h_wall:
                    render_sup_wall+=1
                    rend = i.render()
                    wall_rend.append(i)
                    # Im[i.Ub] = rend[i.Ub]



        #print(render_w,render_w_add,render_w_add2,render_sup_wall,empty_pixel_count)
        render_w=render_w+render_w_add+render_w_add2+render_sup_wall



    milliseconds.append(time.perf_counter()*1000)

    label_deltat.append('walls')

    S_i,wall_ind_i,Xl,Im_ray,POS_l=intersect(screenV,screenP,cell_start, cell_count, cell_objects,cell_size,all_a,all_b,all_X,all_aa,all_bb,all_n,all_ab,all_inv_det,all_opening,all_freq,all_phase,all_tile_z,all_trans_im,all_format,all_wall_im,all_light,all_light_w)
    if key[K_u]:
        plt.imshow(Im_ray/255)
        plt.show()

    milliseconds.append(time.perf_counter() * 1000)

    label_deltat.append('intersect')
    depth=S_i[:,:,-1,None]
    if render_type=='ray':

        uniq, wall_index = np.unique(wall_ind_i, return_inverse=True)
        # wall_index = wall_index.reshape(wall_ind_i.shape)
        #
        wall_rend = np.array(all_walls)[uniq]
        render_w = uniq.size

        # ind_l = [
        #     c // (12 // len(i.wall_im)) if (i.side < 0 and levelD[level]['deco'][i.deco - 1] not in deco_destruc)
        #     else c // (12 // len(i.wall_im2)) if (
        #                 i.side > 0 and levelD[level]['deco'][i.deco - 1] not in deco_destruc)
        #     else min(i.vie, 2) if (levelD[level]['deco'][i.deco - 1] in deco_destruc)
        #     else 0
        #     for i in wall_rend
        # ]
        #
        # wall_im_g = np.concatenate(
        #     [i.wall_im[ind_l[c0]] if i.side < 0 else i.wall_im2[ind_l[c0]] for c0, i in enumerate(wall_rend)],
        #     axis=0)
        # freq_g = np.array([i.freq for i in wall_rend])
        # phase_g = np.array([i.phase_ - i.freq + 1 for i in wall_rend])
        # tile_z_g = np.array([i.tile_z for i in wall_rend])
        # light_g = np.array([light_modif(i.colorL, level, c3) for i in wall_rend])
        #
        # format_g = np.stack([i.format for i in wall_rend], axis=0)
        # i_, j_ = np.indices(wall_index.shape)
        # S_g_r = S_i
        #
        #
        # u = ((1 - S_g_r[:, :, :-1]) * format_g[wall_index, :]).astype(int)
        # G_g = np.mod(np.maximum(u, 0), 120)

        # texture = wall_im_g[120 * wall_index + G_g[:, :, 0], G_g[:, :, 1] + 120 * (
        #             (((-u[:, :, 1] // 120 + phase_g[wall_index]) % freq_g[wall_index])) == 0) * (
        #                                 u[:, :, 0] < 120 + 1000 * tile_z_g[wall_index])] * light_g[wall_index]
        texture=Im_ray
        Im = texture
        # Xsource_g = np.empty((4, len(wall_rend), 3))
        # torch_shine = False
        # for cg, i in enumerate(wall_rend):
        #     if i.ID in light_wall.keys():
        #         Y0 = [np.linalg.norm(source_pos(j) - R_c) for j in light_wall[i.ID]]
        #         X0 = [x for _, x in sorted(zip(Y0, light_wall[i.ID]))]
        #         Xsource_g[:, cg, :] = np.array(
        #             [source_pos(X0[k]) if k < len(X0) else np.array([1e9, 0., 0.]) for k in range(4)])
        #     else:
        #         if uniq[cg]!=0:
        #             torch_shine = True
        #
        # a_g = np.array([i.a[0, 0, :] for i in wall_rend])
        # b_g = np.array([i.b[0, 0, :] for i in wall_rend])
        # x_g = np.array([i.X[0, 0, :] for i in wall_rend])
        #
        # # Xl = S_g_r[:, :, 0, None] * a_g[wall_index, :] + S_g_r[:, :, 1, None] * b_g[wall_index, :] + x_g[wall_index,
        # #                                                                                              :]
        # if torch_shine:
        #     Im = Im * torch_on * TORCHE ** 3
        #     POS = np.linalg.norm(Xl - R_c, axis=-1) ** 0.5
        # else:
        #     POS = np.amin(np.linalg.norm(Xl[:, :, :] - Xsource_g[:, wall_index, :], axis=-1), axis=0)
        #
        # if explo != 0:
        #     explo_R = np.minimum(explo_R,
        #                          np.linalg.norm(Xl - np.array([explo_pt[0], explo_pt[1], 0.]), axis=-1)[:, :, None])

        POS=POS_l



    if moving_cam == True:
        Im_cached = Im
        depth_cached = depth
        POS_cached = POS
        horizon_cached = horizon
        horizon_cached2 = horizon2
    else:
        Im = Im_cached
        depth = depth_cached
        POS = POS_cached
        horizon = horizon_cached
        horizon2 = horizon_cached2
    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('glob_walls')


    # #NUMBA ATTEMPT
    # wall_a=[]
    # wall_b = []
    # wall_X = []
    # for i in wall[0:wall_count]:
    #     wall_a.append(i.a[0,0,:])
    #     wall_b.append(i.b[0, 0, :])
    #     wall_X.append(i.X[0, 0, :])
    #
    # # xx=render_ray_step_sign(scrnL[0],scrnL[1],10,np.array(wall_a),np.array(wall_b),np.array(wall_X),screenV0, screenP0, R_c0)
    # milliseconds.append(time.perf_counter()*1000)
    # label_deltat.append('numba attempt ray tracing deactivated')
    # xx=render_numba(scrnL[0], scrnL[1], 10, np.array(wall_a), np.array(wall_b), np.array(wall_X), screenV,screenP, R_c)
    # # plt.imshow(xx[:,:,-1])
    # # plt.show()
    # milliseconds.append(time.perf_counter()*1000)
    # label_deltat.append('numba attempt solving')
    # #NUMBA ATTEMPT



    if level in [2]:
        if randint(0, 100) == 0:
            level_light = np.array([1, 1, 1.]) * random()
        else:
            level_light = np.minimum(level_light + 0.1 * np.array([1, 1, 1.]), np.array([1, 1, 1.]))

    old_render_sky=False
    # Sky_view=True
    if Sky_view and old_render_sky:
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

    milliseconds.append(time.perf_counter()*1000)
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
        Im = np.where((np.expand_dims(S[:, :, 2],-1)>0)&(np.expand_dims(S[:, :, 2],-1)<depth)& (D_e<(4*explo)+np.random.randint(-1,1,explo_R.shape)), (0.5*255+Im *0.5)* explo_zone(4*explo,D_e), Im)

        # if explo==4:
        #     plt.imshow(Im/255)
        #     plt.show()



    horizon = horizon * np.expand_dims(np.linalg.norm(screen[:, 20, 3:], axis=-1), -1)

    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('lights')

    # ttt0=time.time()
    [i.move() for i in ennemies[0:10]]

    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('movethings')

    if c2 == 23:
        update_map(level_map, x)
    if c % 3 == 0:
        [i.walk() for i in ennemies[0:20]]
    milliseconds.append(time.perf_counter()*1000)
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

    milliseconds.append(time.perf_counter()*1000)
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
    render_w_e=0
    render_w_o = 0
    time_in_render_e=np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
    label_t_render_e=['none']
    time_in_render_o=np.array([0.])
    label_t_render_o=['none']
    incinerate_show = False
    explo_cool = max(0,explo_cool-1)
    for i in thing:
        if i.type_M != 'BOSS':
            if i.norm > np.percentile(depth, 99):
                break
            if i.test_behind():
                render_C += 1
                rend_=i.render()
                # mask=i.Ut.astype(bool)
                # Im[mask] = rend_[mask]
                #Im =  rend_* np.expand_dims(i.Ut, -1) + Im * (1 - np.expand_dims(i.Ut, -1))

                # depth[mask] = i.norm
                if i in ennemies :
                    render_w_e += 1
                    if render_w_e == 1 :  # and c3==1:
                        time_in_render_e = i.time[0]
                        label_t_render_e = i.time[1]
                    else:
                        time_in_render_e += i.time[0]
                        label_t_render_e = i.time[1]
                else:
                    if i.type_M == 28 and i.norm<10:
                        incinerate_show=True

                        if key[K_RETURN] and explo_cool==0:
                            s = pygame.mixer.Sound("son/barril.ogg")
                            s.play()
                            explo_pt=i.x0+np.array([np.cos(-i.orient+pi/2),np.sin(-i.orient+pi/2)])*20
                            explo=5
                            explo_type=2
                            explo_cool=100
                    render_w_o+=1
                    if render_w_o == 1 :  # and c3==1:
                        time_in_render_o = i.time[0]
                        label_t_render_o = i.time[1]
                    else:
                        time_in_render_o += i.time[0]
                        label_t_render_o = i.time[1]
    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('things')

    Im,index_e = thing_render(c,c2,ang[0], ang[1], R_c, all_x_e, Im, S_i,all_RA,all_im_m,all_im_o,all_obj_mon,all_types_e,all_angle,all_ima_m,all_mort,all_attack_range,all_range,all_light_e)

    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('things_parallel')

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

    # IS_rendered=[]
    # for i in IS:
    #     IS_rendered.append(i.render())
    #     #Im = IS_rendered[-1] * 0.5 + Im * (1 - 0.5 * np.expand_dims(i.U, -1))
    #     Im[i.Ub] = IS_rendered[-1][i.Ub]*0.5+Im[i.Ub]*0.5
    if fire:
        Im = np.minimum(Im + 100 * TORCHE, 255)


    Im=np.maximum(Im,0)

    fond = pygame.Surface((2*160-6,2*80-6))#to improve
    Sky_view=True
    if Sky_view:
        if level==6:
            movement+=-sin(ang[0])/ 500
        else:
            movement=0

        fond.blit(SKY0_im,(int(-((-ang[0])%(2*pi)) * 6 * scrnL[0] / (2 * pi)),-int(tan(ang[1]) * scrnL[1] / TAN1 + scrnL[1])))
        fond.blit(SKY0_im, (
        int(-((-ang[0])%(2*pi)) * 6 * scrnL[0] / (2 * pi))+SKY0_im.get_width(), -int(tan(ang[1] ) * scrnL[1] / TAN1 + scrnL[1])))
        fond.blit(LAND0_im,(int(-((-ang[0]+movement)%(2*pi)) * 12 * scrnL[0] / (2 * pi)),-int(tan(ang[1]) * scrnL[1] / TAN1 + scrnL[1])))
        fond.blit(LAND0_im, (
        int(-((-ang[0]+movement)%(2*pi)) * 12 * scrnL[0] / (2 * pi))+LAND0_im.get_width(), -int(tan(ang[1] ) * scrnL[1] / TAN1 + scrnL[1])))

        if level==6:
            [i.calc_norm() for i in op]
            [i.affiche() for i in op]
    else:
        movement=0

    fond0 = pygame.surfarray.make_surface(Im[3:-3, 3:-3])
    fond0.set_colorkey((0, 255, 255))
    fond.blit(fond0,(0,0))
    if setting['smooth'] == True:
        fond = pygame.transform.smoothscale(fond, window)
    else:
        fond = pygame.transform.scale(fond, window)

    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('rendering')

    for i in Boule:
        KILL = i.update()
        if KILL:
            Boule.remove(i)
    [i.affiche() for i in Boule]
    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('boule')


    if shoot==1 and arme!=0 and arme!=4:
        for i in range(len(y_d)):
            depth1=depth[int(depth_[0]*(y_d[i][0]+0.5))][int(depth_[1]*(y_d[i][1]+0.5))]
            impact2=pygame.transform.scale(impact,(int(Ratio*400/depth1),int(Ratio*400/depth1)))
            impact2=pygame.transform.rotate(impact2,360*random())
            fond.blit(impact2,(int(window[0]//2-Ratio*200/depth1+y_d[i][0]*window[0]),int(window[1]//2-Ratio*200/depth1+y_d[i][1]*window[1])))
    GUN_im.set_colorkey((0,0,0))
    fond.blit(GUN_im.convert(), ((int(Ratio*shift_a[arme]+-10*Ratio * (cos(2 * pi * xg / 20)) ** 2), int(10*Ratio * (cos(2 * pi * yg / 20)) ** 2))))

    # fond.blit(GUN_im, ((int(Ratio*shift_a[arme]+-10*Ratio * (cos(2 * pi * xg / 20)) ** 2), int(10*Ratio * (cos(2 * pi * yg / 20)) ** 2))))

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

    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('bleed, explo, effects')

    fenetre.blit(fond, (0, 0))

    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('blit image')
    if incinerate_show:
        if explo_cool==0:
            fenetre.blit(incinerate, (int(0.5 * window[0] - 0.4 * window[1]), int(0.84 * window[1])))
        else:
            fenetre.blit(incinerate2, (int(0.5 * window[0] - 0.4 * window[1]), int(0.84 * window[1])))

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
        v = 1.2 * min(max(24 * (-milliseconds[0] + time.perf_counter()*1000) / 1000, 1),1.5)

    milliseconds.append(time.perf_counter()*1000)
    label_deltat.append('end')

    if c3 % 500 == 2:
        milliT = np.expand_dims(milliseconds, -1)
    else:
        if c3!=1:
            milliT = np.concatenate((milliT, np.expand_dims(milliseconds, -1)), axis=1)

    if c3 == 1:
        milliseconds=np.array(milliseconds)*0
    averaged_time = averaged_time + np.round((np.array(milliseconds) - np.roll(np.array(milliseconds), 1))[1:], 1)

    nb_wall.append(render_w)
    # time_wall.append(np.sum(time_in_render))
    time_behind.append(np.sum(time_in_behind))
    time_tot.append(milliseconds[-1]-milliseconds[0])


    if (c3-1) % 500 == 499 :

        averaged_time = np.round(averaged_time / 500, 1)
        milliseconds = np.mean(milliT, axis=-1)
        timelist = averaged_time#np.round((np.array(milliseconds) - np.roll(np.array(milliseconds), 1))[1:], 1)
        sortingtime = [(x, str(y) + ' ms', str(round(100 * y / np.sum(timelist), 1)) + ' %') for y, x in
                       sorted(zip(timelist, label_deltat))]
        sortingtime.reverse()
        print('Top contribution', sortingtime[:3])
        print('average,framerate', round(1000 / np.sum(averaged_time), 1), '/s | average time',
              round(np.sum(averaged_time), 1), 'ms')
        print(sortingtime)
        # print('detail on wall rendering',np.sum(time_in_render),time_in_render/render_w, label_t_render)
        # print('per_wall',np.sum(time_in_render)/render_w,render_w)
        render_w_e=max(1,render_w_e)
        print('detail on ennemies rendering', np.sum(time_in_render_e),[str(round(i,3)) for i in list(time_in_render_e / render_w_e)] , label_t_render_e)
        print('per_ennemies', np.sum(time_in_render_e) / render_w_e, render_w_e)

        render_w_o = max(1, render_w_o)
        print('detail on object rendering', np.sum(time_in_render_o),
              [str(round(i, 3)) for i in list(time_in_render_o / render_w_o)], label_t_render_o)
        print('per_object', np.sum(time_in_render_o) / render_w_o, render_w_o)
        print('*-----------*')
        ttt = []


        mean_time=np.full(max(nb_wall)+1,0.)
        mean_time_wall = np.full(max(nb_wall) + 1, 0.)
        count=np.full(max(nb_wall)+1,0.)
        for i in range(len(nb_wall)):
            mean_time[nb_wall[i]]+=time_tot[i]
            # mean_time_wall[nb_wall[i]] += time_wall[i]
            count[nb_wall[i]]+=1
        mean_time=np.divide(mean_time,np.maximum(count,1))
        mean_time_wall = np.divide(mean_time_wall, np.maximum(count, 1))
        if  plot_stats :

            fig,ax=plt.subplots(1,2)
            # ax[0].scatter(nb_wall,time_wall,color='blue',label='wall')
            ax[0].scatter(nb_wall, time_behind, color='red', label='behind')
            ax[0].scatter(nb_wall, time_tot, color='orange',label='total')
            ax[0].plot(np.linspace(0,max(nb_wall),max(nb_wall)+1), mean_time, color='maroon', label='total')
            ax[0].plot(np.linspace(0, max(nb_wall), max(nb_wall) + 1), mean_time_wall, color='green', label='wall')
            ax[0].plot(np.linspace(0, max(nb_wall), max(nb_wall) + 1),mean_time-mean_time_wall, color='purple', label='diff')
            ax[0].hlines(1000/24,0,10,color='red',label='24 fps',linestyle='--')
            ax[0].hlines(np.sum(averaged_time), 0, max(nb_wall), color='black', label='mean fps')
            ax[0].set_xlabel('nb_wall')
            ax[0].set_ylabel('time in ms')
            ax[0].legend()
            ax[1].hist(time_tot,bins=100)
            ax[1].set_xlabel('time in ms')


            dataT=milliT - np.roll(milliT, 1,axis=0)
            fig2, ax2 = plt.subplots(5, 5)
            for i in range(5):
                ax2[i][0].set_ylabel('time in ms')
                for j in range(5):
                    if i*5+j<len(label_deltat):
                        ax2[i][j].set_title(label_deltat[i*5+j])
                        ax2[i][j].scatter(nb_wall[1:],dataT[1+i*5+j,:])
                        ax2[3][j].set_xlabel('nb_wall')
                        ax2[i][j].hlines(1000 / 24, 0, max(nb_wall),color='red')

            plt.show()
        nb_wall=[]
        time_wall=[]
        time_behind = []
        time_tot=[]
        averaged_time = np.full((23), 0.)



    clock.tick_busy_loop(24)

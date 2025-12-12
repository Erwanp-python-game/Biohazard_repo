import numpy as np
from math import *
import matplotlib.pyplot as plt
import time
from numba import njit, typed, prange

scrnL = np.array([80, 40])
screen = np.full((*scrnL, 6), 0.0)

depth = np.full((2 * scrnL[0], 2 * scrnL[1], 1), 100.0)
POS = np.full((2 * scrnL[0], 2 * scrnL[1]), 1000.0)

Ang = np.full((*scrnL, 2), 0.0)
I = np.indices(scrnL)
I_n = np.divide(np.moveaxis((np.indices(scrnL)), 0, 2) - 0.5 * scrnL, scrnL[1])
screen[:, :, 1:3] = I_n
Ang = I_n
Ang[:, :, 0] = Ang[:, :, 0] * pi / 4
Ang[:, :, 1] = Ang[:, :, 1] * atan(1 / 2) * 2

screen[:, :, 4] = np.sin(Ang[:, :, 0])
screen[:, :, 5] = np.cos(-Ang[:, :, 1] + pi * 0.5)
screen[:, :, 3] = 1

TAN1 = np.amax(screen[:, :, 5])
TAN2 = np.amax(screen[:, :, 4])

screenV = screen[:, :, 3:]
screenP = screen[:, :, :3]

c=0
R_c = np.array([-1, 0, 0])

# class Wall():
# 	def __init__(self, u, v, w):
# 		self.a = np.full((scrnL[0], scrnL[1], 3), u)
# 		self.b = np.full((scrnL[0], scrnL[1], 3), v)
# 		self.X = np.full((scrnL[0], scrnL[1], 3), w)
#
# 	def render(self):
# 		global depth, POS
# 		self.time = (0, 0)
# 		milliseconds = [time.time() * 1000]
# 		label_m = []
# 		self.M = np.stack((self.a[::2, ::2, :] * 1.01, self.b[::2, ::2, :] * 1.01, -screenV[::2, ::2, :]), axis=-1)
# 		self.B = -self.X[::2, ::2, :] + screenP[::2, ::2, :]
#
# 		# self.S=(np.linalg.solve(self.M.astype(np.float32),self.B.astype(np.float32)))
# 		inv_M = np.linalg.inv(self.M.astype(np.float32))
# 		self.S = np.einsum('...ij,...j->...i', inv_M, self.B.astype(np.float32))
# 		milliseconds.append(time.time() * 1000)
# 		label_m.append('solving')
#
#
# 		for i in range(2):
# 			self.S = self.S.repeat(2, axis=0).repeat(2, axis=1)
# 			self.S[:, :, :-1] = (np.roll(self.S[:, :, :-1], 1, axis=0) + np.roll(self.S[:, :, :-1], 1,axis=1) + np.roll(self.S[:, :, :-1],-1,axis=1) + np.roll(self.S[:, :, :-1], -1, axis=0)) / 4
# 			AbsS = np.absolute(self.S[:, :, -1])
# 			self.S[:, :, -1] = (np.sign(self.S[:, :, -1]) * (np.roll(AbsS, 1, axis=0) + np.roll(AbsS, 1, axis=1) + np.roll(AbsS, -1, axis=1) + np.roll(AbsS,-1,axis=0)) / 4)
# 		milliseconds.append(time.time() * 1000)
# 		label_m.append('upscale')
# 		self.U = np.all(self.S <= depth, axis=-1) & np.all(self.S > 0, axis=-1) & np.all(self.S[:, :, :-1] < 1, axis=-1)
#
#
# 		milliseconds.append(time.time() * 1000)
# 		label_m.append('contour')
#
#
#
# 		Ue = np.expand_dims(self.U, -1)
#
#
# 		self.G = np.mod(
# 			np.maximum(((1 - self.S[:, :, :-1]) * self.format).astype(int), 0) + int(self.phase * c3 * 0.5),
# 			self.borne)
#
# 		milliseconds.append(time.time() * 1000)
# 		label_m.append('indexing')
#
# 		colorL = [1, 1, 1]
#
#
# 		x_ = self.X[0][0][0]
# 		y_ = self.X[0][0][1]
# 		a_ = self.b[0][0][0]
# 		b_ = self.b[0][0][1]
# 		side = b_ * R_c[0] - a_ * R_c[1] + a_ * y_ - b_ * x_
#
# 		if side < 0:
# 			ind = c // (12 // len(self.wall_im))
# 			Ar = np.where(np.stack([self.U] * 3, axis=-1),
# 						  np.moveaxis(self.wall_im[ind][tuple(map(tuple, self.G.T))] * colorL, 1, 0), 0)
# 		else:
# 			ind2 = c // (12 // len(self.wall_im2))  # 2
# 			Ar = np.where(np.stack([self.U] * 3, axis=-1),
# 						  np.moveaxis(self.wall_im2[ind2][tuple(map(tuple, self.G.T))] * colorL, 1, 0), 0)
#
# 		milliseconds.append(time.time() * 1000)
# 		label_m.append('render')
#
# 		filt = 1 - np.expand_dims((Ar == 0).all(2), -1)
# 		depth = np.where(Ue * (filt), self.S[:, :, -1:], depth)
# 		self.U = self.U * filt[:, :, 0]
# 		Xl = ((np.expand_dims(self.S[::2, ::2, 0], -1) * self.a * 1.01 + np.expand_dims(self.S[::2, ::2, 1],
# 																						-1) * self.b * 1.01 + self.X).repeat(
# 			2,
# 			axis=0).repeat(
# 			2, axis=1)) * Ue
#
# 		POS = (POS * (1 - self.U)).astype(np.float32)
# 		D = np.full((2 * scrnL[0], 2 * scrnL[1]), 1000.0).astype(np.float32)
#
# 		if self.ID in light_wall.keys():
# 			Y0 = [np.linalg.norm(source_pos(i) - R_c) for i in light_wall[self.ID]]
# 			X0 = [x for _, x in sorted(zip(Y0, light_wall[self.ID]))]
# 			for i in X0[:min(len(X0), 4)]:
# 				Xsource = source_pos(i)
# 				D = np.minimum(D, np.linalg.norm(Xl - Xsource, axis=-1).astype(np.float32) * self.U)
# 			POS = POS + D
# 			Ar = Ar * level_light
#
#
# 		milliseconds.append(time.time() * 1000)
# 		label_m.append('distance and position light')
# 		self.time = ((np.array(milliseconds) - np.roll(np.array(milliseconds), 1))[1:], label_m)
#
# 		return Ar,milliseconds
#
# t0=time.time()
# w=Wall(np.array([0.,1.1,0.]),np.array([1.1,0.,0.]),np.array([0.,0.,0.]))
# im,t=w.render()
# print(time.time()-t0,t,'t')


@njit
def test():
	for i in range(80):
		for j in range(40):
			x=0
test()
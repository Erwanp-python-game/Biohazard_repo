import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

U=[]
V=[]
for i in range(10):
	U.append(np.random.random((3,1600,1600)))
	V.append(np.random.random((3,1600,1)))
start = time.time()
Time=[]
b=start
for i in range(100):
	a=np.linalg.solve(U[i%10],V[i%10])
	Time.append(time.time()-b)
	b=time.time()
	
end = time.time()

print(end - start)


U=[]
V=[]
for i in range(10):
	U.append(torch.rand((3,1600,1600)))
	V.append(torch.rand((3,1600,1)))
torch.cuda.synchronize()
start = time.time()
Time2=[]
b=start
for i in range(100):

	a=torch.linalg.solve(U[i%10],V[i%10])
	Time2.append(time.time()-b)
	b=time.time()
torch.cuda.synchronize()
end = time.time()

print(end - start)


U=[]
V=[]
for i in range(10):
	U.append(torch.rand((3,1600,1600)).to('cuda'))
	V.append(torch.rand((3,1600,1)).to('cuda'))
torch.cuda.synchronize()
start = time.time()
Time3=[]
b=start
for i in range(100):

	a=torch.linalg.solve(U[i%10],V[i%10])
	Time3.append(time.time()-b)
	b=time.time()
torch.cuda.synchronize()
end = time.time()

print(end - start)
plt.plot(Time)
plt.plot(Time2)
plt.plot(Time3)
plt.yscale('log')
plt.show()

import pickle

import matplotlib.pyplot as plt
import numpy as np
from game import Thing,Object,M_charac,COOLDOWN,DEGAT,level_arme
from level_data import levelD
def load_level_diff(level_name):

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

    difficulty_var=[0.,0.,0.,0.]
    M=0
    ammo=[0,0,0]
    soin_=0
    for i in M_liste:
        if i[-2]:
            M+=1
            monst = Thing(2 * (i[0] - 50), int(i[1]), 1, i[-1])
            difficulty_var[0]+=monst.vie
            if levelD[level]['mon'][int(i[1])]==4:
                difficulty_var[1] += 2*M_charac['degat'][0] / 12
                #M+=2
                difficulty_var[0]+=2*M_charac['life'][0]

            if monst.attack_range:
                difficulty_var[1]+=M_charac['degat'][monst.type_M]/24
            else:
                difficulty_var[1] += M_charac['degat'][monst.type_M]/12
        else:
            objet = Object(2 * (i[0] - 50), int(i[1]), 1, i[-1])
            t=levelD[level]['obj'][int(i[1])]
            if t == 2 or t == 5:
                objet.color = (i[-4]) + 1
            if t==2:
                difficulty_var[2]+=10*DEGAT[objet.color]/COOLDOWN[objet.color]
                ammo[objet.color]+=1
            if t == 1:
                difficulty_var[3] +=20
                soin_+=1

    print(difficulty_var,difficulty_var[1]*difficulty_var[0],'degat x (pt_vie / frame)',difficulty_var[2]*difficulty_var[3],'degat x (pt_vie / frame)')
    return difficulty_var,M,ammo,soin_
print('--------------------')
fig,ax=plt.subplots(2,3)
plt.tight_layout()
ax2 = ax[1][2].twinx()
ax3 = ax[1][1].twinx()
diff=[]
deg_p=[]
deg_m=[]
life_p=[]
life_m=[]

level_nb=4

for i in range(level_nb):
    level=i+1
    TotAr=level_arme[level]
    AMMO = [0, 0, 0, 0]
    AMMO[0] = max(min(TotAr - 1, 1) * 20, 0)
    AMMO[1] = max(min(TotAr - 2, 1) * 20, 0)
    AMMO[2] = max(min(TotAr - 3, 1) * 20, 0)
    AMMO[3] = max(min(TotAr - 4, 1) * 20, 0)
    difficulty_var,M,ammo,soin_=load_level_diff(str(1+i))
    print(ammo,'ammo',M,'M',soin_,'soin')
    if i==2:
        difficulty_var[0]+=700
        difficulty_var[1]+=(25/12+15*3/24)*0.5
        M+=1
    for j in range(4):
        difficulty_var[2]+=AMMO[j]*DEGAT[j+1]/(COOLDOWN[j+1]+1)

    ax[0][0].scatter((difficulty_var[1]*difficulty_var[0]),difficulty_var[2]*difficulty_var[3],color=(i/(level_nb+1),1-i/(level_nb+1),0.))

    ax[0][1].scatter(level, M, color=(i / (level_nb+1), 1 - i / (level_nb+1), 0.))
    ax[1][0].scatter(1+i,(difficulty_var[1]*difficulty_var[0])/(difficulty_var[2]*difficulty_var[3]),color=(i/(level_nb+1),1-i/(level_nb+1),0.))
    diff.append((difficulty_var[1]*difficulty_var[0])/(difficulty_var[2]*difficulty_var[3]))

    ax[1][1].scatter(1 + i+0.1, difficulty_var[0]/M,color='red')
    life_m.append(difficulty_var[0]/M)
    ax3.scatter(1 + i, difficulty_var[3]/M, color='blue')
    life_p.append(difficulty_var[3]/M)
    ax3.tick_params(axis='y', labelcolor='blue')

    ax[1][2].scatter(1 + i+0.1, difficulty_var[1]/M, color='red')
    deg_m.append(difficulty_var[1]/M)
    ax2.scatter(1 + i, difficulty_var[2]/M, color='blue')
    deg_p.append(difficulty_var[2]/M)
    ax2.tick_params(axis='y', labelcolor='blue')

ax[1][1].tick_params(axis='y', labelcolor='red')
ax[1][2].tick_params(axis='y', labelcolor='red')

ax[0][0].plot(np.linspace(0,100000,2),np.linspace(0,100000,2))
ax[0][0].set_xlabel('monster_power')
ax[0][0].set_ylabel('player_power')
ax[0][1].set_xlabel('level')
ax[0][1].set_ylabel('number_monster')
ax[0][2].set_xlabel('monster_lifept')
ax[0][2].set_ylabel('player_lifept')
ax[1][0].set_xlabel('level')
ax[1][0].set_ylabel('difficulty (monst/player)')
ax[1][0].plot(np.linspace(1,level_nb,level_nb),diff,color='black',linestyle='--')
ax[1][1].set_xlabel('level')
ax[1][1].plot(np.linspace(1,level_nb,level_nb)+0.1,life_m,color='red',linestyle='--')
ax3.plot(np.linspace(1,level_nb,level_nb),life_p,color='blue',linestyle='--')

ax[1][1].set_ylabel('monst_life/monst')
ax[1][1].yaxis.label.set_color('red')
ax3.set_ylabel('player_life/monst')
ax3.yaxis.label.set_color('blue')
ax[1][2].yaxis.label.set_color('red')
ax[1][2].set_ylabel('monst_degat/monst')
ax2.set_ylabel('player_deg/monst')
ax2.yaxis.label.set_color('blue')

ax[1][2].plot(np.linspace(1,level_nb,level_nb)+0.1,deg_m,color='red',linestyle='--')
ax2.plot(np.linspace(1,level_nb,level_nb),deg_p,color='blue',linestyle='--')
plt.show()
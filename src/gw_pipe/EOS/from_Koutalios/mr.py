import sys
from os.path import dirname
sys.path.append("/home/gvretinaris/Desktop/post_merger_bilby")

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline as spline
from post_merger_bilby import NR_strains as nr
from post_merger_bilby import global_vars as glb
from post_merger_bilby import config
rcparams = {}
rcparams["axes.linewidth"] = 0.5
rcparams["font.family"] = "serif"
rcparams["font.size"] = 12
# rcparams['legend.fontsize'] = 12
rcparams["mathtext.fontset"] = "stix"
rcparams["text.usetex"] = True
rcparams["figure.dpi"] = 300
rcparams["figure.figsize"] = (
    1920 / rcparams["figure.dpi"],
    1440 / rcparams["figure.dpi"],
)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
# rcparams['lines.linewidth'] = 1

plt.rcParams.update(rcparams)

c=2.9979e10
G=6.67408e-8
Msun=1.989e33
Length = G*Msun/c**2
Time = Length/c
Density = Msun/Length**3

names = ['MPA1','H4','SLY','LS220','MS1B','BHBLP','DD2','2H']

eos = os.listdir()
eos = [i for i in eos if 'K' != i[0] and 'mr' != i[:2]]

for file in eos:
    mr = np.load(file)
    if file[:-4] in names:
        plt.plot(mr[1]*Length/1e5,mr[0],label=file[:-4])
        #mx=np.amax(mr[0])
        #idx=np.where(mr[0]==mx)
        #idx=idx[0][0]
        #if file[:-4] == 'SLY':
            #cs=spline(mr[0][1:idx],mr[1][1:idx])
            #R = cs(1.35)*Length/1.0e5
            #plt.plot(R,1.35,'ko')
        #elif file[:-4] == 'LS220':
            #cs=spline(mr[0][1:idx],mr[1][1:idx])
            #R = cs(1.35)*Length/1.0e5
            #plt.plot(R,1.35,'ko')
        #elif file[:-4] == 'MS1B':
            #cs=spline(mr[0][1:idx],mr[1][1:idx])
            #R = cs(1.35)*Length/1.0e5
            #plt.plot(R,1.35,'ko')
            #R = cs(1.5)*Length/1.0e5
            #plt.plot(R,1.5,'ko')
            #R = cs(1.375)*Length/1.0e5
            #plt.plot(R,1.375,'ko')
        #elif file[:-4] == 'BHBLP':
            #cs=spline(mr[0][1:idx],mr[1][1:idx])
            #R = cs(1.3)*Length/1.0e5
            #plt.plot(R,1.3,'ko')
        #elif file[:-4] == 'DD2':
            #cs=spline(mr[0][1:idx],mr[1][1:idx])
            #R = cs(1.2)*Length/1.0e5
            #plt.plot(R,1.2,'ko')
        #elif file[:-4] == '2H':
            #cs=spline(mr[0][1:idx],mr[1][1:idx])
            #R = cs(1.35)*Length/1.0e5
            #plt.plot(R,1.35,'ko')
    #else:
    #    plt.plot(mr[1]*Length/1e5,mr[0],label=file[:-4],alpha=0.2)

config_obj = config.Config()
config_dict = config_obj.config_dict
strains = config_dict['Run']['waveforms']
hardcoded = strains
#hardcoded = '''THC:0036:R03,
#THC:0019:R05,
#BAM:0088:R01,
#THC:0002:R01,
#THC:0011:R01,
#BAM:0070:R01,
#BAM:0065:R03,
#THC:0010:R01,
#BAM:0002:R02'''

#strains = '/'.join(os.getcwd().split('/')[:-2])+'/NR_strains'
#strains = os.listdir(strains)

#strains = [f"Soultanis/{i}" for i in np.around(np.arange(1.20,1.55,0.05),3)]
for strain in strains:
    data = nr.NumericalData(strain,sampling_frequency=2*8192)
    try:
        file = [i for i in eos if data.metadata_dict['id_eos'].upper() in i][0]
        mr = np.load(file)
        m = data.metadata_dict['id_mass_starA']
        mx=np.amax(mr[0])
        idx=np.where(mr[0]==mx)
        idx=idx[0][0]
        cs=spline(mr[0][1:idx],mr[1][1:idx])
        R = cs(m)*Length/1.0e5
        if strain in hardcoded:
            plt.plot(R,m,'o',color='k',ms=4)
        #else:
            #if m < 1.5:
                #plt.plot(R,m,'ro')
                ##pass
                ##print(strain)
            #else:
                #print(f"{strain},")
                #plt.plot(R,m,'ro')
    except:
        print(strain, data.metadata_dict['id_eos'])

r = np.arange(10.7,18.51,0.05)
y1 = np.ones(r.size)*3
y2 = np.zeros(r.size)

low_line = 0.12074852048946*r - 0.38801323366469
high_line = 0.20414622214340*r - 1.27775721544950

plt.fill_between(r, low_line, y1, interpolate=True, color='darksalmon',alpha=0.25)
plt.fill_between(r, low_line, high_line, interpolate=True, color='yellowgreen',alpha=0.1)
plt.fill_between(r, y2, high_line, interpolate=True, color='darkturquoise',alpha=0.1)


plt.plot(r,low_line,'--',color='gray')
plt.plot(r,high_line,'--',color='gray')

plt.xlabel('$R [R_\odot]$')
plt.ylabel('$M [M_\odot]$')
plt.legend()
plt.axis(xmin=10.7,xmax=17.5,ymin=1,ymax=1.8)
plt.grid()
plt.tight_layout()
plt.savefig('mr.pdf')
plt.show()

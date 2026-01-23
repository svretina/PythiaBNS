#!/usr/bin/env python
import numpy as np
from scipy.interpolate import PchipInterpolator
from gw_pipe import NR_strains as nr
import os
import pandas as pd

# cgs
c = 2.9979e10
G = 6.67408e-8
Msun = 1.989e33

Length = G*Msun/c**2/1e5  # km
Time = Length/c
Density = Msun/Length**3

def get_Lambda_and_R(m, filename):
    data = nr.NumericalData(filename)
    
    eos_name = data.metadata_dict['id_eos'].upper()
    #print(eos_name, filename)
    available = os.listdir("EOS/from_Koutalios")
    if f"{eos_name.lower()}.npy" not in [a.lower() for a in available]:
        #print(filename, eos_name)
        raise ValueError("EOS not yet implemented.")
    else:
        dataMR = np.load(f"EOS/from_Koutalios/{eos_name}.npy")
        dataKL = np.load(f"EOS/from_Koutalios/K_L_{eos_name}.npy")

        mass = dataMR[0]
        radius = dataMR[1]*Length
        L = dataKL[1]
        mass = mass[radius<18]
        L = L[radius<18]
        radius=radius[radius<18]

    data = np.array([np.array([m1, r1, l1])
                    for m1, r1, l1 in sorted(zip(mass, radius, L))])

    mass = data[:, 0]
    radius = data[:, 1]
    L = data[:, 2]

    interpolatorL = PchipInterpolator(mass, L)
    interpolatorR = PchipInterpolator(mass, radius)

    L, R = interpolatorL.__call__(m), interpolatorR.__call__(m)

    return L, R


def Lambda_tilde(filename):

    data = nr.NumericalData(filename)
    m1 = data.metadata_dict["id_mass_starA"]
    m2 = data.metadata_dict["id_mass_starB"]
    #print(filename, data.metadata_dict["id_eos"], m1,m2)
    L1, _ = get_Lambda_and_R(m1, filename)
    L2, _ = get_Lambda_and_R(m2, filename)
    temp = (m1+12.0*m2)*m1**4.0 * L1 + (m2+12.0*m1)*m2**4.0 * L2
    temp2 = (m1+m2)**5
    return 16.0/13.0 * (temp/temp2)


def get_metadata(filename):
    data = nr.NumericalData(filename)
    m1 = data.metadata_dict["id_mass_starA"]
    m2 = data.metadata_dict["id_mass_starB"]
    return data.metadata_dict["id_eos"], m1, m2


def get_predictor(filename):
    # masses to evaluate
    masses = [1.2, 1.4, 1.6, 1.8]
    L = []
    R = []
    Ltilde = []
    kept = []
    try:
        _, _ = get_Lambda_and_R(1, filename)
        Ltilde.append(Lambda_tilde(filename))
        kept.append(filename)
        for mass in masses:
            Ltemp, Rtemp = (get_Lambda_and_R(mass, filename))
            L.append(Ltemp)
            R.append(Rtemp)
    except ValueError:
        print(get_metadata(filename))
        print(f"{filename} Not available")

    R = np.array(R).reshape(len(kept), 4)
    L = np.array(L).reshape(len(kept), 4)

    eos = []
    m1 = []
    m2 = []
    for filename in kept:
        temp1,temp2,temp3 = get_metadata(filename)
        eos.append(temp1.lower())
        m1.append(temp2)
        m2.append(temp3)

    m1 = np.array(m1)
    m2 = np.array(m2)
    Mchirp = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
    q = min(m1,m2)/max(m1,m2)

        
    X = pd.DataFrame([eos,
                      Mchirp,q,
                      R[:,0],R[:,1],R[:,2],R[:,3],
                      Ltilde,
                      L[:,0],L[:,1],L[:,2],L[:,3]])
    X = X.T
    return X

#namelist = os.listdir("NR_strains")

#for name in namelist:
    #L,R =get_Lambda_and_R(1.35, name)

    #print(L,R)

    ##_ = get_predictor(name)

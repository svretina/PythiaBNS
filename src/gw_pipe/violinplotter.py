#!/usr/bin/env python3
import dill
from gw_pipe import config
import matplotlib.pyplot as plt
import numpy as np
import os
rcparams = {}
rcparams["axes.linewidth"] = 0.5
rcparams["font.family"] = "serif"
rcparams["font.size"] = 12
# rcparams['legend.fontsize'] = 12
rcparams["mathtext.fontset"] = "stix"
rcparams["text.usetex"] = True
rcparams["figure.dpi"] = 300
rcparams["figure.figsize"] = (
    1920/rcparams["figure.dpi"],
    1.2*1080/rcparams["figure.dpi"],
)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
# rcparams['lines.linewidth'] = 1

plt.rcParams.update(rcparams)  # update plot parameters

filename = 'result_pickle.pickle'

filenames = [i for i in os.listdir() if 'pickle' in i and 'regress' not in i]

for filename in filenames:
    with open(filename,'rb') as file:
        data = dill.load(file)

    data = dict((key,d[key]) for d in data for key in d)

    config_obj = config.Config()
    config_dict = config_obj.config_dict

    names = config_dict['Run']['waveforms']

    size = len(names)
    pos = np.arange(0,size,1)
    fig,ax = plt.subplots()
    labels = []
    colors = []
    for i in range(size):
        FF = np.array([data[key][0] for key in data.keys() if key[:-2] in names[i]])
        label = [data[key][-1] for key in data.keys() if key[:-2] in names[i]][0]
        labels.append(label)
#    for i,FF in enumerate([FF1,FF2,FF3,FF4,FF5,FF6]):#,FF7,FF8,FF9]):
        violin_plot = plt.violinplot(FF,
                [pos[i]],points=200,showmeans=True, showextrema=False,
                quantiles=[0.0015,0.9985], bw_method=0.5)

#        print(violin_plot['cmeans'].__dir__())
#        exit()
        color = violin_plot['cquantiles'].get_edgecolor()[0]

        colors.append(color)
        plt.vlines([pos[i]],
                np.quantile(FF,0.05),np.quantile(FF,0.95),
                color=color)
        pc = violin_plot["bodies"][0]
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        #pc = violin_plot["cbars"]
        #pc.set_facecolor(color)
        #pc.set_edgecolor(color)
        #pc = violin_plot["cmins"]
        #pc.set_facecolor(color)
        #pc.set_edgecolor(color)
        #pc = violin_plot["cmaxes"]
        #pc.set_facecolor(color)
        #pc.set_edgecolor(color)
        pc = violin_plot["cmeans"]
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc = violin_plot["cquantiles"]
        pc.set_facecolor(color)
        pc.set_edgecolor(color)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation = 90)
                        #label7,
                        #label8,
                        #label9], rotation = 90)
    [ax.get_xticklabels()[i].set_color(colors[i]) for i in range(len(colors))]
    ax.set_ylabel('Fitting Factors')
    ax.set_ylim(0.93,0.99)
    ax.set_yticks(np.arange(0.93,0.991,0.01))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{filename[7:-7]}_violin_plot.png")
    #plt.show()

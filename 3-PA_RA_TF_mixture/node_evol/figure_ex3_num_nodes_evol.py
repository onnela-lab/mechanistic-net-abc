# -*- coding: utf-8 -*-
"""
Code to plot graph of evolution of predictions of three observed data based
on the number of nodes. This is for the 3rd simulated example: with model
(2/3)*PA(4) + (1/6)*RA(4) + (1/6)*TF.
"""

import os
### Base directory for this example on which results and plots will be saved
from mechanistic_net_abc.settings import base_dir_example3

dir_base = base_dir_example3
dir_save_plots = dir_base+"/saved_figures"
# Possibly create a folder to save the simulated data and results
if not os.path.exists(dir_save_plots):
    os.makedirs(dir_save_plots)
    print("Directory created")

### 50 nodes ###
w_pred_50nodes_obs1 = [0.1514, 0.6789, 0.1697]
w_q0025_50nodes_obs1 = [0.0050, 0.4207, 0.0339]
w_q0975_50nodes_obs1 = [0.4151, 0.8580, 0.3000]
mPA_pred_50nodes_obs1 = 4
mPA_vote_50nodes_obs1 = 80
mRA_pred_50nodes_obs1 = 4
mRA_vote_50nodes_obs1 = 200

w_pred_50nodes_obs2 = [0.6755, 0.1361, 0.1884]
w_q0025_50nodes_obs2 = [0.3717, 0.0018, 0.0578]
w_q0975_50nodes_obs2 = [0.8796, 0.3631, 0.3340]
mPA_pred_50nodes_obs2 = 4
mPA_vote_50nodes_obs2 = 200
mRA_pred_50nodes_obs2 = 4
mRA_vote_50nodes_obs2 = 93

w_pred_50nodes_obs3 = [0.7796, 0.0852, 0.1353]
w_q0025_50nodes_obs3 = [0.5785, 0.0014, 0.0347]
w_q0975_50nodes_obs3 = [0.9226, 0.2694, 0.2496]
mPA_pred_50nodes_obs3 = 4
mPA_vote_50nodes_obs3 = 200
mRA_pred_50nodes_obs3 = 4
mRA_vote_50nodes_obs3 = 75

### 100 nodes ###
w_pred_100nodes_obs1 = [0.3313, 0.5512, 0.1176]
w_q0025_100nodes_obs1 = [0.0606, 0.3010, 0.0450]
w_q0975_100nodes_obs1 = [0.5893, 0.8116, 0.2135]
mPA_pred_100nodes_obs1 = 4
mPA_vote_100nodes_obs1 = 195
mRA_pred_100nodes_obs1 = 4
mRA_vote_100nodes_obs1 = 200

w_pred_100nodes_obs2 = [0.7980, 0.0499, 0.1520]
w_q0025_100nodes_obs2 = [0.6420, 0.0009, 0.0573]
w_q0975_100nodes_obs2 = [0.9249, 0.1650, 0.2621]
mPA_pred_100nodes_obs2 = 4
mPA_vote_100nodes_obs2 = 200
mRA_pred_100nodes_obs2 = 4
mRA_vote_100nodes_obs2 = 72

w_pred_100nodes_obs3 = [0.7924, 0.0711, 0.1365]
w_q0025_100nodes_obs3 = [0.6345, 0.0018, 0.0379]
w_q0975_100nodes_obs3 = [0.8968, 0.1891, 0.2376]
mPA_pred_100nodes_obs3 = 4
mPA_vote_100nodes_obs3 = 200
mRA_pred_100nodes_obs3 = 3
mRA_vote_100nodes_obs3 = 74

### 200 nodes ###
w_pred_200nodes_obs1 = [0.6534, 0.1816, 0.1649]
w_q0025_200nodes_obs1 = [0.4714, 0.0407, 0.1039]
w_q0975_200nodes_obs1 = [0.8194, 0.3440, 0.2269]
mPA_pred_200nodes_obs1 = 4
mPA_vote_200nodes_obs1 = 200
mRA_pred_200nodes_obs1 = 4
mRA_vote_200nodes_obs1 = 173

w_pred_200nodes_obs2 = [0.6146, 0.1319, 0.2588]
w_q0025_200nodes_obs2 = [0.4572, 0.0136, 0.1691]
w_q0975_200nodes_obs2 = [0.7633, 0.2694, 0.3244]
mPA_pred_200nodes_obs2 = 4
mPA_vote_200nodes_obs2 = 200
mRA_pred_200nodes_obs2 = 4
mRA_vote_200nodes_obs2 = 146

w_pred_200nodes_obs3 = [0.7041, 0.1497, 0.1461]
w_q0025_200nodes_obs3 = [0.5086, 0.0204, 0.0555]
w_q0975_200nodes_obs3 = [0.8670, 0.3129, 0.2107]
mPA_pred_200nodes_obs3 = 4
mPA_vote_200nodes_obs3 = 200
mRA_pred_200nodes_obs3 = 4
mRA_vote_200nodes_obs3 = 143

### 400 nodes ###
w_pred_400nodes_obs1 = [0.5957, 0.2300, 0.1744]
w_q0025_400nodes_obs1 = [0.4745, 0.1133, 0.1307]
w_q0975_400nodes_obs1 = [0.7207, 0.3374, 0.2283]
mPA_pred_400nodes_obs1 = 4
mPA_vote_400nodes_obs1 = 200
mRA_pred_400nodes_obs1 = 4
mRA_vote_400nodes_obs1 = 197

w_pred_400nodes_obs2 = [0.6922, 0.1374, 0.1704]
w_q0025_400nodes_obs2 = [0.5748, 0.0306, 0.1092]
w_q0975_400nodes_obs2 = [0.8150, 0.2466, 0.2136]
mPA_pred_400nodes_obs2 = 4
mPA_vote_400nodes_obs2 = 200
mRA_pred_400nodes_obs2 = 4
mRA_vote_400nodes_obs2 = 179

w_pred_400nodes_obs3 = [0.6561, 0.1872, 0.1567]
w_q0025_400nodes_obs3 = [0.5187, 0.0605, 0.0938]
w_q0975_400nodes_obs3 = [0.7950, 0.3153, 0.2179]
mPA_pred_400nodes_obs3 = 4
mPA_vote_400nodes_obs3 = 200
mRA_pred_400nodes_obs3 = 4
mRA_vote_400nodes_obs3 = 171

### 800 nodes ###
w_pred_800nodes_obs1 = [0.6137, 0.1965, 0.1898]
w_q0025_800nodes_obs1 = [0.5325, 0.1240, 0.1444]
w_q0975_800nodes_obs1 = [0.6988, 0.2629, 0.2372]
mPA_pred_800nodes_obs1 = 4
mPA_vote_800nodes_obs1 = 200
mRA_pred_800nodes_obs1 = 4
mRA_vote_800nodes_obs1 = 198

w_pred_800nodes_obs2 = [0.6590, 0.1861, 0.1550]
w_q0025_800nodes_obs2 = [0.5716, 0.1087, 0.1159]
w_q0975_800nodes_obs2 = [0.7611, 0.2602, 0.1934]
mPA_pred_800nodes_obs2 = 4
mPA_vote_800nodes_obs2 = 200
mRA_pred_800nodes_obs2 = 4
mRA_vote_800nodes_obs2 = 195

w_pred_800nodes_obs3 = [0.6269, 0.1939, 0.1791]
w_q0025_800nodes_obs3 = [0.5325, 0.1240, 0.1449]
w_q0975_800nodes_obs3 = [0.7151, 0.2697, 0.2109]
mPA_pred_800nodes_obs3 = 4
mPA_vote_800nodes_obs3 = 200
mRA_pred_800nodes_obs3 = 4
mRA_vote_800nodes_obs3 = 200

###############################################################################
### Predictions to draw the figure of evolution predictions with node number
###############################################################################

num_nodes = [50,100,200,400,800]

# Obs 1 #

wPA_obs1 = [w_pred_50nodes_obs1[0], w_pred_100nodes_obs1[0],
            w_pred_200nodes_obs1[0], w_pred_400nodes_obs1[0],
            w_pred_800nodes_obs1[0]]

wPA_q0025_obs1 = [w_q0025_50nodes_obs1[0], w_q0025_100nodes_obs1[0],
                  w_q0025_200nodes_obs1[0], w_q0025_400nodes_obs1[0],
                  w_q0025_800nodes_obs1[0]]

wPA_q0975_obs1 = [w_q0975_50nodes_obs1[0], w_q0975_100nodes_obs1[0],
                  w_q0975_200nodes_obs1[0], w_q0975_400nodes_obs1[0],
                  w_q0975_800nodes_obs1[0]]


wRA_obs1 = [w_pred_50nodes_obs1[1], w_pred_100nodes_obs1[1],
            w_pred_200nodes_obs1[1], w_pred_400nodes_obs1[1],
            w_pred_800nodes_obs1[1]]

wRA_q0025_obs1 = [w_q0025_50nodes_obs1[1], w_q0025_100nodes_obs1[1],
                  w_q0025_200nodes_obs1[1], w_q0025_400nodes_obs1[1],
                  w_q0025_800nodes_obs1[1]]

wRA_q0975_obs1 = [w_q0975_50nodes_obs1[1], w_q0975_100nodes_obs1[1],
                  w_q0975_200nodes_obs1[1], w_q0975_400nodes_obs1[1],
                  w_q0975_800nodes_obs1[1]]


wTF_obs1 = [w_pred_50nodes_obs1[2], w_pred_100nodes_obs1[2],
            w_pred_200nodes_obs1[2], w_pred_400nodes_obs1[2],
            w_pred_800nodes_obs1[2]]

wTF_q0025_obs1 = [w_q0025_50nodes_obs1[2], w_q0025_100nodes_obs1[2],
                  w_q0025_200nodes_obs1[2], w_q0025_400nodes_obs1[2],
                  w_q0025_800nodes_obs1[2]]

wTF_q0975_obs1 = [w_q0975_50nodes_obs1[2], w_q0975_100nodes_obs1[2],
                  w_q0975_200nodes_obs1[2], w_q0975_400nodes_obs1[2],
                  w_q0975_800nodes_obs1[2]]

mPA_obs1 = [mPA_pred_50nodes_obs1, mPA_pred_100nodes_obs1,
            mPA_pred_200nodes_obs1, mPA_pred_400nodes_obs1,
            mPA_pred_800nodes_obs1]

mPA_vote_obs1 = [mPA_vote_50nodes_obs1, mPA_vote_100nodes_obs1,
                 mPA_vote_200nodes_obs1, mPA_vote_400nodes_obs1,
                 mPA_vote_800nodes_obs1]

mRA_obs1 = [mRA_pred_50nodes_obs1, mRA_pred_100nodes_obs1,
            mRA_pred_200nodes_obs1, mRA_pred_400nodes_obs1,
            mRA_pred_800nodes_obs1]

mRA_vote_obs1 = [mRA_vote_50nodes_obs1, mRA_vote_100nodes_obs1,
                 mRA_vote_200nodes_obs1, mRA_vote_400nodes_obs1,
                 mRA_vote_800nodes_obs1]

# Obs 2 #

wPA_obs2 = [w_pred_50nodes_obs2[0], w_pred_100nodes_obs2[0],
            w_pred_200nodes_obs2[0], w_pred_400nodes_obs2[0],
            w_pred_800nodes_obs2[0]]

wPA_q0025_obs2 = [w_q0025_50nodes_obs2[0], w_q0025_100nodes_obs2[0],
                  w_q0025_200nodes_obs2[0], w_q0025_400nodes_obs2[0],
                  w_q0025_800nodes_obs2[0]]

wPA_q0975_obs2 = [w_q0975_50nodes_obs2[0], w_q0975_100nodes_obs2[0],
                  w_q0975_200nodes_obs2[0], w_q0975_400nodes_obs2[0],
                  w_q0975_800nodes_obs2[0]]


wRA_obs2 = [w_pred_50nodes_obs2[1], w_pred_100nodes_obs2[1],
            w_pred_200nodes_obs2[1], w_pred_400nodes_obs2[1],
            w_pred_800nodes_obs2[1]]

wRA_q0025_obs2 = [w_q0025_50nodes_obs2[1], w_q0025_100nodes_obs2[1],
                  w_q0025_200nodes_obs2[1], w_q0025_400nodes_obs2[1],
                  w_q0025_800nodes_obs2[1]]

wRA_q0975_obs2 = [w_q0975_50nodes_obs2[1], w_q0975_100nodes_obs2[1],
                  w_q0975_200nodes_obs2[1], w_q0975_400nodes_obs2[1],
                  w_q0975_800nodes_obs2[1]]


wTF_obs2 = [w_pred_50nodes_obs2[2], w_pred_100nodes_obs2[2],
            w_pred_200nodes_obs2[2], w_pred_400nodes_obs2[2],
            w_pred_800nodes_obs2[2]]

wTF_q0025_obs2 = [w_q0025_50nodes_obs2[2], w_q0025_100nodes_obs2[2],
                  w_q0025_200nodes_obs2[2], w_q0025_400nodes_obs2[2],
                  w_q0025_800nodes_obs2[2]]

wTF_q0975_obs2 = [w_q0975_50nodes_obs2[2], w_q0975_100nodes_obs2[2],
                  w_q0975_200nodes_obs2[2], w_q0975_400nodes_obs2[2],
                  w_q0975_800nodes_obs2[2]]

mPA_obs2 = [mPA_pred_50nodes_obs2, mPA_pred_100nodes_obs2,
            mPA_pred_200nodes_obs2, mPA_pred_400nodes_obs2,
            mPA_pred_800nodes_obs2]

mPA_vote_obs2 = [mPA_vote_50nodes_obs2, mPA_vote_100nodes_obs2,
                 mPA_vote_200nodes_obs2, mPA_vote_400nodes_obs2,
                 mPA_vote_800nodes_obs2]

mRA_obs2 = [mRA_pred_50nodes_obs2, mRA_pred_100nodes_obs2,
            mRA_pred_200nodes_obs2, mRA_pred_400nodes_obs2,
            mRA_pred_800nodes_obs2]

mRA_vote_obs2 = [mRA_vote_50nodes_obs2, mRA_vote_100nodes_obs2,
                 mRA_vote_200nodes_obs2, mRA_vote_400nodes_obs2,
                 mRA_vote_800nodes_obs2]


# Obs 3 #
 
wPA_obs3 = [w_pred_50nodes_obs3[0], w_pred_100nodes_obs3[0],
            w_pred_200nodes_obs3[0], w_pred_400nodes_obs3[0],
            w_pred_800nodes_obs3[0]]

wPA_q0025_obs3 = [w_q0025_50nodes_obs3[0], w_q0025_100nodes_obs3[0],
                  w_q0025_200nodes_obs3[0], w_q0025_400nodes_obs3[0],
                  w_q0025_800nodes_obs3[0]]

wPA_q0975_obs3 = [w_q0975_50nodes_obs3[0], w_q0975_100nodes_obs3[0],
                  w_q0975_200nodes_obs3[0], w_q0975_400nodes_obs3[0],
                  w_q0975_800nodes_obs3[0]]


wRA_obs3 = [w_pred_50nodes_obs3[1], w_pred_100nodes_obs3[1],
            w_pred_200nodes_obs3[1], w_pred_400nodes_obs3[1],
            w_pred_800nodes_obs3[1]]

wRA_q0025_obs3 = [w_q0025_50nodes_obs3[1], w_q0025_100nodes_obs3[1],
                  w_q0025_200nodes_obs3[1], w_q0025_400nodes_obs3[1],
                  w_q0025_800nodes_obs3[1]]

wRA_q0975_obs3 = [w_q0975_50nodes_obs3[1], w_q0975_100nodes_obs3[1],
                  w_q0975_200nodes_obs3[1], w_q0975_400nodes_obs3[1],
                  w_q0975_800nodes_obs3[1]]


wTF_obs3 = [w_pred_50nodes_obs3[2], w_pred_100nodes_obs3[2],
            w_pred_200nodes_obs3[2], w_pred_400nodes_obs3[2],
            w_pred_800nodes_obs3[2]]

wTF_q0025_obs3 = [w_q0025_50nodes_obs3[2], w_q0025_100nodes_obs3[2],
                  w_q0025_200nodes_obs3[2], w_q0025_400nodes_obs3[2],
                  w_q0025_800nodes_obs3[2]]

wTF_q0975_obs3 = [w_q0975_50nodes_obs3[2], w_q0975_100nodes_obs3[2],
                  w_q0975_200nodes_obs3[2], w_q0975_400nodes_obs3[2],
                  w_q0975_800nodes_obs3[2]]

mPA_obs3 = [mPA_pred_50nodes_obs3, mPA_pred_100nodes_obs3,
            mPA_pred_200nodes_obs3, mPA_pred_400nodes_obs3,
            mPA_pred_800nodes_obs3]

mPA_vote_obs3 = [mPA_vote_50nodes_obs3, mPA_vote_100nodes_obs3,
                 mPA_vote_200nodes_obs3, mPA_vote_400nodes_obs3,
                 mPA_vote_800nodes_obs3]

mRA_obs3 = [mRA_pred_50nodes_obs3, mRA_pred_100nodes_obs3,
            mRA_pred_200nodes_obs3, mRA_pred_400nodes_obs3,
            mRA_pred_800nodes_obs3]

mRA_vote_obs3 = [mRA_vote_50nodes_obs3, mRA_vote_100nodes_obs3,
                 mRA_vote_200nodes_obs3, mRA_vote_400nodes_obs3,
                 mRA_vote_800nodes_obs3]


import matplotlib.pyplot as plt

# And only the avg_clustering_coef which is the most important to identify w_TF
xylabelsize = 12
tickslabelsize = 10
legendsize = 12

plt.figure(figsize=(4,6))
plt.axhline(y=2/3, color='black', linestyle='--', label="truth")
plt.plot(num_nodes, wPA_obs1, label="$G_1^*$", color="tab:blue")
plt.fill_between(num_nodes, wPA_q0025_obs1, wPA_q0975_obs1,
                 alpha=0.25, edgecolor="tab:blue", facecolor="tab:blue",
                 linewidth=0)
plt.plot(num_nodes, wPA_obs2, label="$G_2^*$", color="tab:red")
plt.fill_between(num_nodes, wPA_q0025_obs2, wPA_q0975_obs2,
                 alpha=0.25, edgecolor="tab:red", facecolor="tab:red",
                 linewidth=0)
plt.plot(num_nodes, wPA_obs3, label="$G_3^*$", color="tab:green")
plt.fill_between(num_nodes, wPA_q0025_obs3, wPA_q0975_obs3,
                 alpha=0.25, edgecolor="tab:green", facecolor="tab:green",
                 linewidth=0)
plt.xticks(num_nodes)
plt.ylim(0,1)
plt.legend(loc='lower right', ncol=2, prop={'size': legendsize})
plt.xlabel("Number of nodes", size=xylabelsize)
plt.ylabel(r"$\alpha_{PA}$", size=xylabelsize)
plt.tick_params(axis='both', which='major', labelsize=tickslabelsize)
plt.savefig(dir_save_plots+"/num_nodes_evol_wPA.pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/num_nodes_evol_wPA.eps", bbox_inches='tight')
plt.show()


plt.figure(figsize=(4,6))
plt.axhline(y=1/6, color='black', linestyle='--', label="truth")
plt.plot(num_nodes, wRA_obs1, label="$G_1^*$", color="tab:blue")
plt.fill_between(num_nodes, wRA_q0025_obs1, wRA_q0975_obs1,
                 alpha=0.25, edgecolor="tab:blue", facecolor="tab:blue",
                 linewidth=0)
plt.plot(num_nodes, wRA_obs2, label="$G_2^*$", color="tab:red")
plt.fill_between(num_nodes, wRA_q0025_obs2, wRA_q0975_obs2,
                 alpha=0.25, edgecolor="tab:red", facecolor="tab:red",
                 linewidth=0)
plt.plot(num_nodes, wRA_obs3, label="$G_3^*$", color="tab:green")
plt.fill_between(num_nodes, wRA_q0025_obs3, wRA_q0975_obs3,
                 alpha=0.25, edgecolor="tab:green", facecolor="tab:green",
                 linewidth=0)
plt.xticks(num_nodes)
plt.ylim(0,1)
plt.legend(ncol=2, prop={'size': legendsize})
plt.xlabel("Number of nodes", size=xylabelsize)
plt.ylabel(r"$\alpha_{RA}$", size=xylabelsize)
plt.tick_params(axis='both', which='major', labelsize=tickslabelsize)
plt.savefig(dir_save_plots+"/num_nodes_evol_wRA.pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/num_nodes_evol_wRA.eps", bbox_inches='tight')
plt.show()


plt.figure(figsize=(4,6))
plt.axhline(y=1/6, color='black', linestyle='--', label="truth")
plt.plot(num_nodes, wTF_obs1, label="$G_1^*$", color="tab:blue")
plt.fill_between(num_nodes, wTF_q0025_obs1, wTF_q0975_obs1,
                 alpha=0.25, edgecolor="tab:blue", facecolor="tab:blue",
                 linewidth=0)
plt.plot(num_nodes, wTF_obs2, label="$G_2^*$", color="tab:red")
plt.fill_between(num_nodes, wTF_q0025_obs2, wTF_q0975_obs2,
                 alpha=0.25, edgecolor="tab:red", facecolor="tab:red",
                 linewidth=0)
plt.plot(num_nodes, wTF_obs3, label="$G_3^*$", color="tab:green")
plt.fill_between(num_nodes, wTF_q0025_obs3, wTF_q0975_obs3,
                 alpha=0.25, edgecolor="tab:green", facecolor="tab:green",
                 linewidth=0)
plt.xticks(num_nodes)
plt.ylim(0,1)
plt.legend(ncol=2, prop={'size': legendsize})
plt.xlabel("Number of nodes", size=xylabelsize)
plt.ylabel(r"$\alpha_{TF}$", size=xylabelsize)
plt.tick_params(axis='both', which='major', labelsize=tickslabelsize)
plt.savefig(dir_save_plots+"/num_nodes_evol_wTF.pdf", bbox_inches='tight')
plt.savefig(dir_save_plots+"/num_nodes_evol_wTF.eps", bbox_inches='tight')
plt.show()



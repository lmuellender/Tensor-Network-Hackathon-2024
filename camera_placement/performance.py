import numpy as np
import matplotlib.pyplot as plt
import glob

# performance of brute force solver
N_list_bf = np.array(list(range(9,17))+[18,20,22,24,32])
# bf_files = glob('./results_bf/bruteforce_N*.npy')
ene_c_bf = np.zeros(len(N_list_bf))
time_c_bf = np.zeros(len(N_list_bf))
ene_u_bf = np.zeros(len(N_list_bf))
time_u_bf = np.zeros(len(N_list_bf))
C_u_bf = np.array([6,6,6,7,8,9,10,11,11,13,15,16,21])
for i,(N,C_u) in enumerate(zip(N_list_bf[:-1],C_u_bf[:-1])):
    data_c = np.load(f'camera_placement/results_bf/bruteforce_N{N}_C{N//2}.npy', allow_pickle=True)
    data_c = data_c.item()
    ene_c_bf[i] = data_c['energy']
    time_c_bf[i] = data_c['time']
    data_u = np.load(f'camera_placement/results_bf/bruteforce_N{N}_C{C_u}.npy', allow_pickle=True)
    data_u = data_u.item()
    ene_u_bf[i] = data_u['energy']
    time_u_bf[i] = data_u['time']
ene_c_bf[-1] = -46.18
time_c_bf[-1] = 4507.563
ene_u_bf[-1] = -50.023
time_u_bf[-1] = 48150.47
# best solution N=32: [0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0.]

# gurobi

data_gurobi = np.load('camera_placement/result_gurobi_LOCAL.npz')
N_list_gur = np.arange(10,200,2)
times_u_gur = data_gurobi['times_unc']
ene_u_gur = data_gurobi['energies_unc']
times_c_gur = data_gurobi['times_con']
ene_c_gur = data_gurobi['energies_con']

fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
ax[0].plot(N_list_bf, ene_u_bf, 'o-', label='brute force')
ax[0].plot(N_list_gur, ene_u_gur, 'x-', label='gurobi')
ax[1].plot(N_list_bf, ene_c_bf, 'o-', label='brute force')
ax[1].plot(N_list_gur, ene_c_gur, 'x-', label='gurobi')
ax[0].set_ylabel('Energy')
ax[0].set_title('Unconstrained')
ax[1].set_xlabel('Number of sites')
ax[1].set_ylabel('Energy')
ax[1].set_title('Constrained')
ax[0].legend()
ax[1].legend()
fig.tight_layout()

fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
ax[0].plot(N_list_bf, time_c_bf, 'o-', label='brute force')
ax[0].plot(N_list_gur, times_c_gur, 'o-', label='gurobi')
ax[1].plot(N_list_bf, time_u_bf, 'o-', label='brute force')
ax[1].plot(N_list_gur, times_u_gur, 'o-', label='gurobi')
ax[0].set_ylabel('Time [s]')
ax[0].set_yscale('log')
ax[0].set_title('Constrained')
ax[0].legend()
ax[1].set_xlabel('Number of sites')
ax[1].set_ylabel('Time [s]')
ax[1].set_yscale('log')
ax[1].set_title('Unconstrained')
ax[1].legend()
fig.tight_layout()

plt.show()


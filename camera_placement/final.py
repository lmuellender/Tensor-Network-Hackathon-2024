# Loading libraries to handle the problem
from ocp import *


# number of sites
# N = 10
sizes = [N for N in range(9,16)] + [2**(N+4) for N in range(7)]#[N for N in range(16,201,2)]

xi_list = [.25]

# number of cameras
C = 1
P = 25.0

# side of the square defining the simplified model
a = 10

num_shots = 100 # shots for measuring the bitstring
max_bond_dim = 32
cut_ratio = 1e-8
max_iter = 50
tn_type = 6 # (5)TTN, (6)MPS
tensor_backend = 2
statics_method = 2 #1 : sweep, 2 : sweep with space expansion, 4 : imaginary time evolution with TDVP space expansion
                #5 : imaginary time evolution with TDVP single-site, 33 : imaginary nearest-neighbor TEBD.

tau = 0.1
rel_deviation = 1e-8 #tolerance for convergence

seed=42

random_sweep = False
sweep_order=None
# sweep_order=[]
# for i in range(int(np.log2(N)-1)):
#     for j in range(int(2**(i+1))):
#         sweep_order.append((i,j))
# random.shuffle(sweep_order)


np.random.seed(seed)

params, dlist, mlist, slist = everythin_else(sizes, C, P, xi_list, a, seed, tau, max_bond_dim, cut_ratio, max_iter, 
                                             statics_method, tn_type, tensor_backend, num_shots, rel_deviation, random_sweep)

tn_file = open('tn_mps_var'+str(statics_method)+'.dat','a')
print ('# [N, xi, seed, time(s), cost, solution]:',file=tn_file)

results = []
tn_time_list = []
tn_energy_list = []
bit_string_list = []
for ii, elem in enumerate(params):
    print('params = ', elem)
    
    # run the simulation
    start_time = time.time()
    
    slist[ii].run(elem, delete_existing_folder=True, nthreads=1)
    
    dt_tn = time.time() - start_time
    # get the results
    results.append( slist[ii].get_static_obs(elem) )

    energy = results[ii]['energy']
    proj = results[ii]['projective_measurements']

    tn_energy_list.append(energy)
    tn_time_list.append(dt_tn)
    
    bit_strings=[]
    cnt=0
    for bit in proj.keys():
        bit_strings.append([*bit])
        plot_antennas(dlist[ii], status=bit_strings[cnt])
        plt.savefig('camera_placement_N'+str(elem['N'])+'_xi'+str(elem['xi'])+'_var'+str(statics_method)+'_.pdf',bbox_inches='tight')
        cnt+=1
    bit_string_list.append(bit_strings)

    print(elem['N'], elem['xi'], seed, tn_time_list[ii], tn_energy_list[ii], bit_string_list[ii],file=tn_file)

tn_file.close()


plt.close()

mpl.rcParams["figure.dpi"] = 200
plt.rcParams["figure.figsize"] = (6,4)
plt.plot(sizes, tn_time_list, 'x', color="purple",label='MPS_ITE4')
# plt.plot(sizes, ed_time, '+', color="green",label='ED')
plt.legend()
plt.xlabel("$N$")
plt.ylabel("$ Runtime $")
plt.savefig('runtime.png',bbox_inches='tight')
plt.close()


mpl.rcParams["figure.dpi"] = 200
plt.rcParams["figure.figsize"] = (6,4)
plt.plot(sizes, tn_energy_list, 'x', color="purple",label='MPS_ITE4')
# plt.plot(sizes, ed_energy, '+', color="green",label='ED')
plt.legend()
plt.xlabel("$N$")
plt.ylabel("$ Cost $")
plt.savefig('energy.png',bbox_inches='tight')
plt.close()
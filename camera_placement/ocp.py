# Loading libraries to handle the problem
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol
import itertools
import qtealeaves as qtl
from qtealeaves import modeling
from qtealeaves.convergence_parameters import TNConvergenceParameters
from scipy.sparse.linalg import eigs, eigsh
from qiskit.quantum_info import Statevector
import time


#%%

# Utility for plotting the antennas
def plot_antennas(df, status, axes=None):
    """
    Function for plotting the cameras with their field of view.
    
    
    Input:
    -----------------------------------------
       df:     pandas Dataframe with cols 
               'id', 'x_loc', 'y_loc', 'radius'
       status: numpy.array describing the status on/off (0/1) of the sites
       axes:   mpl Axes object to add the cameras.
               If None, new Axes object is created. 
               Default: None.
    
    Returns:
    -----------------------------------------
    None
    
    """
    if axes is None:
        figure, axes = plt.subplots()
    for n, row in df.iterrows():
        color = 'g' if int(status[n]) == 0 else 'b'
        alpha = 0.5 if int(status[n]) == 0 else 0.3
        Drawing_colored_circle = plt.Circle(( row.x_loc , row.y_loc ), row.radius, color=color, alpha=alpha)
        axes.set_aspect( 1 )
        axes.add_artist( Drawing_colored_circle )
    # ax = df.plot(x='x_loc', y='y_loc', kind='scatter', s='area', alpha=0.45)
    df.plot(x='x_loc', y='y_loc', kind='scatter', s=10., color='k', ax=axes)
    for i in df.iterrows():
        v = i[1]['x_loc'], i[1]['y_loc']
        axes.annotate(i[0],v)
    # df[['x_loc','y_loc','id']].apply(lambda x: axes.text(x[0], x[1], int(x[2])),axis=1)
    axes.set_ylabel('Latitude', fontsize=14)
    axes.set_xlabel('Longitude', fontsize=14)

#generate the problem
def circle_overlap(x0, y0, r0, x1, y1, r1):
    """ 
    Function to calculate the overlap between two circles,
    given the coordinates of the center and the radius of each circle.

    
    Input:
    -----------------------------------------
       x0:  float, x-coordinate of first circle
       y0:  float, y-coordinate of first circle
       r0:  float, radius of first circle
       x1:  float, x-coordinate of second circle
       y1:  float, y-coordinate of second circle
       r1:  float, radius of second circle
    
    Returns:
    -----------------------------------------
    float, overlap.
    
    """
    rr0 = r0*r0
    rr1 = r1*r1
    c = np.sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0))
    if c > r0 + r1:
        return 0.0
    if r0 > r1 + c:
        return np.pi*r1**2
    if r1 > r0 + c:
        return np.pi*r0**2
    phi = (np.arccos((rr0+(c*c)-rr1) / (2*r0*c)))
    theta = (np.arccos((rr1+(c*c)-rr0) / (2*r1*c)))
    overlap = theta*rr1 + phi*rr0 - 0.5*np.sqrt((r0+r1-c) * (r0-r1+c) * (r1-r0+c)*(r1+r0+c))
    return overlap

def generate_problem(data, xi, normalize=False):
    """ 
    Function to calculate the overlap matrix and the linear term of the Ising model,
    given the problem parameters.

    
    Input:
    -----------------------------------------
        data: pandas.DataFrame,  data of the cameras.
        xi: float, relative multiplier.
        normalize: bool, return normalized terms. Default: False.
    
    Returns:
    -----------------------------------------
    float, overlap.
    
    """
    assert xi >= 0, r"\xi parameters must be non-negative."
    W = np.zeros((data.shape[0], data.shape[0]))
    for n, row1 in data.iterrows():
        for m, row2 in data.iterrows():
            if m == n:
                continue
            W[n, m] = circle_overlap(row1.x_loc, 
                                     row1.y_loc, 
                                     row1.radius,
                                     row2.x_loc,
                                     row2.y_loc,
                                     row2.radius)

    # We define the quantities in the theory
    A = -xi * data['area'].to_numpy()

    ## Normalize the matrices
    if normalize:
        norm = max([np.max(abs(W)), np.max(abs(A))])
        W = W.copy()/norm
        A = A.copy()/norm
    return W, A

def number_constraint(W, A, C, P, normalize=False):
    """ 
    Function to calculate the overlap matrix and the linear term of the constrained 
    Ising model, given the unconstrained Ising matrix and linear term.

    
    Input:
    -----------------------------------------
       W:           numpy.ndarray, Ising matrix of unconstrained model, shape=(N, N)
       A:           numpy.array, Ising linear term of unconstrained model, shape=(N,)
       C:           int, number of available cameras
       P:           int, penalty factor
       normalize:   bool, normalize the Ising terms? Default: False. 
    
    Returns:
    -----------------------------------------
        W_P:     numpy.ndarray, Ising matrix of constrained model, shape=(N, N)
        A_P:     numpy.array, Ising linear term of constrained model, shape=(N,)
        scaling: float, scaling applied to the matrices (if normalize is False, it is 1.0).
    
    """
    num_sites = W.shape[0]
    assert P >=0, "Penalty must be non-negative"
    assert np.shape(W)[0] > C > 0, "Number of cameras must be in (0, N)"
    W_P = W.copy() + 2 * P*(np.ones((num_sites, num_sites))
                 -np.eye(num_sites))
    A_P = A.copy() + 2 * P * (num_sites - 2*C) * np.ones((num_sites,))
    scaling = 1.0
    if normalize:
        max_w = np.max(W_P)
        max_node = np.max(np.abs(node_weight))
        print(node_weight)
        scaling = max([max_w, max_node])
        W_P *= scaling
        A_P *= scaling
    return W_P, A_P, scaling

def generate_data(N,a,seed):
    np.random.seed(seed)
    # Generate radius of the antennas
    radius = .5*a*(1. + np.random.rand(N))/np.sqrt(N)

    # Distribute sites uniformly but not symmetrically in the square
    sampler = Sobol(2, scramble=False, optimization='lloyd')
    sequence = sampler.random_base2(m=int(np.ceil(np.log2(N))))

    # Save the data of the cameras in a Pandas Dataframe
    xs = [a*x[0] for x in sequence][:N]
    ys = [a*x[1] for x in sequence][:N]

    d = {'id': np.arange(N, dtype=int), 
        'x_loc': xs, 
        'y_loc': ys, 
        'radius': radius,
        'area': np.pi*radius**2}
    data = pd.DataFrame(data=d)

    return data

def model_ocp(params, W_P, A_P, my_ops, max_bond_dim, cut_ratio, max_iter, statics_method, tn_type, tensor_backend, num_shots):
    model_name = lambda params: "CameraPlacement_xi%2.4f" % (params["xi"])

    # Define a general quantum model - 1-dimensional, of size "N", with a given name
    model = modeling.QuantumModel(dim=1, lvals="N", name=model_name)

    # this is the hamiltonian
    
    model += modeling.RandomizedLocalTerm(operator="sz", strength="xi", prefactor=1, coupling_entries=A_P)
    
    model += modeling.TwoBodyAllToAllTerm1D(
        operators=["sz", "sz"], strength=1., prefactor=1, coupling_matrix=W_P
    )

    #observables
    # first define a general TNObservables class
    my_obs = qtl.observables.TNObservables()
    my_obs += qtl.observables.TNObsProjective(num_shots=num_shots)

    # we put it all into the TNConvergenceParameters object
    conv_params = TNConvergenceParameters(max_bond_dimension = max_bond_dim,
                                        cut_ratio = cut_ratio,
                                        max_iter = max_iter,
                                        statics_method=statics_method,
                                        data_type='D', # double precision real
                                        device='cpu', # we are running on CPUs
                                        )

    # input_folder = lambda params : 'input_L%02d_g%2.4f'%(params['L'],params['xi'],)
    # output_folder = lambda params : 'output_L%02d_g%2.4f'%(params['L'],params['xi'],)

    simulation = qtl.QuantumGreenTeaSimulation(model, my_ops, conv_params, my_obs,
                                        tn_type=tn_type,
                                        tensor_backend=tensor_backend,
                                        # folder_name_input=input_folder,
                                        # folder_name_output=output_folder,
                                        store_checkpoints=False
        )
    return model, simulation

def everythin_else(sizes, C, P, xi_list, a, seed, max_bond_dim, cut_ratio, max_iter, statics_method, tn_type, tensor_backend, num_shots, sweep_order=None):
    
    my_ops = qtl.operators.TNSpin12Operators()

    # itertools: 1st one iterates slowly
    params = []
    dlist = []
    mlist=[]
    slist = []
    for size, xi in itertools.product(sizes, xi_list):
        if sweep_order==None:
            params.append({
                    'N' : size, 
                    'xi' : xi,
            })
        else:
            params.append({
                    'N' : size, 
                    'xi' : xi,
                    'sweep_order': sweep_order
            })
        data = generate_data(size,a,seed)

        # actual generation of the problem
        W, A = generate_problem(data, 1.0, normalize=False)
        W_P, A_P, scaling = number_constraint(W, A, C, P=P, normalize=False)

        model, simulation = model_ocp(params, W_P, A_P, my_ops, max_bond_dim, cut_ratio, max_iter, statics_method, tn_type, tensor_backend, num_shots)

        dlist.append(data)
        mlist.append(model)
        slist.append(simulation)

    return params, dlist, mlist, slist

def final():
    
    return


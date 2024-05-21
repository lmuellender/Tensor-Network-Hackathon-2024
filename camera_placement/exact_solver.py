# Loading libraries to handle the problem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol
import itertools
import qtealeaves as qtl
from qtealeaves import modeling
from qtealeaves.convergence_parameters import TNConvergenceParameters
from itertools import combinations
import time

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
        color = 'grey' if int(status[n]) == 0 else 'red'
        alpha = 0.5 if int(status[n]) == 0 else 0.3
        Drawing_colored_circle = plt.Circle(( row.x_loc , row.y_loc ), row.radius, color=color, alpha=alpha)
        axes.set_aspect( 1 )
        axes.add_artist( Drawing_colored_circle )
    # ax = df.plot(x='x_loc', y='y_loc', kind='scatter', s='area', alpha=0.45)
    df.plot(x='x_loc', y='y_loc', kind='scatter', s=10., color='k', ax=axes)
    df[['x_loc','y_loc','id']].apply(lambda x: axes.text(x.iloc[0], x.iloc[1], int(x.iloc[2])),axis=1)
    axes.set_ylabel('Latitude', fontsize=14)
    axes.set_xlabel('Longitude', fontsize=14)

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

def solve_bruteforce(W, A, N, C):
    """ 
    Function to solve the Ising model with a brute force algorithm.
    
    Input:
    -----------------------------------------
        W: numpy.array, overlap matrix.
        A: numpy.array, linear term.
        N: int, number of sites.
        C: int, number of cameras.
    
    Returns:
    -----------------------------------------
    numpy.array, status of the sites.
    
    """
    
    # time the execution
    t0 = time.time()

    # Generate all possible combinations of cameras
    if C > 0:
        comb = list(combinations(range(N), C))
    else:
        comb = [com for sub in range(N) for com in combinations(range(N), sub + 1)]    

    best_state = None
    best_energy = np.inf
    print(f"trying {len(comb)} combinations of cameras for {N} sites and {C} cameras...")
    for c in comb:
        state = np.zeros((N,))
        state[np.array(c)] = 1
        energy = np.dot(state, np.dot(W, state)) + np.sum(np.dot(A, state))
        if energy < best_energy:
            best_state = state
            best_energy = energy

    t1 = time.time()
    t_tot = t1 - t0
    print(f"best solution: {best_state} with energy {best_energy}")
    if C == 0:
        print(f"best solution: {int(np.sum(best_state))} cameras")
    print(f"total time: {t_tot} seconds")

    return best_state, t_tot


if __name__ == "__main__":
    # number of sites
    N = 15

    # number of cameras
    C = N//2 
    # C = 0

    # side of the square defining the simplified model
    a = 10

    # Generate radius of the antennas
    np.random.seed(42)
    radius = 0.5*a*(1. + np.random.rand(N))/np.sqrt(N)

    # Distribute sites uniformly but not symmetrically in the square
    m = int(np.ceil(np.log2(N)))
    sampler = Sobol(2, scramble=False, optimization='lloyd')
    sequence = sampler.random_base2(m=m)

    # Save the data of the cameras in a Pandas Dataframe

    xs = [a*x[0] for x in sequence][:N]
    ys = [a*x[1] for x in sequence][:N]

    d = {'id': np.arange(N, dtype=int), 
        'x_loc': xs, 
        'y_loc': ys, 
        'radius': radius,
        'area': np.pi*radius**2}
    data = pd.DataFrame(data=d)

    # plot the sites
    # plot_antennas(data, status=np.ones((data.shape[0],)))

    # generate the problem
    xi = 1.0
    W, A = generate_problem(data, xi)

    # solve the problem
    best_state, _ = solve_bruteforce(W, A, N, C)

    # plot the solution
    fig, ax = plt.subplots()
    plot_antennas(data, status=best_state, axes=ax)

    plt.show()

# Loading libraries to handle the problem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol
from scipy.special import binom
from itertools import combinations, islice
from multiprocessing import Pool, cpu_count
from functools import partial
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

def generate_matrices(data, xi, normalize=False):
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

def generate_problem(N, xi=0.1):
    """
    Function to generate the problem of camera placement.

    """
    # side of the square defining the simplified model
    a = 10.

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

    # generate the problem
    W, A = generate_matrices(data, xi)

    return W, A, data

def generate_combinations(N, C, chunk_size):
    if C > 0:
        comb_iter = combinations(range(N), C)
    else:
        comb_iter = (com for sub in range(N) for com in combinations(range(N), sub + 1))
    
    while True:
        chunk = list(islice(comb_iter, chunk_size))
        if not chunk:
            break
        yield chunk

def evaluate_combination(c, N, W, A):
        state = -np.ones((N,))
        state[np.array(c)] = 1
        energy = 0.5 * np.dot(state, np.dot(W, state)) + np.sum(np.dot(A, state))
        state = (state + 1) / 2
        return energy, state

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

    # If N>16 Use multiprocessing to evaluate the combinations in parallel
    # chunk size for multiprocessing
    chunk_size = 10000000
    n_chunks = binom(N, C) if C > 0 else 2**N
    n_chunks = np.ceil(n_chunks / chunk_size)

    best_state = None
    best_energy = np.inf
    if N > 16:
        # Create a partial function with N, W, and A as fixed arguments
        print(f"trying many combinations of cameras for {N} sites and {C} cameras...")
        print(f"splitting the problem in {n_chunks} chunks of {chunk_size} combinations each")
        print(f"using {cpu_count()} cores")

        partial_evaluate = partial(evaluate_combination, N=N, W=W, A=A)
        count = 1
        for comb_chunk in generate_combinations(N, C, chunk_size):
            print(f"chunk {count} of {n_chunks}...")
            with Pool(processes=cpu_count()) as pool:
                results = pool.map(partial_evaluate, comb_chunk)
            
            # Find the best combination in the current chunk
            chunk_best_energy, chunk_best_state = min(results, key=lambda x: x[0])
            
            if chunk_best_energy < best_energy:
                best_energy = chunk_best_energy
                best_state = chunk_best_state
            count += 1
    else:
        # Generate all possible combinations of cameras
        if C > 0:
            comb = list(combinations(range(N), C))
        else:
            comb = [com for sub in range(N) for com in combinations(range(N), sub + 1)]    

        print(f"trying {len(comb)} combinations of cameras for {N} sites and {C} cameras...")
        for c in comb:
            energy, state = evaluate_combination(c, N, W, A)
            if energy < best_energy:
                best_energy = energy
                best_state = state

    t1 = time.time()
    t_tot = t1 - t0
    print(f"best solution: {best_state} with energy {best_energy}")
    if C == 0:
        C = int(np.sum(best_state))
        print(f"best solution: {C} cameras")
    print(f"total time: {t_tot} seconds")

    return best_state, best_energy, C, t_tot


if __name__ == "__main__":
    # number of sites
    N_list = [32]
    # N_list = [16]
    xi = .25  # relative multiplier
    
    for N in N_list:
        # generate the problem
        W, A, data = generate_problem(N, xi)
        # number of cameras 
        for C in [N//2, 0]:
            # solve the problem
            best_state, best_energy, C, T = solve_bruteforce(W, A, N, C)

            # save solution
            fig, ax = plt.subplots()
            plot_antennas(data, status=best_state, axes=ax)
            ax.set_title(f"{N} sites, {C} cameras, energy {best_energy:.3f}, time {T:.4f} s")
            fig.savefig(f"camera_placement/results/bruteforce_N{N}_C{C}_tst.pdf", bbox_inches='tight')

    plt.show()

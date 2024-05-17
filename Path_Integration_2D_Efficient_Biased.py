# Code to optimise path-integrating representations of object maps
# Three objects arranged different ways in a room
# Neurons have code the action required to move to each object

# Pull in imports
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, grad, jit, random
import jax.nn as jnn
import optax
import math
import datetime
import os
import jax

# Parameters
num_objects = 3 # CODE IS NOT DONE FOR CHANGING THIS!
L = 4 # CODE CURRENTLY ONLY WORKS FOR EVEN!
T = L**2 # Length of each trajectory
D_Sample = 5 # How many rooms to sample
save_iter = 50000 # How often to save the weights
num_timesteps = 10000000000 # Number of gradient descent steps
learning_rate = 1e-6 # Learning Rate
random_seed = np.random.choice(10000) # Random seed
N = 60 # Number of neurons
mu_fit = 10000
mu_G = 1
mu_W = 1
mu_pos = 10000
fit_thresh = 0.0001
scale = 1
resample_trajs = 1 # How often to resample
bias_probability = 0.5

# Configure file saving
date = datetime.datetime.now()
today = date.strftime('%Y%m%d')
now = date.strftime('%H:%M:%S')

# Make sure folder is there
if not os.path.isdir(f"./data/"):
    os.mkdir(f"./data/")
if not os.path.isdir(f"data/{today}/"):
    os.mkdir(f"data/{today}/")
# Now make a folder in there for this run
savepath = f"data/{today}/{now}_{random_seed}/"
if not os.path.isdir(f"data/{today}/{now}_{random_seed}"):
    os.mkdir(f"data/{today}/{now}_{random_seed}")
    
# Save config dict
config_dict = {'num_objects':num_objects, 'L':L, 'T':T, 'D_Sample':D_Sample, 'save_iter':save_iter, 'learning_rate':learning_rate, 'random_seed':random_seed, 'N':N, 'mu_fit':mu_fit, 'mu_G':mu_G, 'mu_W':mu_W, 'mu_pos':mu_pos, 'fit_thresh':fit_thresh, 'num_timesteps':num_timesteps, 'scale':scale, 'resample_trajs':resample_trajs, 'bias_probability':bias_probability}
np.save(f"data/{today}/{now}_{random_seed}/config_dict", config_dict)

# Set up step matrix
steps = jnp.zeros([4, 2])
steps = steps.at[0,0].set(-1)
steps = steps.at[1,0].set(1)
steps = steps.at[2,1].set(-1)
steps = steps.at[3,1].set(1)

all_object_positions = np.meshgrid(np.arange(L), np.arange(L))
all_object_positions = np.stack([np.ndarray.flatten(all_object_positions[0]), np.ndarray.flatten(all_object_positions[1])]) - 0.5
all_object_positions

def generate_2D_labels(L, D_Sample):
    labels = np.zeros([2*num_objects, L, L, D_Sample])
    object_positions = np.zeros([num_objects*2, D_Sample])
    for d in range(D_Sample):
        if np.random.sample(1) < 1-bias_probability:
            object_positions[:,d] = np.random.choice(L, [num_objects*2])-0.5 # x and y position of each of the three objects
        else:
            object_positions[:2,d] = np.random.choice(L, [2])
            for object in range(1, num_objects):
                object_positions[object*2:(object+1)*2,d] = np.mod(object_positions[(object-1)*2:object*2,d] + 1, L)
            object_positions[:,d] = object_positions[:,d]-0.5

    for object in range(num_objects):
        for room in range(D_Sample):
            object_position = object_positions[object*2:(object+1)*2,room]
            for l_x in range(L):
                for l_y in range(L):
                    distance = object_position - np.array([l_x, l_y])
                    distance[distance > L/2] = distance[distance > L/2] - L
                    distance[distance < -L/2] = distance[distance < -L/2] + L
    
                    #print(object_position, np.array([l_x, l_y]), distance)
    
                    # If further to travel in x
                    if np.abs(distance[0]) > np.abs(distance[1]):
                        # And object to left, go left (0)
                        if distance[0] > 0:
                            labels[2*object, l_x, l_y, room] = -1
                        else: # Go right (1)
                            labels[2*object, l_x, l_y, room] = 1
                    # Further to go in y
                    else:
                        # And object to south, go south (2)
                        if distance[1] > 0:
                            labels[2*object+1, l_x, l_y, room] = -1
                        else: # Go north (3)
                            labels[2*object+1, l_x, l_y, room] = 1 

    return labels, object_positions

def generate_trajectories_random(T, D, L):
    # Set up actions, have to explore the room for the first L**2 - 1 timesteps, then random
    actions = np.random.choice(4, [T,D])
    actions = actions.astype(int)
    
    # Given actions, compute the positions
    positions = np.zeros([2, T, D])
    for t in range(1, T):
        positions[:,t,:] = np.mod(positions[:,t-1,:] + steps[actions[t-1,:]].T, L) # This is for periodic
    positions = positions.astype(int)

    labels, object_positions = generate_2D_labels(L, D)
    
    network_signals = np.zeros([2*num_objects,T,D])
    for d in range(D):
        for t in range(T):
            for input_dim in range(num_objects*2):
                network_signals[input_dim,t,d] = labels[input_dim, positions[0,t,d], positions[1,t,d],d]

    inputs = np.zeros([L**2*num_objects, D])
    for d in range(D):
        for object in range(num_objects):
            this_object_position = object_positions[object*2:(object+1)*2,d]
            one_hot_code = np.where(np.logical_and(all_object_positions[0] == this_object_position[0,None], all_object_positions[1] == this_object_position[1,None]))[0]
            inputs[object*L**2 + one_hot_code, d] = 1
            
    return actions, positions, network_signals, inputs, object_positions

# Define initialising functions, and losses
num_actions = 4
 
# Initialise Ws orthogonal, and others random
def initialise_weights(N, random_seed, init_scale = 0.01):
    W = jnp.zeros([num_actions, N, N+1])
    key = random.PRNGKey(random_seed)

    for n in range(num_actions):
        key, subkey = random.split(key)
        W = W.at[n,:,-1].set(random.normal(subkey, (N,))*init_scale)

        Wp = random.normal(key, (N,N))
        Wp, S, V = jnp.linalg.svd(Wp)
        W = W.at[n,:,:N].set(Wp)

    R = random.normal(key, (2*num_objects, N+1))*init_scale
    I = random.normal(key, (N, L**2*num_objects+1))*init_scale
    params = {'W':W, 'R':R, 'I':I}
    
    return params

@jit
def generate_rep(params, inputs, actions):
    # Inputs dim x traj length x rooms, actions traj length x rooms
    g = jnp.zeros([N, T, D_Sample])
    g = g.at[:,0,:].set(params['I'][:,:-1]@inputs[:,:] + params['I'][:,-1][:,None])

    # For rest we just recurrently go around.
    for t in range(1,T):
        g = g.at[:,t,:].set(jnp.einsum('ijk,ki->ji', params['W'][actions[t-1],:,:-1],g[:,t-1,:]) + params['W'][actions[t-1],:,-1].T)
        
    return g

@jit
def loss_weight(W):
    return jnp.sum(jnp.power(W[:,:-1], 2))

@jit
def loss_act(g):
    return jnp.sum(jnp.power(g, 2))

@jit
def loss_pos(g):
    return jnp.sum(jnn.relu(-g))

@jit
def loss_fit(g, R, outputs):
    preds = jnp.einsum('ij, jkl -> ikl', R[:,:-1], g) + R[:,-1][:,None,None]
    return jnp.sum(jnp.power(outputs - preds, 2))

@jit
def loss(params, inputs, network_signals, actions):
    g = generate_rep(params, inputs, actions)
    
    fitting_loss = loss_fit(g, params['R'], network_signals)  
    
    weight_loss = 0
    for i in range(2):
        weight_loss += loss_weight(params['W'][i,:,:])
        
    weight_loss += loss_weight(params['R'])
    weight_loss += loss_weight(params['I'].T)
    
    return mu_fit*jnn.relu(fitting_loss-fit_thresh) + mu_G*loss_act(g) + mu_W*weight_loss + mu_pos*loss_pos(g)

@jit
def update(params, inputs, network_signals, actions, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = value_and_grad(loss)(params, inputs, network_signals, actions)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value

key = random.PRNGKey(random_seed)
optimizer = optax.adam(learning_rate)
params = initialise_weights(N, random_seed, init_scale = scale)
opt_state = optimizer.init(params)
min_loss = np.infty

# Some mechanics
print_iter = 5000
min_this_step = 1
for step in range(num_timesteps):
    if step % resample_trajs == 0:
        actions, positions, network_signals, inputs, object_positions = generate_trajectories_random(T, D_Sample, L)
        
    params, opt_state, loss_val = update(params,  inputs, network_signals, actions, opt_state)

    if step % print_iter == 0:
        g = generate_rep(params, inputs, actions)
        
        fitting_loss = loss_fit(g, params['R'], network_signals)  
        
        weight_loss_W = 0
        for i in range(2):
            weight_loss_W += loss_weight(params['W'][i,:,:])
        weight_loss_R = loss_weight(params['R'])
        weight_loss_I = loss_weight(params['I'].T)
        
        L_a = loss_act(g)
        L_p = loss_pos(g)
        print(f"Step {step}, Loss: {loss_val:.5f}, Fit: {fitting_loss:.5f}, Act: {L_a:.5f}, Wei: {weight_loss_W:.5f}, R: {weight_loss_R:.5f}, I: {weight_loss_I:.5f}, Pos: {L_p:.5f}")

        min_this_step = 0


    if loss_val < min_loss:
        params_best = params
        min_loss = loss_val
        if min_this_step == 0:
            print(f'New min! {loss_val}')
            min_this_step = 1
            np.save(f"data/{today}/{now}_{random_seed}/params_{step}_{min_loss:.5f}", params_best)
            
np.save(f"data/{today}/{now}_{random_seed}/params_{step}_{min_loss:.5f}", params_best)

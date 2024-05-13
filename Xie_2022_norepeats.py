# Code to optimise representation for Xie et al. 2022
# Monkey is shown three cues, has to remember them, and then preform them.
# We're going to compare the delay period activity

# Pull in imports
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, grad, jit, random
import jax.nn as jnn
import optax
import math
import datetime
import os

# Set up parameters
repeat = 0 # Whether to include repeats or not
reward = 1 # Whether to predict reward or not
num_stim = 6 # How many stimuli (3 in original task)
seq_len = 2 # Seq length
save_iter = 50000 # How often to save the weights
T = 100000000 # Number of gradient descent steps
learning_rate = 1e-5 # Learning Rate
random_seed = np.random.choice(1000000) # Random seed
N = 50 # Number of neurons
mu_fit = 10000 # Fit hyperparameter
mu_G = 10 # Activity hyperparameter
mu_W = 0.1 # Weight hyperparameter
mu_R = 0.1 # Readout hyperparameter
mu_pos = 2000 # Positivity hyperparameter
fit_thresh = 0.01 # Fit threhsold
debias_outputs = 1 # Whether to remove bias from outputs
debias_inputs = 1 # Or inputs
goal_encoding = 1 # What goal encoding to use, circle, one hot, or data generated.

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
config_dict = {'repeat':repeat, 'reward':reward, 'num_stim':num_stim, 'save_iter':save_iter, 'T':T, 'learning_rate':learning_rate, 'random_seed':random_seed, 'N':N, 'mu_fit':mu_fit, 'mu_G':mu_G, 'mu_W':mu_W, 'mu_R':mu_R, 'mu_pos':mu_pos, 'fit_thresh':fit_thresh, 'debias_outputs':debias_outputs, 'debias_inputs':debias_inputs, 'goal_encoding':goal_encoding}
np.save(f"data/{today}/{now}_{random_seed}/config_dict", config_dict)

if goal_encoding == 0:
    generators = np.eye(num_stim)
elif goal_encoding == 1:
    thetas = [2*np.pi/num_stim*i for i in range(num_stim)]
    generators = np.stack([np.cos(thetas), np.sin(thetas)]).T
elif goal_encoding == 2:
    generators = np.load('./best_fitting_symmetric_generators.npy')
    num_stim = generators.shape[0]
data_dim = generators.shape[1]

if generators.shape[0] != num_stim:
    print("NOT CORRECT num_stim!!!")

def generate_sequences(generators, seq_len, repeat, debias_outputs, debias_inputs):
    if seq_len > 4:
        print("SEQUENCES LONGER THAN 4 NOT IMPLEMENTED")

    num_stim, slot_dim = generators.shape
        
    task_len = 2*seq_len + 1
    
    if repeat == 0:
        num_tasks = int(math.factorial(num_stim)/(math.factorial(num_stim-seq_len)))
    else:
        num_tasks = num_stim**seq_len
    num_states = num_tasks*task_len
    
    outputs = np.zeros([slot_dim, num_states])
    inputs = np.zeros([slot_dim, num_states])
    regression_targets = np.zeros([num_stim*seq_len, num_tasks])

    # Must be a smarter way than this dumbness...
    if seq_len == 1:
        counter = 0
        task_counter = 0
        for q in range(num_states):
            inputs[:, counter] = generators[q,:]
            counter += 1
    
            counter += 1
            outputs[:, counter] = generators[q,:]
            counter += 1
    
            regression_targets[q, task_counter] = 1
            task_counter += 1
    
    if seq_len == 2:
        counter = 0
        task_counter = 0
        for q in range(num_stim):
            for q_p in range(num_stim):
                if repeat==0 and q_p == q:
                    continue
                inputs[:, counter] = generators[q,:]
                counter += 1
                inputs[:, counter] = generators[q_p,:]
                counter += 1
    
                counter += 1
                
                outputs[:, counter] = generators[q,:]
                counter += 1
                outputs[:, counter] = generators[q_p,:]
                counter += 1
    
                regression_targets[q, task_counter] = 1
                regression_targets[q_p+num_stim, task_counter] =  1
                task_counter += 1
                
    if seq_len == 3:
        counter = 0
        task_counter = 0
        for q in range(num_stim):
            for q_p in range(num_stim):
                for q_pp in range(num_stim):
                    if repeat==0:
                        if q == q_p or q_p == q_pp or q == q_pp:
                            continue
                    inputs[:, counter] = generators[q,:]
                    counter += 1
                    inputs[:, counter] = generators[q_p,:]
                    counter += 1
                    inputs[:, counter] = generators[q_pp,:]
                    counter += 1
    
                    counter += 1
    
                    outputs[:, counter] = generators[q,:]
                    counter += 1
                    outputs[:, counter] = generators[q_p,:]
                    counter += 1
                    outputs[:, counter] = generators[q_pp,:]
                    counter += 1
    
                    regression_targets[q, task_counter] = 1
                    regression_targets[q_p+num_stim, task_counter] =  1
                    regression_targets[q_pp+2*num_stim, task_counter] =  1
                    task_counter += 1
                    
    if seq_len == 4:
        counter = 0
        task_counter = 0
        for q in range(num_stim):
            for q_p in range(num_stim):
                for q_pp in range(num_stim):
                    for q_ppp in range(num_stim):
                        if repeat==0:
                            if q == q_p or q_p == q_pp or q == q_pp or q == q_ppp or q_p == q_ppp or q_pp == q_ppp:
                                continue
                        inputs[:, counter] = generators[q,:]
                        counter += 1
                        inputs[:, counter] = generators[q_p,:]
                        counter += 1
                        inputs[:, counter] = generators[q_pp,:]
                        counter += 1
                        inputs[:, counter] = generators[q_ppp,:]
                        counter += 1
                        
                        counter += 1
        
                        outputs[:, counter] = generators[q,:]
                        counter += 1
                        outputs[:, counter] = generators[q_p,:]
                        counter += 1
                        outputs[:, counter] = generators[q_pp,:]
                        counter += 1
                        outputs[:, counter] = generators[q_ppp,:]
                        counter += 1
                            
                        regression_targets[q, task_counter] = 1
                        regression_targets[q_p+num_stim, task_counter] =  1
                        regression_targets[q_pp+2*num_stim, task_counter] =  1
                        regression_targets[q_ppp+3*num_stim, task_counter] =  1
                        task_counter += 1
    
    if debias_outputs == 1:
        if np.sum(np.abs(np.mean(outputs, axis = 1)) > 0.0001):
            print('DEBIASING OUTPUTS')
            outputs = outputs - np.mean(outputs, axis = 1)[:,None]
    elif debias_outputs != 0:
        print('SHIFTING OUTPUTS')
        outputs = outputs - debias
    
    if debias_inputs == 1:
        if np.sum(np.abs(np.mean(inputs, axis = 1)) > 0.0001):
            print('DEBIASING INPUTS')
            inputs = inputs - np.mean(inputs, axis = 1)[:,None]
    elif debias_inputs != 0:
        print('SHIFTING INPUTS')
        inputs = inputs - debias
    
    inputs = np.vstack([inputs, np.ones([1, num_states])/np.sqrt(2)])
    return inputs, outputs, regression_targets, num_states, task_len

def get_2_PCs(vecs, num_eigs = 2):
    vecs = vecs - np.mean(vecs, axis = 1)[:,None]
    covar = vecs@vecs.T
    eigvals, eigvecs = np.linalg.eig(covar)
    
    ordering = np.argsort(eigvals)[::-1]
    #print(f"Propotion of Activity Kept: {np.sum(eigvals[ordering[:2]])/np.sum(eigvals)}")
    return np.real(eigvecs[:,ordering[:num_eigs]])

@jit
def generate_rep(params, inputs):
    g = jnp.zeros([N, D])
    g = g.at[:,::task_len].set(params["I"]@inputs[:,::task_len])
    for t in range(1,task_len):
        g = g.at[:,t::task_len].set(params["I"]@inputs[:,t::task_len] + params["W"]@g[:,t-1::task_len])

    g_bias = jnp.vstack([g, jnp.ones(g.shape[1])[None,:]])
    return g_bias   
    
@jit
def loss_pos(g):
    g_neg = (g - jnp.abs(g))/2
    L_pos = -jnp.sum(g_neg)
    return L_pos

@jit
def loss_weight(W):
    return jnp.sum(jnp.power(W, 2))

@jit
def loss_weight_I(I):
    return jnp.sum(jnp.power(I[:,:-1],2))

@jit
def generate_R(g, outputs):
    mat0 = jnp.matmul(g, g.T)
    mat1 = jnp.matmul(g, outputs.T)
    mat2 = jnp.matmul(jnp.linalg.inv(mat0 + 0.0001*jnp.eye(N+1)), mat1)
    return mat2

@jit
def loss_R(R):
    R_sub = R[:-1,:]
    return jnp.sum(jnp.power(R_sub, 2))

@jit
def loss_act(g):
    return jnp.sum(jnp.power(g[:-1,:], 2))

@jit
def loss_fit(g, R, outputs):
    preds = R.T@g
    #preds_demeaned = preds - np.mean(preds, axis = 1)[:,None]
    return jnp.linalg.norm(outputs - preds) 

@jit
def loss(params, inputs, outputs):
    g = generate_rep(params, inputs)
    R = generate_R(g, outputs)
    
    return mu_fit*jnn.relu(loss_fit(g, R, outputs)-fit_thresh) + mu_G*loss_act(g) + mu_W*loss_weight(params["W"]) + mu_R*(loss_weight_I(params["I"]) + loss_R(R)) + mu_pos*loss_pos(g)

@jit
def update(params, inputs, outputs, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = value_and_grad(loss)(params, inputs, outputs)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value

inputs, outputs, regression_targets, D, task_len = generate_sequences(generators, seq_len, repeat, debias_outputs, debias_inputs)

key = random.PRNGKey(random_seed)
W = random.normal(key, (N,N))*0.1
W, S, V = jnp.linalg.svd(W)
I = random.normal(key, (N, data_dim+1))*0.1
optimizer = optax.adam(learning_rate)
# Obtain the `opt_state` that contains statistics for the optimizer.
params = {'W': W, 'I': I}
opt_state = optimizer.init(params)
params_best = params
min_loss = np.infty

for t in range(T):
    params, opt_state, loss = update(params, inputs, outputs, opt_state)

    if t % save_iter == 0:
        g = generate_rep(params, inputs)
        R = generate_R(g, outputs)

        L_f = loss_fit(g, R, outputs)
        L_a = loss_act(g)
        L_w = loss_weight(params["W"])
        L_R = loss_weight(R)
        L_p = loss_pos(g)
        L_I = loss_weight(params["I"])    

        min_this_step = 0
        print(f"Step {t}, Loss: {loss:.5f}, Fit: {L_f:.5f}, Act: {L_a:.5f}, Wei: {L_w:.5f}, R: {L_R:.5f}, Pos: {L_p:.5f}, I: {L_I:.5f}")
        
        np.save(f"data/{today}/{now}_{random_seed}/params_{t}_{min_loss:.5f}", params_best)
        
    if loss < min_loss:
        params_best = params
        min_loss = loss
        if min_this_step == 0:
            print(f'New min! {loss}')
            min_this_step = 1
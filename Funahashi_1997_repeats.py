# Code to infer best weights for Funahashi et al. 1997 task
# Monkey is shown two cues, has to remember them, and then perform them
# We're looking for delay period activity of these neurons

# Pull in imports
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, grad, jit, random
import jax.nn as jnn
import optax
import datetime
import os

# Set up parameters
repeats = 1 # Whether to include repeats or not
reward = 1 # Whether to predict reward or not
num_stim = 3 # How many stimuli (3 in original task)
save_iter = 50000 # How often to save the weights
T = 10000000000 # Number of gradient descent steps
learning_rate = 5e-5 # Learning Rate
random_seed = np.random.choice(10000) # Random seed
N = 30 # Number of neurons
mu_fit = 100000 # Fit hyperparameter
mu_G = 100 # Activity hyperparameter
mu_W = 0.1 # Weight hyperparameter
mu_R = 0.1 # Readout hyperparameter
mu_pos = 10000 # Positivity hyperparameter
fit_thresh = 0.01 # Fit threhsold

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
config_dict = {'repeats':repeats, 'reward':reward, 'num_stim':num_stim, 'save_iter':save_iter, 'T':T, 'learning_rate':learning_rate, 'random_seed':random_seed, 'N':N, 'mu_fit':mu_fit, 'mu_G':mu_G, 'mu_W':mu_W, 'mu_R':mu_R, 'mu_pos':mu_pos, 'fit_thresh':fit_thresh}
np.save(f"data/{today}/{now}_{random_seed}/config_dict", config_dict)
    
# Use these to calculate the number of trials, how long each trial is, and num conditions
num_trials = num_stim*(num_stim-(1-repeats)) 
task_len = 5 + reward
D = task_len*num_trials # Total number of conditions

# Create inputs and outputs
inputs = np.zeros([num_stim, D])
outputs = np.zeros([num_stim+reward, D])
counter = 0
for target_1 in range(num_stim):
    for target_2 in range(num_stim):
        if target_2 != target_1 or repeats:
            inputs[target_1,counter] = 1
            inputs[target_2,counter + 1] = 1

            outputs[target_1,counter+3] = 1
            outputs[target_2,counter+4] = 1
            if reward:
                outputs[num_stim,counter+5] = 1
            counter += task_len
          
# Now stack a bias on the inputs
inputs = np.vstack([inputs, np.ones([1, inputs.shape[1]])])

# Weight init function
def initialise_weights(N, random_seed, init_scale = 0.01):
    W1 = jnp.zeros([N, N+1])

    key = random.PRNGKey(random_seed)
    W = random.normal(key, (N,N))*init_scale
    W, S, V = jnp.linalg.svd(W)
    W1 = W1.at[:,:N].set(W)

    I = random.normal(key, (N, num_stim + 1))*init_scale
    W1 = W1.at[:,-1].set(random.normal(key, (N,))*init_scale)

    params = {'W':W1, 'I':I}
    
    return params

# Rep Generating Function
@jit
def generate_rep(params, inputs):
    g = jnp.zeros([N+1, D])
    g = g.at[-1:,:].set(np.ones([1,g.shape[1]]))
    g = g.at[:-1,::task_len].set(params["I"]@inputs[:,::task_len])
    for t in range(1,task_len):
        g = g.at[:-1,t::task_len].set(params["I"]@inputs[:,t::task_len] + params["W"]@g[:,t-1::task_len])
    return g

# PC extracting function
def get_PCs(vecs, num_eigs = 2):
    vecs = vecs - np.mean(vecs, axis = 1)[:,None]
    covar = vecs@vecs.T
    eigvals, eigvecs = np.linalg.eig(covar)
    
    ordering = np.argsort(eigvals)[::-1]
    #print(f"Propotion of Activity Kept: {np.sum(eigvals[ordering[:2]])/np.sum(eigvals)}")
    return np.real(eigvecs[:,ordering[:num_eigs]])

# Subspace angle calculating function
def calculate_angles(g, inputs, num_eigs = 2):
    inputs_1 = inputs[:3,0::task_len]
    inputs_2 = inputs[:3,1::task_len]
    inputs_both = [inputs_1, inputs_2]
    
    PCs = np.zeros([N, 2, num_eigs])
    for order in range(2):
        diff_vec = []
        for q in range(3):
            indices = np.where(inputs_both[order][q,:] == 1)[0]
            for index in range(len(indices)):
                for index2 in range(index):
                    if index != index2:
                        diff_vec.append(g[:,indices[index]] - g[:,indices[index2]])
        diff_vec = np.stack(diff_vec).T
        PCs[:,order,:] = get_PCs(diff_vec, num_eigs = num_eigs)      

    U, S, V = np.linalg.svd(PCs[:,0,:].T@PCs[:,1,:])
    angles = np.arccos(S)/np.pi*180
    return angles

@jit
def loss_pos(g):
    g_neg = (g[:-1,:] - jnp.abs(g[:-1,:]))/2
    L_pos = -jnp.sum(g_neg)
    return L_pos

@jit
def loss_act(g):
    return jnp.sum(jnp.power(g[:-1,:], 2))

@jit
def loss_weight(W):
    return jnp.sum(jnp.power(W[:,:-1], 2))

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
    return jnp.linalg.norm(generate_R(g, targets))**2

@jit
def loss_fit(g, R, outputs):
    preds = R.T@g
    return jnp.linalg.norm(outputs - preds)

@jit
def loss(params, inputs, outputs):
    g = generate_rep(params, inputs)
    R = generate_R(g, outputs)
    
    return mu_fit*jnn.relu(loss_fit(g, R, outputs)-fit_thresh) + mu_G*loss_act(g) + mu_W*(loss_weight(params["W"])) + mu_R*(loss_weight_I(params["I"]) + loss_weight(R)) + mu_pos*loss_pos(g)

@jit
def update(params, inputs, outputs, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = value_and_grad(loss)(params, inputs, outputs)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value
    
############
# Setup optimisation
optimizer = optax.adam(learning_rate)
params = initialise_weights(N, random_seed)
opt_state = optimizer.init(params)
min_loss = np.infty
params_best = params

for t in range(T):
    params, opt_state, loss = update(params, inputs, outputs, opt_state)

    if t % save_iter == 0:
        g = generate_rep(params, inputs)
        R = generate_R(g, outputs)
        
        angles = calculate_angles(g[:-1,2::task_len], inputs, num_eigs = 2)
        L_f = loss_fit(g, R, outputs)
        L_a = loss_act(g)
        L_w1 = loss_weight(params["W"])
        L_R = loss_weight(R)
        L_p = loss_pos(g)
        L_I = loss_weight(params["I"])    
        print(f"Step {t}, Loss: {loss:.5f}, Fit: {L_f:.5f}, Act: {L_a:.5f}, Wei: {L_w1:.5f}, R: {L_R:.5f}, Pos: {L_p:.5f}, I: {L_I:.5f}, Angle: {angles}")

        min_this_step = 0

    if loss < min_loss:
        params_best = params
        min_loss = loss
        if min_this_step == 0:
            print(f'New min! {loss}')
            np.save(f"data/{today}/{now}_{random_seed}/params_{t}_{min_loss:.5f}", params_best)
            min_this_step = 1

np.save(f"data/{today}/{now}_{random_seed}/params_{t}_{min_loss:.5f}", params_best)

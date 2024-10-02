import glob
import os
import shutil
import time
import pickle
import torch
import disentangled_rnn_params
import numpy as np
import torch.optim as optim
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr

import disentangled_rnn_utils as dru
import disentangled_rnn as dr

# get data
specs = disentangled_rnn_params.get_params()

# Create directories for storing all information about the current run
run_path, train_path, model_path, save_path, script_path, run_name = dru.make_directories(
    base_path='../Summaries_disrnn/')
# Save all python files in current directory to script directory
files = glob.iglob(os.path.join('', '*.py'))
for file in files:
    if os.path.isfile(file):
        shutil.copy2(file, os.path.join(script_path, file))

# Save parameters
np.save(os.path.join(save_path, 'params'), dict(specs))

# Create a logger to write log output to file
logger_sums = dru.make_logger(run_path, 'summaries')

inputs_all, targets_all, h_1_all, h_2_all, teacher_rnn_1, teacher_rnn_2 = dru.get_data(specs.data)
torch.save(teacher_rnn_1.state_dict(), model_path + '/teacher_rnn_1')
torch.save(teacher_rnn_2.state_dict(), model_path + '/teacher_rnn_2')
ds = dru.MyDataset(inputs_all, targets_all, h_1_all, h_2_all, batch_size=specs.data.batch_size, shuffle=True)
# train(ds, specs)

# init student
student_rnn = dr.RNN(specs.model.student_hidden_size, specs.data.input_size, 2 * specs.data.output_size,
                     activation_type=specs.model.student_rnn_type, orthog=False)
# init optimiser
optimizer = optim.Adam(student_rnn.parameters(), lr=specs.train.learning_rate)
pred_beta = specs.train.pred_beta
geco_pred = dru.GECO(specs.train.geco_pars, pred_beta)

losses, pred_losses, activity_losses, activity_l1_losses, act_nonneg_losses, weight_losses = [], [], [], [], [], []
mir_8s, num_used_latents = [], []
forward_times, backward_times = 0, 0
# et = time.time()
print_mir = True
for train_i in range(specs.train.train_steps):
    inputs, targets, h1, h2 = ds.next()
    for param in student_rnn.parameters():
        param.grad = None
    # student forward
    forward_start_time = time.time()
    targets_hat, hiddens = student_rnn(torch.from_numpy(inputs).type(torch.float32))
    # collate inputs for model
    loss, pred_loss, activity_reg, act_l1, act_nonneg, weight_l2 = dr.compute_loss(student_rnn,
                                                                                   torch.from_numpy(targets).type(
                                                                                       torch.float32),
                                                                                   targets_hat, hiddens, specs,
                                                                                   pred_beta=pred_beta)
    # backward pass
    backward_start_time = time.time()
    loss.backward()
    # clip gradients
    torch.nn.utils.clip_grad_norm_(student_rnn.parameters(), max_norm=2, norm_type=2)
    # update model parameters
    optimizer.step()
    stop_time = time.time()
    if train_i % 100 == 0:
        losses.append(loss.detach().numpy())
        pred_losses.append(pred_loss.detach().numpy())
        activity_losses.append(activity_reg.detach().numpy())
        activity_l1_losses.append(act_l1.detach().numpy())
        act_nonneg_losses.append(act_nonneg.detach().numpy())
        weight_losses.append(weight_l2.detach().numpy())

    if specs.train.use_geco:
        pred_beta = geco_pred.update(pred_loss)

    forward_times += backward_start_time - forward_start_time
    backward_times += stop_time - backward_start_time

    # periodically check modularity
    if train_i % 1000 == 0:
        # st = time.time()
        # print('time for 1000 iterations: ', st-et)
        with torch.no_grad():
            targets_hat, hiddens = student_rnn(torch.from_numpy(ds.inputs).type(torch.float32))
        inputs = ds.inputs[:2000]
        targets = ds.targets[:2000]
        h = ds.h_1[:2000]
        latents = hiddens[:2000].detach().numpy().reshape(-1, specs.model.student_hidden_size).T

        if train_i == 0:
            # corrs/mi on input / target
            corr_tar = \
                pearsonr(targets.reshape(-1, 2 * specs.data.output_size)[:, 0],
                         targets.reshape(-1, 2 * specs.data.output_size)[:, 1])[0]
            id1 = dru.histogram_discretize(targets.reshape(-1, 2 * specs.data.output_size)[:, :1].T)
            id2 = dru.histogram_discretize(targets.reshape(-1, 2 * specs.data.output_size)[:, 1:].T)
            mi_tar = mutual_info_score(id1[0, :], id2[0, :])

            # corrs/mi on input / target
            sources = h.reshape(-1, specs.data.hidden_size)
            corr_h = pearsonr(sources[:, 0], sources[:, 1])[0]
            sources = dru.discretize_binning(sources, bins='auto')
            mi_h = dru.normalized_multiinformation(sources)

        # linear_cinfom teacher--latents
        sources = h.reshape(-1, specs.data.hidden_size).T
        a = dru.compute_linear_metrics(sources.T, latents.T, 'continuous')
        mir_8 = a['linear_cinfom']
        mir_8s.append(mir_8)

        msg = "train_i={:.2f}, total_steps={:.2f}".format(train_i, train_i * specs.data.T)
        logger_sums.info(msg)
        # losses unscaled
        msg = '{:0.5f}, {:0.5f}, {:0.5f}, {:0.5f}, {:0.5f}, {:0.5f}'.format(
            loss.detach().numpy(), pred_loss.detach().numpy(), activity_reg.detach().numpy(),
            act_l1.detach().numpy(), act_nonneg.detach().numpy(), weight_l2.detach().numpy())
        logger_sums.info(msg)
        # losses scaled
        msg = '{:0.5f}, {:0.5f}, {:0.5f}, {:0.5f}, {:0.5f}, {:0.5f}'.format(
            loss.detach().numpy(), (pred_beta * pred_loss).detach().numpy(),
            specs.train.act_reg_l2 * activity_reg.detach().numpy(),
            specs.train.act_reg_l1 * act_l1.detach().numpy(),
            specs.train.act_nonneg_reg * act_nonneg.detach().numpy(),
            specs.train.weight_reg_l2 * weight_l2.detach().numpy())
        logger_sums.info(msg)
        msg = '{:0.4f}, {:0.5f}, {:0.5f}, {:0.4f}'.format(
            pred_beta.detach().numpy() if specs.train.use_geco else pred_beta, \
            forward_times / 1000, backward_times / 1000, mir_8)
        logger_sums.info(msg)

        forward_times = 0
        backward_times = 0

    if train_i % 20000 == 0 and train_i > 0:
        with open(save_path + '/mirs_all_' + str(train_i) + '.pickle', 'wb') as handle:
            pickle.dump([student_rnn,
                         (losses, pred_losses, activity_losses, activity_l1_losses, act_nonneg_losses, weight_losses),
                         (mir_8s, num_used_latents, corr_tar, mi_tar, corr_h, mi_h)],
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

        torch.save(student_rnn.state_dict(), model_path + '/student_rnn_' + str(train_i))

os._exit(0)

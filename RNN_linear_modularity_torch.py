import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import joblib as jl
import json
import os
from datetime import datetime
##Define RNN


class RNN(nn.Module):

    def __init__(self,
                 batch_size,
                 hidden_size,
                 input_size,
                 output_size,
                 activation_type='relu',
                 orthog=True,
                 input=True):
        super(RNN, self).__init__()
        self.input = input
        self.hidden_size = hidden_size
        self.activation_type = activation_type
        self.w_rec = nn.Linear(hidden_size, hidden_size, bias=True)
        self.x0 = torch.zeros(batch_size, self.hidden_size)
        self.w_out = nn.Linear(hidden_size, output_size, bias=True)
        self.input = input
        if self.input:
            self.w_in = nn.Linear(input_size, hidden_size, bias=True)
        if orthog:
            _ = self.apply(self._init_weights)
        else:
            _ = self.apply(self._init_weights_uniform)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_weights_uniform(self, module):
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, 0, .01)
            if module.bias is not None:
                nn.init.uniform_(module.weight, 0, .01)

    def forward(self, inputs=None):
        B, T, _ = inputs.shape

        outputs = []
        hiddens = []
        h = self.x0
        for t in range(T):
            if self.input:
                h = self.w_rec(h) + self.w_in(inputs[:, t, ...])
            else:
                h = self.w_rec(h)
            h = self.activation(self.norm(h))
            outputs.append(self.w_out(h))
            hiddens.append(h)
        return torch.stack(outputs, axis=1), torch.stack(hiddens, axis=1)

    def activation(self, x):
        if self.activation_type == 'relu':
            return nn.ReLU()(x)
        elif self.activation_type == 'tanh':
            return torch.tanh(x)
        elif self.activation_type == 'none':
            return x
        else:
            raise ValueError('incorrect activation type')

    def norm(self, x):
        #return x / np.sqrt(np.sum(x**2) + 1e-8)
        return x


def periodic_output(batch_size, T, input_size, freq, t_bin=0.1):
    freq = np.ones(input_size) * freq
    d = np.cos(freq[:, None] @ np.arange(0, T, t_bin)[:, None].T).T

    return np.tile(d, (batch_size, 1, 1))


def periodic_pulse(batch_size, T, input_size, freq, t_bin=0.1, cos=True):
    arr = np.zeros(len(np.arange(0, T, t_bin)))

    arr = np.zeros_like(np.arange(0, T, 0.1))
    for n in range(int(freq * T / 2 / np.pi) + 1):
        if int(2 * np.pi * n / freq / t_bin) < len(arr):
            arr[int(2 * np.pi * n / freq / t_bin)] = 1

    return np.tile(arr[:, None], (batch_size, 1, 1))


def freq_mixing_terms(batch_size, T, input_size, freqs, trig, t_bin=0.1):
    freq1 = np.ones(input_size) * freqs[0]
    freq2 = np.ones(input_size) * freqs[1]
    if trig == "cos":
        d1 = np.cos(
            (freq1 + freq2)[:, None] @ np.arange(0, T, t_bin)[:, None].T)
        d2 = np.cos((freq1[:, None] - freq2[:, None]) @ np.arange(
            0, T, t_bin)[:, None].T)
    elif trig == 'sin':
        d1 = np.sin((freq1[:, None] + freq2[:, None]) @ np.arange(
            0, T, t_bin)[:, None].T)
        d2 = np.sin((freq1[:, None] - freq2[:, None]) @ np.arange(
            0, T, t_bin)[:, None].T)
    d = np.stack([d1, d2], axis=-1)
    return np.tile(d, (batch_size, 1, 1))


def spectral_entropy(signal):
    spectrum = np.fft.fft(signal, axis=0)
    N = len(spectrum)
    power = 1 / N * (np.absolute(spectrum)**2)
    p = power / (np.sum(power) + 1e-8)
    return sum(-p * np.log(p))


def rel_fft_power(signal, freq, d=0.1, delta=0.5):
    spectrum = np.fft.fft(signal, axis=0)
    freq_list = np.fft.fftfreq(len(spectrum), d=d) * 2 * np.pi
    spectrum = spectrum[freq_list >= 0]
    freq_list = freq_list[freq_list >= 0]
    magnitude = (np.absolute(spectrum) + 1e-8)
    freq_ind = np.where(abs(freq_list - freq) <= delta)[0]

    return sum(magnitude[freq_ind]) / sum(magnitude)


def rel_fft_magnitude(signal, freq, d=0.1, delta=0.5):
    spectrum = abs(np.real(np.fft.fft(signal, axis=0)))
    freq_list = np.fft.fftfreq(len(spectrum), d=d) * 2 * np.pi
    spectrum = spectrum[freq_list >= 0]
    freq_list = freq_list[freq_list >= 0]
    magnitude = (spectrum + 1e-8)
    freq_ind = np.where(abs(freq_list - freq) <= delta)[0]

    return sum(magnitude[freq_ind]) / sum(magnitude)


def MI(nm_matrix):
    n, m = nm_matrix.shape
    a = 0
    for i in range(n):
        a += np.max(nm_matrix[i])

    return (a / np.sum(nm_matrix) - 1 / m) / (1 - 1 / m)


def get_zero_input(batch_size, T, input_size):
    return (np.zeros((batch_size, T, input_size)))


def compute_loss(model, y, y_hat, h, pos=True):
    pred_loss = torch.mean((y - y_hat)**2)
    activity_reg = torch.mean(h**2)
    positivity = torch.mean(0.5 * (h - abs(h))**2)
    weight_l2 = 0
    for name, p in model.named_parameters():
        if 'weight' in name:
            weight_l2 += (p**2).sum()

    if not pos:
        return pred_loss + (5e-1 * activity_reg + 1e-2 *
                            weight_l2), pred_loss, activity_reg, weight_l2
    else:
        #return pred_loss * 2 + 5e-1 * activity_reg + 1e-2 * weight_l2 + positivity * 2, pred_loss, activity_reg, weight_l2, positivity
        return pred_loss + 5e-1 * activity_reg + 2e-2 * weight_l2 + positivity * 5, pred_loss, activity_reg, weight_l2, positivity  ## For delta input


def train(net, x, y, train_steps, lr, batch_size=1, pos_reg=True):
    losses, pred_losses, activity_losses, weight_losses, pos_losses = [], [], [], [],[]
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for i in range(train_steps):
        for param in net.parameters():
            param.grad = None
        # student forward
        forward_start_time = time.time()
        targets_hat, hiddens = net(
            x)  #net(torch.from_numpy(x).type(torch.float32))
        # collate inputs for model
        loss, pred_loss, activity_reg, weight_l2, pos = compute_loss(
            net, y, targets_hat, hiddens, pos_reg)
        # backward pass
        backward_start_time = time.time()
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(net.parameters(),
                                       max_norm=2,
                                       norm_type=2)
        # update model parameters

        optimizer.step()
        stop_time = time.time()
        if i % 100 == 0:
            losses.append(loss.detach().numpy())
            pred_losses.append(pred_loss.detach().numpy())
            activity_losses.append(activity_reg.detach().numpy())
            weight_losses.append(weight_l2.detach().numpy())
            pos_losses.append(pos.detach().numpy())
        if i % 1000 == 0:
            print(
                f"iter {i}: loss: {loss.item():.4f}, pred_loss: {pred_loss.item():.4f}, activity_loss: {activity_reg.item():.4f}, weight_loss:{weight_l2.item():.4f}, positivity_loss:{pos.item():.4f}"
            )
    if pos_reg:
        return net, (losses, pred_losses, activity_losses, weight_losses, pos)
    else:
        return net, (losses, pred_losses, activity_losses, weight_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--freqs', nargs='+', type=float, required=False)
    parser.add_argument('--max-iters', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=7.5e-4)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--irrational', action='store_true')
    parser.add_argument('--rational', action='store_true')

    args = parser.parse_args()

    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    exp_logdir = os.path.join(args.savedir, timestamp)
    os.mkdir(exp_logdir)

    T = 20
    batch_size = 1
    hidden_size = args.hidden_size
    input_size = 1
    output_size = 1
    if args.irrational:
        while True:
            kf1 = np.random.uniform(0.5, 4) * np.pi
            kf2 = np.sqrt(np.random.uniform(1, 100))
            if abs(kf1 - kf2) > 1:
                break
    elif args.rational:
        while True:
            kf1 = np.random.randint(1, 20)
            kf2 = np.random.randint(1, 20)
            if abs(kf1 - kf2) > 1 and max([kf1, kf2]) / min([kf1, kf2]) != int(
                    max([kf1, kf2]) / min([kf1, kf2])):
                break
    else:
        kf1, kf2 = args.freqs

    rnn = RNN(batch_size,
              hidden_size,
              2,
              2,
              activation_type='none',
              orthog=True,
              input=True)

    output1 = periodic_output(batch_size, T, 1, kf1)
    output2 = periodic_output(batch_size, T, 1, kf2)

    input1 = periodic_pulse(batch_size, T, 1, kf1)
    input2 = periodic_pulse(batch_size, T, 1, kf2)

    inputs = torch.from_numpy(np.concatenate([input1, input2],
                                             axis=-1)).type(torch.float32)
    targets = torch.from_numpy(np.concatenate([output1, output2],
                                              axis=-1)).type(torch.float32)

    rnn_trained, (losses, pred_losses, activity_losses, weight_losses,
                  pos) = train(rnn,
                               inputs,
                               targets,
                               args.max_iters,
                               args.lr,
                               pos_reg=True)

    with torch.no_grad():
        targets_hat, hs = rnn_trained(inputs)

    fft_results = []
    entropy = []
    rel_mag = []
    rel_power = []

    for i in range(hidden_size):
        fft_result = np.fft.fft(hs[0, :, i] - torch.mean(hs[0, :, i]).numpy(),
                                axis=0)
        frequencies = np.fft.fftfreq(fft_result.shape[0], d=0.1) * 2 * np.pi
        fft_results.append(fft_result)
        entropy.append(
            spectral_entropy(hs[0, :, i] - torch.mean(hs[0, :, i]).numpy()))
        rel_power_temp = []
        rel_mag_temp = []
        rel_power_temp.append(
            rel_fft_power(hs[0, :, i] - torch.mean(hs[0, :, i]).numpy(),
                          kf1,
                          delta=0.2))
        rel_power_temp.append(
            rel_fft_power(hs[0, :, i] - torch.mean(hs[0, :, i]).numpy(),
                          kf2,
                          delta=0.2))
        rel_mag_temp.append(
            rel_fft_magnitude(hs[0, :, i] - torch.mean(hs[0, :, i]).numpy(),
                              kf1,
                              delta=0.2))
        rel_mag_temp.append(
            rel_fft_magnitude(hs[0, :, i] - torch.mean(hs[0, :, i]).numpy(),
                              kf2,
                              delta=0.2))
        rel_mag.append(rel_mag_temp)
        rel_power.append(rel_power_temp)

    result_dict = {
        'hs': hs.numpy(),
        'y_pred': targets_hat.numpy(),
        'y': targets.numpy(),
        'fft': fft_results,
        'fft_freq': frequencies,
        'rel_mag': rel_mag,
        'rel_power': rel_power,
        'entropy': entropy,
        'key_freq': [kf1, kf2],
        'MI_mag': MI(np.array(rel_mag)),
        'MI_power': MI(np.array(rel_power)),
        'total_loss': losses[-1],
        'pred_loss': pred_losses[-1],
        'acitivity_loss': activity_losses[-1],
        'weight_loss': weight_losses[-1]
    }
    json.dump(vars(args), open(os.path.join(exp_logdir, 'args.json'), 'w'))
    jl.dump(result_dict, os.path.join(exp_logdir, 'result_dict.jl'))
    torch.save(rnn_trained.state_dict(),
               os.path.join(exp_logdir, 'rnn_trained.pth'))
    jl.dump(
        {
            'loss': losses,
            'pred_loss': pred_losses,
            'activity_loss': activity_losses,
            'weight_loss': weight_losses
        }, os.path.join(exp_logdir, 'rnn_loss_history.jl'))

import torch
import torch.nn as nn


# get RNN model class and plot data

class RNN(nn.Module):
    def __init__(self, hidden_size, input_size, output_size, activation_type='relu', orthog=True):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.activation_type = activation_type
        self.output_size = output_size
        self.w_rec = nn.Linear(hidden_size, hidden_size, bias=True)
        self.w_in = nn.Linear(input_size, hidden_size, bias=True)
        self.w_out = nn.Linear(hidden_size, output_size, bias=True)
        self.shift = torch.zeros(output_size)
        self.scale = torch.ones(output_size)
        if orthog:
            _ = self.apply(self._init_weights)
        else:
            torch.nn.init.uniform_(self.w_in.weight, a=0.0, b=1.0)
            self.w_in.weight.data = self.w_in.weight.data / (torch.sum(self.w_in.weight ** 2) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        B, T, _ = inputs.shape
        h = torch.zeros(B, self.hidden_size)
        outputs = []
        hiddens = []
        for t in range(T):
            h = self.rnn_step(h, inputs[:, t, ...])
            outputs.append(self.scale * self.w_out(h) - self.shift)
            hiddens.append(h)
        return torch.stack(outputs, axis=1), torch.stack(hiddens, axis=1)

    def rnn_step(self, h, inp):
        h_ = self.w_rec(h) + self.w_in(inp)
        return self.activation(h_)

    def activation(self, x):
        if self.activation_type == 'relu':
            return nn.ReLU()(x)
        elif self.activation_type[:5] == 'relu_':
            return torch.clamp(x, min=0, max=float(self.activation_type[5:]))
        elif self.activation_type[:4] == 'relu':
            return nn.ReLU()(x + float(self.activation_type[4:])) - float(self.activation_type[4:])
        elif self.activation_type[:2] == '+-':
            return torch.clamp(x, min=-float(self.activation_type[2:]), max=float(self.activation_type[2:]))
        elif self.activation_type == 'tanh':
            return torch.tanh(x)
        else:
            raise ValueError('incorrect activation type')

    def limit_outputs(self, inputs, low=-1.0, high=1.0):
        with torch.no_grad():
            t, _ = self.forward(inputs)
            t = t.reshape(-1, self.output_size)
            # rescale
            self.scale = (high - low) / (t.max(dim=0)[0] - t.min(dim=0)[0])
            t = t * self.scale
            self.shift = t.min(dim=0)[0] - low


def compute_loss(model, y, y_hat, h, specs, pred_beta=1.0):
    pred_loss = torch.mean((y - y_hat) ** 2)
    act_l2 = torch.mean(h ** 2)
    act_l1 = torch.mean(torch.abs(h))
    act_nonneg = torch.mean(torch.nn.ReLU()(-h))
    weight_l2 = 0
    for name, p in model.named_parameters():
        if 'weight' in name:
            weight_l2 += (p ** 2).mean()

    loss = (pred_beta * pred_loss +
            specs.train.act_reg_l2 * act_l2 +
            specs.train.weight_reg_l2 * weight_l2 +
            specs.train.act_reg_l1 * act_l1 +
            specs.train.act_nonneg_reg * act_nonneg)
    return loss, pred_loss, act_l2, act_l1, act_nonneg, weight_l2

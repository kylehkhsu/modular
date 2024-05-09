"""
Minimal non-linear teacher-student experiment script
"""

import argparse
import os

from datetime import datetime
import joblib
import json

import jax.numpy as jnp
import jax
import equinox as eqx
import optax
import tqdm
import pprint
import einops

from model import ShootingModel, ModularTeacher
import utils

SHOOTING_LAMBDAS_TEACHER = {
    'target': 1e+7,
    'transition': 1,
    'activation_energy': 1,
    'activation_positivity': 1,
    'readout_energy': 1,
    'transition_energy': 1
}

SHOOTING_LAMBDAS_STUDENT = {
    'target': 1000,
    'transition': 1,
    'activation_energy': 1,
    'activation_positivity': 1000,
    'readout_energy': 1,
    'transition_energy': 1
}


@eqx.filter_jit
def train_step(model, optimizer_state, optimizer, y, key, **kwargs):

    (_, aux), grad = eqx.filter_value_and_grad(model.loss,
                                               has_aux=True)(model, y, key,
                                                             **kwargs)
    update, optimizer_state = optimizer.update(grad, optimizer_state, model)
    model = eqx.apply_updates(model, update)
    return model, optimizer_state, aux


@eqx.filter_jit
def ts_train_step(teacher, student, optimizer_state, optimizer, key, **kwargs):

    x0 = student.sigma * jax.random.normal(key, (student.K, student.D))
    ## compute forward pass from teacher; get {'x', 'y'}
    teacher_pred = teacher.forward(x0)
    teacher_pred = jnp.stack(
        [teacher_pred[k][1].squeeze() for k in range(len(teacher))]).T
    ## concat predicted output from teacher

    student_x0 = jnp.reshape(x0, (student.K * student.D, ))

    (_, aux), grad = eqx.filter_value_and_grad(student.loss,
                                               has_aux=True)(student,
                                                             student_x0,
                                                             teacher_pred, key,
                                                             **kwargs)
    update, optimizer_state = optimizer.update(grad, optimizer_state, student)
    student = eqx.apply_updates(student, update)
    return student, optimizer_state, aux


def generate_data(D, freqs, sampling_rate, duration, key, noise=False):
    time_bins = jnp.arange(0, duration, sampling_rate)
    omega_t = einops.rearrange(time_bins, 't->t ()') * jnp.array(freqs)
    data = jnp.sin(omega_t)
    if noise:
        data += jax.random.normal(key, data.shape) * 0.02

    x_init = jax.random.uniform(key,
                                (int(duration / sampling_rate), len(freqs), D),
                                minval=0,
                                maxval=1)
    return data, x_init


def experiment(**kwargs):

    if not os.path.isdir(kwargs['logdir']):
        os.makedirs(kwargs['logdir'])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_logdir = os.path.join(kwargs['logdir'], timestamp)
    os.mkdir(exp_logdir)

    key = jax.random.PRNGKey(kwargs['seed'])
    ## Generate oscillatory data and train teacher on this first
    y_true, _ = generate_data(kwargs['D'], kwargs['freqs'], 0.05, 10, key)

    ## initialise teacher and fit with the oscillatory output
    teacher = ModularTeacher([
        ShootingModel(K=1,
                      D=kwargs['D'],
                      T=kwargs['T'],
                      key=key,
                      non_linearity=kwargs['non_linearity'],
                      output_non_linearity=None,
                      orthogonal_init=True,
                      lambdas=SHOOTING_LAMBDAS_TEACHER,
                      sigma=kwargs['sigma']) for _ in range(kwargs['K'])
    ])

    teacher_aux = {k: None for k in range(kwargs['K'])}
    loss_history = {k: {} for k in range(kwargs['K'])}
    for i, single_module in enumerate(teacher.modules):

        optimizer = optax.adam(kwargs['lr_teacher'])
        optimizer_state = optimizer.init(single_module)
        for n in tqdm.trange(kwargs['max_iters_teacher']):
            key, subkey = jax.random.split(key)
            single_module, optimizer_state, aux = train_step(single_module,
                                                             optimizer_state,
                                                             optimizer,
                                                             y_true[:,
                                                                    i][:,
                                                                       None],
                                                             key=subkey)
            if n % kwargs['log_frequency'] == 0:
                for k, v, in aux['losses'].items():
                    if k in loss_history[i].keys():
                        loss_history[i][k].append(v.item())
                    else:
                        loss_history[i][k] = [v.item()]
            if n % kwargs['print_frequency'] == 0:
                pprint.pprint(aux['losses'])
        teacher_aux[i] = aux
        teacher.modules[i] = single_module

    joblib.dump(teacher_aux, os.path.join(exp_logdir, 'final_teacher_aux.jl'))
    joblib.dump(loss_history, os.path.join(exp_logdir, 'loss_history.jl'))

    ## Now train student on it

    student_x0 = jnp.concat(
        [teacher.modules[i].x0 for i in range(kwargs['K'])])

    student = ShootingModel(K=kwargs['K'],
                            D=kwargs['D'],
                            T=kwargs['T'],
                            key=key,
                            x0=student_x0,
                            non_linearity=kwargs['non_linearity'],
                            output_non_linearity=None,
                            orthogonal_init=True,
                            lambdas=SHOOTING_LAMBDAS_STUDENT,
                            sigma=kwargs['sigma'])

    ## training loop: initialise x0, get y from teacher and train student with the x0, y pair

    optimizer = optax.adam(kwargs['lr_student'])
    optimizer_state = optimizer.init(student)
    student_loss_history = {}

    for n in tqdm.trange(kwargs['max_iters_student']):
        key, subkey = jax.random.split(key)
        ## initialise random x0
        """
        student, optimizer_state, aux = ts_train_step(teacher,
                                                      student,
                                                      optimizer_state,
                                                      optimizer,
                                                      key=subkey)
        """

        student, optimizer_state, aux = train_step(student,
                                                   optimizer_state,
                                                   optimizer,
                                                   y_true,
                                                   key=subkey)
        if n % kwargs['log_frequency'] == 0:
            for k, v, in aux['losses'].items():
                if k in student_loss_history.keys():
                    student_loss_history[k].append(v.item())
                else:
                    student_loss_history[k] = [v.item()]
        if n % kwargs['print_frequency'] == 0:
            pprint.pprint(aux['losses'])
    teacher_out = teacher.forward(aux['x'][0, :].reshape(student.K, student.D))
    teacher_activity = jnp.concat(
        [teacher_out[k][0].squeeze() for k in range(len(teacher))], axis=-1)

    json.dump(kwargs, open(os.path.join(exp_logdir, 'args.json'), 'w'))
    joblib.dump(student_loss_history,
                os.path.join(exp_logdir, 'student_loss_history.jl'))
    joblib.dump(aux, os.path.join(exp_logdir, 'final_student_aux.jl'))
    joblib.dump({"x": teacher_activity},
                os.path.join(exp_logdir, 'teacher_activity.jl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Args for non-linear RNN teacher student setup")
    ### Model params
    parser.add_argument("--K", type=int, default=2, help="Number of modules")
    parser.add_argument("--D",
                        type=int,
                        default=5,
                        help="Latent dimensionality")
    parser.add_argument("--T", type=int, default=200, help="Sequence length")
    parser.add_argument("--sigma",
                        type=float,
                        default=0.001,
                        help="Scaling for model weight initalisation")
    parser.add_argument("--non-linearity",
                        type=str,
                        default="tanh",
                        help="nonlinearity type")
    parser.add_argument("--freqs", type=int, nargs='+')

    ### Training params
    parser.add_argument("--max-iters-teacher",
                        type=int,
                        default=1,
                        help="Update iters")
    parser.add_argument("--max-iters-student",
                        type=int,
                        default=300000,
                        help="Update iters")
    parser.add_argument("--lr-teacher",
                        type=float,
                        default=1e-3,
                        help="learning rate")
    parser.add_argument("--lr-student",
                        type=float,
                        default=1e-3,
                        help="learning rate")
    parser.add_argument("--seed",
                        type=int,
                        default=111,
                        help="Initial key for jax.random")
    ##TODO maybe we need to change the LAMBDA params

    ### Log params
    parser.add_argument("--logdir",
                        type=str,
                        default="nonlinear_logs_oscillatory",
                        help="Base logdir")
    parser.add_argument("--log-frequency",
                        type=int,
                        default=1000,
                        help="Loss logging frequency")
    parser.add_argument("--print-frequency",
                        type=int,
                        default=5000,
                        help="Printing frequency")

    args = parser.parse_args()

    experiment(**vars(args))

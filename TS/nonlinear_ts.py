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

SHOOTING_LAMBDAS = {
    'target': 10000,
    'transition': 1,
    'activation_energy': 1,
    'activation_positivity': 1,
    'readout_energy': 1,
    'transition_energy': 1
}


@eqx.filter_jit
def train_step(teacher, student, optimizer_state, optimizer, key, **kwargs):

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


def experiment(**kwargs):

    key = jax.random.PRNGKey(kwargs['seed'])

    ## initialise teacher
    teacher = ModularTeacher([
        ShootingModel(K=1,
                      D=kwargs['D'],
                      T=kwargs['T'],
                      key=key,
                      non_linearity=kwargs['non_linearity'],
                      lambdas=SHOOTING_LAMBDAS,
                      sigma=kwargs['sigma']) for _ in range(kwargs['K'])
    ])
    ## initialise student

    student = ShootingModel(K=kwargs['K'],
                            D=kwargs['D'],
                            T=kwargs['T'],
                            key=key,
                            non_linearity=kwargs['non_linearity'],
                            lambdas=SHOOTING_LAMBDAS,
                            sigma=kwargs['sigma'])

    ## training loop: initialise x0, get y from teacher and train student with the x0, y pair

    optimizer = optax.adam(kwargs['lr'])
    optimizer_state = optimizer.init(student)
    loss_history = {}

    if not os.path.isdir(kwargs['logdir']):
        os.makedirs(kwargs['logdir'])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_logdir = os.path.join(kwargs['logdir'], timestamp)
    os.mkdir(exp_logdir)

    for n in tqdm.trange(kwargs['max_iters']):
        key, subkey = jax.random.split(key)
        ## initialise random x0
        student, optimizer_state, aux = train_step(teacher,
                                                   student,
                                                   optimizer_state,
                                                   optimizer,
                                                   key=subkey)
        if n % kwargs['log_frequency'] == 0:
            for k, v, in aux['losses'].items():
                if k in loss_history.keys():
                    loss_history[k].append(v.item())
                else:
                    loss_history[k] = [v.item()]
        if n % kwargs['print_frequency'] == 0:
            pprint.pprint(aux['losses'])
    teacher_out = teacher.forward(aux['x'][0, :].reshape(student.K, student.D))
    teacher_activity = jnp.concat(
        [teacher_out[k][0].squeeze() for k in range(len(teacher))], axis=-1)

    json.dump(kwargs, open(os.path.join(exp_logdir, 'args.json'), 'w'))
    joblib.dump(loss_history, os.path.join(exp_logdir, 'loss_history.jl'))
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
    parser.add_argument("--T", type=int, default=100, help="Sequence length")
    parser.add_argument("--sigma",
                        type=float,
                        default=1,
                        help="Scaling for model weight initalisation")
    parser.add_argument("--non-linearity",
                        type=str,
                        default="tanh",
                        help="nonlinearity type")

    ### Training params
    parser.add_argument("--max-iters",
                        type=int,
                        default=50000,
                        help="Update iters")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Initial key for jax.random")
    ##TODO maybe we need to change the LAMBDA params

    ### Log params
    parser.add_argument("--logdir",
                        type=str,
                        default="nonlinear_logs",
                        help="Base logdir")
    parser.add_argument("--log-frequency",
                        type=int,
                        default=100,
                        help="Loss logging frequency")
    parser.add_argument("--print-frequency",
                        type=int,
                        default=1000,
                        help="Printing frequency")

    args = parser.parse_args()

    experiment(**vars(args))

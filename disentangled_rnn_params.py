from disentangled_rnn_utils import DotDict as Dd


def get_params():
    data = Dd()
    model = Dd()
    train = Dd()

    data.batch_size = 128
    data.T = 20
    data.hidden_size = 2
    data.input_size = 2
    data.output_size = 1
    data.condition_type = "oversample_diagonals"  # corner_cut_both  oversample_diagonal_2 oversample_diagonals
    data.teacher_rnn_type = 'tanh'
    data.every_n = 1
    data.low = -0.5
    data.high = 0.5
    data.gradient = 1.0
    data.intercept_y = -0.0
    data.intercept_x = -0.1
    data.resample_input = True
    data.resample_output = False
    data.xl = -0.2
    data.xh = 0.6
    data.yl = -0.2
    data.yh = 0.6
    data.prop_batch_oversample = 1.0
    data.n_data = 10000
    data.targets_scale = 0.7

    model.student_hidden_size = 64
    model.student_rnn_type = 'relu'

    train.train_steps = 60001
    train.pred_beta = 10.0
    train.learning_rate = 2e-3
    train.act_reg_l2 = 1e0
    train.act_reg_l1 = 0.0  # 2e-2
    train.act_nonneg_reg = 0.0  # 1.0
    train.weight_reg_l2 = 1e0
    train.use_geco = False,
    train.geco_pars = {'threshold': 3e-4,
                       'alpha': 0.9,
                       'gamma': 2e-1,
                       }

    return Dd({'data': data,
               'model': model,
               'train': train})

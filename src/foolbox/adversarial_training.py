from src import foolbox as fb
from src.foolbox.test_attack import criterion_dict, attack_dict
from seml.utils import merge_dicts
import torch

class AttackModel:

    def __init__(self, model, attack_name, attack_loss, bounds, attack_params, criterion_params, attack_kwargs=None, conf_cal_adv_training=None):
        self.fb_model = fb.PyTorchModel(model, bounds)
        self.attack_kwargs = {}
        if attack_kwargs is not None:
            self.attack_kwargs = attack_kwargs
        self.criterion_params = criterion_params
        self.attack_name = attack_name
        self.attack_loss = attack_loss
        self.attack_params = attack_params

        self.conf_cal_adv_training = conf_cal_adv_training

    def attack(self, input_data, labels):
        criterion = criterion_dict[self.attack_loss](labels, **self.criterion_params)
        atk_fun = attack_dict[self.attack_name](self.attack_kwargs)
        _, pert_X, measure = atk_fun(self.fb_model, input_data, criterion, **self.attack_params)
        if self.conf_cal_adv_training is None:
            calibrated_labels = labels
        else:
            if not self.conf_cal_adv_training['do_calibration']:
                calibrated_labels = labels
            else:
                calibrated_labels = self.get_conf_cali_labels(input_data, pert_X, labels)

        return pert_X[0], calibrated_labels

    def get_conf_cali_labels(self, input_data, pert_X, labels):
        dev = input_data.device
        nr_classes = self.conf_cal_adv_training['nr_classes']
        nr_points = input_data.shape[0]
        eps_max = torch.tensor(self.conf_cal_adv_training['max_dist']).to(dev)
        rho = torch.tensor(self.conf_cal_adv_training['rho']).to(dev)
        if self.conf_cal_adv_training['norm'] == 'L2':
            dist = torch.norm(torch.reshape((input_data - pert_X[0]), (nr_points, -1)), 2, -1)
            one_t = torch.tensor([1.0]).to(dev)
            lambda_dist = torch.pow((one_t -torch.min(one_t, dist/eps_max)), rho)
            # mix onehot with uniform
            uni_labels = (1.0/nr_classes)*torch.ones(nr_points, nr_classes).to(dev)
            hot_labels = torch.zeros(nr_points, nr_classes).to(dev)
            hot_labels[range(nr_points), labels] = 1.0
            calibrated_labels = torch.mm(torch.diag(lambda_dist), hot_labels) + torch.mm(torch.diag(1-lambda_dist), uni_labels)
            print('TODO: adapt loss functions (label shape: nr_points x nr_classes) + model specific stuff (soft)')
        else:
            print('This calibration norm is not implemented.')
            calibrated_labels = labels
        return calibrated_labels


def parse_adversarial_training_config(config):
    atk_params_in = None
    atk_params_out = None
    if config is not None:
        atk_params_shared = {}
        if 'shared' in config:
            if 'bounds' in config['shared']:
                bounds = config['shared']['bounds']
                config['shared']['bounds'] = [float(bounds[0]), float(bounds[1])]
            atk_params_shared.update(config['shared'])
        if 'in_data' in config:
            atk_params_in = merge_dicts(atk_params_shared, config['in_data'])
        if 'out_data' in config:
            atk_params_out = merge_dicts(atk_params_shared, config['out_data'])
    return atk_params_in, atk_params_out


def parse_config_and_create_attacks(config, model):
    if config is None:  # no adversarial training config
        return None, None
    atk_params_in, atk_params_out = parse_adversarial_training_config(config)

    if 'conf_cal_adv_training' in config:
        conf_cal_adv_training = config['conf_cal_adv_training']
    else:
        conf_cal_adv_training = None

    in_atk = None
    if atk_params_in is not None:
        in_atk = AttackModel(model, **atk_params_in, conf_cal_adv_training=conf_cal_adv_training)
    out_atk = None
    if atk_params_out is not None:
        out_atk = AttackModel(model, **atk_params_out, conf_cal_adv_training=conf_cal_adv_training)
    return in_atk, out_atk
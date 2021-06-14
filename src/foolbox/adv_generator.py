import foolbox.run_attack
import foolbox as fb
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
import foolbox as fb
from foolbox.criteria import Misclassification, Alpha0, DiffEntropy, DistUncertainty
import torchvision.models as models
import torch



# use it to generate adverarials for training
class Adv_Generator():
    def __init__(self, attack_type, attack_data, criterion_type, loss_name, punished_measure, eps, bounds, training_procedure, norm_adv, rho, nr_classes, threshold):
        self.attack_type = attack_type
        self.attack_data = attack_data # start of the attack, 'in' our 'out'
        self.criterion_type = criterion_type
        self.loss_name = loss_name
        self.punished_measure = punished_measure
        self.eps = eps
        self.bounds = bounds
        self.training_procedure = training_procedure # 'adv',  'confadv'
        self.norm_adv = norm_adv
        self.rho = rho
        self.nr_classes = nr_classes
        self.threshold = threshold


        # noise attacks can be used for randomized smoothing training
        self.attack_dict = { 'FGSM_Linf': fa.FGSM(),
                        'FGSM_L2': fa.FGM(), 
                        'PGD_Linf': fa.LinfProjectedGradientDescentAttack(),
                        'PGD_L2': fa.L2ProjectedGradientDescentAttack(),
                        'Noise_Gaussian_L2': fa.L2AdditiveGaussianNoiseAttack(), 
                        'Noise_Uniform_Linf': fa.LinfAdditiveUniformNoiseAttack(),
                        'Noise_Uniform_L2': fa.L2AdditiveUniformNoiseAttack(),
                        'CW_L2': fa.L2CarliniWagnerAttack(),
        }


        self.attack = self.attack_dict[attack_type]

    # model changes during training, thus required as argument
    def get_adv_batch(self, model, X, Y):
        fmodel = fb.PyTorchModel(model, bounds=self.bounds)

        Xp, Yp = ep.astensors(X, Y)

        if self.criterion_type == 'Misclassification':
            criterion = Misclassification(Yp)
        elif self.criterion_type == 'Alpha0':
            criterion = Alpha0(Yp, self.attack_data, self.threshold)
        elif self.criterion_type == 'DiffEntropy':
            criterion = DiffEntropy(Yp, self.attack_data, self.threshold)
        elif self.criterion_type == 'DistUncertainty':
            criterion = DistUncertainty(Yp, self.attack_data, self.threshold)
        else:
            print('Choosen criterion is not available for in data.')
            return None
        
        d1, d2, measure = self.attack(fmodel, Xp, criterion, epsilons=[self.eps], loss_name=self.loss_name, data_type=self.attack_data, punished_measure=self.punished_measure)


        measure_list = measure.raw.detach().cpu().numpy()[0].tolist()
        adv_d1 = d1[0].raw
        adv_d2 = d2[0].raw

        return adv_d1, adv_d2




    def compute_conf_labels(self, model, X, Y, adv):
        dist = self.get_dist(X, adv)
        lambda_dist = self.get_lambda(dist)
        Y_mod = self.get_mod_labels(Y, lambda_dist)
        return dist, lambda_dist, Y_mod



    def get_dist(self, X, A):
        mat = torch.reshape(torch.abs(X-A),((X-A).shape[0], -1))
        if self.norm_adv == 'L2':
            dist = torch.norm(mat, 2, 1)
        elif self.norm_adv == 'Linf':
            dist, inds = torch.max(mat, 1)
        else:
            dist = None
            print('This norm is not implemented, returning None')
        return dist


    def get_lambda(self, dist):
        lambda_dist = (1.0 - torch.min(dist, torch.ones_like(dist)))**self.rho
        return lambda_dist

    def get_mod_labels(self, Y, lambda_dist):
        y_uni = (1.0 - lambda_dist)*(1/self.nr_classes)
        Y_mod = y_uni.repeat(10, 1).T
        Y_mod[range(Y.shape[0]), Y] = lambda_dist + Y_mod[range(Y.shape[0]), Y]
        return Y_mod
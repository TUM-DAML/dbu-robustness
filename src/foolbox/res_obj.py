import numpy as np
import re
import collections
import pickle
import os
from threading import Timer
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy
import scipy.special


import matplotlib as mpl
from IPython.display import IFrame
mpl.use('pgf')



class Attack_res():
    def __init__(self, base_model, in_dataset, ood_dataset_list, attack_type, criterion_type, loss_type, punished_measure, eps, measure_dict, data_dict, label_dict, adv1_dict, adv2_dict, t_dict, threshold):
        
        self.base_model = base_model
        self.in_dataset = in_dataset
        self.ood_dataset_list = ood_dataset_list
        self.attack_type = attack_type
        self.criterion_type = criterion_type
        self.loss_type = loss_type
        self.punished_measure = punished_measure 
        self.eps = eps
        self.measure_dict = measure_dict
        self.data_dict = data_dict
        self.label_dict = label_dict
        self.adv1_dict = adv1_dict
        self.adv2_dict = adv2_dict
        self.t_dict = t_dict
        self.threshold = threshold


        self.nr_classes = len(set(label_dict[in_dataset]))
        self.ood_alpha = np.ones(self.nr_classes)
        self.ood_alpha0 = np.sum(self.ood_alpha)

        self.id_alpha = np.ones(self.nr_classes)
        self.id_alpha0 = 100.0
        self.id_alpha[0] = self.id_alpha0 - (self.nr_classes-1)*1.0 # 0 is the correct class

        # Compute theoretical value of uncertainty measure for in data and out data
        if criterion_type == 'Alpha0':
            self.label_id = self.id_alpha0
            self.label_ood = self.ood_alpha0
        
        elif criterion_type == 'DiffEntropy':
            id_log_term = np.sum(scipy.special.loggamma(self.id_alpha)) - scipy.special.loggamma(self.id_alpha0)
            id_digamma_term = np.sum((self.id_alpha -1.0) * (scipy.special.digamma(self.id_alpha) - scipy.special.digamma(self.id_alpha0)))
            self.label_id = id_log_term - id_digamma_term

            ood_log_term = np.sum(scipy.special.loggamma(self.ood_alpha)) - scipy.special.loggamma(self.ood_alpha0)
            ood_digamma_term = np.sum((self.ood_alpha -1.0) * (scipy.special.digamma(self.ood_alpha) - scipy.special.digamma(self.ood_alpha0)))
            self.label_ood = ood_log_term - ood_digamma_term

        elif criterion_type == 'DistUncertainty':
            id_probs = self.id_alpha / self.id_alpha0
            id_total_uncertainty = -1.0 * np.sum(id_probs * np.log(id_probs + 0.00001))
            id_digamma_term = scipy.special.digamma(self.id_alpha + 1.0) - scipy.special.digamma(self.id_alpha0 +1.0)
            id_dirichlet_mean = self.id_alpha / self.id_alpha0
            id_exp_data_uncertainty = -1 * np.sum(id_dirichlet_mean * id_digamma_term)
            self.label_id = id_total_uncertainty - id_exp_data_uncertainty

            ood_probs = self.ood_alpha / self.ood_alpha0
            ood_total_uncertainty = -1.0 * np.sum(ood_probs * np.log(ood_probs + 0.00001))
            ood_digamma_term = scipy.special.digamma(self.ood_alpha + 1.0) - scipy.special.digamma(self.ood_alpha0 +1.0)
            ood_dirichlet_mean = self.ood_alpha / self.ood_alpha0
            ood_exp_data_uncertainty = -1 * np.sum(ood_dirichlet_mean * ood_digamma_term)
            self.label_ood = ood_total_uncertainty - ood_exp_data_uncertainty

        else:
            self.label_id = None
            self.label_ood = None
        
    
    
    def store_res_obj(self, store_path):
        fh = open(store_path, 'wb')
        pickle.dump(self, fh)
        fh.close()            



def load_res_obj(load_path):
        fh = open(load_path, 'rb')
        res = pickle.load(fh)
        fh.close()
        return res
                





def figsize(scale, ratio_yx=None):
    fig_width_pt = 505.89 #397.48499  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio
    if ratio_yx is None:
        ratio_yx = golden_mean
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * ratio_yx  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def sns_facetsize(tot_width=0.95, ratio_yx_facet=1.6, nrows=1, ncols=1):
    ratio_yx = ratio_yx_facet * nrows / ncols
    size = figsize(tot_width, ratio_yx)
    height_facet = size[1] / nrows
    ratio_xy_facet = 1 / ratio_yx_facet
    return height_facet, ratio_xy_facet  
    
    


# Customized newfig and savefig functions
def newfig(width, ratio_yx=None, style='whitegrid', subplots=True, nrows=1, ncols=1):
    sns.set(style=style, palette='colorblind', color_codes=True)
    
    if subplots:
        return plt.subplots(
                nrows, ncols,
                figsize=figsize(width, ratio_yx=ratio_yx))
    else:
        return plt.subplots(figsize=figsize(width, ratio_yx=ratio_yx))


def savefig(filename, fig=None, tight={'pad': 0.5},
        dpi=600, format='pgf', preview=None,
        close_fig=True, remove_preview_file_after=1, **kwargs):
    if fig is None:
        fig = plt.gca().figure
    if tight:
        fig.tight_layout(**tight)
    if os.path.splitext(filename)[1] == format:
        filepath = filename
    else:
        filepath = f"{filename}.{format}"
        filepath2 = f"{filename}.pdf"
    fig.savefig(filepath2, dpi=dpi, **kwargs)
    if close_fig:
        if fig is None:
            plt.close()
        else:
            plt.close(fig)
    if preview is not None:
        if preview == format:
            preview_path = filepath
        else:
            while True:
                rnd_int = np.random.randint(np.iinfo(np.uint32).max)
                preview_path = f"preview_tmp{rnd_int}.{preview}"
                if not os.path.exists(preview_path):
                    break
            fig.savefig(preview_path, dpi=dpi, **kwargs)
            Timer(remove_preview_file_after, os.remove, args=[preview_path]).start()
        return IFrame(os.path.relpath(preview_path), width=700, height=500)






def get_plot_params(nr_curves):
    sns.set()

    if np.mod(nr_curves, 2) == 0:
        nr_p1 = int(nr_curves / 2)
        nr_p2 = int(nr_curves / 2)
    else:
        nr_p1 = int((nr_curves+1)/2)
        nr_p2 = int((nr_curves-1)/2)

    
    color_list_p1 = sns.color_palette('inferno', nr_p1)
    color_list_p2 = sns.color_palette('viridis', nr_p2)
    color_list_p2.reverse()
    color_list = color_list_p1 + color_list_p2
    sns.palplot(color_list)
    
    
    
    # parameters for latex
    pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
    "text.latex.preamble": [
        r"\usepackage[utf8]{inputenc}",  # use utf8 fonts
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath}",
        r"\newcommand*{\mat}[1]{\boldsymbol{#1}}",
    ],
    "pgf.preamble": [
        r"\usepackage[utf8]{inputenc}",  # use utf8 fonts
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath}",
        r"\newcommand*{\mat}[1]{\boldsymbol{#1}}",
        ],
    }
    mpl.rcParams.update(pgf_with_latex)
    # Set line widths
    default_params = {
        'grid.linewidth': 0.5,
        'grid.color': '.8',
        'axes.linewidth': 0.75,
        'axes.edgecolor': '.7',
    }
    
    
    return color_list, pgf_with_latex, default_params
    





def plot_aucroc(model, res_path_acc, res_list):
    eps_list = []
    measure_list = []
    t_list = []
    label_list = []

    model_list = []
    id_list = []
    #ood_list = []
    criterion_list = []
    loss_list = []
    #punished_list = []
    #threshold_list = []

    
    for res in res_list:
        eps_list.append(res.eps)
        t_list.append(res.t_dict)

        # must all be the same 
        model_list.append(res.base_model)
        id_list.append(res.in_dataset)
        criterion_list.append(res.criterion_type)
        loss_list.append(res.loss_type)
    
    if len(set(model_list)) == len(set(id_list)) == len(set(criterion_list)) == len(set(loss_list)) == 1:
        base_model = res_list[0].base_model
        data_types = list(res_list[0].t_dict)
        in_data = res_list[0].in_dataset
        out_data = res_list[0].ood_dataset_list
        attack_type = res_list[0].attack_type
        criterion_type = res_list[0].criterion_type
        loss_type = res_list[0].loss_type
        punished_measure = res_list[0].punished_measure

        # accumulate and plot times
        color_list, pgf_with_latex, default_params = get_plot_params(len(data_types))
        save_path_curr = '{}_res_attack={}_criterion={}_loss={}_punishedm={}_time'.format(base_model, attack_type, criterion_type, loss_type, punished_measure)
        fig, ax = newfig(0.6)
        for j in range(len(data_types)):
            dt = data_types[j]
            t_curr = []
            for i in range(len(eps_list)):
                t_curr.append(t_list[i][dt])
            ax.plot(eps_list, t_curr, label='{}'.format(dt), color=color_list[j])

        ax.set_xlabel('Epsilon')
        ax.set_ylabel('Mean Time [s]')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
        ax.legend(loc=1) # 0 best, 1 upper right, 2 upper left, 3 lower left, 4 lower right, 5 right, 6 center ....
        ax.set_title('Attacking {}'.format(criterion_type))
        save_path_curr = save_path_curr.replace('.','_')
        save_path_curr = save_path_curr.replace(' ','')
        savefig('{}{}'.format(res_path_acc, save_path_curr))

        if criterion_type == 'Misclassification':
                print('AUC-plots etc. not implemented for criterion Misclassification.')
        else:            
            color_list, pgf_with_latex, default_params = get_plot_params(len(eps_list))
            if np.abs(res_list[0].eps) < 1e-10:
                 measure_in_orig = res_list[0].measure_dict[in_data]
            else:
                print('Forgot to compute measures for original data. Please add eps=0.0 in your experiments.')
            for dt in out_data:
                measure_out_orig = res_list[0].measure_dict[dt]
                
                for i in range(len(eps_list)):
                    eps = eps_list[i]
                    measure_in = res_list[i].measure_dict[in_data]
                    measure_out = res_list[i].measure_dict[dt]

                    # only on in data
                    save_path_curr = '{}_res_attack={}_criterion={}_loss={}_punishedm={}_in={}_out={}_attack=in_auc'.format(base_model, attack_type, criterion_type, loss_type, punished_measure, in_data, out_data)
                    fig, ax = newfig(0.6)






    else:
        print('No plots available.')


    
    mpl.rcParams.update(mpl.rcParamsDefault)

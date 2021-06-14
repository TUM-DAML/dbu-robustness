import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import logging
import torch
from sacred import Experiment
import seml
from seml.utils import flatten
import copy

from src.datasets.Dataset import get_dataset
from src.models.ensemble_networks.EnsembleN import EnsembleN
from src.models.ensemble_networks.train import train as train_teacher
from src.models.distribution_distillation_networks.DistributionDistilledN import DistributionDistilledN
from src.models.distribution_distillation_networks.train import train as train_student
from src.models.distribution_distillation_networks.test import test
from src.foolbox.adversarial_training import parse_config_and_create_attacks

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(
        # Dataset parameters
        seed_dataset,  # Seed to shuffle dataset. int
        dataset_name,  # Dataset name. string
        ood_dataset_names,  # OOD dataset names.  list of strings
        split,  # Split for train/val/test sets. list of floats

        # Architecture parameters
        seed_model,  # Seed to init model. int
        directory_model,  # Path to save model. string
        architecture,  # Encoder architecture name. string
        input_dims,  # Input dimension. List of ints
        output_dim,  # Output dimension. int
        hidden_dims,  # Hidden dimensions. list of ints
        kernel_dim,  # Input dimension. int
        k_lipschitz,  # Lipschitz constant. float or None (if no lipschitz)
        n_networks,  # Number of networks in ensemble. int

        # Training parameters
        directory_results,  # Path to save resutls. string
        max_epochs,  # Maximum number of epochs for training
        patience,  # Patience for early stopping. int
        frequency,  # Frequency for early stopping test. int
        batch_size,  # Batch size. int
        lr,  # Learning rate. float
        loss,  # Loss name. string

        adversarial_training=None,  # adversarial training
        rs_sigma=None,  # rs training
        ):

    logging.info('Received the following configuration:')
    logging.info(f'DATASET | '
                 f'seed_dataset {seed_dataset} - '
                 f'dataset_name {dataset_name} - '
                 f'ood_dataset_names {ood_dataset_names} - '
                 f'split {split}')
    logging.info(f'ARCHITECTURE | '
                 f' seed_model {seed_model} - '
                 f' architecture {architecture} - '
                 f' input_dims {input_dims} - '
                 f' output_dim {output_dim} - '
                 f' hidden_dims {hidden_dims} - '
                 f' kernel_dim {kernel_dim} - '
                 f' k_lipschitz {k_lipschitz} - '
                 f' n_networks {n_networks}')
    logging.info(f'TRAINING | '
                 f' max_epochs {max_epochs} - '
                 f' patience {patience} - '
                 f' frequency {frequency} - '
                 f' batch_size {batch_size} - '
                 f' lr {lr} - '
                 f' loss {loss}')

    if adversarial_training is not None:
        logging.info(f'ADVERSARIAL TRAINING | '
                     " - ".join([f"{k}: {v}" for k,v in adversarial_training.items()]))
        adversarial_training = copy.deepcopy(adversarial_training)

    ##################
    ## Load dataset ##
    ##################
    train_loader, val_loader, test_loader, N = get_dataset(dataset_name,
                                                           batch_size=batch_size,
                                                           split=split,
                                                           seed=seed_dataset)

    #################
    ## Train model ##
    #################
    teacher_model = EnsembleN(input_dims=input_dims,
                              output_dim=output_dim,
                              hidden_dims=hidden_dims,
                              kernel_dim=kernel_dim,
                              architecture=architecture,
                              k_lipschitz=k_lipschitz,
                              n_networks=n_networks,
                              batch_size=batch_size,
                              lr=lr,
                              loss='CE',
                              seed=seed_model)
    model = DistributionDistilledN(input_dims=input_dims,
                                   output_dim=output_dim,
                                   hidden_dims=hidden_dims,
                                   kernel_dim=kernel_dim,
                                   architecture=architecture,
                                   k_lipschitz=k_lipschitz,
                                   batch_size=batch_size,
                                   lr=lr,
                                   loss=loss,
                                   seed=seed_model)
    full_config_dict = {'seed_dataset': seed_dataset,
                        'dataset_name': dataset_name,
                        'split': split,
                        'seed_model': seed_model,
                        'architecture': architecture,
                        'input_dims': input_dims,
                        'output_dim': output_dim,
                        'hidden_dims': hidden_dims,
                        'kernel_dim': kernel_dim,
                        'k_lipschitz': k_lipschitz,
                        'n_networks': n_networks,
                        'max_epochs': max_epochs,
                        'patience': patience,
                        'frequency': frequency,
                        'batch_size': batch_size,
                        'lr': lr,
                        'loss': loss,
                        # adversarial training
                        'adversarial_training': adversarial_training,
                        'rs_sigma': rs_sigma
                        }
    full_config_name = ''
    for k, v in full_config_dict.items():
        if isinstance(v, dict):
            v = flatten(v)
            v = [str(val) for key, val in v.items()]
            v = "-".join(v)
        full_config_name += str(v) + '-'
    full_config_name = full_config_name[:-1]
    model_path = f'{directory_model}/model-ddn-{full_config_name}'

    in_atk, out_atk = parse_config_and_create_attacks(adversarial_training, model)

    if False:
        teacher_model.load_state_dict(torch.load(f'{model_path}'.replace('ddn', 'en').replace('Dir-LL', 'CE'))['model_state_dict'])
    else:
        train_teacher(teacher_model,
                      train_loader,
                      val_loader,
                      max_epochs=max_epochs,
                      frequency=frequency,
                      patience=patience,
                      model_path=f'{model_path}-teacher',
                      full_config_dict=full_config_dict)
        teacher_model.load_state_dict(torch.load(f'{model_path}-teacher')['model_state_dict'])
    train_losses, val_losses, train_accuracies, val_accuracies = train_student(teacher_model,
                                                                               model,
                                                                               train_loader,
                                                                               val_loader,
                                                                               max_epochs=max_epochs,
                                                                               frequency=frequency,
                                                                               patience=patience,
                                                                               model_path=model_path,
                                                                               in_data_attack=in_atk,
                                                                               rs_sigma=rs_sigma,
                                                                               full_config_dict=full_config_dict)

    ################
    ## Test model ##
    ################
    ood_dataset_loaders = {}
    for ood_dataset_name in ood_dataset_names:
        ood_train_loader, ood_val_loader, ood_test_loader, N = get_dataset(ood_dataset_name,
                                                                           batch_size=batch_size,
                                                                           split=split,
                                                                           seed=seed_dataset)
        ood_dataset_loaders[ood_dataset_name] = ood_test_loader
    result_path = f'{directory_results}/results-ddn-{full_config_name}.csv'
    model.load_state_dict(torch.load(f'{model_path}')['model_state_dict'])
    metrics = test(model, test_loader, ood_dataset_loaders, result_path)

    results = {
        'model_path': model_path,
        'result_path': result_path,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'fail_trace': seml.evaluation.get_results
    }

    return {**results, **metrics}

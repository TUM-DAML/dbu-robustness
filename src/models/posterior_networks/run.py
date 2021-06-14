import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import logging
import torch
from sacred import Experiment
import seml
from seml.utils import flatten
import copy

from src.datasets.Dataset import get_dataset
from src.models.posterior_networks.PosteriorN import PosteriorN
from src.models.posterior_networks.train import train, train_sequential
from src.models.posterior_networks.test import test
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
        latent_dim,  # Latent dimension. int
        no_density,  # Use density estimation or not. boolean
        density_type,  # Density type. string
        n_density,  # Number of density components. int
        k_lipschitz,  # Lipschitz constant. float or None (if no lipschitz)
        budget_function,  # Budget function name applied on class count. name


        # Training parameters
        directory_results,  # Path to save resutls. string
        max_epochs,  # Maximum number of epochs for training
        patience,  # Patience for early stopping. int
        frequency,  # Frequency for early stopping test. int
        batch_size,  # Batch size. int
        lr,  # Learning rate. float
        loss,  # Loss name. string
        training_mode,  # 'joint' or 'sequential' training. string
        regr,  # Regularization factor in Bayesian loss. float

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
                 f' latent_dim {latent_dim} - '
                 f' no_density {no_density} - '
                 f' density_type {density_type} - '
                 f' n_density {n_density} - '
                 f' k_lipschitz {k_lipschitz} - '
                 f' budget_function {budget_function}')
    logging.info(f'TRAINING | '
                 f' max_epochs {max_epochs} - '
                 f' patience {patience} - '
                 f' frequency {frequency} - '
                 f' batch_size {batch_size} - '
                 f' lr {lr} - '
                 f' loss {loss} - '
                 f' training_mode {training_mode} - '
                 f' regr {regr}')

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
    model = PosteriorN(N=N,
                       input_dims=input_dims,
                       output_dim=output_dim,
                       hidden_dims=hidden_dims,
                       kernel_dim=kernel_dim,
                       latent_dim=latent_dim,
                       architecture=architecture,
                       k_lipschitz=k_lipschitz,
                       no_density=no_density,
                       density_type=density_type,
                       n_density=n_density,
                       budget_function=budget_function,
                       batch_size=batch_size,
                       lr=lr,
                       loss=loss,
                       regr=regr,
                       seed=seed_model)

    full_config_dict = {'seed_dataset': seed_dataset,
                        'dataset_name': dataset_name,
                        'split': split,
                        'seed_model': seed_model,
                        'architecture': architecture,
                        # 'N': N,
                        'input_dims': input_dims,
                        'output_dim': output_dim,
                        'hidden_dims': hidden_dims,
                        'kernel_dim': kernel_dim,
                        'latent_dim': latent_dim,
                        'no_density': no_density,
                        'density_type': density_type,
                        'n_density': n_density,
                        'k_lipschitz': k_lipschitz,
                        'budget_function': budget_function,
                        'max_epochs': max_epochs,
                        'patience': patience,
                        'frequency': frequency,
                        'batch_size': batch_size,
                        'lr': lr,
                        'loss': loss,
                        'training_mode': training_mode,
                        'regr': regr,

                        # adversarial training
                        'adversarial_training': adversarial_training,
                        'rs_sigma': rs_sigma
                        }
    full_config_name = ''
    for k, v in full_config_dict.items():
        if isinstance(v, dict):
            v = flatten(v)
            v = [str(val) for key,val in v.items()]
            v = "-".join(v)
        full_config_name += str(v) + '-'
    full_config_name = full_config_name[:-1]
    model_path = f'{directory_model}/model-dpn-{full_config_name}'

    in_atk, _ = parse_config_and_create_attacks(adversarial_training, model)

    if training_mode == 'joint':
        train_losses, val_losses, train_accuracies, val_accuracies = train(model,
                                                                           train_loader,
                                                                           val_loader,
                                                                           max_epochs=max_epochs,
                                                                           frequency=frequency,
                                                                           patience=patience,
                                                                           model_path=model_path,
                                                                           in_data_attack=in_atk,
                                                                           rs_sigma=rs_sigma,
                                                                           full_config_dict=full_config_dict,)
    elif training_mode == 'sequential':
        assert not no_density
        train_losses, val_losses, train_accuracies, val_accuracies = train_sequential(model,
                                                                                       train_loader,
                                                                                       val_loader,
                                                                                       max_epochs=max_epochs,
                                                                                       frequency=frequency,
                                                                                       patience=patience,
                                                                                       model_path=model_path,
                                                                                       full_config_dict=full_config_dict)
    else:
        raise NotImplementedError
    ################
    ## Test model ##
    ################
    ood_dataset_loaders = {}
    for ood_dataset_name in ood_dataset_names:
        ood_train_loader, ood_val_loader, ood_test_loader, N = get_dataset(ood_dataset_name,
                                                                           batch_size=batch_size,
                                                                           split=[.95, .05],
                                                                           seed=seed_dataset)
        ood_dataset_loaders[ood_dataset_name] = ood_test_loader
        # ood_dataset_loaders[ood_dataset_name] = ood_train_loader
    result_path = f'{directory_results}/results-dpn-{full_config_name}'
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

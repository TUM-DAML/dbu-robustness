import inspect
import torch
from src.datasets.Dataset import get_dataset
from src.models.posterior_networks.PosteriorN import PosteriorN
from src.models.evidential_networks.EvidentialN import EvidentialN
from src.models.distribution_distillation_networks.DistributionDistilledN import DistributionDistilledN
from src.models.kl_prior_networks.KLPN import KLPN

create_model = {'postnet': PosteriorN,
                'evnet': EvidentialN,
                'ddnet': DistributionDistilledN,
                'priornet': KLPN,
                'rev_kl_priornet': KLPN}


def load_model(directory_model, name_model, model_type, batch_size_eval=1024):
    model_path = directory_model + name_model
    map_location = None
    if not torch.cuda.is_available():
        map_location = "cpu"

    # Select arguments for model creation
    args = inspect.getfullargspec(create_model[model_type])[0][1:]
    config_dict = torch.load(f'{model_path}', map_location=map_location)['model_config_dict']
    _, _, _, config_dict['N'] = get_dataset(config_dict['dataset_name'],
                                            batch_size=config_dict['batch_size'],
                                            split=config_dict['split'],
                                            seed=config_dict['seed_dataset'],
                                            test_shuffle_seed=None,
                                            batch_size_eval=batch_size_eval)
    config_dict['seed'] = config_dict['seed_model']
    filtered_config_dict = {arg: config_dict[arg] for arg in args}

    # Create model
    model = create_model[model_type](**filtered_config_dict)


    # Load weights
    model.load_state_dict(torch.load(f'{model_path}', map_location=map_location)['model_state_dict'])
    model.eval()

    return model

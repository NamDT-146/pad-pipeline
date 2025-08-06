from .siamesenetwork import SiameseNetwork, create_siamese_model
from .mobilenetv2 import MobileNetV2Network, create_mobilenetv2_model
from .minutiaenet import MinutiaeNetSiamese, create_minutiaenet_model
from .minutiaenet_simple import SimpleMinutiaeNetSiamese, create_simple_minutiaenet_model

NETWORK = {
    'siamese': SiameseNetwork,
    'mobilenetv2': MobileNetV2Network,
    'minutiaenet': MinutiaeNetSiamese,
    'minutiaenet_simple': SimpleMinutiaeNetSiamese,
}

FACTORY = {
    'siamese': create_siamese_model,
    'mobilenetv2': create_mobilenetv2_model,
    'minutiaenet': create_minutiaenet_model,
    'minutiaenet_simple': create_simple_minutiaenet_model,
}

def get_architecture(network_type='siamese', args=None, device=None, **kwargs):
    if network_type in FACTORY:
        # Prefer factory for device handling
        if device is not None:
            # Merge args and kwargs
            all_args = {**(args or {}), **kwargs}
            return FACTORY[network_type](device, **all_args)
        else:
            # Merge args and kwargs
            all_args = {**(args or {}), **kwargs}
            return FACTORY[network_type](**all_args)
    elif network_type in NETWORK:
        # Merge args and kwargs
        all_args = {**(args or {}), **kwargs}
        return NETWORK[network_type](**all_args)
    else:
        raise ValueError(f"Unknown network type: {network_type}. Available types: {list(NETWORK.keys())}")
    

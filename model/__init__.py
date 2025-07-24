from .siamesenetwork import SiameseNetwork, create_siamese_model
from .mobilenetv2 import MobileNetV2Network, create_mobilenetv2_model

NETWORK = {
    'siamese': SiameseNetwork,
    'mobilenetv2': MobileNetV2Network,
}

FACTORY = {
    'siamese': create_siamese_model,
    'mobilenetv2': create_mobilenetv2_model,
}

def get_architecture(network_type='siamese', args=None, device=None):
    if network_type in FACTORY:
        # Prefer factory for device handling
        if device is not None:
            return FACTORY[network_type](device, **(args or {}))
        else:
            return FACTORY[network_type](**(args or {}))
    elif network_type in NETWORK:
        return NETWORK[network_type](**(args or {}))
    else:
        raise ValueError(f"Unknown network type: {network_type}. Available types: {list(NETWORK.keys())}")
    

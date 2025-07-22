from .siamesenetwork import SiameseNetwork, create_siamese_model

NETWORK = {
    'siamese': SiameseNetwork,
}

def get_architecture(network_type='siamese', args=None):
    if network_type in NETWORK:
        return NETWORK[network_type](**(args or {}))
    else:
        raise ValueError(f"Unknown network type: {network_type}. Available types: {list(NETWORK.keys())}")
    

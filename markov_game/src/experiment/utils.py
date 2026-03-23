import os
import random
import numpy as np
import torch
import inspect
import re
import pandas as pd
from omegaconf import OmegaConf

def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    # Set seed for the gym environment
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)


def check_all_keys_exist(config_a, config_b):
    """
    Check whether all keys in config_a exist in config_b.
    Return True as long as the key exists, even if the value is None.
    """
    def _check_recursive(a_dict, b_dict, path=""):
        for key, value in a_dict.items():
            current_path = f"{path}.{key}" if path else key
            
            # Check whether the key exists (value can be None)
            if key not in b_dict:
                return False, current_path
            
            # If the value is a nested dictionary, check recursively
            if OmegaConf.is_dict(value):
                if not OmegaConf.is_dict(b_dict[key]):
                    return False, current_path
                result, missing_key = _check_recursive(value, b_dict[key], current_path)
                if not result:
                    return False, missing_key
        
        return True, None
    
    return _check_recursive(config_a, config_b)



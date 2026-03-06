import collections
from typing import Iterable

import jax
from jax import numpy as jnp


def update_nested_pytree(pytree, update: dict):
    """
    Update a nested pytree with another nested dictionary
    :param pytree:
    :param update:
    :return:
    """
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            pytree = pytree.replace(**{k: update_nested_pytree(eval(f"pytree.{k}"), v)})
        else:
            pytree = pytree.replace(
                **{k: jnp.array(v) if isinstance(v, Iterable) else v}
            )
    return pytree


def remove_non_list_entries(
    d,
    list_parameters=(
        "asset_range",
        "consumption_preferences",
        "prices",
        "consumption_tax_rate",
        "hidden_layers",
    ),
    matrix_parameters=(),
    omit_parameters=("consumption_preferences",),
):
    """
    Remove entries from a dictionary that are not lists (except for the hidden_layers entry)
    Applies to nested dictionaries like config_DQN
    :param d: Nested dictionary
    :param list_parameters: List of parameters that should be lists
    :return:
    """

    def filter_value(k, v):
        if k in omit_parameters:
            return None
        if isinstance(v, dict):
            recursive_output = remove_non_list_entries(
                v, list_parameters, matrix_parameters, omit_parameters)
            if len(recursive_output) > 0:
                return recursive_output
            else:
                return None
        elif isinstance(v, list) and (
            (k not in list_parameters and k not in matrix_parameters)
            or (k in list_parameters and isinstance(v[0], list))
            or (k in matrix_parameters and isinstance(v[0][0], list))
        ):
            return v
        else:
            return None

    return {
        k: filter_value(k, v) for k, v in d.items() if filter_value(k, v) is not None
    }


def check_if_jittable(func):
    """Check whether a function is JIT-compatible.
    Example: check_if_jittable(jax.numpy.add)"""
    
    print(f"=== JIT compatibility check for function {func.__name__} ===")
    
    # 1. Check if the jit decorator is applied
    if hasattr(func, '__wrapped__'):
        print("✓ JIT decorator is applied")
        print(f"  Original function: {func.__wrapped__}")
    else:
        print("? JIT decorator is not directly applied")
    
    # 2. Check the function's type
    print(f"Function type: {type(func)}")
    
    # 3. Check JAX internal attributes
    if hasattr(func, '_cpp_jitted_f'):
        print("✓ This is a JAX JIT-compiled function")
    elif hasattr(func, 'f'):
        print("✓ Possibly a JAX-wrapped function")
    
    # 4. Try actually JIT-ing the function
    try:
        # Attempt to wrap with jax.jit using a simple test
        jitted_func = jax.jit(func)
        print("✓ Can be wrapped with jax.jit()")
        return True
    except Exception as e:
        print(f"✗ Failed to JIT: {e}")
        return False

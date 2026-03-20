import inspect
from garage.np.algos import RLAlgorithm

from .. import algos as algos_module
from .. import follower as follower_module


def _collect_classes(module):
    classes = {}
    exported_names = getattr(module, '__all__', None)

    if exported_names is None:
        candidates = inspect.getmembers(module, inspect.isclass)
        for name, cls in candidates:
            if cls.__module__.startswith(module.__name__):
                classes[name] = cls
        return classes

    for name in exported_names:
        value = getattr(module, name, None)
        if inspect.isclass(value):
            classes[name] = value
    return classes


def _is_rl_algorithm_class(cls):
    return inspect.isclass(cls) and issubclass(cls, RLAlgorithm)


def _algo_name(algo_class):
    name = getattr(algo_class, 'name', None)
    if isinstance(name, str) and name:
        return name
    return algo_class.__name__


_algo_classes = {
    **_collect_classes(algos_module),
    **_collect_classes(follower_module),
}

# Get all RLAlgorithm classes from the src.algos and src.follower modules, and create a mapping from their names to the classes.
algos = {_algo_name(algo_class): algo_class
         for algo_class in _algo_classes.values()
         if _is_rl_algorithm_class(algo_class)}

def get_algo(algo: str):
    """Get the algorithm class corresponding to the given name."""
    try:
        return algos[algo]
    except KeyError:
        raise ValueError(f"Unknown algorithm: {algo}")

def get_init_kwarg_names(cls):
    """Get the names of keyword-only arguments in the __init__ method of the given class."""
    sig = inspect.signature(cls.__init__)
    return set([k for k, v in sig.parameters.items()
                if k != 'self' and v.kind == inspect.Parameter.KEYWORD_ONLY])

def kwargs_from_cfg(cfg, algo_cls):
    """Extract the keyword arguments for constructing an algorithm from a config dict."""
    if not algo_cls in algos.values():
        raise ValueError(f"Unknown algorithm class: {algo_cls}")
    def _get_all_kwarg_names(c):
        parents = c.__bases__
        if RLAlgorithm in parents:
            return get_init_kwarg_names(c)
        else:
            _kwargs = set()
            for p in parents:
                _kwargs |= _get_all_kwarg_names(p)
            return _kwargs | get_init_kwarg_names(c)
    kwargs_set = _get_all_kwarg_names(algo_cls)
    kwargs = {k: v for k, v in cfg.items() if k in kwargs_set}
    return kwargs

if __name__ == '__main__':
    print("Available algorithms:")
    for name, algo_class in algos.items():
        print(f"  {name}: {algo_class.name}")
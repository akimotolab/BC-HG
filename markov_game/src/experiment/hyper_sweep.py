import itertools
import multiprocessing
import random
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf
import dill

class Sweeper(object):
    def __init__(self, hyper_config):
        self.hyper_config = hyper_config
        self.timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    def __iter__(self):
        for config in itertools.product(*[val if isinstance(val, list) else [val] 
                                          for val in self.hyper_config.values()]):
            kwargs = {key:config[i] for i, key in enumerate(self.hyper_config.keys())}
            kwargs['datetime'] = self.timestamp
            yield kwargs

# Perform sweep while maintaining the hierarchical structure of hyper_config
class HierarchicalSweeper(object):
    def __init__(self, hyper_config):
        self.hyper_config = hyper_config
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def __iter__(self):
        vals = []
        for key, val in self.hyper_config.items():
            if isinstance(val, dict) or OmegaConf.is_dict(val):
                if key != 'sweep':
                    val = [d for d, _ in HierarchicalSweeper(val)]
                else:
                    val = [val]
            elif isinstance(val, list) or OmegaConf.is_list(val):
                val = val
            else:
                val = [val]          
            vals.append(val)
        for config in itertools.product(*vals):
            kwargs = {key:config[i] for i, key in enumerate(self.hyper_config.keys())}
            yield kwargs, self.timestamp
            
def kwargs_wrapper(args_method):
    args, method = args_method
    return method(args)

def kwargs_wrapper_for_garage(args_method):
    args, method = args_method
    ctxt, params = args_for_experiments(args)
    return method(ctxt, cfg=params)

def args_for_experiments(args):
    sweeped_args = ''
    for k, sw_args in args['sweep'].items():
        if sw_args is not None and len(sw_args) > 0:
            if k == "env":
                sweeped_args += f'_E'
            elif k == "leader":
                sweeped_args += f'_L'
            elif k == "follower":
                sweeped_args += f'_F'
            else:
                raise ValueError(f"Unknown sweep key: {k}") 
            for sw_arg in sw_args:
                value = args[k][sw_arg]
                arg_str = "{:.0e}".format(value) if isinstance(value, float) else str(value) 
                sweeped_args += f'_{sw_arg}_{arg_str}'

    if ('name' in args 
        and args['name'] is not None
        and len(args['name']) > 0):
        prefix = f"{args['datetime']}_{args['name']}"
        exp_name = f"{args['name']}{sweeped_args}"
    else:
        prefix = args['datetime']
        exp_name = sweeped_args
    # Hint: name = exp_name + seed
    name = f"{exp_name}_seed_{args['seed']}" if len(exp_name) > 0 else f"seed_{args['seed']}"

    ctxt = {k:v for k,v in args['ctxt'].items()}
    ctxt['prefix'] = f'experiment/{prefix}'
    ctxt['name'] = name
    # Hint: log_dir = data/local/{prefix}/{name} if ctxt['log_dir'] is None
    for k,v in args['ctxt'].items():
        _ = ctxt.pop(k) if v is None else None  # Remove ctxt if not set, so default values are used instead of None
    args['exp_name'] = exp_name
    return ctxt, args

def run_sweep_parallel_ctxt(run_method, params, num_cpu=multiprocessing.cpu_count()):
    sweeper= HierarchicalSweeper(params)
    pool = multiprocessing.Pool(num_cpu, init_process, (pcount,))
    exp_args = []
    for config, timestamp in sweeper:
        config['experiment']['datetime'] = timestamp
        exp_args.append((config, dill.dumps(run_method)))
    random.shuffle(exp_args)
    results = pool.map(kwargs_wrapper_for_garage, exp_args)
    return results, pcount.value

def run_sweep_parallel(run_method, params, num_cpu=multiprocessing.cpu_count()):
    sweeper = Sweeper(params)
    pool = multiprocessing.Pool(num_cpu, init_process, (pcount,))
    exp_args = [(config, run_method) for config in sweeper]
    random.shuffle(exp_args)
    results = pool.map(kwargs_wrapper, exp_args)
    return results, pcount.value

def run_sweep_parallel_with_pbar(run_method, params, num_cpu=multiprocessing.cpu_count()):
    sweeper = Sweeper(params)
    pool = multiprocessing.Pool(num_cpu, init_process, (pcount,))
    exp_args = [(config, run_method) for config in sweeper]
    random.shuffle(exp_args)
    imap = pool.imap(kwargs_wrapper, exp_args)
    results = list(tqdm(imap, total=len(exp_args)))
    return results, pcount.value

def init_process(pcount):
    #　pid is the process ID of each child process and is a global variable in each child process scope.
    global pid
    pid = pcount.value
    pcount.value += 1

# The number of generated child processes.
pcount = multiprocessing.Value('I', 0)

lock = multiprocessing.Lock()
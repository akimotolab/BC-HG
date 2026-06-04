from .bchg import BCHG
from .bchg_discrete import BCHGDiscrete
from .bchg_opt import BCHG_Opt
from .bchg_discrete_opt import BCHGDiscrete_Opt
from .bchg_discrete_subopt import BCHGDiscrete_Subopt

class Baseline(BCHG):
    name = "Baseline"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._no_guidance = True

class BaselineDiscrete(BCHGDiscrete):
    name = "BaselineDiscrete"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._no_guidance = True

class Baseline_Opt(BCHG_Opt):
    name = "Baseline_Opt"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._no_guidance = True

class BaselineDiscrete_Opt(BCHGDiscrete_Opt):
    name = "BaselineDiscrete_Opt"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._no_guidance = True

class BaselineDiscrete_Subopt(BCHGDiscrete_Subopt):
    name = "BaselineDiscrete_Subopt"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._no_guidance = True

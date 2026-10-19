# --- gpytorch compat (newer, botorch-compatible gpytorch) ---------------------
# Newer gpytorch/linear_operator removed LinearOperator.evaluate() in favour of
# to_dense(); this codebase still calls .evaluate() on kernel/lazy tensors, so
# re-add the alias if missing. No-op on older gpytorch (which still has it).
try:
    from linear_operator.operators import LinearOperator as _LinearOperator
    if not hasattr(_LinearOperator, "evaluate"):
        _LinearOperator.evaluate = _LinearOperator.to_dense
except Exception:
    pass

from .gpregression import GPR
#from .lmgp_short import LMGP
#from .lmgp_short2 import LMGP
#from .lmgp_short3 import LMGP
#from .FCNN import FFNN
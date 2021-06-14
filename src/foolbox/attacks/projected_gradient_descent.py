from typing import Optional

from .gradient_descent_base import L1BaseGradientDescent
from .gradient_descent_base import L2BaseGradientDescent
from .gradient_descent_base import LinfBaseGradientDescent


class L1ProjectedGradientDescentAttack(L1BaseGradientDescent):
    """L1 Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        steps: int = 50,
        random_start: bool = True,
        lambda_diffa: float = 2.0,     # used to weight terms of alphadist loss
        lambda_ce_uncert: float = 0.1, # used to weight uncertainty-term in crossentropy_x loss (x = alpha0, diffE, distU)
        lambda_alpha0: float = 0.01,  # map alpha0 for in data alpha0 is in [10, 100] (in: high, out:low), L2 distance is in [0,1]
        lambda_diffE: float = 0.05,   # map diffE, diffE is in  [-32.45, -12.80] if alpha0=100 (in: low, out: high)
        lambda_distU: float = 3.0,    # map distU, distU is in [0.038, 0.3736] if alpha0=100 (in: low, out: high)

    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
            lambda_diffa= lambda_diffa, 
            lambda_ce_uncert=lambda_ce_uncert,
            lambda_alpha0=lambda_alpha0,
            lambda_diffE=lambda_diffE,
            lambda_distU=lambda_distU,
        )


class L2ProjectedGradientDescentAttack(L2BaseGradientDescent):
    """L2 Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        steps: int = 50,
        random_start: bool = True,
        lambda_diffa: float = 2.0,     # used to weight terms of alphadist loss
        lambda_ce_uncert: float = 0.1, # used to weight uncertainty-term in crossentropy_x loss (x = alpha0, diffE, distU)
        lambda_alpha0: float = 0.01,  # map alpha0 for in data alpha0 is in [10, 100] (in: high, out:low), L2 distance is in [0,1]
        lambda_diffE: float = 0.05,   # map diffE, diffE is in  [-32.45, -12.80] if alpha0=100 (in: low, out: high)
        lambda_distU: float = 3.0,    # map distU, distU is in [0.038, 0.3736] if alpha0=100 (in: low, out: high)
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
            lambda_diffa=lambda_diffa,
            lambda_ce_uncert=lambda_ce_uncert,
            lambda_alpha0=lambda_alpha0,
            lambda_diffE=lambda_diffE,
            lambda_distU=lambda_distU,
        )


class LinfProjectedGradientDescentAttack(LinfBaseGradientDescent):
    """Linf Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon (defaults to 0.01 / 0.3).
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.01 / 0.3,
        abs_stepsize: Optional[float] = None,
        steps: int = 40,
        random_start: bool = True,
        lambda_diffa: float = 2.0,     # used to weight terms of alphadist loss
        lambda_ce_uncert: float = 0.1, # used to weight uncertainty-term in crossentropy_x loss (x = alpha0, diffE, distU)
        lambda_alpha0: float = 0.01,  # map alpha0 for in data alpha0 is in [10, 100] (in: high, out:low), L2 distance is in [0,1]
        lambda_diffE: float = 0.05,   # map diffE, diffE is in  [-32.45, -12.80] if alpha0=100 (in: low, out: high)
        lambda_distU: float = 3.0,    # map distU, distU is in [0.038, 0.3736] if alpha0=100 (in: low, out: high)
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
            lambda_diffa=lambda_diffa,
            lambda_ce_uncert=lambda_ce_uncert,
            lambda_alpha0=lambda_alpha0,
            lambda_diffE=lambda_diffE,
            lambda_distU=lambda_distU,
        )

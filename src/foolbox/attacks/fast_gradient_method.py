from .gradient_descent_base import L1BaseGradientDescent
from .gradient_descent_base import L2BaseGradientDescent
from .gradient_descent_base import LinfBaseGradientDescent


class L1FastGradientAttack(L1BaseGradientDescent):
    """Fast Gradient Method (FGM) using the L1 norm

    Args:
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(
        self, 
        *, 
        random_start: bool = False,
        lambda_diffa: float = 2.0,     # used to weight terms of alphadist loss
        lambda_ce_uncert: float = 0.1, # used to weight uncertainty-term in crossentropy_x loss (x = alpha0, diffE, distU)
        lambda_alpha0: float = 0.01,  # map alpha0 for in data alpha0 is in [10, 100] (in: high, out:low), L2 distance is in [0,1]
        lambda_diffE: float = 0.05,   # map diffE, diffE is in  [-32.45, -12.80] if alpha0=100 (in: low, out: high)
        lambda_distU: float = 3.0,    # map distU, distU is in [0.038, 0.3736] if alpha0=100 (in: low, out: high)
    ):
        super().__init__(
            rel_stepsize=1.0, 
            steps=1, 
            random_start=random_start,
            lambda_diffa=lambda_diffa,
            lambda_ce_uncert=lambda_ce_uncert,
            lambda_alpha0=lambda_alpha0,
            lambda_diffE=lambda_diffE,
            lambda_distU=lambda_distU,
        )


class L2FastGradientAttack(L2BaseGradientDescent):
    """Fast Gradient Method (FGM)

    Args:
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(
        self, 
        *, random_start: bool = False,
        lambda_diffa: float = 2.0,     # used to weight terms of alphadist loss
        lambda_ce_uncert: float = 0.1, # used to weight uncertainty-term in crossentropy_x loss (x = alpha0, diffE, distU)
        lambda_alpha0: float = 0.01,  # map alpha0 for in data alpha0 is in [10, 100] (in: high, out:low), L2 distance is in [0,1]
        lambda_diffE: float = 0.05,   # map diffE, diffE is in  [-32.45, -12.80] if alpha0=100 (in: low, out: high)
        lambda_distU: float = 3.0,    # map distU, distU is in [0.038, 0.3736] if alpha0=100 (in: low, out: high)
        ):
        super().__init__(
            rel_stepsize=1.0, 
            steps=1, 
            random_start=random_start,
            lambda_diffa=lambda_diffa,
            lambda_ce_uncert=lambda_ce_uncert,
            lambda_alpha0=lambda_alpha0,
            lambda_diffE=lambda_diffE,
            lambda_distU=lambda_distU,
        )


class LinfFastGradientAttack(LinfBaseGradientDescent):
    """Fast Gradient Sign Method (FGSM)

    Args:
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(
        self, 
        *, random_start: bool = False,
        lambda_diffa: float = 2.0,     # used to weight terms of alphadist loss
        lambda_ce_uncert: float = 0.1, # used to weight uncertainty-term in crossentropy_x loss (x = alpha0, diffE, distU)
        lambda_alpha0: float = 0.01,  # map alpha0 for in data alpha0 is in [10, 100] (in: high, out:low), L2 distance is in [0,1]
        lambda_diffE: float = 0.05,   # map diffE, diffE is in  [-32.45, -12.80] if alpha0=100 (in: low, out: high)
        lambda_distU: float = 3.0,    # map distU, distU is in [0.038, 0.3736] if alpha0=100 (in: low, out: high)
        ):
        super().__init__(
            rel_stepsize=1.0, 
            steps=1, 
            random_start=random_start,
            lambda_diffa=lambda_diffa,
            lambda_ce_uncert=lambda_ce_uncert,
            lambda_alpha0=lambda_alpha0,
            lambda_diffE=lambda_diffE,
            lambda_distU=lambda_distU,
        )

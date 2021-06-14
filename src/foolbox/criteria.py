"""
===============================================================================
:mod:`foolbox.criteria`
===============================================================================

Criteria are used to define which inputs are adversarial.
We provide common criteria for untargeted and targeted adversarial attacks,
e.g. :class:`Misclassification` and :class:`TargetedMisclassification`.
New criteria can easily be implemented by subclassing :class:`Criterion`
and implementing :meth:`Criterion.__call__`.

Criteria can be combined using a logical and ``criterion1 & criterion2``
to create a new criterion.


:class:`Misclassification`
===============================================================================

.. code-block:: python

   from foolbox.criteria import Misclassification
   criterion = Misclassification(labels)

.. autoclass:: Misclassification
   :members:


:class:`TargetedMisclassification`
===============================================================================

.. code-block:: python

   from foolbox.criteria import TargetedMisclassification
   criterion = TargetedMisclassification(target_classes)

.. autoclass:: TargetedMisclassification
   :members:


:class:`Criterion`
===============================================================================

.. autoclass:: Criterion
   :members:
   :special-members: __call__
"""
from typing import TypeVar, Any
from abc import ABC, abstractmethod
import eagerpy as ep
import torch


T = TypeVar("T")


class Criterion(ABC):
    """Abstract base class to implement new criteria."""

    @abstractmethod
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def __call__(self, perturbed: T, outputs: T) -> T:
        """Returns a boolean tensor indicating which perturbed inputs are adversarial.

        Args:
            perturbed: Tensor with perturbed inputs ``(batch, ...)``.
            outputs: Tensor with model outputs for the perturbed inputs ``(batch, ...)``.

        Returns:
            A boolean tensor indicating which perturbed inputs are adversarial ``(batch,)``.
        """
        ...

    def __and__(self, other: "Criterion") -> "Criterion":
        return _And(self, other)


class _And(Criterion):
    def __init__(self, a: Criterion, b: Criterion):
        super().__init__()
        self.a = a
        self.b = b

    def __repr__(self) -> str:
        return f"{self.a!r} & {self.b!r}"

    def __call__(self, perturbed: T, outputs: T) -> T:
        args, restore_type = ep.astensors_(perturbed, outputs)
        a = self.a(*args)
        b = self.b(*args)
        is_adv = ep.logical_and(a, b)
        return restore_type(is_adv)


class Misclassification(Criterion):
    """Considers those perturbed inputs adversarial whose predicted class
    differs from the label.

    Args:
        labels: Tensor with labels of the unperturbed inputs ``(batch,)``.
    """

    def __init__(self, labels: Any):
        super().__init__()
        self.labels: ep.Tensor = ep.astensor(labels)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed, outputs
        
        classes = outputs_.argmax(axis=-1)
        assert classes.shape == self.labels.shape
        is_adv = classes != self.labels
        return restore_type(is_adv)



class TargetedMisclassification(Criterion):
    """Considers those perturbed inputs adversarial whose predicted class
    matches the target class.

    Args:
        target_classes: Tensor with target classes ``(batch,)``.
    """

    def __init__(self, target_classes: Any):
        super().__init__()
        self.target_classes: ep.Tensor = ep.astensor(target_classes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target_classes!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed, outputs

        classes = outputs_.argmax(axis=-1)
        assert classes.shape == self.target_classes.shape
        is_adv = classes == self.target_classes
        return restore_type(is_adv)





"""
Added criteria for attacking uncertainty estimation
"""



class Alpha0(Criterion):
    """Computes alpha0.

    Args:
        target_classes: Tensor with target classes ``(batch,)``.
    """

    def __init__(self, labels: Any, start_data: Any, threshold: Any):
        super().__init__()
        self.labels: ep.Tensor = ep.astensor(labels)
        self.start_data = start_data
        self.threshold = threshold

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        alpha0 = outputs.sum(axis=1)
        del perturbed, outputs
        
        if not(self.threshold is None):
            if self.start_data == 'in':
                is_adv = alpha0 < self.threshold
            elif self.start_data == 'out' or self.start_data == 'random':
                is_adv = alpha0 > self.threshold
            else:
                is_adv = None
        else:
            is_adv = alpha0

        return is_adv


class DiffEntropy(Criterion):
    """Computes differential entropy

    Args:
        target_classes: Tensor with target classes ``(batch,)``.
    """

    def __init__(self, labels: Any, start_data: Any, threshold):
        super().__init__()
        self.labels: ep.Tensor = ep.astensor(labels)
        self.start_data = start_data
        self.threshold = threshold

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        eps = 1e-6
        alpha = outputs.raw + eps
        alpha0 = alpha.sum(-1)
        del perturbed, outputs

        log_term = torch.sum(torch.lgamma(alpha), axis=1) - torch.lgamma(alpha0)
        digamma_term = torch.sum((alpha - 1.0) * (
                    torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))),
                                 axis=1)
        differential_entropy = log_term - digamma_term
        # scores = - differential_entropy.cpu().detach().numpy()
        
        diffE = ep.astensor(differential_entropy)

        if not(self.threshold is None):
            if self.start_data == 'in': 
                is_adv = diffE > self.threshold
            elif self.start_data == 'out' or self.start_data == 'random':
                is_adv = diffE < self.threshold
            else:
                is_adv = None
        else:
            is_adv = diffE

        return is_adv
        

class DistUncertainty(Criterion):
    """Computes differential entropy

    Args:
        target_classes: Tensor with target classes ``(batch,)``.
    """

    def __init__(self, labels: Any, start_data: Any, threshold: Any):
        super().__init__()
        self.labels: ep.Tensor = ep.astensor(labels)
        self.start_data = start_data
        self.threshold = threshold

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        alpha = outputs.raw
        alpha0 = torch.sum(alpha, axis=1)
        probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)

        del perturbed, outputs

        total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=1)

        digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) +1.0)
        dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        exp_data_uncertainty = -1 * torch.sum(dirichlet_mean * digamma_term, dim=1)
        
        distributional_uncertainty = total_uncertainty - exp_data_uncertainty

        distU = ep.astensor(distributional_uncertainty)

        if not(self.threshold is None):
            if self.start_data == 'in': 
                is_adv = distU > self.threshold
            elif self.start_data == 'out' or self.start_data == 'random':
                is_adv = distU < self.threshold
            else:
                is_adv = None
        else:
            is_adv = distU

        return is_adv


class MisclassificationBest(Criterion):
    """Considers those perturbed inputs adversarial whose predicted class
    differs from the label.

    Args:
        labels: Tensor with labels of the unperturbed inputs ``(batch,)``.
    """

    def __init__(self, labels: Any):
        super().__init__()
        self.labels = labels

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        alpha = outputs

        classes = alpha.argmax(axis=-1)
        assert classes.shape == self.labels.shape
        is_adv = classes != self.labels
        # print(is_adv.sum())
        # print(perturbed[is_adv][0], classes[0], self.labels[0])
        if is_adv.sum() > 0:
            best_perturbed = perturbed[is_adv][0]
        else:
            best_perturbed = perturbed[0]
        return best_perturbed


class Alpha0Best(Criterion):
    """Computes alpha0.

    Args:
        target_classes: Tensor with target classes ``(batch,)``.
    """

    def __init__(self, labels: Any, start_data: Any, threshold: Any):
        super().__init__()
        self.labels = labels
        self.start_data = start_data
        self.threshold = threshold

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        alpha0 = outputs.sum(-1)

        if self.start_data == 'in':
            best_perturbed = perturbed[alpha0.min(0)[1]]
        elif self.start_data == 'out':
            best_perturbed = perturbed[alpha0.max(0)[1]]
        else:
            raise NotImplementedError

        return best_perturbed


class DiffEntropyBest(Criterion):
    """Computes differential entropy

    Args:
        target_classes: Tensor with target classes ``(batch,)``.
    """

    def __init__(self, labels: Any, start_data: Any, threshold):
        super().__init__()
        self.labels = labels
        self.start_data = start_data
        self.threshold = threshold

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        eps = 1e-6
        alpha = outputs + eps
        alpha0 = alpha.sum(-1)

        log_term = torch.sum(torch.lgamma(alpha), axis=1) - torch.lgamma(alpha0)
        digamma_term = torch.sum((alpha - 1.0) * (
                torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))),
                                 axis=1)
        differential_entropy = log_term - digamma_term
        # scores = - differential_entropy.cpu().detach().numpy()

        if self.start_data == 'in':
            best_perturbed = perturbed[differential_entropy.max(0)[1]]
        elif self.start_data == 'out':
            best_perturbed = perturbed[differential_entropy.min(0)[1]]
        else:
            raise NotImplementedError

        return best_perturbed


class DistUncertaintyBest(Criterion):
    """Computes differential entropy

    Args:
        target_classes: Tensor with target classes ``(batch,)``.
    """

    def __init__(self, labels: Any, start_data: Any, threshold: Any):
        super().__init__()
        self.labels: ep.Tensor = ep.astensor(labels)
        self.start_data = start_data
        self.threshold = threshold

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        alpha = outputs
        alpha0 = torch.sum(alpha, axis=-1)
        probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)

        total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=1)

        digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(
            alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) + 1.0)
        dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        exp_data_uncertainty = -1 * torch.sum(dirichlet_mean * digamma_term, dim=1)

        distributional_uncertainty = total_uncertainty - exp_data_uncertainty

        if self.start_data == 'in':
            best_perturbed = perturbed[distributional_uncertainty.max(0)[1]]
        elif self.start_data == 'out':
            best_perturbed = perturbed[distributional_uncertainty.min(0)[1]]
        else:
            raise NotImplementedError

        return best_perturbed

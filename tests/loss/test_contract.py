"""
Properties every loss has, checked for each loss in :data:`LOSSES` and each task it claims, on
batches with ID and OOD targets.
"""

import copy
import inspect
import textwrap
from dataclasses import dataclass
from typing import Iterator, List, Optional

import pytest
import torch
from torch import Tensor, nn

from pytorch_ood import loss
from pytorch_ood.api import Task
from pytorch_ood.utils import is_known
from tests.helpers.components import LOSSES
from tests.helpers.losses import LOSS_INPUTS, SHAPES


@dataclass
class LossCall:
    """A loss with its arguments, called on a copy so that stateful losses start equal."""

    criterion: nn.Module
    args: List
    #: position of the target among ``args``; not always last (EnergyMarginLoss)
    target_at: int

    @property
    def target(self) -> Tensor:
        return self.args[self.target_at]

    def rows(self, index: Tensor) -> List:
        """The arguments with one entry per sample, indexed; the others unchanged."""
        n = self.target.shape[0]
        return [
            a[index] if isinstance(a, Tensor) and a.shape[:1] == (n,) else a for a in self.args
        ]

    def with_grad(self) -> List:
        """The arguments, with floating-point tensors replaced by copies that require grad."""
        return [
            a.clone().requires_grad_(True)
            if isinstance(a, Tensor) and a.is_floating_point()
            else a
            for a in self.args
        ]

    def __call__(self, target: Optional[Tensor] = None, args: Optional[List] = None) -> Tensor:
        args = list(self.args if args is None else args)
        if target is not None:
            args[self.target_at] = target
        with torch.random.fork_rng():
            torch.manual_seed(0)  # losses that sample, e.g. virtual outliers
            return copy.deepcopy(self.criterion)(*args)


def _call(cls: type, task: Task, reduction: Optional[str] = None) -> LossCall:
    with torch.random.fork_rng():
        torch.manual_seed(0)  # the builders draw random inputs
        criterion, args = LOSS_INPUTS[cls](SHAPES[task])
    if reduction is not None:
        criterion.reduction = reduction
    args = list(args)
    target_at = max(
        i for i, a in enumerate(args) if isinstance(a, Tensor) and not a.is_floating_point()
    )
    # every case gets ID and OOD targets
    args[target_at] = args[target_at].clone()
    args[target_at].view(-1)[::3] = -1
    return LossCall(criterion, args, target_at)


def _unsupervised(cls: type) -> bool:
    return not cls.info.supervised


def _cases(applies=lambda cls: True, marks=None) -> List:
    marks = marks or {}
    return [
        pytest.param(cls, task, id=f"{cls.__name__}-{task.value}", marks=marks.get(cls, ()))
        for cls in LOSSES
        if applies(cls)
        for task in cls.info.tasks
    ]


def _has_reduction(cls: type) -> bool:
    return "reduction" in inspect.signature(cls.__init__).parameters


DROPS_OOD_ROWS = pytest.mark.xfail(
    strict=True, reason="reduction='none' returns only the ID rows, not one value per target"
)
NONE_SHAPE_BROKEN = {
    loss.DeepSVDDLoss: DROPS_OOD_ROWS,
    loss.LogitNorm: DROPS_OOD_ROWS,
    loss.VirtualOutlierSynthesizingRegLoss: DROPS_OOD_ROWS,
}


# --- reductions ---------------------------------------------------------------------------------


@pytest.mark.parametrize("cls, task", _cases(_has_reduction, marks=NONE_SHAPE_BROKEN))
def test_none_gives_one_value_per_target(cls: type, task: Task):
    call = _call(cls, task, "none")

    value = call()

    assert value.shape == call.target.shape
    if _unsupervised(cls):
        assert (value[~is_known(call.target)] == 0).all()


@pytest.mark.parametrize(
    "cls, task",
    _cases(
        # documented: with "none", VOSRegLoss adds its batch regularizer to every element
        lambda cls: _has_reduction(cls) and cls is not loss.VOSRegLoss,
    ),
)
def test_sum_is_the_sum_of_none(cls: type, task: Task):
    torch.testing.assert_close(_call(cls, task, "sum")(), _call(cls, task, "none")().sum())


# --- batches ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls, task",
    # documented: EnergyMarginLoss requires ID and OOD samples in every batch
    _cases(lambda cls: cls is not loss.EnergyMarginLoss),
)
def test_all_ood_batch_gives_a_finite_scalar_with_gradients(cls: type, task: Task):
    call = _call(cls, task)

    value = call(torch.full_like(call.target, -1), call.with_grad())

    assert value.shape == ()
    assert torch.isfinite(value)
    value.backward()


@pytest.mark.parametrize("cls, task", _cases())
def test_invariant_to_permuting_the_batch(cls: type, task: Task, seed: torch.Generator):
    call = _call(cls, task)
    perm = torch.randperm(call.target.shape[0], generator=seed)

    torch.testing.assert_close(call(), call(args=call.rows(perm)))


# --- unsupervised losses discard OOD samples ----------------------------------------------------

UNSUPERVISED = [pytest.param(cls, id=cls.__name__) for cls in LOSSES if _unsupervised(cls)]


def test_unsupervised_losses_are_all_covered():
    assert len(UNSUPERVISED) == 8


@pytest.mark.parametrize("cls", UNSUPERVISED)
def test_outliers_do_not_change_an_unsupervised_loss(cls: type):
    call = _call(cls, Task.CLASSIFICATION)

    with pytest.warns(UserWarning):
        mixed = call()

    torch.testing.assert_close(mixed, call(args=call.rows(is_known(call.target))))


@pytest.mark.parametrize("cls", UNSUPERVISED)
def test_only_outliers_give_zero_loss_and_zero_gradients(cls: type):
    call = _call(cls, Task.CLASSIFICATION)
    inputs = call.with_grad()

    with pytest.warns(UserWarning):
        value = call(torch.full_like(call.target, -1), inputs)

    assert value.shape == ()
    assert value.item() == 0
    value.backward()
    for a in inputs:
        if isinstance(a, Tensor) and a.requires_grad:
            assert (a.grad == 0).all()


@pytest.mark.filterwarnings("error::UserWarning")
@pytest.mark.parametrize("cls", UNSUPERVISED)
def test_unsupervised_loss_does_not_warn_without_outliers(cls: type):
    call = _call(cls, Task.CLASSIFICATION)

    call(call.target.clamp(min=0))


@pytest.mark.filterwarnings("error::UserWarning")
def test_mchad_passes_only_id_samples_to_its_unsupervised_parts():
    _call(loss.MCHADLoss, Task.CLASSIFICATION)()


# --- docstring examples -------------------------------------------------------------------------

#: losses whose contract differs from ``loss(logits, targets)``, so they need a usage example
NEED_EXAMPLE = [
    loss.CACLoss,
    loss.CenterLoss,
    loss.ConfidenceLoss,
    loss.DeepSADLoss,
    loss.DeepSVDDLoss,
    loss.EnergyMarginLoss,
    loss.IILoss,
    loss.MCHADLoss,
    loss.ObjectosphereLoss,
    loss.VOSRegLoss,
    loss.VirtualOutlierSynthesizingRegLoss,
]


def _code_blocks(doc: Optional[str]) -> Iterator[str]:
    """The content of every ``.. code-block:: python`` directive in a docstring."""
    lines = (doc or "").splitlines()
    for i, line in enumerate(lines):
        if line.strip() != ".. code-block:: python":
            continue
        indent = len(line) - len(line.lstrip())
        block = []
        for body in lines[i + 1 :]:
            if body.strip() and len(body) - len(body.lstrip()) <= indent:
                break
            block.append(body)
        yield textwrap.dedent("\n".join(block)).strip()


@pytest.mark.parametrize("cls", NEED_EXAMPLE, ids=lambda cls: cls.__name__)
def test_losses_with_a_different_signature_have_an_example(cls: type):
    assert list(_code_blocks(cls.__doc__)), "no usage example"


# unsupervised losses warn about the OOD samples they discard
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "cls, code",
    [
        pytest.param(cls, code, id=f"{cls.__name__}-{i}")
        for cls in LOSSES
        for i, code in enumerate(_code_blocks(cls.__doc__))
    ],
)
def test_docstring_example_runs(cls: type, code: str):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        exec(compile(code, f"<{cls.__name__} example>", "exec"), {})

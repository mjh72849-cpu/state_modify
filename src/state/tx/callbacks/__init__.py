import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch.optim import Optimizer

from ..models import PerturbationModel
from .batch_speed_monitor import BatchSpeedMonitorCallback
from .model_flops_utilization import ModelFLOPSUtilizationCallback
from .cumulative_flops import CumulativeFLOPSCallback
from .concise_progress import ConciseProgressCallback

__all__ = [
    "PerturbationModel",
    "BatchSpeedMonitorCallback",
    "ModelFLOPSUtilizationCallback",
    "CumulativeFLOPSCallback",
    "ConciseProgressCallback",
]


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer
    ) -> None:
        pl_module.log("train/gradient_norm", gradient_norm(pl_module))
        for group in optimizer.param_groups:
            name = group.get("name")
            if name:
                pl_module.log(
                    f"train/gradient_norm_{name}",
                    gradient_norm_from_parameters(group["params"]),
                )


def gradient_norm(model):
    return gradient_norm_from_parameters(model.parameters())


def gradient_norm_from_parameters(parameters):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm

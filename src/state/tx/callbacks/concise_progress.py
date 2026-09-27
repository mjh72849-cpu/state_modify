import time

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class ConciseProgressCallback(Callback):
    """Emit one compact console line every N optimizer steps.

    Lightning's tqdm bar refreshes for every microbatch.  That is useful in an
    interactive terminal but creates very large redirected/tmux logs.  This
    callback is deliberately keyed to ``global_step`` so its cadence remains
    stable when gradient accumulation changes.
    """

    def __init__(self, logging_interval: int = 50) -> None:
        super().__init__()
        if logging_interval <= 0:
            raise ValueError("logging_interval must be positive")
        self.logging_interval = logging_interval
        self._last_logged_step = 0
        self._started_at: float | None = None

    def on_train_start(self, trainer, pl_module) -> None:
        self._started_at = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if step <= 0 or step == self._last_logged_step or step % self.logging_interval:
            return

        elapsed = max(time.monotonic() - (self._started_at or time.monotonic()), 1e-9)
        rate = step / elapsed
        remaining = max(int(trainer.max_steps) - step, 0)
        eta_hours = remaining / max(rate, 1e-9) / 3600
        metrics = trainer.callback_metrics

        def scalar(name: str) -> float | None:
            value = metrics.get(name)
            if value is None:
                return None
            try:
                return float(value.detach().cpu())
            except AttributeError:
                return float(value)

        train_loss = scalar("train_loss")
        decoder_loss = scalar("decoder_loss")
        gradient_norm = scalar("train/gradient_norm")
        decoder_weight = float(getattr(pl_module, "decoder_loss_weight", 1.0))
        total_loss = (
            train_loss + decoder_weight * decoder_loss
            if train_loss is not None and decoder_loss is not None
            else None
        )

        loss_text = ""
        if train_loss is not None:
            loss_text += f" train_loss={train_loss:.6f}"
        if decoder_loss is not None:
            loss_text += f" decoder_loss={decoder_loss:.6f}"
        if total_loss is not None:
            loss_text += f" total_loss={total_loss:.6f}"
        if gradient_norm is not None:
            loss_text += f" gradient_norm={gradient_norm:.4f}"
        rank_zero_info(
            "TRAIN_PROGRESS optimizer_step=%d/%d optimizer_steps_per_second=%.4f eta_hours=%.2f%s",
            step,
            int(trainer.max_steps),
            rate,
            eta_hours,
            loss_text,
        )
        self._last_logged_step = step

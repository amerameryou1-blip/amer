from __future__ import annotations

import argparse
import dataclasses
import glob
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from training.tpu.data.fast_fen import ensure_fast_fen_built
from training.tpu.data.xla_input import BinaryShardConfig, count_records, create_dataloader
from training.tpu.models.nnue_tpu import NnueTpuModel
from training.tpu.optim.ema import ExponentialMovingAverage
from training.tpu.optim.lookahead import Lookahead


@dataclasses.dataclass(slots=True)
class TrainConfig:
    train_glob: str
    val_glob: str | None
    output_dir: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_steps: int
    min_lr_scale: float
    num_workers: int
    prefetch_factor: int
    clip_grad_norm: float
    ema_decay: float
    lookahead_alpha: float
    lookahead_k: int
    seed: int
    log_every: int
    compile_fast_fen: bool
    save_every: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Mythos NNUE on TPU with PyTorch/XLA")
    parser.add_argument("--train-glob", type=str, required=True)
    parser.add_argument("--val-glob", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts/nnue")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--lookahead-alpha", type=float, default=0.5)
    parser.add_argument("--lookahead-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--compile-fast-fen", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        train_glob=args.train_glob,
        val_glob=args.val_glob,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr_scale=args.min_lr_scale,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        clip_grad_norm=args.clip_grad_norm,
        ema_decay=args.ema_decay,
        lookahead_alpha=args.lookahead_alpha,
        lookahead_k=args.lookahead_k,
        seed=args.seed,
        log_every=args.log_every,
        compile_fast_fen=args.compile_fast_fen,
        save_every=args.save_every,
    )


def discover_shards(pattern: str | None) -> list[str]:
    if not pattern:
        return []
    return sorted(glob.glob(pattern))


def xla_world_size() -> int:
    if hasattr(xm, "xrt_world_size"):
        return max(1, int(xm.xrt_world_size()))
    try:
        import torch_xla.runtime as xr

        return max(1, int(xr.world_size()))
    except Exception:
        return 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_scale: float) -> LambdaLR:
    def schedule(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return max(1e-6, float(step + 1) / float(max(1, warmup_steps)))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda=schedule)


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> Lookahead:
    base_optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return Lookahead(base_optimizer, alpha=cfg.lookahead_alpha, k=cfg.lookahead_k)


def evaluate(model: torch.nn.Module, loader, max_batches: int | None = None) -> float:
    model.eval()
    running_loss = 0.0
    batches = 0

    with torch.no_grad():
        for features, targets in loader:
            predictions = model(features)
            loss = F.smooth_l1_loss(predictions, targets)
            running_loss += loss.detach().item()
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

    return 0.0 if batches == 0 else running_loss / batches


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    step: int,
    model: torch.nn.Module,
    optimizer: Lookahead,
    scheduler: LambdaLR,
    ema: ExponentialMovingAverage,
    val_loss: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "val_loss": val_loss,
    }
    xm.save(checkpoint, output_dir / f"checkpoint-epoch{epoch:03d}.pt")


def train_worker(index: int, cfg: TrainConfig) -> None:
    train_shards = discover_shards(cfg.train_glob)
    if not train_shards:
        raise RuntimeError(f"no training shards matched pattern: {cfg.train_glob}")

    val_shards = discover_shards(cfg.val_glob)
    if cfg.compile_fast_fen and xm.is_master_ordinal():
        ensure_fast_fen_built(verbose=True)
    xm.rendezvous("fast-fen-build")

    seed_everything(cfg.seed + index)
    device = xm.xla_device()

    model = NnueTpuModel().to(device)
    optimizer = build_optimizer(model, cfg)
    ema = ExponentialMovingAverage(model, decay=cfg.ema_decay, device="cpu")

    total_records = sum(count_records(path) for path in train_shards)
    steps_per_epoch = max(1, math.ceil(total_records / cfg.batch_size / xla_world_size()))
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = create_scheduler(optimizer.optimizer, cfg.warmup_steps, total_steps, cfg.min_lr_scale)

    train_dataset_cfg = BinaryShardConfig(
        shard_paths=train_shards,
        records_per_batch=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        shuffle_files=True,
        seed=cfg.seed,
        drop_last=False,
        flip_to_stm=True,
    )
    val_dataset_cfg = BinaryShardConfig(
        shard_paths=val_shards,
        records_per_batch=cfg.batch_size,
        num_workers=max(1, cfg.num_workers // 2),
        prefetch_factor=max(2, cfg.prefetch_factor // 2),
        shuffle_files=False,
        seed=cfg.seed,
        drop_last=False,
        flip_to_stm=True,
    )

    global_step = 0
    output_dir = Path(cfg.output_dir)

    for epoch in range(cfg.epochs):
        _, train_loader = create_dataloader(train_dataset_cfg, epoch=epoch)
        train_device_loader = pl.MpDeviceLoader(train_loader, device)

        model.train()
        running_loss = 0.0
        running_batches = 0

        for step_in_epoch, (features, targets) in enumerate(train_device_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = F.smooth_l1_loss(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            xm.optimizer_step(optimizer, barrier=False)
            scheduler.step()
            ema.update(model)

            running_loss += loss.detach().item()
            running_batches += 1
            global_step += 1

            if step_in_epoch % cfg.log_every == 0:
                reduced_loss = xm.mesh_reduce(
                    "train_loss",
                    running_loss / max(1, running_batches),
                    lambda losses: sum(losses) / len(losses),
                )
                if xm.is_master_ordinal():
                    xm.master_print(
                        f"epoch={epoch + 1} step={step_in_epoch} global_step={global_step} "
                        f"loss={reduced_loss:.5f} lr={scheduler.get_last_lr()[0]:.6e}"
                    )
                running_loss = 0.0
                running_batches = 0

        val_loss = 0.0
        if val_shards:
            ema.apply_shadow(model)
            _, val_loader = create_dataloader(val_dataset_cfg, epoch=epoch)
            val_device_loader = pl.MpDeviceLoader(val_loader, device)
            val_loss = evaluate(model, val_device_loader, max_batches=200)
            reduced_val_loss = xm.mesh_reduce("val_loss", val_loss, lambda losses: sum(losses) / len(losses))
            ema.restore(model)
            if xm.is_master_ordinal():
                xm.master_print(f"epoch={epoch + 1} validation_loss={reduced_val_loss:.5f}")
                val_loss = reduced_val_loss

        if (epoch + 1) % cfg.save_every == 0 and xm.is_master_ordinal():
            save_checkpoint(output_dir, epoch + 1, global_step, model, optimizer, scheduler, ema, val_loss)

        xm.rendezvous(f"epoch-{epoch}-complete")


def main() -> None:
    cfg = parse_args()
    xmp.spawn(train_worker, args=(cfg,), nprocs=None, start_method="fork")


if __name__ == "__main__":
    main()

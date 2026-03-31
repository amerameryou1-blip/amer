from __future__ import annotations

import hashlib
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from training.tpu.data.fast_fen import decode_binary_records


def _xla_world_info() -> tuple[int, int]:
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        if hasattr(xm, "xrt_world_size"):
            world_size = int(xm.xrt_world_size())
        else:
            world_size = int(xr.world_size())
        return xm.get_ordinal(), max(1, world_size)
    except Exception:
        return 0, 1


def count_records(path: str | Path) -> int:
    file_path = Path(path)
    if not file_path.exists():
        return 0

    count = 0
    with file_path.open("rb") as handle:
        while True:
            raw_length = handle.read(2)
            if len(raw_length) < 2:
                break
            fen_length = int.from_bytes(raw_length, byteorder="little", signed=False)
            handle.seek(fen_length + 1, io.SEEK_CUR)
            count += 1
    return count


def _read_batch_blob(handle, records_per_batch: int) -> bytes:
    batch = bytearray()
    records = 0
    while records < records_per_batch:
        raw_length = handle.read(2)
        if len(raw_length) < 2:
            break
        fen_length = int.from_bytes(raw_length, byteorder="little", signed=False)
        payload = handle.read(fen_length + 1)
        if len(payload) < fen_length + 1:
            break
        batch.extend(raw_length)
        batch.extend(payload)
        records += 1
    return bytes(batch)


@dataclass(slots=True)
class BinaryShardConfig:
    shard_paths: Sequence[str]
    records_per_batch: int = 2048
    num_workers: int = 2
    prefetch_factor: int = 4
    shuffle_files: bool = True
    seed: int = 1337
    drop_last: bool = False
    flip_to_stm: bool = True


class FenBatchIterableDataset(IterableDataset):
    def __init__(self, config: BinaryShardConfig) -> None:
        super().__init__()
        self.config = config
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _assigned_paths(self) -> list[str]:
        ordinal, world_size = _xla_world_info()
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        worker_count = worker_info.num_workers if worker_info is not None else 1

        global_workers = max(1, world_size * worker_count)
        global_rank = ordinal * worker_count + worker_id

        paths = list(self.config.shard_paths)
        if self.config.shuffle_files:
            seed_material = f"{self.config.seed}:{self._epoch}:{global_rank}".encode("utf-8")
            seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
            random.Random(seed).shuffle(paths)

        return [path for index, path in enumerate(paths) if index % global_workers == global_rank]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for shard_path in self._assigned_paths():
            with open(shard_path, "rb") as handle:
                while True:
                    blob = _read_batch_blob(handle, self.config.records_per_batch)
                    if not blob:
                        break

                    features, targets, count = decode_binary_records(
                        blob,
                        flip_to_stm=self.config.flip_to_stm,
                        prefer_cpp=True,
                    )
                    if count == 0:
                        continue
                    if self.config.drop_last and count < self.config.records_per_batch:
                        continue
                    yield features.contiguous(), targets.contiguous()


def create_dataloader(config: BinaryShardConfig, epoch: int = 0) -> tuple[FenBatchIterableDataset, DataLoader]:
    dataset = FenBatchIterableDataset(config)
    dataset.set_epoch(epoch)
    kwargs = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": config.num_workers,
        "pin_memory": False,
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
    loader = DataLoader(**kwargs)
    return dataset, loader

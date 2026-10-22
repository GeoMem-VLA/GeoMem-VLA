# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
# ZeRO-2/3 partition gradients on every microbatch and cannot enter model.no_sync().
# Keep Accelerate's accumulation counter (and optimizer/scheduler update boundaries),
# while letting DeepSpeed own gradient reduction and accumulation. Read the same launch
# environment as Accelerator's default plugin so CLI accumulation values stay effective.
# This trainer iterates across epochs and consumes extra evaluation batches; do not flush
# a partial accumulation at a dataloader boundary, which would disagree with DeepSpeed.
gradient_accumulation_plugin = GradientAccumulationPlugin(
    num_steps=int(os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "1")),
    sync_each_batch=True,
    sync_with_dataloader=False,
)
accelerator = Accelerator(
    deepspeed_plugin=deepspeed_plugin,
    gradient_accumulation_plugin=gradient_accumulation_plugin,
)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.resume_state_path = None
        self.weight_only_resume = False
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._init_checkpointing()
        if self.resume_state_path:
            # Validate before overwriting the run's original config snapshots.
            self._validate_resume_signature(self.resume_state_path)
        self._save_initial_configs()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        # Keep scheduler stepping exactly once per optimizer update. Preparing it with Accelerate
        # would wrap it in AcceleratedScheduler, whose split_batches=False behaviour advances the
        # underlying schedule world_size times per call. Register the raw scheduler only for state IO.
        self.accelerator.register_for_checkpointing(self.lr_scheduler)

        if self.resume_state_path:
            self._load_checkpoint(self.resume_state_path)
            logger.info(
                f"Strict resume restored model, optimizer, scheduler, scaler and RNG at "
                f"step {self.completed_steps}: {self.resume_state_path}"
            )
        elif self.weight_only_resume:
            # Legacy checkpoints have no optimizer/scheduler/RNG state. Advancing the scheduler
            # preserves the old behaviour, but this is explicitly a warm restart, not exact resume.
            self._adjust_lr_scheduler_for_resume()

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group=os.environ.get("WANDB_RUN_GROUP", "vla-train"),
            )

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_state, self.completed_steps = self._get_latest_training_state(self.checkpoint_dir)
            if resume_state:
                self.resume_state_path = resume_state
                self.resume_from_checkpoint = resume_state
                logger.info(
                    f"Strict-resume state selected: {resume_state}, steps: {self.completed_steps}"
                )
                return

            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                self.weight_only_resume = True
                logger.warning(
                    "Only a legacy model-only checkpoint was found. Model weights will be loaded "
                    "and the LR scheduler advanced, but optimizer and RNG state cannot be recovered; "
                    f"this is a WARM RESTART, not an exact resume: {self.resume_from_checkpoint} "
                    f"(step {self.completed_steps})"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            skip_pretrained_keys = getattr(self.config.trainer, "skip_pretrained_keys", None)
            self.model = self.load_pretrained_backbones(
                self.model,
                pretrained_checkpoint,
                reload_modules=reload_modules,
                skip_pretrained_keys=skip_pretrained_keys,
            )
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Restore a complete Accelerate/DeepSpeed checkpoint on every rank."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _resume_signature(self) -> dict:
        """Fields that must match for an exact optimizer/scheduler continuation."""
        vd = self.config.datasets.vla_data
        trainer = self.config.trainer
        def plain(value):
            if isinstance(value, AccessTrackedConfig):
                value = value.unwrap()
            return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value

        return {
            "world_size": int(self.accelerator.num_processes),
            "gradient_accumulation_steps": int(self.accelerator.gradient_accumulation_steps),
            "per_device_batch_size": int(vd.per_device_batch_size),
            "data_root_dir": str(Path(vd.data_root_dir).expanduser().resolve()),
            "data_mix": str(vd.data_mix),
            "max_train_steps": int(trainer.max_train_steps),
            "num_warmup_steps": int(trainer.num_warmup_steps),
            "lr_scheduler_type": str(trainer.lr_scheduler_type),
            "scheduler_specific_kwargs": plain(trainer.scheduler_specific_kwargs),
            "learning_rate": plain(trainer.learning_rate),
            "optimizer": plain(trainer.optimizer),
            "gradient_clipping": trainer.gradient_clipping,
            "freeze_modules": trainer.get("freeze_modules", ""),
            "seed": int(getattr(self.config, "seed", 3047)),
            "eval_interval": int(trainer.eval_interval),
            "sequential_step_sampling": bool(vd.get("sequential_step_sampling", False)),
            "group_sampling_mode": str(vd.get("group_sampling_mode", "consecutive")),
            "group_size": int(vd.get("group_size", 1)),
        }

    def _validate_resume_signature(self, state_path: str) -> None:
        marker = Path(state_path) / "_SUCCESS"
        try:
            saved = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid strict-resume marker {marker}: {exc}") from exc
        saved_signature = saved.get("resume_signature")
        current_signature = self._resume_signature()
        if not isinstance(saved_signature, dict) or not isinstance(current_signature, dict):
            raise RuntimeError(
                f"Invalid strict-resume signature: saved={saved_signature}, "
                f"current={current_signature}"
            )
        directory_step = int(Path(state_path).name.split("_")[1])
        if int(saved.get("steps", -1)) != directory_step or self.completed_steps != directory_step:
            raise RuntimeError(
                f"Strict-resume step mismatch: directory={directory_step}, marker={saved.get('steps')}, "
                f"selected={self.completed_steps}"
            )
        # Extending a run to more total steps legitimately changes the schedule
        # horizon without invalidating the restored optimizer/RNG state: this
        # process rebuilds the scheduler with the new horizon and restores its
        # step counter from the checkpoint. Allow those two fields to differ.
        horizon_changes = {
            key: (saved_signature.get(key), current_signature.get(key))
            for key in ("max_train_steps", "num_warmup_steps")
            if saved_signature.get(key) != current_signature.get(key)
        }
        if horizon_changes:
            logger.warning(f"Resuming with changed training horizon (allowed): {horizon_changes}")
        for key in horizon_changes:
            saved_signature.pop(key, None)
            current_signature.pop(key, None)

        if saved_signature != current_signature:
            raise RuntimeError(
                "Refusing an inexact strict resume because the training signature changed. "
                f"saved={saved_signature}, current={current_signature}. Use a new RUN_ID for a "
                "new configuration, or set IS_RESUME=False for an intentional fresh run."
            )

    def _prune_resume_states(self) -> None:
        """Keep only the newest complete strict states; portable model checkpoints are retained."""
        limit = int(getattr(self.config.trainer, "resume_checkpoint_limit", 2))
        if limit <= 0 or not self.accelerator.is_main_process:
            return
        checkpoint_root = Path(self.checkpoint_dir).resolve()
        states: list[tuple[int, Path]] = []
        for path in checkpoint_root.iterdir():
            match = re.fullmatch(r"steps_(\d+)_state", path.name)
            if match and path.is_dir() and (path / "_SUCCESS").is_file():
                states.append((int(match.group(1)), path.resolve()))
        for _, old_state in sorted(states)[:-limit]:
            if old_state.parent != checkpoint_root or not re.fullmatch(r"steps_\d+_state", old_state.name):
                raise RuntimeError(f"Refusing to prune unexpected checkpoint path: {old_state}")
            shutil.rmtree(old_state)
            logger.info(f"Pruned old strict-resume state (portable weights retained): {old_state}")

    def _save_checkpoint(self):
        """Save exact training state plus a portable model-only checkpoint."""
        save_format = getattr(self.config.trainer, "save_format", "pt")
        if save_format not in {"pt", "safetensors"}:
            raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
        state_path = Path(self.checkpoint_dir) / f"steps_{self.completed_steps}_state"
        self.accelerator.wait_for_everyone()
        if state_path.exists():
            if (state_path / "_SUCCESS").is_file():
                raise FileExistsError(f"Refusing to overwrite complete training state: {state_path}")
            # Every rank sees the same shared path. Rank 0 removes only this exact, incomplete
            # output-specific directory; barriers keep other ranks out until cleanup completes.
            if self.accelerator.is_main_process:
                expected_parent = Path(self.checkpoint_dir).resolve()
                resolved = state_path.resolve()
                if resolved.parent != expected_parent or resolved.name != f"steps_{self.completed_steps}_state":
                    raise RuntimeError(f"Refusing to clean unexpected partial state path: {resolved}")
                shutil.rmtree(resolved)
        self.accelerator.wait_for_everyone()

        # All ranks must participate in DeepSpeed checkpoint saving. _SUCCESS is written only
        # after every rank completes, so a partial directory can never be auto-selected.
        self.accelerator.save_state(str(state_path), safe_serialization=False)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            marker = {
                "steps": self.completed_steps,
                "resume_signature": self._resume_signature(),
            }
            marker_tmp = state_path / "_SUCCESS.tmp"
            marker_tmp.write_text(json.dumps(marker, indent=2) + "\n")
            os.replace(marker_tmp, state_path / "_SUCCESS")

        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

        # ZeRO-3 state consolidation is collective, so every rank must participate even though
        # only rank 0 serializes the portable model checkpoint.
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()
        self._prune_resume_states()
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        # This dataset is deterministic in (epoch, index). Resume at the exact next local batch
        # within the synthetic epoch; no video decode is performed for skipped batches because
        # Accelerate replaces the batch sampler with SkipBatchSampler.
        if self.resume_state_path and self.completed_steps > 0:
            eval_batches_consumed = self.completed_steps // int(self.config.trainer.eval_interval)
            total_microbatches_consumed = (
                self.completed_steps * self.accelerator.gradient_accumulation_steps
                + eval_batches_consumed
            )
            local_batches_per_epoch = len(self.vla_train_dataloader)
            self.vla_epoch_count, microbatches_to_skip = divmod(
                total_microbatches_consumed, local_batches_per_epoch
            )
            self.vla_train_dataloader.dataset.set_epoch(self.vla_epoch_count)
            if hasattr(self.vla_train_dataloader, "set_epoch"):
                self.vla_train_dataloader.set_epoch(self.vla_epoch_count)
            if microbatches_to_skip:
                resume_loader = self.accelerator.skip_first_batches(
                    self.vla_train_dataloader, num_batches=microbatches_to_skip
                )
                # skip_first_batches constructs a new DataLoaderShard whose iteration starts at
                # zero; carry the epoch across or it would silently reseed the mixture as epoch 0.
                if hasattr(resume_loader, "set_epoch"):
                    resume_loader.set_epoch(self.vla_epoch_count)
                self.vla_iter = iter(resume_loader)
                logger.info(
                    f"Resume dataloader position: epoch={self.vla_epoch_count}, "
                    f"local_microbatches_skipped={microbatches_to_skip}"
                )
                return
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            did_update = bool(self.accelerator.sync_gradients)
            if did_update:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if did_update and self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            if did_update:
                self._log_metrics(step_metrics)

            if did_update and self.completed_steps % self.config.trainer.save_interval == 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        eval_model = self.accelerator.unwrap_model(self.model)
        # A grouped eval batch is one self-contained memory sequence. Start empty so history from a
        # previous eval cannot leak in; clear afterwards so eval state cannot affect later calls.
        if hasattr(eval_model, "reset_memory"):
            eval_model.reset_memory()
        output_dict = eval_model.predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )
        if hasattr(eval_model, "reset_memory"):
            eval_model.reset_memory()

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            nonfinite = (~torch.isfinite(total_loss.detach())).to(dtype=torch.int32)
            if dist.is_initialized():
                dist.all_reduce(nonfinite, op=dist.ReduceOp.MAX)
            if bool(nonfinite.item()):
                raise FloatingPointError(
                    f"Non-finite training loss on at least one rank at optimizer step "
                    f"{self.completed_steps + 1}"
                )

            self.accelerator.backward(total_loss)

            grad_norm = None
            if self.config.trainer.gradient_clipping is not None and self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )
                if torch.is_tensor(grad_norm) and not torch.isfinite(grad_norm.detach()).all():
                    raise FloatingPointError(
                        f"Non-finite gradient norm at optimizer step {self.completed_steps + 1}"
                    )

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
            # AcceleratedOptimizer.zero_grad() is a no-op during non-sync microsteps. Keeping it
            # here (after step) therefore preserves accumulated gradients correctly for G>1.
            self.optimizer.zero_grad()

        # Surface every scalar the framework produced (e.g. action_only_loss and the
        # geo/* auxiliary breakdown) so the full loss composition is visible in wandb,
        # not just the combined optimized loss. `action_dit_loss` is kept as the total
        # optimized loss for continuity with existing runs.
        metrics = {"action_dit_loss": action_loss.item()}
        if torch.is_tensor(grad_norm):
            metrics["grad_norm"] = grad_norm.detach().float().cpu().item()

        def _to_scalar(v):
            if torch.is_tensor(v):
                return v.item() if v.numel() == 1 else None
            if isinstance(v, (int, float)):
                return float(v)
            return None

        for k, v in output_dict.items():
            if k == "action_loss":
                continue  # already logged as action_dit_loss
            scalar = _to_scalar(v)
            if scalar is not None:
                metrics[k] = scalar

        return metrics

    def _finalize_training(self):
        """Training end processing."""
        final_state = Path(self.checkpoint_dir) / f"steps_{self.completed_steps}_state" / "_SUCCESS"
        if not final_state.is_file():
            self._save_checkpoint()

        save_format = getattr(self.config.trainer, "save_format", "pt")
        # Collective for ZeRO-3; cheap/local under ZeRO-2.
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # Seed before constructing new action/GeoPlan/memory parameters, not only the train loop.
    rank = dist.get_rank() if dist.is_initialized() else 0
    set_seed((cfg.seed if hasattr(cfg, "seed") else 3047) + rank)

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    # Training and all final checkpoint collectives have already completed.  On memory-tight
    # full-FT runs NCCL can still request CUDA memory while destroying the communicator; if that
    # allocation fails, propagating the teardown exception incorrectly turns a completed run into
    # a failed torchrun job and prevents serial launchers from starting the next task.  Release
    # cached blocks first and make process-group teardown best-effort.  Process exit remains the
    # ultimate cleanup boundary even when NCCL reports a shutdown-only error.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception as exc:
            logger.warning(f"Final distributed barrier failed after training completed: {exc}")
        try:
            dist.destroy_process_group()
        except Exception as exc:
            logger.warning(f"Distributed teardown failed after training completed; ignoring: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)

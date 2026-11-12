"""Train MXT / MXT+ on one or more embodiments."""
import contextlib
import fcntl
import gc
import math
import os
import random
from datetime import datetime
from itertools import repeat

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm

from egox.data.data import load_data
from egox.data.sampler import capture_rng_state, restore_rng_state
from egox.models.mxt.model import MXT

from .checkpoints import (SNAPSHOT, atomic_save, export_metadata, load_snapshot,
                          resume_model_config, stats_to_numpy, stats_to_tensors)


def compute_dict_mean(dicts):
    """Mean of every (nested) scalar across a list of loss dicts."""
    result = {}
    for key in {k for d in dicts for k in d}:
        values = [d[key] for d in dicts if key in d]
        if isinstance(values[0], dict):
            result[key] = compute_dict_mean(values)
        else:
            result[key] = sum(float(v) for v in values) / len(values)
    return result


def cosine_scale(step, total_steps, min_ratio):
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def save_plot(path, train_losses, val_losses):
    plt.figure()
    plt.plot(np.linspace(0, len(train_losses), len(train_losses)), train_losses, label='train')
    plt.plot(np.linspace(0, len(train_losses), len(val_losses)), val_losses, label='validation')
    plt.tight_layout()
    plt.legend()
    plt.title('loss curve')
    plt.savefig(path)
    plt.close('all')


def train(args, model_cfg, episodes=None):
    """Prepare a locked run, restore training state and execute optimizer updates."""
    torch.cuda.set_device(args.gpu_id)
    run_dir = f'{args.ckpt_dir}_{args.task_name}'
    if not args.auto_resume:
        run_dir += datetime.now().strftime('_%Y%m%d_%H%M%S')
    os.makedirs(run_dir, exist_ok=args.auto_resume)
    with open(os.path.join(run_dir, '.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # one process per run directory
        resume_dir = args.resume_dir
        if resume_dir is None and args.auto_resume and os.path.isfile(os.path.join(run_dir, SNAPSHOT)):
            resume_dir = run_dir
        snapshot = load_snapshot(resume_dir) if resume_dir else None
        norm_stats = None
        if snapshot is not None:
            if os.path.realpath(run_dir) == os.path.realpath(resume_dir):
                export_metadata(snapshot, run_dir)  # Repair stale exports, even for a completed run.
            saved_config = snapshot['config']
            model_cfg = resume_model_config(saved_config['policy_config'], model_cfg)
            args.warn_resume_changes(saved_config)
            norm_stats = stats_to_numpy(snapshot['norm_stats'])
        vel_rebalance = (dict(x=args.vel_rebalance_weight_x, y=args.vel_rebalance_weight_y,
                              yaw=args.vel_rebalance_weight_yaw, threshold=args.vel_rebalance_threshold)
                         if args.vel_rebalance else None)
        train_loader, val_loader, norm_stats, embodiments = load_data(
            args.dataset_dir, model_cfg['embodiment_args'], args.batch_size, args.chunk_size, args.seed,
            per_emb_batch_size=args.per_emb_batch_size, sample_weights=args.sample_weights,
            train_ratio=args.train_ratio,
            main_cam_size=(args.main_cam_height, args.main_cam_width),
            wrist_cam_size=(args.wrist_cam_height, args.wrist_cam_width),
            vel_rebalance=vel_rebalance, norm_stats=norm_stats, episode_index=episodes)

        try:
            if args.wandb:
                run_id = None
                if args.auto_resume and resume_dir == run_dir and snapshot is not None:
                    run_id = snapshot['wandb_run_id']
                wandb.init(project=args.wandb_name, name=os.path.basename(run_dir), id=run_id, resume='allow' if run_id else None)
                wandb.config.update(args.to_dict(), allow_val_change=True)

            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.benchmark = args.cudnn_benchmark
            model = MXT(**model_cfg, norm_stats=norm_stats).cuda()
            optimizer = torch.optim.AdamW(
                model.param_groups(args.lr_tokenizer, args.lr_action_head, args.lr_dream_head, args.lr_trunk),
                lr=args.lr, weight_decay=args.weight_decay)
            base_lrs = [group['lr'] for group in optimizer.param_groups]

            start_step = 0
            if snapshot is not None:
                model.load_state_dict(snapshot['model'])
                optimizer.load_state_dict(snapshot['optimizer'])
                for group, lr in zip(optimizer.param_groups, base_lrs):
                    group['lr'] = lr
                    group['weight_decay'] = args.weight_decay
                start_step = snapshot['step'] + 1
                print(f'Resumed training at step {start_step}')
            elif args.pretrained_path:
                model.load_pretrained(torch.load(args.pretrained_path, map_location='cuda', weights_only=True))
            model.update_ema(decay=None)  # start the EMA encoders from the current BatchNorm statistics
            print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
            if start_step >= args.num_steps:
                print(f'Training already complete after {start_step} updates')
                return
            if snapshot is not None:
                train_loader.load_state_dict(snapshot['train_loader'])
                val_loader.load_state_dict(snapshot['val_loader'])
                restore_rng_state(snapshot['rng'])  # after initialization, which consumes randomness
                snapshot.clear()  # Release restored checkpoint tensors.

            def save_snapshot(step):
                # Store the zero-based index of the last completed update for resume.
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                         'rng': capture_rng_state(), 'train_loader': train_loader.state_dict(),
                         'val_loader': val_loader.state_dict(),
                         'config': {**args.to_dict(), 'policy_config': model_cfg},
                         'norm_stats': stats_to_tensors(norm_stats),
                         'wandb_run_id': wandb.run.id if wandb.run is not None else None}
                atomic_save(state, os.path.join(run_dir, SNAPSHOT))
                export_metadata(state, run_dir)
                atomic_save(state['model'], os.path.join(run_dir, 'policy_last.ckpt'))

            amp = torch.autocast('cuda', dtype=torch.bfloat16) if args.bf16 else contextlib.nullcontext()
            train_iter = (batch for loader in repeat(train_loader) for batch in loader)
            # With --per_emb_batch_size every loader batch holds one embodiment: take one of each per step.
            batches_per_step = len(embodiments) if args.per_emb_batch_size else 1
            train_losses, val_losses = [], []
            for step in tqdm(range(start_step, args.num_steps)):
                if step % args.validate_every == 0:
                    with torch.inference_mode(), amp:
                        model.eval()
                        val_dicts = []
                        for i, batch in enumerate(val_loader):
                            val_dicts.append(model.compute_loss(batch))
                            if i + 1 >= args.max_val_batches:
                                break
                        summary = compute_dict_mean(val_dicts)
                    val_losses.append(summary['all_emb_loss'])
                    print(f"Val loss: {summary['all_emb_loss']:.5f}")
                    if args.wandb:
                        wandb.log({f'val_{k}': v for k, v in summary.items()}, step=step)
                    del val_dicts, summary

                model.train()
                lr_scale = 1.0
                if args.lr_schedule == 'cosine':
                    lr_scale = cosine_scale(step, args.num_steps, args.cosine_min_lr_ratio)
                    for group, lr in zip(optimizer.param_groups, base_lrs):
                        group['lr'] = lr * lr_scale
                optimizer.zero_grad(set_to_none=True)
                # One embodiment at a time, so only its activations are resident; gradients accumulate.
                log = {'all_emb_loss': 0.0}
                for _ in range(batches_per_step):
                    batch = next(train_iter)
                    for emb in batch:
                        with amp:
                            losses = model.compute_loss({emb: batch[emb]})
                        losses['all_emb_loss'].backward()
                        log['all_emb_loss'] += losses['all_emb_loss'].item()
                        log[emb] = losses[emb]
                train_losses.append(log['all_emb_loss'])
                optimizer.step()
                model.update_ema()
                if args.wandb:
                    log['lr_scale'] = lr_scale
                    log['lr_max'] = max(group['lr'] for group in optimizer.param_groups)
                    wandb.log(log, step=step)

                completed_steps = step + 1
                if completed_steps % args.save_every == 0:
                    # Numbered checkpoints are model weights; optimizer state is in training_last.ckpt.
                    torch.save(model.state_dict(), os.path.join(run_dir, f'policy_step_{completed_steps}_seed_{args.seed}.ckpt'))
                    save_plot(os.path.join(run_dir, f'train_val_seed_{args.seed}.png'), train_losses, val_losses)
                if completed_steps % args.snapshot_every == 0:
                    save_snapshot(step)
                    gc.collect()
                del log, batch
            if args.num_steps % args.snapshot_every:
                save_snapshot(args.num_steps - 1)
        finally:
            if args.wandb:
                wandb.finish()

# /// script
# requires-python = ">=3.10"
# dependencies = ["fire", "numpy", "torch", "ogbench", "matplotlib", "scipy", "MQE"]
# ///

"""
stage 2 critic diagnostics on held-out ogbench validation trajectories

- predicted steps-to-go (d / |log gamma|) vs true steps between two states of the same trajectory
- monotonicity: for a fixed start, does the predicted distance grow with the true gap
- action invariance: d(s, (s, a)) in steps, which the paper drives to zero

run

    python examples/ogbench_diagnostics.py --ckpt runs/scene-play-v0/seed0/ckpt_1000000.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr, pearsonr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MQE import MQE, MRN
from ogbench_mqe import EncoderMLP, GPUDataset

class Scaled(nn.Module):
    """pre-branch checkpoints wrapped the mrn heads in a constant scale"""

    def __init__(self, net, scale):
        super().__init__()
        self.net, self.scale = net, scale

    def forward(self, x):
        return self.net(x) * self.scale

def build_mqe(config, obs_dim, action_dim):
    # checkpoints written before the mrn_scale flag existed have unwrapped mrn heads
    wrap = (lambda net: Scaled(net, config['mrn_scale'])) if 'mrn_scale' in config else (lambda net: net)
    return MQE(
        state_encoder = EncoderMLP(obs_dim, config['latent_dim'], config['hidden_dim']),
        state_action_encoder = EncoderMLP(obs_dim + action_dim, config['latent_dim'], config['hidden_dim']),
        metric_residual_network = MRN(
            sym_network = wrap(EncoderMLP(config['latent_dim'], config['latent_dim'], config['hidden_dim'], depth = 1)),
            asym_network = wrap(EncoderMLP(config['latent_dim'], config['latent_dim'], config['hidden_dim'], depth = 1)),
            distance_groups = config['distance_groups'],
            normalize_by_dim = config.get('mrn_normalize_by_dim', False)
        ),
        critic_ensemble = config.get('ensemble', 1),
        discount_factor = config['discount'],
        waypoint_discount = config['waypoint_discount'],
        next_timestep_prob = config['next_timestep_prob'],
        action_invariance_loss_weight = config['action_invariance_loss_weight'],
        paired_loss_weight = config['paired_loss_weight']
    )

@torch.no_grad()
def main(
    ckpt,
    num_pairs = 20000,
    max_gap = 500,
    device = 'cuda',
    seed = 0,
    out_path = None
):
    import ogbench
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    torch.manual_seed(seed)
    ckpt = Path(ckpt)
    state = torch.load(ckpt, map_location = device, weights_only = False)
    config = state['config']
    env_name = config['env_name']

    _, val_dataset = ogbench.make_env_and_datasets(env_name, compact_dataset = True, dataset_only = True)
    data = GPUDataset(val_dataset, device)

    mqe = build_mqe(config, data.obs_dim, data.action_dim).to(device)

    # checkpoints written before the critic-ensemble branch have a single 'critic.' prefix; map them onto member 0 (other members stay at init and are ignored via ensemble_reduce)
    sd = state['mqe']
    if any(k.startswith('critic.') for k in sd):
        sd = {('critics.0.' + k[len('critic.'):]) if k.startswith('critic.') else k: v for k, v in sd.items()}
        missing, unexpected = mqe.load_state_dict(sd, strict = False)
        assert not unexpected, unexpected
        legacy = True
    else:
        mqe.load_state_dict(sd)
        legacy = False
    mqe.eval()

    # pairs (s_t, s_{t+k}) from the same validation trajectory, k uniform in [1, min(max_gap, remaining)]

    t = data.sample_starts(num_pairs)
    remaining = (data.ep_end[t] - t).clamp(max = max_gap)
    k = (torch.rand(num_pairs, device = device) * (remaining - 1)).round().long() + 1
    s, g = data.obs[t], data.obs[t + k]

    red = dict(ensemble_reduce = None) if legacy else {}
    first = (lambda d: d[0]) if legacy else (lambda d: d)
    pred_steps = first(mqe.predict_distance(s, g, return_steps = True, **red))
    pred_steps_q = first(mqe.predict_distance(s, g, actions = data.acts[t], return_steps = True, **red))

    true = k.float().cpu().numpy()
    pred = pred_steps.cpu().numpy()
    pred_q = pred_steps_q.cpu().numpy()

    # monotonicity: same start, two gaps k1 < k2

    t2 = data.sample_starts(num_pairs)
    rem2 = (data.ep_end[t2] - t2).clamp(max = max_gap)
    keep = rem2 >= 2
    t2, rem2 = t2[keep], rem2[keep]
    k1 = (torch.rand(len(t2), device = device) * (rem2 - 1)).round().long() + 1
    k2 = (torch.rand(len(t2), device = device) * (rem2 - 1)).round().long() + 1
    swap = k1 > k2
    k1, k2 = torch.where(swap, k2, k1), torch.where(swap, k1, k2)
    distinct = k2 > k1
    t2, k1, k2 = t2[distinct], k1[distinct], k2[distinct]
    d1 = first(mqe.predict_distance(data.obs[t2], data.obs[t2 + k1], return_steps = True, **red))
    d2 = first(mqe.predict_distance(data.obs[t2], data.obs[t2 + k2], return_steps = True, **red))
    monotonic = float((d2 > d1).float().mean())

    # action invariance: d(s, (s, a)) in steps for dataset (s, a)

    t3 = data.sample_starts(num_pairs)
    enc_s = mqe.critic.state_encoder(data.obs[t3])
    enc_sa = mqe.critic.state_action_encoder((data.obs[t3], data.acts[t3]))
    inv_steps = mqe.critic.metric_residual_network(enc_s, enc_sa) / abs(np.log(config['discount']))
    inv_steps = inv_steps.cpu().numpy()

    # binned calibration: mean predicted steps per true-gap bin

    bins = np.array([1, 2, 5, 10, 20, 50, 100, 200, 350, 501])
    calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (true >= lo) & (true < hi)
        if m.sum() > 0:
            calib.append(dict(gap_lo = int(lo), gap_hi = int(hi - 1), n = int(m.sum()), true_mean = float(true[m].mean()), pred_mean = float(pred[m].mean()), pred_median = float(np.median(pred[m]))))

    results = dict(
        ckpt = str(ckpt),
        env_name = env_name,
        step = state['step'],
        num_pairs = num_pairs,
        spearman_v = float(spearmanr(true, pred).correlation),
        pearson_v = float(pearsonr(true, pred)[0]),
        spearman_q = float(spearmanr(true, pred_q).correlation),
        monotonic_fraction = monotonic,
        num_monotonic_pairs = int(len(t2)),
        action_invariance_steps_mean = float(inv_steps.mean()),
        action_invariance_steps_p95 = float(np.percentile(inv_steps, 95)),
        pred_steps_mean = float(pred.mean()),
        true_steps_mean = float(true.mean()),
        calibration = calib
    )

    out_path = Path(out_path) if out_path is not None else ckpt.parent / f'diagnostics_{state["step"]}.json'
    out_path.write_text(json.dumps(results, indent = 2))

    fig, axes = plt.subplots(1, 2, figsize = (11, 4.5), dpi = 130)
    ax = axes[0]
    ax.scatter(true, pred, s = 2, alpha = 0.15, color = '#2563eb')
    lim = max(true.max(), np.percentile(pred, 99))
    ax.plot([0, lim], [0, lim], color = '#111827', lw = 1, ls = '--', label = 'identity')
    ax.set_xlabel('true steps between states (same trajectory)')
    ax.set_ylabel('predicted steps  d(s, g) / |log γ|')
    ax.set_ylim(0, lim)
    ax.set_title(f'{env_name} @ {state["step"]}: spearman {results["spearman_v"]:.3f}')
    ax.legend()
    ax = axes[1]
    ax.plot([c['true_mean'] for c in calib], [c['pred_mean'] for c in calib], marker = 'o', color = '#2563eb', label = 'mean predicted')
    ax.plot([c['true_mean'] for c in calib], [c['pred_median'] for c in calib], marker = 's', color = '#16a34a', label = 'median predicted')
    ax.plot([0, lim], [0, lim], color = '#111827', lw = 1, ls = '--', label = 'identity')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('true steps (binned)'); ax.set_ylabel('predicted steps')
    ax.set_title(f'monotonic {monotonic:.3f}   d(s,(s,a)) mean {inv_steps.mean():.2f} steps')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_suffix('.png'), bbox_inches = 'tight')

    print(json.dumps({k: v for k, v in results.items() if k != 'calibration'}, indent = 2))
    for c in calib:
        print(f"gap {c['gap_lo']:>3}-{c['gap_hi']:<3}  n {c['n']:>5}  true {c['true_mean']:7.1f}  pred mean {c['pred_mean']:7.1f}  median {c['pred_median']:7.1f}")
    print(f'saved {out_path} and {out_path.with_suffix(".png")}')

if __name__ == '__main__':
    fire.Fire(main)

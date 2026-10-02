# /// script
# requires-python = ">=3.10"
# dependencies = ["fire", "numpy", "torch", "ogbench", "MQE"]
# ///

"""
train MQE on an OGBench state-based dataset and evaluate with the OGBench protocol

defaults follow the authors' released code (critic ensemble of 2, normalized q, paired actor goals, mrn distance / sqrt(dim));
hyperparameters follow table 2 / table 3 of the paper (batch 256, latent 512, MLP (512, 512, 512) with layer norm,
8 MRN components, discount 0.995, waypoint discount 0.95, next-step probability 0.2, 1M gradient steps,
50 evaluation episodes per task at 800k / 900k / 1M steps)

run

    python examples/ogbench_mqe.py --env_name scene-play-v0 --alpha 1.0 --seed 0
    python examples/ogbench_mqe.py --env_name antmaze-giant-stitch-v0 --alpha 0.03 --seed 0
"""

from __future__ import annotations

import json
import time
from math import log
from pathlib import Path

import fire
import numpy as np
import torch
from torch import nn, Tensor
from torch.distributions import Distribution

from MQE import MQE, MRN
from MQE.MQE import ActionDistribution

# networks - ogbench style MLP: (dense -> gelu -> layernorm) x depth -> dense

class EncoderMLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden = 512, depth = 3, layer_norm = True):
        super().__init__()
        layers = []
        d = dim_in
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.GELU())
            if layer_norm:
                layers.append(nn.LayerNorm(hidden))
            d = hidden
        layers.append(nn.Linear(d, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = torch.cat(x, dim = -1)
        return self.net(x)

# ddpg + bc action distribution: deterministic mean, unit-std gaussian log-likelihood for the bc term

class DeterministicDist(Distribution):
    arg_constraints = {}
    has_rsample = True

    def __init__(self, mean: Tensor):
        super().__init__(batch_shape = mean.shape[:-1], event_shape = mean.shape[-1:], validate_args = False)
        self.mean_ = mean

    @property
    def mean(self):
        return self.mean_

    def rsample(self, sample_shape = torch.Size()):
        return self.mean_.clamp(-1., 1.)

    def sample(self, sample_shape = torch.Size()):
        return self.rsample(sample_shape).detach()

    def log_prob(self, value):
        return -0.5 * (value - self.mean_) ** 2

class DDPGAction(ActionDistribution):
    @property
    def expansion_factor(self):
        return 1

    def forward(self, x):
        return DeterministicDist(x)

class GoalConditionedPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden = 512, depth = 3):
        super().__init__()
        self.net = EncoderMLP(obs_dim * 2, action_dim, hidden = hidden, depth = depth)
        self.action_dist = DDPGAction()

    def forward(self, state, goal):
        return self.action_dist(self.net((state, goal)))

# dataset on gpu with trajectory-window sampling

class GPUDataset:
    def __init__(self, dataset, device, segment_len = None):
        obs = torch.from_numpy(dataset['observations']).to(device)
        acts = torch.from_numpy(dataset['actions']).to(device)
        valids = torch.from_numpy(dataset['valids']).to(device).bool()

        # compact dataset: the last state of every trajectory has valids = 0 (no action follows it)

        final_idxs = torch.nonzero(~valids).squeeze(-1)
        ep_end = torch.empty(len(obs), dtype = torch.long, device = device)
        start = 0
        ep_start = torch.empty(len(obs), dtype = torch.long, device = device)
        for e in final_idxs.tolist():
            ep_end[start:e + 1] = e
            ep_start[start:e + 1] = start
            start = e + 1

        # stitching test: cut every trajectory into windows of `segment_len` transitions.
        # adjacent windows share their boundary state (window j covers states s + jL .. s + (j + 1)L),
        # and every sampled waypoint / goal is restricted to the window its start state begins, so no
        # training signal spans more than `segment_len` steps

        if segment_len is not None:
            idx = torch.arange(len(obs), device = device)
            chunk_end = ep_start + ((idx - ep_start) // segment_len + 1) * segment_len
            ep_end = torch.minimum(chunk_end, ep_end)

        self.obs, self.acts, self.ep_end = obs, acts, ep_end
        self.segment_len = segment_len
        self.valid_idxs = torch.nonzero(valids).squeeze(-1)
        self.max_window = int((ep_end[self.valid_idxs] - self.valid_idxs).max()) + 1
        self.device = device
        self.obs_dim, self.action_dim = obs.shape[-1], acts.shape[-1]
        self.num_episodes = len(final_idxs)
        self.num_segments = int((self.ep_end[self.valid_idxs].unique()).numel())

    def sample_starts(self, batch):
        return self.valid_idxs[torch.randint(0, len(self.valid_idxs), (batch,), device = self.device)]

    def sample_windows(self, batch, discount):
        """windows s_t .. s_{t+K} with K ~ min(Geom(1 - discount), steps-to-episode-end), K >= 1"""
        t = self.sample_starts(batch)
        remaining = self.ep_end[t] - t
        k = torch.empty((batch,), device = self.device).geometric_(1. - discount).long().clamp(min = 1) # cuda geometric_ can return 0
        k = torch.minimum(k, remaining)
        lens = k + 1

        # static window length (max steps-to-episode-end + 1) so no host sync is needed per step

        idxs = t[:, None] + torch.arange(self.max_window, device = self.device)[None, :]
        idxs = torch.minimum(idxs, self.ep_end[t][:, None])

        return self.obs[idxs], self.acts[idxs], lens

    def sample_actor_batch(self, batch):
        """(s_t, a_t, g) with g uniformly sampled from the future of the same trajectory"""
        t = self.sample_starts(batch)
        remaining = self.ep_end[t] - t
        offset = (torch.rand((batch,), device = self.device) * remaining).round().long().clamp(min = 1)
        g = self.obs[t + offset]
        return self.obs[t], self.acts[t], g

# evaluation following ogbench

@torch.no_grad()
def evaluate(env, policy, device, num_episodes, num_tasks = 5, act_mean = None, act_std = None):
    # single-observation inference is faster on cpu than on a busy gpu, so evaluate a cpu copy

    import copy
    policy = copy.deepcopy(policy).cpu().eval()
    device = 'cpu'
    results = {}
    for task_id in range(1, num_tasks + 1):
        successes = []
        for _ in range(num_episodes):
            ob, info = env.reset(options = dict(task_id = task_id, render_goal = False))
            goal = torch.from_numpy(np.asarray(info['goal'], dtype = np.float32)).to(device)[None]
            done = False
            success = 0.
            while not done:
                state = torch.from_numpy(np.asarray(ob, dtype = np.float32)).to(device)[None]
                action = policy(state, goal).mean[0]
                if act_mean is not None:
                    action = action * act_std + act_mean   # back to the env's raw action space
                action = action.clamp(-1., 1.).cpu().numpy()
                ob, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                success = float(info['success'])
            successes.append(success)
        results[f'task{task_id}'] = float(np.mean(successes))
    results['overall'] = float(np.mean([results[f'task{i}'] for i in range(1, num_tasks + 1)]))
    return results

# main

def main(
    env_name = 'scene-play-v0',
    seed = 0,
    alpha = 1.0,
    steps = 1_000_000,
    batch_size = 256,
    lr = 3e-4,
    discount = 0.995,
    waypoint_discount = 0.95,
    next_timestep_prob = 0.2,
    latent_dim = 512,
    hidden_dim = 512,
    distance_groups = 8,
    paired_loss_weight = 0.5,
    action_invariance_loss_weight = 1.0,
    eval_steps = (200_000, 400_000, 600_000, 800_000, 900_000, 1_000_000),
    eval_episodes = 50,
    log_interval = 1000,
    out_dir = None,
    device = 'cuda',
    compile = True,
    q_normalize = True,            # divide the actor's q term by its detached mean magnitude (ogbench ddpg+bc convention, authors' code)
    ensemble = 2,                  # critic ensemble size; the actor uses the pessimistic member (authors' code)
    mrn_normalize_by_dim = True,   # divide the mrn distance by sqrt(latent_dim) (authors' code)
    actor_goal = 'paired',         # 'paired': same-trajectory future goal for the q term (authors' code); 'random': goals permuted across the batch (eq. 15 / pre-branch repo)
    actor_final_init_scale = 0.01, # small init of the actor's mean layer (authors' code); None for the default init
    gcbc = False,                  # goal-conditioned behavior cloning baseline: actor trained with the bc term only, critic not trained
    normalize_actions = False,     # standardize each action dimension by its dataset mean / std for training; the policy is un-normalized at execution
    segment_len = None             # stitching test: train only on windows of this many transitions cut from each trajectory (None = full trajectories)
):
    import ogbench

    if env_name.startswith('maniskill'):
        # registers the goal-conditioned maniskill envs and expects the converted datasets in ~/.ogbench/data (see examples/maniskill_gc.py)
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import maniskill_gc  # noqa: F401

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(out_dir) if out_dir is not None else Path('runs') / env_name / f'seed{seed}'
    out_dir.mkdir(parents = True, exist_ok = True)

    config = {k: v for k, v in locals().items() if k in (
        'env_name', 'seed', 'alpha', 'steps', 'batch_size', 'lr', 'discount', 'waypoint_discount', 'next_timestep_prob',
        'latent_dim', 'hidden_dim', 'distance_groups', 'paired_loss_weight', 'action_invariance_loss_weight', 'eval_episodes', 'q_normalize', 'ensemble', 'mrn_normalize_by_dim', 'actor_goal', 'actor_final_init_scale', 'gcbc', 'normalize_actions', 'segment_len')}
    (out_dir / 'config.json').write_text(json.dumps(config, indent = 2))

    import os
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name, compact_dataset = True, dataset_dir = os.environ.get('OGBENCH_DATA_DIR', '~/.ogbench/data'))
    data = GPUDataset(train_dataset, device, segment_len = segment_len)
    obs_dim, action_dim = data.obs_dim, data.action_dim

    # optional per-dimension action standardization (statistics over transitions that have an action)

    act_mean = act_std = None
    q_action_clamp = (-1., 1.)

    if normalize_actions:
        valid_acts = data.acts[data.valid_idxs]
        act_mean = valid_acts.mean(dim = 0)
        act_std = valid_acts.std(dim = 0).clamp(min = 1e-3)
        data.acts = (data.acts - act_mean) / act_std
        q_action_clamp = ((-1. - act_mean) / act_std, (1. - act_mean) / act_std)   # the env's [-1, 1] bounds in normalized space
        print('action mean', [round(v, 3) for v in act_mean.tolist()], 'std', [round(v, 3) for v in act_std.tolist()])
    print(f'{env_name}: {len(data.obs)} states, {data.num_episodes} episodes, {data.num_segments} training segments (segment_len {segment_len}), obs {obs_dim}, act {action_dim}')

    mqe = MQE(
        state_encoder = EncoderMLP(obs_dim, latent_dim, hidden_dim),
        state_action_encoder = EncoderMLP(obs_dim + action_dim, latent_dim, hidden_dim),
        metric_residual_network = MRN(
            sym_network = EncoderMLP(latent_dim, latent_dim, hidden_dim, depth = 1),
            asym_network = EncoderMLP(latent_dim, latent_dim, hidden_dim, depth = 1),
            distance_groups = distance_groups,
            normalize_by_dim = mrn_normalize_by_dim
        ),
        discount_factor = discount,
        waypoint_discount = waypoint_discount,
        next_timestep_prob = next_timestep_prob,
        action_invariance_loss_weight = action_invariance_loss_weight,
        paired_loss_weight = paired_loss_weight,
        critic_ensemble = ensemble
    ).to(device)

    policy = GoalConditionedPolicy(obs_dim, action_dim, hidden_dim).to(device)

    if actor_final_init_scale is not None:
        # the authors initialise the actor's mean layer with a small (1e-2) variance-scaling init so initial actions are near zero
        final = policy.net.net[-1]
        nn.init.orthogonal_(final.weight, gain = actor_final_init_scale)
        nn.init.zeros_(final.bias)

    critic_params = list(mqe.parameters())

    # torch.compile fuses the (batch x batch x groups x dim) quasimetric elementwise chain, ~1.9x faster on a 3080 ti

    def critic_loss_fn(s, a, l):
        return mqe(s, a, lens = l)

    def policy_loss_fn(s, a, g):
        return mqe.extract_policy(
            policy, s, a, g,
            bc_loss_weight = alpha,
            normalize_q = q_normalize,
            cross_batch_goals = (actor_goal == 'random'),
            action_clamp = q_action_clamp
        )

    if gcbc:
        def policy_loss_fn(s, a, g):
            bc_loss = -policy(s, g).log_prob(a).sum(dim = -1).mean()
            return alpha * bc_loss, (torch.zeros((), device = s.device), bc_loss)

    if compile:
        critic_loss_fn = torch.compile(critic_loss_fn)
        policy_loss_fn = torch.compile(policy_loss_fn)

    critic_opt = torch.optim.Adam(critic_params, lr = lr)
    policy_opt = torch.optim.Adam(policy.parameters(), lr = lr)

    log_file = open(out_dir / 'train_log.csv', 'w')
    log_file.write('step,critic_loss,multistep_loss,action_invariance_loss,policy_loss,q_loss,bc_loss,mean_window_len,steps_per_sec\n')
    eval_results = []

    accum = torch.zeros(7, device = device)
    last_time = time.time()

    for step in range(1, steps + 1):

        # critic

        states, actions, lens = data.sample_windows(batch_size, discount)

        if gcbc:
            critic_loss = multistep_loss = invariance_loss = torch.zeros((), device = device)
        else:
            critic_loss, (multistep_loss, invariance_loss) = critic_loss_fn(states, actions, lens)

            critic_opt.zero_grad(set_to_none = True)
            critic_loss.backward()
            critic_opt.step()

        # policy (ddpg + bc); critic frozen for this step

        for p in critic_params:
            p.requires_grad_(False)

        s, a, g = data.sample_actor_batch(batch_size)
        policy_loss, (q_loss, bc_loss) = policy_loss_fn(s, a, g)

        policy_opt.zero_grad(set_to_none = True)
        policy_loss.backward()
        policy_opt.step()

        for p in critic_params:
            p.requires_grad_(True)

        accum += torch.stack([t.detach() for t in (critic_loss, multistep_loss, invariance_loss, policy_loss, q_loss, bc_loss, lens.float().mean())])

        if step % log_interval == 0:
            accum_host = accum.tolist()
            now = time.time()
            sps = log_interval / (now - last_time)
            last_time = now
            row = [step] + [v / log_interval for v in accum_host] + [sps]
            log_file.write(','.join(f'{v:.6g}' if isinstance(v, float) else str(v) for v in row) + '\n')
            log_file.flush()
            print(f'step {step:>8}  critic {row[1]:.4f}  multistep {row[2]:.4f}  inv {row[3]:.5f}  policy {row[4]:.4f}  q {row[5]:.4f}  bc {row[6]:.4f}  win {row[7]:.1f}  {sps:.1f} it/s', flush = True)
            if not np.isfinite(row[1]):
                raise RuntimeError('non-finite critic loss')
            accum.zero_()

        if step in eval_steps:
            t0 = time.time()
            result = evaluate(env, policy, device, eval_episodes, act_mean = None if act_mean is None else act_mean.cpu(), act_std = None if act_std is None else act_std.cpu())
            result['step'] = step
            result['eval_seconds'] = time.time() - t0
            eval_results.append(result)
            (out_dir / 'eval.json').write_text(json.dumps(eval_results, indent = 2))
            print(f'eval @ {step}: ' + '  '.join(f'{k} {v:.3f}' for k, v in result.items() if k.startswith(('task', 'overall'))) + f'  ({result["eval_seconds"]:.0f}s)', flush = True)

            torch.save(dict(mqe = mqe.state_dict(), policy = policy.state_dict(), config = config, step = step, act_mean = act_mean, act_std = act_std), out_dir / f'ckpt_{step}.pt')
            last_time = time.time()

    log_file.close()

    # protocol number: mean of the last three evaluations (800k / 900k / 1M on the paper's schedule)
    final = [r['overall'] for r in eval_results[-3:]]
    summary = dict(config = config, eval = eval_results, final_success = float(np.mean(final)) if final else None)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent = 2))
    print(f'final success (mean of 800k/900k/1M): {summary["final_success"]}')

if __name__ == '__main__':
    fire.Fire(main)

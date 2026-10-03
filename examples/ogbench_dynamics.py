"""
state-space dynamics model for ogbench datasets: ensemble of mlps predicting the normalized state delta

    python examples/ogbench_dynamics.py --env_name cube-single-play-v0 --out runs/dynamics/cube-single-play-v0.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np
import torch
from torch import nn

class DynamicsEnsemble(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden = 512, depth = 3, members = 5):
        super().__init__()
        def mlp():
            layers, d = [], obs_dim + act_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.GELU(), nn.LayerNorm(hidden)]
                d = hidden
            layers.append(nn.Linear(d, obs_dim))
            return nn.Sequential(*layers)
        self.members = nn.ModuleList([mlp() for _ in range(members)])
        self.register_buffer('obs_mean', torch.zeros(obs_dim)); self.register_buffer('obs_std', torch.ones(obs_dim))
        self.register_buffer('act_mean', torch.zeros(act_dim)); self.register_buffer('act_std', torch.ones(act_dim))
        self.register_buffer('delta_mean', torch.zeros(obs_dim)); self.register_buffer('delta_std', torch.ones(obs_dim))

    def forward(self, obs, act, member = None):
        """next observation in raw units; member None -> mean over the ensemble"""
        x = torch.cat([(obs - self.obs_mean) / self.obs_std, (act - self.act_mean) / self.act_std], dim = -1)
        if member is not None:
            d = self.members[member](x)
        else:
            d = torch.stack([m(x) for m in self.members]).mean(0)
        return obs + d * self.delta_std + self.delta_mean

    def forward_all(self, obs, act):
        x = torch.cat([(obs - self.obs_mean) / self.obs_std, (act - self.act_mean) / self.act_std], dim = -1)
        d = torch.stack([m(x) for m in self.members])
        return obs.unsqueeze(0) + d * self.delta_std + self.delta_mean

def transitions(dataset, device):
    obs = torch.from_numpy(dataset['observations']).to(device)
    acts = torch.from_numpy(dataset['actions']).to(device)
    valids = torch.from_numpy(dataset['valids']).to(device).bool()
    idx = torch.nonzero(valids).squeeze(-1)           # states that have an action and a successor in the same episode
    return obs[idx], acts[idx], obs[idx + 1]

def main(env_name = 'cube-single-play-v0', out = None, steps = 40000, batch = 1024, lr = 1e-3, members = 5, device = 'cuda', seed = 0):
    import ogbench
    torch.manual_seed(seed)
    train, val = ogbench.make_env_and_datasets(env_name, compact_dataset = True, dataset_only = True)
    s, a, s2 = transitions(train, device)
    vs, va, vs2 = transitions(val, device)
    model = DynamicsEnsemble(s.shape[-1], a.shape[-1], members = members).to(device)
    delta = s2 - s
    model.obs_mean.copy_(s.mean(0)); model.obs_std.copy_(s.std(0).clamp(min = 1e-3))
    model.act_mean.copy_(a.mean(0)); model.act_std.copy_(a.std(0).clamp(min = 1e-3))
    model.delta_mean.copy_(delta.mean(0)); model.delta_std.copy_(delta.std(0).clamp(min = 1e-4))
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n = len(s)
    for step in range(1, steps + 1):
        # each member sees its own bootstrap batch
        loss = 0.
        for m in range(members):
            i = torch.randint(0, n, (batch,), device = device)
            pred = model(s[i], a[i], member = m)
            loss = loss + (((pred - s2[i]) - model.delta_mean) / model.delta_std).pow(2).mean()
        opt.zero_grad(set_to_none = True); loss.backward(); opt.step(); sched.step()
        if step % 5000 == 0:
            with torch.no_grad():
                v1 = (model(vs, va) - vs2).pow(2).mean().sqrt().item()
            print(f'step {step}: train loss {loss.item() / members:.4f}  val 1-step rmse (raw) {v1:.5f}', flush = True)

    # open-loop rollout error on validation episodes
    model.eval()
    with torch.no_grad():
        vobs = torch.from_numpy(val['observations']).to(device); vact = torch.from_numpy(val['actions']).to(device)
        vvalid = torch.from_numpy(val['valids']).to(device).bool()
        H = 20
        starts = torch.nonzero(vvalid).squeeze(-1)
        ok = torch.ones_like(starts, dtype = torch.bool)
        for h in range(H):
            ok &= vvalid[(starts + h).clamp(max = len(vvalid) - 1)]
        starts = starts[ok][:4096]
        pred = vobs[starts]
        errs = []
        for h in range(H):
            pred = model(pred, vact[starts + h])
            errs.append((pred - vobs[starts + h + 1]).norm(dim = -1).mean().item())
        scale = (vobs[starts + H] - vobs[starts]).norm(dim = -1).mean().item()
        print('open-loop rollout error (L2 in raw obs units) at h = 1, 5, 10, 20:', [round(errs[i], 4) for i in (0, 4, 9, 19)], f'| mean true displacement over 20 steps: {scale:.4f}')

    out = Path(out) if out else Path('runs/dynamics') / f'{env_name}.pt'
    out.parent.mkdir(parents = True, exist_ok = True)
    torch.save(dict(state_dict = model.state_dict(), obs_dim = s.shape[-1], act_dim = a.shape[-1], members = members, env_name = env_name, rollout_errs = errs), out)
    print('saved', out)

if __name__ == '__main__':
    fire.Fire(main)

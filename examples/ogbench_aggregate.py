"""aggregate stage 1 eval results across arms and seeds and compare with the paper's table 4

arms:
  raw        runs/<env>/seed*/eval.json            repo as written
  faithful   runs/faithful/<env>/seed*/eval.json   repo critic + actor matching the authors' code (harness-side, pre-branch)
  branch     runs/branch/<env>/seed*/eval.json     match-authors-actor branch, library defaults
  reference  runs/reference/OGBench/<env>/*/eval.csv   the authors' jax implementation
"""

import csv
import json
from pathlib import Path

import fire
import numpy as np

PAPER = {
    # env: (mqe success %, best baseline success %, best baseline name, gcbc success %) from table 4 of the paper (8 seeds)
    'scene-play-v0': (76.8, 51.3, 'GCIQL', 5.4),
    'antmaze-giant-stitch-v0': (35.1, 9.2, 'TMD', 2.7),
    # cube-single-play is not in the MQE paper; baselines from the OGBench paper (arXiv 2410.20092) table 2: GCBC 6, GCIVL 53, GCIQL 68, QRL 5, CRL 19, HIQL 15
    'cube-single-play-v0': (float('nan'), 68.0, 'GCIQL (OGBench paper)', 6.0),
    # maniskill: no published goal-conditioned numbers; final = mean of the last three evaluations of the 300k schedule
    'maniskill-pickcube-play-v0': (float('nan'), float('nan'), 'n/a', float('nan')),
    'maniskill-pushcube-play-v0': (float('nan'), float('nan'), 'n/a', float('nan')),
}
FINAL_STEPS = (800_000, 900_000, 1_000_000)

def load_harness_curve(seed_dir):
    f = seed_dir / 'eval.json'
    if not f.exists():
        return None
    return {e['step']: e['overall'] for e in json.loads(f.read_text())}

def load_reference_curve(exp_dir):
    f = exp_dir / 'eval.csv'
    if not f.exists():
        return None
    curve = {}
    for row in csv.DictReader(open(f)):
        step = int(float(row['step']))
        if step > 1:
            curve[step] = float(row['evaluation/overall_success'])
    return curve

def collect(runs_dir):
    runs_dir = Path(runs_dir)
    arms = {}
    for env in PAPER:
        arms.setdefault('raw', {})[env] = {d.name: c for d in sorted((runs_dir / env).glob('seed*')) if (c := load_harness_curve(d))} if (runs_dir / env).exists() else {}
        arms.setdefault('faithful', {})[env] = {d.name: c for d in sorted((runs_dir / 'faithful' / env).glob('seed*')) if (c := load_harness_curve(d))} if (runs_dir / 'faithful' / env).exists() else {}
        arms.setdefault('branch', {})[env] = {d.name: c for d in sorted((runs_dir / 'branch' / env).glob('seed*')) if (c := load_harness_curve(d))} if (runs_dir / 'branch' / env).exists() else {}
        arms.setdefault('branch_alpha3', {})[env] = {d.name: c for d in sorted((runs_dir / 'branch_alpha3' / env).glob('seed*')) if (c := load_harness_curve(d))} if (runs_dir / 'branch_alpha3' / env).exists() else {}
        arms.setdefault('branch_alpha10', {})[env] = {d.name: c for d in sorted((runs_dir / 'branch_alpha10' / env).glob('seed*')) if (c := load_harness_curve(d))} if (runs_dir / 'branch_alpha10' / env).exists() else {}
        arms.setdefault('branch_norm', {})[env] = {d.name: c for d in sorted((runs_dir / 'branch_norm' / env).glob('seed*')) if (c := load_harness_curve(d))} if (runs_dir / 'branch_norm' / env).exists() else {}
        arms.setdefault('gcbc_norm', {})[env] = {d.name: c for d in sorted((runs_dir / 'gcbc_norm' / env).glob('seed*')) if (c := load_harness_curve(d))} if (runs_dir / 'gcbc_norm' / env).exists() else {}
        arms.setdefault('gcbc', {})[env] = {d.name: c for d in sorted((runs_dir / 'gcbc' / env).glob('seed*')) if (c := load_harness_curve(d))} if (runs_dir / 'gcbc' / env).exists() else {}
        ref_dir = runs_dir / 'reference' / 'OGBench' / env
        arms.setdefault('reference', {})[env] = {f'seed{i}': c for i, d in enumerate(sorted(ref_dir.glob('*'))) if (c := load_reference_curve(d))} if ref_dir.exists() else {}
    return arms

def main(runs_dir = 'runs'):
    arms = collect(runs_dir)
    lines = []
    for env, (paper_mqe, paper_base, base_name, paper_gcbc) in PAPER.items():
        lines.append(f'## {env}   (paper table 4: MQE {paper_mqe}, best baseline {paper_base} {base_name}, GCBC {paper_gcbc})')
        lines.append('')
        steps = sorted({s for arm in arms.values() for c in arm.get(env, {}).values() for s in c})
        lines.append('| arm | seed | ' + ' | '.join(f'{s // 1000}k' for s in steps) + ' | final (mean 800k-1M) |')
        lines.append('|---|---|' + '---|' * (len(steps) + 1))
        for arm_name, arm in arms.items():
            finals = []
            for seed, c in arm.get(env, {}).items():
                row = [f'{c[s] * 100:.1f}' if s in c else '-' for s in steps]
                fin = [c[s] for s in FINAL_STEPS if s in c] if not env.startswith('maniskill') else [c[s] for s in sorted(c)[-3:]]
                final = f'{np.mean(fin) * 100:.1f}' if len(fin) == 3 else ('(partial) ' + f'{np.mean(fin) * 100:.1f}' if fin else '-')
                if len(fin) == 3:
                    finals.append(np.mean(fin) * 100)
                lines.append(f'| {arm_name} | {seed} | ' + ' | '.join(row) + f' | {final} |')
            if len(finals) > 1:
                lines.append(f'| {arm_name} | mean | ' + ' | '.join('' for _ in steps) + f' | **{np.mean(finals):.1f} ± {np.std(finals, ddof = 1) / np.sqrt(len(finals)):.1f}** |')
        lines.append('')
    out = '\n'.join(lines)
    print(out)

if __name__ == '__main__':
    fire.Fire(main)

"""collect stage 2 diagnostics json files into one markdown table"""

import json
from pathlib import Path

import fire

def main(runs_dir = 'runs'):
    rows = []
    for f in sorted(Path(runs_dir).glob('**/diagnostics_*.json')):
        if 'smoke' in f.parts:
            continue
        d = json.loads(f.read_text())
        arm = f.parts[1] if f.parts[1] in ('faithful', 'qnorm', 'gcbc', 'branch', 'branch_alpha3', 'branch_alpha10', 'branch_norm', 'gcbc_norm') else 'raw'
        env = f.parts[2] if arm != 'raw' else f.parts[1]
        seed = f.parent.name
        rows.append((env, arm, seed, d['step'], d['spearman_v'], d['spearman_q'], d['monotonic_fraction'], d['action_invariance_steps_mean'], d['action_invariance_steps_p95'], d['calibration']))
    rows.sort()
    lines = ['| env | arm | seed | step | spearman d(s,g) | spearman d((s,a),g) | monotonic frac | d(s,(s,a)) mean steps | p95 | pred steps for true gap 1 / 10-19 / 100-199 (mean) |', '|---|---|---|---|---|---|---|---|---|---|']
    for env, arm, seed, step, sv, sq, mono, inv, inv95, calib in rows:
        c = {f"{x['gap_lo']}-{x['gap_hi']}": x['pred_mean'] for x in calib}
        g1 = c.get('1-1', float('nan')); g10 = c.get('10-19', float('nan')); g100 = c.get('100-199', float('nan'))
        lines.append(f'| {env} | {arm} | {seed} | {step // 1000}k | {sv:.2f} | {sq:.2f} | {mono:.2f} | {inv:.1f} | {inv95:.1f} | {g1:.1f} / {g10:.1f} / {g100:.1f} |')
    print('\n'.join(lines))

if __name__ == '__main__':
    fire.Fire(main)

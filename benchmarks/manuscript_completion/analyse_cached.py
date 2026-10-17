"""Recompute cache reuse and calibration surcharges from saved warm samples."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def validate_samples(path):
    rows=read_csv(path);samples=read_csv(str(path)+'.samples.csv')
    for r in rows:
        for operation,column in [('cache_build','cache_build_ms'),('spectrum_lookup','cached_spectrum_select_ms'),('cached_factor','cached_factor_ms'),('independent_recompute','independent_recompute_ms')]:
            values=[float(s['ms']) for s in samples if s['operation']==operation and s['rank'] in ('0',r['rank'])]
            if len(values)!=int(r['repeats']) or not math.isclose(float(r[column]),statistics.median(values),rel_tol=1e-12):raise ValueError('median/sample mismatch: '+str(path))
    return rows


def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=root/'results_v2');p.add_argument('--out',type=Path,default=root);p.add_argument('--validation',type=Path,default=root/'cached_validation.json');a=p.parse_args()
    paths=sorted(path for path in a.results.glob('*.csv') if not path.name.endswith('.samples.csv'))
    rows=[r for path in paths for r in validate_samples(path)]
    calibration_paths=sorted(path for path in (a.results/'calibration').glob('*.csv') if not path.name.endswith('.samples.csv'))
    calibration_rows=rows+[r for path in calibration_paths for r in validate_samples(path)]
    calibration={}
    for name in sorted({r['matrix'] for r in rows}):
        grid=[r for r in calibration_rows if r['matrix']==name and r['rank']=='16']
        if len(grid)!=9:raise ValueError('calibration requires nine states per matrix')
        calibration[name]=sum(float(r['independent_recompute_ms']) for r in grid)
    out=[]
    for r in rows:
        build,cached,fresh=map(float,(r['cache_build_ms'],r['cached_factor_ms'],r['independent_recompute_ms']))
        saving=fresh-cached
        out.append(dict(r,per_use_saving_ms=saving,break_even_reuses=math.ceil(build/saving) if saving>0 else None,
            rank16_grid_calibration_ms=calibration[r['matrix']] if r['rank']=='16' else None,
            break_even_including_rank16_grid=math.ceil((calibration[r['matrix']]+build)/saving) if saving>0 and r['rank']=='16' else None))
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'cached_costs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in out for k in r}));w.writeheader();w.writerows(out)
    with (a.out/'calibration_grid.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(calibration_rows[0]));w.writeheader();w.writerows(calibration_rows)
    validation=json.loads(a.validation.read_text(encoding='utf-8'))
    lines=['# Resident cache and calibration cost','',
        'The break-even counts below compare against independent solves for every request. They do not compare against an optimal final-output cache: repeated identical outputs can be retained, and a known maximum-rank factorization supports smaller-rank prefixes. The reviewer revision uses the explicit three-rank cost model in reviewer_evidence.json instead of these repeated-rank counts as a manuscript claim.',
        '',
        'All medians use seven saved warm samples. The first cache build is recorded separately as cold and excluded from these medians. Effective input, seed, width40, power2, precision and output are matched. Timings use synchronized wall time and include per-operation allocations; export and independent FP64 checks are outside. Spectrum lookup is a single-value copy, not an adaptive rank-selection algorithm.',
        '', '| matrix | k | build ms | cached factors ms | fresh solve ms | cache break-even | with rank16 nine-state grid |','|---|---:|---:|---:|---:|---:|---:|']
    for r in out:
        lines.append(f"| {r['matrix']} | {r['rank']} | {float(r['cache_build_ms']):.3f} | {float(r['cached_factor_ms']):.3f} | {float(r['independent_recompute_ms']):.3f} | {r['break_even_reuses']} | {r['break_even_including_rank16_grid'] or '-'} |")
    lines+=['', 'For N repeated requests on an unchanged matrix and retained state, Ccache(N)=Cbuild+N*Cfactor and Cfresh(N)=N*Tfresh. The cache-only break-even is ceil(Cbuild/(Tfresh-Cfactor)). A separate conservative scenario pays Ccal for all nine complete-factor candidates at rank16 and rebuilds the chosen cache, giving ceil((Ccal+Cbuild)/(Tfresh-Cfactor)); Ccal='+str(calibration)+'. The predeclared (40,2) state isolates reuse, not target attainment or the fastest feasible setting. CUDA-event replay costs are not mixed with these wall-time costs.',
        '',f"Six cached/fresh pairs: max direct residual difference {max(abs(x['cached_residual']-x['independent_residual']) for x in validation):.4e}; max cached U/V orthogonality {max(max(x['cached_u_orth'],x['cached_v_orth']) for x in validation):.4e}."]
    (a.out/'cached_reuse_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest=dict(repeats=7,warmups=1,seed=20261015,calibration='nine complete-factor medians at rank16 plus a new cache build',source_sha256=hashlib.sha256((root/'cached_reuse.cu').read_bytes()).hexdigest(),sample_files={str(path.relative_to(a.results)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(a.results.rglob('*.samples.csv'))})
    (a.out/'corrected_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':main()

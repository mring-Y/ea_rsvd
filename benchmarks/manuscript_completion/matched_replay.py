"""Post hoc, fixed-rank routing ablation with a common output-quality test.

Unlike the historical monitor ablation, both searches accept only rho_k<=tau.
This consumes stored full-energy diagnostics and measured CUDA-event costs;
it does not simulate incremental reuse or time a live controller.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

WIDTHS=(24,32,40)
ORDER=[(w,q) for w in WIDTHS for q in (0,1,2)]
POLICIES=('fixed_largest','rank_residual','residual_pair')


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)


def replay(states,policy,tau=.2,budget=9):
    current=(40,2) if policy=='fixed_largest' else (24,0)
    steps=[];observed=[];attained=False
    for step in range(budget):
        row=dict(states[current]);row['step']=step;steps.append(row)
        if row['valid']:
            observed.append(row)
            if row['rho_k']<=tau:
                attained=True;break
        if policy=='fixed_largest':
            break
        w,q=current
        if policy=='rank_residual' or not row['valid']:
            candidates=[key for key in ORDER if key not in {(x['width'],x['power']) for x in steps}]
            if not candidates:break
            current=candidates[0]
        elif row['rho_l']>tau and q<2:
            current=(w,q+1)
        elif w<40:
            current=(WIDTHS[WIDTHS.index(w)+1],0)
        elif q<2:
            current=(w,q+1)
        else:
            break
        visited={(r['width'],r['power']) for r in steps}
        if current in visited:
            remaining=[key for key in ORDER if key not in visited]
            if not remaining:break
            current=remaining[0]
    best=min(observed,key=lambda x:(x['rho_k'],x['cost_ms'],x['width'],x['power'])) if observed else None
    known=[r['cost_ms'] for r in steps if r['cost_ms'] is not None and math.isfinite(r['cost_ms'])]
    result=dict(policy=policy,attained=attained,evaluated_states=len(steps),invalid_states=sum(not r['valid'] for r in steps),
        cumulative_ms=sum(known) if len(known)==len(steps) else None,known_cumulative_ms=sum(known),unmeasured_cost_states=len(steps)-len(known),
        output_rho_k=best['rho_k'] if best else None,output_rho_l=best['rho_l'] if best else None,
        output_width=best['width'] if best else None,output_power=best['power'] if best else None,
        stop='target' if attained else ('single_fixed_state' if policy=='fixed_largest' else 'route_exhausted'))
    return result,steps


def analyse(path,out,dataset):
    groups={}
    for r in read_csv(path):
        w,q=int(r['width']),int(r['power'])
        rl,rk=float(r.get('exact_projection_residual') or 'nan'),float(r.get('exact_rank_residual') or 'nan')
        orth=float(r.get('q_orthogonality_max_error') or 'nan')
        cost=float(r.get('gpu_pipeline_ms') or 'nan')
        valid=r.get('status',r.get('state_status','ok')) in ('ok','executed') and all(math.isfinite(x) for x in (rl,rk,orth,cost)) and orth<=1e-4 and cost>=0
        groups.setdefault((r['matrix'],r['seed']),{})[(w,q)]=dict(matrix=r['matrix'],seed=r['seed'],width=w,power=q,rho_l=rl,rho_k=rk,cost_ms=cost if math.isfinite(cost) else None,valid=valid)
    results=[];paths=[];offline=[]
    for (matrix,seed),states in sorted(groups.items()):
        if set(states)!=set(ORDER):raise ValueError('incomplete grid: '+matrix+'/'+seed)
        for policy in POLICIES:
            result,steps=replay(states,policy)
            results.append(dict(dataset=dataset,matrix=matrix,seed=seed,**result))
            paths.extend(dict(dataset=dataset,policy=policy,**step) for step in steps)
        feasible=[r for r in states.values() if r['valid'] and r['rho_k']<=.2]
        best=min(feasible,key=lambda r:(r['cost_ms'],r['rho_k'],r['width'],r['power'])) if feasible else None
        costs=[r['cost_ms'] for r in states.values()]
        offline.append(dict(matrix=matrix,seed=seed,attained=bool(best),calibration_ms=sum(costs) if all(c is not None for c in costs) else None,selected_ms=best['cost_ms'] if best else None))
    summaries=[]
    for matrix in sorted({r['matrix'] for r in results})+['ALL']:
        for policy in POLICIES:
            group=[r for r in results if r['policy']==policy and (matrix=='ALL' or r['matrix']==matrix)]
            summaries.append(dict(dataset=dataset,matrix=matrix,policy=policy,groups=len(group),attained=sum(r['attained'] for r in group),
                mean_states=statistics.fmean(r['evaluated_states'] for r in group),mean_cumulative_ms=statistics.fmean(r['cumulative_ms'] for r in group) if all(r['cumulative_ms'] is not None for r in group) else None,unknown_cost_groups=sum(r['cumulative_ms'] is None for r in group),
                mean_output_rho_k=statistics.fmean(r['output_rho_k'] for r in group if r['output_rho_k'] is not None),invalid_states=sum(r['invalid_states'] for r in group)))
    out=Path(out);write_csv(out/'matched_policy_groups.csv',results);write_csv(out/'matched_policy_paths.csv',paths);write_csv(out/'matched_policy_summary.csv',summaries);write_csv(out/'offline_calibration.csv',offline)
    manifest=dict(dataset=dataset,source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),analysis_scope='post hoc measured-state replay; no live-controller or cache benefit timed',rank=16,tau_k=.2,tau_l=.2,max_states=9,
        acceptance='valid and rho_k<=0.20 for all policies',baseline='fixed largest (40,2), one evaluation',
        scalar_route='width-major order, q=0,1,2 at each width',pair_route='increase q when rho_l>0.20 and q<2; otherwise expand width resetting q=0; at width40 use remaining powers',
        output='first target reached; otherwise lowest valid observed rho_k, ties by cost,width,power',
        failure='invalid candidates cannot satisfy target; retain their cost/count and advance in unvisited width-major order; unknown failed costs make the cumulative cost unavailable, never zero',
        offline_rule='evaluate full grid, select cheapest valid target, ties by rho_k,width,power; no target remains infeasible',
        timer='sum stored probe-free CUDA-event pipeline intervals, not complete-factor wall times')
    (out/'matched_protocol.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    for r in summaries:
        if r['matrix']=='ALL':print(r)
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--grid',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--dataset',default='grid');a=p.parse_args()
    analyse(a.grid,a.out,a.dataset)


if __name__=='__main__':main()

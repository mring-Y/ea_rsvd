"""Non-interactive measurement of the two-input resident cache and cost grid."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from audit_factors import read_mm,write_csr

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--matrix-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--repeats',type=int,default=7);p.add_argument('--timeout',type=float,default=300);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'calibration').mkdir();inputs={};status=[]
    for name in ('west2021','lp_osa_30'):
        options=[a.matrix_root/(name+'.mtx'),a.matrix_root/name/(name+'.mtx'),a.matrix_root/name/name/(name+'.mtx')]
        path=next((x for x in options if x.is_file()),None)
        if path is None:status.append(dict(matrix=name,status='missing_input'));continue
        inputs[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        m,n,offsets,columns,values=read_mm(path);csr=a.out/(name+'.csr');write_csr(csr,m,n,offsets,columns,values)
        for ell in (24,32,40):
            for q in (0,1,2):
                case=f'{name}_l{ell}_q{q}';base=a.out if (ell,q)==(40,2) else a.out/'calibration'
                csv=base/((name if (ell,q)==(40,2) else case)+'.csv');factors=base/'factors'/(name if (ell,q)==(40,2) else case);factors.mkdir(parents=True)
                print('START',case,flush=True)
                command=[str(a.binary.resolve()),str(csr.resolve()),str(csv.resolve()),str(ell),str(q),'20261015',str(a.repeats),str(factors.resolve())]
                try:
                    done=subprocess.run(command,stdin=subprocess.DEVNULL,timeout=a.timeout,check=False)
                    item=dict(case=case,status='ok' if done.returncode==0 else 'failed',returncode=done.returncode)
                except subprocess.TimeoutExpired:item=dict(case=case,status='timeout')
                status.append(item);print('DONE',item,flush=True)
                (a.out/'run_status.json').write_text(json.dumps(dict(inputs=inputs,cases=status),indent=2)+'\n')
    (a.out/'run_status.json').write_text(json.dumps(dict(inputs=inputs,cases=status),indent=2)+'\n')
    if any(r['status']!='ok' for r in status):raise SystemExit('Some cases incomplete; see run_status.json. No successful-completion claim is made.')

if __name__=='__main__':main()

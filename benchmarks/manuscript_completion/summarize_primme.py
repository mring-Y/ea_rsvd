"""Matrix summaries with seeds and ranks nested within each matrix."""
import argparse,csv,math
from pathlib import Path
def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def geo(values):return math.exp(sum(math.log(x) for x in values)/len(values))
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with a.input.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    matrix=[];loo=[]
    for device in sorted({r['device'] for r in rows}):
        group=[r for r in rows if r['device']==device];names=sorted({r['matrix'] for r in group})
        for name in names:
            values=[float(r['speedup_rsvd_over_primme']) for r in group if r['matrix']==name]
            matrix.append(dict(device=device,matrix=name,groups=len(values),geometric_mean_primme_over_rsvd=geo(values)))
            keep=[float(r['speedup_rsvd_over_primme']) for r in group if r['matrix']!=name]
            loo.append(dict(device=device,excluded_matrix=name,retained_groups=len(keep),geometric_mean_primme_over_rsvd=geo(keep)))
    write(a.out/'primme_matrix_summary.csv',matrix);write(a.out/'primme_leave_one_out.csv',loo)
if __name__=='__main__':main()

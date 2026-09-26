"""Nominal reader agreement; original pre-consensus pairs required."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score,confusion_matrix

def analyze(d,iterations=10000,seed=0):
    required=['Case_ID','Cohort','M.P.','P.G.','Consensus']
    if any(c not in d for c in required):raise ValueError('Missing required columns: '+str(required))
    if d.empty or d[required].isna().any().any() or d.Case_ID.duplicated().any():raise ValueError('Empty input, missing values or duplicate case IDs; no rows silently excluded.')
    for c in ['M.P.','P.G.','Consensus']:
        if not d[c].isin(range(4)).all():raise ValueError('Labels must be integers 0–3.')
    if iterations<1:raise ValueError('iterations must be positive')
    out={}
    for name,g in [('All',d),*list(d.groupby('Cohort'))]:
        a=g['M.P.'].to_numpy();b=g['P.G.'].to_numpy();rng=np.random.default_rng(seed);boot=[]
        for _ in range(iterations):
            ix=rng.integers(0,len(g),len(g));k=cohen_kappa_score(a[ix],b[ix])
            if np.isfinite(k):boot.append(k)
        k=cohen_kappa_score(a,b);kw=cohen_kappa_score(a,b,weights='quadratic')
        out[str(name)]={'n':len(g),'agreement':int((a==b).sum()),'disagreement':int((a!=b).sum()),'nominal_kappa':float(k) if np.isfinite(k) else None,'ci95':np.quantile(boot,[.025,.975]).tolist() if boot else None,'quadratic_kappa':float(kw) if np.isfinite(kw) else None,'matrix_MP_rows_PG_columns':confusion_matrix(a,b,labels=range(4)).tolist(),'bootstrap_iterations':iterations,'valid_replicates':len(boot),'seed':seed,'weighted_scope':'Secondary category-code sensitivity analysis; not severity agreement.'}
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=10000);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    result=analyze(pd.read_excel(a.input),a.iterations,a.seed)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(a.output)

#!/usr/bin/env python3
"""Audit hard-label metrics from published confusion matrices or original NPZ probabilities.
Examples:
 python export_metrics.py --published-matrices --output confusion_matrix_audit.json
 python export_metrics.py --input external_probs.npz --positive-indices 2 3 --threshold 0.584 --output external_audit.json
Rows of exported confusion matrices are TRUE; columns are PREDICTED.
NPZ files must contain class_names and one unambiguous pair of *_probs, *_true arrays.
This script never manufactures probabilities from a confusion matrix.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score,roc_curve
NAMES=['normal','osteophytes','parasyndesmophytes','syndesmophytes']
# Transpose of Figure 1, whose rows are predicted and columns are reference.
PUBLISHED={
'internal':np.array([[36,2,0,0],[1,31,1,0],[0,1,11,0],[0,0,0,12]]).T,
'external':np.array([[51,3,0,1],[4,49,4,1],[0,1,16,0],[0,1,0,19]]).T}
def cp(k,n):
 if not n:return [None,None]
 return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
def rate(k,n):return {'estimate':float(k/n) if n else None,'ci95':cp(k,n),'numerator':int(k),'denominator':int(n)}
def vals(cm):
 n=cm.sum();r=cm.sum(1);c=cm.sum(0);tp=np.diag(cm);f=np.divide(2*tp,r+c,out=np.zeros(len(r),float),where=(r+c)>0)
 rec=np.divide(tp,r,out=np.full(len(r),np.nan),where=r>0)
 pe=(r*c).sum()/n**2;acc=tp.sum()/n
 w=(np.arange(len(r))[:,None]-np.arange(len(r))[None,:])**2
 den=(w*np.outer(r,c)/n).sum();kw=1-(w*cm).sum()/den if den else np.nan
 return {'accuracy':acc,'balanced_accuracy':np.mean(rec),'macro_f1':np.mean(f),'unweighted_kappa':(acc-pe)/(1-pe) if pe<1 else np.nan,'quadratic_weighted_kappa':kw,'per_class_f1':f}
def audit(cm,names,iterations=2000,seed=0):
 cm=np.asarray(cm,int);n=int(cm.sum());k=len(names);v=vals(cm)
 true,pred=np.where(cm>=0);pair=np.repeat(np.arange(k*k),cm.ravel())
 rng=np.random.default_rng(seed);boot=[]
 for _ in range(iterations):
  b=np.bincount(rng.choice(pair,size=n,replace=True),minlength=k*k).reshape(k,k)
  z=vals(b);z['per_class_f1']=np.where(b.sum(1)>0,z['per_class_f1'],np.nan);boot.append(z)
 out={'n':n,'class_names':names,'confusion_matrix_true_by_predicted':cm.tolist(),'bootstrap':{'iterations':iterations,'seed':seed,'generator':'numpy.default_rng / PCG64','method':'percentile','pair_order':'true class then predicted class, repeated by cell count','note':'Reconstructed pairs are equivalent for label metrics; no patient identifiers or probabilities are reconstructed.'},'overall':{},'per_class':{}}
 for metric in ['balanced_accuracy','macro_f1','unweighted_kappa','quadratic_weighted_kappa']:
  a=np.array([b[metric] for b in boot]);a=a[np.isfinite(a)]
  out['overall'][metric]={'estimate':float(v[metric]),'ci95':np.quantile(a,[.025,.975]).tolist(),'valid_replicates':len(a)}
 out['overall']['accuracy']=rate(int(np.trace(cm)),n)
 dist=abs(np.arange(k)[:,None]-np.arange(k)[None,:]);out['overall']['within_one_category']=rate(int(cm[dist<=1].sum()),n);out['overall']['mae']=float((cm*dist).sum()/n)
 for i,name in enumerate(names):
  tp=int(cm[i,i]);fn=int(cm[i].sum()-tp);fp=int(cm[:,i].sum()-tp);tn=n-tp-fn-fp
  a=np.array([b['per_class_f1'][i] for b in boot]);a=a[np.isfinite(a)]
  out['per_class'][name]={'TP':tp,'FN':fn,'FP':fp,'TN':tn,'sensitivity':rate(tp,tp+fn),'specificity':rate(tn,tn+fp),'PPV':rate(tp,tp+fp),'NPV':rate(tn,tn+fn),'f1':{'estimate':float(v['per_class_f1'][i]),'ci95':np.quantile(a,[.025,.975]).tolist(),'valid_replicates':len(a),'zero_observed_errors':fp+fn==0,'note':'A degenerate bootstrap interval does not imply absence of population uncertainty.' if fp+fn==0 else ''}}
 return out
def load_npz(path):
 with np.load(path,allow_pickle=False) as z:
  pk=[x for x in z.files if x=='probs' or x.endswith('_probs')];tk=[x for x in z.files if x=='true' or x.endswith('_true')]
  if len(pk)!=1 or len(tk)!=1 or 'class_names' not in z:raise ValueError('Require class_names and exactly one probs/true key pair.')
  p=np.asarray(z[pk[0]],float);raw=np.asarray(z[tk[0]]);names=[str(x) for x in z['class_names']]
 if p.ndim!=2 or raw.ndim!=1 or len(raw)!=len(p) or p.shape[1]!=len(names) or len(set(names))!=len(names):raise ValueError('Inconsistent shapes or duplicate names.')
 if not np.all(np.isfinite(p)) or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Invalid probabilities.')
 y=raw.astype(int)
 if not np.array_equal(raw,y) or np.any(y<0) or np.any(y>=len(names)):raise ValueError('Invalid reference labels.')
 return p,y,names
def main():
 ap=argparse.ArgumentParser(description=__doc__);g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--published-matrices',action='store_true');g.add_argument('--input');ap.add_argument('--output',required=True);ap.add_argument('--positive-indices',nargs='+',type=int);ap.add_argument('--threshold',type=float);ap.add_argument('--iterations',type=int,default=2000);ap.add_argument('--seed',type=int,default=0);a=ap.parse_args()
 if a.published_matrices:
  out={'provenance':'Reanalysis of manuscript Figure 1 only; probability-based results are not verified.','cohorts':{s:audit(c,NAMES,a.iterations,a.seed) for s,c in PUBLISHED.items()}}
 else:
  p,y,names=load_npz(a.input);k=len(names);pred=p.argmax(1);cm=np.bincount(y*k+pred,minlength=k*k).reshape(k,k);out=audit(cm,names,a.iterations,a.seed)
  one=np.eye(k)[y];out['mean_ovr_brier']=float(np.mean((p-one)**2));out['macro_auc']=float(roc_auc_score(y,p,multi_class='ovr',labels=np.arange(k))) if len(np.unique(y))==k else None
  out['micro_auc']=float(roc_auc_score(one.ravel(),p.ravel()));out['per_class_auc']={name:float(roc_auc_score(y==i,p[:,i])) if 0<(y==i).sum()<len(y) else None for i,name in enumerate(names)}
  if a.positive_indices is not None:
   pos=a.positive_indices
   if len(set(pos))!=len(pos) or not pos or min(pos)<0 or max(pos)>=k or len(pos)==k:raise ValueError('Invalid positive class indices.')
   yy=np.isin(y,pos);score=p[:,pos].sum(1)
   if len(np.unique(yy))<2:raise ValueError('Binary AUC needs both reference groups.')
   auc=float(roc_auc_score(yy,score));rng=np.random.default_rng(a.seed);boots=[]
   for _ in range(a.iterations):
    idx=rng.integers(0,len(y),len(y))
    if len(np.unique(yy[idx]))==2:boots.append(roc_auc_score(yy[idx],score[idx]))
   out['binary']={'positive_class_names':[names[i] for i in pos],'auc':auc,'auc_ci95':np.quantile(boots,[.025,.975]).tolist(),'valid_auc_replicates':len(boots)}
   # Optimum is explicitly exploratory unless this input is the internal derivation set.
   fpr,tpr,thr=roc_curve(yy,score,drop_intermediate=False);valid=np.isfinite(thr);jj=tpr[valid]-fpr[valid];th=thr[valid];out['binary']['youden_on_this_input']=float(th[np.argmax(jj)])
   if a.threshold is not None:
    yp=score>=a.threshold;tp=int((yy&yp).sum());fn=int((yy&~yp).sum());fp=int((~yy&yp).sum());tn=int((~yy&~yp).sum())
    out['binary']['fixed_threshold']={'threshold':a.threshold,'TP':tp,'FN':fn,'FP':fp,'TN':tn,'sensitivity':rate(tp,tp+fn),'specificity':rate(tn,tn+fp),'PPV':rate(tp,tp+fp),'NPV':rate(tn,tn+fn),'accuracy':rate(tp+tn,len(y))}
 Path(a.output).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(a.output)
if __name__=='__main__':main()

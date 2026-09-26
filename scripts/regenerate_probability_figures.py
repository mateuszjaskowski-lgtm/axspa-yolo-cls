from pathlib import Path
import numpy as np, json, hashlib, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import argparse
from export_metrics import load_npz
from class_mapping import probability_order
ap=argparse.ArgumentParser(description='Regenerate probability Figures 2–4 from authorized source arrays.')
ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();inp=a.inputs;repo=a.output;repo.mkdir(parents=True,exist_ok=True)
script_dir=Path(__file__).resolve().parent
names=['Normal','Osteophytes','Parasyndesmophytes','Syndesmophytes'];data={}
for c in ['internal','external']:
 src=inp/(c+'_probs.npz')
 p,y,source_names=load_npz(src)
 if probability_order(source_names)!=list(range(4)):raise ValueError('Canonical class order required.')
 data[c]=(p,y)
summary={'input_files':{},'limitations':'File-level reanalysis of user-supplied probabilities; checkpoint identity, original training and cohort independence not independently verified.'}
for c,(p,y) in data.items():
 f=inp/(c+'_probs.npz');summary['input_files'][f.name]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'n':len(y)}
s=data['internal'][0][:,2:].sum(1);t=data['internal'][1]>=2
fpr,tpr,thr=roc_curve(t,s,drop_intermediate=False);threshold=float(thr[np.argmax(tpr-fpr)])
p,y=data['external'];s=p[:,2:].sum(1);t=y>=2
fpr,tpr,thr=roc_curve(t,s,drop_intermediate=False);idx=np.argmax(tpr-fpr);youden=float(thr[idx])
valid=np.where(tpr>=.9)[0];maxspec=max(1-fpr[valid]);ties=valid[np.isclose(1-fpr[valid],maxspec)];idx=ties[np.argmax(tpr[ties])];s90=float(thr[idx])
summary['thresholds']={'internal_youden':threshold,'external_exploratory_youden':youden,'external_exploratory_sensitivity90':s90,'rule90':'Maximum specificity subject to sensitivity >=0.90; among tied specificity values choose highest sensitivity.','decision':'sum of float64-converted columns 2 and 3 >= full-precision threshold; text values rounded for display only.'}
import subprocess,sys
for cohort in ['internal','external']:
 subprocess.run([sys.executable,str(script_dir/'export_metrics.py'),'--input',str(inp/(cohort+'_probs.npz')),'--positive-indices','2','3','--threshold',repr(threshold),'--output',str(repo/(cohort+'_probability_audit.json'))],check=True)
summary['calibration']={}
for c,(p,y) in data.items():
 bins=[]
 for k in range(4):
  ix=np.minimum((p[:,k]*10).astype(int),9);b=[]
  for j in range(10):
   m=ix==j;n=int(m.sum())
   b.append({'lower':j/10,'upper':(j+1)/10,'n':n,'mean_probability':float(p[m,k].mean()) if n else None,'observed_fraction':float((y[m]==k).mean()) if n else None})
  bins.append(b)
 summary['calibration'][c]=bins
(repo/'probability_provenance_and_calibration.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
for ax,(c,(p,y)),letter in zip(axs,data.items(),'AB'):
 aucs=[]
 for k,n in enumerate(names):
  f,t,_=roc_curve(y==k,p[:,k]);a=roc_auc_score(y==k,p[:,k]);aucs.append(a);ax.plot(f,t,label=f'{n} (AUC = {a:.3f})')
 ax.plot([0,1],[0,1],'--',color='.6',lw=1);ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='1 - Specificity',ylabel='Sensitivity',title=f'{letter}. {c.capitalize()} cohort (n={len(y)})\nMacro-AUC = {np.mean(aucs):.3f}');ax.legend(loc='lower right',fontsize=8);ax.grid(alpha=.15)
fig.savefig(repo/'Figure2.tif',dpi=300,pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
for ax,(c,(p,y)),letter in zip(axs,data.items(),'AB'):
 for k,n in enumerate(names):
  b=summary['calibration'][c][k];ax.plot([z['mean_probability'] for z in b if z['n']],[z['observed_fraction'] for z in b if z['n']],'.-',label=n)
 ax.plot([0,1],[0,1],'--',color='.6',lw=1);ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Mean predicted probability',ylabel='Observed fraction positive',title=f'{letter}. {c.capitalize()} cohort (n={len(y)})\nMean one-versus-rest Brier = {np.mean((p-np.eye(4)[y])**2):.4f}');ax.legend(fontsize=8);ax.grid(alpha=.15)
fig.savefig(repo/'Figure3.tif',dpi=300,pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,5.4),layout='constrained')
for c,(p,y) in data.items():
 yy=y>=2;s=p[:,2:].sum(1);f,t,_=roc_curve(yy,s);audit=json.loads((repo/(c+'_probability_audit.json')).read_text())['binary'];lo,hi=audit['auc_ci95']
 ax.plot(f,t,label=f'{c.capitalize()} AUC {audit["auc"]:.3f} (95% CI {lo:.3f}–{hi:.3f})')
 m=s>=threshold;tp=(m&yy).sum();fp=(m&~yy).sum()
 ax.scatter(fp/(~yy).sum(),tp/yy.sum(),marker='o' if c=='internal' else 's',color='green' if c=='internal' else 'purple',s=60,zorder=4,label=f'{c.capitalize()} at internal-derived threshold')
ax.plot([0,1],[0,1],'--',color='.6',lw=1);ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='1 - Specificity',ylabel='Sensitivity',title=f'Binary dominant-category grouping\nInternal-derived threshold = {threshold:.6f} (display rounded)');ax.legend(loc='lower right',fontsize=8);ax.grid(alpha=.15)
fig.savefig(repo/'Figure4.tif',dpi=300,pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)
print(json.dumps(summary['thresholds'],indent=2))

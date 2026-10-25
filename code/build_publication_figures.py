#!/usr/bin/env python3
"""Re-layout audited spinal morphology results; never train or modify source data.
Usage: python code/build_publication_figures.py --source . --output reproduced_figures
Optional --include-private-cam requires the separately held CAM inputs/provenance.
Requires numpy, matplotlib, Pillow, scikit-learn; versions recorded in inventory.
All CI values and calibration bins are read from the audited evaluation JSON.
"""
from pathlib import Path
import argparse, csv, hashlib, json, platform, io, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import Normalize
from PIL import Image
import PIL
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parent.parent)
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent/'reproduced_figures')
parser.add_argument('--include-private-cam',action='store_true',help='Generate Figure 5 only when separately held audited CAM image inputs are available')
a=parser.parse_args(); ROOT=a.source.resolve(); OUT=a.output.resolve(); OUT.mkdir(parents=True,exist_ok=True)
RES=ROOT/'results' if (ROOT/'results/analysis').is_dir() else ROOT
BASE=RES/'analysis/yolo11l-cls/seed_0'
WIDTH=7.05
CLASS=['normal','osteofity','parasyndesmofity','syndesmofity']
EN=['Normal','Osteophytes','Parasyndesmophytes','Syndesmophytes']
WRAP=['Normal','Osteophytes','Para-\nsyndesmophytes','Syndesmophytes']
CO=['#2864A5','#C45C19','#248B72','#9760A6']
SPLITS=['val','test']; TITLE={'val':'Internal validation (n = 95)','test':'External test (n = 150)'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7,'axes.linewidth':.7,
'lines.linewidth':1.4,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white',
'image.interpolation':'none','image.resample':False})
inv={'primary_model':'yolo11l-cls','training_seed':0,'journal_width_mm':WIDTH*25.4,
     'environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,
                    'Pillow':PIL.__version__,'scikit-learn':sklearn.__version__},'figures':[],'validation_checks':[]}
sources={}; caps=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_json(p):
 sources[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
def read_csv(p):
 sources[str(p.relative_to(ROOT))]=sha(p)
 with p.open(newline='') as f:return list(csv.DictReader(f))
def tidy(ax):
 ax.spines[['top','right']].set_visible(False)
 ax.grid(alpha=.13,lw=.6,zorder=0)
def roc_axes(ax):
 tidy(ax);ax.plot([0,1],[0,1],color='#888888',ls='--',lw=.8,zorder=0)
 ax.set(xlim=(-.025,1.025),ylim=(-.025,1.025),xlabel='False-positive rate (1 - specificity)',ylabel='True-positive rate (sensitivity)')
 ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6));ax.set_aspect('equal')
def save(fig,stem,caption,src,notes=None):
 # Atomic writes prevent partial artifacts when file previews or packaging run concurrently.
 # Fixed physical sizes: no tight bounding-box resize that changes journal width.
 for suffix,dpi,fmt in [('.png',600,'png'),('.pdf',600,'pdf'),('_preview.png',160,'png')]:
  buf=io.BytesIO();kwargs={'metadata':{'Creator':'build_publication_figures.py','Title':stem}} if fmt=='pdf' else {}
  fig.savefig(buf,format=fmt,dpi=dpi,**kwargs);blob=buf.getvalue()
  if fmt=='png':
   verify_image=Image.open(io.BytesIO(blob));verify_image.load()
  target=OUT/(stem+suffix);tmp=OUT/(stem+suffix+'.tmp')
  with tmp.open('wb') as stream:
   stream.write(blob);stream.flush();os.fsync(stream.fileno())
  os.replace(tmp,target)
  assert target.stat().st_size==len(blob)

 dims=list(Image.open(OUT/f'{stem}.png').size)
 inv['figures'].append({'id':stem,'caption':caption,'files':[f'{stem}.png',f'{stem}.pdf'],
                       'physical_inches':list(fig.get_size_inches()),'png_pixels':dims,'png_dpi':600,
                       'sources':src,'notes':notes or [],'sha256':{f'{stem}.{e}':sha(OUT/f'{stem}.{e}') for e in ['png','pdf']}})
 caps.append(stem+'\n'+caption+'\n');plt.close(fig)
D={s:read_json(BASE/f'{s}_metrics.json') for s in SPLITS}
P={s:read_csv(BASE/f'{s}_predictions.csv') for s in SPLITS}
y={s:np.array([int(r['true_label']) for r in P[s]]) for s in SPLITS}
prob={s:np.array([[float(r['p_'+c]) for c in CLASS] for r in P[s]],dtype=np.float64) for s in SPLITS}
for s in SPLITS:
 assert D[s]['class_names']==CLASS
 assert np.array_equal(confusion_matrix(y[s],prob[s].argmax(1),labels=range(4)),D[s]['confusion_matrix_true_rows_predicted_columns'])
 assert abs(np.mean(y[s]==prob[s].argmax(1))-D[s]['metrics']['accuracy']['estimate'])<1e-12
inv['validation_checks'].append('Both confusion matrices independently recomputed from the audited per-image probabilities and matched exactly.')
SRC=[str((BASE/f'{s}_{n}').relative_to(ROOT)) for s in SPLITS for n in ['metrics.json','predictions.csv']]

# Figure 1: both count and row percentage displayed, preserving integer evidence.
fig,axs=plt.subplots(1,2,figsize=(WIDTH,4.08))
fig.subplots_adjust(left=.18,right=.985,bottom=.26,top=.86,wspace=.60)
for ax,s,ltr in zip(axs,SPLITS,'AB'):
 cm=np.array(D[s]['confusion_matrix_true_rows_predicted_columns']);norm=cm/cm.sum(1,keepdims=True)
 ax.imshow(norm,vmin=0,vmax=1,cmap='Blues',interpolation='none')
 for i in range(4):
  for j in range(4):
   ax.text(j,i,f'{cm[i,j]}\n({norm[i,j]:.1%})',ha='center',va='center',fontsize=8,color='white' if norm[i,j]>.5 else '#202020',linespacing=1.4)
 ax.set_xticks(range(4),WRAP,rotation=48,ha='right',rotation_mode='anchor');ax.set_yticks(range(4),WRAP,fontsize=7.4)
 ax.set(xlabel='Predicted class')
 ax.tick_params(length=0,pad=5)
 ax.set_title(f'{ltr}  {TITLE[s]}\nAccuracy: {int(np.trace(cm))}/{int(cm.sum())} ({np.trace(cm)/cm.sum():.1%})',pad=10)
 ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True);ax.grid(which='minor',color='white',lw=1);ax.tick_params(which='minor',length=0)
 for sp in ax.spines.values():sp.set_visible(False)
fig.text(.017,.56,'Reference class',rotation=90,va='center',fontsize=8)
save(fig,'Figure_1_confusion_matrices','Figure 1. Confusion matrices for the primary YOLO11l-cls model (reference seed 0): (A) internal validation and (B) external test. Rows represent reference classes and columns predicted classes. Cells show counts and percentages normalized within each reference-class row; shading encodes the same row percentages. All four classes are nominal. The external cohort was held out until the final evaluation.',SRC)

# Figure 2: actual unsmoothed empirical OvR curves.
fig,axs=plt.subplots(1,2,figsize=(WIDTH,3.65))
fig.subplots_adjust(left=.085,right=.985,bottom=.16,top=.88,wspace=.34)
for ax,s,ltr in zip(axs,SPLITS,'AB'):
 roc_axes(ax)
 for k,(name,color) in enumerate(zip(EN,CO)):
  f,t,_=roc_curve(y[s]==k,prob[s][:,k],drop_intermediate=False);auc=roc_auc_score(y[s]==k,prob[s][:,k]);assert abs(auc-D[s]['metrics'][CLASS[k]+'_ovr_auc']['estimate'])<1e-12
  ax.plot(f,t,color=color,label=f'{name}: {auc:.3f}')
 ax.set_title(f'{ltr}  {TITLE[s]}',pad=10)
 m=D[s]['metrics']['macro_ovr_auc'];ax.text(.97,.37,f'Macro AUC {m["estimate"]:.3f}\n95% CI {m["ci_low"]:.3f}-{m["ci_high"]:.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=7.5)
 ax.legend(loc='lower right',frameon=False,handlelength=1.7,labelspacing=.6,fontsize=6.6)
save(fig,'Figure_2_ovr_roc','Figure 2. One-versus-rest receiver operating characteristic (ROC) curves for the primary model (seed 0) in (A) internal validation and (B) external test. Legends give class-specific areas under the curve (AUC); the displayed macro AUC is the unweighted mean of the four class-specific AUCs. The 95% confidence intervals are the previously computed image-level percentile bootstrap intervals (2000 replicates) for the fixed fitted model. The dashed diagonal is the no-discrimination reference. Curves were reconstructed directly from saved probabilities without smoothing.',SRC)

# Figure 3: no line connects sparse bins; explicit support and saved exact intervals.
fig=plt.figure(figsize=(WIDTH,4.6));gs=fig.add_gridspec(2,2,height_ratios=[3.1,1],left=.085,right=.985,bottom=.13,top=.9,wspace=.34,hspace=.1)
for col,(s,ltr) in enumerate(zip(SPLITS,'AB')):
 ax=fig.add_subplot(gs[0,col]);hist=fig.add_subplot(gs[1,col],sharex=ax)
 cal=D[s]['calibration']['top_label'];bins=cal['bins'];non=[b for b in bins if b['n']]
 x=np.array([b['mean_probability'] for b in non]);obs=np.array([b['observed_fraction'] for b in non])
 lo=np.array([b['ci_low'] for b in non]);hi=np.array([b['ci_high'] for b in non]);n=np.array([b['n'] for b in non])
 tidy(ax);ax.plot([0,1],[0,1],ls='--',color='#888888',lw=.8)
 ax.errorbar(x,obs,yerr=[obs-lo,hi-obs],fmt='none',ecolor=CO[0],elinewidth=1,capsize=2,alpha=.85)
 ax.scatter(x,obs,s=24,color=CO[0],edgecolor='white',linewidth=.4,zorder=5)
 ax.set(xlim=(-.02,1.025),ylim=(-.045,1.045),ylabel='Observed proportion correct')
 ax.set_title(f'{ltr}  {TITLE[s]}',pad=9);ax.tick_params(labelbottom=False)
 ax.text(.04,.95,f'Top-label ECE: {cal["ece"]:.3f}\nMulticlass Brier: {D[s]["metrics"]["multiclass_brier"]["estimate"]:.3f}',transform=ax.transAxes,va='top',fontsize=7.2,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':2})
 centers=np.array([(b['lower']+b['upper'])/2 for b in bins]);counts=np.array([b['n'] for b in bins]);assert counts.sum()==D[s]['n']
 hist.bar(centers,counts,width=.083,color=CO[0],alpha=.75,zorder=2)
 for xx,nn in zip(centers,counts):
  if nn:hist.text(xx,nn+max(counts)*.045,str(nn),ha='center',fontsize=7)
 hist.set(ylim=(0,max(counts)*1.3),xlabel='Predicted confidence',ylabel='Images (n)');hist.set_xticks(np.linspace(0,1,6));tidy(hist);hist.grid(False)
save(fig,'Figure_3_calibration','Figure 3. Exploratory top-label calibration for the primary model (seed 0): (A) internal validation and (B) external test. Confidence is the maximum of the four predicted class probabilities; the observed outcome is correctness of the predicted class. Points show the mean confidence and observed fraction correct in each nonempty, prespecified equal-width bin (10 bins over [0,1]); vertical bars are exact 95% Clopper-Pearson intervals. Histograms show all bin counts. Empty bins have no calibration point, and points are not joined. The diagonal represents perfect calibration. ECE denotes the count-weighted absolute calibration error. Multiclass Brier score sums squared probability errors over the four classes before averaging across images. Most images occupy the highest-confidence bin; sparse remaining bins and small evaluation cohorts limit calibration conclusions. No recalibration was fitted.',SRC,['Calibration bins, point estimates and confidence intervals are used unchanged from audited JSON.'])

# Figure 4: derive nothing on test, plot same locked threshold in both cohorts.
th=D['val']['binary_morphology']['threshold']['threshold'];assert abs(th-.27126939594745636)<1e-15
fig,axs=plt.subplots(1,2,figsize=(WIDTH,3.65));fig.subplots_adjust(left=.085,right=.985,bottom=.16,top=.88,wspace=.34)
for ax,s,ltr in zip(axs,SPLITS,'AB'):
 roc_axes(ax);target=np.isin(y[s],[2,3]);score=prob[s][:,2:].sum(1)
 f,t,_=roc_curve(target,score,drop_intermediate=False);auc=roc_auc_score(target,score)
 assert abs(auc-D[s]['metrics']['binary_auc']['estimate'])<1e-12
 b=D[s]['binary_morphology'];sens=b['sensitivity']['estimate'];spec=b['specificity']['estimate'];pred=score>=th
 assert int(np.sum(target&pred))==b['tp'] and int(np.sum(~target&pred))==b['fp']
 ax.plot(f,t,color=CO[0]);ax.scatter([1-spec],[sens],color=CO[1],marker='D',s=30,zorder=10,edgecolors='white',linewidths=.5,clip_on=False)
 ax.set_title(f'{ltr}  {TITLE[s]}',pad=10)
 ci=D[s]['metrics']['binary_auc']
 ax.text(.97,.52,f'AUC {auc:.4f}\n95% CI {ci["ci_low"]:.4f}-{ci["ci_high"]:.4f}\n\nThreshold = {th:.4f}\nSensitivity: {b["tp"]}/{b["tp"]+b["fn"]} ({sens:.1%})\nSpecificity: {b["tn"]}/{b["tn"]+b["fp"]} ({spec:.1%})',transform=ax.transAxes,ha='right',va='top',fontsize=7.5,linespacing=1.4)
save(fig,'Figure_4_binary_morphology_roc','Figure 4. Exploratory binary morphology ROC curves for the primary model (seed 0): (A) internal validation and (B) external test. The positive group combines parasyndesmophytes and syndesmophytes; the negative group combines normal morphology and osteophytes. The score is P(parasyndesmophytes) + P(syndesmophytes). Orange diamonds indicate the fixed threshold 0.27126939594745636, selected by maximizing Youden\'s index in internal validation and applied unchanged to the external test. Confidence intervals for AUC are the saved 2000-replicate image-level bootstrap intervals. Operating-point uncertainty does not include uncertainty in selecting the validation threshold. This grouping concerns radiographic morphology and is not a clinical diagnosis of axial spondyloarthritis.',SRC)

# Figure 5: reuse unmodified audited raster panels. No map recoloring/recalculation.
if a.include_private_cam:
 CAM=RES/'eigen_cam';panels=read_csv(CAM/'eigen_cam_panels.csv');prov=read_json(CAM/'eigen_cam_provenance.json');audit=read_json(ROOT/'review_supplement/cam_audit.json')
 assert len(panels)==8 and ''.join(p['panel'] for p in panels)=='abcdefgh'
 assert audit['checks']['exact_probability_equality_all150'] and audit['checks']['selection_matches_locked_predictions']
 fig=plt.figure(figsize=(WIDTH,9.2))
 # Explicit inch-based positions keep every heading between, rather than over, image rows.
 for k,p in enumerate(panels):
  row,col=divmod(k,2);head_in=9.0-2.07*row;left_in=.20+3.55*col;side_in=1.43;gap_in=.12
  ax0=fig.add_axes([left_in/WIDTH,(head_in-.45-side_in)/9.2,side_in/WIDTH,side_in/9.2])
  ax1=fig.add_axes([(left_in+side_in+gap_in)/WIDTH,(head_in-.45-side_in)/9.2,side_in/WIDTH,side_in/9.2])
  tr=EN[CLASS.index(p['true_class'])];pr=EN[CLASS.index(p['pred_class'])];pc=float(p['p_'+p['pred_class']]);pdisplay='p > 0.999' if .9995<=pc<1 else f'p = {pc:.3f}'
  for ax,key in [(ax0,'input_png'),(ax1,'overlay_png')]:
   f=CAM/p[key];sources[str(f.relative_to(ROOT))]=sha(f);im=np.asarray(Image.open(f));assert im.shape[:2]==(640,640)
   ax.imshow(im,interpolation='none',resample=False);ax.axis('off')
  ax0.set_title('Model input',fontsize=7,pad=3);ax1.set_title('Eigen-CAM overlay',fontsize=7,pad=3)
  fig.text(left_in/WIDTH,head_in/9.2,f'({p["panel"]})  Reference: {tr}',fontsize=7.8,weight='bold',ha='left',va='bottom')
  fig.text(left_in/WIDTH,(head_in-.18)/9.2,f'Prediction: {pr} ({pdisplay})',fontsize=7.5,ha='left',va='bottom')
  # Keep exact original NPZs referenced and hashed; do not rewrite pixel or map data.
  f=CAM/p['raw_arrays_npz'];sources[str(f.relative_to(ROOT))]=sha(f)
 cbax=fig.add_axes([.31,.045,.38,.013]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='turbo'),cax=cbax,orientation='horizontal',ticks=[0,.5,1]);cb.ax.tick_params(labelsize=7,length=2);cb.set_label('Relative map intensity (normalized per image)',fontsize=7.5,labelpad=3)
 cam_src=[str((CAM/'eigen_cam_panels.csv').relative_to(ROOT)),str((CAM/'eigen_cam_provenance.json').relative_to(ROOT)),str((ROOT/'review_supplement/cam_audit.json').relative_to(ROOT))]+[str((CAM/p[key]).relative_to(ROOT)) for p in panels for key in ['input_png','overlay_png','raw_arrays_npz']]
 save(fig,'Figure_5_eigen_cam','Figure 5. Exploratory Eigen-CAM examples for the primary YOLO11l-cls model (seed 0) on the external test cohort. Panels (a-d) show one correctly classified example per reference class; (e-h) show four incorrect classifications. Within each pair, the left image is the exact saved 640 x 640 model input and the right is its audited Eigen-CAM overlay. The displayed p is the predicted-class probability, not the intensity of the map. Selection follows the fixed lexicographic rule recorded in the panel ledger: first correct case per reference class; for errors, first case per available reference class followed by remaining paths up to four examples. No confidence-based selection was used. The source PNGs had been prepared manually around the examined spinal segment before radiologist labeling; the computational transform preserved the entire prepared PNG by aspect-preserving resizing and square padding. Maps use the first principal-component projection of centered activations from model.9, clipped at zero, normalized separately per image, and bilinearly resized. The unchanged overlay is 40% turbo heatmap and 60% model input. All 150 external probability vectors matched the locked evaluation exactly in the recorded replay. These class-agnostic maps are qualitative illustrations; they neither validate lesion localization nor establish causal reasoning or exclude shortcut learning. Source identifiers, array files and hashes remain available in the panel ledger and provenance record.',cam_src,['Original input and overlay PNG pixels and numerical map arrays are not modified; only page layout and English labeling changed. PDF embeds original raster panels with vector text. Source images are 640 x 640; exporting the composite at 600 dpi does not create additional source detail.','All eight previously selected cases retained in original a-h order. No need to split or select only favorable examples.'])
else:
 inv['validation_checks'].append('Figure 5 omitted: private CAM pixel inputs are not included in the public repository.')

# Supplement S1: author-confirmed historical screening vs retained image inventory.
# Initial record counts are not asserted to be unique-patient counts.
flow_source = {
 'source': 'Author-confirmed screening counts and accepted dataset reconciliation, as instructed for this final revision.',
 'lodz': {'historical_screening_records': 686, 'quality_exclusions': 9, 'original_png_inventory': 677, 'duplicate_training_copies_removed': 3, 'retained_unique_images': 674, 'train': 579, 'validation': 95},
 'konskie': {'historical_screening_records': 155, 'quality_exclusions': 5, 'retained_unique_images': 150, 'external_test': 150},
 'clarifications': ['Historical screening record counts are not asserted to represent unique patients.', 'One image per patient in the retained dataset was confirmed by the author.', 'Severe degenerative changes were not an exclusion criterion.', 'Prior spinal surgery or instrumentation, fractures unrelated to axSpA, and DISH were exclusion criteria; no additional cases meeting these criteria were present in the assessed pool.']
}
(OUT/'cohort_flow_source.json').write_text(json.dumps(flow_source,indent=2)+'\n')
assert 686-9==677 and 677-3==674 and 579+95==674 and 155-5==150
fig,ax=plt.subplots(figsize=(WIDTH,6.2));fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.975);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,text,fill='#F2F6FA',edge='#54718C',fs=8):
 ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.009,rounding_size=0.008',linewidth=.8,edgecolor=edge,facecolor=fill));ax.text(x,y,text,ha='center',va='center',fontsize=fs,linespacing=1.35)
def arrow(x1,y1,x2,y2):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,color='#54718C',linewidth=.8))
L=.265;R=.75;W=.43
ax.text(L,.975,'Łódź: development cohort',ha='center',fontsize=10,weight='bold');ax.text(R,.975,'Końskie: external cohort',ha='center',fontsize=10,weight='bold')
box(L,.88,W,.095,'Historical screening\n686 examination records');box(R,.88,W,.095,'Historical screening\n155 examination records')
arrow(L,.823,L,.776);arrow(R,.823,R,.776)
box(L,.72,W,.10,'Quality screening: 9 excluded\nArtifacts or poor image quality',fill='#FFF4E8',edge='#BE864A');box(R,.72,W,.10,'Quality screening: 5 excluded\nArtifacts or poor image quality',fill='#FFF4E8',edge='#BE864A')
arrow(L,.662,L,.615);arrow(R,.662,R,.245)
box(L,.56,W,.095,'Original image inventory\n677 PNG files')
arrow(L,.504,L,.456)
box(L,.40,W,.10,'Duplicate-copy removal\n3 copies removed from training',fill='#FFF4E8',edge='#BE864A')
arrow(L,.342,L,.295)
box(L,.24,W,.095,'Final development cohort\n674 unique images',fill='#EAF5F0',edge='#4B8270');box(R,.19,W,.10,'Final external test cohort\n150 unique images',fill='#EAF5F0',edge='#4B8270')
arrow(L,.184,.145,.127);arrow(L,.184,.375,.127)
box(.145,.076,.20,.08,'Training\nn = 579',fill='#EAF5F0',edge='#4B8270');box(.375,.076,.20,.08,'Validation\nn = 95',fill='#EAF5F0',edge='#4B8270')
save(fig,'Figure_S1_cohort_flow','Figure S1. Cohort screening and image-inventory reconciliation. The initial counts of 686 and 155 refer to historical examination records screened in Łódź and Końskie, respectively, and are not presented as counts of unique patients. Quality screening excluded nine and five records, respectively, owing to artifacts or poor image quality. The original Łódź image inventory contained 677 PNG files; removing three duplicate training copies yielded 674 retained images, with the original retained folder assignments preserved (579 training and 95 internal validation). The external test cohort contained 150 images. The author confirmed one image per patient in the retained dataset. Prior spinal surgery or instrumentation, fractures unrelated to axSpA, and diffuse idiopathic skeletal hyperostosis were exclusion criteria, but no additional cases meeting them occurred in the assessed pool. Severe degenerative changes were not an exclusion criterion. This diagram distinguishes clinical screening counts from subsequent technical duplicate-copy removal.',['cohort_flow_source.json'],['Source numerical values explicitly supplied or confirmed by the author; file inventory consistency is separate from unique-patient identity validation.'])

# Supplement: all seeds, no best external seed selection.
allruns=read_csv(RES/'analysis/metrics_all_runs.csv');summary=read_csv(RES/'analysis/five_seed_summary.csv')
MODELS=['yolo11l-cls','yolo26l-cls','resnet50','efficientnet_b0'];ML=['YOLO11l-cls','YOLO26l-cls','ResNet50','EfficientNet-B0'];seedcols=['#000000','#2864A5','#C45C19','#248B72','#9760A6'];jitter=np.linspace(-.17,.17,5)
fig,axs=plt.subplots(1,2,figsize=(WIDTH,3.65));fig.subplots_adjust(left=.085,right=.985,bottom=.24,top=.86,wspace=.32)
for ax,metric,ltr,yl in zip(axs,['accuracy','macro_f1'],'AB',['External accuracy (%)','External macro F1 (%)']):
 for i,model in enumerate(MODELS):
  vals=[]
  for seed in range(5):
   row=next(r for r in allruns if r['model']==model and r['split']=='test' and int(r['seed'])==seed and r['metric']==metric);v=float(row['estimate'])*100;vals.append(v)
   ax.scatter(i+jitter[seed],v,c=seedcols[seed],marker='*' if seed==0 else 'o',s=66 if seed==0 else 26,zorder=5,linewidths=.4,edgecolors='white')
  ss=next(r for r in summary if r['model']==model and r['split']=='test' and r['metric']==metric)
  assert abs(np.mean(vals)-100*float(ss['mean']))<1e-10 and abs(np.std(vals,ddof=1)-100*float(ss['sample_sd']))<1e-10
  ax.errorbar(i,float(ss['mean'])*100,yerr=float(ss['sample_sd'])*100,fmt='_',color='#333333',capsize=5,lw=1.15,markersize=13,zorder=3)
 ax.set(xticks=range(4),xticklabels=ML,ylabel=yl,xlim=(-.45,3.45),ylim=(77,98));ax.tick_params(axis='x',rotation=25);tidy(ax);ax.set_title(f'{ltr}  Five fixed training seeds',pad=10)
handles=[Line2D([0],[0],color='none',marker='*' if i==0 else 'o',markerfacecolor=seedcols[i],markeredgecolor='white',markersize=9 if i==0 else 6,label=f'Seed {i}'+(' (reference)' if i==0 else '')) for i in range(5)]
fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.005),columnspacing=1.1)
save(fig,'Figure_S2_seed_robustness','Figure S2. External-test (A) accuracy and (B) macro F1 for all 20 training runs: four model pipelines and five fixed seeds each, evaluated on the same 150 images. Points show individual runs; stars identify the prespecified reference seed 0. Gray bars show the mean plus/minus one sample standard deviation across the five seeds, not confidence intervals. Models remain in a fixed order rather than an external-performance ranking. Seed variability does not increase the number of independent evaluation cases. Comparisons concern full training pipelines with different optimization settings. The external cohort was held out until the final evaluation.',[str((RES/'analysis/metrics_all_runs.csv').relative_to(ROOT)),str((RES/'analysis/five_seed_summary.csv').relative_to(ROOT))])

# Supplement: descriptive repeatability of mistakes across fixed YOLO11 seeds.
pp=[read_csv(RES/f'analysis/yolo11l-cls/seed_{seed}/test_predictions.csv') for seed in range(5)]
ids=[r['image_id'] for r in pp[0]]
for rows in pp:assert [r['image_id'] for r in rows]==ids
wrong=np.array([[int(r['true_label'])!=int(r['predicted_label']) for r in rows] for rows in pp]);nw=wrong.sum(0);hist=np.bincount(nw,minlength=6);assert hist.tolist()==[108,21,8,4,7,2]
fig,axs=plt.subplots(1,2,figsize=(WIDTH,3.65));fig.subplots_adjust(left=.085,right=.985,bottom=.23,top=.85,wspace=.36)
ax=axs[0];bars=ax.bar(range(6),hist,color=['#248B72']+['#2864A5']*4+['#C45C19'],width=.7)
for b,n in zip(bars,hist):ax.text(b.get_x()+b.get_width()/2,n+2,str(n),ha='center',fontsize=8)
ax.set(xlabel='Number of seeds with an incorrect prediction',ylabel='External images (n)',xticks=range(6),ylim=(0,125));ax.set_title('A  Error recurrence across five seeds',pad=10);tidy(ax);ax.grid(axis='x',visible=False)
ax=axs[1];truth=np.array([int(r['true_label']) for r in pp[0]]);allcorrect=[];variable=[];allwrong=[]
for k in range(4):
 mask=truth==k;allcorrect.append(int(((nw==0)&mask).sum()));variable.append(int(((nw>0)&(nw<5)&mask).sum()));allwrong.append(int(((nw==5)&mask).sum()))
left=np.zeros(4)
for values,color,label in [(allcorrect,'#248B72','Correct in all seeds'),(variable,'#2864A5','Incorrect in 1-4 seeds'),(allwrong,'#C45C19','Incorrect in all seeds')]:
 ax.barh(range(4),values,left=left,height=.65,color=color,label=label)
 for k,(v,base) in enumerate(zip(values,left)):
  if v>=3:ax.text(base+v/2,k,str(v),va='center',ha='center',color='white',fontsize=7.5)
 left+=values
ax.set(yticks=range(4),yticklabels=['Normal (55)','Osteophytes (54)','Parasyndesmophytes (20)','Syndesmophytes (21)'],xlabel='External images (n)',xlim=(0,57));ax.invert_yaxis();ax.tick_params(axis='y',labelsize=7);ax.set_title('B  Recurrence by reference class',pad=10);tidy(ax);ax.grid(axis='y',visible=False)
# Long class labels placed within left margin of right subplot; extra wspace avoids overlap.
fig.subplots_adjust(wspace=.65)
fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.0),ncol=3,frameon=False,fontsize=7)
save(fig,'Figure_S3_error_recurrence','Figure S3. Post-evaluation descriptive analysis of primary-model errors across five training seeds on the same external cohort (n = 150). (A) Number of images misclassified in 0-5 seeds. (B) Counts by reference class: correct in all five seeds, incorrect in one to four seeds, or incorrect in all five seeds. The counts of errors for seeds 0-4 were 12, 24, 22, 14 and 15. Overall, 108 images were classified correctly by every seed, 42 were misclassified at least once and two were misclassified in every seed. This analysis describes training variability; it is not an ensemble evaluation, an independent validation, or evidence that any reference label is incorrect.',[str((RES/f'analysis/yolo11l-cls/seed_{seed}/test_predictions.csv').relative_to(ROOT)) for seed in range(5)])

inv['sources_sha256']=sources
if a.include_private_cam:
 inv['validation_checks'].append('All eight CAM cases, original input PNGs, overlays and array files preserved and referenced by hash.')
inv['validation_checks']+=['All class-specific ROC areas and binary areas recomputed and matched audited point estimates.','Threshold-operating-point counts independently checked without selecting any test threshold.','Calibration bin counts sum to 95 and 150.','All twenty displayed seed point estimates, sample means and sample standard deviations matched the audited summaries.','Error recurrence distribution independently reconstructed: [108, 21, 8, 4, 7, 2].']
inv['script_sha256']=sha(Path(__file__))
(OUT/'figure_inventory.json').write_text(json.dumps(inv,indent=2)+'\n')
(OUT/'figure_captions.txt').write_text('\n'.join(caps))
print(json.dumps({'figures':len(inv['figures']),'source_files_hashed':len(sources),'output':str(OUT)},indent=2))

for item in inv['figures']:
 for name in item['files']:
  assert sha(OUT/name)==item['sha256'][name]
 Image.open(OUT/(item['id']+'.png')).load()
print('Final output hashes and PNG decoding: PASS')

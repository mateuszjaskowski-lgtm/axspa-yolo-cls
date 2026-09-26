#!/usr/bin/env python3
"""Exact paired McNemar tests for aligned models. No test is run without original paired outputs.
Input NPZ must include image_ids, true, probs, class_names. All files must match IDs,
reference labels and class order exactly; reorder deliberately upstream if required.
Usage: python compare_predictions.py --inputs yolo11.npz yolo26.npz resnet50.npz effb0.npz --output comparisons.json
"""
import argparse,json,itertools
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True);args=p.parse_args()
 if len(args.inputs)<2:raise ValueError('At least two model files are required.')
 records=[]
 for f in args.inputs:
  with np.load(f,allow_pickle=False) as z:
   ids=z['image_ids'];y=z['true'];probs=z['probs'];names=z['class_names']
  if len(np.unique(ids))!=len(ids) or y.ndim!=1 or probs.shape!=(len(y),len(names)) or len(ids)!=len(y):raise ValueError('Invalid shapes/IDs in '+f)
  if not np.all(np.isfinite(probs)) or np.any(probs<0) or np.any(probs>1) or not np.allclose(probs.sum(1),1,atol=1e-5):raise ValueError('Invalid probabilities in '+f)
  if not np.array_equal(y,y.astype(int)) or np.any(y<0) or np.any(y>=len(names)):raise ValueError('Invalid labels in '+f)
  if records and any(not np.array_equal(v,w) for v,w in zip((ids,y,names),records[0][1:4])):raise ValueError('Model inputs are not identically aligned: '+f)
  records.append((str(Path(f).name),ids,y,names,probs.argmax(1)==y))
 out=[]
 for a,b in itertools.combinations(records,2):
  ab=int((a[4]&~b[4]).sum());ba=int((~a[4]&b[4]).sum());n=ab+ba
  out.append({'model_a':a[0],'model_b':b[0],'a_correct_b_wrong':ab,'a_wrong_b_correct':ba,'n_discordant':n,'p_exact_two_sided':float(binomtest(ab,n,.5).pvalue) if n else 1.,'accuracy_difference_a_minus_b':float(a[4].mean()-b[4].mean())})
 # Holm adjustment over all reported pairwise comparisons.
 order=sorted(range(len(out)),key=lambda i:out[i]['p_exact_two_sided']);last=0.
 for rank,i in enumerate(order):
  last=max(last,min(1.,(len(out)-rank)*out[i]['p_exact_two_sided']));out[i]['p_holm']=last
 Path(args.output).write_text(json.dumps({'method':'exact two-sided McNemar; Holm adjustment across comparisons','comparisons':out},indent=2)+'\n')
if __name__=='__main__':main()

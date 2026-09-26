"""Evaluate every image in a four-class folder; preserve IDs and canonical order."""
import argparse,json,hashlib,platform
from pathlib import Path
import numpy as np
from class_mapping import NAMES,category,probability_order

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--imgsz',type=int,default=640)
    a=p.parse_args()
    if not a.model.is_file() or not a.data.is_dir():raise FileNotFoundError('Model/data missing.')
    if a.output.exists():raise FileExistsError('Choose a new output path to preserve earlier evaluations.')
    if a.output.suffix!='.npz':raise ValueError('Output must end in .npz')
    dirs=sorted(x for x in a.data.iterdir() if x.is_dir())
    if sorted(category(x.name) for x in dirs)!=list(range(4)):raise ValueError('Require exactly four category folders.')
    supported={'.png','.jpg','.jpeg','.tif','.tiff','.bmp'};items=[]
    for folder in dirs:
        files=sorted(x for x in folder.rglob('*') if x.is_file())
        if not files:raise ValueError(f'Empty category: {folder}')
        for f in files:
            if f.suffix.lower() not in supported:raise ValueError(f'Unsupported file in evaluation directory: {f}')
            items.append((f,category(folder.name)))
    if any(x.is_file() for x in a.data.iterdir()):raise ValueError('Unexpected loose files in dataset root.')
    from ultralytics import YOLO
    import ultralytics,torch
    model=YOLO(str(a.model));order=probability_order(model.names);probs=[];ids=[];labels=[]
    for f,label in items:
        results=model.predict(str(f),imgsz=a.imgsz,verbose=False)
        if len(results)!=1 or results[0].probs is None:raise ValueError(f'No classification result: {f}')
        v=np.asarray(results[0].probs.data.cpu().numpy(),dtype=float)[order]
        if v.shape!=(4,) or not np.isfinite(v).all() or (v<0).any() or (v>1).any() or not np.isclose(v.sum(),1,atol=1e-5):raise ValueError(f'Invalid probabilities: {f}')
        probs.append(v);labels.append(label);ids.append(f.relative_to(a.data).as_posix())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,probs=np.array(probs),true=np.array(labels),image_ids=np.array(ids),class_names=np.array(NAMES))
    meta={'n':len(labels),'class_names':NAMES,'imgsz':a.imgsz,'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'python':platform.python_version(),'ultralytics':ultralytics.__version__,'torch':torch.__version__,'scope':'New inference run; not proof of historical prediction provenance.'}
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print(a.output)
if __name__=='__main__':main()

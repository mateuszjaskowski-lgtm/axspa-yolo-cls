"""Single-image morphology classification using validated checkpoint class names."""
import argparse, json
from pathlib import Path
import numpy as np
from class_mapping import NAMES, probability_order

def predict(image_path, model_path, imgsz=640):
    from ultralytics import YOLO
    image_path, model_path = Path(image_path), Path(model_path)
    if not image_path.is_file() or not model_path.is_file():
        raise FileNotFoundError('An existing image and local model checkpoint are required.')
    model = YOLO(str(model_path))
    order = probability_order(model.names)
    result = model.predict(source=str(image_path), imgsz=imgsz, verbose=False)[0]
    if result.probs is None:
        raise ValueError('A classification checkpoint is required.')
    probs = np.asarray(result.probs.data.cpu().numpy(), dtype=float)[order]
    if probs.shape != (4,) or not np.isfinite(probs).all() or (probs < 0).any() or not np.isclose(probs.sum(),1,atol=1e-5):
        raise ValueError('Invalid classification probabilities.')
    return {'predicted_category':NAMES[int(probs.argmax())],
            'class_probabilities':dict(zip(NAMES,probs.tolist())),
            'binary_dominant_category_score':float(probs[2:].sum()),
            'scope':'Morphological classification; not clinical axSpA diagnosis.'}

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--imgsz',type=int,default=640);a=p.parse_args()
    print(json.dumps(predict(a.image,a.model,a.imgsz),indent=2))

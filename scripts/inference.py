"""
Minimal inference script for axSpA radiographic classification.

Usage:
    python inference.py --image path/to/radiograph.jpg --model best.pt

Requirements:
    - ultralytics >= 8.3.246
    - Python >= 3.10

Output:
    - Predicted grade (0-3)
    - Confidence score
    - All class probabilities
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics==8.3.246")
    sys.exit(1)


CLASS_NAMES = {
    0: "Normal",
    1: "Osteophytes",
    2: "Parasyndesmophytes",
    3: "Syndesmophytes",
}


def predict(image_path: Path, model_path: Path, imgsz: int = 640) -> dict:
    """
    Run YOLO11l-cls inference on a single radiograph.

    Args:
        image_path: Path to input radiograph (JPG, PNG, or TIF)
        model_path: Path to trained YOLO11l-cls weights (.pt file)
        imgsz: Input image size (default 640, matching training)

    Returns:
        dict with:
            - predicted_grade (int, 0-3)
            - predicted_label (str)
            - confidence (float, 0-1)
            - class_probabilities (dict, all 4 class probs)
    """
    # Validate paths
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    # Load model
    model = YOLO(str(model_path))

    # Predict
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        verbose=False,
    )

    # Extract classification probabilities
    result = results[0]
    if result.probs is None:
        raise ValueError(
            "Model did not return classification probabilities. "
            "Ensure this is a YOLO-cls model (not YOLO detection)."
        )

    probs = result.probs.data.cpu().numpy()
    predicted_grade = int(probs.argmax())
    confidence = float(probs[predicted_grade])

    return {
        "predicted_grade": predicted_grade,
        "predicted_label": CLASS_NAMES[predicted_grade],
        "confidence": confidence,
        "class_probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="axSpA radiographic classification inference"
    )
    parser.add_argument(
        "--image", "-i", type=Path, required=True,
        help="Path to input radiograph (JPG/PNG/TIF)"
    )
    parser.add_argument(
        "--model", "-m", type=Path, required=True,
        help="Path to YOLO11l-cls weights (.pt)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640, matching training)"
    )
    args = parser.parse_args()

    result = predict(args.image, args.model, args.imgsz)

    # Pretty print
    print(f"\n{'='*60}")
    print(f"Image: {args.image.name}")
    print(f"{'='*60}")
    print(f"Predicted grade: {result['predicted_grade']} ({result['predicted_label']})")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"\nClass probabilities:")
    for label, prob in result["class_probabilities"].items():
        bar = "█" * int(prob * 40)
        print(f"  {label:22s} {prob:.3f}  {bar}")
    print()

    # Clinical interpretation
    if result["predicted_grade"] >= 2:
        print("⚠️  axSpA-related structural change detected — clinical correlation advised.")
    else:
        print("ℹ️  No axSpA-related structural change detected on this radiograph.")
    print()


if __name__ == "__main__":
    main()

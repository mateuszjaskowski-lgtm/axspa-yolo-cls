# Example Predictions

Sample outputs from `scripts/inference.py` on representative radiographs.

## Correctly classified — Grade 0 (Normal)

```
Predicted grade: 0 (Normal)
Confidence: 1.000

Class probabilities:
  Normal                 1.000  ████████████████████████████████████████
  Osteophytes            0.000
  Parasyndesmophytes     0.000
  Syndesmophytes         0.000

ℹ️  No axSpA-related structural change detected on this radiograph.
```

## Correctly classified — Grade 2 (Parasyndesmophytes)

```
Predicted grade: 2 (Parasyndesmophytes)
Confidence: 1.000

Class probabilities:
  Normal                 0.000
  Osteophytes            0.000
  Parasyndesmophytes     1.000  ████████████████████████████████████████
  Syndesmophytes         0.000

⚠️  axSpA-related structural change detected — clinical correlation advised.
```

## Correctly classified — Grade 3 (Syndesmophytes)

```
Predicted grade: 3 (Syndesmophytes)
Confidence: 1.000

Class probabilities:
  Normal                 0.000
  Osteophytes            0.000
  Parasyndesmophytes     0.000
  Syndesmophytes         1.000  ████████████████████████████████████████

⚠️  axSpA-related structural change detected — clinical correlation advised.
```

## Misclassification example — Grade 2 predicted as Grade 1

```
True label: 2 (Parasyndesmophytes)
Predicted grade: 1 (Osteophytes)
Confidence: 0.99

Class probabilities:
  Normal                 0.005
  Osteophytes            0.987  ███████████████████████████████████████
  Parasyndesmophytes     0.008
  Syndesmophytes         0.000

⚠️  Confident misclassification — illustrates the morphological overlap
    between intermediate inflammatory new bone formation (parasyndesmophytes)
    and degenerative osteophytes. See Figure 5g in the manuscript for
    Eigen-CAM saliency map showing the anatomical basis of this error.
```

See the manuscript Figure 5 for full Eigen-CAM visualizations demonstrating
model attention on vertebral body margins and disc spaces.

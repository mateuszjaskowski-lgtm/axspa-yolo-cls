!pip -q install ultralytics scikit-learn pandas
ZIP = "/sacroiliitis train val_ new.zip"   # <- wklej swoją nazwę z list(uploaded.keys())
!mkdir -p /content/dataset
!unzip -q "$ZIP" -d /content/dataset
!find /content/dataset -maxdepth 2 -type d -print

# export_metrics_cls.py
# Walidacja YOLO-cls + eksport metryk per klasa (precision/recall/F1) + summary.json + INFERENCE TIME
from ultralytics import YOLO
from pathlib import Path
import os, json, time
import numpy as np
import pandas as pd

# === 1) USTAWIENIA – PODMIEŃ NA SWOJE ===
WEIGHTS = "/content/model_- 30 september 2025 14_14.pt"
DATA    = "/content/dataset/sacroiliitis train val_ new"  # root z train/val(/test)
SPLIT   = "val"  # <- daj dokładnie nazwę Twojego podfolderu (np. "val" / "val_new" / "val_final")
PROJECT = "runs/val"
NAME    = "exp1"

# === NOWE: parametry pomiaru inference time ===
IMGSZ = 640          # rozmiar obrazu (ten sam co przy treningu)
DEVICE = 0           # GPU 0 (użyj 'cpu' dla CPU-only)
N_WARMUP = 10        # liczba warmup runs (nie liczone)
N_MEASURE = 200      # liczba pomiarów (używanych do statystyki)

# === 2) Sanity check ścieżek ===
DATA = str(Path(DATA))  # normalizacja
split_dir = Path(DATA) / SPLIT
if not split_dir.is_dir():
    subdirs = [d.name for d in Path(DATA).iterdir() if d.is_dir()]
    raise FileNotFoundError(f"Nie istnieje: {split_dir}\nW {DATA} są: {subdirs}")

# === 3) Walidacja (KLASYFIKACJA) ===
model = YOLO(WEIGHTS)
results = model.val(data=DATA, split=SPLIT, project=PROJECT, name=NAME, plots=True)

# Katalog runa
save_dir = Path(getattr(results, "save_dir", Path(PROJECT) / NAME))
save_dir.mkdir(parents=True, exist_ok=True)

# === 4) Confusion Matrix -> pandas/numpy + etykiety ===
cm_df = results.confusion_matrix.to_df()
# jeżeli to Polars, zamień na pandas
try:
    cm_df = cm_df.to_pandas()
except Exception:
    pass

# jeśli tabela ma kolumnę z etykietami w pierwszej kolumnie — ustaw jako index
if cm_df.shape[0] != cm_df.shape[1]:
    cm_df = cm_df.set_index(cm_df.columns[0])

cm_numeric = cm_df.apply(pd.to_numeric, errors="coerce")
cm = cm_numeric.to_numpy(dtype=float)

# etykiety klas
names = getattr(results, "names", None)
if isinstance(names, dict):
    classes = [names[i] for i in range(cm.shape[0])]
elif isinstance(names, (list, tuple)):
    classes = list(names)[:cm.shape[0]]
else:
    classes = list(map(str, cm_df.index))

# === 5) precision / recall / F1 per klasa ===
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
support = cm.sum(axis=1)

precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp+fp)!=0)
recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp+fn)!=0)
f1        = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(tp), where=(precision+recall)!=0)

per_class = pd.DataFrame({
    "class": classes,
    "support": support.astype(int),
    "precision": precision,
    "recall": recall,
    "f1": f1
}).sort_values("class").reset_index(drop=True)

# === 6) Zapisy CSV/XLSX ===
csv_path  = save_dir / "per_class_metrics.csv"
xlsx_path = save_dir / "per_class_metrics.xlsx"
per_class.to_csv(csv_path, index=False)
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
    per_class.to_excel(w, sheet_name=f"per_class_{SPLIT}", index=False)

# ==========================================================================
# === 7) NOWE: POMIAR INFERENCE TIME =========================================
# ==========================================================================
print("\n" + "=" * 60)
print("POMIAR INFERENCE TIME")
print("=" * 60)

# Znajdź wszystkie obrazy w split directory (rekurencyjnie)
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
test_images = [
    p for p in split_dir.rglob('*')
    if p.suffix.lower() in image_extensions and p.is_file()
]

if not test_images:
    print(f"⚠️  Nie znaleziono obrazów w {split_dir}")
    inference_stats = None
else:
    print(f"Znaleziono {len(test_images)} obrazów w {split_dir}")
    print(f"Warmup: {N_WARMUP} runs, Measurement: {N_MEASURE} runs")
    print(f"Image size: {IMGSZ}×{IMGSZ}, Device: {DEVICE}")

    # Wybierz reprezentatywny obraz (pierwszy)
    test_img = str(test_images[0])
    print(f"Test image: {test_img}")

    # ---- WARMUP (te predykcje NIE są liczone) ----
    print(f"\nWarmup ({N_WARMUP} runs)...")
    for _ in range(N_WARMUP):
        _ = model.predict(test_img, verbose=False, imgsz=IMGSZ, device=DEVICE)

    # ---- POMIAR ----
    print(f"Measuring ({N_MEASURE} runs)...")
    times_ms = []
    for i in range(N_MEASURE):
        start = time.perf_counter()
        _ = model.predict(test_img, verbose=False, imgsz=IMGSZ, device=DEVICE)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

    times_arr = np.array(times_ms)

    # Statystyki
    inference_stats = {
        "n_warmup": N_WARMUP,
        "n_measure": N_MEASURE,
        "imgsz": IMGSZ,
        "device": str(DEVICE),
        "test_image": test_img,
        "mean_ms": float(np.mean(times_arr)),
        "median_ms": float(np.median(times_arr)),
        "std_ms": float(np.std(times_arr)),
        "min_ms": float(np.min(times_arr)),
        "max_ms": float(np.max(times_arr)),
        "p25_ms": float(np.percentile(times_arr, 25)),
        "p75_ms": float(np.percentile(times_arr, 75)),
        "p95_ms": float(np.percentile(times_arr, 95)),
        "throughput_img_per_sec": float(1000.0 / np.mean(times_arr)),
    }

    print(f"\n--- INFERENCE TIME RESULTS ---")
    print(f"  Median:     {inference_stats['median_ms']:.2f} ms/image")
    print(f"  Mean:       {inference_stats['mean_ms']:.2f} ms/image")
    print(f"  Std:        {inference_stats['std_ms']:.2f} ms")
    print(f"  Min/Max:    {inference_stats['min_ms']:.2f} / {inference_stats['max_ms']:.2f} ms")
    print(f"  P25/P75:    {inference_stats['p25_ms']:.2f} / {inference_stats['p75_ms']:.2f} ms")
    print(f"  P95:        {inference_stats['p95_ms']:.2f} ms")
    print(f"  Throughput: {inference_stats['throughput_img_per_sec']:.1f} images/second")
    print(f"\n>>> DO MANUSKRYPTU (median): {inference_stats['median_ms']:.1f} ms per image <<<")

    # Zapisz szczegółowe pomiary do CSV
    pd.DataFrame({"iteration": range(1, N_MEASURE+1), "time_ms": times_ms}).to_csv(
        save_dir / "inference_time_raw.csv", index=False
    )

    # Zapisz statystyki do JSON
    with open(save_dir / "inference_time.json", "w") as f:
        json.dump(inference_stats, f, indent=2)

# ==========================================================================
# === 8) Summary JSON (rozszerzony o inference time) =======================
# ==========================================================================
def js(x):
    try:
        if hasattr(x, "item"):
            return x.item()
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    if isinstance(x, Path):
        return str(x)
    return x

summary = {
    "split": SPLIT,
    "top1": js(getattr(results, "top1", None)),
    "top5": js(getattr(results, "top5", None)),
    "loss": js(getattr(results, "loss", None)),
    "num_images": js(getattr(results, "images", None)),
    "save_dir": str(save_dir),
    "inference_time": inference_stats,   # <-- NOWE
}

with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("ZAPISANE PLIKI:")
print("=" * 60)
print(f"- {csv_path}")
print(f"- {xlsx_path}")
print(f"- {save_dir / 'summary.json'}")
if inference_stats:
    print(f"- {save_dir / 'inference_time.json'}")
    print(f"- {save_dir / 'inference_time_raw.csv'}")
print(f"\nKatalog runa: {save_dir}")

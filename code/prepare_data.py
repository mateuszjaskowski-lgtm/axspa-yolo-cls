"""Verify and copy the accepted 824-image dataset without changing its inputs.

Two registered PNG archives are accepted. Both contain the same 824 paths,
labels, split assignments, dimensions and decoded RGB pixels. The original PNG
representation produces the manifest used by the completed study. The registered
PNG byte variant produces its own exact file-hash manifest for future reproduction;
it cannot replace original PNG bytes in original-checkpoint locked replay.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

ORIGINAL_PNG_ARCHIVE_SHA256 = "ba4c5ece78779f67b209f839cec75f70aa480c07bc8ee1dee008f4594a96613d"
PNG_BYTE_VARIANT_ARCHIVE_SHA256 = "51f54607bb58b2d5492dd129e6c31b0198e67ce58a3efc19c4e1d4156aafd615"
PNG_BYTE_VARIANT_MANIFEST_SHA256 = "f98cc4992fe3da8140ed1ab033361e86188fc3e9c9c04d814ac09af3bb869642"
ACCEPTED_ARCHIVES = {
    ORIGINAL_PNG_ARCHIVE_SHA256: "original_png_representation",
    PNG_BYTE_VARIANT_ARCHIVE_SHA256: "registered_rgb_equivalent_png_byte_variant",
}
CLEAN_MANIFEST_SHA256 = "3016eaeeb8af3745c64b159c5189549fadde3753aa08d232ea7e7d450f4e4dfd"
CLASSES = ["normal", "osteofity", "parasyndesmofity", "syndesmofity"]
EXPECTED_COUNTS = {
    "train": [211, 210, 81, 77], "val": [37, 34, 12, 12], "test": [55, 54, 20, 21]
}

def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def canonical_rgb_hash(im):
    rgb = im.convert("RGB")
    return hashlib.sha256(f"RGB:{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).with_name("reference_clean_manifest.csv"))
    args = ap.parse_args(argv)
    input_sha256 = sha_file(args.archive)
    input_kind = ACCEPTED_ARCHIVES.get(input_sha256)
    if input_kind is None:
        raise ValueError("Archive is not one of the two explicitly registered 824-image archives.")
    byte_variant = input_sha256 == PNG_BYTE_VARIANT_ARCHIVE_SHA256
    if sha_file(args.manifest) != CLEAN_MANIFEST_SHA256:
        raise ValueError("Accepted 824-image reference manifest checksum mismatch.")
    with args.manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    reference = {r["relative_path"]: r for r in rows}
    if len(rows) != 824 or len(reference) != 824:
        raise ValueError("Expected 824 unique reference manifest paths.")
    counts = {s: [sum(r["split"] == s and r["class_name"] == c for r in rows) for c in CLASSES]
              for s in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected accepted class counts: {counts}")
    for r in rows:
        if int(r["label"]) != CLASSES.index(r["class_name"]):
            raise ValueError(f"Reference label disagrees with the class order: {r['relative_path']}")
    expected_input_paths = set(reference)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="axspa_verified_", dir=args.output_dir.parent))
    new_data = temp / "dataset"
    independent_hashes = {}
    actual_file_hashes = {}
    alpha_stats = []
    mode_counts = Counter()
    try:
        with zipfile.ZipFile(args.archive) as z:
            files = {}
            for info in z.infolist():
                if info.is_dir():
                    continue
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts or len(path.parts) != 4 or path.parts[0] != "baza danych":
                    raise ValueError(f"Unexpected archive path: {info.filename}")
                rel = str(PurePosixPath(*path.parts[1:]))
                if rel in files:
                    raise ValueError(f"Repeated ZIP member: {rel}")
                files[rel] = info
            if set(files) != expected_input_paths:
                missing = sorted(expected_input_paths - set(files))
                unexpected = sorted(set(files) - expected_input_paths)
                raise ValueError(f"ZIP paths do not match the accepted 824-image path set. "
                                 f"Missing: {missing}; unexpected: {unexpected}")
            for r in rows:
                rel = r["relative_path"]
                blob = z.read(files[rel])
                actual_file_hashes[rel] = hashlib.sha256(blob).hexdigest()
                if not byte_variant and actual_file_hashes[rel] != r["sha256"]:
                    raise ValueError(f"Image-byte mismatch: {rel}")
                with Image.open(io.BytesIO(blob)) as im:
                    im.load()
                    if im.size != (int(r["width"]), int(r["height"])):
                        raise ValueError(f"Dimensions differ: {rel}")
                    mode_counts[im.mode] += 1
                    independent_hashes[rel] = canonical_rgb_hash(im)
                    if independent_hashes[rel] != r["independent_rgb_sha256"]:
                        raise ValueError(f"Decoded RGB pixels differ from the accepted reference: {rel}")
                    if "A" in im.getbands():
                        a = np.asarray(im.getchannel("A"))
                        alpha_stats.append({"relative_path": rel,
                                            "transparent_fraction": float(np.mean(a == 0)),
                                            "partial_alpha_fraction": float(np.mean((a > 0) & (a < 255)))})
                dest = new_data / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(blob)
        if len(set(actual_file_hashes.values())) != 824:
            raise ValueError("File-byte hashes do not confirm exact-image uniqueness.")
        if len(set(independent_hashes.values())) != 824:
            raise ValueError("Decoded RGB hashes do not confirm exact-image uniqueness.")
        output_rows = [{**r, "sha256": actual_file_hashes[r["relative_path"]],
                        "independent_rgb_sha256": independent_hashes[r["relative_path"]]}
                       for r in rows]
        with (temp / "clean_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)
        clean_manifest_sha256 = sha_file(temp / "clean_manifest.csv")
        expected_manifest_sha256 = PNG_BYTE_VARIANT_MANIFEST_SHA256 if byte_variant else CLEAN_MANIFEST_SHA256
        if clean_manifest_sha256 != expected_manifest_sha256:
            raise ValueError("Generated manifest differs from the registered manifest for this PNG representation.")
        summary = {
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "source_archive_sha256": input_sha256,
            "source_archive_kind": input_kind,
            "reference_manifest_sha256": CLEAN_MANIFEST_SHA256,
            "clean_manifest_sha256": clean_manifest_sha256,
            "source_image_entries": 824,
            "prepared_image_entries": 824,
            "counts": counts,
            "classes": CLASSES,
            "exact_file_byte_uniqueness_confirmed": True,
            "exact_decoded_rgb_uniqueness_confirmed": True,
            "paths_labels_and_split_assignments_match_reference": True,
            "rgb_pixels_and_dimensions_match_reference": True,
            "source_modes": dict(mode_counts),
            "alpha_statistics": alpha_stats,
            "source_png_bytes_modified": False,
            "pixel_hash_note": "The pixel_sha256 column is preserved reference metadata; independent_rgb_sha256 uses SHA256(b'RGB:{width}x{height}:' + RGB bytes).",
            "alpha_handling": "Training/evaluation loaders use PIL convert('RGB'), discarding alpha without compositing. Underlying RGB values are retained.",
            "patient_independence": "One image per patient was reported by the author; independent record linkage was not available. Image-hash checks do not establish patient independence.",
            "input_preparation": "M.J. prepared PNGs with Windows Snipping Tool by outlining the examined spinal segment with surrounding tissues before radiologist labeling; the preparer did not know the labels. No manual brightness/contrast adjustment or CLAHE was applied. This script performs no additional cropping or image enhancement.",
        }
        if byte_variant:
            summary["png_byte_variant"] = {
                "scope": "Future reproduction only; not a replacement for original locked PNG hashes or original-checkpoint replay inputs.",
                "reference_clean_manifest_sha256": CLEAN_MANIFEST_SHA256,
                "current_clean_manifest_sha256": PNG_BYTE_VARIANT_MANIFEST_SHA256,
                "rgb_pixels_dimensions_paths_and_labels_match_reference": True,
                "pixel_sha256_note": "The pixel_sha256 column retains its reference definition; independent_rgb_sha256 is the verified canonical RGB hash for both representations.",
                "original_archived_files_modified": False,
                "file_byte_hash_mapping": [
                    {"relative_path": r["relative_path"],
                     "original_png_sha256": r["sha256"],
                     "current_source_sha256": actual_file_hashes[r["relative_path"]],
                     "independent_rgb_sha256": independent_hashes[r["relative_path"]]}
                    for r in rows],
            }
        (temp / "data_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.output_dir.exists():
            old_manifest = args.output_dir / "clean_manifest.csv"
            if not old_manifest.exists() or sha_file(old_manifest) != summary["clean_manifest_sha256"]:
                raise FileExistsError("Existing destination is different; choose a new output directory.")
            for r in output_rows:
                if sha_file(args.output_dir / "dataset" / r["relative_path"]) != r["sha256"]:
                    raise ValueError("Existing prepared image changed; use a fresh output directory.")
            actual_output_paths = {p.relative_to(args.output_dir / "dataset").as_posix()
                                   for p in (args.output_dir / "dataset").rglob("*") if p.is_file()}
            if actual_output_paths != expected_input_paths:
                raise ValueError("Existing prepared dataset has missing or unexpected files; use a fresh output directory.")
            # Preserve the first preparation record; record later verification separately.
            verification_dir = args.output_dir / "verification_invocations"
            verification_dir.mkdir(exist_ok=True)
            verification_file = verification_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".json")
            verification_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("Existing accepted dataset verified without overwriting.")
        else:
            os.replace(temp, args.output_dir)
        print(json.dumps({"output_dir": str(args.output_dir.resolve()), "counts": counts,
                          "source_archive_kind": input_kind, "source_image_entries": 824,
                          "prepared_image_entries": 824,
                          "manifest_sha256": summary["clean_manifest_sha256"]}, indent=2))
    finally:
        if temp.exists():
            shutil.rmtree(temp)

if __name__ == "__main__":
    main()

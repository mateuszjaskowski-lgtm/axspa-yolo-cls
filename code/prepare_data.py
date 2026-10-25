"""Verify the supplied archive and create a separate deduplicated dataset.

No original images are modified. Three explicitly verified archives are accepted:
the historical 827-entry archive and its 824-entry version with the three
approved copies already removed. Both produce identical retained PNG bytes and
the same clean manifest. The already-clean archive is not deduplicated again.
The registered 824-image PNG byte variant has identical decoded RGB pixels,
dimensions and assignments. It produces a separate manifest of its actual file
hashes for future reproduction only; original-checkpoint locked replay still
requires the original PNG bytes and historical manifest.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

ARCHIVE_SHA256 = "c4eb673704b93daaceaea208a3d1fd202d900c2ef8d013c9b73a07a37a320bdf"
ALREADY_CLEAN_ARCHIVE_SHA256 = "ba4c5ece78779f67b209f839cec75f70aa480c07bc8ee1dee008f4594a96613d"
PNG_BYTE_VARIANT_ARCHIVE_SHA256 = "51f54607bb58b2d5492dd129e6c31b0198e67ce58a3efc19c4e1d4156aafd615"
PNG_BYTE_VARIANT_MANIFEST_SHA256 = "f98cc4992fe3da8140ed1ab033361e86188fc3e9c9c04d814ac09af3bb869642"
ACCEPTED_ARCHIVES = {
    ARCHIVE_SHA256: "historical_original",
    ALREADY_CLEAN_ARCHIVE_SHA256: "already_deduplicated",
    PNG_BYTE_VARIANT_ARCHIVE_SHA256: "registered_rgb_equivalent_png_byte_variant",
}
MANIFEST_SHA256 = "b2b4d559fcee93f4f93f8b613d204fe78d6d7f506685910917616ab3f8e83746"
CLEAN_MANIFEST_SHA256 = "3016eaeeb8af3745c64b159c5189549fadde3753aa08d232ea7e7d450f4e4dfd"
CLASSES = ["normal", "osteofity", "parasyndesmofity", "syndesmofity"]
PAIRS = [
    ("train/normal/263.png", "train/normal/264.png"),
    ("train/normal/84.png", "train/normal/85.png"),
    ("train/osteofity/114.png", "train/osteofity/115.png"),
]
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
    ap.add_argument("--manifest", type=Path, default=Path(__file__).with_name("original_manifest.csv"))
    args = ap.parse_args(argv)
    input_sha256 = sha_file(args.archive)
    input_kind = ACCEPTED_ARCHIVES.get(input_sha256)
    if input_kind is None:
        raise ValueError("Archive is not one of the three explicitly registered study archives.")
    byte_variant = input_sha256 == PNG_BYTE_VARIANT_ARCHIVE_SHA256
    reference = None
    if byte_variant:
        reference_path = Path(__file__).with_name("reference_clean_manifest.csv")
        if sha_file(reference_path) != CLEAN_MANIFEST_SHA256:
            raise ValueError("Frozen RGB-reference manifest checksum mismatch.")
        with reference_path.open(newline="", encoding="utf-8") as handle:
            reference_rows = list(csv.DictReader(handle))
        reference = {r["relative_path"]: r for r in reference_rows}
        if len(reference_rows) != 824 or len(reference) != 824:
            raise ValueError("Expected 824 unique frozen RGB-reference records.")
    if sha_file(args.manifest) != MANIFEST_SHA256:
        raise ValueError("Historical manifest checksum mismatch.")
    with args.manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f); fields = list(reader.fieldnames); rows = list(reader)
    if len(rows) != 827:
        raise ValueError("Expected 827 original entries.")
    by_path = {r["relative_path"]: r for r in rows}
    if len(by_path) != 827:
        raise ValueError("Duplicate manifest paths.")
    omitted = {b for _, b in PAIRS}
    retained = [r for r in rows if r["relative_path"] not in omitted]
    # The fixed historical manifest documents the approved duplicate pairs even
    # when the removed copies are absent from the already-clean input archive.
    for keep, drop in PAIRS:
        if any(by_path[keep][field] != by_path[drop][field]
               for field in ("sha256", "pixel_sha256", "width", "height")):
            raise ValueError(f"Historical duplicate pair no longer matches: {keep}, {drop}")
    input_rows = rows if input_kind == "historical_original" else retained
    expected_input_paths = {r["relative_path"] for r in input_rows}
    removed_this_invocation = sorted(omitted) if input_kind == "historical_original" else []
    previously_removed = sorted(omitted) if input_kind != "historical_original" else []
    counts = {s: [sum(r["split"] == s and r["class_name"] == c for r in retained) for c in CLASSES]
              for s in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected cleaned class counts: {counts}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="axspa_verified_", dir=args.output_dir.parent))
    new_data = temp / "dataset"
    independent_hashes = {}; actual_file_hashes = {}; alpha_stats = []; mode_counts = Counter()
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
                raise ValueError(f"ZIP paths do not match the verified {input_kind} path set. "
                                 f"Missing: {missing}; unexpected: {unexpected}")
            for r in input_rows:
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
                    if byte_variant and independent_hashes[rel] != reference[rel]["independent_rgb_sha256"]:
                        raise ValueError(f"Registered PNG variant has changed RGB pixels: {rel}")
                    if "A" in im.getbands():
                        a = np.asarray(im.getchannel("A"))
                        alpha_stats.append({"relative_path": rel,
                                            "transparent_fraction": float(np.mean(a == 0)),
                                            "partial_alpha_fraction": float(np.mean((a > 0) & (a < 255)))})
                if rel not in omitted:
                    dest = new_data / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(blob)
            for keep, drop in PAIRS:
                if input_kind == "historical_original" and independent_hashes[keep] != independent_hashes[drop]:
                    raise ValueError(f"Accepted pair no longer matches: {keep}, {drop}")
        groups = defaultdict(list)
        for r in retained:
            groups[independent_hashes[r["relative_path"]]].append(r["relative_path"])
        if any(len(v) > 1 for v in groups.values()):
            raise ValueError("Unexpected duplicate remains after the three specified omissions.")
        output_rows = [{**r, "sha256": actual_file_hashes[r["relative_path"]],
                        "independent_rgb_sha256": independent_hashes[r["relative_path"]]}
                       for r in retained]
        with (temp / "clean_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields + ["independent_rgb_sha256"])
            writer.writeheader()
            writer.writerows(output_rows)
        clean_manifest_sha256 = sha_file(temp / "clean_manifest.csv")
        expected_manifest_sha256 = PNG_BYTE_VARIANT_MANIFEST_SHA256 if byte_variant else CLEAN_MANIFEST_SHA256
        if clean_manifest_sha256 != expected_manifest_sha256:
            raise ValueError("Generated clean manifest differs from the previously verified clean manifest.")
        summary = {
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "source_archive_sha256": input_sha256,
            "source_archive_kind": input_kind,
            "historical_original_archive_sha256": ARCHIVE_SHA256,
            "historical_manifest_sha256": MANIFEST_SHA256,
            "clean_manifest_sha256": clean_manifest_sha256,
            "historical_image_entries": 827,
            "source_image_entries": len(input_rows), "retained_unique_images": 824,
            "counts": counts, "classes": CLASSES,
            "omitted_duplicate_copies": sorted(omitted),
            "omitted_duplicate_copies_note": "Approved omissions relative to the historical manifest, not necessarily removals in this invocation.",
            "removed_duplicate_copies_this_invocation": removed_this_invocation,
            "previously_removed_duplicate_copies": previously_removed,
            "exact_cross_split_duplicates_after_deduplication": [],
            "source_modes": dict(mode_counts), "alpha_statistics": alpha_stats,
            "source_png_bytes_modified": False,
            "pixel_hash_note": "Historical pixel_sha256 preserved as metadata; independent_rgb_sha256 uses SHA256(b'RGB:{width}x{height}:' + RGB bytes).",
            "alpha_handling": "Training/evaluation loaders use PIL convert('RGB'), discarding alpha without compositing. Underlying RGB values are retained.",
            "patient_independence": "One image per patient was reported by the author; independent record linkage was not available. Image-hash checks do not establish patient independence.",
            "input_preparation": "M.J. prepared PNGs with Windows Snipping Tool by outlining the examined spinal segment with surrounding tissues before radiologist labeling; the preparer did not know the labels. No manual brightness/contrast adjustment or CLAHE was applied. This script performs no additional manual cropping or image enhancement.",
        }
        if byte_variant:
            summary["png_byte_variant"] = {
                "scope": "Future reproduction only; not a replacement for original locked PNG hashes or original-checkpoint replay inputs.",
                "reference_clean_manifest_sha256": CLEAN_MANIFEST_SHA256,
                "current_clean_manifest_sha256": PNG_BYTE_VARIANT_MANIFEST_SHA256,
                "rgb_pixels_dimensions_paths_and_labels_match_reference": True,
                "historical_pixel_sha256_note": "The pixel_sha256 column remains historical metadata with its original definition; independent_rgb_sha256 is the independently verified canonical RGB hash for both representations.",
                "original_archived_files_modified": False,
                "file_byte_hash_mapping": [
                    {"relative_path": r["relative_path"],
                     "historical_source_sha256": r["sha256"],
                     "current_source_sha256": actual_file_hashes[r["relative_path"]],
                     "independent_rgb_sha256": independent_hashes[r["relative_path"]]}
                    for r in retained],
            }
        (temp / "data_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        if args.output_dir.exists():
            old_manifest = args.output_dir / "clean_manifest.csv"
            if not old_manifest.exists() or sha_file(old_manifest) != summary["clean_manifest_sha256"]:
                raise FileExistsError("Existing destination is different; choose a new output directory.")
            for r in output_rows:
                if sha_file(args.output_dir / "dataset" / r["relative_path"]) != r["sha256"]:
                    raise ValueError("Existing prepared image changed; use a fresh output directory.")
            actual_output_paths = {p.relative_to(args.output_dir / "dataset").as_posix()
                                   for p in (args.output_dir / "dataset").rglob("*") if p.is_file()}
            if actual_output_paths != {r["relative_path"] for r in retained}:
                raise ValueError("Existing prepared dataset has missing or unexpected files; use a fresh output directory.")
            # Keep the original preparation record while recording which input
            # archive was verified during this later invocation.
            verification_dir = args.output_dir / "verification_invocations"
            verification_dir.mkdir(exist_ok=True)
            verification_file = verification_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".json")
            verification_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("Existing clean dataset verified; retained without overwriting.")
        else:
            os.replace(temp, args.output_dir)
        print(json.dumps({"output_dir": str(args.output_dir.resolve()), "counts": counts,
                          "source_archive_kind": input_kind, "source_image_entries": len(input_rows),
                          "removed_this_invocation": len(removed_this_invocation),
                          "previously_removed": len(previously_removed),
                          "unique_images": 824, "manifest_sha256": summary["clean_manifest_sha256"]}, indent=2))
    finally:
        if temp.exists():
            shutil.rmtree(temp)

if __name__ == "__main__":
    main()

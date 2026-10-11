"""All-pairs multi-metric candidate generation; thresholds never confirm identity."""
import csv
import itertools
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import yaml
import _bootstrap
from blood_cell_detection.data import sha, save_json
RAW = ROOT / 'data/raw/bccd'
SPLITS = ('train', 'val', 'test')

SCAN_VERSION = "all-pairs-phash-dhash-correlation-v1"
PARAMETERS = {"phash": "32x32 INTER_AREA grayscale, top-left 8x8 DCT; median excluding DC; 64 bits",
              "dhash": "9x8 INTER_AREA grayscale, horizontal right>left, 64 bits",
              "correlation": "128x96 INTER_AREA grayscale, Pearson correlation of centered pixels, float64",
              "candidate_rule": "(pHash<=12 and (dHash<=24 or corr>=0.60)) OR (dHash<=12 and corr>=0.65) OR corr>=0.80",
              "display_order": "pHash distance ascending, then IDs; all candidates shown",
              "seed": 42, "random_sampling": False}


def bit_hash(bits):
    return sum(int(value) << i for i, value in enumerate(bits.flatten()))


def scan():
    (ROOT / 'outputs/dataset_analysis').mkdir(parents=True, exist_ok=True)
    paths = sorted((RAW / "BCCD/JPEGImages").glob("*.jpg"))
    images, ph, dh, direct = {}, {}, {}, []
    upstream = {}
    for split in SPLITS:
        for i in (RAW / f"BCCD/ImageSets/Main/{split}.txt").read_text().split():
            if i in upstream:
                raise RuntimeError("Overlapping upstream IDs")
            upstream[i] = split
    proposed = dict(upstream)
    correction = yaml.safe_load((ROOT / "configs/split_corrections.yaml").read_text(encoding="utf-8"))
    for item in correction["corrections"]:
        if proposed[item["image_id"]] != item["from"] or "test" in (item["from"], item["to"]):
            raise RuntimeError("Invalid or test-affecting split correction")
        proposed[item["image_id"]] = item["to"]
    for p in paths:
        with Image.open(p) as im:
            gray = np.asarray(im.convert("L"))
        images[p.stem] = p
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(small.astype(np.float32))[:8, :8]
        ph[p.stem] = bit_hash(dct > np.median(dct.flatten()[1:]))
        small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        dh[p.stem] = bit_hash(small[:, 1:] > small[:, :-1])
        v = cv2.resize(gray, (128, 96), interpolation=cv2.INTER_AREA).astype(np.float64).flatten()
        v -= v.mean()
        norm = np.linalg.norm(v)
        if norm == 0:
            raise RuntimeError(f"Constant grayscale image: {p.stem}")
        direct.append(v / norm)
    corr = np.asarray(direct) @ np.asarray(direct).T
    ids = list(images)
    candidates, checked = [], 0
    for i, j in itertools.combinations(range(len(ids)), 2):
        checked += 1
        a, b = ids[i], ids[j]
        pd = (ph[a] ^ ph[b]).bit_count()
        dd = (dh[a] ^ dh[b]).bit_count()
        similarity = float(corr[i, j])
        candidate = (pd <= 12 and (dd <= 24 or similarity >= .60)) or (dd <= 12 and similarity >= .65) or similarity >= .80
        if candidate:
            candidates.append({"a": a, "b": b, "upstream_a": upstream[a], "upstream_b": upstream[b],
                "proposed_corrected_a": proposed[a], "proposed_corrected_b": proposed[b],
                "phash_distance": pd, "dhash_distance": dd, "grayscale_correlation": round(similarity, 12),
                "a_sha256": sha(images[a]), "b_sha256": sha(images[b])})
    candidates.sort(key=lambda p: (p["phash_distance"], p["a"], p["b"]))
    fields = ["a", "b", "upstream_a", "upstream_b", "proposed_corrected_a", "proposed_corrected_b",
              "phash_distance", "dhash_distance", "grayscale_correlation", "a_sha256", "b_sha256"]
    with (ROOT / "outputs/dataset_analysis/near_duplicate_candidates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    sanity = any({p["a"], p["b"]} == {"BloodImage_00037", "BloodImage_00038"} for p in candidates)
    if not sanity or checked != len(paths) * (len(paths)-1)//2:
        raise RuntimeError("Known-pair sanity or all-pairs coverage failed")
    report = {"scan_version": SCAN_VERSION, "parameters": PARAMETERS, "images": len(paths),
              "total_pairs_checked": checked, "candidates": len(candidates), "known_pair_detected": sanity,
              "image_sha256": {i: sha(p) for i, p in images.items()},
              "candidate_csv_sha256": sha(ROOT / "outputs/dataset_analysis/near_duplicate_candidates.csv"),
              "decision_policy": "Candidates require visual and multi-metric review; do not automatically reassign any candidate. Stop if confirmed group touches test."}
    save_json(ROOT / "outputs/dataset_analysis/near_duplicate_scan.json", report)
    lines = ["# Full near-duplicate candidate scan", "", f"Version: {SCAN_VERSION}",
             f"All {len(paths)} images, {checked} distinct unordered pairs checked; {len(candidates)} candidates.",
             f"Known 00037/00038 sanity check: {sanity}", "", "## Fixed parameters", ""]
    lines += [f"- {k}: {v}" for k, v in PARAMETERS.items()]
    lines += ["", "Thresholds generate candidates only. No algorithmic duplicate confirmation or sample relocation.",
              "Correlation uses aligned resized grayscale; hashes compare appearance. Neither rules out shifted/rotated/cropped related fields or patient/slide relationships.",
              "Candidate CSV contains source hashes and upstream/proposed corrected membership. Proposed membership is not split approval.", "", "## Candidates", "",
              "| a | b | upstream | proposed corrected | pHash | dHash | gray corr |", "|---|---|---|---|---:|---:|---:|"]
    for p in candidates:
        lines.append(f"| {p['a']} | {p['b']} | {p['upstream_a']}/{p['upstream_b']} | {p['proposed_corrected_a']}/{p['proposed_corrected_b']} | {p['phash_distance']} | {p['dhash_distance']} | {p['grayscale_correlation']:.6f} |")
    group_path = ROOT / "configs/near_duplicate_groups.yaml"
    if group_path.exists():
        review = yaml.safe_load(group_path.read_text(encoding="utf-8"))
        lines += ["", "## Explicit review decisions", "", review["review_source"],
                  "No new candidate is approved by this section; prepare_dataset.py verifies complete candidate review and stops on test groups."]
        for group in review["confirmed_groups"]:
            lines.append(f"- {group['group_id']}: {', '.join(group['members'])}; {group['status']}; destination {group['destination']}. {group['reason']}")
    (ROOT / "outputs/dataset_analysis/near_duplicate_scan.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    out = ROOT / "outputs/dataset_analysis"
    out.mkdir(parents=True, exist_ok=True)
    # Six pairs per page; every candidate is shown, without random subsampling.
    for page, start in enumerate(range(0, len(candidates), 6)):
        selected = candidates[start:start+6]
        fig, axes = plt.subplots(len(selected), 2, figsize=(12, 4.6*len(selected)), squeeze=False)
        for row, p in zip(axes, selected):
            for ax, key in zip(row, ("a", "b")):
                with Image.open(images[p[key]]) as im:
                    ax.imshow(im)
                ax.set_title(f"{p[key]} | upstream={p['upstream_'+key]} | proposed corrected={p['proposed_corrected_'+key]}\npHash={p['phash_distance']} dHash={p['dhash_distance']} corr={p['grayscale_correlation']:.6f}", fontsize=10)
                ax.axis("off")
        fig.suptitle("Multi-metric candidates: visual review required (not automatic duplicates)")
        fig.tight_layout()
        name = "near_duplicate_candidates.png" if page == 0 else f"near_duplicate_candidates_{page+1:02d}.png"
        fig.savefig(out / name, dpi=120, metadata={"Software": SCAN_VERSION})
        plt.close(fig)
    print({k: report[k] for k in ("images", "total_pairs_checked", "candidates", "known_pair_detected")})
    for p in candidates:
        print(p["a"], p["b"], p["upstream_a"], p["upstream_b"], p["phash_distance"], p["dhash_distance"], p["grayscale_correlation"])


if __name__ == "__main__":
    scan()

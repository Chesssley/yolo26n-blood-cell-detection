"""Pinned acquisition, read-only auditing and deterministic VOC conversion."""
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import hashlib
import io
import json
import shutil
import urllib.request

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("RBC", "WBC", "Platelets")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def fetch(root=ROOT):
    destination = root / "data/raw/bccd"
    require(not destination.exists(), "Raw exists; refusing overwrite")
    provenance = load_json(root / "docs/evidence/provenance.json")
    commit = provenance["upstream_commit"]
    with urllib.request.urlopen(provenance["archive_url"], timeout=60) as response:
        data = response.read()
    require(hashlib.sha256(data).hexdigest() == provenance["archive_sha256"], "Archive hash mismatch")
    with ZipFile(io.BytesIO(data)) as archive:
        require(archive.testzip() is None, "Archive CRC failure")
        prefix = f"BCCD_Dataset-{commit}/"
        for info in archive.infolist():
            p = Path(info.filename.removeprefix(prefix))
            require(info.filename.startswith(prefix) and not p.is_absolute() and
                    ".." not in p.parts and ".git" not in p.parts and ":" not in str(p), "Unsafe entry")
        destination.mkdir(parents=True)
        for info in archive.infolist():
            if not info.is_dir():
                p = destination / info.filename.removeprefix(prefix)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(archive.read(info))
    require(sha(destination / "LICENSE") == provenance["license_sha256"], "License hash mismatch")
    print("Pinned source acquired. Raw defects are retained; continue with audit and preparation.")


def read_raw(root=ROOT, raw=None):
    """Reject unknown defects; apply only exact registered corrections in memory."""
    base = Path(raw or root / "data/raw/bccd") / "BCCD"
    provenance = load_json(root / "docs/evidence/provenance.json")
    registry = read_yaml(root / "configs/annotation_corrections.yaml")
    require(registry["upstream_commit"] == provenance["upstream_commit"], "Registry provenance mismatch")
    rules = {(x["image_id"], x["class"], tuple(x["original_bbox"])): x for x in registry["corrections"]}
    require(len(rules) == len(registry["corrections"]), "Duplicate correction rules")
    source_hashes = load_json(root / "docs/evidence/raw_manifest.json")
    raw_root = base.parent
    require(set(source_hashes) == {p.relative_to(raw_root).as_posix() for p in raw_root.rglob("*") if p.is_file()},
            "Raw tree differs from pinned source")
    require(all(sha(raw_root / p) == expected for p, expected in source_hashes.items()), "Raw source hash drift")
    records, images, raw_counts, hits, pixel_hashes = {}, {}, Counter(), Counter(), {}
    for xml in sorted((base / "Annotations").glob("*.xml")):
        x = ET.parse(xml).getroot()
        image = base / "JPEGImages" / (xml.stem + ".jpg")
        with Image.open(image) as im:
            im.load()
            require(im.size == (640, 480) and im.mode == "RGB", "Image format drift")
            pixel_hashes[xml.stem] = hashlib.sha256(im.tobytes()).hexdigest()
        require(x.findtext("filename") == image.name and
                [int(x.findtext("size/" + k)) for k in ("width", "height", "depth")] == [640, 480, 3],
                "VOC image metadata mismatch")
        objects = []
        for obj in x.findall("object"):
            cls = obj.findtext("name")
            require(cls in CLASSES, "Unknown class")
            bbox = tuple(float(obj.findtext("bndbox/" + k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            raw_counts[cls] += 1
            key = xml.stem, cls, bbox
            x1, y1, x2, y2 = bbox
            if key in rules:
                require(rules[key]["action"] == "drop_invalid_zero_area_bbox" and
                        (x1 == x2 or y1 == y2), "Unsupported correction")
                hits[key] += 1
            else:
                require(all(np.isfinite(bbox)) and 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480,
                        f"Unregistered annotation defect: {xml.stem}")
                objects.append((cls, bbox))
        records[xml.stem], images[xml.stem] = objects, image
    require(set(records) == {p.stem for p in (base / "JPEGImages").glob("*.jpg")}, "Unpaired image/XML")
    frozen = load_json(root / "docs/evidence/frozen_results.json")["dataset"]
    require(len(records) == frozen["images"] and dict(raw_counts) == frozen["raw_counts"], "Raw count drift")
    require(all(hits[k] == 1 for k in rules), "Registry entries must each match once")
    counts = Counter(c for objects in records.values() for c, _ in objects)
    require(dict(counts) == frozen["modeling_counts"], "Modeling count drift")
    manifest = load_json(root / "configs/splits/split_manifest.json")
    for p, expected in manifest["config_sha256"].items():
        require(sha(root / p) == expected, "Correction configuration hash drift")
    splits = {s: (root / f"configs/splits/{s}.txt").read_text(encoding="utf-8").split()
              for s in ("train", "val", "test")}
    require({s: len(ids) for s, ids in splits.items()} == frozen["splits"], "Split count drift")
    flat = [i for ids in splits.values() for i in ids]
    require(len(flat) == len(set(flat)) == len(records) and set(flat) == set(records), "Split overlap/coverage")
    for s in splits:
        require(sha(root / f"configs/splits/{s}.txt") == manifest["corrected_split_hashes"][s], "Split hash drift")
    require((root / "configs/splits/test.txt").read_bytes() == (base / "ImageSets/Main/test.txt").read_bytes(),
            "Test membership changed")
    group = read_yaml(root / "configs/near_duplicate_groups.yaml")["confirmed_groups"]
    membership = {i: s for s, ids in splits.items() for i in ids}
    for g in group:
        require({membership[i] for i in g["members"]} == {g["destination"]}, "Acquisition group leakage")
        require(all(sha(images[i]) == g["image_sha256"][i] for i in g["members"]), "Group image drift")
    for hashes in ({i: sha(p) for i, p in images.items()}, pixel_hashes):
        seen = {}
        for i, h in hashes.items():
            require(h not in seen or seen[h] == membership[i], "Exact content leakage")
            seen[h] = membership[i]
    return records, images, splits


def label_text(objects):
    rows = []
    for cls, (x1, y1, x2, y2) in objects:
        values = ((x1 + x2) / 1280, (y1 + y2) / 960, (x2 - x1) / 640, (y2 - y1) / 480)
        rows.append(str(CLASSES.index(cls)) + " " + " ".join(f"{v:.12f}" for v in values))
    return ("\n".join(rows) + "\n").encode("utf-8")


def audit(root=ROOT):
    records, _, splits = read_raw(root)
    result = {"images": len(records), "raw_objects": 4888, "modeling_objects": sum(map(len, records.values())),
              "registered_zero_area_objects": 2, "split_counts": {s: len(ids) for s, ids in splits.items()},
              "near_duplicate_scope": "Frozen reviewed group; use find_near_duplicates.py for a new all-pairs scan"}
    save_json(root / "outputs/dataset_analysis/audit.json", result)
    print(json.dumps(result, indent=2))


def prepare(root=ROOT):
    records, images, splits = read_raw(root)
    # A fresh all-pairs scan is a prerequisite, not replaced by a known-group check.
    scan = load_json(root / "outputs/dataset_analysis/near_duplicate_scan.json")
    require(scan["images"] == len(records) and scan["total_pairs_checked"] == len(records)*(len(records)-1)//2,
            "Incomplete all-pairs audit")
    require(scan["image_sha256"] == {i: sha(p) for i, p in images.items()}, "Scan input drift")
    csv_path = root / "outputs/dataset_analysis/near_duplicate_candidates.csv"
    require(sha(csv_path) == scan["candidate_csv_sha256"], "Candidate audit drift")
    import csv
    pairs = {tuple(sorted((r["a"], r["b"]))) for r in csv.DictReader(csv_path.open(encoding="utf-8"))}
    review = read_yaml(root / "configs/near_duplicate_groups.yaml")
    require(review["scan_version"] == scan["scan_version"], "Scan version drift")
    reviewed = {tuple(sorted(g["members"])) for g in review["confirmed_groups"]}
    reviewed |= {tuple(sorted((g["a"], g["b"]))) for g in review.get("rejected_candidates", []) if g.get("reason")}
    require(pairs <= reviewed and len(pairs) == scan["candidates"], "Unreviewed acquisition candidates")
    target = root / "data/processed/bccd"
    require(not target.exists(), "Processed exists; refusing overwrite")
    target.mkdir(parents=True)
    for s, ids in splits.items():
        for kind in ("images", "labels"):
            (target / kind / s).mkdir(parents=True)
        for i in ids:
            shutil.copyfile(images[i], target / f"images/{s}/{i}.jpg")
            (target / f"labels/{s}/{i}.txt").write_bytes(label_text(records[i]))
    check_processed(root)


def check_processed(root=ROOT):
    records, images, splits = read_raw(root)
    hashes = {}
    max_error = 0.0
    for s, ids in splits.items():
        for kind, suffix in (("images", ".jpg"), ("labels", ".txt")):
            folder = root / f"data/processed/bccd/{kind}/{s}"
            actual = {p.stem for p in folder.glob("*" + suffix)}
            require(actual == set(ids), "Unexpected processed membership")
        for i in ids:
            image = root / f"data/processed/bccd/images/{s}/{i}.jpg"
            label = root / f"data/processed/bccd/labels/{s}/{i}.txt"
            require(sha(image) == sha(images[i]) and label.read_bytes() == label_text(records[i]), "Conversion drift")
            for row, (_, box) in zip(label.read_text().splitlines(), records[i]):
                _, xc, yc, w, h = map(float, row.split())
                restored = ((xc-w/2)*640, (yc-h/2)*480, (xc+w/2)*640, (yc+h/2)*480)
                max_error = max(max_error, max(abs(a-b) for a, b in zip(restored, box)))
            hashes[image.relative_to(root).as_posix()] = sha(image)
            hashes[label.relative_to(root).as_posix()] = sha(label)
    require(max_error < 1e-8, "Coordinate round-trip error")
    return {"status": "PASS", "artifact_sha256": hashes, "max_round_trip_error_px": max_error}

"""Render documentation tables from immutable published evidence, without ML imports."""
import argparse
import _bootstrap
from blood_cell_detection.data import ROOT, CLASSES, load_json, require, sha

METRICS = ("Precision", "Recall", "mAP50", "mAP50-95")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def load():
    for p, h in load_json(ROOT / "docs/evidence/manifest.json")["public_sha256"].items():
        require(sha(ROOT / p) == h, "Published evidence drift: " + p)
    return load_json(ROOT / "docs/evidence/frozen_results.json")


def blocks(r):
    e, t = r["experiments"], r["test"]
    final = table(list(METRICS), [[f"{t['overall'][k]*100:.2f}%" for k in METRICS]])
    design = table(["Experiment", "Input", "cls_pw", "Purpose", "Validation mAP50-95"],
        [[i, e[i]["config"]["imgsz"], e[i]["config"]["cls_pw"], purpose, f"{e[i]['overall']['mAP50-95']:.6f}"]
         for i, purpose in zip(e, ("Baseline", "Class weighting", "Resolution trade-off"))])
    validation = table(["Experiment", *METRICS], [[i, *[f"{e[i]['overall'][k]:.6f}" for k in METRICS]] for i in e])
    classes = table(["Class", "E1 validation AP50-95", "E2 validation AP50-95", "E3 validation AP50-95", "E2 test AP50-95"],
        [[c, *[f"{e[i]['per_class'][c]['map50_95']:.6f}" for i in e], f"{t['per_class'][c]['map50_95']:.6f}"] for c in CLASSES])
    efficiency = table(["Metric", "640 (E2)", "512 (E3)", "Change"], [[x["metric"]+" ("+x["unit"]+")",
        f"{float(x['E2']):.4f}", f"{float(x['E3']):.4f}", f"{float(x['percent_change']):+.2f}%"] for x in r["efficiency"]])
    errors = table(["Class", "TP", "FP", "FN"], [[x[k] for k in ("domain", "TP", "FP", "FN")] for x in r["errors"]])
    gap = table(["Metric", "Test minus E2 validation (percentage points)"],
                [[k, f"{100*(t['overall'][k]-e['E2']['overall'][k]):+.4f}"] for k in METRICS])
    return dict(FINAL=final, DESIGN=design, VALIDATION=validation, CLASSES=classes, EFFICIENCY=efficiency, ERRORS=errors, GAP=gap)


def render(verify=False):
    r = load()
    values = blocks(r)
    values["ACCURACY_DELTA"] = f"{100*(r['experiments']['E3']['overall']['mAP50-95']-r['experiments']['E2']['overall']['mAP50-95']):+.3f}"
    for name in ("README.md", "docs/results.md", "docs/experiments.md"):
        source = (ROOT / "docs/templates" / (name.replace("/", "_") + ".in")).read_text(encoding="utf-8")
        for key, value in values.items():
            source = source.replace("{{" + key + "}}", value)
        require("{{" not in source, "Unresolved documentation placeholder")
        target = ROOT / name
        if verify:
            require(target.read_text(encoding="utf-8") == source, "Rendered table/document drift: " + name)
        else:
            target.write_text(source, encoding="utf-8")
    print("Frozen evidence and rendered documentation: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    render(parser.parse_args().verify)

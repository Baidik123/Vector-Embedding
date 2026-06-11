#!/usr/bin/env python3
import os, sys

IMAGE_DIR = r"C:\Users\baidik.bora\Downloads\Testing Jambs"

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "all_image_to_image_similarity_estimates.csv")

CACHE_PATH = os.path.join(os.path.dirname(__file__), "calibration_cache.json")

ALGORITHMS = [
    "dhash",
    "dino_small",
    "dino_large",
    "dino_giant",
    "dino_base",
    "clip",
    "efficientnet",
    "convnext",
    "resnet",
    "hybrid",
]

DHASH_WEIGHT      = 0.4
DINO_WEIGHT       = 0.3
DINO_LARGE_WEIGHT = 0.3

HASH_SIZE = 16
MAX_BITS  = HASH_SIZE * HASH_SIZE

import csv, json, time
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import imagehash

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoProcessor
    from torchvision import models as tv_models
    import torchvision.transforms as tv_transforms
    from sklearn.neighbors import NearestNeighbors
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] torch/transformers not installed — only dHash will run.")

CATEGORY_MAP = {
    "100% similar":     "Exact Duplicate",
    "Very similar":     "Likely Duplicate",
    "Minor similarity": "Similar – Same Family",
    "Just related":     "Similar – Related",
    "Different":        "Different",
}

CATEGORY_RANK = {
    "Exact Duplicate":       5,
    "Likely Duplicate":      4,
    "Similar – Same Family": 3,
    "Similar – Related":     2,
    "Different":             1,
}

def preprocess_image(image: Image.Image) -> Image.Image:
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)
    h, w = inverted.shape
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[int(0.85 * h):, :] = 0
    mask[:, int(0.80 * w):] = 0
    masked = cv2.bitwise_and(inverted, inverted, mask=mask)
    blurred = cv2.GaussianBlur(masked, (3, 3), 0)
    return Image.fromarray(blurred)

def dhash_score(img1: Image.Image, img2: Image.Image) -> dict:
    h1 = imagehash.dhash(preprocess_image(img1), hash_size=HASH_SIZE)
    h2 = imagehash.dhash(preprocess_image(img2), hash_size=HASH_SIZE)
    dist = h1 - h2
    return {"score": round(1.0 - dist / MAX_BITS, 6), "hamming": dist}

def knn_similarity(emb1, emb2) -> float:
    e1 = F.normalize(emb1, p=2, dim=1).numpy()
    e2 = F.normalize(emb2, p=2, dim=1).numpy()
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(e1)
    dist, _ = nn.kneighbors(e2)
    return float(1.0 / (1.0 + dist[0][0]))

def knn_to_cosine_scale(knn_sim: float) -> float:
    dist = (1.0 / knn_sim) - 1.0
    return float(1.0 - (dist ** 2) / 2.0)

_models = {}

def _load_hf_model(name: str, hf_path: str):
    if name not in _models:
        print(f"  [load] {hf_path} ...", end=" ", flush=True)
        t = time.time()
        proc  = AutoProcessor.from_pretrained(hf_path)
        model = AutoModel.from_pretrained(hf_path)
        model.eval()
        _models[name] = (proc, model)
        print(f"done ({time.time()-t:.1f}s)")
    return _models[name]

def _load_cnn_model(name: str, factory, weights):
    if name not in _models:
        print(f"  [load] {name} ...", end=" ", flush=True)
        t = time.time()
        _models[name] = factory(weights=weights)
        _models[name].eval()
        print(f"done ({time.time()-t:.1f}s)")
    return _models[name]

_cnn_transform = None
def get_cnn_transform():
    global _cnn_transform
    if _cnn_transform is None:
        _cnn_transform = tv_transforms.Compose([
            tv_transforms.Resize(256),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])
    return _cnn_transform

def _cnn_preprocess(image: Image.Image):
    preprocessed = preprocess_image(image).convert("RGB")
    return get_cnn_transform()(preprocessed).unsqueeze(0)

def run_dino_small(img1, img2) -> float:
    proc, model = _load_hf_model("dino_small", "facebook/dinov2-small")
    def embed(img):
        inp = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            return model(**inp).last_hidden_state.mean(dim=1)
    knn = knn_similarity(embed(img1), embed(img2))
    return knn_to_cosine_scale(knn)

def run_dino_large(img1, img2) -> float:
    proc, model = _load_hf_model("dino_large", "facebook/dinov2-large")
    def embed(img):
        inp = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            return model(**inp).last_hidden_state.mean(dim=1)
    knn = knn_similarity(embed(img1), embed(img2))
    return knn_to_cosine_scale(knn)

def run_dino_giant(img1, img2) -> float:
    proc, model = _load_hf_model("dino_giant", "facebook/dinov2-giant")
    def embed(img):
        inp = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            return model(**inp).last_hidden_state[:, 0, :]
    knn = knn_similarity(embed(img1), embed(img2))
    return knn_to_cosine_scale(knn)

def run_dino_base(img1, img2) -> float:
    proc, model = _load_hf_model("dino_base", "facebook/dinov2-base")
    def embed(img):
        inp = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            return model(**inp).last_hidden_state.mean(dim=1)
    knn = knn_similarity(embed(img1), embed(img2))
    return knn_to_cosine_scale(knn)

def run_clip(img1, img2) -> float:
    proc, model = _load_hf_model("clip", "openai/clip-vit-base-patch32")
    def embed(img):
        inp = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            return model.vision_model(pixel_values=inp["pixel_values"]).last_hidden_state.mean(dim=1)
    knn = knn_similarity(embed(img1), embed(img2))
    return knn_to_cosine_scale(knn)

def run_efficientnet(img1, img2) -> float:
    import copy
    base = _load_cnn_model("efficientnet", tv_models.efficientnet_b4,
                           tv_models.EfficientNet_B4_Weights.DEFAULT)
    key = "efficientnet_infer"
    if key not in _models:
        m = copy.deepcopy(base)
        m.classifier = torch.nn.Identity()
        m.eval()
        _models[key] = m
    m = _models[key]
    def embed(img):
        with torch.no_grad():
            return m(_cnn_preprocess(img))
    knn = knn_similarity(embed(img1), embed(img2))
    return knn_to_cosine_scale(knn)

def run_convnext(img1, img2) -> float:
    import copy
    base = _load_cnn_model("convnext", tv_models.convnext_tiny,
                           tv_models.ConvNeXt_Tiny_Weights.DEFAULT)
    key = "convnext_infer"
    if key not in _models:
        m = copy.deepcopy(base)
        m.classifier = torch.nn.Sequential()
        m.eval()
        _models[key] = m
    m = _models[key]
    def embed(img):
        with torch.no_grad():
            out = m(_cnn_preprocess(img))
            return out.view(out.size(0), -1)
    knn = knn_similarity(embed(img1), embed(img2))
    return knn_to_cosine_scale(knn)

def run_resnet(img1, img2) -> float:
    import copy
    base = _load_cnn_model("resnet", tv_models.resnet50,
                           tv_models.ResNet50_Weights.DEFAULT)
    key = "resnet_infer"
    if key not in _models:
        m = copy.deepcopy(base)
        m.fc = torch.nn.Identity()
        m.eval()
        _models[key] = m
    m = _models[key]
    def embed(img):
        with torch.no_grad():
            return m(_cnn_preprocess(img))
    knn = knn_similarity(embed(img1), embed(img2))
    return knn_to_cosine_scale(knn)

def load_images(image_dir: str) -> dict:
    images = {}
    img_dir = Path(image_dir)
    for f in img_dir.iterdir():
        if f.suffix.lower() == ".png":
            stem = f.stem
            try:
                images[stem] = Image.open(f).convert("RGB")
            except Exception as e:
                print(f"[WARN] Could not load {f}: {e}")
    return images

def load_pairs(csv_path: str) -> list:
    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i1 = str(int(row["image_1"]))
            i2 = str(int(row["image_2"]))
            if i1 == i2:
                continue
            cat_raw = row["category"].strip()
            label   = CATEGORY_MAP.get(cat_raw, cat_raw)
            mid_pct = (float(row["estimated_similarity_min_pct"]) +
                       float(row["estimated_similarity_max_pct"])) / 2.0
            pairs.append({
                "img1":     i1,
                "img2":     i2,
                "gt_label": label,
                "gt_rank":  CATEGORY_RANK.get(label, 0),
                "gt_score": round(mid_pct / 100.0, 4),
            })
    return pairs

def compute_all_scores(pairs: list, images: dict, algorithms: list) -> list:
    results = [dict(p) for p in pairs]

    runners = {
        "dino_small":   run_dino_small,
        "dino_large":   run_dino_large,
        "dino_giant":   run_dino_giant,
        "dino_base":    run_dino_base,
        "clip":         run_clip,
        "efficientnet": run_efficientnet,
        "convnext":     run_convnext,
        "resnet":       run_resnet,
    }

    for algo in algorithms:
        print(f"\n{'─'*60}")
        print(f"  Algorithm: {algo}")
        print(f"{'─'*60}")

        is_dhash  = (algo == "dhash")
        is_hybrid = (algo == "hybrid")

        if not is_dhash and not is_hybrid:
            if not HAS_TORCH:
                print(f"  [SKIP] torch not available")
                for r in results:
                    r[algo] = None
                continue
            algo_fn = runners.get(algo)
            if algo_fn is None:
                print(f"  [SKIP] unknown algorithm: {algo}")
                continue

        for i, row in enumerate(results):
            i1, i2 = row["img1"], row["img2"]
            img1, img2 = images.get(i1), images.get(i2)

            if img1 is None or img2 is None:
                row[algo] = None
                continue

            try:
                if is_dhash:
                    out = dhash_score(img1, img2)
                    row["dhash"]   = out["score"]
                    row["hamming"] = out["hamming"]
                elif is_hybrid:
                    d = row.get("dhash")
                    s = row.get("dino_small")
                    l = row.get("dino_large")
                    if d is None or s is None or l is None:
                        row["hybrid"] = None
                    else:
                        row["hybrid"] = round(
                            d * DHASH_WEIGHT + s * DINO_WEIGHT + l * DINO_LARGE_WEIGHT, 6
                        )
                else:
                    row[algo] = round(algo_fn(img1, img2), 6)
            except Exception as e:
                print(f"  [ERROR] pair ({i1},{i2}): {e}")
                row[algo] = None

            if (i + 1) % 50 == 0 or (i + 1) == len(results):
                print(f"  {i+1}/{len(results)} pairs done", flush=True)

    return results

CATEGORY_WEIGHTS = {
    "Exact Duplicate":       5,
    "Likely Duplicate":      4,
    "Similar – Same Family": 3,
    "Similar – Related":     2,
    "Different":             1,
}

def score_to_band(score: float, thresholds: list) -> str:
    bands = ["Exact Duplicate", "Likely Duplicate", "Similar – Same Family",
             "Similar – Related", "Different"]
    for i, t in enumerate(thresholds):
        if score >= t:
            return bands[i]
    return "Different"


def find_optimal_thresholds(scores: list, labels: list, n_candidates: int = 200) -> dict:
    arr = np.array(scores)
    pcts  = np.linspace(1, 99, n_candidates)
    cands = sorted(set(np.percentile(arr, pcts).tolist()), reverse=True)

    weights  = np.array([CATEGORY_WEIGHTS.get(l, 1) for l in labels])
    best_acc = -1.0
    best_t   = [0.973, 0.902, 0.781, 0.699]

    for t1 in cands:
        for t2 in [c for c in cands if c < t1]:
            for t3 in [c for c in cands if c < t2]:
                for t4 in [c for c in cands if c < t3]:
                    predicted = [score_to_band(s, [t1, t2, t3, t4]) for s in scores]
                    correct   = np.array([int(p == l) for p, l in zip(predicted, labels)])
                    acc       = float(np.average(correct, weights=weights))
                    if acc > best_acc:
                        best_acc = acc
                        best_t   = [t1, t2, t3, t4]

    predicted = [score_to_band(s, best_t) for s in scores]
    per_class = {}
    for label in CATEGORY_WEIGHTS:
        idx = [i for i, l in enumerate(labels) if l == label]
        if not idx:
            continue
        correct = sum(1 for i in idx if predicted[i] == label)
        per_class[label] = {"total": len(idx), "correct": correct,
                            "accuracy": round(correct / len(idx), 3)}

    return {
        "thresholds":        [round(t, 4) for t in best_t],
        "weighted_accuracy": round(best_acc, 4),
        "per_class":         per_class,
    }


def dhash_find_optimal_thresholds(hammings: list, labels: list) -> dict:
    cands   = sorted(set(hammings))
    weights = np.array([CATEGORY_WEIGHTS.get(l, 1) for l in labels])

    best_acc = -1.0
    best_h   = [7, 25, 56, 77]

    for h1 in cands:
        for h2 in [c for c in cands if c > h1]:
            for h3 in [c for c in cands if c > h2]:
                for h4 in [c for c in cands if c > h3]:
                    predicted = []
                    for h in hammings:
                        if   h <= h1: predicted.append("Exact Duplicate")
                        elif h <= h2: predicted.append("Likely Duplicate")
                        elif h <= h3: predicted.append("Similar – Same Family")
                        elif h <= h4: predicted.append("Similar – Related")
                        else:         predicted.append("Different")
                    correct = np.array([int(p == l) for p, l in zip(predicted, labels)])
                    acc     = float(np.average(correct, weights=weights))
                    if acc > best_acc:
                        best_acc = acc
                        best_h   = [h1, h2, h3, h4]

    score_equivalents = [round(1.0 - h / MAX_BITS, 4) for h in best_h]

    predicted = []
    for h in hammings:
        if   h <= best_h[0]: predicted.append("Exact Duplicate")
        elif h <= best_h[1]: predicted.append("Likely Duplicate")
        elif h <= best_h[2]: predicted.append("Similar – Same Family")
        elif h <= best_h[3]: predicted.append("Similar – Related")
        else:                predicted.append("Different")

    per_class = {}
    for label in CATEGORY_WEIGHTS:
        idx = [i for i, l in enumerate(labels) if l == label]
        if not idx:
            continue
        correct = sum(1 for i in idx if predicted[i] == label)
        per_class[label] = {"total": len(idx), "correct": correct,
                            "accuracy": round(correct / len(idx), 3)}

    return {
        "hamming_thresholds": best_h,
        "score_thresholds":   score_equivalents,
        "weighted_accuracy":  round(best_acc, 4),
        "per_class":          per_class,
    }

BAND_DESCRIPTIONS = {
    "Exact Duplicate":       "These images are virtually identical.",
    "Likely Duplicate":      "Very similar with only minor differences.",
    "Similar – Same Family": "Significant structural similarity — same design family.",
    "Similar – Related":     "Moderately similar — related drawing type.",
    "Different":             "Substantially different drawings.",
}

ALGO_CONST_MAP = {
    "dino_small":   "DINO_BANDS",
    "dino_large":   "DINO_LARGE_BANDS",
    "dino_giant":   "DINO_GIANT_BANDS",
    "dino_base":    "DINO_BASE_BANDS",
    "clip":         "CLIP_BANDS",
    "efficientnet": "EFFICIENTNET_BANDS",
    "convnext":     "CONVNEXT_BANDS",
    "resnet":       "RESNET_BANDS",
    "hybrid":       "HYBRID_BANDS",
}

def print_report(algo_results: dict):
    print("\n" + "=" * 70)
    print("  THRESHOLD CALIBRATION REPORT")
    print("=" * 70)

    band_labels = ["Exact Duplicate", "Likely Duplicate",
                   "Similar – Same Family", "Similar – Related"]

    for algo, res in algo_results.items():
        if res is None:
            continue
        print(f"\n{'─'*70}")
        print(f"  {algo.upper()}")
        print(f"{'─'*70}")
        print(f"  Weighted accuracy: {res['weighted_accuracy']:.1%}")
        print(f"\n  Per-class accuracy:")
        for label, info in res.get("per_class", {}).items():
            bar = "█" * int(info["accuracy"] * 20)
            print(f"    {label:<28} {info['accuracy']:.1%}  {bar}  ({info['correct']}/{info['total']})")

        print(f"\n  ─── Paste into Backend/main.py ──────────────────────────────────")
        if algo == "dhash":
            ht = res["hamming_thresholds"]
            st = res["score_thresholds"]
            print(f"  DHASH_BANDS = [")
            for i, label in enumerate(band_labels):
                desc = BAND_DESCRIPTIONS[label]
                print(f'      ({ht[i]},   "{label}",       "{desc}"),')
            print(f'      (256, "Different",              "Substantially different drawings."),')
            print(f"  ]")
            print(f"\n  # Score equivalents: {st}")
        else:
            thresholds = res.get("thresholds", [])
            const_name = ALGO_CONST_MAP.get(algo, algo.upper() + "_BANDS")
            print(f"  {const_name} = [")
            for i, label in enumerate(band_labels):
                desc = BAND_DESCRIPTIONS[label]
                t = thresholds[i] if i < len(thresholds) else "???"
                print(f'      ({t}, "{label}",       "{desc}"),')
            print(f'      (0.0, "Different",              "Substantially different drawings."),')
            print(f"  ]")

    print("\n" + "=" * 70)
    print("  Copy the BANDS above into Backend/main.py to apply calibrated thresholds.")
    print("=" * 70)


def save_report_json(algo_results: dict, output_path: str):
    with open(output_path, "w") as f:
        json.dump(algo_results, f, indent=2)
    print(f"\n  Full results saved to: {output_path}")

def main():
    print("=" * 70)
    print("  IMAGE SIMILARITY THRESHOLD CALIBRATION")
    print("=" * 70)

    print(f"\n[1/4] Loading labeled pairs from: {CSV_PATH}")
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        alt = Path(__file__).parent / "all_image_to_image_similarity_estimates.csv"
        if alt.exists():
            pairs = load_pairs(str(alt))
        else:
            sys.exit(f"ERROR: CSV not found at {CSV_PATH}. Edit CSV_PATH in CONFIG.")
    else:
        pairs = load_pairs(str(csv_path))
    print(f"  {len(pairs)} labeled pairs loaded.")

    print(f"\n[2/4] Loading images from: {IMAGE_DIR}")
    images = load_images(IMAGE_DIR)
    if not images:
        sys.exit(f"ERROR: No PNG files found in {IMAGE_DIR}. Edit IMAGE_DIR in CONFIG.")
    print(f"  {len(images)} images loaded: {sorted(images.keys())}")

    missing = set()
    for p in pairs:
        for k in (p["img1"], p["img2"]):
            if k not in images:
                missing.add(k)
    if missing:
        print(f"  [WARN] {len(missing)} image IDs in CSV not found in folder: {sorted(missing)}")

    valid_pairs = [p for p in pairs if p["img1"] in images and p["img2"] in images]
    print(f"  {len(valid_pairs)} / {len(pairs)} pairs can be evaluated.")

    cache_path = Path(CACHE_PATH)
    if cache_path.exists():
        print(f"\n[3/4] Loading cached scores from: {cache_path}")
        with open(cache_path) as f:
            results = json.load(f)
        print(f"  Cache loaded ({len(results)} rows).")
    else:
        print(f"\n[3/4] Computing similarity scores ...")
        results = compute_all_scores(valid_pairs, images, ALGORITHMS)
        with open(cache_path, "w") as f:
            json.dump(results, f)
        print(f"\n  Scores cached to: {cache_path}")

    print(f"\n[4/4] Optimising thresholds ...")
    algo_results = {}

    for algo in ALGORITHMS:
        rows_with_score = [r for r in results if r.get(algo) is not None]
        if not rows_with_score:
            print(f"  [SKIP] {algo} — no scores available")
            continue

        scores = [r[algo] for r in rows_with_score]
        labels = [r["gt_label"] for r in rows_with_score]

        print(f"  Optimising {algo} over {len(scores)} pairs ...", end=" ", flush=True)
        t0 = time.time()

        if algo == "dhash":
            hammings = [r.get("hamming", int((1 - r["dhash"]) * MAX_BITS))
                        for r in rows_with_score]
            res = dhash_find_optimal_thresholds(hammings, labels)
        else:
            res = find_optimal_thresholds(scores, labels, n_candidates=100)

        algo_results[algo] = res
        print(f"done ({time.time()-t0:.1f}s)  weighted_acc={res['weighted_accuracy']:.1%}")

    print_report(algo_results)

    report_path = Path(__file__).parent / "calibration_report.json"
    save_report_json(algo_results, str(report_path))


if __name__ == "__main__":
    main()

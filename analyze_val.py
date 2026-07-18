from pathlib import Path
import csv, shutil
import numpy as np
import cv2
from ultralytics import YOLO

WEIGHTS = r"C:\harness_dataset_v2\runs\detect\runs_harness\yolo11s_v1\weights\best.pt"
VAL_IMAGES = Path("images/val")
VAL_LABELS = Path("labels/val")
CONF = 0.25
IOU_MATCH = 0.5
OUT = Path(r"C:\harness_dataset_v2\runs\detect\runs_harness\analysis")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(label_path, w, h):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            t = line.split()
            if len(t) != 5:
                continue
            _, xc, yc, bw, bh = map(float, t)
            x1 = (xc - bw/2) * w; y1 = (yc - bh/2) * h
            x2 = (xc + bw/2) * w; y2 = (yc + bh/2) * h
            boxes.append([x1, y1, x2, y2])
    return boxes


def match(preds, gts):
    used = set(); tp = 0; ious = []
    for p in preds:
        best, bj = 0, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(p, g)
            if v > best:
                best, bj = v, j
        if best >= IOU_MATCH:
            tp += 1; used.add(bj); ious.append(best)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn, (float(np.mean(ious)) if ious else 0.0)

def main():
    model = YOLO(WEIGHTS)
    OUT.mkdir(parents=True, exist_ok=True)

    model.predict(source=str(VAL_IMAGES), conf=CONF, save=True,
                  project=r"C:\harness_dataset_v2\runs\detect\runs_harness",
                  name="predict_all", exist_ok=True)

    rows = []
    imgs = sorted(VAL_IMAGES.glob("*.jpg"))
    for ip in imgs:
        img = cv2.imread(str(ip)); h, w = img.shape[:2]
        gts = load_gt(VAL_LABELS / (ip.stem + ".txt"), w, h)
        res = model.predict(source=str(ip), conf=CONF, verbose=False)[0]
        preds = [b.tolist() for b in res.boxes.xyxy.cpu().numpy()]
        confs = res.boxes.conf.cpu().numpy().tolist()
        tp, fp, fn, miou = match(preds, gts)
        rows.append({"name": ip.name, "gt": len(gts), "pred": len(preds),
                     "tp": tp, "fp": fp, "fn": fn, "mean_iou": round(miou, 3),
                     "errors": fp + fn,
                     "img": img, "preds": preds, "confs": confs, "gts": gts})

    with open(OUT / "per_image_stats.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "gt", "pred", "tp", "fp", "fn", "mean_iou", "errors"])
        for r in rows:
            wr.writerow([r["name"], r["gt"], r["pred"], r["tp"], r["fp"],
                         r["fn"], r["mean_iou"], r["errors"]])

    def draw(r, path):
        img = r["img"].copy()
        for g in r["gts"]:
            cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 200, 0), 2)
        for p, c in zip(r["preds"], r["confs"]):
            cv2.rectangle(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 255), 2)
            cv2.putText(img, f"{c:.2f}", (int(p[0]), int(p[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(path), img)

    good_dir = OUT / "examples_good"; bad_dir = OUT / "examples_bad"
    for d in (good_dir, bad_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()

    good = sorted([r for r in rows if r["gt"] > 0 and r["errors"] == 0],
                  key=lambda r: -r["mean_iou"])[:10]
    bad = sorted([r for r in rows if r["errors"] > 0],
                 key=lambda r: (-r["errors"], r["mean_iou"]))[:10]

    for i, r in enumerate(good):
        draw(r, good_dir / f"good_{i:02d}_iou{r['mean_iou']:.2f}_{r['name']}")
    for i, r in enumerate(bad):
        draw(r, bad_dir / f"bad_{i:02d}_fp{r['fp']}_fn{r['fn']}_{r['name']}")

    tot_tp = sum(r["tp"] for r in rows); tot_fp = sum(r["fp"] for r in rows); tot_fn = sum(r["fn"] for r in rows)
    print(f"Кадров: {len(rows)} | TP={tot_tp} FP={tot_fp} FN={tot_fn}")
    print(f"Удачных отобрано: {len(good)} | Неудачных: {len(bad)}")
    print(f"Зелёный = разметка (GT), Красный = предсказание модели (conf).")
    print(f"Всё в: {OUT.resolve()}")


if __name__ == "__main__":
    main()
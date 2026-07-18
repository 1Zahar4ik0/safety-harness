from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11s.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        seed=42,
        patience=25,
        cache=True,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,

        project="runs_harness",
        name="yolo11s_v1",
    )

    metrics = model.val()
    print(metrics.box.map, metrics.box.map50, metrics.box.mp, metrics.box.mr)
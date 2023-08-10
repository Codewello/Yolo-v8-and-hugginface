from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")

model.predict('./data/images/val/building workers_106.jpeg', save=True, show=True, conf=0.7, save_txt=True)
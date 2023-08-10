from ultralytics import YOLO

model = YOLO('yolov8n.yaml')

results = model.train(data='data_config.yaml', epochs=100 , imgsz=640 , workers=1, batch=3 )
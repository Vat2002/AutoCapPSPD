from ultralytics import YOLO


class InitializeModel:
    def train_model(model_path):
        yolo_model = YOLO(model_path)
        results = yolo_model.train(data='coco128.yaml', epochs=2, imgsz=640, workers=8, lr0=0.01, lrf=0.01)
        return results

    def load_model(model_path):
        yolo_model = YOLO(model_path)
        print("Model Loaded:" + model_path)
        return yolo_model


InitializeModel.load_model('models/weights/best.pt')

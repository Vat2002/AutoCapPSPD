from detection import ImageDetection
from train import InitializeModel


class Main:
    def callFunctions(model_path, image_path):
        InitializeModel.load_model(model_path)
        ImageDetection.detect_objects_onimage(model_path,image_path)


Main.callFunctions('models/weights/best.pt','images/4.jpg')

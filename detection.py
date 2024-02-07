import cv2
from train import InitializeModel


class ImageDetection:
    def detect_objects_onimage(model_path, image_path):
        # Load the image
        image = cv2.imread(image_path)
        model = InitializeModel.load_model(model_path)
        results = model.predict(image)
        result = results[0]
        output = []
        for box in result.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            class_id = box.cls[0].item()
            prob = round(box.conf[0].item(), 2)
            output.append([x1, y1, x2, y2, result.names[class_id], prob])

            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put the class name and probability on the image
            label = f"{result.names[class_id]}: {prob * 100}%"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image
        cv2.imshow('Detected Objects', image)
        cv2.waitKey(0)  # Wait for a key press to close
        cv2.destroyAllWindows()
        return output


detection = ImageDetection.detect_objects_onimage('models/yolov8n.pt', 'test_images/4.jpg')

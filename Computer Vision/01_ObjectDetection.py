import torch
import cv2
import numpy as np
from time import time
from ultralytics import YOLO

from supervision import ColorPalette
from supervision import BoxAnnotator, Detections

class ObjectDetection():

    def __init__(self, capture_index):
        
        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = BoxAnnotator(color=ColorPalette.DEFAULT,thickness=4, text_thickness=3, text_scale=1.5)


    def load_model(self):

        model = YOLO("yolov8m.pt")
        model.fuse()

        return model
    
    def predict(self, frame):

        results = self.model(frame)

        return results
    
    def plot_bboxes(self, results, frame):

        xyxys = []
        confidences = []
        class_ids = []

        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if class_id == 0:

                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int)),

        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )

        self.labels = [
            f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
        
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)

        return frame
    

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:

            start_time = time()

            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)

            # cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5)

            cv2.imshow("YOLOv8 Detection", frame)

            if cv2.waitKey(5) & 0xFF == 27:

                break

        cap.release()
        cv2.destroyAllWindows()



detector = ObjectDetection(capture_index=0)
detector.__call__()







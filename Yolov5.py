import torch
import yolov5
import os


class YoloDetectorV5:
    def __init__(self, conf_threshold=0.25, expected_objs=['person']):

        self._model_path = "best.pt"

        self._model = yolov5.load(self._model_path)

        # set model parameters
        self._model.conf = conf_threshold  # NMS confidence threshold
        self._model.iou = 0.45  # NMS IoU threshold
        self._model.agnostic = False  # NMS class-agnostic
        self._model.multi_label = False  # NMS multiple labels per box
        self._model.max_det = 1000  # maximum number of detections per image

        self._objects_to_be_detected = expected_objs

        self._classes = self._model.names
        self._number_of_classes = len(self._classes)

    def process_image(self, image):
        # results = self._model(image)

        # inference with larger input size
        # results = self._model(image, size=1280)

        # inference with test time augmentation
        results = self._model(image, augment=False)

        # parse results
        predictions = results.pred[0]

        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        detected_objects = []
        for i in range(len(boxes)):
            class_idx = int(categories[i])
            label = self._classes[class_idx]
            if label in self._objects_to_be_detected:
                box = boxes[i].to(torch.int32).tolist()
                detected_objects.append({"name": label, "box": box})

        return detected_objects
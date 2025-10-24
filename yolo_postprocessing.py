import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum

class detection_classes_type(Enum):
    TRUNK = 0
    END_OF_ROW = 1
    POLE = 2

    def __str__(self):
        return self.name

    def __int__(self):
        return self.value

    @staticmethod
    def names():
        return [member.name for member in detection_classes_type]


class Detection:
    bbox: np.ndarray
    score: float
    class_id: int
    class_name: str

    def __init__(self, _class, _score, _bbox):
        self.class_detected = _class
        self.score = _score
        self.bbox = _bbox
        self.class_name = str(_class)
        self.class_id= int(_class)

    def as_dict(self) -> dict:
        return {
            'bbox': self.bbox.tolist(),
            'score': float(self.score),
            'class_id': int(self.class_detected),
            'class_name': str(self.class_detected)
        }

    def __repr__(self):
        return f"Detection(class={self.class_name}, score={self.score:.3f}, bbox={self.bbox})"


class YOLOPostprocessor:
    def __init__(
            self,
            score_threshold: float = 0.5,
            nms_threshold: float = 0.45,
            box_format: str = 'xywh'
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.box_format = box_format

    def filter_by_score(
            self,
            boxes: np.ndarray,
            scores: np.ndarray,
            classes: np.ndarray,
            threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if threshold is None:
            threshold = self.score_threshold

        mask = scores >= threshold
        return boxes[mask], scores[mask], classes[mask]

    def filter_by_class(
            self,
            boxes: np.ndarray,
            scores: np.ndarray,
            classes: np.ndarray,
            target_classes: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.isin(classes, target_classes)
        return boxes[mask], scores[mask], classes[mask]

    def organize_by_class(
            self,
            boxes: np.ndarray,
            scores: np.ndarray,
            classes: np.ndarray
    ) -> Dict[str, List[Detection]]:

        detections_dict = {}
        detections_dict['all'] = []

        for box, score, class_id in zip(boxes, scores, classes):
            _class = detection_classes_type(int(class_id))
            class_name = str(_class)

            detection = Detection(
                _bbox=box,
                _score=float(score),
                _class=_class)

            # Add to specific class list
            if not class_name in detections_dict:
                detections_dict[class_name] = []
            detections_dict[class_name].append(detection)
            detections_dict['all'].append(detection)

        for key in detections_dict:
            detections_dict[key].sort(key=lambda d: d.score, reverse=True)

        return detections_dict

    def apply_nms_per_class(
            self,
            boxes: np.ndarray,
            scores: np.ndarray,
            classes: np.ndarray,
            iou_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if iou_threshold is None:
            iou_threshold = self.nms_threshold

        if len(boxes) == 0:
            return boxes, scores, classes

        kept_indices = []
        unique_classes = np.unique(classes)

        for class_id in unique_classes:
            class_mask = classes == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = np.where(class_mask)[0]

            nms_indices = self._nms(class_boxes, class_scores, iou_threshold)
            kept_indices.extend(class_indices[nms_indices])

        kept_indices = np.array(kept_indices)
        return boxes[kept_indices], scores[kept_indices], classes[kept_indices]

    def _nms(
            self,
            boxes: np.ndarray,
            scores: np.ndarray,
            iou_threshold: float
    ) -> np.ndarray:
        if len(boxes) == 0:
            return np.array([], dtype=int)

        if self.box_format != 'xyxy':
            boxes = self._convert_to_xyxy(boxes, self.box_format)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def _convert_to_xyxy(self, boxes: np.ndarray, from_format: str) -> np.ndarray:
        boxes = boxes.copy()

        if from_format == 'xywh':
            # [x, y, w, h] -> [x1, y1, x2, y2]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        elif from_format == 'cxcywh':
            # [cx, cy, w, h] -> [x1, y1, x2, y2]
            boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return boxes

    def process(
            self,
            boxes: np.ndarray,
            scores: np.ndarray,
            classes: np.ndarray,
            apply_nms: bool = True,
            target_classes: Optional[List[int]] = None
    ) -> Dict[str, List[Detection]]:

        if len(boxes) == 0:
            return {name: [] for name in detection_classes_type.names() + ['all']}

        boxes, scores, classes = self.filter_by_score(boxes, scores, classes)

        if len(boxes) == 0:
            return {name: [] for name in detection_classes_type.names() + ['all']}

        # 2. Filter by class if specified
        if target_classes is not None:
            boxes, scores, classes = self.filter_by_class(
                boxes, scores, classes, target_classes
            )

        if len(boxes) == 0:
            return {name: [] for name in detection_classes_type.names() + ['all']}

        if apply_nms:
            boxes, scores, classes = self.apply_nms_per_class(
                boxes, scores, classes
            )

        detections_dict = self.organize_by_class(boxes, scores, classes)

        return detections_dict



# ========== Example Usage ==========

if __name__ == "__main__":

    postprocessor = YOLOPostprocessor(
        score_threshold=0.5,
        nms_threshold=0.45,
        box_format='xywh'
    )

    print("YOLO Postprocessor - Vineyard Detection Example")

    np.random.seed(42)
    n_detections = 15

    boxes = np.random.rand(n_detections, 4) * 500  # Random boxes
    boxes[:, 2:] = np.random.rand(n_detections, 2) * 100 + 50  # w, h

    scores = np.random.rand(n_detections) * 0.5 + 0.3  # scores 0.3-0.8
    classes = np.random.randint(0, 3, n_detections)  # classes 0, 1, 2

    print(f"Raw detections: {n_detections}")
    print(f"Score range: {scores.min():.2f} - {scores.max():.2f}")
    print(f"Classes distribution: {np.bincount(classes.astype(int))}")

    print("Processing with default threshold (0.5)...")

    detections_dict = postprocessor.process(
        boxes=boxes,
        scores=scores,
        classes=classes,
        apply_nms=True
    )

    print("Detections by Class:")
    for class_name in ['trunk', 'end_of_row', 'pole']:
        detections = detections_dict[class_name]
        print(f"{class_name.upper()}:")
        print(f"  Count: {len(detections)}")
        if len(detections) > 0:
            print(f"  Top 3 scores: {[f'{d.score:.3f}' for d in detections[:3]]}")
            print(f"  Sample bbox: {detections[0].bbox}")

    print("🎯 Filtering only TRUNKS (class 0):")

    trunks_only = postprocessor.process(
        boxes=boxes,
        scores=scores,
        classes=classes,
        apply_nms=True,
        target_classes=[0]  # Only trunk
    )

    print(f"Trunks detected: {len(trunks_only['trunk'])}")
    print(f"Other classes: {sum(len(trunks_only[c]) for c in ['end_of_row', 'pole'])}")

    print("Using higher threshold (0.7):")

    postprocessor.score_threshold = 0.7
    high_conf_detections = postprocessor.process(
        boxes=boxes,
        scores=scores,
        classes=classes,
        apply_nms=True
    )

    print("Export format example:")

    if len(detections_dict['trunk']) > 0:
        trunk_dict = detections_dict['trunk'][0].as_dict()
        print(f"\nFirst trunk detection as dict:")
        import json

        print(json.dumps(trunk_dict, indent=2))
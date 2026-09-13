"""Person detection and tracking for the keyframe stage.

Tracking rather than plain detection matters here: in a fight bodies occlude
each other constantly, and a detector run frame by frame loses and regains
people, which makes the person count jump around. Persisting track IDs keeps a
partially hidden person in the count until they actually leave the scene.

The boxes are reused later as a weak spatial reference for the Grad-CAM
pointing-game metric.
"""

import numpy as np

PERSON_CLASS = 0  # COCO


class PersonTracker:
    def __init__(self, weights="yolo12n.pt", conf=0.35, imgsz=640,
                 tracker="bytetrack.yaml"):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker

    def track_frame(self, frame):
        """-> (boxes xyxy, track ids, confidences) for people only."""
        res = self.model.track(frame, persist=True, classes=[PERSON_CLASS],
                               conf=self.conf, imgsz=self.imgsz,
                               tracker=self.tracker, verbose=False)
        r = res[0]
        if r.boxes is None or r.boxes.id is None:
            if r.boxes is None or len(r.boxes) == 0:
                return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)
            # detections without IDs still count as people present
            return (r.boxes.xyxy.cpu().numpy(),
                    np.full(len(r.boxes), -1, int),
                    r.boxes.conf.cpu().numpy())
        return (r.boxes.xyxy.cpu().numpy(),
                r.boxes.id.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy())

    def person_counts(self, video_path, stride=1):
        """Per-frame person count and boxes for a whole video."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        counts, boxes_per_frame, i = [], [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                boxes, ids, _ = self.track_frame(frame)
                counts.append(len(boxes))
                boxes_per_frame.append(boxes)
            i += 1
        cap.release()
        return np.asarray(counts), boxes_per_frame


def candidate_frames(person_counts, motion_scores, min_people=2,
                     motion_threshold=None):
    """A frame is a candidate when enough people are present *and* something
    is moving. Both conditions are necessary; either alone fires constantly in
    a busy scene or on an empty panning shot."""
    from .frame_difference import estimate_threshold

    pc = np.asarray(person_counts)
    ms = np.asarray(motion_scores, float)
    n = min(len(pc), len(ms))
    pc, ms = pc[:n], ms[:n]

    if motion_threshold is None:
        motion_threshold = estimate_threshold(ms)

    mask = (pc >= min_people) & (ms >= motion_threshold)
    return np.where(mask)[0], motion_threshold

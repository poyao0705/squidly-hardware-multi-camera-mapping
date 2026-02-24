import os
import mediapipe as mp


LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "detection/face_landmarker.task")

class Detector:
    def __init__(self):
        self.timestamp_ts = 0
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
            )
        )

    def detect(self, frame):
        # Convert the BGR image to RGB
        # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # # Process the image and find hands
        # results = self.hands.process(img_rgb)
        # return results
        h, w = frame.shape[:2]
        mp_image = mp.Image(mp.ImageFormat.SRGB, frame)
        result = self._landmarker.detect_for_video(mp_image, self.timestamp_ts)
        self.timestamp_ts += 33

        if not result.face_landmarks:
            return None

        lms = result.face_landmarks[0]
        xs = [int(lms[i].x * w) for i in LEFT_EYE_LANDMARKS]
        ys = [int(lms[i].y * h) for i in LEFT_EYE_LANDMARKS]

        x1 = min(xs) - 5
        y1 = min(ys) - 2
        x2 = max(xs) + 5
        y2 = max(ys) + 2

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        return x1, y1, x2, y2
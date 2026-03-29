# pyright: reportAttributeAccessIssue=false

import cv2
from bbox_transfer import DEFAULT_R, DEFAULT_T, project_bbox_to_cam2
from detector import Detector


def draw_bbox(frame, bbox, color, label):
    if bbox is None:
        return

    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def main():
    print("Hello from squidly-hardware-multi-camera-mapping!")
    # Open two cameras
    cap0 = cv2.VideoCapture(1)
    cap1 = cv2.VideoCapture(2)

    if not cap0.isOpened():
        raise RuntimeError("Camera 0 not available")
    if not cap1.isOpened():
        raise RuntimeError("Camera 1 not available")
    
    detector0 = Detector()
    detector1 = Detector()

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        # Add labels to each frame
        cv2.putText(frame0, "Camera 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame1, "Camera 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Detect faces in each frame
        bbox0 = detector0.detect(frame0)
        bbox1 = detector1.detect(frame1)
        projected_bbox0 = project_bbox_to_cam2(
            bbox1,
            R=DEFAULT_R,
            T=DEFAULT_T,
            image_shape_cam2=frame0.shape,
        )

        # Draw bounding boxes if faces are detected
        draw_bbox(frame0, bbox0, (0, 0, 255), "Cam0 detect")
        draw_bbox(frame1, bbox1, (0, 255, 0), "Cam1 detect")

        # Combine side by side
        combined = cv2.hconcat([frame0, frame1])

        projection_view = frame0.copy()
        cv2.putText(
            projection_view,
            "Projection View",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        draw_bbox(projection_view, bbox0, (0, 0, 255), "Cam0 detect")
        draw_bbox(projection_view, projected_bbox0, (255, 0, 0), "Projected from Cam1")

        projection_view_w = projection_view.shape[1]
        _, w = combined.shape[:2]
        projection_padded = cv2.copyMakeBorder(
            projection_view,
            0,
            0,
            0,
            w - projection_view_w,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        v_combined = cv2.vconcat([combined, projection_padded])

        cv2.imshow("Dual Webcam", v_combined)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

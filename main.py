import cv2
from detector import Detector

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

        # Draw bounding boxes if faces are detected
        if bbox0:
            x1, y1, x2, y2 = bbox0
            cv2.rectangle(frame0, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if bbox1:
            x1, y1, x2, y2 = bbox1
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Combine side by side
        combined = cv2.hconcat([frame0, frame1])
        # Pad frame0 to match width of combined
        frame0_w = frame0.shape[1]
        _, w = combined.shape[:2]
        frame0_padded = cv2.copyMakeBorder(frame0, 0, 0, 0, w - frame0_w, cv2.BORDER_CONSTANT, value=0)

        # Draw bbox0 on the left half (frame0 region)
        if bbox0 is not None:
            x1, y1, x2, y2 = bbox0
            cv2.rectangle(frame0_padded, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame0_padded, "Cam0", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Draw bbox1 on the frame0 region (no offset)
        if bbox1 is not None:
            x1, y1, x2, y2 = bbox1
            cv2.rectangle(frame0_padded, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame0_padded, "Cam1", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        v_combined = cv2.vconcat([combined, frame0_padded])

        cv2.imshow("Dual Webcam", v_combined)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

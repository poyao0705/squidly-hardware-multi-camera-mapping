import cv2


def main():
    print("Hello from squidly-hardware-multi-camera-mapping!")
    # Open two cameras
    cap0 = cv2.VideoCapture(1)
    cap1 = cv2.VideoCapture(2)

    if not cap0.isOpened():
        raise RuntimeError("Camera 0 not available")
    if not cap1.isOpened():
        raise RuntimeError("Camera 1 not available")

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        # Resize to same size (important if different cameras)
        frame0 = cv2.resize(frame0, (640, 480))
        frame1 = cv2.resize(frame1, (640, 480))

        # Combine side by side
        combined = cv2.hconcat([frame0, frame1])

        cv2.imshow("Dual Webcam", combined)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

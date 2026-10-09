import cv2

for i in range(3):
    print(f"Testing camera index {i}...")
    cam = cv2.VideoCapture(i)

    if not cam.isOpened():
        print("Could not open camera.")
        continue

    print("Press Q to close this preview and test the next camera.")

    while True:
        ret, frame = cam.read()

        if not ret:
            print("No frame received.")
            break

        cv2.imshow(f"Camera Index {i}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

print("Camera test completed.")
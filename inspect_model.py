import cv2

for index in range(5):
    print(f"\nOpening camera index {index}...")

    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print(f"Index {index}: unavailable")
        camera.release()
        continue

    ret, frame = camera.read()

    if ret:
        print(f"Index {index}: displaying preview")
        cv2.imshow(
            f"CAMERA INDEX {index} - Press Q for next",
            frame
        )

        while True:
            ret, frame = camera.read()

            if ret:
                cv2.imshow(
                    f"CAMERA INDEX {index} - Press Q for next",
                    frame
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        print(f"Index {index}: no frame received")

    camera.release()
    cv2.destroyAllWindows()

print("\nCamera identification finished.")
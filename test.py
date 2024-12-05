from inference.models.utils import get_roboflow_model
import cv2
import os
from dotenv import load_dotenv

load_dotenv()


MODEL_NAME = "bola-oren"
MODEL_VERSION = "5"
API_KEY = os.getenv("API_KEY")


print("\n\n========== myudakk ==========")

CAP = cv2.VideoCapture(int(input("Video Capture (0,1,2,3) > ")))
CONFIDENCE = float(input("CONFIDENCE (0.5,0.6,0.7,0.8,0.9,1.0) > "))
IOU_THRESHOLD = float(input("IOU_THRESHOLD (0.5,0.6,0.7,0.8,0.9,1.0) > "))
print(f"MODEL_NAME = {MODEL_NAME}")
print(f"MODEL_VERSION = {MODEL_VERSION}")
print(f"API_KEY = {API_KEY}")


if not CAP.isOpened():
    print("Error: Could not open camera.")
    exit()

MODEL = get_roboflow_model(
    model_id="{}/{}".format(MODEL_NAME, MODEL_VERSION),
    api_key=API_KEY,
)

while 1:
    ret, frame = CAP.read()
    if not ret:
        print("Error gblog")

    results = MODEL.infer(
        image=frame, confidence=CONFIDENCE, iou_threshold=IOU_THRESHOLD
    )

    if results[0].predictions:
        prediction = results[0].predictions[0]
        print(prediction)

        x_center = int(prediction.x)
        y_center = int(prediction.y)
        width = int(prediction.width)
        height = int(prediction.height)

        x0 = x_center - width // 2
        y0 = y_center - height // 2
        x1 = x_center + width // 2
        y1 = y_center + height // 2

        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 10)
        cv2.putText(
            frame,
            "BOLA MERAH",
            (x0, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2,
        )

    cv2.imshow("BOLA", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

CAP.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2

model = YOLO("court_keypoints.pt")


image_path = "../test_images/image1.jpg"
results = model(image_path)

annotated_frame = results[0].plot()
cv2.imshow("Pose Estimation", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

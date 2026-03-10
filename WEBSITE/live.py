import cv2
from ultralytics import YOLO

# Load YOLO model
model_path = "best.pt"  # Replace with the path to your YOLOv8 or YOLOv11 model weights
model = YOLO(model_path)

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Function to draw bounding boxes and labels
def draw_boxes(frame, results):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID
            label = f"{model.names[cls]} {conf:.2f}"
            color = (0, 255, 0)  # Green for bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform inference
    results = model(frame)

    # Draw detections on the frame
    draw_boxes(frame, results)

    # Display the frame
    cv2.imshow("Live YOLO Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
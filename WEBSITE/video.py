import cv2
import numpy as np
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_FROM = 'daminmain@gmail.com'  # Replace with your email
EMAIL_PASSWORD = 'kpqtxqskedcykwjz'  # Replace with your app password
EMAIL_TO = 'bashith67@gmail.com'  # Replace with recipient's email

def send_email(subject, body, image_path=None):
    """Send an email with optional image attachment."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = subject
        
        # Attach the email body
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach the image if provided
        if image_path:
            with open(image_path, 'rb') as img_file:
                img = MIMEImage(img_file.read(), name=os.path.basename(image_path))
                msg.attach(img)
        
        # Send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Load the YOLO model
model = YOLO("best.pt")  # Ensure best.pt is trained for landslide detection

# Open the video file
cap = cv2.VideoCapture("testing2.mp4")

# Get the video's width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video.mp4", fourcc, 30, (width, height))

# Flag to send email
alert_sent = False

# Process each frame
frame_id = 0  # Track frame number for debugging
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame, conf=0.25)  # Adjust confidence threshold if needed

    # Draw the detection results on the frame
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        label = model.names[int(box.cls[0])]  # Detected label
        confidence = round(box.conf[0].item(), 2)  # Confidence score
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check if a 'landslide' is detected and send an email if not already sent
        if label == 'landslide' and not alert_sent:
            output_image_path = f"frame_with_landslide_{frame_id}.jpg"
            cv2.imwrite(output_image_path, frame)  # Save the frame with detection
            send_email(
                subject="Landslide Detected",
                body="A landslide has been detected in the video.",
                image_path=output_image_path
            )
            alert_sent = True

    # Write the frame to the output video
    out.write(frame)

    # Optional: Print detection info for debugging
    print(f"Processed Frame {frame_id}")
    frame_id += 1

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

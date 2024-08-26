from ultralytics import YOLO
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

# Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="TWWx3s94S7ZX3Kg2rq2e"
)

# Load the trained YOLO model
model = YOLO('runs/detect/train2/weights/best.pt')

# Path to the video
video_path = 'Copy of cover_0015.avi'
output_path = 'annotated_output_with_bounce4.mp4'  # Output video path

# Open video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Slow down factor (e.g., 2 means the video will be half as fast)
slowdown_factor = 2
new_fps = fps / slowdown_factor

# Initialize VideoWriter with MP4 codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))

def detect_pitch(frame):
    """
    Function to detect the pitch area in the frame using Roboflow API.
    """
    # Save the current frame to a temporary image file
    cv2.imwrite('current_frame.jpg', frame)

    # Infer using the Roboflow model
    result = CLIENT.infer('current_frame.jpg', model_id="cricket-pitch-t9j9g/2")

    # Extract bounding box coordinates for the pitch
    if result['predictions']:
        pitch = result['predictions'][0]
        x, y, w, h = pitch['x'], pitch['y'], pitch['width'], pitch['height']
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2
    else:
        print("Error: No pitch detected.")
        return None, None, None, None

def draw_pitch_length_annotations(frame, pitch_x1, pitch_y1, pitch_x2, pitch_y2):
    """
    Draw pitch length annotations on the frame based on detected pitch coordinates.
    """
    # Define colors for each region
    colors = {
        "Short": (0, 0, 255),   # Red
        "Good": (0, 255, 0),    # Green
        "Full": (255, 0, 0),    # Blue
        "Yorker": (0, 255, 255) # Yellow
    }

    # Ensure the coordinates are integers
    pitch_x1, pitch_y1, pitch_x2, pitch_y2 = map(int, [pitch_x1, pitch_y1, pitch_x2, pitch_y2])

    # Calculate y-coordinates for the length annotations based on pitch height
    yorker_length_y = int(pitch_y1 + 0.10 * (pitch_y2 - pitch_y1))
    full_length_y = int(pitch_y1 + 0.25 * (pitch_y2 - pitch_y1))
    good_length_y = int(pitch_y1 + 0.40 * (pitch_y2 - pitch_y1))
    short_length_y = int(pitch_y1 + 0.55 * (pitch_y2 - pitch_y1))

    # Draw pitch length regions on the frame
    cv2.rectangle(frame, (pitch_x1, short_length_y), (pitch_x2, pitch_y2), colors["Short"], 2)
    cv2.rectangle(frame, (pitch_x1, good_length_y), (pitch_x2, short_length_y), colors["Good"], 2)
    cv2.rectangle(frame, (pitch_x1, full_length_y), (pitch_x2, good_length_y), colors["Full"], 2)
    cv2.rectangle(frame, (pitch_x1, yorker_length_y), (pitch_x2, full_length_y), colors["Yorker"], 2)

    # Add text labels
    cv2.putText(frame, "Short", (pitch_x1 + 10, short_length_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Short"], 2)
    cv2.putText(frame, "Good", (pitch_x1 + 10, good_length_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Good"], 2)
    cv2.putText(frame, "Full", (pitch_x1 + 10, full_length_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Full"], 2)
    cv2.putText(frame, "Yorker", (pitch_x1 + 10, yorker_length_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Yorker"], 2)

def classify_bounce(ball_y, pitch_y1, pitch_y2):
    """
    Classify the bounce position based on the y-coordinate relative to the pitch.
    """
    yorker_length_y = int(pitch_y1 + 0.10 * (pitch_y2 - pitch_y1))
    full_length_y = int(pitch_y1 + 0.20 * (pitch_y2 - pitch_y1))
    good_length_y = int(pitch_y1 + 0.35 * (pitch_y2 - pitch_y1))
    short_length_y = int(pitch_y1 + 0.50 * (pitch_y2 - pitch_y1))

    if ball_y >= short_length_y:
        return "Short"
    elif ball_y >= good_length_y:
        return "Good"
    elif ball_y >= full_length_y:
        return "Full"
    elif ball_y >= yorker_length_y:
        return "Yorker"
    else:
        return "Beyond Yorker"

# Initialize variables to track the ball's position and detect bounce
prev_ball_y = None
bounce_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pitch detection using Roboflow API
    pitch_x1, pitch_y1, pitch_x2, pitch_y2 = detect_pitch(frame)

    # Ensure that the pitch was detected before proceeding
    if pitch_x1 is None:
        print("Error: Pitch detection failed. Skipping frame.")
        continue

    # Perform detection
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Draw the pitch lines and pitch length annotations on the annotated frame
    draw_pitch_length_annotations(annotated_frame, pitch_x1, pitch_y1, pitch_x2, pitch_y2)

    # Initialize variables to track the highest confidence box
    highest_confidence = 0
    best_box = None
    best_label = None

    for detection in results[0].boxes:
        # Extract bounding box coordinates and confidence score
        x1, y1, x2, y2 = detection.xyxy[0].tolist()
        confidence = detection.conf[0].tolist()  # Confidence score of the detection

        # Update the best box if the current confidence is higher
        if confidence > highest_confidence:
            highest_confidence = confidence
            best_box = (x1, y1, x2, y2)
            best_label = detection.cls[0].tolist()  # Label of the detection

    if best_box is not None:
        x1, y1, x2, y2 = best_box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Detect bounce
        if prev_ball_y is not None:
            if not bounce_detected and cy > prev_ball_y:  # Ball was going downwards
                bounce_detected = True
                print(f"Bounce detected at ({cx}, {cy})")
                bounce_position = classify_bounce(cy, pitch_y1, pitch_y2)
                cv2.putText(annotated_frame, f"Bounce: {bounce_position}", (cx, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            elif cy < prev_ball_y:
                bounce_detected = False  # Reset for next potential bounce detection

        prev_ball_y = cy

        # Define colors for different lengths
        colors = {
            "Short Length": (0, 0, 255),     # Red
            "Good Length": (0, 255, 0),      # Green
            "Full Length": (255, 0, 0),      # Blue
            "Yorker Length": (0, 255, 255),  # Yellow
            "Beyond Yorker": (255, 255, 255) # White
        }

        # Draw the bounding box and ball landing classification
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), colors.get(best_label, (255, 255, 255)), 2)
        
    # Write the annotated frame to the output video
    out.write(annotated_frame)

# Release the video capture and writer objects
cap.release()
out.release()

from ultralytics import YOLO
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

# Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="TWWx3s94S7ZX3Kg2rq2e"
)

CLIENT_POSE = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="TWWx3s94S7ZX3Kg2rq2e"  # Use the same API key
)

# Load the trained YOLO model
model = YOLO('runs/detect/train2/weights/best.pt')

# Path to the video
video_path = 'Copy of cover_0015.avi'
output_path = 'annotated_output_with_continuous_display12.mp4'  # Updated output video path

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

def detect_pose(frame):
    """
    Function to detect the cricket pose in the frame using Roboflow API.
    """
    cv2.imwrite('current_frame_pose.jpg', frame)
    result = CLIENT_POSE.infer('current_frame_pose.jpg', model_id="cricket_pose_segmentation/2")
    
    print("Pose detection result:", result)  # Add this line to inspect the response

    if result['predictions']:
        pose = result['predictions'][0]
        print("Pose prediction details:", pose)  # Add this line to inspect the details of the pose
        # Adjust this based on actual key names in the response
        return pose.get('class', "Unknown")  # Use the actual key name found in the response
    else:
        print("Error: No pose detected.")
        return "Unknown"

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
    full_length_y = int(pitch_y1 + 0.17 * (pitch_y2 - pitch_y1))
    good_length_y = int(pitch_y1 + 0.25 * (pitch_y2 - pitch_y1))
    short_length_y = int(pitch_y1 + 0.40 * (pitch_y2 - pitch_y1))

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
    Returns both a general classification and a more specific length description.
    """
    yorker_length_y = int(pitch_y1 + 0.10 * (pitch_y2 - pitch_y1))
    full_length_y = int(pitch_y1 + 0.20 * (pitch_y2 - pitch_y1))
    good_length_y = int(pitch_y1 + 0.35 * (pitch_y2 - pitch_y1))
    short_length_y = int(pitch_y1 + 0.50 * (pitch_y2 - pitch_y1))

    if ball_y >= short_length_y:
        return "Short", "Very Short"
    elif ball_y >= good_length_y:
        if ball_y >= (good_length_y + short_length_y) / 2:
            return "Short", "Back of a Length"
        else:
            return "Good", "Good Length"
    elif ball_y >= full_length_y:
        if ball_y >= (full_length_y + good_length_y) / 2:
            return "Good", "Full of a Length"
        else:
            return "Full", "Full"
    elif ball_y >= yorker_length_y:
        return "Yorker", "Yorker"
    else:
        return "Beyond Yorker", "Full Toss"

# Initialize variables to track the ball's position and detect bounce
prev_ball_y = None
bounce_detected = False
pitch_x1, pitch_y1, pitch_x2, pitch_y2 = None, None, None, None

# New variables to store the last detected bounce information
last_bounce_classification = "N/A"
last_length_description = "N/A"
last_bounce_coordinates = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pitch detection using Roboflow API
    detected_pitch_x1, detected_pitch_y1, detected_pitch_x2, detected_pitch_y2 = detect_pitch(frame)
    shot_type = detect_pose(frame)

    # Perform detection
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Track the pitch area
    if detected_pitch_x1 is not None:
        pitch_x1, pitch_y1, pitch_x2, pitch_y2 = detected_pitch_x1, detected_pitch_y1, detected_pitch_x2, detected_pitch_y2

    # Draw the pitch lines and pitch length annotations only when bounce is detected
    if bounce_detected and pitch_x1 is not None:
        draw_pitch_length_annotations(annotated_frame, pitch_x1, pitch_y1, pitch_x2, pitch_y2)
        bounce_detected = False  # Reset after drawing annotations

    # Initialize variables to track the highest confidence box that matches the circular shape criteria
    highest_confidence_box = None
    highest_confidence = -1

    for r in results[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        width = x2 - x1
        height = y2 - y1

        # Aspect ratio check to ensure detection is circular
        aspect_ratio = width / height

        if 0.9 <= aspect_ratio <= 1.1:
            if r.conf[0] > highest_confidence:
                highest_confidence = r.conf[0]
                highest_confidence_box = (x1, y1, x2, y2)

    # If a circle-like object is detected, process it
    if highest_confidence_box:
        x1, y1, x2, y2 = highest_confidence_box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if prev_ball_y is not None and center_y > prev_ball_y:  # Check if the ball is falling down
            if pitch_y1 is not None and pitch_y2 is not None:
                if center_y >= pitch_y1 and center_y <= pitch_y2:
                    bounce_detected = True
                    last_bounce_classification, last_length_description = classify_bounce(center_y, pitch_y1, pitch_y2)
                    last_bounce_coordinates = (center_x, center_y)

        # Update previous ball position
        prev_ball_y = center_y

    # Display bounce classification on the frame
    cv2.putText(annotated_frame, f"Bounced: {last_bounce_classification} ({last_length_description})", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the detected pose (e.g., "Cover Drive" or "Unknown")
    cv2.putText(annotated_frame, f"Shot Type: {shot_type}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Write the frame to the output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()
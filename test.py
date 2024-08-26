from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Path to the video
video_path = 'Copy of cover_0015.avi'
output_path = 'annotated_output_with_length4.mp4'  # Output video path

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

# Capture the first frame to establish fixed line positions
ret, reference_frame = cap.read()
if not ret:
    print(f"Error: Could not read the first frame of video")
    cap.release()
    out.release()
    exit()

# Define lengths as a proportion of the reference frame height
def get_fixed_length_lines(frame_height):
    short_length_y = int(0.65 * frame_height)
    good_length_y = int(0.55 * frame_height)
    full_length_y = int(0.45 * frame_height)
    yorker_length_y = int(0.35 * frame_height)

    return {
        "short_length_y": short_length_y,
        "good_length_y": good_length_y,
        "full_length_y": full_length_y,
        "yorker_length_y": yorker_length_y
    }

# Get fixed line positions based on the reference frame
fixed_lines = get_fixed_length_lines(height)

def classify_ball_length(ball_x, ball_y, frame_width, fixed_lines):
    short_length_y = fixed_lines["short_length_y"]
    good_length_y = fixed_lines["good_length_y"]
    full_length_y = fixed_lines["full_length_y"]
    yorker_length_y = fixed_lines["yorker_length_y"]

    short_region_x = int(0.25 * frame_width)
    good_region_x = int(0.35 * frame_width)
    full_region_x = int(0.45 * frame_width)
    yorker_region_x = int(0.55 * frame_width)

    if ball_y >= short_length_y:
        length = "Short Pitched" if ball_x >= short_region_x else "Short Length"
    elif ball_y >= good_length_y:
        length = "Good Length" if ball_x >= good_region_x else "Full Length"
    elif ball_y >= full_length_y:
        length = "Full Length" if ball_x >= full_region_x else "Yorker Length"
    elif ball_y >= yorker_length_y:
        length = "Yorker Length" if ball_x >= yorker_region_x else "Beyond Yorker"
    else:
        length = "Beyond Yorker"
    
    return length

def draw_pitch_lines(frame, fixed_lines):
    # Define colors for each region
    colors = {
        "Short": (0, 0, 255),   # Red
        "Good": (0, 255, 0),    # Green
        "Full": (255, 0, 0),    # Blue
        "Yorker": (0, 255, 255) # Yellow
    }

    # Draw pitch regions on the frame
    cv2.rectangle(frame, (0, fixed_lines["short_length_y"]), (width, fixed_lines["good_length_y"]), colors["Short"], 2)
    cv2.rectangle(frame, (0, fixed_lines["good_length_y"]), (width, fixed_lines["full_length_y"]), colors["Good"], 2)
    cv2.rectangle(frame, (0, fixed_lines["full_length_y"]), (width, fixed_lines["yorker_length_y"]), colors["Full"], 2)
    cv2.rectangle(frame, (0, fixed_lines["yorker_length_y"]), (width, 0), colors["Yorker"], 2)

    # Optionally add text labels
    cv2.putText(frame, "Short", (50, fixed_lines["short_length_y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors["Short"], 2)
    cv2.putText(frame, "Good", (50, fixed_lines["good_length_y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors["Good"], 2)
    cv2.putText(frame, "Full", (50, fixed_lines["full_length_y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors["Full"], 2)
    cv2.putText(frame, "Yorker", (50, fixed_lines["yorker_length_y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors["Yorker"], 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Draw the pitch lines on the annotated frame
    draw_pitch_lines(annotated_frame, fixed_lines)

    # Draw bounding boxes and classify ball length
    for detection in results[0].boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = detection.xyxy[0].tolist()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        # Classify the ball's length and region based on its position
        ball_length = classify_ball_length(cx, cy, width, fixed_lines)
        print(f"Ball landed in {ball_length}")

        # Define colors for different lengths
        colors = {
            "Short Length": (0, 0, 255),     # Red
            "Good Length": (0, 255, 0),      # Green
            "Full Length": (255, 0, 0),      # Blue
            "Yorker Length": (0, 255, 255),  # Yellow
            "Beyond Yorker": (255, 255, 255) # White
        }

        # Draw the bounding box and ball landing classification
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), colors.get(ball_length, (255, 255, 255)), 2)
        cv2.putText(annotated_frame, ball_length, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors.get(ball_length, (255, 255, 255)), 2)

    # Write annotated frame to output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()

print(f"Annotated video saved as {output_path}")

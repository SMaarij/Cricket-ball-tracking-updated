# Cricket Ball Tracking and Length Classification Project

This project is a computer vision-based system for tracking a cricket ball during play and classifying its bounce position into categories such as **yorker**, **full**, **good**, or **short**. Using advanced object detection techniques and custom algorithms, the system analyzes video footage to detect the ball's motion, identify the bounce point, and classify the length based on predefined pitch zones.

---

## Key Features

### 1. Ball and Pitch Detection
- Utilizes a YOLOv8 model trained on cricket-specific datasets to detect the cricket ball and the pitch in each video frame.
- Dynamically annotates the pitch area and marks specific pitch lengths for classification.

### 2. Bounce Point Detection
- Tracks the cricket ball across consecutive frames and calculates its velocity in the vertical direction.
- Identifies the bounce point as the moment when the ball’s velocity changes direction, transitioning from downward motion (falling) to upward motion (rising).

### 3. Length Classification
- Divides the pitch into predefined zones: **yorker**, **full**, **good**, and **short**, based on pitch dimensions.
- Uses the detected bounce position to classify the ball length in real-time.

### 4. Dynamic Pitch Handling
- Accounts for camera motion by dynamically detecting the pitch region in every frame.
- Ensures consistent annotation of pitch zones even when the camera angle changes.

### 5. Robust Tracking
- Maintains accuracy despite variations in video quality and environmental factors like lighting or camera movement.
- Handles scenarios where the pitch lines become partially or fully obscured by camera panning.

---

## How Bounce Position is Calculated

### 1. Velocity Calculation:
- Tracks the ball's position in each frame using YOLOv8 detections.
- Calculates the vertical velocity between consecutive frames:
  \[
  v_y = y_{current} - y_{previous}
  \]
  where \( y_{current} \) and \( y_{previous} \) are the ball's vertical coordinates in consecutive frames.

### 2. Direction Change Detection:
- Monitors the sign of the vertical velocity:
  - **Downward motion:** \( v_y > 0 \)
  - **Upward motion:** \( v_y < 0 \)
- Identifies the bounce point as the frame where \( v_y \) changes from positive to negative.

### 3. Frame Validation:
- Applies smoothing techniques to ensure the detected bounce point is accurate and not influenced by noise or missed detections.

---

## Usage

1. Train the YOLOv8 model with cricket ball and pitch datasets.
2. Process video footage to detect the ball and pitch in real-time.
3. Use the bounce detection algorithm to classify ball lengths dynamically.
4. Output annotated frames or videos highlighting the bounce position and corresponding ball length classification.

---

## Applications
- **Cricket Coaching and Training**: Enhanced video analysis for player improvement.
- **Live Cricket Broadcasting**: Automatic insights for on-air commentary.
- **Research**: Detailed studies on cricket ball behavior and pitch performance.

---

## Conclusion

This project demonstrates how modern computer vision techniques can be applied to sports analytics, providing precise and actionable insights for players, coaches, and analysts.

---

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import time


# === 1. Define Model ===
class SkeletonLSTM(nn.Module):
    def __init__(self, input_size=75, hidden_size=128, num_layers=2, num_classes=3):
        super(SkeletonLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, joints, coords = x.shape
        x = x.view(batch_size, seq_len, joints * coords)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1])
        return out


# === 2. Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkeletonLSTM()
model.load_state_dict(torch.load("src/models/pytorchModel.pth", map_location=device))
model.to(device)
model.eval()


# === 3. Utility Functions ===
def convert_mediapipe_to_ntu25(landmarks):
    def avg(*ids): return np.mean([landmarks[i] for i in ids], axis=0)

    ntu = [None] * 25
    ntu[0] = avg(24, 23)
    ntu[1] = avg(11, 12, 23, 24)
    ntu[2] = avg(9, 10, 11, 12)
    ntu[3] = landmarks[0]
    ntu[4] = landmarks[11]
    ntu[5] = landmarks[13]
    ntu[6] = landmarks[15]
    ntu[7] = avg(19, 17, 15)
    ntu[8] = landmarks[12]
    ntu[9] = landmarks[14]
    ntu[10] = landmarks[16]
    ntu[11] = avg(16, 18, 20)
    ntu[12] = landmarks[23]
    ntu[13] = landmarks[25]
    ntu[14] = landmarks[27]
    ntu[15] = landmarks[31]
    ntu[16] = landmarks[24]
    ntu[17] = landmarks[26]
    ntu[18] = landmarks[28]
    ntu[19] = landmarks[32]
    ntu[20] = avg(11, 12)
    ntu[21] = avg(17, 19)
    ntu[22] = landmarks[21]
    ntu[23] = avg(18, 20)
    ntu[24] = landmarks[22]
    return np.array(ntu)


def rotate_skeleton_upright(skel):
    if skel.shape != (25, 3): return skel
    try:
        hip = skel[0]
        neck = skel[20]
        r_sh = skel[8]
        l_sh = skel[4]
        y_axis = neck - hip
        y_axis /= np.linalg.norm(y_axis) + 1e-8
        x_axis = l_sh - r_sh
        x_axis /= np.linalg.norm(x_axis) + 1e-8
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-8
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis) + 1e-8
        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        return skel @ R
    except:
        return skel


def normalize_skeleton_scale(skeleton):
    hip = skeleton[0]
    neck = skeleton[20]
    spine_len = np.linalg.norm(neck - hip) + 1e-8
    return skeleton / spine_len


# === 4. Init Pose Estimation ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2)

# === 5. Video and Labels ===
cap = cv2.VideoCapture(0)
sequence = []
labels = ['Clapping', 'Hand Waving', 'Hopping']
label_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
current_label = ""
confidence = 0.0
fps = 0
prev_time = time.time()

# === 6. NTU Edges for Drawing ===
# Group edges by body parts for different colors
body_edges = [(0, 1), (1, 20), (20, 2), (2, 3)]
left_arm_edges = [(20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (21, 22)]
right_arm_edges = [(20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (23, 24)]
left_leg_edges = [(0, 12), (12, 13), (13, 14), (14, 15)]
right_leg_edges = [(0, 16), (16, 17), (17, 18), (18, 19)]

# Colors for different body parts
body_color = (255, 51, 153)  # Pink
left_arm_color = (0, 255, 255)  # Yellow
right_arm_color = (0, 165, 255)  # Orange
left_leg_color = (153, 0, 255)  # Purple
right_leg_color = (255, 0, 102)  # Red-Pink

# Joint colors by importance
central_joint_color = (0, 0, 255)  # Red - central joints
limb_joint_color = (0, 255, 0)  # Green - limb joints
extremity_joint_color = (255, 0, 0)  # Blue - extremities

# Define central, limb and extremity joints
central_joints = [0, 1, 2, 3, 20]
limb_joints = [4, 5, 8, 9, 12, 13, 16, 17]
extremity_joints = [6, 7, 10, 11, 14, 15, 18, 19, 21, 22, 23, 24]

# === 7. Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret: break

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Process frame for pose detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # Create a dark semi-transparent overlay for UI elements
    h, w = frame.shape[:2]
    ui_overlay = np.zeros((h, w, 3), dtype=np.uint8)

    if results.pose_landmarks:
        # 3D World landmarks
        landmarks_world = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        joints = convert_mediapipe_to_ntu25(landmarks_world)

        if joints is not None and len(joints) == 25:
            # Preprocessing: Center → Align → Scale
            normed = joints - joints[0]
            aligned = rotate_skeleton_upright(normed)
            aligned = normalize_skeleton_scale(aligned)

            sequence.append(aligned)
            if len(sequence) > 25:
                sequence.pop(0)

            # Prediction
            if len(sequence) == 25:
                input_tensor = torch.tensor([sequence], dtype=torch.float32).to(device)
                with torch.no_grad():
                    pred = model(input_tensor)
                    softmax = torch.nn.functional.softmax(pred, dim=1)[0]
                    pred_class = torch.argmax(pred, dim=1).item()
                    current_label = labels[pred_class]
                    confidence = softmax[pred_class].item()

        # === Enhanced 2D Skeleton Drawing ===
        h, w = frame.shape[:2]
        landmarks_2d = results.pose_landmarks.landmark
        joint_coords = convert_mediapipe_to_ntu25(
            np.array([[lm.x * w, lm.y * h, lm.z] for lm in landmarks_2d])
        )

        if joint_coords is not None:
            # Draw edges with different colors by body part
            # Body (spine)
            for i, j in body_edges:
                x1, y1 = int(joint_coords[i][0]), int(joint_coords[i][1])
                x2, y2 = int(joint_coords[j][0]), int(joint_coords[j][1])
                cv2.line(frame, (x1, y1), (x2, y2), body_color, 3)

            # Left arm
            for i, j in left_arm_edges:
                x1, y1 = int(joint_coords[i][0]), int(joint_coords[i][1])
                x2, y2 = int(joint_coords[j][0]), int(joint_coords[j][1])
                cv2.line(frame, (x1, y1), (x2, y2), left_arm_color, 3)

            # Right arm
            for i, j in right_arm_edges:
                x1, y1 = int(joint_coords[i][0]), int(joint_coords[i][1])
                x2, y2 = int(joint_coords[j][0]), int(joint_coords[j][1])
                cv2.line(frame, (x1, y1), (x2, y2), right_arm_color, 3)

            # Left leg
            for i, j in left_leg_edges:
                x1, y1 = int(joint_coords[i][0]), int(joint_coords[i][1])
                x2, y2 = int(joint_coords[j][0]), int(joint_coords[j][1])
                cv2.line(frame, (x1, y1), (x2, y2), left_leg_color, 3)

            # Right leg
            for i, j in right_leg_edges:
                x1, y1 = int(joint_coords[i][0]), int(joint_coords[i][1])
                x2, y2 = int(joint_coords[j][0]), int(joint_coords[j][1])
                cv2.line(frame, (x1, y1), (x2, y2), right_leg_color, 3)

            # Draw joints with different colors based on joint type
            for i, (x, y, _) in enumerate(joint_coords):
                x, y = int(x), int(y)

                # Determine joint color and size based on its category
                if i in central_joints:
                    color = central_joint_color
                    size = 8
                elif i in limb_joints:
                    color = limb_joint_color
                    size = 6
                else:  # extremity joints
                    color = extremity_joint_color
                    size = 4

                # Draw joint
                cv2.circle(frame, (x, y), size, color, -1)
                cv2.circle(frame, (x, y), size + 1, (255, 255, 255), 1)  # White outline

    # Create UI panel with action recognition results
    # Add semi-transparent background panel
    panel_overlay = frame.copy()
    cv2.rectangle(panel_overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(panel_overlay, 0.3, frame, 0.7, 0)

    # Add UI elements
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show capture status
    buffer_status = f"Buffer: {len(sequence)}/25"
    cv2.putText(frame, buffer_status, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display action and confidence if we have a prediction
    if current_label:
        label_color = label_colors[labels.index(current_label)]
        cv2.putText(frame, f"Action: {current_label}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

        # Add confidence bar
        conf_percent = int(confidence * 100)
        bar_width = int(200 * confidence)
        cv2.rectangle(frame, (100, 110), (100 + bar_width, 115), label_color, -1)
        cv2.rectangle(frame, (100, 110), (300, 115), (255, 255, 255), 1)
        cv2.putText(frame, f"{conf_percent}%", (305, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add legend in bottom-left
    y_offset = h - 160
    # Draw legend title
    cv2.rectangle(frame, (10, y_offset - 30), (180, y_offset + 130), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, y_offset - 30), (180, y_offset + 130), (255, 255, 255), 1)
    cv2.putText(frame, "SKELETON LEGEND", (20, y_offset - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw color codes for body parts
    cv2.line(frame, (20, y_offset + 10), (40, y_offset + 10), body_color, 3)
    cv2.putText(frame, "Spine", (50, y_offset + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.line(frame, (20, y_offset + 30), (40, y_offset + 30), left_arm_color, 3)
    cv2.putText(frame, "Left Arm", (50, y_offset + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.line(frame, (20, y_offset + 50), (40, y_offset + 50), right_arm_color, 3)
    cv2.putText(frame, "Right Arm", (50, y_offset + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.line(frame, (20, y_offset + 70), (40, y_offset + 70), left_leg_color, 3)
    cv2.putText(frame, "Left Leg", (50, y_offset + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.line(frame, (20, y_offset + 90), (40, y_offset + 90), right_leg_color, 3)
    cv2.putText(frame, "Right Leg", (50, y_offset + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw joint type legend
    cv2.circle(frame, (25, y_offset + 110), 5, central_joint_color, -1)
    cv2.putText(frame, "Main Joints", (50, y_offset + 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the result
    cv2.imshow("Enhanced Action Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

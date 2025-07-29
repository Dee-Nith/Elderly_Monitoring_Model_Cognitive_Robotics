import cv2
import mediapipe as mp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from gaze import get_gaze_direction
from hand import get_hand_preshape
from trajectory import get_trajectory_direction
from logger import ActionLogger

# === TRAINING DUMMY MODEL INSIDE SCRIPT ===
X_train = pd.DataFrame({
    'gaze': ['left', 'right', 'center', 'left', 'right', 'center'],
    'hand': ['gripping', 'open', 'gripping', 'open', 'gripping', 'open'],
    'trajectory': ['left', 'right', 'center', 'left', 'right', 'center']
})
y_train = ['cup', 'medication', 'remote', 'cup', 'emergency_button', 'remote']

encoders = {}
X_encoded = pd.DataFrame()
for col in X_train.columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_train[col])
    encoders[col] = le

action_encoder = LabelEncoder()
y_encoded = action_encoder.fit_transform(y_train)

clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_encoded, y_encoded)

# === SYSTEM CONFIG ===
log_every = 10
logger = ActionLogger(filename="elderly_activity_log.csv")

# === SETUP MEDIAPIPE ===
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(max_num_hands=1)
face_mesh = mp_face.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose()

# === OPEN CAMERA AND SETUP VIDEO WRITER ===
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter(
    "elderly_monitoring_output.avi",
    cv2.VideoWriter_fourcc(*'XVID'),
    20,
    (frame_width, frame_height)
)

trajectory_history = []
frame_count = 0

def trigger_alert(predicted_action):
    if predicted_action in ["medication", "emergency_button"]:
        print(f"[ALERT] High-priority action detected: {predicted_action}")
        return True
    return False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to access webcam.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hand = hands.process(frame_rgb)
    results_face = face_mesh.process(frame_rgb)
    results_pose = pose.process(frame_rgb)

    h, w, _ = frame.shape
    gaze = "unknown"
    hand_shape = "unknown"
    trajectory_dir = "unknown"

    if results_face.multi_face_landmarks:
        face_landmarks = results_face.multi_face_landmarks[0].landmark
        gaze = get_gaze_direction(face_landmarks, w, h)

    if results_hand.multi_hand_landmarks:
        hand_landmarks = results_hand.multi_hand_landmarks[0].landmark
        hand_shape = get_hand_preshape(hand_landmarks, w, h)

    if results_pose.pose_landmarks:
        wrist = results_pose.pose_landmarks.landmark[16]
        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)
        trajectory_history.append((wrist_x, wrist_y))
        if len(trajectory_history) > 10:
            trajectory_history.pop(0)
        trajectory_dir = get_trajectory_direction(trajectory_history)

    try:
        input_data = pd.DataFrame([{
            'gaze': gaze,
            'hand': hand_shape,
            'trajectory': trajectory_dir
        }])
        for col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])
        prediction = clf.predict(input_data)[0]
        predicted_action = action_encoder.inverse_transform([prediction])[0]
    except Exception:
        predicted_action = "unknown"

    alert_flag = trigger_alert(predicted_action)

    if frame_count % log_every == 0:
        logger.log(frame_count, gaze, hand_shape, trajectory_dir, predicted_action)

    # === Overlay Prediction Information ===
    color = (0, 0, 255) if alert_flag else (0, 255, 0)
    cv2.putText(frame, f'Gaze: {gaze}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f'Hand: {hand_shape}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f'Trajectory: {trajectory_dir}', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f'Action: {predicted_action}', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if alert_flag:
        cv2.putText(frame, "⚠️ ALERT", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # === Show and Save Video ===
    cv2.imshow("Elderly Monitoring System", frame)
    out.write(frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

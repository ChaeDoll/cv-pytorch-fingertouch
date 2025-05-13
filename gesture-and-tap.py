import cv2
import torch
import numpy as np
import mediapipe as mp
import pickle
from models.model import LSTMModel
from utils.util import FingerTapDetector, calculate_angles, recognize_action

# --- 메인 비디오 처리 루프 ---
def process_video(cap, model, actions, seq_length, width, height, device):
    # Load calibration for external camera
    with open('./utils/camera_calibration.pkl', 'rb') as f:
        calib_data = pickle.load(f)
    camera_matrix = calib_data['camera_matrix'] # calibration
    dist_coeffs = calib_data['dist_coeffs'] # calibration

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    seq = []
    action_seq = []
    mode = "None"
    tap_detector = FingerTapDetector()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.undistort(img, camera_matrix, dist_coeffs) # calibration
        img = cv2.flip(img, 1)
        result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                angle = calculate_angles(joint)
                feature_vector = np.concatenate([joint.flatten(), angle])
                seq.append(feature_vector)

                if len(seq) >= seq_length:
                    input_data = torch.tensor(np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0))
                    action, action_seq = recognize_action(model, input_data, actions, action_seq, device)
                    if action is not None:
                        mode = action

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                # Enter 모드일 때만 작동
                if mode == "enter": 
                    tap_detector.detect_tap(img, res, width, height)

        cv2.putText(img, f'Mode : {mode}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255) if mode != "None" else (128, 128, 128), 3)
        cv2.imshow('img', img)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# --- 실행 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features = 99
    num_classes = 4
    model = LSTMModel(input_size=features, hidden_size=64, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('models/gesture_model.pth'))
    model.to(device)

    actions = ['enter', 'setting', 'power on', 'power off']
    seq_length = 30
    cap = cv2.VideoCapture(1) # CAM으로 연결 (0 is labtop, 1 is external_camera)
    width, height = 640, 480
    process_video(cap, model, actions, seq_length, width, height, device)

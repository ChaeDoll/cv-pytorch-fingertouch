import cv2
import torch
import numpy as np
import mediapipe as mp
from ..models.model import LSTMModel
from ..utils.util import *  # 전처리, landmark 추출 등 유틸 함수가 있다고 가정

# 손동작 인식
def recognize_action(model, input_data, actions, action_seq, device):
    model.eval()
    input_data = input_data.to(device)
    with torch.no_grad():
        y_pred = model(input_data).squeeze()
        i_pred = int(torch.argmax(y_pred))
        conf = torch.softmax(y_pred, dim=0)[i_pred].item()
    if conf < 0.4:
        return None, action_seq
    action = actions[i_pred]
    action_seq.append(action)
    if len(action_seq) < 7:
        return None, action_seq
    if action_seq[-7:] == [action]*7:
        return action, []
    return None, action_seq

# 각도를 계산하는 함수
def calculate_angles(joint):
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]  # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]  # Child joint
    v = v2 - v1  # [20, 3]
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # Normalize

    angle = np.arccos(np.einsum('nt,nt->n', v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
    return np.degrees(angle)  # Convert radian to degree


# 비디오 처리 루프 함수
def process_video(cap, model, actions, seq_length, width, height, device):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, 
                           min_detection_confidence=0.6, 
                           min_tracking_confidence=0.6)
    seq = []
    action_seq = []
    gesture_count = 0
    mode = "None"
    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting.")
            break
        result = hands.process(img)
        
        cv2.putText(img, f'Gesture change count: {gesture_count}', (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                # 손가락 좌표와 각도 추출
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                angle = calculate_angles(joint)
                feature_vector = np.concatenate([joint.flatten(), angle])
                seq.append(feature_vector)
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                if len(seq) < seq_length:
                    continue
                input_data = torch.tensor(np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0))
                # 손동작 인식
                action, action_seq = recognize_action(model, input_data, actions, action_seq, device)
                # 손동작 인식 후 화면에 제스처 출력
                if action is not None:
                    mode = action

        # 현재 모드 화면에 출력
        cv2.putText(img, f'Mode : {mode}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, 
                    (0, 0, 255) if mode != "None" else (128, 128, 128), 3)

        cv2.imshow('img', img)  # ROI가 그려진 이미지만 'img'에 표시

        # ESC 키를 누르면 종료
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #서버에 연결
    device = torch.device('cuda')

    # Hyperparameters
    features = 99
    num_classes = 4

    model = LSTMModel(input_size=features, hidden_size=64, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('innopia_code/models/gesture_model.pth'))
    model.to(device)

    actions = ['enter', 'setting', 'power on', 'power off']
    seq_length = 30

    cap = cv2.VideoCapture(0)
    width, height = 640, 480
    print(f'width: {width}, height: {height}')
    process_video(cap, model, actions, seq_length, width, height, device)
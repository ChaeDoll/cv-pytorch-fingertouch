import time
import cv2
import numpy as np
import torch
from collections import deque

# --- 손가락 탭 감지 클래스 ---
class FingerTapDetector:
    def __init__(self):
        self.index_y_history = deque(maxlen=10)
        self.tap_threshold = 20
        self.stability_threshold = 3
        self.stability_check_delay = 7
        self.stability_required = 5
        self.tap_attempt_timeout = self.stability_check_delay + self.stability_required + 10

        self.stability_count = 0
        self.tap_attempt_frames = 0
        self.tap_tracking = False
        self.tap_positions = []
        self.display_time = 3.0

    def calculate_velocity(self):
        if len(self.index_y_history) < 2:
            return 0
        return self.index_y_history[-1] - self.index_y_history[-2]

    def detect_tap(self, frame, hand_landmarks, width, height):
        index_finger_tip = hand_landmarks.landmark[8]
        x, y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)
        self.index_y_history.append(y)
        current_velocity = self.calculate_velocity()

        cv2.putText(frame, f"Velocity: {current_velocity:.1f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if not self.tap_tracking and current_velocity > self.tap_threshold:
            self.tap_tracking = True
            self.stability_count = 0
            self.tap_attempt_frames = 0
            print("Tap attempt detected")

        elif self.tap_tracking:
            self.tap_attempt_frames += 1
            if self.tap_attempt_frames <= self.stability_check_delay:
                cv2.putText(frame, "Waiting for finger to land...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                if abs(current_velocity) < self.stability_threshold:
                    self.stability_count += 1
                    cv2.putText(frame, f"Stability: {self.stability_count}/{self.stability_required}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if self.stability_count >= self.stability_required:
                        self.tap_positions.append((x, y, time.time()))
                        print(f"Tap completed at ({x}, {y})")
                        self.tap_tracking = False
                else:
                    self.stability_count = 0
                    cv2.putText(frame, "Unstable motion", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if abs(current_velocity) > self.tap_threshold * 0.8:
                        print("Tap attempt canceled due to fast movement")
                        self.tap_tracking = False

            if self.tap_attempt_frames > self.tap_attempt_timeout:
                print("Tap attempt timeout")
                self.tap_tracking = False

        now = time.time()
        active_taps = []
        for tap_x, tap_y, tap_time in self.tap_positions:
            if now - tap_time < self.display_time:
                cv2.circle(frame, (tap_x, tap_y), 30, (0, 0, 255), 2)
                cv2.putText(frame, f'({tap_x}, {tap_y})', (tap_x + 10, tap_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"{self.display_time - (now - tap_time):.1f}s", (tap_x - 20, tap_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                active_taps.append((tap_x, tap_y, tap_time))
        self.tap_positions = active_taps

# --- 각도 계산 ---
def calculate_angles(joint):
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]  # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]  # Child joint
    v = v2 - v1  # [20, 3]
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # Normalize
    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
    return np.degrees(angle)

# --- 제스처 인식 ---
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

# ------------------------------------------------------------

class Button:
    def __init__(self, pos, size, text):
        self.pos = pos  # [x, y] 중앙 좌표
        self.size = size  # [width, height] 크기
        self.text = text  # 버튼에 표시할 텍스트

def draw(img, buttons, width, height, size_ratio, position_ratio):
    for button in buttons:
        # 버튼 위치 및 크기를 비율로 조정
        x = round(button.pos[0] * position_ratio)
        y = round(button.pos[1] * position_ratio)
        w = round(button.size[0] * size_ratio)
        h = round(button.size[1] * size_ratio)
        
        # 화면 밖으로 나가지 않도록 좌표 보정
        rect_x1 = max(0, x - w // 2)
        rect_y1 = max(0, y - h // 2)
        rect_x2 = min(width, x + w // 2)
        rect_y2 = min(height, y + h // 2)
        
        # 사각형 그리기
        cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), thickness=2)
        
        # 텍스트 그리기 (텍스트가 버튼 왼쪽 상단에 배치되도록 조정)
        padding = 10  # 왼쪽 상단에서 떨어진 위치
        text_x = rect_x1 + padding
        text_y = rect_y1 + padding + cv2.getTextSize(button.text, cv2.FONT_HERSHEY_PLAIN, 3, 2)[0][1]  # 높이 고려
        
        cv2.putText(img, button.text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return img

def draw_legend(img, key_map, width, height, size_ratio):
    overlay = img.copy()
    output = img.copy()

    # 동적으로 텍스트의 최대 너비와 텍스트 개수에 따라 높이를 계산
    max_text_width = 0
    text_height = 0
    padding = 20
    line_height = 25  # 한 줄당 높이
    
    # 텍스트 크기 측정하여 최대 너비와 총 높이 계산
    for k, v in key_map.items():
        text_size = cv2.getTextSize(f'{k} : {v}', cv2.FONT_HERSHEY_PLAIN, round(2 * size_ratio), 2)[0]
        max_text_width = max(max_text_width, text_size[0])
        text_height += line_height

    # 사각형 너비 및 높이 동적으로 설정 (텍스트의 최대 길이와 줄 수에 맞춰)
    rect_width = max_text_width + padding * 2  # 텍스트 길이에 맞춰 사각형 너비 결정
    rect_height = text_height + padding  # 텍스트 줄 수에 맞춰 사각형 높이 결정

    # 사각형 좌상단 및 우하단 좌표 설정
    rect_x1 = 0
    rect_y1 = 0
    rect_x2 = min(width, rect_x1 + rect_width)
    rect_y2 = min(height, rect_y1 + rect_height)

    # 사각형 그리기
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)

    # 텍스트 출력
    i = 0
    for k, v in key_map.items():
        text_x = rect_x1 + padding
        text_y = rect_y1 + padding + (i * line_height) + round(2 * size_ratio * 10)  # 줄 간격 고려
        cv2.putText(overlay, f'{k} : {v}', (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, round(2 * size_ratio), (0, 255, 0), 2)
        i += 1

    # 반응형으로 설정된 사각형과 텍스트를 화면에 그리기
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    return output

def draw_input(img, text, width, height, size_ratio):
    overlay = img.copy()
    output = img.copy()

    # 텍스트가 포함된 사각형을 그리는 부분 (화면 크기에 맞춰 가변적으로 조정)
    rect_x1 = max(0, round(80 / 100 * width))
    rect_y1 = max(0, round(90 / 100 * height))
    rect_x2 = min(width, round(99 / 100 * width))
    rect_y2 = min(height, round(95 / 100 * height))

    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, round(2 * size_ratio), 2)[0]
    text_x = max(0, min(round(85 / 100 * width), width - text_size[0]))
    text_y = max(text_size[1], min(round(93 / 100 * height), height))
    
    cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, round(2 * size_ratio), (0, 255, 0), 2)

    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    return output

def display_gesture(img, gesture_text, width, height, size_ratio = 1.0, padding = 20):
    output = img.copy()

    # 텍스트의 크기 계산
    text_size = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 2 * size_ratio, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    # 왼쪽 하단 위치 계산 (화면에서 일정 간격 떨어짐)
    text_x = padding
    text_y = height - padding

    # 인식된 제스처 텍스트 출력
    cv2.putText(output, gesture_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2 * size_ratio, (0, 255, 0), 2)

    return output

def display_click_status(img, click_status, width, height, size_ratio = 1.0):
    output = img.copy()

    # 텍스트의 크기 계산
    text_size = cv2.getTextSize(click_status, cv2.FONT_HERSHEY_SIMPLEX, 2 * size_ratio, 2)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    # 왼쪽 하단 위치 계산 (화면에서 일정 간격 떨어짐)
    padding = 10
    text_x = padding
    text_y = round(height*0.92) #height - padding

    # 인식된 제스처 텍스트 출력
    cv2.putText(output, click_status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2 * size_ratio, (0, 255, 0), 2)

    return output

def vertical_symmetry(x, y, height):
    new_y = height - y
    return x, new_y

def horizontal_symmetry(x, y, width):
    new_x = width - x
    return new_x, y

def drawROI(img, corners, size_ratio):
    cpy = img.copy()

    c1 = [(192, 192, 255), (192, 102, 255), (102, 192, 255), (192, 255, 255)]
    c2 = (128, 128, 255)

    # ROI를 그리는 코드 (화면 크기에 맞춰 조정)
    for i, pt in enumerate(corners):
        cv2.circle(cpy, tuple(pt.astype(int)), round(25 * size_ratio), c1[i], -1, cv2.LINE_AA)

    cv2.line(cpy, tuple(corners[0].astype(int)), tuple(corners[1].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[1].astype(int)), tuple(corners[2].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[2].astype(int)), tuple(corners[3].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[3].astype(int)), tuple(corners[0].astype(int)), c2, 2, cv2.LINE_AA)

    return cv2.addWeighted(img, 0.3, cpy, 0.7, 0)

def convert_position(pt1, pt2, pers):
    transformed_pt1 = np.dot(pers, pt1)
    transformed_pt2 = np.dot(pers, pt2)
    
    # 변환된 좌표 계산 후 반환
    return (round(transformed_pt1[0]), round(transformed_pt1[1])), (round(transformed_pt2[0]), round(transformed_pt2[1]))

# 손가락이 사각형 안에 있는지 확인하는 함수
def is_finger_in_rectangle(finger_pos, button):
    finger_x, finger_y = finger_pos
    x, y = button.pos
    w, h = button.size
    # 좌상단 및 우하단 좌표 계산
    top_left = (x - w // 2, y - h // 2)
    bottom_right = (x + w // 2, y + h // 2)
    
    top_left_x, top_left_y = top_left
    bottom_right_x, bottom_right_y = bottom_right

    return top_left_x <= finger_x <= bottom_right_x and top_left_y <= finger_y <= bottom_right_y

def validate_corners(corners, y_threshold=10):
    # 좌표를 분리
    top_left, top_right, bottom_left, bottom_right = corners
    
    # y축 값 비교
    top_y_similar = abs(top_left[1] - top_right[1]) <= y_threshold
    bottom_y_similar = abs(bottom_left[1] - bottom_right[1]) <= y_threshold
    
    # x축 값 비교 (상단의 길이가 하단의 길이보다 길어야 함)
    top_width = abs(top_right[0] - top_left[0])
    bottom_width = abs(bottom_right[0] - bottom_left[0])
    width_condition = top_width > bottom_width
    
    # 모든 조건을 만족해야 True
    return top_y_similar and bottom_y_similar and width_condition

def find_green_corners(img):
    lower_green = np.array([30, 65, 65])
    upper_green = np.array([85, 255, 255])

    # HSV 색 공간으로 변환 및 초록색 필터링
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 엣지 검출 및 윤곽선 찾기
    edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    try:
        # 가장 큰 윤곽선 찾기
        largest_contour = max(contours, key=cv2.contourArea)

        # 사다리꼴 꼭짓점 근사화
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # 꼭짓점이 4개인 경우만 처리
        if len(approx) == 4:
            corners = [tuple(pt[0]) for pt in approx]

            # 좌상단, 우상단, 좌하단, 우하단 순서로 정렬
            corners = sorted(corners, key=lambda x: (x[1], x[0]))  # y값 기준으로 정렬
            top_points = sorted(corners[:2], key=lambda x: x[0])
            bottom_points = sorted(corners[2:], key=lambda x: x[0])
            sorted_corners = [top_points[0], top_points[1], bottom_points[0], bottom_points[1]]

            # 꼭짓점 출력
            for idx, point in enumerate(sorted_corners):
                cv2.circle(img, point, 10, (0, 0, 255), -1)
                cv2.putText(img, f'{idx}', point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return sorted_corners, img
        else:
            return [], img
    except:
        return None, img
    
def convert_position(point, pers):
    # 변환된 동차 좌표 계산
    transformed_point = np.dot(pers, np.array([point[0], point[1], 1]))

    # 변환된 유클리드 좌표 계산 (동차좌표를 유클리드 좌표로 변환)
    transformed_point = transformed_point / transformed_point[2]
    transformed_x, transformed_y = round(transformed_point[0]), round(transformed_point[1])

    return transformed_x, transformed_y
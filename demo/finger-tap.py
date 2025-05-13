import cv2
import mediapipe as mp
import time
from collections import deque

class FingerTapDetector:
    def __init__(self):
        # MediaPipe 핸드 모듈 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 손가락 탭 감지 관련 변수
        self.index_y_history = deque(maxlen=10)  # 검지 y좌표 기록
        self.tap_threshold = 30  # 탭으로 인식할 속도 임계값 (양수: 아래로 내려가는 속도)
        # 안정 상태 관련 변수
        self.stability_threshold = 3  # 안정 상태로 판단할 움직임 임계값
        self.stability_check_delay = 7  # 안정 상태 확인 시작 전 대기 프레임 수
        self.stability_count = 0  # 안정 상태 유지 프레임 카운트
        self.stability_required = 5  # 탭으로 인식하기 위한 최소 안정 프레임 수
        # 탭 시도 타임아웃 관련
        self.tap_attempt_frames = 0  # 탭 시도 후 경과한 프레임 수 
        self.tap_attempt_timeout = self.stability_check_delay + self.stability_required + 10  # 탭 시도 최대 프레임 수
        # 상태 변수
        self.tap_tracking = False  # 탭 시도 감지 중인지 여부
        self.tap_positions = []  # 탭 위치 저장 [(x, y, timestamp), ...]
        self.display_time = 3.0  # 원 표시 시간 (초)
    
    def calculate_velocity(self):
        """최근 y좌표 기록을 기반으로 속도 계산"""
        if len(self.index_y_history) < 2:
            return 0
        
        # 최근 두 프레임 간의 y좌표 변화량 계산
        return self.index_y_history[-1] - self.index_y_history[-2]
    
    def detect_tap(self, frame):
        """현재 프레임에서 손 감지 및 탭 동작 인식"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        height, width, _ = frame.shape
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 손 랜드마크 그리기
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # 검지 끝 좌표 (8번 랜드마크)
                index_finger_tip = hand_landmarks.landmark[8]
                x, y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)
                
                # 검지 끝 표시
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                
                # y좌표 기록 및 속도 계산
                self.index_y_history.append(y)
                current_velocity = self.calculate_velocity()
                
                # 화면에 속도 표시
                cv2.putText(frame, f"Velocity: {current_velocity:.1f}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # 탭 감지 로직
                if not self.tap_tracking and current_velocity > self.tap_threshold:
                    # 빠른 하강 감지 -> 탭 시도 시작
                    self.tap_tracking = True
                    self.stability_count = 0
                    self.tap_attempt_frames = 0
                    print("Tap attempt detected")
                
                elif self.tap_tracking:
                    # 탭 시도 프레임 카운트 증가
                    self.tap_attempt_frames += 1
                    
                    # 안정 상태 확인을 지연시킴 (손가락이 바닥에 닿을 때까지 대기)
                    if self.tap_attempt_frames <= self.stability_check_delay:
                        # 대기 중에는 상태 정보만 표시
                        cv2.putText(frame, "Waiting for finger to land...", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        # 안정 상태 확인 시작
                        if abs(current_velocity) < self.stability_threshold:
                            self.stability_count += 1
                            # 디버그 정보 표시
                            cv2.putText(frame, f"Stability: {self.stability_count}/{self.stability_required}", 
                                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            # 충분한 시간 동안 안정 상태 유지 시 탭으로 인식
                            if self.stability_count >= self.stability_required:
                                self.tap_positions.append((x, y, time.time()))
                                print(f"Tap completed at ({x}, {y})")
                                self.tap_tracking = False
                        else:
                            # 안정 상태가 아니면 카운트 리셋
                            self.stability_count = 0
                            # 디버그 정보 표시
                            cv2.putText(frame, "Unstable motion", (50, 100), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            # 너무 빠르게 움직이면 탭 시도 취소
                            if abs(current_velocity) > self.tap_threshold * 0.8:
                                print("Tap attempt canceled due to fast movement")
                                self.tap_tracking = False

                    # 탭 시도 타임아웃 체크
                    if self.tap_attempt_frames > self.tap_attempt_timeout:
                        print("Tap attempt timeout")
                        self.tap_tracking = False

        else:
            # 손이 감지되지 않으면 탭 추적 중단
            self.tap_tracking = False
        
        # 인식된 탭 위치에 원 표시
        current_time = time.time()
        active_taps = []
        
        for tap_x, tap_y, tap_time in self.tap_positions:
            if current_time - tap_time < self.display_time:
                # 표시 시간 내의 탭은 유지
                cv2.circle(frame, (tap_x, tap_y), 30, (0, 0, 255), 2)
                
                # 남은 시간 표시
                remaining = self.display_time - (current_time - tap_time)
                cv2.putText(frame, f"{remaining:.1f}s", (tap_x - 20, tap_y - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                active_taps.append((tap_x, tap_y, tap_time))
        
        # 활성 탭 목록 업데이트
        self.tap_positions = active_taps
        
        return frame

def main():
    cap = cv2.VideoCapture(1)
    detector = FingerTapDetector()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("카메라를 열지 못했습니다.")
            break
        
        # 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)
        
        # 탭 감지 및 시각화
        processed_frame = detector.detect_tap(frame)
        
        # 결과 표시
        cv2.imshow('Finger Tap Detection', processed_frame)
        
        # ESC 키를 누르면 종료
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """세 점 사이의 각도를 계산 (a: 시작, b: 중점, c: 끝)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

cap = cv2.VideoCapture(0)

# 상태 변수
recording = False
feedback_ready = False
frames_buffer = []  # 동작 데이터 저장
status_text = "Waiting for User..."

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 이미지 처리
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 1. 어깨 너비로 거리 측정 (가까이 왔는지 확인)
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        dist = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))

        # 2. 발차기 감지 로직 (발목이 골반보다 높을 때)
        hip_height = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        ankle_height = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        
        # 발차기 시작: 발이 골반 위로 올라감 & 카메라와 멀리 있음
        if ankle_height < hip_height and dist < 0.2:
            if not recording:
                recording = True
                frames_buffer = [] # 새 녹화 시작
                status_text = "Kicking Detected! Analyzing..."

        # 녹화 중 데이터 수집
        if recording:
            frames_buffer.append(landmarks)
            # 발이 다시 내려오면 녹화 종료
            if ankle_height > hip_height + 0.1: 
                recording = False
                feedback_ready = True
                status_text = "Analysis Done. Come closer to see feedback!"

        # 3. 피드백 모드 전환 (카메라에 가까이 다가오면 실행)
        if feedback_ready and dist > 0.4: # 어깨가 화면에 크게 보이면
            status_text = "FEEDBACK: Check your knee angle!"
            # 여기서 frames_buffer를 분석하여 화면에 피드백 텍스트를 띄움
            # 예: max_angle = max([calculate_angle(...) for l in frames_buffer])
            
            # 피드백 확인 후 초기화 (3초 뒤)
            cv2.putText(image, "Great Kick!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            # feedback_ready = False # 주석 해제 시 1회 노출 후 초기화

        # 스켈레톤 그리기
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 상단 상태바 표시
    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Taekwondo Coach', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
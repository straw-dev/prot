import cv2
import mediapipe as mp

# 1. 기초 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# model_complexity: 0(빠름), 1(보통), 2(정확함 - GPU 권장)
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

vid_path = input("path: ")
if vid_path=='0':
    vid_path = int(vid_path)
else:
    vid_path = vid_path.strip().replace('"','')

cap = cv2.VideoCapture(vid_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 2. 이미지 전처리
    frame = cv2.flip(frame, 1) # 거울 모드
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    # 3. 랜드마크 출력 및 좌표 추출
    if results.pose_landmarks:
        # 스켈레톤(뼈대) 그리기
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # 점 설정
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2) # 선 설정
        )

        # 예시: 오른쪽 어깨 좌표만 따로 가져오고 싶을 때
        # landmarks = results.pose_landmarks.landmark
        # r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
        #               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

    # 4. 결과창 출력
    cv2.imshow('Full Body Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp

# 1. 사용할 모듈 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils # 선을 그려주는 도구
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. 카메라 연결
cap = cv2.VideoCapture(0)

print("자세 감지를 시작합니다. 종료하려면 'q'를 누르세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 성능을 위해 이미지 형식을 변경 (BGR -> RGB)
    # MediaPipe는 RGB 이미지를 사용합니다.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # 4. 자세 감지 실행
    results = pose.process(image)

    # 5. 다시 화면에 그리기 위해 설정 변경 (RGB -> BGR)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 6. 결과(랜드마크)가 있다면 화면에 그리기
    if results.pose_landmarks:
        # 인체 관절 포인트와 연결선을 화면에 그림
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # 점 색상
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)  # 선 색상
        )

    # 7. 결과 출력
    cv2.imshow('Mediapipe Pose Detection', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
pose.close()
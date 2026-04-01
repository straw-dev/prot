import cv2
import mediapipe as mp

# 아래처럼 solutions를 직접 임포트해 보세요
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# MediaPipe Pose 및 그리기 도구 초기화
'''mp_pose = mp.solutions.pose
 mp_drawing = mp.solutions.drawing_utils'''
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 노트북 내장 카메라 연결 (0번은 기본 내장 카메라)
cap = cv2.VideoCapture(0)

print("프로그램을 종료하려면 'q'를 누르세요.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        break

    # 성능을 위해 이미지 쓰기 불가능 설정 및 BGR에서 RGB로 변환
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 자세 감지 수행
    results = pose.process(image)

    # 다시 화면에 그리기 위해 RGB에서 BGR로 변환 및 쓰기 가능 설정
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 랜드마크가 감지된 경우 화면에 그리기
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    # 결과 영상 출력
    cv2.imshow('MediaPipe Real-time Pose Detection', image)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
pose.close()
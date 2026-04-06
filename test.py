import cv2
import mediapipe as mp

# 사용할 모듈 정의
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

print("✅ 미디어파이프 로드 성공! 카메라를 켭니다...")

# 카메라 테스트
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # 이미지 전처리 (좌우 반전 및 RGB 변환)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # 다시 BGR로 변환하여 화면 표시 준비
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 손이 감지되면 그리기
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Hands Test', image)
    
    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import sys
import os

import sys
import os

# 1. 아까 'pip show'에서 확인한 Location 경로를 아래에 정확히 넣어줘
# (예: r"C:\Users\i\AppData\Local\Programs\Python\Python311\Lib\site-packages")
real_path = r"여기에_아까_복사한_경로를_넣어줘"

if real_path not in sys.path:
    sys.path.insert(0, real_path)

import mediapipe as mp

# 이제 확인해보자
try:
    print("✅ 버전:", mp.__version__)
    print("✅ 솔루션 체크:", hasattr(mp, 'solutions'))
    mp_hands = mp.solutions.hands
    print("🔥 드디어 성공! 이제 친구들이랑 작업 계속해!")
except Exception as e:
    print(f"❌ 이래도 안된다고? 에러 내용: {e}")
    
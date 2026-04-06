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

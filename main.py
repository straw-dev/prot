import cv2
import mediapipe as mp

print("OpenCV 버전:", cv2.__version__)
print("Mediapipe 로드 성공!")

# 간단하게 카메라가 켜지는지만 확인
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 찾을 수 없습니다. (웹캠 연결 확인!)")
else:
    print("카메라 연결 성공! 'q'를 누르면 종료됩니다.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
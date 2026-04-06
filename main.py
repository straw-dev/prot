import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

def main():
    while True:
        attempt = 0
        success, img = cap.read()
        while not success and attempt <5:
            time.sleep(0.2)
            success, img = cap.read()
            attempt +=1
        if not success:
            print(f"failid to read frame after {attempt} attemps.")
            break
        img = cv2.flip(img, 1)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
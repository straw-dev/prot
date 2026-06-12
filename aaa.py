import cv2
import mediapipe as mp
import numpy as np
import math
import random

# --- 1. 초기 설정 및 변수 선언 ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 웹캠 설정
cap = cv2.VideoCapture(0)
cam_width, cam_height = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

# 이미지 로드 및 초기화
img_path = 'character.png'
img_original = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if img_original is None:
    print(f"'{img_path}' 파일을 찾을 수 없어 임의의 사각형으로 대체합니다.")
    img_original = np.zeros((150, 150, 4), dtype=np.uint8)
    cv2.rectangle(img_original, (0, 0), (150, 150), (255, 200, 100, 255), -1)

# 캐릭터 기본 크기 설정
orig_h, orig_w = img_original.shape[:2]
scale_factor = 150 / max(orig_h, orig_w)
img_base = cv2.resize(img_original, (int(orig_w * scale_factor), int(orig_h * scale_factor)))
base_h, base_w = img_base.shape[:2]

# 캐릭터 상태 및 물리 변수
current_w, current_h = base_w, base_h
obj_cx, obj_cy = cam_width // 2, cam_height // 2
vx, vy = 0, 0       
friction = 0.98     
return_speed = 0.15 

# 🎯 점수 및 타겟(사각형) 변수 추가
score = 0
target_size = 120  # 목표 사각형의 크기
target_x = 0
target_y = 0

# 목표 사각형을 화면 내 랜덤한 위치로 재배치하는 함수
def respawn_target():
    global target_x, target_y
    # 사각형이 화면 바깥으로 잘리지 않도록 마진을 두고 랜덤 배치
    target_x = random.randint(100, cam_width - 100 - target_size)
    target_y = random.randint(150, cam_height - 100 - target_size)

# 첫 번째 타겟 생성
respawn_target()

prev_index_pos = None
state = "IDLE"
init_finger_dist_x = 1.0
init_finger_dist_y = 1.0

def get_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def overlay_image(background, overlay, cx, cy):
    h, w = overlay.shape[:2]
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = x1 + w, y1 + h

    if x1 < 0 or y1 < 0 or x2 > background.shape[1] or y2 > background.shape[0]:
        return background

    sub_bg = background[y1:y2, x1:x2]
    
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        alpha_img = np.expand_dims(alpha, axis=2)
        sub_bg[:] = (1.0 - alpha_img) * sub_bg + alpha_img * overlay[:, :, :3]
    else:
        sub_bg[:] = overlay[:, :, :3]
        
    background[y1:y2, x1:x2] = sub_bg
    return background

# --- 2. 메인 루프 ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h_cam, w_cam, _ = frame.shape

    # 🎯 화면에 랜덤 목표 사각형 그리기 (노란색 사각형)
    cv2.rectangle(frame, (target_x, target_y), (target_x + target_size, target_y + target_size), (0, 255, 255), 3)
    cv2.putText(frame, "TARGET", (target_x, target_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hand_inside_idx = False
    obj_x1, obj_y1 = obj_cx - current_w // 2, obj_cy - current_h // 2
    obj_x2, obj_y2 = obj_cx + current_w // 2, obj_cy + current_h // 2

    target_w, target_h = base_w, base_h

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            landmarks = {}
            for idx, lm in enumerate(hand_landmarks.landmark):
                landmarks[idx] = (int(lm.x * w_cam), int(lm.y * h_cam))

            thumb = landmarks[4]
            index = landmarks[8]
            middle = landmarks[12]

            if obj_x1 < index[0] < obj_x2 and obj_y1 < index[1] < obj_y2:
                hand_inside_idx = True

            thumb_index_dist = get_dist(thumb, index)
            index_middle_dist = get_dist(index, middle)

            # --- 인터랙션 로직 ---
            if thumb_index_dist < 45 and hand_inside_idx and state != "STRETCH":
                state = "GRAB"
                obj_cx, obj_cy = index[0], index[1]
                vx, vy = 0, 0
                
            elif state == "GRAB" and thumb_index_dist >= 45:
                if hand_inside_idx:
                    state = "STRETCH"
                    init_finger_dist_x = max(abs(thumb[0] - index[0]), 5)
                    init_finger_dist_y = max(abs(thumb[1] - index[1]), 5)
                else:
                    state = "IDLE"
                    
            elif state == "STRETCH":
                current_dist_x = abs(thumb[0] - index[0])
                current_dist_y = abs(thumb[1] - index[1])
                
                target_w = int(base_w * (current_dist_x / init_finger_dist_x) * 1.5)
                target_h = int(base_h * (current_dist_y / init_finger_dist_y) * 1.5)
                
                target_w = np.clip(target_w, base_w, base_w * 4)
                target_h = np.clip(target_h, base_h, base_h * 4)
                
                current_w, current_h = target_w, target_h
                
                if thumb_index_dist > 220:
                    state = "IDLE"

            elif index_middle_dist < 35 and hand_inside_idx:
                state = "IDLE"
                if prev_index_pos is not None:
                    dx = index[0] - prev_index_pos[0]
                    dy = index[1] - prev_index_pos[1]
                    vx = dx * 1.5
                    vy = dy * 1.5
                else:
                    vx = (index[0] - obj_cx) * 1.0
                    vy = (index[1] - obj_cy) * 1.0

            prev_index_pos = index
    else:
        prev_index_pos = None

    # --- 3. 크기 원복 및 물리 연산 ---
    if state != "STRETCH":
        target_w, target_h = base_w, base_h
        current_w += int((target_w - current_w) * return_speed)
        current_h += int((target_h - current_h) * return_speed)

    current_w = max(current_w, 10)
    current_h = max(current_h, 10)

    img_display = cv2.resize(img_base, (current_w, current_h))

    if state == "IDLE":
        obj_cx += int(vx)
        obj_cy += int(vy)
        vx *= friction
        vy *= friction

        if obj_cx - current_w // 2 < 0 or obj_cx + current_w // 2 > cam_width:
            vx *= -0.9
            obj_cx = np.clip(obj_cx, current_w // 2, cam_width - current_w // 2)
        if obj_cy - current_h // 2 < 0 or obj_cy + current_h // 2 > cam_height:
            vy *= -0.9
            obj_cy = np.clip(obj_cy, current_h // 2, cam_height - current_h // 2)

    # 🎯 [핵심 추가] 충돌 검사 (캐릭터 중심점이 목표 사각형 내부에 들어왔는지 체크)
    if target_x < obj_cx < target_x + target_size and target_y < obj_cy < target_y + target_size:
        score += 1          # 점수 획득
        respawn_target()    # 타겟을 새로운 랜덤 위치로 이동

    # 합성 및 출력
    try:
        frame = overlay_image(frame, img_display, obj_cx, obj_cy)
    except Exception as e:
        pass

    # 🎯 UI 텍스트 출력 (상단에 스코어 보드 배치)
    cv2.putText(frame, f"SCORE: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, f"State: {state}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Interactive Character Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
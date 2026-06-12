import cv2
import mediapipe as mp
import numpy as np
import math

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

# 실시간 크기 및 물리 변수
current_w, current_h = base_w, base_h
obj_cx, obj_cy = cam_width // 2, cam_height // 2
vx, vy = 0, 0       # 이동 속도
friction = 0.98     # 마찰력을 살짝 줄여서 더 멀리, 시원하게 날아가도록 수정 (기존 0.95)
return_speed = 0.15 # 제자리로 돌아오는 속도 (말랑한 느낌 유지)

# 이전 프레임의 손가락 위치 저장용 (밀기 속도 정밀 계산용)
prev_index_pos = None

STATE_IDLE = "IDLE"
STATE_GRAB = "GRAB"
STATE_STRETCH = "STRETCH"
state = STATE_IDLE

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
            
            # [액션 1] 잡고 이동하기
            if thumb_index_dist < 45 and hand_inside_idx and state != STATE_STRETCH:
                state = STATE_GRAB
                obj_cx, obj_cy = index[0], index[1]
                vx, vy = 0, 0
                
            elif state == STATE_GRAB and thumb_index_dist >= 45:
                if hand_inside_idx:
                    state = STATE_STRETCH
                    # 늘리기 기준점을 타이트하게 잡아 반응성 극대화
                    init_finger_dist_x = max(abs(thumb[0] - index[0]), 5)
                    init_finger_dist_y = max(abs(thumb[1] - index[1]), 5)
                else:
                    state = STATE_IDLE
                    
            # [액션 2] 고무줄처럼 늘리기 (대폭 가속화)
            elif state == STATE_STRETCH:
                current_dist_x = abs(thumb[0] - index[0])
                current_dist_y = abs(thumb[1] - index[1])
                
                # 가속 계수(1.5)를 곱해 조금만 벌려도 팍팍 커지도록 수정
                target_w = int(base_w * (current_dist_x / init_finger_dist_x) * 1.5)
                target_h = int(base_h * (current_dist_y / init_finger_dist_y) * 1.5)
                
                # 최대 확대 제한 확대 (4배까지 가능)
                target_w = np.clip(target_w, base_w, base_w * 4)
                target_h = np.clip(target_h, base_h, base_h * 4)
                
                # 손가락을 늘리는 중에는 딜레이 없이 즉시 크기 반영
                current_w, current_h = target_w, target_h
                
                if thumb_index_dist > 220: # 더 크게 늘릴 수 있도록 한계치 상향
                    state = STATE_IDLE

            # [액션 3] 밀어서 날리기 (속도 대폭 상향)
            elif index_middle_dist < 35 and hand_inside_idx:
                state = STATE_IDLE
                if prev_index_pos is not None:
                    # 이전 프레임 대비 손가락의 움직임 변화량(속도)을 직접 계산
                    dx = index[0] - prev_index_pos[0]
                    dy = index[1] - prev_index_pos[1]
                    
                    # 튕겨나가는 힘의 가속도 증가
                    vx = dx * 1.5
                    vy = dy * 1.5
                else:
                    vx = (index[0] - obj_cx) * 1.0
                    vy = (index[1] - obj_cy) * 1.0

            prev_index_pos = index
    else:
        prev_index_pos = None

    # --- 3. 크기 원복 및 물리 연산 ---
    if state != STATE_STRETCH:
        # 늘리기 상태가 아닐 때만 보간(Lerp)을 이용해 천천히 돌아옴
        target_w, target_h = base_w, base_h
        current_w += int((target_w - current_w) * return_speed)
        current_h += int((target_h - current_h) * return_speed)

    current_w = max(current_w, 10)
    current_h = max(current_h, 10)

    img_display = cv2.resize(img_base, (current_w, current_h))

    # IDLE 상태일 때 날아가기 연산
    if state == STATE_IDLE:
        obj_cx += int(vx)
        obj_cy += int(vy)
        vx *= friction
        vy *= friction

        # 벽면 충돌 시 더 탱글하게 튕기도록 반발계수 조정
        if obj_cx - current_w // 2 < 0 or obj_cx + current_w // 2 > cam_width:
            vx *= -0.9
            obj_cx = np.clip(obj_cx, current_w // 2, cam_width - current_w // 2)
        if obj_cy - current_h // 2 < 0 or obj_cy + current_h // 2 > cam_height:
            vy *= -0.9
            obj_cy = np.clip(obj_cy, current_h // 2, cam_height - current_h // 2)

    # 합성 및 출력
    try:
        frame = overlay_image(frame, img_display, obj_cx, obj_cy)
    except Exception as e:
        pass

    cv2.putText(frame, f"State: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Interactive Character", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
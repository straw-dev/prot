import os
import cv2
import mediapipe as mp
from ultralytics import YOLO

# 윈도우 크기 정의
TARGET_WIDTH = 960
TARGET_HEIGHT = 540

# 1. YOLOv8 및 MediaPipe 초기화 (속도 최적화 세팅)
yolo_model = YOLO("yolov8n.pt") 

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# model_complexity를 1에서 0(가장 빠름)으로 하향 조정
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 동영상 경로 입력 받기
while True:
    video_path = input("분석할 동영상 파일의 경로를 입력하세요 (0 입력 시 웹캠): ").strip()
    if video_path == '0':
        video_path = 0
        break
    elif os.path.exists(video_path):
        break
    else:
        print("파일을 찾을 수 없습니다. 올바른 경로를 입력해주세요.\n")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    exit()

# 고정된 크기의 OpenCV 윈도우 생성
cv2.namedWindow("Select Target Number", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select Target Number", TARGET_WIDTH, TARGET_HEIGHT)
cv2.namedWindow("Taekwondo Pose Coaching", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Taekwondo Pose Coaching", TARGET_WIDTH, TARGET_HEIGHT)

# [단계 1] 첫 프레임 처리 (YOLO 초기화)
ret, first_frame = cap.read()
if not ret:
    print("영상을 읽을 수 없습니다.")
    cap.release()
    exit()

h_max, w_max, _ = first_frame.shape

# 첫 프레임 탐지 (imgsz=320으로 속도 업)
yolo_results = yolo_model.track(first_frame, persist=True, classes=0, imgsz=320, verbose=False)

detected_persons = {}
if yolo_results[0].boxes.id is not None:
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)
    
    selection_frame = first_frame.copy()
    for box, obj_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        detected_persons[obj_id] = (x1, y1, x2, y2)
        
        cv2.rectangle(selection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(selection_frame, f"No. {obj_id}", (x1 + 10, y1 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    resized_selection = cv2.resize(selection_frame, (TARGET_WIDTH, TARGET_HEIGHT))
    cv2.imshow("Select Target Number", resized_selection)
    cv2.waitKey(1)
    
    while True:
        try:
            selected_id = int(input(f"분석할 사람의 번호(No.)를 입력하세요 {list(detected_persons.keys())}: "))
            if selected_id in detected_persons:
                break
            else:
                print("목록에 있는 번호를 입력해주세요.")
        except ValueError:
            print("숫자만 입력 가능합니다.")
            
    cv2.destroyWindow("Select Target Number")
else:
    print("[알림] 사람이 감지되지 않았습니다.")
    exit()


# 💡 연산 주기를 조절하기 위한 변수 선언
frame_count = 0
SKIP_FRAMES = 4  # YOLO는 4프레임에 한 번만 실행 (원하는 대로 조절 가능)
last_target_box = None

# [단계 2] 실시간 영상 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    frame_count += 1
    
    # 💡 지정된 주기에 도달했거나, 이전에 기억된 상자가 없을 때만 YOLO 작동
    if frame_count % SKIP_FRAMES == 0 or last_target_box is None:
        # imgsz=320 옵션으로 추적 속도 가속화
        yolo_results = yolo_model.track(frame, persist=True, classes=0, imgsz=320, verbose=False)
        
        target_box = None
        if yolo_results[0].boxes.id is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, obj_id in zip(boxes, ids):
                if obj_id == selected_id:
                    target_box = list(map(int, box))
                    last_target_box = target_box  # 최근 위치 기억
                    break
    else:
        # YOLO를 건너뛰는 프레임에서는 직전에 저장된 상자 위치를 그대로 재사용
        target_box = last_target_box
                
    # 타겟 박스가 존재하는 경우 분석 진행
    if target_box is not None:
        tx1, ty1, tx2, ty2 = target_box
        tw, th = tx2 - tx1, ty2 - ty1
        
        # 마진 축소 (15% -> 10%로 줄여 크롭 이미지 크기를 작게 만듦 -> MediaPipe 연산 속도 향상)
        margin_w = int(tw * 0.10)
        margin_h = int(th * 0.10)
        
        y1, y2 = max(0, ty1 - margin_h), min(h_max, ty2 + margin_h)
        x1, x2 = max(0, tx1 - margin_w), min(w_max, tx2 + margin_w)
        
        roi_frame = frame[y1:y2, x1:x2]
        
        if roi_frame.size > 0:
            roi_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(roi_rgb)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    roi_frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )
                
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(display_frame, f"ANALYZING (ID: {selected_id})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        cv2.putText(display_frame, "TARGET LOST", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    resized_display = cv2.resize(display_frame, (TARGET_WIDTH, TARGET_HEIGHT))
    cv2.imshow("Taekwondo Pose Coaching", resized_display)
    
    # waitKey 값을 10에서 1로 낮춰 입출력 딜레이 최소화
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# 1. 분석할 동영상 경로 입력 받기
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

# 원본 영상 해상도 정보 가져오기
w_max = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_max = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 윈도우 최대 제한 크기 설정
MAX_WIDTH = 960
MAX_HEIGHT = 540

# ⭐ [추가] 원본 비율을 유지하면서 출력 창 크기 계산하기
# 가로 비율과 세로 비율 중 어느 쪽이 제한 크기에 먼저 걸리는지 계산
scale_w = MAX_WIDTH / w_max
scale_h = MAX_HEIGHT / h_max
scale = min(scale_w, scale_h)

# 최종적으로 팝업창이 가질 해상도 (원본 비율 유지됨)
TARGET_WIDTH = int(w_max * scale)
TARGET_HEIGHT = int(h_max * scale)

# OpenCV 윈도우 생성 및 계산된 크기로 설정
cv2.namedWindow('TaeKwonDo Side Kick Landmark', cv2.WINDOW_NORMAL)
cv2.resizeWindow('TaeKwonDo Side Kick Landmark', TARGET_WIDTH, TARGET_HEIGHT)

# 2. MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3. 데이터 저장 리스트 초기화
left_knee_angles = []
right_knee_angles = []
frames = []
frame_count = 0

# 4. 각도 계산 함수 정의
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    vector_ba = a - b
    vector_bc = c - b
    
    cosine_angle = np.dot(vector_ba, vector_bc) / (np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

print(f"\n[알림] 원본 해상도: {w_max}x{h_max} -> 출력 해상도: {TARGET_WIDTH}x{TARGET_HEIGHT}")
print("[알림] 동영상 분석을 시작합니다. 'q'를 누르면 중간에 종료할 수 있습니다.")

# 5. 동영상 재생 및 프레임별 각도 추출 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 원본 이미지 픽셀 좌표 기준 연산 (비율 왜곡 방지)
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w_max, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h_max]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w_max, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h_max]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w_max, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h_max]
        
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w_max, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h_max]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w_max, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h_max]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w_max, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h_max]
        
        l_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        r_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        left_knee_angles.append(l_knee_angle)
        right_knee_angles.append(r_knee_angle)
        frames.append(frame_count)
        
        annotated_image = frame.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 글자 크기(폰트 스케일)를 영상 출력 크기에 맞춰 유동적으로 조절
        font_scale = max(0.5, TARGET_WIDTH / 1000.0)
        thickness = max(1, int(TARGET_WIDTH / 500))

        cv2.putText(annotated_image, f"Left Knee: {int(l_knee_angle)} deg", 
                    (int(w_max*0.05), int(h_max*0.08)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(annotated_image, f"Right Knee: {int(r_knee_angle)} deg", 
                    (int(w_max*0.05), int(h_max*0.15)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        
        # ⭐ [수정] 원본 비율이 유지되도록 계산된 TARGET 크기로만 리사이즈
        resized_display = cv2.resize(annotated_image, (TARGET_WIDTH, TARGET_HEIGHT))
        cv2.imshow('TaeKwonDo Side Kick Landmark', resized_display)
        
    else:
        if left_knee_angles:
            left_knee_angles.append(left_knee_angles[-1])
            right_knee_angles.append(right_knee_angles[-1])
        else:
            left_knee_angles.append(0)
            right_knee_angles.append(0)
        frames.append(frame_count)
        
        # 랜드마크 검출 실패 시에도 원본 비율로 리사이즈 출력
        resized_display = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        cv2.imshow('TaeKwonDo Side Kick Landmark', resized_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[알림] 사용자에 의해 분석이 중단되었습니다.")
        break

cap.release()
cv2.destroyAllWindows()
pose.close()

# 6. Matplotlib 그래프 출력 부분 (이전과 동일)
if frames:
    plt.figure(figsize=(12, 6))
    plt.plot(frames, left_knee_angles, label='Left Knee Angle', color='blue')
    plt.plot(frames, right_knee_angles, label='Right Knee Angle', color='red', linestyle='--')

    max_l_angle = max(left_knee_angles)
    max_r_angle = max(right_knee_angles)
    
    if max_l_angle >= max_r_angle:
        peak_frame = frames[left_knee_angles.index(max_l_angle)]
        peak_angle = max_l_angle
        label_text = f'Left Knee Peak: {int(peak_angle)}°'
    else:
        peak_frame = frames[right_knee_angles.index(max_r_angle)]
        peak_angle = max_r_angle
        label_text = f'Right Knee Peak: {int(peak_angle)}°'

    plt.axvline(x=peak_frame, color='gray', linestyle=':', label='Kicking Peak Frame')
    plt.annotate(label_text, xy=(peak_frame, peak_angle), 
                 xytext=(peak_frame + (frame_count * 0.02), peak_angle - 15),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    plt.title('TaeKwonDo Side Kick Knee Angle Time-Series Analysis')
    plt.xlabel('Frame Number')
    plt.ylabel('Knee Angle (degrees)')
    plt.ylim(0, 190)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\n[알림] 영상 분석이 끝나고 각도 변화 그래프가 성공적으로 출력되었습니다.")

    #C:\Users\dimas\OneDrive\사진\Camera Roll\sidekick(1).mp4
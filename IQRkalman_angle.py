import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. 순수 수학 공식 기반 1차원 칼만 필터 클래스 정의
# ==========================================
class SimpleKalmanFilter:
    def __init__(self, process_noise=0.01, measurement_noise=0.5):
        self.Q = process_noise       # 프로세스 노이즈 (작을수록 변화가 부드러움)
        self.R = measurement_noise   # 측정 노이즈 (클수록 원본의 튀는 에러를 무시함)
        self.X_estimated = 0.0       # 최종 최적 추정값 (필터링된 각도)
        self.P = 1.0                 # 오차 공분산
        self.initialized = False

    def filter(self, measurement):
        if not self.initialized:
            self.X_estimated = measurement
            self.initialized = True
            return self.X_estimated

        # 1) 예측 (Predict)
        X_predicted = self.X_estimated
        P_predicted = self.P + self.Q

        # 2) 수정 및 업데이트 (Update)
        kalman_gain = P_predicted / (P_predicted + self.R)
        self.X_estimated = X_predicted + kalman_gain * (measurement - X_predicted)
        self.P = (1 - kalman_gain) * P_predicted

        return self.X_estimated

# ==========================================
# 0-1. [추가] IQR 기반 이상치 검출 및 제거 함수 정의
# ==========================================
IQR_MULTIPLIER = 1.5  # 이상치 판단 기준 (일반적으로 1.5 배수 사용)

def remove_outlier_by_iqr(current_angle, angle_history, diff_history):
    """
    최근 프레임 간 변화량의 IQR을 계산하여, 통계적 범위를 넘어서는 대형 노이즈를 직전 값으로 보정
    """
    # 이전에 기록된 최종 각도가 없으면(첫 프레임) 그대로 반환
    if not angle_history:
        return current_angle
        
    # 직전 최종 각도와의 변화량(절댓값) 계산
    current_diff = abs(current_angle - angle_history[-1])
    
    # 데이터가 아직 부족할 때는(초반 5프레임 미만) 기록 수집을 위해 그냥 통과
    if len(diff_history) < 5:
        diff_history.append(current_diff)
        return current_angle
        
    # 최근 5개 변화량에 대한 Q1, Q3 및 IQR 연산
    q1 = np.percentile(diff_history, 25)
    q3 = np.percentile(diff_history, 75)
    iqr = q3 - q1
    
    # 이상치 판단의 임계 상한선 설정
    upper_bound = q3 + (IQR_MULTIPLIER * iqr)
    
    # 태권도 동작의 고속 특성을 감안해 최소 변화 한계선(15도) 보장 (IQR이 극도로 작아 오류나는 것 방지)
    if upper_bound < 15.0:
        upper_bound = 15.0
        
    # 만약 현재 변화량이 통계적 상한선을 초과했다면 '뒷사람 간섭 등에 의한 이상치'로 판단
    if current_diff > upper_bound:
        # 현재의 에러 데이터를 버리고, 가장 믿을 수 있는 '직전 프레임의 각도'를 임시 유지
        return angle_history[-1]
    else:
        # 정상 움직임일 경우 변화량 기록을 업데이트하고 현재 각도를 통과시킴
        diff_history.append(current_diff)
        if len(diff_history) > 5:
            diff_history.pop(0)  # 최근 5개만 유지하는 슬라이딩 윈도우 구조
        return current_angle

# ==========================================

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

# 원본 비율을 유지하면서 출력 창 크기 계산하기
scale_w = MAX_WIDTH / w_max
scale_h = MAX_HEIGHT / h_max
scale = min(scale_w, scale_h)

TARGET_WIDTH = int(w_max * scale)
TARGET_HEIGHT = int(h_max * scale)

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

# ⭐ [추가] IQR 필터에 사용될 다리별 최근 변화량 저장소 리스트
left_diffs = []
right_diffs = []

# ⭐ [최적화] 칼만 필터 인스턴스 생성 및 피드백 주신 검증된 R 값(0.01) 적용
kf_left = SimpleKalmanFilter(process_noise=0.01, measurement_noise=0.01)
kf_right = SimpleKalmanFilter(process_noise=0.01, measurement_noise=0.01)

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
        
        # 원본 이미지 픽셀 좌표 기준 연산
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w_max, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h_max]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w_max, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h_max]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w_max, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h_max]
        
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w_max, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h_max]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w_max, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h_max]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w_max, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h_max]
        
        # [1단계] 미필터링 순수 원본 각도 계산
        raw_l_angle = calculate_angle(left_hip, left_knee, left_ankle)
        raw_r_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # [2단계] ⭐ [IQR 필터 통과] 툭 떨어지거나 튀는 초대형 이상치 아웃
        iqr_l_angle = remove_outlier_by_iqr(raw_l_angle, left_knee_angles, left_diffs)
        iqr_r_angle = remove_outlier_by_iqr(raw_r_angle, right_knee_angles, right_diffs)
        
        # [3단계] ⭐ [칼만 필터 통과] 통과된 데이터의 미세 떨림 보정 및 고속 트래킹 감지
        l_knee_angle = kf_left.filter(iqr_l_angle)
        r_knee_angle = kf_right.filter(iqr_r_angle)
        
        left_knee_angles.append(l_knee_angle)
        right_knee_angles.append(r_knee_angle)
        frames.append(frame_count)
        
        annotated_image = frame.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        font_scale = max(0.5, TARGET_WIDTH / 1000.0)
        thickness = max(1, int(TARGET_WIDTH / 500))

        # 화면에도 최종 하이브리드 필터(IQR + 칼만)를 거친 결과가 출력됩니다.
        cv2.putText(annotated_image, f"Left Knee: {int(l_knee_angle)} deg", 
                    (int(w_max*0.05), int(h_max*0.08)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(annotated_image, f"Right Knee: {int(r_knee_angle)} deg", 
                    (int(w_max*0.05), int(h_max*0.15)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        
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
        
        resized_display = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        cv2.imshow('TaeKwonDo Side Kick Landmark', resized_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[알림] 사용자에 의해 분석이 중단되었습니다.")
        break

cap.release()
cv2.destroyAllWindows()
pose.close()

# 6. Matplotlib 그래프 출력 부분
if frames:
    plt.figure(figsize=(12, 6))
    plt.plot(frames, left_knee_angles, label='Left Knee Angle (IQR + Kalman)', color='blue')
    plt.plot(frames, right_knee_angles, label='Right Knee Angle (IQR + Kalman)', color='red', linestyle='--')

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

    plt.title('TaeKwonDo Knee Angle Analysis (IQR Outlier Filtering & Kalman Filter)')
    plt.xlabel('Frame Number')
    plt.ylabel('Knee Angle (degrees)')
    plt.ylim(0, 190)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\n[알림] 영상 분석이 끝나고 IQR과 칼만 필터가 모두 결합된 각도 변화 그래프가 성공적으로 출력되었습니다.")
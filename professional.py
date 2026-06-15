import cv2
import mediapipe as mp
import math
import os
import matplotlib.pyplot as plt

def calculate_angle(p1, p2, p3):
    """
    p2(중심축)를 원점으로 평행이동한 뒤, atan2를 이용하여 p1과 p3 사이의 사잇각을 구합니다.
    """
    theta1 = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    theta2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
    
    angle = math.degrees(abs(theta2 - theta1))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def calculate_torso_angle(shoulder, hip):
    """
    어깨와 골반 중심을 잇는 벡터가 가상 수평선(X축)과 이루는 동경 각도를 구합니다.
    """
    angle = math.degrees(math.atan2(hip[1] - shoulder[1], hip[0] - shoulder[0]))
    return abs(angle)

def analyze_expert_video_final():
    # 1. 영상 파일 경로 입력 받기
    video_path = input("분석할 전문가 영상 파일 경로를 입력하세요: ").strip('"')
    if not os.path.exists(video_path):
        print("파일이 존재하지 않습니다. 경로를 다시 확인해주세요.")
        return

    # 2. OpenCV 및 MediaPipe 초기화
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # 데이터 저장 및 시각화용 리스트
    frame_numbers = []
    knee_angles = []
    torso_angles = []
    
    lowest_y_value = 1.0  # MediaPipe Y축은 위로 갈수록 0에 수렴
    strike_frame = 0
    active_foot = "Unknown"
    strike_frame_image = None # 타격 순간의 원본 프레임 이미지를 저장할 변수

    print("\n--- [1단계] 실시간 영상 분석 및 데이터 추출을 시작합니다 ---")
    print("※ 영상 분석 창에서 'q' 키를 누르면 중간에 중단할 수 있습니다.")

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # 좌우 관절 추출
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]

            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]

            # 주 수행발 판별 및 데이터 계산
            if landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y < landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y:
                knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                torso_angle = calculate_torso_angle(l_shoulder, l_hip)
                ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                current_foot = "Left"
            else:
                knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                torso_angle = calculate_torso_angle(r_shoulder, r_hip)
                ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                current_foot = "Right"

            # 리스트에 프레임별 데이터 적재
            frame_numbers.append(frame_count)
            knee_angles.append(knee_angle)
            torso_angles.append(torso_angle)

            # 저번 버전처럼 부드러운 재생 중 실시간 관절 오버레이 시각화
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Frame: {frame_count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Knee Angle: {knee_angle:.1f} deg", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Torso Angle: {torso_angle:.1f} deg", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 최대 타격 순간 탐색 및 해당 프레임 화면 이미지 '박제' 복사본 저장
            if ankle_y < lowest_y_value:
                lowest_y_value = ankle_y
                strike_frame = frame_count
                active_foot = current_foot
                # 나중에 띄워줄 수 있도록 타격 순간의 랜드마크가 그려진 화면 프레임을 통째로 복사해서 저장해둡니다.
                strike_frame_image = frame.copy()

        # 실시간 화면 출력 (저번 방식 복구)
        cv2.imshow("Real-time Taekwondo Analysis Window", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            print("\n[알림] 사용자에 의해 분석이 중단되었습니다.")
            break

    cap.release()
    cv2.destroyAllWindows() # 실시간 재생 창은 닫아줍니다.

    # [논리 오류 수정] 무릎 최소 각도는 반드시 타격 프레임 '이전' 구간에서만 탐색
    min_knee_angle = 180.0
    min_knee_frame = 0
    for i in range(len(frame_numbers)):
        if frame_numbers[i] < strike_frame:
            if knee_angles[i] < min_knee_angle and knee_angles[i] > 30.0:
                min_knee_angle = knee_angles[i]
                min_knee_frame = frame_numbers[i]

    # 타격 순간의 최종 데이터 추출
    strike_idx = frame_numbers.index(strike_frame)
    max_knee_angle_at_strike = knee_angles[strike_idx]
    torso_angle_at_strike = torso_angles[strike_idx]

    # ------------------------------------------------------------------
    # 🌟 [신규 추가] 2단계: 최정점 타격 순간 프레임을 화면에 "짠" 하고 고정 출력
    # ------------------------------------------------------------------
    if strike_frame_image is not None:
        print(f"\n--- [2단계] 타격 순간 고정 화면 검증창이 활성화되었습니다 (Frame: {strike_frame}) ---")
        h, w, _ = strike_frame_image.shape
        # 상단과 하단에 검은색 정량적 데이터 배너 박스 오버레이 생성
        cv2.rectangle(strike_frame_image, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.rectangle(strike_frame_image, (0, h-40), (w, h), (0, 0, 0), -1)
        
        cv2.putText(strike_frame_image, f"[CRITICAL EVENT] MAXIMUM STRIKE POINT", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(strike_frame_image, f"Strike Frame: {strike_frame} f | Knee: {max_knee_angle_at_strike:.1f} deg | Torso: {torso_angle_at_strike:.1f} deg", 
                    (20, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 사용자가 아무 키나 누를 때까지 무한대기(박제) 상태로 윈도우 창 유지
        cv2.imshow("STRIKE MOMENT CAPTURE (Press Any Key to Continue)", strike_frame_image)
        print("※ 고정된 타격 순간 창을 보고 캡처하신 후, 아무 키나 누르면 다음 그래프 창으로 넘어갑니다.")
        cv2.waitKey(0) # 0은 아무 키나 누를 때까지 멈춰있는 옵션입니다.
        cv2.destroyAllWindows()

    pose.close()

    # 3. 콘솔 정량적 데이터 레포트 출력
    print("\n==================================================")
    print("           [ 전문가 옆차기 정량 데이터 분석 결과 ]")
    print("==================================================")
    print(f" 총 분석 범위 : 1 ~ {frame_count} 프레임(frame)")
    print(f" 판정 수행발  : {active_foot} Foot")
    print("--------------------------------------------------")
    print(f" 1. 무릎 최대 수축 (예비 동작) 순간 (타격 이전 구간):")
    print(f"   - 포착 프레임: {min_knee_frame} frame")
    print(f"   - 무릎 최소 사잇각: {min_knee_angle:.2f} 도(Degree)")
    print("--------------------------------------------------")
    print(f" 2. 최정점 타격 순간 (발목 최고 고도 시점):")
    print(f"   - 포착 프레임: {strike_frame} frame")
    print(f"   - 타격 시 무릎 최대 사잇각: {max_knee_angle_at_strike:.2f} 도(Degree)")
    print(f"   - 타격 시 상체 제어 기울기: {torso_angle_at_strike:.2f} 도(Degree)")
    print("==================================================")

    # 4. 시계열 무릎 각도 그래프 시각화 출력 
    print("\n--- [3단계] 시계열 무릎 각도 그래프 분석창을 출력합니다 ---")
    plt.figure(figsize=(10, 5))
    plt.plot(frame_numbers, knee_angles, label='Knee Joint Angle', color='g', linewidth=2)
    
    # Key-Event 수직선 및 마커 표시
    plt.axvline(x=min_knee_frame, color='orange', linestyle='--', label=f'Min Knee (F:{min_knee_frame})')
    plt.scatter(min_knee_frame, min_knee_angle, color='orange', s=100, zorder=5)
    
    plt.axvline(x=strike_frame, color='red', linestyle='--', label=f'Strike Point (F:{strike_frame})')
    plt.scatter(strike_frame, max_knee_angle_at_strike, color='red', s=100, zorder=5)
    
    # 그래프 정보 레이아웃 설정
    plt.title('Taekwondo Kicking Time-Series Knee Angle Analysis', fontsize=14, pad=15)
    plt.xlabel('Time Flow (Frame Number)', fontsize=11)
    plt.ylabel('Joint Angle (Degree)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    # 그래프 하단에 타격 순간 프레임과 상체 각도 정보 주석 명시
    plt.figtext(0.15, 0.02, f"※ Target Strike Event [Frame {strike_frame}] -> Torso Alignment Angle: {torso_angle_at_strike:.1f}°", 
                fontsize=11, color='blue', weight='bold')
    
    plt.tight_layout(pad=3.0)
    plt.show()

if __name__ == "__main__":
    analyze_expert_video_final()
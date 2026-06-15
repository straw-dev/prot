import cv2
import mediapipe as mp
import math
import os
import matplotlib.pyplot as plt

def calculate_angle(p1, p2, p3):
    """[이론적 배경 2.5] atan2 함수를 이용한 관절 내각 연산 (0~180도 보정)"""
    theta1 = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    theta2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
    
    angle = math.degrees(abs(theta2 - theta1))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def get_line_slope_and_angle(shoulder_mid, hip_mid):
    """두 어깨 중점과 두 골반 중점을 잇는 상체 선의 기울기와 수직선 기준 각도를 반환합니다."""
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1] 
    
    if dx == 0:
        slope = float('inf')
    else:
        slope = dy / dx
        
    angle_from_vertical = abs(math.degrees(math.atan2(dx, -dy))) if dy != 0 else 90.0
    return slope, angle_from_vertical

def resize_frame_fixed_width(frame, target_width=400):
    """영상의 가로 크기를 요청하신 400px로 고정하고 원본 비율에 맞춰 세로 크기를 조절합니다."""
    h, w, _ = frame.shape
    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)
    return cv2.resize(frame, (target_width, target_height))

def run_taekwondo_feedback_system_v4():
    # 1. 영상 파일 경로 입력 및 전처리 초기화
    video_path = input("분석할 태권도 옆차기 영상 파일 경로를 입력하세요: ").strip('"')
    if not os.path.exists(video_path):
        print("파일 경로를 찾을 수 없습니다. 다시 확인해 주세요.")
        return

    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    TH_KNEE_MIN = 90.0
    TH_KNEE_MAX = 170.0

    raw_frames = []
    raw_landmarks_data = []

    print("\n[1단계] 영상 프레임을 읽어 관절 로그 데이터를 수집하는 중입니다...")
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        raw_frames.append(frame.copy())
        raw_landmarks_data.append(results.pose_landmarks)

    cap.release()

    if frame_count == 0:
        print("영상을 로드하지 못했습니다.")
        return

    # 2. 시간적 구간 분할 및 차는발(좌/우) 자동 판별 연산
    print("[2단계] 인체역학적 고도 변위 분석을 통한 실시간 수행발 판별 연산 중...")
    left_y_displacements = []
    right_y_displacements = []
    valid_indices = []

    for idx, landmarks in enumerate(raw_landmarks_data):
        if landmarks:
            valid_indices.append(idx)
            left_y_displacements.append(landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
            right_y_displacements.append(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)

    if not valid_indices:
        print("영상 내에서 포즈를 검출하지 못했습니다.")
        return

    left_diff = max(left_y_displacements) - min(left_y_displacements)
    right_diff = max(right_y_displacements) - min(right_y_displacements)

    if left_diff > right_diff:
        kicking_side = "LEFT"
        print(f" -> [판정 완료] 차는발 방향: '왼발(LEFT)' / 지지발 방향: '오른발(RIGHT)'")
    else:
        kicking_side = "RIGHT"
        print(f" -> [판정 완료] 차는발 방향: '오른발(RIGHT)' / 지지발 방향: '왼발(LEFT)'")

    # 3. 순수 픽셀 데이터 변환 및 전방 보간 처리
    print("[3단계] 미검출 구간 결측치 보간 및 상체 중심 축 변환 연산 중...")
    cleaned_landmarks_pixel = []
    last_valid_pixel_coords = None

    if kicking_side == "LEFT":
        k_hip_idx, k_knee_idx, k_ankle_idx, k_toe_idx = 23, 25, 27, 31
        s_hip_idx, s_ankle_idx, s_heel_idx = 24, 28, 30
    else:
        k_hip_idx, k_knee_idx, k_ankle_idx, k_toe_idx = 24, 26, 28, 32
        s_hip_idx, s_ankle_idx, s_heel_idx = 23, 27, 29

    for idx in range(frame_count):
        landmarks = raw_landmarks_data[idx]
        frame = raw_frames[idx]
        h, w, _ = frame.shape

        if landmarks is None:
            if last_valid_pixel_coords is not None:
                cleaned_landmarks_pixel.append(last_valid_pixel_coords.copy())
            else:
                cleaned_landmarks_pixel.append(None)
            continue

        l_dict = landmarks.landmark
        
        l_shoulder = [l_dict[11].x * w, l_dict[11].y * h]
        r_shoulder = [l_dict[12].x * w, l_dict[12].y * h]
        l_hip = [l_dict[23].x * w, l_dict[23].y * h]
        r_hip = [l_dict[24].x * w, l_dict[24].y * h]
        
        shoulder_mid = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
        hip_mid = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]

        curr_pixels = {
            'k_hip': [l_dict[k_hip_idx].x * w, l_dict[k_hip_idx].y * h],
            'k_knee': [l_dict[k_knee_idx].x * w, l_dict[k_knee_idx].y * h],
            'k_ankle': [l_dict[k_ankle_idx].x * w, l_dict[k_ankle_idx].y * h],
            'k_toe': [l_dict[k_toe_idx].x * w, l_dict[k_toe_idx].y * h],
            's_hip': [l_dict[s_hip_idx].x * w, l_dict[s_hip_idx].y * h],
            's_ankle': [l_dict[s_ankle_idx].x * w, l_dict[s_ankle_idx].y * h],
            's_heel': [l_dict[s_heel_idx].x * w, l_dict[s_heel_idx].y * h],
            'sh_mid': shoulder_mid,
            'hp_mid': hip_mid
        }

        cleaned_landmarks_pixel.append(curr_pixels)
        last_valid_pixel_coords = curr_pixels

    # 4. 핵심 이벤트 감지 및 역학 변수 연산
    print("[4단계] 최정점 타격 프레임 및 타격 이전 무릎 최소 각도 추적 중...")
    analyzed_frames_idx = []
    knee_angles = []
    
    lowest_k_ankle_y = float('inf')
    strike_frame_num = 0

    for idx in range(frame_count):
        p = cleaned_landmarks_pixel[idx]
        if p is None: continue
        if p['k_ankle'][1] < lowest_k_ankle_y:
            lowest_k_ankle_y = p['k_ankle'][1]
            strike_frame_num = idx + 1

    min_knee_angle = 180.0
    min_knee_frame_num = 0

    for idx in range(frame_count):
        p = cleaned_landmarks_pixel[idx]
        if p is None: continue

        k_angle = calculate_angle(p['k_hip'], p['k_knee'], p['k_ankle'])
        analyzed_frames_idx.append(idx + 1)
        knee_angles.append(k_angle)

        if (idx + 1) < strike_frame_num:
            if k_angle < min_knee_angle and k_angle > 30.0:
                min_knee_angle = k_angle
                min_knee_frame_num = idx + 1

    strike_array_idx = analyzed_frames_idx.index(strike_frame_num)
    strike_knee_angle = knee_angles[strike_array_idx]

    # 5. [수정] 부호 왜곡 없는 물리적 위치(dx) 기반 상체 채점 및 피드백 도출
    print("[5단계] 척추 중심선 상대위치 기반 맞춤형 채점 및 피드백 도출 중...")
    p_strike = cleaned_landmarks_pixel[strike_frame_num - 1]
    
    # 두 중점 간의 순수한 X축 거리 차이(dx)와 수직선 기준 누운 절대 각도 오차 추출
    dx = p_strike['sh_mid'][0] - p_strike['hp_mid'][0]
    torso_slope, torso_angle_err = get_line_slope_and_angle(p_strike['sh_mid'], p_strike['hp_mid'])

    # [항목 1] 무릎 점수
    s1_err_prep = max(0.0, min_knee_angle - TH_KNEE_MIN)
    s1_err_strike = max(0.0, TH_KNEE_MAX - strike_knee_angle)
    score_1 = max(10.0, 100.0 - (s1_err_prep * 1.5 + s1_err_strike * 1.5))

    # [항목 2] 상체 각도 유지 채점 (물리적 X축 상대 위치 판정 방식 도입)
    if kicking_side == "RIGHT":
        # 오른발을 오른쪽으로 차는데 상체가 타격 방향(우측)으로 과도하게 쓰러진 경우 (dx > 0 이고 일정 각도 이상 누웠을 때)
        if dx < 0 and torso_angle_err > 15.0: 
            score_2 = max(10.0, 100.0 - (torso_angle_err * 10.0))
            torso_status_text = f"Over-leaned to Right (Err: {torso_angle_err:.1f} deg)"
        else: 
            score_2 = 100.0
            torso_status_text = "Good (Well-Aligned)"
    else:
        # 왼발을 왼쪽으로 차는데 상체가 타격 방향(좌측)으로 과도하게 쓰러진 경우 (dx < 0 이고 일정 각도 이상 누웠을 때)
        if dx > 0 and torso_angle_err > 15.0: 
            score_2 = max(10.0, 100.0 - (torso_angle_err * 10.0))
            torso_status_text = f"Over-leaned to Left (Err: {torso_angle_err:.1f} deg)"
        else: 
            score_2 = 100.0
            torso_status_text = "Good (Well-Aligned)"

    # [항목 3] 지지발 뒤축 회전도 점수
    s_ankle_x = p_strike['s_ankle'][0]
    s_heel_x = p_strike['s_heel'][0]
    heel_alignment_error = abs(s_ankle_x - s_heel_x)
    score_3 = max(20.0, min(100.0, 100.0 - (heel_alignment_error * 0.8)))

    # [항목 4] 차는발 뒤축 고도 점수
    k_toe_y = p_strike['k_toe'][1]
    k_heel_y = p_strike['k_ankle'][1]
    if k_heel_y < k_toe_y:
        score_4 = 100.0
    else:
        score_4 = max(30.0, 100.0 - (abs(k_heel_y - k_toe_y) * 2.0))

    # 최저 성취도 기반 취약점 판정
    score_list = [score_1, score_2, score_3, score_4]
    lowest_score = min(score_list)
    lowest_idx = score_list.index(lowest_score)

    feedback_messages = [
        "타격 전 무릎을 가슴 안쪽으로 더 깊게 접었다가 강하게 뻗어 차세요.",
        "차는 발에 상체가 과도하게 누워 중심이 무너질 수 있으니 허리와 상체를 더 세우세요.",
        "디딤발의 회전이 부족하니 뒤축이 차는 방향을 바라보도록 완전히 돌려주세요.",
        "타격 시 발끝이 들려있으니 발날 뒤축을 발가락보다 높게 들어 올려 타격하세요."
    ]
    final_feedback_text = f"우선 보완점 피드백: {feedback_messages[lowest_idx]}"

    # 6. 실시간 비디오 시각화 재생 인터페이스 (크기 가로 400 고정)
    print("\n[6단계] 실시간 영상 분석 오버레이 화면을 출력합니다.")
    for idx in range(frame_count):
        display_frame = raw_frames[idx].copy()
        p = cleaned_landmarks_pixel[idx]
        raw_landmarks = raw_landmarks_data[idx]

        if raw_landmarks:
            mp_drawing.draw_landmarks(display_frame, raw_landmarks, mp_pose.POSE_CONNECTIONS)

        if p:
            k_angle = calculate_angle(p['k_hip'], p['k_knee'], p['k_ankle'])
            cv2.putText(display_frame, f"Frame: {idx+1}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Knee: {k_angle:.1f} deg", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.line(display_frame, (int(p['sh_mid'][0]), int(p['sh_mid'][1])), (int(p['hp_mid'][0]), int(p['hp_mid'][1])), (255, 0, 255), 3)

        # 가로 400픽셀 고정 리사이징
        resized_display = resize_frame_fixed_width(display_frame, target_width=400)
        cv2.imshow("Real-time Taekwondo Analysis Window", resized_display)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    # 7. 최종 최정점 타격 순간 고정 화면 '박제' 출력 (가로 400 고정, 고칠점 텍스트 배제)
    strike_img = raw_frames[strike_frame_num - 1].copy()
    strike_landmarks = raw_landmarks_data[strike_frame_num - 1]
    
    if strike_landmarks:
        mp_drawing.draw_landmarks(strike_img, strike_landmarks, mp_pose.POSE_CONNECTIONS)
        
    cv2.line(strike_img, (int(p_strike['sh_mid'][0]), int(p_strike['sh_mid'][1])), (int(p_strike['hp_mid'][0]), int(p_strike['hp_mid'][1])), (255, 0, 255), 4)
    
    # 가로 400픽셀 고정 리사이징 후 배너 오버레이 처리
    strike_img_resized = resize_frame_fixed_width(strike_img, target_width=400)
    s_h, s_w, _ = strike_img_resized.shape
    
    # 텍스트 오버레이용 검은색 하단 배너 영역 설정
    cv2.rectangle(strike_img_resized, (0, 0), (s_w, 35), (0, 0, 0), -1)
    cv2.rectangle(strike_img_resized, (0, s_h-40), (s_w, s_h), (20, 20, 20), -1)
    
    # 고칠점 텍스트는 완전히 제외하고 프레임 정보와 4가지 지표 점수만 깔끔하게 구성 (가독성 향상)
    cv2.putText(strike_img_resized, f"[EVENT] STRIKE MOMENT (Frame {strike_frame_num})", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.putText(strike_img_resized, f"Scores: [{score_1:.0f} | {score_2:.0f} | {score_3:.0f} | {score_4:.0f}]", (15, s_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
    
    # 터미널 창 최종 콘솔 리포트 출력
    print(f"\n==================================================")
    print(f"       [ 시스템 종합 정량적 피드백 레포트 결과 ]")
    print(f"==================================================")
    print(f" 판정된 수행발 방향       : {kicking_side} FOOT")
    print(f" 1. 무릎 최대 수축 시점   : 제 {min_knee_frame_num} 프레임 (사잇각: {min_knee_angle:.1f}°)")
    print(f" 2. 최정점 타격 순간 시점 : 제 {strike_frame_num} 프레임 (사잇각: {strike_knee_angle:.1f}°)")
    print(f" 3. 타격 순간 상체 정렬   : {torso_status_text}")
    print(f"--------------------------------------------------")
    print(f" [세부 항목별 성취 점수 지표]")
    print(f"  - 항목1 (무릎 접기/펴기 성취도) : {score_1:.2f} 점 / 100점")
    print(f"  - 항목2 (상체 각도 유지도)      : {score_2:.2f} 점 / 100점")
    print(f"  - 항목3 (지지발 뒤축 회전도)    : {score_3:.2f} 점 / 100점")
    print(f"  - 항목4 (차는발 뒤축 고도성)    : {score_4:.2f} 점 / 100점")
    print(f"--------------------------------------------------")
    print(final_feedback_text)
    print(f"==================================================")
    
    cv2.imshow("STRIKE MOMENT CAPTURE (Press Any Key to Continue)", strike_img_resized)
    print("\n※ 화면에 박제된 타격 순간 검증창(가로 400px 고정) 확인 후, 아무 키나 누르면 시계열 그래프창으로 이동합니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 8. Matplotlib 기반 시계열 분석 그래프 출력
    plt.figure(figsize=(10, 5))
    plt.plot(analyzed_frames_idx, knee_angles, label='Knee Joint Angle', color='forestgreen', linewidth=2.5)
    plt.axvline(x=min_knee_frame_num, color='darkorange', linestyle='--', label=f'Min Knee Prep (F:{min_knee_frame_num})')
    plt.scatter(min_knee_frame_num, min_knee_angle, color='darkorange', s=120, zorder=5)
    plt.axvline(x=strike_frame_num, color='crimson', linestyle='--', label=f'Strike Max Point (F:{strike_frame_num})')
    plt.scatter(strike_frame_num, strike_knee_angle, color='crimson', s=120, zorder=5)
    
    plt.title('Taekwondo Kicking Time-Series Knee Angle Analysis', fontsize=14, pad=15, weight='bold')
    plt.xlabel('Time Flow (Frame Number)', fontsize=12)
    plt.ylabel('Joint Angle (Degree °)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    summary_text = f"Side: {kicking_side} | Strike F: {strike_frame_num} | Torso: {torso_status_text} | Primary Vulnerability: Item {lowest_idx+1}"
    plt.figtext(0.12, 0.02, f"※ System Analytics: {summary_text}", fontsize=10, color='navy', weight='bold')
    plt.tight_layout(pad=3.5)
    plt.show()

if __name__ == "__main__":
    run_taekwondo_feedback_system_v4()
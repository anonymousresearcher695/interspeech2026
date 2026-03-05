#!/usr/bin/env python3
import pandas as pd
import re
import sys
import os

def get_dominant_speaker(speaker_str):
    if not isinstance(speaker_str, str): return None
    # 점유율이 가장 높은 첫 번째 화자 추출
    m = re.search(r"(SPEAKER_\d+)", speaker_str)
    return m.group(1) if m else None

def process_and_predict_labels(df, date, output_dir):
    """
    분석 로직 (V6.7):
    1. 화자 식별(DJ): 시작하자마자(2초 이내) 등장하는 시보/광고 화자 필터링 후, 발화량 상위 3명 중 가장 먼저 등장한 사람
    2. 화자 식별(GUEST): 대화 턴(Turn-taking) 기반
    3. 통계표 추출: dj_stats.csv 파일 별도 생성
    4. 세그먼트 라벨링: predicted_labels.csv 에 저장
    5. 팝송 방어(Protection)
    """
    
    # 1. 기초 화자 데이터 정리
    df['Dominant_Speaker'] = df.apply(
        lambda row: get_dominant_speaker(row.get('Speakers', '')) if row['Type'] == 'speech' else None,
        axis=1
    )
    
    total_broadcast_time = df['Stop Time'].max()
    all_speakers = [s for s in df['Dominant_Speaker'].unique() if s]
    
    # --- [Step 1] 화자별 통계 수집 및 DJ 식별 ---
    speaker_metrics = []
    for spk in all_speakers:
        spk_rows = df[df['Dominant_Speaker'] == spk]
        total_dur = spk_rows['Duration'].sum()
        first_app = spk_rows['Start Time'].min()
        last_app = spk_rows['Stop Time'].max()
        
        opening_ratio = (first_app / total_broadcast_time) * 100
        active_range = ((last_app - first_app) / total_broadcast_time) * 100
        
        speaker_metrics.append({
            'Speaker': spk, 
            'Total_Duration': round(total_dur, 2), 
            'Opening_Ratio': round(opening_ratio, 2), 
            'Active_Range': round(active_range, 2),
            'First_App': round(first_app, 2),
            'Last_App': round(last_app, 2)
        })

    m_df = pd.DataFrame(speaker_metrics)
    
    # 🚀 [추가된 필터링 로직]: 0~2초 사이에 등장하는 시보/광고 화자 무조건 탈락
    # (화자 분리 오류로 여러 광고 목소리가 하나로 묶여 발화량이 높아져도, 무조건 0초대에 등장하므로 여기서 걸러짐)
    filtered_df = m_df[m_df['First_App'] > 2.0]
    
    # 혹시라도 필터링 후 남는 화자가 없다면 (오류 방지) 원본 사용
    if filtered_df.empty:
        filtered_df = m_df

    # 발화량 기준 상위 3명 후보 선정
    top_candidates = filtered_df.sort_values(by='Total_Duration', ascending=False).head(3).copy()
    
    # 후보 3명 중 가장 먼저 입을 연 사람을 DJ로 확정
    dj_id = top_candidates.sort_values(by='First_App', ascending=True).iloc[0]['Speaker']
    
    print(f"👑 DJ 확정 (시보 필터링 후, 상위 3인 중 최초 등장): {dj_id}")

    # --- [Step 2] 게스트 식별 (대화 턴 기반) ---
    speech_df = df[df['Type'] == 'speech'].copy().reset_index(drop=True)
    interaction_counts = {spk: 0 for spk in all_speakers}
    
    for i in range(len(speech_df)):
        spk = speech_df.loc[i, 'Dominant_Speaker']
        if spk and spk != dj_id:
            prev_spk = speech_df.loc[i-1, 'Dominant_Speaker'] if i > 0 else None
            next_spk = speech_df.loc[i+1, 'Dominant_Speaker'] if i < len(speech_df)-1 else None
            
            # DJ와 번갈아가며 대화했는지 확인
            if prev_spk == dj_id or next_spk == dj_id:
                interaction_counts[spk] += 1

    candidates = [(spk, interaction_counts[spk]) for spk in all_speakers if spk != dj_id]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    guest_list = []
    if candidates:
        top_guest_spk, top_guest_cnt = candidates[0]
        cutoff_value = max(12, top_guest_cnt * 0.2) # 최소 12번 이상, 1등의 20% 이상
        guest_list = [spk for spk, cnt in candidates if cnt >= 20 or cnt >= cutoff_value]
        print(f"👥 GUEST 확정: {guest_list}")

    # --- [Step 3] 통계 파일(dj_stats.csv) 저장 로직 ---
    # 각 화자별로 최종 Role(역할)을 부여하여 통계표에 추가
    roles = []
    for spk in m_df['Speaker']:
        if spk == dj_id:
            roles.append('DJ')
        elif spk in guest_list:
            roles.append('GUEST')
        else:
            roles.append('AD')
    m_df['Role'] = roles
    
    # 통계 파일 저장
    stats_csv = os.path.join(output_dir, f"{date}-dj_stats.csv")
    m_df.sort_values(by='Total_Duration', ascending=False).to_csv(stats_csv, index=False, encoding='utf-8-sig')
    print(f"📊 Speaker stats saved to {stats_csv}")

    # --- [Step 4 & 5] 세그먼트 라벨링 및 팝송 방어 ---
    predicted_labels = []
    rescue_count = 0

    for idx, row in df.iterrows():
        row_type = row['Type']
        text = str(row['Transcript']) if pd.notna(row['Transcript']) else ""
        spk = row['Dominant_Speaker']
        
        if row_type == 'music':
            predicted_labels.append('MUSIC')
        elif row_type == 'silence':
            predicted_labels.append('SILENCE')
        elif row_type == 'speech':
            if spk == dj_id:
                predicted_labels.append('DJ')
            elif spk in guest_list:
                predicted_labels.append('GUEST')
            else:
                # [팝송 방어 로직] 영어가 대부분이면 MUSIC으로 구출
                english_chars = len(re.findall(r'[a-zA-Z]', text))
                korean_chars = len(re.findall(r'[가-힣]', text))
                
                if english_chars > 10 and korean_chars < 5:
                    predicted_labels.append('MUSIC')
                    rescue_count += 1
                else:
                    predicted_labels.append('AD')
        else:
            predicted_labels.append('UNKNOWN')
            
    df['Predicted_Label'] = predicted_labels
    print(f"🎵 팝송 보호 작동: {rescue_count}개의 영어 가사 세그먼트를 MUSIC으로 구출했습니다.")
    
    return df

def main():
    if len(sys.argv) != 2:
        print("Usage: python dj_stat_v6_7.py <YYYYMMDD>")
        sys.exit(1)

    date = sys.argv[1]
    # 디렉토리 경로는 본인 환경에 맞게 수정하세요 (예: baechulsu -> 프로그램명)
    base_dir = os.path.join(os.environ.get("RADIO_DATA_ROOT", "/path/to/radio/data"), date, "transcript")
    
    input_csv = os.path.join(base_dir, f"{date}_with_speaker_ratio.csv")
    labels_csv = os.path.join(base_dir, f"{date}_inference_result_ratio.csv")

    if not os.path.exists(input_csv):
        print(f"❌ Input not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    
    # 라벨링 및 통계 생성 적용
    labeled_df = process_and_predict_labels(df, date, base_dir)
    
    # 상세 라벨링 결과를 새로운 CSV 파일로 저장
    labeled_df.to_csv(labels_csv, index=False, encoding='utf-8-sig')
    print(f"💾 Labeled dataset saved to {labels_csv}")

if __name__ == "__main__":
    main()
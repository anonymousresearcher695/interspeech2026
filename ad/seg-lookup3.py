# ============================================================
# seg-lookup3.py
# ============================================================
# 역할: 클립들을 Panako DB와 비교하여 매칭 결과 추출
#
# 동작 흐름:
# 1. clips/19940713/ 폴더의 모든 클립(1192개)을 가져옴
# 2. 각 클립을 Panako DB와 비교 (panako query 명령어)
# 3. 매칭된 결과를 CSV로 저장
#
# 예시:
# - DB에 19940712.mp3가 저장되어 있음
# - 19940713의 클립들을 DB와 비교
# - 같은 광고가 있으면 매칭됨!
# ============================================================

import os
import subprocess
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse


def query_panako(file_path):
    """
    Panako의 query 명령어를 실행하고 결과를 반환
    
    panako query clip_000236_708000.mp3
    → DB에 저장된 오디오 중 이 클립과 유사한 구간을 찾음
    
    Args:
        file_path: 클립 MP3 경로
    
    Returns:
        (file_path, stdout 결과)
    """
    try:
        result = subprocess.run(
            ["panako", "query", file_path],  # ⭐ 핵심 명령어
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return file_path, result.stdout
    except Exception as e:
        print(f"Error querying {file_path}: {e}")
        return file_path, None


def parse_query_result(result):
    """
    Panako query 결과를 파싱하여 매칭 정보 추출
    
    Panako 출력 형식 (세미콜론 구분):
    순번;...;Query경로;Query시작;Query끝;Match경로;MatchID;Match시작;Match끝;점수;...
    
    예시:
    1;...;clip_000236.mp3;0.0;25.0;19940712.mp3;123;570.28;594.62;302;...
    
    → 이 클립이 19940712.mp3의 570~594초 구간과 매칭됨 (점수 302)
    """
    matches = []
    lines = result.splitlines()

    for line in lines:
        # ============================================================
        # 유효하지 않은 줄 건너뛰기
        # ============================================================
        # - 빈 줄
        # - "null" 포함 (매칭 실패)
        # - "-1.000" 포함 (매칭 실패)
        if not line.strip() or "null" in line or "-1.000" in line:
            continue

        # ============================================================
        # 세미콜론으로 분리하여 파싱
        # ============================================================
        fields = line.split(";")
        
        # 필드가 13개 이상이고, 첫 번째가 숫자(순번)인 경우만 처리
        if len(fields) >= 13 and fields[0].strip().isdigit():
            try:
                query_path = fields[2].strip()          # 쿼리 클립 경로
                query_start = float(fields[3].strip())  # 클립 내 시작 (보통 0)
                query_stop = float(fields[4].strip())   # 클립 내 끝 (보통 25)
                match_path = fields[5].strip()          # 매칭된 원본 파일 (19940712.mp3)
                match_id = fields[6].strip()            # 매칭 ID
                match_start = float(fields[7].strip())  # ⭐ 원본에서 매칭된 시작 시간
                match_stop = float(fields[8].strip())   # ⭐ 원본에서 매칭된 끝 시간
                match_score = int(fields[9].strip())    # ⭐ 매칭 점수 (높을수록 정확)
                time_factor = float(fields[10].strip().replace("%", ""))
                frequency_factor = float(fields[11].strip().replace("%", ""))
                seconds_with_match = float(fields[12].strip())

                matches.append({
                    "Query Path": query_path,
                    "Query Start": query_start,
                    "Query Stop": query_stop,
                    "Match Path": match_path,
                    "Match ID": match_id,
                    "Match Start": match_start,      # 중요: DB 원본에서의 위치
                    "Match Stop": match_stop,        # 중요: DB 원본에서의 위치
                    "Match Score": match_score,      # 중요: 매칭 신뢰도
                    "Time Factor (%)": time_factor,
                    "Frequency Factor (%)": frequency_factor,
                    "Seconds with Match (%)": seconds_with_match
                })
            except Exception as e:
                # 파싱 실패해도 전체 중단 안 함
                print(f"\n⚠️ 데이터 파싱 건너뜀: {line} | 사유: {e}")

    return matches if matches else None


def process_file(file_path):
    """
    단일 파일 처리: query 실행 → 결과 파싱
    """
    file_path, query_result = query_panako(file_path)
    if query_result:
        matches = parse_query_result(query_result)
        return matches
    return None


def process_directory_parallel(input_dir, output_csv):
    """
    디렉토리의 모든 MP3 클립을 병렬로 처리
    
    Args:
        input_dir: 클립 폴더 (예: clips/19940713/)
        output_csv: 결과 저장 파일
    """
    # ============================================================
    # Step 1: 클립 파일 목록 수집
    # ============================================================
    mp3_files = [
        os.path.join(input_dir, f) 
        for f in sorted(os.listdir(input_dir)) 
        if f.endswith(".mp3")
    ]
    print(f"Found {len(mp3_files)} MP3 files in {input_dir}")
    # 예: 1192개 클립

    # ============================================================
    # Step 2: 병렬 처리 (ThreadPoolExecutor)
    # ============================================================
    results = []
    with ThreadPoolExecutor() as executor:
        # 모든 클립에 대해 process_file 함수를 병렬 실행
        future_to_file = {
            executor.submit(process_file, file): file 
            for file in mp3_files
        }

        # tqdm으로 진행률 표시하면서 결과 수집
        for future in tqdm(as_completed(future_to_file), total=len(mp3_files), desc="Processing files"):
            try:
                matches = future.result()
                if matches:
                    results.extend(matches)  # 매칭 결과 누적
            except Exception as e:
                print(f"Error processing file: {e}")

    # ============================================================
    # Step 3: 결과를 CSV로 저장
    # ============================================================
    if results:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=[
                "Query Path", "Query Start", "Query Stop",
                "Match Path", "Match ID", "Match Start", "Match Stop",
                "Match Score", "Time Factor (%)", "Frequency Factor (%)", "Seconds with Match (%)"
            ])
            writer.writeheader()
            writer.writerows(results)

        print(f"\n🎉 성공! Results saved to {output_csv}")
    else:
        print("\nNo matches found. CSV file not created.")


# ============================================================
# 메인 실행부
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process MP3 files in a directory using Panako and save the results."
    )
    parser.add_argument("input_dir", type=str, 
                        help="Path to the input directory containing MP3 files.")
    parser.add_argument("compare_date", type=str, 
                        help="compare date in panako.")
    args = parser.parse_args()

    # 출력 파일명: 19940713-19940712-compare.csv
    # → 19940713 클립들을 19940712 DB와 비교한 결과
    output_csv = (
        os.path.basename(args.input_dir.rstrip("/")) 
        + "-" 
        + os.path.basename(args.compare_date) 
        + "-compare.csv"
    )
    
    process_directory_parallel(args.input_dir, output_csv)


## 핵심 요약

### 동작 흐름
# clips/19940713/
# ├── clip_000000_000000.mp3  → panako query → 매칭 없음
# ├── clip_000001_003000.mp3  → panako query → 매칭 없음
# ├── ...
# ├── clip_000236_708000.mp3  → panako query → 매칭! (570~594초, 점수 302)
# ├── clip_000237_711000.mp3  → panako query → 매칭! (573~597초, 점수 298)
# └── ...

# 결과: 19940713-19940712-compare.csv

# Match Score의 의미
# 점수: 300+ 이면, 매우 확실한 매칭(같은 광고)
# 점수: 100~300 사이면, 어느정도 확실
# 점수 50~100 사이면, 애매함(오탐 가능)
# 점수 50 미만이면, 거의 노이즈



# 출력 CSV 예시(19940713-19940712-compare.csv):
# Query Path,Query Start,Query Stop,Match Path,Match ID,Match Start,Match Stop,Match Score,Time Factor (%),Frequency Factor (%),Seconds with Match (%)
# clip_000236_708000.mp3,0.12,24.61,19940712.mp3,19940712,570.28,594.62,302,1.006,1.0,0.88
# clip_000252_756000.mp3,0.42,24.33,19940712.mp3,19940712,618.35,642.17,264,1.004,1.0,0.88
# clip_000038_114000.mp3,4.12,21.98,19940712.mp3,19940712,0.32,18.14,58,1.002,1.0,0.61

# ## 컬럼별 해석

# | 컬럼 | 예시 | 의미 |
# |------|------|------|
# | Query Path | clip_000236_708000.mp3 | 19940713의 708초 위치 클립 |
# | Query Start/Stop | 0.12 ~ 24.61 | 클립 내 매칭 구간 (거의 전체) |
# | Match Path | 19940712.mp3 | DB에 저장된 기준 파일 |
# | Match ID | 19940712 | 매칭된 파일 ID |
# | **Match Start** | **570.28** | **DB 파일에서 매칭된 시작 (초)** |
# | **Match Stop** | **594.62** | **DB 파일에서 매칭된 끝 (초)** |
# | **Match Score** | **302** | **매칭 신뢰도 (높을수록 좋음)** |
# | Time Factor | 1.006 | 속도 비율 (1.0 = 동일) |
# | Frequency Factor | 1.0 | 피치 비율 (1.0 = 동일) |
# | Seconds with Match | 0.88 | 클립 중 매칭된 비율 (88%) |

# ## 해석 예시
# clip_000236_708000.mp3 → 19940712의 570~594초와 매칭 (Score: 302)

# 의미:
# - 19940713의 708초 부근 = 19940712의 570초 부근
# - 같은 광고가 둘 다 나왔다!
# - Score 302 → 확실한 매칭

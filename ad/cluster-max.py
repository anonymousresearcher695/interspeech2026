#!/usr/bin/env python3
# ============================================================
# cluster-max.py
# ============================================================
# 수정 사항:
# 1. GAP_THRESHOLD 추가 - 연속 광고 인식
#    - 광고1 끝나고 광고2 시작까지 gap이 GAP_THRESHOLD 이내면 같은 블록으로 인식
# 2. OPENING_FILTER = 0 (첫 광고도 탐지하도록)
# 3. 클립 길이 20초 기준으로 동작
# ============================================================

import pandas as pd
import argparse
import os

# ============================================================
# 필터링 설정
# ============================================================
SCORE_THRESHOLD = 100      # 이 점수 이하는 오탐으로 간주하여 제거
OPENING_FILTER = 0         # 0으로 변경 - 오프닝 광고도 탐지
GAP_THRESHOLD = 30         # 30초 이내 gap이면 연속 광고로 인식 (같은 클러스터)


def find_non_overlapping_scores(sorted_cluster, cluster_id):
    """
    클러스터 내에서 겹치지 않는 최고 점수 매칭들을 선택
    """
    non_overlapping = []
    used_intervals = []

    for row in sorted_cluster:
        is_overlapping = False
        
        for start, stop in used_intervals:
            if not (row["Match Stop"] <= start or row["Match Start"] >= stop):
                is_overlapping = True
                break

        if not is_overlapping:
            row_with_cluster = row.copy()
            row_with_cluster["Cluster ID"] = cluster_id
            non_overlapping.append(row_with_cluster)
            used_intervals.append((row["Match Start"], row["Match Stop"]))

    return non_overlapping


def find_high_scores_in_clusters(data, gap_threshold):
    """
    연속된 매칭들을 클러스터로 묶고, 각 클러스터에서 최적 매칭 선택
    
    클러스터링 원리 (수정됨):
    - Match Start 순으로 정렬된 상태에서
    - 현재 매칭의 Start가 이전 매칭의 Stop + GAP_THRESHOLD보다 작으면 같은 클러스터
    - 아니면 새 클러스터 시작
    
    예시 (GAP_THRESHOLD=30):
    매칭1: 570~594  ─┐
    매칭2: 618~642   ├─ 클러스터 1 (gap=24초 < 30초, 연속 광고!)
    매칭3: 660~684  ─┘
    매칭4: 800~824  ─── 클러스터 2 (gap=116초 > 30초, 새 광고 블록)
    """
    clusters = []
    current_cluster = []

    for _, row in data.iterrows():
        if not current_cluster:
            current_cluster.append(row)
        else:
            last = current_cluster[-1]
            
            # ============================================================
            # 핵심 수정: GAP_THRESHOLD 추가
            # 이전 매칭 끝 + gap_threshold 이내면 같은 클러스터
            # ============================================================
            if row["Match Start"] <= last["Match Stop"] + gap_threshold:
                current_cluster.append(row)
            else:
                clusters.append(current_cluster)
                current_cluster = [row]

    if current_cluster:
        clusters.append(current_cluster)

    cluster_info = []
    all_non_overlapping_scores = []

    for cluster_id, cluster in enumerate(clusters, start=1):
        sorted_cluster = sorted(cluster, key=lambda x: x["Match Score"], reverse=True)
        non_overlapping_scores = find_non_overlapping_scores(sorted_cluster, cluster_id)
        all_non_overlapping_scores.extend(non_overlapping_scores)

        cluster_start = min(item["Match Start"] for item in cluster)
        cluster_stop = max(item["Match Stop"] for item in cluster)
        cluster_size = len(cluster)
        cluster_info.append({
            "Cluster ID": cluster_id,
            "Cluster Start": cluster_start,
            "Cluster Stop": cluster_stop,
            "Cluster Size": cluster_size,
            "Duration": round(cluster_stop - cluster_start, 2)
        })

    return all_non_overlapping_scores, cluster_info


def main(input_file, score_threshold, opening_filter, gap_threshold):
    """
    메인 함수
    """
    data = pd.read_csv(input_file)
    print(f"📂 입력: {input_file}")
    print(f"   원본 매칭 수: {len(data)}개")
    print(f"   설정: score_threshold={score_threshold}, opening_filter={opening_filter}, gap_threshold={gap_threshold}")

    # Step 2: 필터링
    before_count = len(data)
    data = data[data["Match Score"] >= score_threshold]
    print(f"   Score < {score_threshold} 제거: {before_count - len(data)}개 제거 → {len(data)}개 남음")
    
    if opening_filter > 0:
        before_count = len(data)
        data = data[data["Match Start"] >= opening_filter]
        print(f"   오프닝 ({opening_filter}초 이내) 제거: {before_count - len(data)}개 제거 → {len(data)}개 남음")

    if len(data) == 0:
        print("⚠️ 필터링 후 매칭이 없습니다.")
        # 빈 결과 파일 생성
        output_file = os.path.splitext(input_file)[0] + "-ad-result.csv"
        pd.DataFrame(columns=["Cluster ID", "Match Start", "Match Stop", "Duration", "Match Score", "Query Path"]).to_csv(output_file, index=False)
        return

    data["Duration"] = data["Match Stop"] - data["Match Start"]

    # Step 3: 정렬
    data = data.sort_values(by="Match Start")

    # Step 4: 클러스터링 (GAP_THRESHOLD 적용)
    non_overlapping_scores, cluster_info = find_high_scores_in_clusters(data, gap_threshold)
    print(f"   클러스터(광고 블록) 수: {len(cluster_info)}개")
    print(f"   최종 광고 매칭: {len(non_overlapping_scores)}개")

    # 클러스터 정보 출력
    print(f"\n   [클러스터 상세]")
    for ci in cluster_info:
        print(f"   - 클러스터 {ci['Cluster ID']}: {ci['Cluster Start']:.1f}~{ci['Cluster Stop']:.1f}초 (duration={ci['Duration']:.1f}초, 매칭={ci['Cluster Size']}개)")

    # Step 5: 결과 저장
    output_file = os.path.splitext(input_file)[0] + "-ad-result.csv"
    cluster_info_file = os.path.splitext(input_file)[0] + "-cluster-info.csv"

    result = pd.DataFrame(non_overlapping_scores)
    result = result.sort_values(by="Match Start")
    result["Duration"] = round(result["Match Stop"] - result["Match Start"], 2)
    result[["Cluster ID", "Match Start", "Match Stop", "Duration", "Match Score", "Query Path"]].to_csv(output_file, index=False)
    print(f"\n💾 광고 결과: {output_file}")

    cluster_info_df = pd.DataFrame(cluster_info)
    cluster_info_df.to_csv(cluster_info_file, index=False)
    print(f"💾 클러스터 정보: {cluster_info_file}")
    
    print("\n✅ 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a CSV file to find non-overlapping match scores in clusters."
    )
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("--score_threshold", type=int, default=100,
                        help="Minimum Match Score to keep (default: 100)")
    parser.add_argument("--opening_filter", type=int, default=0,
                        help="Ignore matches before this time in seconds (default: 0)")
    parser.add_argument("--gap_threshold", type=int, default=30,
                        help="Max gap between matches to consider same ad block (default: 30)")
    args = parser.parse_args()

    main(args.input_file, args.score_threshold, args.opening_filter, args.gap_threshold)
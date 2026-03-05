#!/usr/bin/env python3
"""
merge_ad_results.py
===================
7일치 Panako 결과를 병합하여 광고 DB를 구축합니다.

사용법:
  python merge_ad_results.py

출력:
  - results_7days/merged_ads.csv (병합된 광고 목록)
  - results_7days/ad_summary.csv (광고별 요약)
"""

import os
import glob
import pandas as pd
from collections import defaultdict

RESULTS_DIR = "results_7days"
OUTPUT_MERGED = os.path.join(RESULTS_DIR, "merged_ads.csv")
OUTPUT_SUMMARY = os.path.join(RESULTS_DIR, "ad_summary.csv")


def load_all_results():
    """모든 whisper 결과 파일 로드"""
    all_data = []
    
    pattern = os.path.join(RESULTS_DIR, "*-whisper.csv")
    files = glob.glob(pattern)
    
    print(f"📂 발견된 결과 파일: {len(files)}개")
    
    for f in sorted(files):
        # 파일명에서 날짜 추출: 19940713-19940712-whisper.csv
        basename = os.path.basename(f)
        parts = basename.replace("-whisper.csv", "").split("-")
        query_date = parts[0]
        db_date = parts[1]
        
        df = pd.read_csv(f)
        df['query_date'] = query_date
        df['db_date'] = db_date
        
        all_data.append(df)
        print(f"  {basename}: {len(df)}개 광고")
    
    if all_data:
        merged = pd.concat(all_data, ignore_index=True)
        return merged
    return None


def extract_ad_keywords(text):
    """광고 텍스트에서 주요 키워드 추출 (간단한 버전)"""
    if pd.isna(text) or not text:
        return ""
    
    # 알려진 광고 키워드
    keywords = [
        "인터메조", "민들레", "폴리그램", "수학독본", "수학 독본",
        "비너스", "피너스", "슬링크", "슬링컵", "짜짜로니",
        "레모나", "영단어", "우선순위", "과학원", "김한길",
        "엑센트", "현대자동차", "카페리쉬", "맥스웰", "펩시",
        "고려원", "한길사", "김영사", "해냄", "비전"
    ]
    
    found = []
    text_lower = text.lower()
    for kw in keywords:
        if kw.lower() in text_lower:
            found.append(kw)
    
    return ", ".join(found)


def cluster_similar_ads(df):
    """유사한 광고끼리 클러스터링 (시간대 기준)"""
    # Match Start 기준으로 비슷한 시간대 그룹핑
    df = df.sort_values('Match Start').reset_index(drop=True)
    
    groups = []
    current_group = []
    
    for idx, row in df.iterrows():
        if not current_group:
            current_group.append(row)
        else:
            # 이전 광고와 60초 이내면 같은 그룹
            last_end = current_group[-1]['Match Stop']
            if row['Match Start'] - last_end < 60:
                current_group.append(row)
            else:
                groups.append(current_group)
                current_group = [row]
    
    if current_group:
        groups.append(current_group)
    
    return groups


def create_ad_summary(df):
    """광고별 요약 생성"""
    summaries = []
    
    # 키워드 추출
    df['keywords'] = df['transcript'].apply(extract_ad_keywords)
    
    # 날짜별로 그룹핑하여 출현 횟수 계산
    keyword_stats = defaultdict(lambda: {'count': 0, 'dates': set(), 'transcripts': []})
    
    for idx, row in df.iterrows():
        keywords = row['keywords']
        if keywords:
            for kw in keywords.split(", "):
                keyword_stats[kw]['count'] += 1
                keyword_stats[kw]['dates'].add(row['query_date'])
                if len(keyword_stats[kw]['transcripts']) < 3:
                    keyword_stats[kw]['transcripts'].append(row['transcript'][:100])
    
    for kw, stats in sorted(keyword_stats.items(), key=lambda x: -x[1]['count']):
        summaries.append({
            'keyword': kw,
            'count': stats['count'],
            'days_appeared': len(stats['dates']),
            'dates': ', '.join(sorted(stats['dates'])),
            'sample_transcript': stats['transcripts'][0] if stats['transcripts'] else ''
        })
    
    return pd.DataFrame(summaries)


def main():
    print("=" * 60)
    print("🔄 7일치 Panako 결과 병합")
    print("=" * 60)
    
    # 1. 모든 결과 로드
    df = load_all_results()
    
    if df is None or len(df) == 0:
        print("❌ 결과 파일이 없습니다.")
        return
    
    print(f"\n총 광고 클립: {len(df)}개")
    
    # 2. 병합된 결과 저장
    df.to_csv(OUTPUT_MERGED, index=False, encoding='utf-8-sig')
    print(f"💾 병합 결과 저장: {OUTPUT_MERGED}")
    
    # 3. 광고 요약 생성
    summary_df = create_ad_summary(df)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding='utf-8-sig')
    print(f"💾 광고 요약 저장: {OUTPUT_SUMMARY}")
    
    # 4. 요약 출력
    print("\n" + "=" * 60)
    print("📊 광고 키워드 통계 (출현 빈도순)")
    print("=" * 60)
    
    for idx, row in summary_df.head(15).iterrows():
        print(f"  {row['keyword']}: {row['count']}회 ({row['days_appeared']}일)")
    
    print("\n✅ 완료!")


if __name__ == "__main__":
    main()
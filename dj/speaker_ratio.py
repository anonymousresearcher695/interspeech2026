#!/usr/bin/env python3
# whisperx_to_csv.py
import sys
import os
import re
import pandas as pd
from pathlib import Path

def parse_whisperx_txt(txt_path):
    segments = []
    # WhisperX 출력 형식: [0.000 - 5.000] SPEAKER_00: Hello world.
    pattern = re.compile(r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(SPEAKER_\d+|UNKNOWN):\s*(.*)")

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                segments.append({
                    "Type":       "speech",
                    "Start Time": float(m.group(1)),
                    "Stop Time":  float(m.group(2)),
                    "Duration":   float(m.group(2)) - float(m.group(1)),
                    "Transcript": m.group(4).strip(),
                    "Speakers":   f"{m.group(3)}:1.00s(1.000)",
                })

    # 1. 시간순 정렬
    sorted_speech = sorted(segments, key=lambda x: x["Start Time"])
    
    # 2. 간격(Gap) 계산 및 Music 타입 추가
    final_segments = []
    prev_end_time = 0.0

    for seg in sorted_speech:
        curr_start = seg["Start Time"]
        
        # 현재 말하기 시작 전과 이전 말하기 종료 사이의 간격 계산
        gap_duration = curr_start - prev_end_time
        
        # 간격이 100초 이상이면 Music 세그먼트 삽입
        if gap_duration >= 100.0:
            final_segments.append({
                "Type":       "Music",
                "Start Time": prev_end_time,
                "Stop Time":  curr_start,
                "Duration":   gap_duration,
                "Transcript": "",
                "Speakers":   "",
            })
        
        final_segments.append(seg)
        prev_end_time = seg["Stop Time"]

    return final_segments


def main():
    if len(sys.argv) != 2:
        print("Usage: python whisperx_to_csv.py <YYYYMMDD>")
        sys.exit(1)

    date_str   = sys.argv[1]
    # 경로 설정 (사용자 환경에 맞춤)
    base_dir   = Path(os.environ.get("RADIO_DATA_ROOT", "/path/to/radio/data")) / date_str / "transcript"
    txt_path   = base_dir / f"{date_str}_whisperx.txt"
    output_csv = base_dir / f"{date_str}_with_speaker_ratio.csv"

    if not txt_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {txt_path}")
        sys.exit(1)

    segments = parse_whisperx_txt(txt_path)
    print(f"📥 처리 완료 세그먼트(Speech + Music): {len(segments)}")

    df = pd.DataFrame(segments, columns=[
        "Type", "Start Time", "Stop Time", "Duration", "Transcript", "Speakers"
    ])
    
    # 저장
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 저장 완료: {output_csv}")


if __name__ == "__main__":
    main()
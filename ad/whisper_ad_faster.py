#!/usr/bin/env python3
# ============================================================
# whisper_ad_faster.py
# ============================================================
# 역할: Panako로 탐지된 광고 클립들을 Whisper로 전사
#
# 동작 흐름:
# 1. cluster-max.py 결과 (ad-result.csv) 로드
# 2. 각 클립의 Query Path에서 MP3 파일 읽기
# 3. faster-whisper로 한국어 전사
# 4. 전사 결과를 CSV에 추가하여 저장
#
# 목적:
# - 광고 클립의 텍스트를 얻어서 제품명 추출에 사용
# - 예: "비너스 슬림컷 브라" → 제품명: 비너스
# ============================================================

import csv
import os
import argparse
from tqdm import tqdm
from faster_whisper import WhisperModel

# -----------------------------
# Config
# -----------------------------
DEFAULT_MODEL = "large-v3"   # 모델 크기: tiny / base / small / medium / large-v2 / large-v3
DEVICE = "cuda"              # GPU 사용: "cuda", CPU만: "cpu"
COMPUTE_TYPE = "float16"     # cuda: float16, cpu: int8 (메모리 절약)


def load_rows(csv_path):
    """
    CSV 파일 로드
    
    입력 예시 (ad-result.csv):
    Cluster ID,Match Start,Match Stop,Duration,Match Score,Query Path
    2,570.28,594.62,24.34,302,clip_000236_708000.mp3
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def transcribe_audio(model, audio_path):
    """
    faster-whisper로 오디오 전사
    
    Args:
        model: WhisperModel 인스턴스
        audio_path: MP3 클립 경로
    
    Returns:
        {
            "transcript": "비너스 슬림컷 브라...",
            "language": "ko",
            "avg_logprob": -0.25  # 신뢰도 (0에 가까울수록 좋음)
        }
    """
    try:
        # ============================================================
        # Whisper 전사 실행
        # ============================================================
        segments, info = model.transcribe(
            audio_path,
            language="ko",       # 한국어 강제 지정
            beam_size=5,         # 빔 서치 크기 (높을수록 정확, 느림)
            vad_filter=True      # Voice Activity Detection (묵음 구간 스킵)
        )

        texts = []
        avg_logprobs = []

        # 세그먼트별 텍스트 수집
        for seg in segments:
            texts.append(seg.text.strip())
            if seg.avg_logprob is not None:
                avg_logprobs.append(seg.avg_logprob)

        return {
            "transcript": " ".join(texts),  # 전체 텍스트
            "language": info.language,       # 감지된 언어
            "avg_logprob": round(sum(avg_logprobs) / len(avg_logprobs), 4) if avg_logprobs else ""
        }

    except Exception as e:
        return {
            "transcript": "",
            "language": "",
            "avg_logprob": "",
            "error": str(e)
        }


def main():
    # ============================================================
    # 인자 파싱
    # ============================================================
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Panako ad result CSV (ad-result.csv)")
    ap.add_argument("--output", required=True, help="Output CSV with Whisper transcripts")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model size")
    args = ap.parse_args()

    # ============================================================
    # Step 1: 광고 클립 목록 로드
    # ============================================================
    rows = load_rows(args.input)
    print(f"[INFO] loaded {len(rows)} ad clips")

    # ============================================================
    # Step 2: Whisper 모델 로드
    # ============================================================
    print(f"[INFO] loading faster-whisper model: {args.model}")
    model = WhisperModel(
        args.model,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )

    # ============================================================
    # Step 3: 각 클립 전사
    # ============================================================
    out_rows = []

    for r in tqdm(rows, desc="Whisper (faster)"):
        audio_path = r["Query Path"]  # 클립 MP3 경로

        # 파일 존재 확인
        if not os.path.isfile(audio_path):
            r["transcript"] = ""
            r["language"] = ""
            r["avg_logprob"] = ""
            r["error"] = "file not found"
            out_rows.append(r)
            continue

        # 전사 실행
        tr = transcribe_audio(model, audio_path)

        # 결과 추가
        r["transcript"] = tr.get("transcript", "")
        r["language"] = tr.get("language", "")
        r["avg_logprob"] = tr.get("avg_logprob", "")
        r["error"] = tr.get("error", "")

        out_rows.append(r)

    # ============================================================
    # Step 4: 결과 CSV 저장
    # ============================================================
    fieldnames = list(out_rows[0].keys())
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"[DONE] saved → {args.output}")


if __name__ == "__main__":
    main()


## 핵심 요약

### 입력 → 출력
# ```
# 입력: 19940713-19940712-compare-ad-result.csv
#       (14개 광고 클립)

# 출력: 19940713-whisper.csv
#       (14개 광고 + 전사 텍스트)
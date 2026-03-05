#!/usr/bin/env python3
"""
DJ/GUEST 블록 요약 벤치마크 - OpenAI 요약 + 수동 채점용 CSV 출력

사용법:
    python benchmark_summary.py 20260114
    python benchmark_summary.py 20260114 /path/to/base_dir
    python benchmark_summary.py 20260114 /path/to/base_dir sk-xxxx   ← API키 직접 전달

    또는 환경변수로:
    export OPENAI_API_KEY="sk-xxxx"
    python benchmark_summary.py 20260114

입력:
    {date}-blocks.csv       - block_type / start / end / text 컬럼
    {date}-truth-blocks.csv - gt_block_type / gt_start / gt_end / summary 컬럼

출력:
    {date}-summary_eval.csv - GT 요약 + 예측 요약 + score(빈칸) 나란히
"""

import sys, os
import pandas as pd
from pathlib import Path

# .env 파일에서 환경변수 로드 (python-dotenv)
try:
    from dotenv import load_dotenv
    # 스크립트 위치 → 상위 폴더 순으로 .env 탐색
    _env_path = Path(__file__).parent / ".env"
    if not _env_path.exists():
        _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv 없으면 os.environ만 사용

DATE     = sys.argv[1] if len(sys.argv) > 1 else "20260114"
BASE_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent
API_KEY  = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("OPENAI_API_KEY", "")

_transcript_dir = BASE_DIR / "movie" / DATE / "transcript"
DATA_DIR = _transcript_dir if _transcript_dir.exists() else BASE_DIR

BLOCKS_CSV = DATA_DIR / f"{DATE}-blocks.csv"
GT_CSV     = DATA_DIR / f"{DATE}-truth-blocks.csv"
OUT_CSV    = DATA_DIR / f"{DATE}-summary_eval.csv"

TARGET_TYPES = {"DJ", "GUEST", "DJ/GUEST"}

SUMMARY_PROMPT = (
    "아래는 한국 라디오 방송 DJ 진행 구간의 트랜스크립트입니다.\n"
    "핵심 내용을 1~2문장으로 한국어로 요약하세요. "
    "소개된 음악, 청취자 사연, DJ 멘트 주제를 포함하세요.\n\n"
    "### 트랜스크립트:\n{text}\n\n### 요약:\n"
)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ── 데이터 로드 ───────────────────────────────────────────────────
def load_blocks(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    rename = {}
    for c in df.columns:
        lc = c.lower().replace(" ", "_")
        if lc == "start_time":        rename[c] = "start"
        elif lc == "stop_time":       rename[c] = "end"
        elif lc == "transcript":      rename[c] = "text"
        elif lc == "predicted_label": rename[c] = "block_type"
    df = df.rename(columns=rename)
    df = df[df["block_type"].isin(TARGET_TYPES)].copy()
    df["text"]  = df["text"].fillna("").astype(str)
    df["start"] = df["start"].astype(float)
    df["end"]   = df["end"].astype(float)
    return df.reset_index(drop=True)


def load_gt(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[df["gt_block_type"].isin(TARGET_TYPES)].copy()
    df["summary"]  = df["summary"].fillna("").astype(str)
    df["gt_start"] = pd.to_numeric(df["gt_start"], errors="coerce")
    df["gt_end"]   = pd.to_numeric(df["gt_end"],   errors="coerce")
    return df.dropna(subset=["gt_start", "gt_end"]).reset_index(drop=True)


# ── 블록 매칭 (IoU) ───────────────────────────────────────────────
def match_blocks(pred_df, gt_df, iou_threshold=0.3):
    """예측 블록 ↔ GT 블록 1:1 매칭. [(pred_idx, gt_idx), ...] 반환"""
    pairs = []
    used_gt = set()
    for pi, pred in pred_df.iterrows():
        best_iou, best_gi = 0.0, -1
        for gi, gt in gt_df.iterrows():
            if gi in used_gt:
                continue
            inter = max(0.0, min(pred["end"], gt["gt_end"]) - max(pred["start"], gt["gt_start"]))
            union = max(pred["end"], gt["gt_end"]) - min(pred["start"], gt["gt_start"])
            iou   = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0 and best_iou >= iou_threshold:
            pairs.append((pi, best_gi))
            used_gt.add(best_gi)
    return pairs


# ── OpenAI 요약 ───────────────────────────────────────────────────
def summarize_openai(text, client, model="gpt-4o-mini"):
    if not text.strip():
        return ""
    prompt = SUMMARY_PROMPT.format(text=text[:3000])
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"   ⚠️  OpenAI error: {e}")
        return ""


# ── 메인 ─────────────────────────────────────────────────────────
def main():
    print(f"\n📂 날짜: {DATE}")
    print(f"   blocks : {BLOCKS_CSV}")
    print(f"   GT     : {GT_CSV}")

    if not BLOCKS_CSV.exists():
        print(f"\n❌ blocks 파일 없음: {BLOCKS_CSV}")
        return
    if not GT_CSV.exists():
        print(f"\n❌ GT 파일 없음: {GT_CSV}")
        return

    pred_df = load_blocks(str(BLOCKS_CSV))
    gt_df   = load_gt(str(GT_CSV))
    pairs   = match_blocks(pred_df, gt_df)

    print(f"\n📊 예측 블록: {len(pred_df)}개  |  GT 블록: {len(gt_df)}개  |  매칭: {len(pairs)}개")
    unmatched_gt = len(gt_df) - len(pairs)
    if unmatched_gt > 0:
        print(f"   ⚠️  매칭 안 된 GT 블록: {unmatched_gt}개 (IoU < 0.3)")

    if not HAS_OPENAI:
        print("\n❌ openai 패키지 없음. 설치: pip install openai")
        return

    if not API_KEY:
        print("\n❌ OpenAI API 키가 없습니다. 다음 중 하나로 설정하세요:")
        print("   1) 스크립트와 같은 폴더에 .env 파일 생성:")
        print("         OPENAI_API_KEY=sk-...")
        print("   2) export OPENAI_API_KEY='sk-...'")
        print("   3) python benchmark_summary.py {date} {base_dir} sk-...")
        return

    client = OpenAI(api_key=API_KEY)
    rows = []

    print(f"\n🔹 OpenAI 요약 시작 ({len(pairs)}개 블록)...")
    for idx, (pi, gi) in enumerate(pairs, 1):
        pred = pred_df.loc[pi]
        gt   = gt_df.loc[gi]

        print(f"   [{idx}/{len(pairs)}] {pred['block_type']}  "
              f"{pred['start']:.1f}~{pred['end']:.1f}s", end=" ... ", flush=True)

        pred_summary = summarize_openai(pred["text"], client)
        print("✅")

        rows.append({
            "block_type":   pred["block_type"],
            "start":        pred["start"],
            "end":          pred["end"],
            "gt_summary":   gt["summary"],
            "pred_summary": pred_summary,
            "score":        "",   # ← 직접 채점 (예: 1~5)
            "comment":      "",   # ← 메모용
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n💾 저장 완료: {OUT_CSV}")
    print(f"   → score 컬럼을 직접 채워주세요 (예: 1~5점)")
    print(f"\n{'─'*60}")
    for _, row in df_out.iterrows():
        print(f"\n[{row['block_type']}] {row['start']:.1f}~{row['end']:.1f}s")
        print(f"  GT  : {row['gt_summary']}")
        print(f"  Pred: {row['pred_summary']}")


if __name__ == "__main__":
    main()
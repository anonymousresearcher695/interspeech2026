#!/usr/bin/env python3
# dj_merge_block_whisperx.py
# WhisperX 출력 기반 블록 병합 (Predicted_Label 기준)
# DJ/GUEST 인접 시 DJ/GUEST로 병합, DJ 단독은 DJ 유지

import pandas as pd
import sys
import os


def merge_blocks_by_label(df):
    blocks = []
    current_rows = []
    current_label = None

    def flush():
        if not current_rows:
            return
        text = " ".join(
            str(r["Transcript"]) for r in current_rows
            if pd.notna(r.get("Transcript")) and str(r.get("Transcript", "")).strip()
        ).strip()

        speakers = set()
        for r in current_rows:
            spk = r.get("Dominant_Speaker", "")
            if spk and pd.notna(spk):
                speakers.add(spk)

        blocks.append({
            "block_type":    current_label,
            "start":         current_rows[0]["Start Time"],
            "end":           current_rows[-1]["Stop Time"],
            "duration":      round(current_rows[-1]["Stop Time"] - current_rows[0]["Start Time"], 2),
            "segments":      len(current_rows),
            "speaker_count": len(speakers),
            "speakers":      ",".join(sorted(speakers)),
            "text":          text,
        })
        current_rows.clear()

    for _, row in df.iterrows():
        label = row.get("Predicted_Label", "UNKNOWN")
        
        if label == "SILENCE":   # ← 이거 추가
            continue

        if label != current_label:
            flush()
            current_label = label

        current_rows.append(row.to_dict())

    flush()
    return pd.DataFrame(blocks)


def should_merge(a, b):
    """같은 타입이거나, DJ↔GUEST 조합이면 병합"""
    if a == b:
        return True
    if {a, b} <= {"DJ", "GUEST"}:
        return True
    return False


def merged_label(a, b):
    if a == b:
        return a
    if {a, b} <= {"DJ", "GUEST"}:
        return "DJ/GUEST"
    return a


def merge_consecutive_same_blocks(blocks_df):
    if len(blocks_df) == 0:
        return blocks_df

    merged = []
    current = blocks_df.iloc[0].to_dict()

    for i in range(1, len(blocks_df)):
        row = blocks_df.iloc[i]

        if should_merge(current["block_type"], row["block_type"]):
            current["block_type"]    = merged_label(current["block_type"], row["block_type"])
            current["end"]           = row["end"]
            current["duration"]      = round(current["end"] - current["start"], 2)
            current["segments"]     += row["segments"]

            curr_spk = set(current["speakers"].split(",")) if current["speakers"] else set()
            new_spk  = set(row["speakers"].split(","))     if row["speakers"]     else set()
            combined = sorted(curr_spk | new_spk)
            current["speakers"]      = ",".join(combined)
            current["speaker_count"] = len(combined)

            if row["text"]:
                current["text"] = (current["text"] + " " + row["text"]).strip()
        else:
            merged.append(current)
            current = row.to_dict()

    merged.append(current)
    return pd.DataFrame(merged)


def main():
    if len(sys.argv) != 2:
        print("Usage: python dj_merge_block_whisperx.py <YYYYMMDD>")
        sys.exit(1)

    date     = sys.argv[1]
    base_dir = os.path.join(os.environ.get("RADIO_DATA_ROOT", "/path/to/radio/data"), date, "transcript")

    input_csv  = os.path.join(base_dir, f"{date}_inference_result_ratio.csv")
    output_csv = os.path.join(base_dir, f"{date}-blocks.csv")

    if not os.path.exists(input_csv):
        print(f"❌ Input CSV missing: {input_csv}")
        sys.exit(1)

    print(f"📥 Loading: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"   Total segments : {len(df)}")
    print(f"   Label counts   :\n{df['Predicted_Label'].value_counts()}")

    print("\n🧱 Merging blocks by Predicted_Label...")
    blocks = merge_blocks_by_label(df)

    print("🔗 Merging consecutive same-type blocks (DJ+GUEST → DJ/GUEST)...")
    final_blocks = merge_consecutive_same_blocks(blocks)

    final_blocks.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n✅ Saved: {output_csv}")
    print(f"   Total blocks:\n{final_blocks['block_type'].value_counts()}")
    print(f"\n📊 Block preview:")
    print(final_blocks[["block_type", "start", "end", "duration", "segments"]].to_string(index=False))


if __name__ == "__main__":
    main()
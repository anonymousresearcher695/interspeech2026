#!/usr/bin/env python3
import whisperx
from whisperx.diarize import DiarizationPipeline
import torch
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def is_hallucination(text):
    text = text.strip()
    if len(text) < 2: return True
    blacklist = ["한글자막", "자막 by", "Subtitle", "시청해 주셔서", "구독과 좋아요", "알림 설정", "좋아요", "구독", "다음 주에 만나요", "다음 영상에서"]
    for word in blacklist:
        if word.lower() in text.lower():
            if len(text) < 15 or text == word: return True
    if len(text) > 5 and len(set(text)) < 3: return True
    if len(text) > 20:
        words = text.split()
        if len(words) > 4 and len(set(words)) < len(words) / 2: return True
    return False


def seconds_to_srt_time(seconds):
    """초 → SRT 타임코드 (HH:MM:SS,mmm)"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ============================================================
# 경로 설정
# ============================================================
if len(sys.argv) != 2:
    print("Usage: python whisperx_radio.py <date_or_filepath>")
    exit(1)

INPUT_ARG = sys.argv[1]
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ Error: HF_TOKEN 환경 변수가 설정되지 않았습니다.")
    exit(1)

if os.path.isfile(INPUT_ARG):
    AUDIO_FILE = INPUT_ARG
    DATE = Path(AUDIO_FILE).stem
    OUTPUT_DIR = Path(AUDIO_FILE).parent.parent / "transcript"
else:
    DATE = INPUT_ARG
    _data_root = os.environ.get("RADIO_DATA_ROOT", "/path/to/radio/data")
    BASE_DIR = os.path.join(_data_root, DATE)
    AUDIO_FILE = f"{BASE_DIR}/mp3/{DATE}.mp3"
    OUTPUT_DIR = Path(f"{BASE_DIR}/transcript")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TEXT = OUTPUT_DIR / f"{DATE}_whisperx.txt"
OUTPUT_SRT  = OUTPUT_DIR / f"{DATE}_whisperx.srt"

if not os.path.exists(AUDIO_FILE):
    print(f"❌ Audio file not found: {AUDIO_FILE}")
    exit(1)

# ============================================================
# WhisperX 실행
# ============================================================
device = "cuda"
batch_size = 32
compute_type = "float16"

print(f"🚀 [A6000] Loading WhisperX large-v3 for: {DATE}")

model = whisperx.load_model("large-v3", device, compute_type=compute_type, language="ko")
audio = whisperx.load_audio(AUDIO_FILE)
result = model.transcribe(audio, batch_size=batch_size)

print("⏳ Aligning timestamps...")
model_a, metadata = whisperx.load_align_model(language_code="ko", device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print("⏳ Diarizing speakers...")
diarize_model = DiarizationPipeline(token=HF_TOKEN, device=device)
diarize_segments = diarize_model(audio)
final_result = whisperx.assign_word_speakers(diarize_segments, result)

# ============================================================
# 결과 저장
# ============================================================
print(f"💾 Saving results to: {OUTPUT_DIR}")

hallucination_count = 0
valid_segments = []

for segment in final_result["segments"]:
    text = segment['text'].strip()
    if is_hallucination(text):
        hallucination_count += 1
        continue
    valid_segments.append({
        "start":   segment['start'],
        "end":     segment['end'],
        "speaker": segment.get('speaker', 'UNKNOWN'),
        "text":    text,
    })

# TXT 저장
with open(OUTPUT_TEXT, "w", encoding="utf-8") as f:
    for seg in valid_segments:
        f.write(f"[{seg['start']:07.2f} - {seg['end']:07.2f}] {seg['speaker']}: {seg['text']}\n")

# SRT 저장
with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
    for idx, seg in enumerate(valid_segments, start=1):
        start_tc = seconds_to_srt_time(seg['start'])
        end_tc   = seconds_to_srt_time(seg['end'])
        f.write(f"{idx}\n")
        f.write(f"{start_tc} --> {end_tc}\n")
        f.write(f"[{seg['speaker']}] {seg['text']}\n")
        f.write("\n")

print(f"✅ ALL DONE! Filtered: {hallucination_count}")
print(f"   📄 TXT: {OUTPUT_TEXT}")
print(f"   🎬 SRT: {OUTPUT_SRT}")
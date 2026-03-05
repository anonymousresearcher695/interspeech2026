# Automatic Structuring of Radio Broadcast Streams

This repository contains the implementation for our paper submitted to Interspeech 2026.

> **Anonymous submission** — author information withheld for double-blind review.

---

## Overview

We propose a system for automatically segmenting radio broadcast audio into four semantic categories: **DJ talk/Guest talk**, **Music**, and **Advertisement**. The system combines a speaker-diarization-based baseline with two complementary modules that improve music and advertisement detection.

### Pipeline

```

Preprocessed Audio (.mp3)
    |
    v
[1. ASR + Speaker Diarization]  (WhisperX)
    |
    v
[2. Baseline Labeling]          ('DJ/GUEST'/MUSIC/AD via speaker statistics)
    |
    +---> [3. Music Detection]  (Audio acoustic features: dB, RMS..)
    |
    +---> [4. Ad Detection]      (Audio fingerprinting: Panako)
                |
                +---> [5. Individual Ad Identification]  (LLM: GPT-4o)
```

---

## Method

### Stage 0 - Preprocessing data
Vocals only, Musics only


### Stage 1 & 2 — Baseline (Speaker-Diarization-Based Labeling)

**`dj/whisperX.py`** — Runs WhisperX (large-v3) with speaker diarization on the raw broadcast audio, producing a time-stamped transcript with per-segment speaker IDs.

**`dj/speaker_ratio.py`** — Parses the WhisperX output into a structured CSV. Gaps of 100 seconds or more between speech segments are inserted as `Music` segments.

**`dj/dj_stat.py`** — Labels each segment using the following heuristics:
- **DJ**: Among the top-3 speakers by total duration (excluding those appearing in the first 2 seconds, which are typically time-signal or pre-roll), the one who speaks earliest is identified as the DJ.
- **GUEST**: Speakers who alternate turns with the DJ at least 20 times (or above a relative threshold).
- **MUSIC**: Segments inserted from long silence gaps (>= 100 s).
- **AD**: All remaining unlabeled segments.

**`dj/merge_block.py`** — Merges consecutive same-type segments into blocks. Adjacent DJ and GUEST segments are merged into `DJ/GUEST` blocks.

### Stage 3 — Music detection (Audio Feature-Based)

The 100-second gap heuristic in the baseline misses short music segments and is sensitive to speech-over-music. This module operates on source-separated audio (music stem and vocal stem, e.g. via Demucs) to detect music regions more accurately.

**`music/detect_selection_music.py`** — `SelectionMusicDetector` runs a 4-step pipeline:
1. **High-energy detection**: computes RMS energy on the music stem; frames above -38 dB are marked as candidates.
2. **Segment merging**: adjacent candidate segments within a 20-second gap are merged.
3. **Duration filter**: merged segments shorter than 60 seconds are discarded.
4. **Vocal/music refinement**: for each remaining segment, per-frame RMS is compared between the vocal and music stems. Frames where music is active and vocals are absent (instrumental) or closely mixed (singing) are retained. Segments passing a 70% smoothed validity threshold and lasting at least 100 seconds are emitted as final music intervals, saved to `{date}-selection_music.csv`.

**`music/extract_playlist.py`** — Uses GPT-4o to extract song title and artist information by analyzing DJ talk context (up to 400 s window) surrounding each detected music block. Handles ASR transcription errors via prompt-level correction rules.

**`music/auto_eval.py`** — Batch runner: iterates over a date range and calls `extract_playlist.py` for each date that has a `selection_music.csv` present.

**`music/evaluate_music_blocks.py`** / **`music/evaluate_music_overall_all.py`** — Evaluation scripts comparing detected music intervals against manually annotated ground truth (SBS, KBS, MBC) using IoU-based metrics.

### Stage 4 — Advertisement Detection (Audio Fingerprinting)

**`ad/cluster-max.py`** — Clusters candidate clips into ad blocks. Clips within a 30-second gap are merged into the same cluster.

**`ad/whisper_ad_faster.py`** — Transcribes detected ad clips using faster-whisper to enable product-name-level identification.

---

## Requirements

```bash
pip install -r requirements.txt
```

External dependencies:
- [WhisperX](https://github.com/m-bain/whisperX)
- [Panako](https://github.com/JorenSix/Panako) (audio fingerprinting)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- HuggingFace token for pyannote speaker diarization

---

## Setup

```bash
# Required environment variables
export RADIO_DATA_ROOT=/path/to/your/broadcast/data
export HF_TOKEN=your_huggingface_token
export OPENAI_API_KEY=your_openai_key   # only for playlist extraction
```

Expected data directory structure:
```
$RADIO_DATA_ROOT/
└── {YYYYMMDD}/
    ├── mp3/
    │   └── {YYYYMMDD}.mp3
    └── transcript/          # outputs written here
```

---

## Usage

```bash
# Stage 1: ASR + diarization
python dj/whisperX.py 20241124

# Stage 2: Parse to CSV
python dj/speaker_ratio.py 20241124

# Stage 2: Baseline labeling
python dj/dj_stat.py 20241124

# Stage 2: Merge into blocks
python dj/merge_block.py 20241124

# Stage 4: Ad detection (Panako must be indexed beforehand)
python ad/detector.py --date 20241124 --input-dir /path/to/clips/

# Stage 4: Cluster ad results
python ad/cluster-max.py panako_clips_20241124.csv

# Stage 5: Playlist extraction (requires OpenAI key)
python music/extract_playlist.py MBC 20241124
```

---

## Data

Sample outputs for 30 broadcast episodes (KBS, MBC, SBS) are provided in `data_sample`.

The raw audio files are not included due to copyright restrictions.

---

## License

MIT License

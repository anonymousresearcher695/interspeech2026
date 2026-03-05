import pandas as pd
import sys
import os
import json
import numpy as np
from openai import OpenAI
from config import OUTPUT_DIR

# ===========================
# 1. Configuration & Client Setup
# ===========================
def load_config():
    """Load API key from config.py"""
    config = {}
    try:
        import config as config_module
        config = {
            'openai_api_key': getattr(config_module, 'OPENAI_API_KEY', None)
        }
        return config
    except ImportError:
        return {}

def get_openai_client(config):
    """Initialize OpenAI client for GPT-4o"""
    api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: OpenAI API key not found!")
        sys.exit(1)
    return OpenAI(api_key=api_key)

# ===========================
# 2. Data Loading & Context Extraction
# ===========================
def load_data(channel, date_str):
    """Load full blocks and detected music intervals"""
    channel_lower = channel.lower()
    base_path = os.path.join(OUTPUT_DIR, channel_lower, date_str)
    
    blocks_path = os.path.join(base_path, f"{date_str}-blocks.csv")
    selection_path = os.path.join(base_path, f"{date_str}-selection_music.csv")
    
    if not os.path.exists(blocks_path):
        print(f"❌ Full blocks CSV not found: {blocks_path}")
        return None, None
    if not os.path.exists(selection_path):
        print(f"❌ Selection music CSV not found: {selection_path}")
        return None, None
        
    return pd.read_csv(blocks_path), pd.read_csv(selection_path)

def get_optimized_context(full_df, target_start, target_end, window_sec=400):
    """
    Extracts timeline context. Increased window to 400s 
    to capture delayed DJ commentary after long songs.
    """
    search_start = target_start - window_sec
    search_end = target_end + window_sec
    
    context_df = full_df[(full_df['end'] >= search_start) & (full_df['start'] <= search_end)].copy()
    
    context_lines = []
    for _, row in context_df.iterrows():
        # Tag the music interval being analyzed
        is_target = " >>> [분석 대상 음악 구간] <<< " if (abs(row['start'] - target_start) < 0.5) else ""
        block_type = row['block_type']
        text = row['text'] if pd.notna(row['text']) else ""
        
        line = f"- {row['start']:.1f}s ~ {row['end']:.1f}s | [{block_type}]{is_target}: {text}"
        context_lines.append(line)
        
    return "\n".join(context_lines)

# ===========================
# 3. LLM Processing with ASR Error Correction
# ===========================
def extract_song_info_with_llm(client, context_text, start_time, end_time):
    """Sends context to GPT-4o with strong ASR error correction rules"""
    
    prompt = f"""당신은 라디오 선곡표 추출 전문가입니다. 
제공된 [라디오 타임라인 맥락]에서 **{start_time:.1f}s ~ {end_time:.1f}s** 구간에 재생된 곡 정보를 추출하세요.

[라디오 타임라인 맥락]
{context_text}

[추출 및 ASR 오타 보정 규칙 - 필독]
1. **ASR 오타 허용**: 전사 데이터는 음성 인식 결과이므로 고유명사에 오타가 많습니다. 
   - 예: '제이리빗' -> '제이레빗', '바보레츠' -> '바버렛츠', '버블진트' -> '버벌진트'
   - 발음이 유사하다면 해당 아티스트/곡으로 인정하고 '표준 맞춤법'으로 교정하여 출력하세요.
2. **사후 소개 우선**: DJ가 "방금 들으신 곡은 ~", "듣고 왔습니다"라고 말하면, 그 멘트 바로 앞의 [분석 대상 음악 구간]이 해당 곡입니다.
3. **사전 예고**: DJ가 "다음 곡은 ~입니다", "듣겠습니다"라고 하면 바로 뒤의 음악 구간이 해당 곡입니다.
4. **절대 금지**: 이전 블록에서 추출한 정보를 현재 블록에 재사용하지 마세요. 증거가 없으면 비워두세요.

[출력 형식 (JSON)]
```json
{{
    "songs": [
        {{
            "title": "교정된 표준 곡명",
            "artist": "교정된 표준 아티스트명",
            "confidence": "high/medium/low",
            "reasoning": "타임라인의 구체적인 구절(예: '첫 곡은요 제이리빗...')을 근거로 설명"
        }}
    ]
}}
```"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a radio playlist auditor. You excel at correcting ASR transcription errors and matching DJ talk to the correct music blocks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception:
        return {"songs": []}

# ===========================
# 4. Execution Flow
# ===========================
def process_playlist(channel, date_str):
    print(f"\n{'='*70}\n[ {channel.lower()} | {date_str} ] Semantic Music Extraction\n{'='*70}\n")
    
    config = load_config()
    client = get_openai_client(config)
    blocks_df, selection_df = load_data(channel, date_str)
    
    if blocks_df is None or selection_df is None:
        return None

    results = []
    for i, row in selection_df.iterrows():
        print(f"[{i+1}/{len(selection_df)}] Interval: {row['start']:.1f}s ~ {row['end']:.1f}s")
        
        # Increase window to 400s to handle long talk segments
        context_text = get_optimized_context(blocks_df, row['start'], row['end'], window_sec=400)
        
        llm_result = extract_song_info_with_llm(client, context_text, row['start'], row['end'])

        if llm_result.get('songs'):
            for song in llm_result['songs']:
                print(f"  ✨ Extracted: {song['title']} - {song['artist']} ({song['confidence']})")
                results.append({
                    'music_index': i + 1,
                    'start': row['start'], 'end': row['end'],
                    'title': song['title'], 'artist': song['artist'],
                    'confidence': song['confidence'],
                    'reasoning': song.get('reasoning', '')
                })
        else:
            print(f"  ❌ No definitive info found.")
            results.append({
                'music_index': i + 1, 'start': row['start'], 'end': row['end'],
                'title': None, 'artist': None, 'confidence': 'none',
                'reasoning': 'No mention in surrounding context.'
            })
    return results

def save_output(channel, date_str, playlist):
    save_dir = os.path.join(OUTPUT_DIR, channel.lower(), date_str)
    os.makedirs(save_dir, exist_ok=True)
    
    pd.DataFrame(playlist).to_csv(os.path.join(save_dir, f"{date_str}-playlist.csv"), index=False, encoding='utf-8-sig')
    print(f"\n✅ Results saved to: {save_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_playlist.py <CHANNEL> <YYYYMMDD>")
        sys.exit(1)
        
    channel_in = sys.argv[1]
    date_in = sys.argv[2]
    
    final_playlist = process_playlist(channel_in, date_in)
    
    if final_playlist:
        save_output(channel_in, date_in, final_playlist)
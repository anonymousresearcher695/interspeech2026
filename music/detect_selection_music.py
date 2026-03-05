import librosa
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from config import OUTPUT_DIR

class SelectionMusicDetector:
    def __init__(self, date_str):
        self.date_str = date_str
        self.base_dir = f"{OUTPUT_DIR}{date_str}"
        
        self.musics_mp3 = f"{self.base_dir}/{date_str}_musics.mp3"
        self.vocals_mp3 = f"{self.base_dir}/{date_str}_vocals.mp3"
    
    def detect(self):
        print(f"\n{'='*70}")
        print(f"🎵 Selection Music Detection: {self.date_str}")
        print(f"{'='*70}\n")
        
        print("📍 [Step 1/4] Detecting high-energy regions...")
        candidates = self.detect_high_energy_regions()
        print(f"   → {len(candidates)} candidates found")
        
        print("\n🔗 [Step 2/4] Merging adjacent segments...")
        merged_raw = self.merge_adjacent(candidates, gap=20)
        print(f"   → {len(candidates)} → {len(merged_raw)} after merge")
        
        print("\n⏱️  [Step 3/4] Filtering by duration (≥ 60s)...")
        merged_raw = [c for c in merged_raw if c['duration'] >= 60]
        print(f"   → {len(merged_raw)} regions remain")
        
        print("\n🎤 [Step 4/4] Extracting true music & Fade-out separation...")
        selections = []
        for i, seg in enumerate(merged_raw):
            print(f"   Segment {i+1} [Raw: {seg['start']:.1f}s - {seg['end']:.1f}s]:")
            
            refined_segs = self.refine_and_split_segment(seg)
            
            if refined_segs:
                for j, r_seg in enumerate(refined_segs):
                    selections.append(r_seg)
                    print(f"      → Valid Music {j+1}: {r_seg['start']:.1f}s - {r_seg['end']:.1f}s (duration: {r_seg['duration']:.1f}s)")
            else:
                print(f"      → DJ/Ad Only")
        
        if selections:
            print(f"\n   Final segments:")
            for i, seg in enumerate(selections, 1):
                print(f"      [{i}] {seg['start']:.1f}s - {seg['end']:.1f}s "
                      f"(duration: {seg['duration']:.1f}s)")
        else:
            print("\n   No valid selection music found.")
        
        self.save_timestamps(selections)
        self.print_statistics(selections)
        
        return selections
    
    def detect_high_energy_regions(self):
        y, sr = librosa.load(self.musics_mp3, sr=22050)
        frame_length = int(sr * 0.1)
        hop_length = int(sr * 0.05)
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms))
        
        high_energy = rms_db > -38
        times = librosa.frames_to_time(range(len(rms_db)), sr=sr, hop_length=hop_length)
        
        segments = []
        in_music = False
        start = 0
        for i, is_high in enumerate(high_energy):
            if is_high and not in_music:
                start = times[i]
                in_music = True
            elif not is_high and in_music:
                if times[i] - start > 3:
                    segments.append({'start': start, 'end': times[i], 'duration': times[i] - start})
                in_music = False
        return segments
    
    def merge_adjacent(self, segments, gap=20):
        if not segments: return []
        sorted_segs = sorted(segments, key=lambda x: x['start'])
        merged = []
        current = sorted_segs[0].copy()
        
        for next_seg in sorted_segs[1:]:
            if next_seg['start'] - current['end'] <= gap:
                current['end'] = next_seg['end']
                current['duration'] = current['end'] - current['start']
            else:
                merged.append(current)
                current = next_seg.copy()
        merged.append(current)
        return merged
    
    def refine_and_split_segment(self, segment):
        try:
            vocals, sr = librosa.load(self.vocals_mp3, offset=segment['start'], duration=segment['duration'])
            musics, _ = librosa.load(self.musics_mp3, offset=segment['start'], duration=segment['duration'])
            
            min_len = min(len(vocals), len(musics))
            vocals, musics = vocals[:min_len], musics[:min_len]
            
            frame_length = int(sr * 0.5)
            hop_length = int(sr * 0.25)
            frame_duration = hop_length / sr  
            
            v_rms = librosa.feature.rms(y=vocals, frame_length=frame_length, hop_length=hop_length)[0]
            m_rms = librosa.feature.rms(y=musics, frame_length=frame_length, hop_length=hop_length)[0]
            
            v_db = librosa.amplitude_to_db(v_rms, ref=1.0)
            m_db = librosa.amplitude_to_db(m_rms, ref=1.0)
            m_mean = np.mean(m_db)
            
            
            is_active = m_db > (m_mean - 15)
            is_inst = is_active & (v_db < (m_db - 20))
            is_song = is_active & (np.abs(m_db - v_db) <= 15)
            valid_frames = is_inst | is_song
            
            window_frames = int(10 / frame_duration)
            if len(valid_frames) < window_frames:
                return []
                
            smoothed = pd.Series(valid_frames).rolling(window=window_frames, center=True).mean().fillna(0).values
            valid_indices = np.where(smoothed >= 0.70)[0]
            if len(valid_indices) == 0:
                return []
                
            
            gap_frames = int(15 / frame_duration)
            split_points = np.where(np.diff(valid_indices) > gap_frames)[0]
            
            sub_segments = []
            start_idx = 0
            for sp in split_points:
                sub_segments.append((valid_indices[start_idx], valid_indices[sp]))
                start_idx = sp + 1
            sub_segments.append((valid_indices[start_idx], valid_indices[-1]))
            
            
            padding_frames = int(2.5 / frame_duration)
            max_frame = len(m_db) - 1
            padded_segments = []
            for s_idx, e_idx in sub_segments:
                p_s = max(0, s_idx - padding_frames)
                p_e = min(max_frame, e_idx + padding_frames)
                padded_segments.append((p_s, p_e))
                
        
            final_segments = []
            dip_frames_threshold = int(1.5 / frame_duration)
            
            for s_idx, e_idx in padded_segments:
                seg_db = m_db[s_idx:e_idx+1]
                
                is_dip = (seg_db < (m_mean - 15)) | (seg_db < -40)
                dip_indices = np.where(is_dip)[0]
                
                if len(dip_indices) == 0:
                    final_segments.append((s_idx, e_idx))
                    continue
                    
                diffs = np.diff(dip_indices)
                break_points = np.where(diffs > 1)[0]
                
                block_starts = [dip_indices[0]] + dip_indices[break_points + 1].tolist()
                block_ends = dip_indices[break_points].tolist() + [dip_indices[-1]]
                
                last_idx = 0
                for b_start, b_end in zip(block_starts, block_ends):
                    if (b_end - b_start + 1) >= dip_frames_threshold:
                        if b_start > last_idx:
                            final_segments.append((s_idx + last_idx, s_idx + b_start - 1))
                        last_idx = b_end + 1
                        
                
                if last_idx <= (e_idx - s_idx):
                    final_segments.append((s_idx + last_idx, e_idx))
                    
            refined_segs = []
            for s_idx, e_idx in final_segments:
                s_time = segment['start'] + (s_idx * frame_duration)
                e_time = segment['start'] + (e_idx * frame_duration)
                dur = e_time - s_time
                
                if dur >= 100:
                    refined_segs.append({
                        'start': s_time,
                        'end': e_time,
                        'duration': dur
                    })
                    
            return refined_segs
            
        except Exception as e:
            print(f"      ⚠️ Trim failed: {e}")
            return []
            
    def save_timestamps(self, segments):
        csv_file = f"{self.base_dir}/{self.date_str}-selection_music.csv"
        pd.DataFrame(segments).to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        txt_file = f"{self.base_dir}/{self.date_str}-selection_timestamps.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"# Selection Music Timestamps: {self.date_str}\n")
            f.write(f"# Total: {len(segments)} segments\n\n")
            for i, seg in enumerate(segments, 1):
                f.write(f"[{i}] {self.format_time(seg['start'])} - {self.format_time(seg['end'])} ({seg['duration']:.1f}s)\n")
        print(f"\n💾 Saved:\n   {csv_file}\n   {txt_file}")
    
    def format_time(self, seconds):
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def print_statistics(self, segments):
        if not segments: return
        total = sum(s['duration'] for s in segments)
        avg = np.mean([s['duration'] for s in segments])
        
        print(f"\n{'='*70}\n📊 Statistics\n{'='*70}")
        print(f"\n🎵 Total: {len(segments)} songs")
        print(f"⏱️  Duration: {total:.1f}s ({total/60:.1f}min)")
        print(f"📏 Average: {avg:.1f}s ({avg/60:.1f}min)\n\n📋 List:")
        for i, seg in enumerate(segments, 1):
            print(f"   [{i}] {self.format_time(seg['start'])} - {self.format_time(seg['end'])} ({seg['duration']/60:.1f}min)")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect_selection_music.py <YYYYMMDD>")
        sys.exit(1)
    detector = SelectionMusicDetector(sys.argv[1])
    detector.detect()
import os
import sys
import shutil
import subprocess
from datetime import datetime, timedelta

from config import BASE_DIR
from config import OUTPUT_DIR

# modify here! --------------------------
START_DATE = "20260101"       
END_DATE   = "20260115"
BROADCAST = "KBS" 
# modify here! ---------------------------

BASE_PATH = BASE_DIR


def run_command(cmd):
    print(f"🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def pipeline(date_str):
    print(f"Starting Pipeline for {date_str}...")

    # print("[Step 1] Evaluating music blocks for date...")
    # run_command(["python", "evaluate_music_blocks.py", BROADCAST , date_str])

    print("[Step 2] extract to LLM...")
    run_command(["python", "extract_playlist.py", BROADCAST , date_str])

    print(f"\n🎉 All Done for {date_str}!")


def main():
    start = datetime.strptime(START_DATE, "%Y%m%d")
    end = datetime.strptime(END_DATE, "%Y%m%d")

    current = start
    processed = 0
    skipped = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"Start date: {START_DATE}")
    print(f"End date: {END_DATE}")
    print(f"{'='*60}\n")
    
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        
        print(f"\n{'='*60}")
        print(f"date: {date_str}")
        print(f"{'='*60}")

        target_dir = os.path.join(OUTPUT_DIR, BROADCAST.lower(), date_str, f"{date_str}-selection_music.csv")
        if os.path.exists(target_dir):
            print(f"founded file: {target_dir}")
            try:
                pipeline(date_str)
                print(f"{date_str} Complete!")
                processed += 1
            except subprocess.CalledProcessError as e:
                print(f"{date_str} An Error occured!")
                failed += 1
                with open("error_log.txt", "a") as f:
                    f.write(f"{date_str}: Pipeline Failed - {e}\n")
        else:
            print(f"file not found (Skip): {target_dir}")
            skipped += 1
        
        current += timedelta(days=1)

    # print("[Final process] overall evaiatoion")
    # run_command(["python", "evaluate_music_overall_all.py"])
    

if __name__ == "__main__":
    main()

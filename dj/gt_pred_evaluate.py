#!/usr/bin/env python3
"""
Block Classification Evaluation System
블록 분류 성능 평가 (Ground Truth vs Predicted)

평가 지표:
1. Block Classification Accuracy (블록 분류 정확도)
2. Temporal Accuracy (시간 정확도)
3. Block Boundary Precision (경계 정확도)
4. Confusion Matrix (혼동 행렬)
5. Per-Class Performance (클래스별 성능)
"""
import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 평가에 사용할 레이블 (4개 클래스)
# DJ/GUEST → DJ 로 정규화: 게스트는 항상 DJ 코너 안에서만 등장하므로
# DJ/GUEST 구간도 "DJ 있음"으로 취급
ALL_LABELS = ['DJ', 'MUSIC', 'AD', 'GUEST']

# 타입 정규화 규칙
TYPE_NORMALIZE = {
    'DJ/GUEST': 'DJ',
}

# Ground Truth
GROUND_TRUTH = {
    '20241123': [
        {'start': 09.05, 'end': 28.55, 'type': 'AD'},
        {'start': 29.17, 'end': 142.05, 'type': 'DJ'},  
        {'start': 42.05, 'end': 335.99, 'type': 'MUSIC' }, #Juice (Breakbot Mix),Lizzo
        {'start': 335.99, 'end': 449.60, 'type': 'DJ'},
        {'start': 449.60, 'end': 549.67, 'type': 'AD'},
        {'start': 569.78, 'end': 629.61, 'type': 'AD'},
        {'start': 645.09, 'end': 791.20, 'type': 'MUSIC'}, # I Ain't Worried,OneRepublic
        {'start': 791.20, 'end': 943.88, 'type': 'DJ'},
        {'start': 943.88, 'end': 1381.02, 'type': 'MUSIC'},#Loving You, Minnie Riperton / Your Song (From "Rocketman"),Taron Egerton
        {'start': 1381.02, 'end': 1538.09, 'type': 'DJ' },
        {'start': 1538.09, 'end': 1662.94, 'type': 'MUSIC'}, #봄날은 간다,조유리
        {'start': 1745.00, 'end': 1825.09, 'type': 'AD'},
        {'start': 1925.15, 'end': 2075.98, 'type': 'MUSIC'}, #The Life of Riley (From "Inside Out 2"/Score),Andrea Datzman
        {'start': 2075.98, 'end': 2431.82, 'type': 'DJ'},
        {'start': 2431.82, 'end': 2609.62, 'type': 'MUSIC'}, #내게 온 사랑,이형석
        {'start': 2609.62, 'end': 2717.10, 'type': 'DJ'},
        {'start': 2717.10, 'end': 2983.88, 'type': 'MUSIC'}, #처음부터 지금까지,Ryu (류)
        {'start': 2983.88, 'end': 3123.81, 'type': 'AD'},
        {'start': 3158.27, 'end': 3269.47, 'type': 'DJ'},
        {'start': 3269.47, 'end': 3445.14, 'type': 'MUSIC'}, #첫눈처럼 너에게 가겠다,에일리
        {'start': 3445.14, 'end': 3469.44, 'type': 'AD'},
        {'start': 3489.98, 'end': 3549.47, 'type': 'AD'},
        {'start': 3589.00, 'end': 3620.05, 'type': 'AD'},
    ],
        '20241124': [
        {'start': 8.72, 'end': 28.23, 'type': 'AD'},
        {'start': 36.10, 'end': 144.34,'type': 'DJ'},
        {'start': 144.34, 'end': 342.95,'type': 'MUSIC'}, #A Winter Story (From 'Love Letter' Soundtrack),Remedios
        {'start': 342.95, 'end': 429.15,'type': 'DJ'},
        {'start': 429.15, 'end': 529.06,'type': 'AD'},
        {'start': 549.09, 'end': 608.97,'type': 'AD'},
        {'start': 608.97, 'end': 732.44,'type': 'MUSIC'}, #Moon River (From 'Breakfast at Tiffany's' Original Soundtrack),Audrey Hepburn
        {'start': 732.44, 'end': 888.38,'type': 'DJ'},
        {'start': 888.38, 'end': 1423.54,'type': 'MUSIC'}, #함께 있으면 좋을 사람,최재훈/가만히 눈을 감고 (New Version),정재욱
        {'start': 1423.54, 'end': 1548.77,'type': 'DJ'},
        {'start': 1548.77, 'end': 1663.22,'type': 'MUSIC'}, #You’re Christmas To Me,Sam Ryder
        {'start': 1744.57, 'end': 1823.86,'type': 'AD'},
        {'start': 1925.52, 'end': 2114.34,'type': 'MUSIC'}, #Eli's Theme (From "Let the Right One In"),City Of Prague Philharmonic Orchestra/Woo Hoo,5.6.7.8's
        {'start': 2114.34, 'end': 2228.81,'type': 'DJ'},
        {'start': 2228.81, 'end': 2406.25,'type': 'MUSIC'}, #Vacation,The Go-Go's
        {'start': 2406.25, 'end': 2508.18,'type': 'DJ'},
        {'start': 2508.18, 'end': 2614.67,'type': 'MUSIC'}, #누가 죄인인가,(정성화 ,조재윤 ,배정남 ,이현우)/Do You Hear The People Sing?,(Aaron Tveit ,Eddie Redmayne ,Students ,Les M)/Overture/And All That Jazz,Catherine Zeta-Jones
        {'start': 2614.67, 'end': 2633.05,'type': 'DJ'}, 
        {'start': 2633.05, 'end': 2740.20,'type': 'MUSIC'},#짜파구리,정재일
        {'start': 2740.20, 'end': 2792.44,'type': 'DJ'},
        {'start': 2792.44, 'end': 2992.32,'type': 'MUSIC'}, #Roxie,Renee Zellweger
        {'start': 2992.32, 'end': 3132.14,'type': 'AD'},
        {'start': 3166.76, 'end': 3255.89,'type': 'DJ'},
        {'start': 3255.89, 'end': 3444.63,'type': 'MUSIC'}, #Yes Sir, I Can Boogie,Baccara
        {'start': 3444.63, 'end': 3469.15,'type': 'AD'},
        {'start': 3489.76, 'end': 3549.44,'type': 'AD'},
        {'start': 3589.17, 'end': 3620.06,'type': 'AD'},
    ],
          '20241125': [
        {'start': 8.49, 'end': 28.02, 'type': 'AD'},
        {'start': 28.73, 'end': 134.71, 'type': 'DJ'},
        {'start': 134.71, 'end': 374.40, 'type': 'MUSIC'}, #You Got Me,Colbie Caillat
        {'start': 374.40, 'end': 513.08, 'type': 'DJ'},
        {'start': 513.08, 'end': 613.03, 'type': 'AD'},
        {'start': 633.12, 'end': 699.09, 'type': 'AD'},
        {'start': 713.55, 'end': 865.85, 'type': 'DJ'},
        {'start': 865.85, 'end': 967.37, 'type': 'MUSIC'}, #Hello Zepp,Charlie Clouser
        {'start': 967.37, 'end': 1093.34, 'type': 'DJ'}, 
        {'start': 1093.34, 'end': 1337.45, 'type': 'MUSIC'},#Ghostbusters,Ray Parker Jr.
        {'start': 1337.45, 'end': 1465.88, 'type': 'DJ'},
        {'start': 1465.88, 'end': 1662.88, 'type': 'MUSIC'}, #Streets Of Philadelphia,Bruce Springsteen
        {'start': 1744.37, 'end': 1823.61, 'type': 'AD'},
        {'start': 1925.07, 'end': 2050.25, 'type': 'MUSIC'}, #블루시티,Various Artists
        {'start': 2050.25, 'end': 2056.00, 'type': 'DJ'},
        {'start': 2056.00, 'end': 2071.86, 'type': 'MUSIC'}, #The Batman Theme,Danny Elfman
        {'start': 2071.86, 'end': 2327.07, 'type': 'DJ'},
        {'start': 2327.07, 'end': 2443.63, 'type': 'MUSIC'}, #Let It Snow! Let It Snow! Let It Snow!,Robbie Williams
        {'start': 2443.63, 'end': 2594.80, 'type': 'DJ'},
        {'start': 2594.80, 'end': 2766.20, 'type': 'MUSIC'}, #Snow Frolice,Various Artists
        {'start': 2766.20, 'end': 2906.16, 'type': 'AD'},
        {'start': 2940.55, 'end': 3142.23, 'type': 'DJ'}, #하모니,(제아 ,이영현)
        {'start': 3142.23, 'end': 3444.58, 'type': 'MUSIC'},
        {'start': 3449.67, 'end': 3469.09, 'type': 'AD'},
        {'start': 3489.50, 'end': 3549.34, 'type': 'AD'},
        {'start': 3588.47, 'end': 3620.06, 'type': 'AD'},
    ],
        '20241126': [
        {'start': 11.24, 'end': 30.96, 'type': 'AD'},
        {'start': 39.18, 'end': 139.31, 'type': 'DJ'},
        {'start': 139.31, 'end': 589.49, 'type': 'MUSIC'}, #Defying Gravity (Feat. Ariana Grande),Cynthia Erivo
        {'start': 589.49, 'end': 799.35, 'type': 'DJ'},
        {'start': 799.35, 'end': 909.44, 'type': 'AD'},
        {'start': 899.44, 'end': 999.34, 'type': 'AD'},
        {'start': 999.34, 'end': 1174.53, 'type': 'DJ'},
        {'start': 1174.53, 'end': 1426.16, 'type': 'MUSIC'}, #The Whole Nine Yards,(吉俣良 / Ryo Yoshimata)
        {'start': 1426.16, 'end': 1563.01, 'type': 'DJ'},
        {'start': 1563.01, 'end': 1665.57, 'type': 'MUSIC'}, #City Of Stars / May Finally Come True (Feat. Ryan Gosling, Emma Stone),Justin Hurwitz
        {'start': 1747.30, 'end': 1826.50, 'type': 'AD'},
        {'start': 1928.05, 'end': 1944.31, 'type': 'MUSIC'}, #Oompa Loompa,(Hugh Grant ,Timothée Chalamet)
        {'start': 1944.31, 'end': 2338.27, 'type': 'DJ/GUEST'},
        {'start': 2338.27, 'end': 2534.01, 'type': 'MUSIC'}, #Just Look Up (From Don’t Look Up),(Ariana Grande ,Kid Cudi)
        {'start': 2534.01, 'end': 2956.59, 'type': 'DJ/GUEST'},
        {'start': 2956.59, 'end': 3096.48, 'type': 'AD'},
        {'start': 3116.94, 'end': 3136.97, 'type': 'AD'},
        {'start': 3151.54, 'end': 3261.98, 'type': 'DJ'},
        {'start': 3261.98, 'end': 3447.47, 'type': 'MUSIC'}, #Popular,Ariana Grande
        {'start': 3452.46, 'end': 3471.82, 'type': 'AD'},
        {'start': 3492.27, 'end': 3551.63, 'type': 'AD'},
        {'start': 3591.26, 'end': 3620.05, 'type': 'AD'},
    ],
            '20241127': [
        {'start': 11.11, 'end': 30.53, 'type': 'AD'},
        {'start': 63.93, 'end': 130.17, 'type': 'DJ'},
        {'start': 130.17, 'end': 358.74, 'type': 'MUSIC'}, #Remembering You,Steven Curtis Chapman
        {'start': 358.74, 'end': 523.15, 'type': 'DJ'},
        {'start': 523.15, 'end': 602.74, 'type': 'AD'},
        {'start': 622.98, 'end': 743.15, 'type': 'AD'},
        {'start': 763.21, 'end': 1276.59, 'type': 'DJ/GUEST'},
        {'start': 1276.59, 'end': 1471.87, 'type': 'MUSIC'}, #1도 없어,Apink (에이핑크)
        {'start': 1471.87, 'end': 1630.67, 'type': 'DJ/GUEST'},
        {'start': 1630.67, 'end': 1665.62, 'type': 'MUSIC'}, #가오만사성,두번째달
        {'start': 1747.13, 'end': 1826.52, 'type': 'AD'},
        {'start': 1927.61, 'end': 2428.66, 'type': 'DJ/GUEST'},
        {'start': 2428.66, 'end': 2648.69, 'type': 'MUSIC'}, #괜찮아도 괜찮아 (That's okay),도경수(D.O.)
        {'start': 2648.69, 'end': 2951.24, 'type': 'DJ/GUEST'},
        {'start': 2951.24, 'end': 3091.40, 'type': 'AD'},
        {'start': 3111.72, 'end': 3131.59, 'type': 'AD'},
        {'start': 3183.77, 'end': 3375.48, 'type': 'DJ'},
        {'start': 3375.48, 'end': 3452.11, 'type': 'MUSIC'}, #어른,Sondia
        {'start': 3452.11, 'end': 3471.67, 'type': 'AD'},
        {'start': 3492.08, 'end': 3551.72, 'type': 'AD'},
        {'start': 3591.16, 'end': 3620.06, 'type': 'AD'},
    ],
                '20241128': [
        {'start': 11.00, 'end': 30.54, 'type': 'AD'},
        {'start': 63.66,'end': 141.19, 'type': 'DJ'}, 
        {'start': 141.19,'end': 268.56, 'type': 'MUSIC'}, #Prologue,John Williams
        {'start': 268.56,'end': 465.07, 'type': 'DJ'},
        {'start': 465.07,'end': 544.61, 'type': 'AD'},
        {'start': 564.97,'end': 685.00, 'type': 'AD'},
        {'start': 705.07,'end': 854.03, 'type': 'DJ'},
        {'start': 854.03,'end': 1044.18, 'type': 'MUSIC'}, #Immortals (From "Big Hero 6"/Soundtrack),Fall Out Boy
        {'start': 1044.18,'end': 1211.19, 'type': 'DJ'},
        {'start': 1211.19,'end': 1408.69, 'type': 'MUSIC'}, #Under the Tree (from “That Christmas”),Ed Sheeran
        {'start': 1408.69,'end': 1509.55, 'type': 'DJ'},
        {'start': 1509.55,'end': 1625.23, 'type': 'MUSIC'}, #Eres Tu,Mocedades
        {'start': 1796.97,'end': 1876.49, 'type': 'AD'},
        {'start': 1888.50,'end': 2416.98, 'type': 'DJ/GUEST'},
        {'start': 2416.98,'end': 2602.96, 'type': 'MUSIC'}, #저 너머로 (엔딩 크레딧 버전) (feat. Te Vaka), 나연(TWICE)
        {'start': 2602.96,'end': 2923.98, 'type': 'DJ/GUEST'},
        {'start': 2923.98,'end': 3063.81, 'type': 'AD'},
        {'start': 3084.04,'end': 3103.90, 'type': 'AD'},
        {'start': 3158.68,'end': 3255.84, 'type': 'DJ'},
        {'start': 3255.84,'end': 3411.71, 'type': 'MUSIC'}, #Christmas Tree,V
        {'start': 3411.71,'end': 3611.57, 'type': 'AD'},
        {'start': 3451.75,'end': 3511.58, 'type': 'AD'},
        {'start': 3550.98,'end': 3590.01, 'type': 'AD'},
    ],
        '20241129': [
        {'start': 11.00, 'end': 30.54, 'type': 'AD'},
        {'start': 38.50, 'end': 142.38, 'type': 'DJ'},
        {'start': 142.38, 'end': 311.13, 'type': 'MUSIC'}, #Such A Night (LP Version),Dr. John
        {'start': 311.13, 'end': 469.80, 'type': 'DJ'},
        {'start': 469.80, 'end': 570.13, 'type': 'AD'},
        {'start': 590.17, 'end': 690.17, 'type': 'AD'},
        {'start': 710.39, 'end': 884.78, 'type': 'DJ'},
        {'start': 884.78, 'end': 1058.29, 'type': 'MUSIC'}, #Bellbottoms,The Jon Spencer Blues Explosion
        {'start': 1058.29, 'end': 1300.47, 'type': 'DJ'},
        {'start': 1300.47, 'end': 1665.31, 'type': 'MUSIC'}, #Jump,Girls Aloud/#Do You Want to Build a Snowman? (From "Frozen"/Soundtrack Version),(Kristen Bell ,Agatha Lee Monn ,Katie Lopez)
        {'start': 1746.86, 'end': 1826.51, 'type': 'AD'}, 
        {'start': 1927.75, 'end': 2082.83, 'type': 'DJ'}, 
        {'start': 2082.83, 'end': 2096.92, 'type': 'MUSIC'}, #Have Yourself A Merry Little Christmas,98°
        {'start': 2096.92, 'end': 2111.10, 'type': 'GUEST'},
        {'start': 2111.10, 'end': 2120.13, 'type': 'MUSIC'}, #Have Yourself A Merry Little Christmas,98°
        {'start': 2120.13, 'end': 2669.66, 'type': 'DJ/GUEST'},
        {'start': 2669.66, 'end': 2694.62, 'type': 'MUSIC'}, #??
        {'start': 2694.62, 'end': 2778.70, 'type': 'DJ/GUEST'},
        {'start': 2778.70, 'end': 2797.99, 'type': 'MUSIC'},
        {'start': 2797.99, 'end': 2807.77, 'type': 'DJ/GUEST'}, 
        {'start': 2807.77, 'end': 2819.77, 'type': 'MUSIC'},#Raiders March (From "Indiana Jones and the Kingdom of the Crystal Skull" / Soundtrack Ver,John Williams
        {'start': 2819.77, 'end': 2880.17, 'type': 'DJ/GUEST'},
        {'start': 2880.17, 'end': 2913.19, 'type': 'MUSIC'}, #??
        {'start': 2913.19, 'end': 3004.29, 'type': 'DJ/GUEST'},
        {'start': 3004.29, 'end': 3018.31, 'type': 'MUSIC'},
        {'start': 3018.31, 'end': 3091.68, 'type': 'DJ/GUEST'}, 
        {'start': 3091.68, 'end': 3231.47, 'type': 'AD'},
        {'start': 3251.90, 'end': 3271.66, 'type': 'AD'},
        {'start': 3312.24, 'end': 3447.05, 'type': 'DJ'},
        {'start': 3452.09, 'end': 3471.55, 'type': 'AD'},
        {'start': 3491.83, 'end': 3551.67, 'type': 'AD'},
        {'start': 3590.88, 'end': 3620.06, 'type': 'AD'},
    ],
    

}



    
    
class BlockEvaluator:
    def __init__(self, date_str):
        self.date_str = date_str
        self.gt_blocks = GROUND_TRUTH.get(date_str, [])

        if not self.gt_blocks:
            print(f"\n❌ Error: No Ground Truth data for {date_str}")
            print(f"   Available dates: {', '.join(GROUND_TRUTH.keys())}")
            sys.exit(1)

    def load_predicted_blocks(self, csv_file):
        df = pd.read_csv(csv_file)
        predicted = []
        for _, row in df.iterrows():
            predicted.append({
                'start': float(row['start']),
                'end': float(row['end']),
                'type': row['block_type'].strip()
            })
        return predicted

    def evaluate_all(self, predicted_blocks):
        print(f"\n{'='*70}")
        print(f"📊 Block Classification Evaluation: {self.date_str}")
        print(f"{'='*70}\n")

        print("📍 [1/5] Block Classification Accuracy (IoU)...")
        block_metrics = self.block_classification_accuracy(predicted_blocks)

        print("\n⏱️  [2/5] Temporal Accuracy...")
        temporal_acc = self.temporal_accuracy(predicted_blocks)

        print("\n🎯 [3/5] Block Boundary Evaluation...")
        boundary_metrics = self.boundary_evaluation(predicted_blocks)

        print("\n🔀 [4/5] Confusion Matrix...")
        conf_matrix, labels = self.confusion_matrix_analysis(predicted_blocks)

        print("\n📈 [5/5] Per-Class Performance...")
        class_report = self.per_class_performance(predicted_blocks)

        self.print_summary(
            block_metrics, temporal_acc, boundary_metrics,
            conf_matrix, labels, class_report
        )

        return {
            'block_accuracy': block_metrics,
            'temporal_accuracy': temporal_acc,
            'boundary_metrics': boundary_metrics,
            'confusion_matrix': conf_matrix,
            'class_report': class_report
        }

    def block_classification_accuracy(self, predicted):
        correct_50 = 0
        correct_75 = 0
        total = len(self.gt_blocks)

        matched_pred_50 = set()
        matched_pred_75 = set()

        for gt in self.gt_blocks:
            gt_type = gt['type']
            gt_duration = gt['end'] - gt['start']

            best_iou = 0.0
            best_pred_idx = -1

            for i, pred in enumerate(predicted):
                if pred['type'] != gt_type:
                    continue

                overlap_start = max(gt['start'], pred['start'])
                overlap_end = min(gt['end'], pred['end'])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > 0:
                    pred_duration = pred['end'] - pred['start']
                    union = gt_duration + pred_duration - overlap
                    iou = overlap / union

                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = i

            if best_iou >= 0.5 and best_pred_idx not in matched_pred_50:
                correct_50 += 1
                matched_pred_50.add(best_pred_idx)

            if best_iou >= 0.75 and best_pred_idx not in matched_pred_75:
                correct_75 += 1
                matched_pred_75.add(best_pred_idx)

        acc_50 = (correct_50 / total * 100) if total > 0 else 0
        acc_75 = (correct_75 / total * 100) if total > 0 else 0

        print(f"   Total GT blocks: {total}")
        print(f"   Correct (IoU >= 0.5): {correct_50} ({acc_50:.4f}%)")
        print(f"   Correct (IoU >= 0.75): {correct_75} ({acc_75:.4f}%)")

        return {'acc_50': acc_50, 'acc_75': acc_75}

    def temporal_accuracy(self, predicted):
        total_duration = max(
            self.gt_blocks[-1]['end'],
            predicted[-1]['end'] if predicted else 0
        )

        correct_time = 0
        for t in range(int(total_duration)):
            gt_label = self.get_label_at_time(self.gt_blocks, t)
            pred_label = self.get_label_at_time(predicted, t)
            if gt_label == pred_label:
                correct_time += 1

        accuracy = (correct_time / total_duration * 100) if total_duration > 0 else 0

        print(f"   Total duration: {total_duration:.0f}s ({total_duration/60:.1f}min)")
        print(f"   Correct time: {correct_time}s ({correct_time/60:.1f}min)")
        print(f"   Accuracy: {accuracy:.4f}%")

        return accuracy

    def get_label_at_time(self, blocks, time):
        for block in blocks:
            if block['start'] <= time < block['end']:
                raw = block['type']
                return TYPE_NORMALIZE.get(raw, raw)
        return 'SILENCE'

    def boundary_evaluation(self, predicted, tolerance=5):
        gt_boundaries = [b['end'] for b in self.gt_blocks[:-1]]
        pred_boundaries = [b['end'] for b in predicted[:-1]] if len(predicted) > 0 else []

        correct_recall = 0
        for gt_b in gt_boundaries:
            if any(abs(gt_b - pred_b) <= tolerance for pred_b in pred_boundaries):
                correct_recall += 1

        correct_precision = 0
        for pred_b in pred_boundaries:
            if any(abs(pred_b - gt_b) <= tolerance for gt_b in gt_boundaries):
                correct_precision += 1

        recall = (correct_recall / len(gt_boundaries) * 100) if len(gt_boundaries) > 0 else 0
        precision = (correct_precision / len(pred_boundaries) * 100) if len(pred_boundaries) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        print(f"   GT Boundaries: {len(gt_boundaries)} | Predicted Boundaries: {len(pred_boundaries)}")
        print(f"   Precision (±{tolerance}s): {precision:.4f}%")
        print(f"   Recall    (±{tolerance}s): {recall:.4f}%")
        print(f"   F1-Score  (±{tolerance}s): {f1_score:.4f}%")

        return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

    def confusion_matrix_analysis(self, predicted):
        total_duration = int(max(
            self.gt_blocks[-1]['end'],
            predicted[-1]['end'] if predicted else 0
        ))

        y_true = []
        y_pred = []

        for t in range(total_duration):
            gt_label = self.get_label_at_time(self.gt_blocks, t)
            pred_label = self.get_label_at_time(predicted, t)
            if gt_label != 'SILENCE':
                y_true.append(gt_label)
                y_pred.append(pred_label)

        # DJ/GUEST 포함 5개 클래스
        labels = ALL_LABELS
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        print("\n   Confusion Matrix:")
        print(f"\n   Predicted →   {'  '.join(labels)}")
        print("       ↓")
        for i, label in enumerate(labels):
            row_str = f"   {label:8s}      "
            for j in range(len(labels)):
                pct = (cm[i][j] / cm[i].sum() * 100) if cm[i].sum() > 0 else 0
                row_str += f"{pct:5.4f}%  "
            print(row_str)

        return cm, labels

    def per_class_performance(self, predicted):
        total_duration = int(max(
            self.gt_blocks[-1]['end'],
            predicted[-1]['end'] if predicted else 0
        ))

        y_true = []
        y_pred = []

        for t in range(total_duration):
            gt_label = self.get_label_at_time(self.gt_blocks, t)
            pred_label = self.get_label_at_time(predicted, t)
            if gt_label != 'SILENCE':
                y_true.append(gt_label)
                y_pred.append(pred_label)

        # DJ/GUEST 포함 5개 클래스
        labels = ALL_LABELS
        report = classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0
        )

        print("\n   Per-Class Performance:")
        print(f"\n              Precision  Recall  F1-Score  Support")
        for label in labels:
            metrics = report[label]
            print(f"   {label:10s}   {metrics['precision']:.4f}     "
                  f"{metrics['recall']:.4f}    {metrics['f1-score']:.4f}     "
                  f"{int(metrics['support'])}")

        print(f"\n   Macro Avg      {report['macro avg']['precision']:.4f}     "
              f"{report['macro avg']['recall']:.4f}    "
              f"{report['macro avg']['f1-score']:.4f}")

        return report

    def print_summary(self, block_metrics, temporal_acc, boundary_metrics,
                     conf_matrix, labels, class_report):
        print(f"\n{'='*70}")
        print(f"📊 Evaluation Summary")
        print(f"{'='*70}\n")

        print(f"Block Acc (IoU >= 0.5):          {block_metrics['acc_50']:.4f}%")
        print(f"Block Acc (IoU >= 0.75):         {block_metrics['acc_75']:.4f}%")
        print(f"Temporal Accuracy:               {temporal_acc:.4f}%")
        print(f"Boundary F1-Score (±5s):         {boundary_metrics['f1_score']:.4f}%")

        print(f"\nPer-Class F1-Scores:")
        for label in ALL_LABELS:
            f1 = class_report[label]['f1-score']
            print(f"  {label:10s}: {f1:.4f}")

        print(f"\nMacro Average F1-Score:          "
              f"{class_report['macro avg']['f1-score']:.4f}")

        print(f"\n{'='*70}\n")

    def save_results(self, results, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Evaluation Results: {self.date_str}\n\n")

            f.write(f"Block Acc (IoU >= 0.5): {results['block_accuracy']['acc_50']:.4f}%\n")
            f.write(f"Block Acc (IoU >= 0.75): {results['block_accuracy']['acc_75']:.4f}%\n")
            f.write(f"Temporal Accuracy: {results['temporal_accuracy']:.4f}%\n")

            b_metrics = results['boundary_metrics']
            f.write(f"Boundary Precision: {b_metrics['precision']:.4f}%\n")
            f.write(f"Boundary Recall: {b_metrics['recall']:.4f}%\n")
            f.write(f"Boundary F1-Score: {b_metrics['f1_score']:.4f}%\n\n")

            f.write("Per-Class Performance:\n")
            # DJ/GUEST 포함 5개 클래스
            for label in ALL_LABELS:
                metrics = results['class_report'][label]
                f.write(f"  {label}: P={metrics['precision']:.4f}, "
                       f"R={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}\n")

        print(f"💾 Results saved: {output_file}")

    def plot_results(self, results, output_path=None):
        fig = plt.figure(figsize=(16, 12))

        ax1 = plt.subplot(2, 2, 1)
        self.plot_overall_metrics(ax1, results)

        ax2 = plt.subplot(2, 2, 2)
        self.plot_confusion_matrix(ax2, results)

        ax3 = plt.subplot(2, 2, 3)
        self.plot_per_class_performance(ax3, results)

        ax4 = plt.subplot(2, 2, 4)
        self.plot_class_distribution(ax4)

        plt.suptitle(f'Block Classification Evaluation: {self.date_str}',
                     fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()

        if output_path is None:
            output_path = f"{self.date_str}_evaluation.png"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved: {output_path}")
        plt.close()

    def plot_overall_metrics(self, ax, results):
        metrics = [
            results['block_accuracy']['acc_50'],
            results['temporal_accuracy'],
            results['boundary_metrics']['f1_score']
        ]

        labels = ['Block Acc\n(IoU>= 0.5)', 'Temporal\nAccuracy', 'Boundary\nF1-Score']
        colors = ['#2ecc71', '#3498db', '#9b59b6']

        bars = ax.bar(labels, metrics, color=colors, alpha=0.8, edgecolor='black')

        for bar in bars:
            height = bar.get_height()
            truncated = int(height * 10000) / 10000
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{truncated:.4f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% Target')

    def plot_confusion_matrix(self, ax, results):
        cm = results['confusion_matrix']
        # DJ/GUEST 포함 5개 클래스
        labels = ALL_LABELS
        cm_percent = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100

        im = ax.imshow(cm_percent, cmap='Blues', aspect='auto')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('True', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (%)', fontsize=14, fontweight='bold')

        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{cm_percent[i, j]:.1f}%',
                             ha="center", va="center",
                             color="white" if cm_percent[i, j] > 50 else "black",
                             fontsize=9, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Percentage (%)', fontsize=10)

    def plot_per_class_performance(self, ax, results):
        # DJ/GUEST 포함 5개 클래스
        labels = ALL_LABELS
        precision = [results['class_report'][l]['precision'] for l in labels]
        recall = [results['class_report'][l]['recall'] for l in labels]
        f1 = [results['class_report'][l]['f1-score'] for l in labels]

        x = np.arange(len(labels))
        width = 0.25

        bars1 = ax.bar(x - width, precision, width, label='Precision',
                      color='#e74c3c', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, recall, width, label='Recall',
                      color='#f39c12', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score',
                      color='#2ecc71', alpha=0.8, edgecolor='black')

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    def plot_class_distribution(self, ax):
        # DJ/GUEST 포함 5개 클래스
        class_counts = {label: 0 for label in ALL_LABELS}

        for block in self.gt_blocks:
            duration = block['end'] - block['start']
            block_type = block['type']
            if block_type in class_counts:
                class_counts[block_type] += duration

        # 0인 클래스 제외 (파이차트에서 보기 좋게)
        filtered = {k: v for k, v in class_counts.items() if v > 0}
        labels = list(filtered.keys())
        sizes = list(filtered.values())
        colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'][:len(labels)]
        explode = [0.05] * len(labels)

        wedges, texts, autotexts = ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90
        )

        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax.set_title('Ground Truth Distribution (by duration)',
                    fontsize=14, fontweight='bold')

        legend_labels = [f'{l}: {s/60:.1f}min' for l, s in zip(labels, sizes)]
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 0, 0.5, 1))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_blocks.py <YYYYMMDD>")
        print("Example: python evaluate_blocks.py 20260101")
        sys.exit(1)

    date_str = sys.argv[1]

    base_dir = f"/mnt/home_dnlab/jhjung/radio/movie/{date_str}/transcript/"

    possible_files = [
        f"{date_str}-blocks.csv"
    ]

    csv_file = None
    for filename in possible_files:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            csv_file = filepath
            break

    if not csv_file:
        print(f"❌ Error: CSV file not found in {base_dir}")
        print(f"   Looked for: {', '.join(possible_files)}")
        sys.exit(1)

    evaluator = BlockEvaluator(date_str)

    print(f"📥 Loading predicted blocks: {csv_file}")
    predicted_blocks = evaluator.load_predicted_blocks(csv_file)
    print(f"   → Loaded {len(predicted_blocks)} blocks")

    results = evaluator.evaluate_all(predicted_blocks)

    output_file = os.path.join(base_dir, f"{date_str}-evaluation.txt")
    evaluator.save_results(results, output_file)

    print(f"\n📊 Generating visualization...")
    output_graph = os.path.join(base_dir, f"{date_str}_evaluation.png")
    evaluator.plot_results(results, output_graph)

    print(f"\n✅ Evaluation completed!")
    print(f"   Text: {output_file}")
    print(f"   Graph: {output_graph}")
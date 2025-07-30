# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import sys
import io
import csv

# Windows에서 한글 출력을 위한 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# matplotlib 한글 폰트 설정
mpl.rcParams['font.family'] = 'Malgun Gothic'  # Windows의 맑은 고딕
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 코사인 유사도 계산 함수들
def cal_score_bmk(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)
    
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)).item()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# CSV 파일에서 테스트 데이터 읽기
test_pairs = []
try:
    with open('quiz.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 스킵
        for row in reader:
            if len(row) >= 3:
                # (문장1, 문장2, 정답) 형태로 저장
                test_pairs.append((row[0], row[1], int(row[2])))
    print(f"CSV 파일에서 {len(test_pairs)}개의 문장 쌍을 로드했습니다.")
except FileNotFoundError:
    print("quiz.csv 파일을 찾을 수 없습니다. 기본 테스트 데이터를 사용합니다.")
    # 기본 테스트 데이터 (파일이 없을 경우)
    test_pairs = [
        # 유사한 문장 쌍들 (GT = 1)
        ("영화를 보는 것을 좋아한다", "영화 감상을 즐긴다", 1),
        ("오늘 날씨가 정말 좋다", "날씨가 매우 화창하다", 1),
        ("저녁을 먹었다", "저녁 식사를 했다", 1),
        ("공부하는 것이 어렵다", "학습이 힘들다", 1),
        ("운동을 매일 한다", "매일 운동한다", 1),
        ("책을 읽는 것을 좋아한다", "독서를 즐긴다", 1),
        ("커피가 맛있다", "커피의 맛이 좋다", 1),
        ("음악을 듣고 있다", "음악 감상 중이다", 1),
        
        # 비유사한 문장 쌍들 (GT = 0)
        ("영화를 좋아한다", "영화가 싫다", 0),
        ("날씨가 좋다", "비가 많이 온다", 0),
        ("공부를 한다", "게임을 한다", 0),
        ("아침을 먹었다", "잠을 잔다", 0),
        ("운동을 좋아한다", "운동을 싫어한다", 0),
        ("책을 읽는다", "TV를 본다", 0),
        ("커피를 마신다", "차를 싫어한다", 0),
        ("음악이 시끄럽다", "조용한 것을 좋아한다", 0),
    ]

# 모델 로드
print("모델 로딩 중...")
# Model 1: BM-K/KoSimCSE-roberta
model1_name = 'BM-K/KoSimCSE-roberta'
bmk_model = AutoModel.from_pretrained(model1_name)
bmk_tokenizer = AutoTokenizer.from_pretrained(model1_name)

# Model 2: snunlp/KR-SBERT-V40K-klueNLI-augSTS
model2_name = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
snu_model = AutoModel.from_pretrained(model2_name)
snu_tokenizer = AutoTokenizer.from_pretrained(model2_name)

# 각 모델의 유사도 점수 계산
bmk_scores = []
snu_scores = []
ground_truth = []

print("\n유사도 계산 중...")
for sent1, sent2, gt in test_pairs:
    # BM-K 모델
    inputs = bmk_tokenizer([sent1, sent2], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings, _ = bmk_model(**inputs, return_dict=False)
    bmk_score = cal_score_bmk(embeddings[0][0], embeddings[1][0])
    bmk_scores.append(bmk_score)
    
    # SNU 모델
    encoded_input = snu_tokenizer([sent1, sent2], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = snu_model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    snu_score = cosine_similarity(sentence_embeddings[0].numpy(), sentence_embeddings[1].numpy())
    snu_scores.append(snu_score)
    
    ground_truth.append(gt)
    
    print(f"\n문장1: {sent1}")
    print(f"문장2: {sent2}")
    print(f"GT: {gt}, {model1_name}: {bmk_score:.3f}, {model2_name}: {snu_score:.3f}")

# 성능 평가
def calculate_accuracy(scores, threshold=0.5):
    predictions = [1 if score > threshold else 0 for score in scores]
    correct = sum([1 for pred, gt in zip(predictions, ground_truth) if pred == gt])
    return correct / len(ground_truth)

# 최적 임계값 찾기
best_bmk_threshold = max([(calculate_accuracy(bmk_scores, t), t) for t in np.arange(0, 1, 0.01)], key=lambda x: x[0])[1]
best_snu_threshold = max([(calculate_accuracy(snu_scores, t), t) for t in np.arange(0, 1, 0.01)], key=lambda x: x[0])[1]

print(f"\n최적 임계값:")
print(f"  {model1_name}: {best_bmk_threshold:.2f}")
print(f"  {model2_name}: {best_snu_threshold:.2f}")
print(f"\n정확도:")
print(f"  {model1_name}: {calculate_accuracy(bmk_scores, best_bmk_threshold):.2%}")
print(f"  {model2_name}: {calculate_accuracy(snu_scores, best_snu_threshold):.2%}")

# 시각화
plt.figure(figsize=(15, 10))

# 1. 코사인 유사도 분포
plt.subplot(2, 2, 1)
x = range(len(test_pairs))
# ground_truth를 기반으로 유사/비유사 인덱스 분리
similar_idx = [i for i, gt in enumerate(ground_truth) if gt == 1]
dissimilar_idx = [i for i, gt in enumerate(ground_truth) if gt == 0]

# 모델명을 짧게 표시하기 위해 마지막 부분만 사용
model1_short_label = model1_name.split('/')[-1]
model2_short_label = model2_name.split('/')[-1]

# 유사/비유사별로 점수 플롯
plt.scatter([x[i] for i in similar_idx], [bmk_scores[i] for i in similar_idx], color='blue', marker='o', label=f'{model1_short_label} (유사)', s=100)
plt.scatter([x[i] for i in dissimilar_idx], [bmk_scores[i] for i in dissimilar_idx], color='blue', marker='x', label=f'{model1_short_label} (비유사)', s=100)
plt.scatter([x[i] for i in similar_idx], [snu_scores[i] for i in similar_idx], color='red', marker='o', label=f'{model2_short_label} (유사)', s=100)
plt.scatter([x[i] for i in dissimilar_idx], [snu_scores[i] for i in dissimilar_idx], color='red', marker='x', label=f'{model2_short_label} (비유사)', s=100)
plt.axhline(y=best_bmk_threshold, color='blue', linestyle='--', alpha=0.5, label=f'{model1_short_label} 임계값: {best_bmk_threshold:.2f}')
plt.axhline(y=best_snu_threshold, color='red', linestyle='--', alpha=0.5, label=f'{model2_short_label} 임계값: {best_snu_threshold:.2f}')
plt.xlabel('문장 쌍 인덱스')
plt.ylabel('코사인 유사도')
plt.title('모델별 코사인 유사도 분포')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. ROC 커브
plt.subplot(2, 2, 2)
fpr_bmk, tpr_bmk, _ = roc_curve(ground_truth, bmk_scores)
fpr_snu, tpr_snu, _ = roc_curve(ground_truth, snu_scores)
roc_auc_bmk = auc(fpr_bmk, tpr_bmk)
roc_auc_snu = auc(fpr_snu, tpr_snu)

plt.plot(fpr_bmk, tpr_bmk, color='blue', label=f'{model1_short_label} (AUC = {roc_auc_bmk:.3f})')
plt.plot(fpr_snu, tpr_snu, color='red', label=f'{model2_short_label} (AUC = {roc_auc_snu:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC 커브')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. 유사/비유사별 점수 분포 (Box plot)
plt.subplot(2, 2, 3)
# ground_truth를 기반으로 유사/비유사 점수 분리
bmk_similar_scores = [bmk_scores[i] for i in similar_idx]
bmk_dissimilar_scores = [bmk_scores[i] for i in dissimilar_idx]
snu_similar_scores = [snu_scores[i] for i in similar_idx]
snu_dissimilar_scores = [snu_scores[i] for i in dissimilar_idx]

# 모델명을 짧게 표시하기 위해 마지막 부분만 사용
model1_short = model1_name.split('/')[-1]
model2_short = model2_name.split('/')[-1]

data_for_plot = {
    f'{model1_short}\n유사': bmk_similar_scores,
    f'{model1_short}\n비유사': bmk_dissimilar_scores,
    f'{model2_short}\n유사': snu_similar_scores,
    f'{model2_short}\n비유사': snu_dissimilar_scores
}
box_plot = plt.boxplot(data_for_plot.values(), labels=data_for_plot.keys())

# x축 레이블을 45도 회전하여 겹치지 않도록 함
plt.xticks(rotation=45, ha='right')
plt.ylabel('코사인 유사도')
plt.title('모델별 유사/비유사 코사인 유사도 분포')
plt.grid(True, alpha=0.3)

# 4. 성능 지표 테이블
plt.subplot(2, 2, 4)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 예측값 계산
bmk_predictions = [1 if score > best_bmk_threshold else 0 for score in bmk_scores]
snu_predictions = [1 if score > best_snu_threshold else 0 for score in snu_scores]

# 혼동 행렬 계산
cm_bmk = confusion_matrix(ground_truth, bmk_predictions)
cm_snu = confusion_matrix(ground_truth, snu_predictions)

# 클래스별 정확도 계산
bmk_tn, bmk_fp, bmk_fn, bmk_tp = cm_bmk.ravel()
snu_tn, snu_fp, snu_fn, snu_tp = cm_snu.ravel()

# 성능 지표 계산
table_data = []
# 헤더
headers = ['모델', '전체 정확도', '유사 정확도', '비유사 정확도', 'F1 Score']

# BM-K 모델 성능
bmk_overall_acc = (bmk_tp + bmk_tn) / len(ground_truth)
bmk_similar_acc = bmk_tp / (bmk_tp + bmk_fn) if (bmk_tp + bmk_fn) > 0 else 0
bmk_dissimilar_acc = bmk_tn / (bmk_tn + bmk_fp) if (bmk_tn + bmk_fp) > 0 else 0
bmk_f1 = f1_score(ground_truth, bmk_predictions)

# SNU 모델 성능
snu_overall_acc = (snu_tp + snu_tn) / len(ground_truth)
snu_similar_acc = snu_tp / (snu_tp + snu_fn) if (snu_tp + snu_fn) > 0 else 0
snu_dissimilar_acc = snu_tn / (snu_tn + snu_fp) if (snu_tn + snu_fp) > 0 else 0
snu_f1 = f1_score(ground_truth, snu_predictions)

# 테이블 데이터 구성 - 모델명을 더 짧게 표시
model1_table_name = model1_name.split('/')[-1]
model2_table_name = model2_name.split('/')[-1]

# 긴 모델명이 있을 경우를 대비하여 최대 길이 제한
if len(model1_table_name) > 20:
    model1_table_name = model1_table_name[:17] + '...'
if len(model2_table_name) > 20:
    model2_table_name = model2_table_name[:17] + '...'

table_data = [
    [model1_table_name, f'{bmk_overall_acc:.1%}', f'{bmk_similar_acc:.1%}', f'{bmk_dissimilar_acc:.1%}', f'{bmk_f1:.3f}'],
    [model2_table_name, f'{snu_overall_acc:.1%}', f'{snu_similar_acc:.1%}', f'{snu_dissimilar_acc:.1%}', f'{snu_f1:.3f}']
]

# 축 숨기기
ax = plt.gca()
ax.axis('off')

# 테이블 생성 (각 컬럼의 너비를 다르게 설정)
# 첫 번째 컬럼(모델명)을 더 넓게 설정
col_widths = [0.40, 0.15, 0.15, 0.15, 0.15]  # 첫 컬럼을 더 넓게 (합이 1.0)
table = plt.table(cellText=table_data,
                  colLabels=headers,
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1],
                  colWidths=col_widths)

# 테이블 스타일링
table.auto_set_font_size(False)
table.set_fontsize(8)  # 폰트 크기를 줄임
table.scale(1.5, 2.5)  # 너비를 더 넓게, 높이도 약간 높게 조정

# 헤더 스타일
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 성능이 더 좋은 셀 강조
for col in range(1, len(headers)):
    bmk_val = float(table_data[0][col].rstrip('%f'))
    snu_val = float(table_data[1][col].rstrip('%f'))
    
    if bmk_val > snu_val:
        table[(1, col)].set_facecolor('#E3F2FD')
        table[(1, col)].set_text_props(weight='bold')
    elif snu_val > bmk_val:
        table[(2, col)].set_facecolor('#FFEBEE')
        table[(2, col)].set_text_props(weight='bold')

plt.title('모델별 분류 성능 비교', pad=20, fontsize=12, weight='bold')

plt.tight_layout()
# 하단 여백 추가하여 회전된 레이블이 잘 보이도록 함
plt.subplots_adjust(bottom=0.1)
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 성능 요약
print("\n=== 성능 비교 요약 ===")
print(f"{model1_name}:")
print(f"  - AUC: {roc_auc_bmk:.3f}")
print(f"  - 최적 임계값: {best_bmk_threshold:.2f}")
print(f"  - 정확도: {calculate_accuracy(bmk_scores, best_bmk_threshold):.2%}")
print(f"\n{model2_name}:")
print(f"  - AUC: {roc_auc_snu:.3f}")
print(f"  - 최적 임계값: {best_snu_threshold:.2f}")
print(f"  - 정확도: {calculate_accuracy(snu_scores, best_snu_threshold):.2%}")

# 평균 점수 차이
similar_diff_bmk = np.mean(bmk_similar_scores) - np.mean(bmk_dissimilar_scores)
similar_diff_snu = np.mean(snu_similar_scores) - np.mean(snu_dissimilar_scores)
print(f"\n유사-비유사 평균 코사인 유사도 차이:")
print(f"  - {model1_name}: {similar_diff_bmk:.3f}")
print(f"  - {model2_name}: {similar_diff_snu:.3f}")
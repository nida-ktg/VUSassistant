# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:25:19 2026

@author: hatice
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:21:42 2026

@author: BUSE
"""

import pandas as pd
import numpy as np
import warnings
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score
# ====================================================================
# BAŞLANGIÇ VE SİSTEM AYARLARI
# ====================================================================
print("[SİSTEM] Başlatılıyor... Veri belleğe (RAM) alınıyor.")
warnings.filterwarnings("ignore")

# Optuna'nın deneme (trial) loglarını tamamen susturur, sadece hataları basar.
optuna.logging.set_verbosity(optuna.logging.WARNING)
print("[SİSTEM] Optuna Motoru Sessiz Moda Alındı. Lütfen Bekleyin (CPU Yoğun İşlem)...")

# ====================================================================
# MODÜL 1: VERİ İÇE AKTARIMI VE İZOLASYON (DATA LEAKAGE DEFANSI)
# ====================================================================
df = pd.read_csv('nihai_genetik_veri.csv')

X = df.drop(columns=['CLASS'])
y = df['CLASS']

# Stratify=y ile dengeli bölme (Rapor Kriteri)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"[BAŞARI] X_train (Eğitim Seti) Boyutu: {X_train.shape}")
print(f"[BAŞARI] X_test (Test Seti) Boyutu: {X_test.shape}")

# ====================================================================
# MODÜL 2: OPTUNA İLE BAYESYEN OPTİMİZASYON (F2 SKORU ODAKLI)
# ====================================================================
def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        # Hastaları kaçırmamak için azınlık sınıfını (1) daha ağır basıyoruz
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 3.0, 8.0), 
        
        # Ağaç Mimarisi ve Overfitting Frenleri
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 30, 90),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
        'verbose': -1,
        'random_state': 42
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        model = lgb.LGBMClassifier(**param)
        model.fit(
            X_tr, y_tr, 
            eval_set=[(X_va, y_va)], 
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        
        preds_proba = model.predict_proba(X_va)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_va, preds_proba)
        
        # RAPORDA VADEDİLEN F2-SKORU FORMÜLÜ (Hastayı atlama cezasını artırır)
        f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10) 
        cv_scores.append(np.max(f2_scores))

    return np.mean(cv_scores)

# Motoru 30 deneme ile ateşliyoruz (Sessiz çalışacak, bitene kadar bekle)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30) 

print("\n[BAŞARI] Optimizasyon Bitti.")
print(f"[ANALİZ] Ulaşılan En İyi F2 Skoru: {study.best_value:.4f}")

# ====================================================================
# MODÜL 3: NİHAİ MODEL EĞİTİMİ VE KLİNİK EŞİK (THRESHOLD)
# ====================================================================
print("\n[İŞLEM] Nihai Model Eğitiliyor...")
best_params = study.best_params
best_params.update({'objective': 'binary', 'verbose': -1, 'random_state': 42})

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train, y_train)

test_proba = final_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, test_proba)

# F2 ile Eşiği (Threshold) Hesaplıyoruz
f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
best_threshold_idx = np.argmax(f2_scores)
best_threshold = thresholds[best_threshold_idx]

print(f"[ANALİZ] Varsayılan 0.50 Eşiği İPTAL EDİLDİ.")
print(f"[BAŞARI] F2 Optimize Edilmiş Klinik Karar Eşiği (Threshold): {best_threshold:.4f}")

y_pred_dynamic = (test_proba >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_dynamic)

print("\n=======================================================")
print("[KARMAŞIKLIK MATRİSİ] - GERÇEK DÜNYA PERFORMANSI")
print("=======================================================")
print(f"Gerçek Negatif (TN - Sağlıklıya sağlıklı dedi)  : {cm[0][0]}")
print(f"Yanlış Pozitif (FP - Sağlıklıya patojenik dedi) : {cm[0][1]} (Gereksiz Alarm - Kabul Edilebilir)")
print(f"Yanlış Negatif (FN - PATOJENİĞİ ATLADI!)        : {cm[1][0]} (ÖLÜMCÜL HATA - Optimize Edildi)")
print(f"Gerçek Pozitif (TP - Patojeniği doğru buldu)    : {cm[1][1]}")
print("=======================================================")
# ROC-AUC (Modelin genel ayrım gücü)
roc_auc = roc_auc_score(y_test, test_proba)
# PR-AUC (Dengesiz verilerde modelin ne kadar 'keskin' olduğu)
pr_auc = average_precision_score(y_test, test_proba)

print(f"\n[TELEMETRİ] ROC-AUC Skoru: {roc_auc:.4f}")
print(f"[TELEMETRİ] PR-AUC Skoru  : {pr_auc:.4f}")


# ====================================================================
# MODÜL 5: DUAL-CORE ENSEMBLE (LIGHTGBM + XGBOOST VOTING)
# ====================================================================
print("\n[SİSTEM] Modül 5 Başlatılıyor: Dual-Core (LGBM + XGBoost) Motoru Ateşleniyor...")

from sklearn.ensemble import VotingClassifier
import xgboost as xgb

# 1. LightGBM Motoru (Optuna'dan çıkan en iyi ayarlarla zaten elimizde)
# final_model değişkeni Modül 3'te eğitilmişti.

# 2. XGBoost Motoru (LightGBM'in mimarisine benzer defansif ayarlarla)
print("[İŞLEM] XGBoost Yardımcı Motoru Hazırlanıyor...")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=5.0, # Hastaları 5 kat değerli gör
    learning_rate=0.05,
    max_depth=6,
    alpha=1.0, # L1 Regularization
    lambda_param=1.0, # L2 Regularization (isim çakışmasını önlemek için lambda_param kullanılır)
    random_state=42,
    n_jobs=-1 # Tüm CPU çekirdeklerini kullan
)
plt.figure(figsize=(8, 6))

# cmap='Blues' yaparak temayı maviye çektik
# vmin ve vmax değerleri, yüksek sayıların (TN) rengi boğmasını engellemek için hala kritik.
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                 annot_kws={"size": 16, "weight": "bold"},
                 xticklabels=['Tahmin: Benign (0)', 'Tahmin: Patojenik (1)'],
                 yticklabels=['Gerçek: Benign (0)', 'Gerçek: Patojenik (1)'],
                 vmin=0, vmax=5000) 

plt.title('Nihai Klinik Risk Matrisi (Dual-Core Ensemble)', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Gerçek Sınıflar', fontsize=12, fontweight='bold')
plt.xlabel('Modelin Tahminleri', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
# 3. Voting (Oylama) Sınıflandırıcısı
# 'soft' oylama: İki model de olasılık üretir, sistem bu olasılıkların ortalamasını alır.
print("[İŞLEM] Modeller Birleştiriliyor (Soft Voting)...")
ensemble_model = VotingClassifier(
    estimators=[('lgbm', final_model), ('xgb', xgb_model)],
    voting='soft'
)

# Çift motoru tüm eğitim verisiyle eğit
print("[İŞLEM] Çift Motor Eğitiliyor (Fan Sesi Yükselebilir)...")
ensemble_model.fit(X_train, y_train)

# 4. Ensemble Test ve Yeni Dinamik Eşik
test_proba_ens = ensemble_model.predict_proba(X_test)[:, 1]

# F2 Skoru ile Eşiği Çift Motor için tekrar optimize ediyoruz
precisions_ens, recalls_ens, thresholds_ens = precision_recall_curve(y_test, test_proba_ens)
f2_scores_ens = (5 * precisions_ens * recalls_ens) / (4 * precisions_ens + recalls_ens + 1e-10)
best_threshold_ens = thresholds_ens[np.argmax(f2_scores_ens)]

print(f"\n[BAŞARI] Dual-Core Optimize Edilmiş Klinik Eşik: {best_threshold_ens:.4f}")

# 5. Yeni Kararlar ve Karmaşıklık Matrisi
y_pred_ens = (test_proba_ens >= best_threshold_ens).astype(int)
cm_ens = confusion_matrix(y_test, y_pred_ens)

print("\n=======================================================")
print("[KARMAŞIKLIK MATRİSİ] - DUAL-CORE ENSEMBLE (LGBM + XGB)")
print("=======================================================")
print(f"Gerçek Negatif (TN - Sağlıklı)       : {cm_ens[0][0]}")
print(f"Yanlış Pozitif (FP - Gereksiz Alarm) : {cm_ens[0][1]}")
print(f"Yanlış Negatif (FN - Kaçan Hasta)    : {cm_ens[1][0]} (Son Limit)")
print(f"Gerçek Pozitif (TP - Bulunan Hasta)  : {cm_ens[1][1]}")
print("=======================================================")
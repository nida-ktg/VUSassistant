# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:01:16 2026

@author: BUSE
"""

import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler

VERI_YOLU =r"C:\Users\BUSE\OneDrive\Desktop\veri temizleme\clinvar_conflicting.csv"
print("SISTEM BASLATILIYOR... veri okunuyor")
#ilk 1000 satırı çekiyoruz
df=pd.read_csv(VERI_YOLU , nrows=70000)

#verileri sayısal veri haline getiriyoruz 
sayisal_kolonlar=['AF_ESP','AF_EXAC','AF_TGP','CADD_PHRED','CADD_RAW','LoFtool','CLASS']
df_filtreli=df[sayisal_kolonlar]

# verinin anatomisini ekrana yazdırıyoruz 
print("\n---MATRİS BOYUTU ---")
print(df_filtreli.shape)

print("\n---EKSİK VERİLER (NaN)---")
print(df_filtreli.isnull().sum())

##VERI TEMIZLIGI

print("\n VERİ TEMİZLİĞİ BAŞLIYOR...")

# 1. MICE ALGORİTMASI İLE EKSİK VERİLERİ (NaN) DOLDURMA
print("MICE Algoritması çalışıyor... (Eksik veriler makine öğrenmesiyle tahmin ediliyor)")
imputer = IterativeImputer(random_state=42)
# fit_transform veriyi numpy dizisine çevirir, biz tekrar Pandas DataFrame'e alıyoruz
df_doldurulmus = pd.DataFrame(imputer.fit_transform(df_filtreli), columns=df_filtreli.columns)

print("İşlem sonrası toplam eksik veri (NaN) sayısı:", df_doldurulmus.isnull().sum().sum())


# 2. ROBUST SCALER İLE UÇ DEĞERLERİ (OUTLIERS) BASKILAMA VE NORMALİZASYON
print("RobustScaler çalışıyor... (Aykırı değerler eziliyor)")
scaler = RobustScaler()

# CLASS sütunu bizim "0 veya 1" olan hedefimiz (Patojenik mi değil mi?). Onu ölçeklendirmemize gerek yok, ayırıyoruz.
X_ozellikler = df_doldurulmus.drop('CLASS', axis=1)
y_hedef = df_doldurulmus['CLASS']

# Sadece özellikleri (AF_ESP, CADD vs.) ölçeklendiriyoruz
X_olcekli = pd.DataFrame(scaler.fit_transform(X_ozellikler), columns=X_ozellikler.columns)

# Temizlenmiş ve ölçeklenmiş özellikleri, CLASS hedefimizle tekrar birleştiriyoruz
df_temiz = pd.concat([X_olcekli, y_hedef], axis=1)

print("\n--- İŞLEM TAMAM: YAPAY ZEKAYA GİRECEK TEMİZ MATRİS HAZIR ---")
print(df_temiz.head())

df_temiz.to_csv("teknofest_temiz_veri.csv", index=False)
print("Veri başarıyla Hard Disk'e yazıldı. Görev Tamamlandı!")
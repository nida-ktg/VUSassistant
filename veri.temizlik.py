import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib

VERI_YOLU = "clinvar_conflicting.csv"
print("SISTEM BASLATILIYOR... Veri okunuyor")

# low_memory=False: İşlemciye "Büyük dosya, tipleri karıştırma, yavaş oku ama doğru oku" diyoruz.
df = pd.read_csv(VERI_YOLU, nrows=61000, low_memory=False)
# CLASS sütunu bizim cevap anahtarımız. Temizlik yaparken makinenin bunu görmemesi lazım.
y_hedef = df['CLASS']

# axis=1 demek "o sütunu sütun bütünlüğünü bozmadan işleme sok " demektir.
df_ozellikler = df.drop('CLASS', axis=1)
print("Ana tablodaki kolon sayısı (CLASS hariç):", df_ozellikler.shape[1])

#kolonların projemize yararlı olduğunu düşündüğümüz kısımlarını filtreledik
sayisal_kolonlar = ['AF_ESP', 'AF_EXAC', 'AF_TGP', 'CADD_PHRED', 'CADD_RAW', 'LoFtool', 'BLOSUM62', 'DISTANCE', 'STRAND']
metin_kolonlari = ['CHROM', 'Consequence', 'IMPACT', 'BIOTYPE', 'CLNVC', 'ORIGIN', 'SIFT', 'PolyPhen']

#kolonları çekiyoruz
df_sayisal_kisim = df_ozellikler[sayisal_kolonlar]
df_metin_kisim = df_ozellikler[metin_kolonlari]

#sayısallaştırıyoruz
print("Kolonlar 0-1 matrislerine dönüştürülüyor...")
df_sayisallasmis = pd.get_dummies(df_metin_kisim, columns=metin_kolonlari)

#sayısal verilerin matrislerini birleştiriyoruz
df_islem = pd.concat([df_sayisal_kisim, df_sayisallasmis], axis=1)

print("Genişletilmiş Yeni Kolon Sayısı:", df_islem.shape[1])

# MICE algoritmasıyla doldurma işlemini yapıyoruz
# random_state=42: Deneyi her tekrarladığımızda aynı bilimsel sonucu almak için.
imputer = IterativeImputer(random_state=42)

doldurulmus_matris = imputer.fit_transform(df_islem)
df_doldurulmus = pd.DataFrame(doldurulmus_matris, columns=df_islem.columns)

print("İşlem sonrası kalan eksik veri (NaN) sayısı:", df_doldurulmus.isnull().sum().sum())
from sklearn.preprocessing import RobustScaler
# Sıfıra bölünme hatasını (Sistem çökmesini) engellemek için paydaya 0.0001 ekliyoruz.
df_doldurulmus['Risk_Indeksi'] = df_doldurulmus['CADD_PHRED'] / (df_doldurulmus['AF_ESP'] + 0.0001)
print("Toplam Sensör Sayısı (Risk İndeksi Dahil):", df_doldurulmus.shape[1])

# aykırı değerleri robust scaler ile eziyoruz
scaler = RobustScaler()
X_final = pd.DataFrame(scaler.fit_transform(df_doldurulmus), columns=df_doldurulmus.columns)


joblib.dump(scaler, "models/scaler.pkl")
print("[SİSTEM] RobustScaler modeli 'models' klasörüne başarıyla mühürlendi.")
# class ı geri ekliyoruz
df_temiz = pd.concat([X_final, y_hedef.reset_index(drop=True)], axis=1)

df_temiz.to_csv("nihai_genetik_veri.csv", index=False)

print("\n--- İŞLEM TAMAM ---")
print(f"Nihai Matris Boyutu: {df_temiz.shape}")
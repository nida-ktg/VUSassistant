from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import numpy as np
import re
import sqlite3
import hashlib
from datetime import datetime

# Senin kendi yerel motorların
from rule_engine import check_medical_rules
from xai_explainer import generate_shap_plot

app = FastAPI(title="VUSassistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. VERİTABANI KURULUMU (KURUM EKLENDİ) ---
def init_db():
    conn = sqlite3.connect('vus_klinik.db', check_same_thread=False)
    c = conn.cursor()
    # Kurum sütunu eklendi
    c.execute('''CREATE TABLE IF NOT EXISTS kullanicilar 
                 (email TEXT PRIMARY KEY, ad_soyad TEXT, kurum TEXT, sifre_hash TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS hasta_dosyalari 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, doktor_email TEXT, tc TEXT, ad TEXT, varyant TEXT, sonuc TEXT, yuzde REAL, tarih TEXT)''')
    conn.commit()
    return conn

db_conn = init_db()

def hash_sifre(sifre):
    return hashlib.sha256(sifre.encode()).hexdigest()

# --- 2. YAPAY ZEKA YÜKLEMESİ ---
try:
    model = joblib.load("models/ensemble_model.pkl") 
    with open("feature_list.json", "r") as f:
        FEATURE_NAMES = json.load(f)
    print("✅ Modeller ve Veritabanı Hazır!")
except Exception as e:
    print(f"❌ HATA: {e}")

# --- 3. VERİ ŞABLONLARI ---
class KayitVerisi(BaseModel):
    ad_soyad: str
    kurum: str
    email: str
    sifre: str

class GirisVerisi(BaseModel):
    email: str
    sifre: str

class AnalizVerisi(BaseModel):
    genetik_veri: str
    tc: str
    ad: str
    varyant: str
    doktor_email: str

# --- 4. ENDPOINTLER ---
@app.post("/kayit")
async def kayit_ol(veri: KayitVerisi):
    c = db_conn.cursor()
    c.execute("SELECT * FROM kullanicilar WHERE email=?", (veri.email,))
    if c.fetchone():
        raise HTTPException(status_code=400, detail="Bu e-posta zaten kayıtlı!")
    
    c.execute("INSERT INTO kullanicilar (email, ad_soyad, kurum, sifre_hash) VALUES (?, ?, ?, ?)", 
              (veri.email, veri.ad_soyad, veri.kurum, hash_sifre(veri.sifre)))
    db_conn.commit()
    return {"mesaj": "Kayıt Başarılı"}

@app.post("/giris")
async def giris_yap(veri: GirisVerisi):
    c = db_conn.cursor()
    c.execute("SELECT ad_soyad, kurum FROM kullanicilar WHERE email=? AND sifre_hash=?", 
              (veri.email, hash_sifre(veri.sifre)))
    res = c.fetchone()
    if not res:
        raise HTTPException(status_code=401, detail="Hatalı e-posta veya şifre!")
    return {"ad_soyad": res[0], "kurum": res[1]}

@app.get("/hastalar")
async def hastalari_getir(doktor_email: str):
    c = db_conn.cursor()
    c.execute("SELECT tc, ad, varyant, sonuc, yuzde, tarih FROM hasta_dosyalari WHERE doktor_email=? ORDER BY id DESC", (doktor_email,))
    kayitlar = c.fetchall()
    return {"hastalar": [{"tc": k[0], "ad": k[1], "varyant": k[2], "sonuc": k[3], "yuzde": k[4], "tarih": k[5]} for k in kayitlar]}

@app.post("/analiz")
async def analiz_et(veri: AnalizVerisi):
    try:
        # Analiz mantığı (Scaler iptal edilmiş haliyle)
        temiz_metin = re.sub(r'[;\t\n]+', ',', veri.genetik_veri.strip())
        ham_parcalar = [p.strip() for p in temiz_metin.split(',') if p.strip()]
        beklenen_uzunluk = len(FEATURE_NAMES)
        girilen_uzunluk = len(ham_parcalar)

        if girilen_uzunluk != beklenen_uzunluk:
            raise HTTPException(
                status_code=400, 
                detail=f"Hata: Tam olarak {beklenen_uzunluk} adet parametre girmelisiniz. (Siz {girilen_uzunluk} adet girdiniz). Lütfen verinizi kontrol edin."
            )
        temizlenmis_sayilar = [float(p) for p in ham_parcalar]
    

        df_hasta = pd.DataFrame([temizlenmis_sayilar], columns=FEATURE_NAMES)
        olasiliklar = model.predict_proba(df_hasta)[0]
        patojenik_ihtimali = float(olasiliklar[1])
        risk_yuzdesi = round(patojenik_ihtimali * 100, 1)

        ai_karari = "PATOJENİK" if patojenik_ihtimali >= 0.50 else "BENIGN"

        cadd_val = float(df_hasta['CADD_PHRED'].iloc[0]) if 'CADD_PHRED' in df_hasta.columns else 0.0
        af_val = float(df_hasta['AF_ESP'].iloc[0]) if 'AF_ESP' in df_hasta.columns else 0.0
        kural_sonucu = check_medical_rules(cadd_score=cadd_val, af_esp=af_val)

        if kural_sonucu.override_ml:
            ai_karari = "PATOJENİK"
            risk_yuzdesi = 99.9

        su_an = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c = db_conn.cursor()
        c.execute("INSERT INTO hasta_dosyalari (doktor_email, tc, ad, varyant, sonuc, yuzde, tarih) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (veri.doktor_email, veri.tc, veri.ad, veri.varyant, ai_karari, risk_yuzdesi, su_an))
        db_conn.commit()

        return {
            "basarili": True, "karar": ai_karari, "yuzde": risk_yuzdesi,
            "kural": kural_sonucu.explanation, "shap_resmi": generate_shap_plot(df_hasta)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
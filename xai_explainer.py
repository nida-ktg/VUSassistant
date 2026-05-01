"""
Created on Fri May 1 21:27:17 2026

@author: Nida
"""

# XAI (Açıklanabilir Yapay Zeka)
# Burada hastanın genetik verisi üzerinden modelin tahminini nasıl yaptığına dair bir açıklama (SHAP) grafiği üretiyoruz.

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import base64
import json
from io import BytesIO


#Eğitilmiş modelinin yüklenmesi  (ensemble XGBoost modeli)
model = joblib.load("models/xgb_model.pkl")
#model = final_model (LightGBM)
#model = joblib.load("models/lgbm_model.pkl")


#SHAP açıklayıcı (tree-based modeller için)
explainer = shap.TreeExplainer(model)



#Feature isimlerinin yüklenmesi (model ile tam uyum için)
with open("feature_list.json") as f:
    FEATURE_NAMES = json.load(f)


def generate_shap_plot(hasta_verisi):
    """
    Hastanın genetik verisini alır ve SHAP açıklama grafiği üretir.
    
    Input:
        hasta_verisi -> dict veya pandas DataFrame
    
    Output:
        base64 formatında SHAP grafiği (web için uygun)
    """

    # Eğer veri dict olarak geldiyse DataFrame'e çevir
    if isinstance(hasta_verisi, dict):
        hasta_verisi = pd.DataFrame([hasta_verisi])

    # Eksik feature varsa sıfır ile tamamla
    for col in FEATURE_NAMES:
        if col not in hasta_verisi.columns:
            hasta_verisi[col] = 0

    # Feature sırasını modele uygun hale getir
    hasta_verisi = hasta_verisi[FEATURE_NAMES]

    # SHAP değerlerini hesapla
    shap_values = explainer.shap_values(hasta_verisi)

    # Grafik oluştur
    plt.clf()
    #plt.figure()

    '''
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=hasta_verisi.columns
    )
    '''

    
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=hasta_verisi.iloc[0],
            feature_names=hasta_verisi.columns
        ),
        show=False
    )

    plt.gcf()


    # Grafiği memory buffer'a al
    buffer = BytesIO()
    #plt.savefig(buffer, format="png", bbox_inches="tight")
    fig = plt.gcf()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    # Base64'e çevir (web'e göndermek için)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    plt.close()

    return img_base64


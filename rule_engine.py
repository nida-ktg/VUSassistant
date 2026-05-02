# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:12:17 2026

@author: zeyne
"""

def check_medical_rules(cadd_phred, af_esp, model_prediction):

    uyarilar = []

   
    if model_prediction == 0 and cadd_phred >= 20.0:
        uyarilar.append("DIKKAT: Yapay zeka zararsiz dedi ancak CADD_PHRED skoru kritik seviyede (>=20). MANUEL INCELEME GEREK!")

    if model_prediction == 0 and af_esp < 0.01 and cadd_phred >= 15.0:
        uyarilar.append("DIKKAT: Varyant toplumda cok nadir (AF_ESP < 0.01) ve CADD skoru supheli. Gozden kacmis olabilir!")

    if len(uyarilar) > 0:
        return " | ".join(uyarilar)
    else:
        return "Kural Kontrolu: Ihlal Yok (Klinik Olarak Guvenli)"


if __name__ == "__main__":
    print("[TEST] Medikal Kural Motoru Baslatiliyor...")
    

    test_cadd = 24.5
    test_af = 0.005
    test_tahmin = 0 
    
    sonuc = check_medical_rules(cadd_phred=test_cadd, af_esp=test_af, model_prediction=test_tahmin)
    print("\n[CIKTI]:", sonuc)
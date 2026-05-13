from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


CADD_PATHOGENIC_THRESHOLD    = 20.0   
CADD_CRITICAL_THRESHOLD      = 30.0   
AF_ESP_COMMON_THRESHOLD      = 0.01   
AF_ESP_VERY_RARE_THRESHOLD   = 0.0001 
SIFT_DAMAGING_THRESHOLD      = 0.05   
POLYPHEN_PROBABLY_DAMAGING   = 0.908 
POLYPHEN_POSSIBLY_DAMAGING   = 0.446 



@dataclass
class RuleResult:

    override_ml: bool            = False
    risk_level: str              = "LOW"
    warnings: list[str]          = field(default_factory=list)
    explanation: str             = ""

    def to_dict(self) -> dict:
        return {
            "override_ml": self.override_ml,
            "risk_level":  self.risk_level,
            "warnings":    self.warnings,
            "explanation": self.explanation,
        }


def check_medical_rules(
    cadd_score: float,
    af_esp:     float,
    sift_score: Optional[float] = None,
    polyphen_score: Optional[float] = None,
) -> RuleResult:




    result   = RuleResult()
    warnings = []
    score    = 0  
  
    if cadd_score >= CADD_CRITICAL_THRESHOLD:
        msg = (
            f"DİKKAT: CADD SKORU ÇOK YÜKSEK ({cadd_score:.1f} ≥ {CADD_CRITICAL_THRESHOLD}), "
            f"MANUEL İNCELEME GEREKİYOR — Yapay zeka kararı geçersiz sayılabilir."
        )
        warnings.append(msg)
        logger.warning(msg)
        result.override_ml = True 
        score += 50

    elif cadd_score >= CADD_PATHOGENIC_THRESHOLD:
        msg = (
            f"UYARI: CADD skoru klinik eşiği aştı ({cadd_score:.1f} ≥ {CADD_PATHOGENIC_THRESHOLD}). "
            f"Patojenik potansiyel yüksek."
        )
        warnings.append(msg)
        score += 25


    if af_esp < AF_ESP_VERY_RARE_THRESHOLD:
        msg = (
            f"DİKKAT: Varyant ultra-nadir (AF={af_esp:.6f} < {AF_ESP_VERY_RARE_THRESHOLD}). "
            f"Populasyonda neredeyse görülmüyor — patojenik risk artıyor."
        )
        warnings.append(msg)
        score += 20

    if af_esp > AF_ESP_COMMON_THRESHOLD:
        msg = (
            f"BİLGİ: Varyant yaygın polimorfizm aralığında (AF={af_esp:.4f} > {AF_ESP_COMMON_THRESHOLD}). "
            f"Benign olma ihtimali yüksek — ML 'Patojenik' dediyse dikkatli ol."
        )
        warnings.append(msg)
        score -= 15   


    if sift_score is not None:
        if sift_score <= SIFT_DAMAGING_THRESHOLD:
            msg = (
                f"UYARI: SIFT skoru fonksiyon kaybına işaret ediyor "
                f"({sift_score:.3f} ≤ {SIFT_DAMAGING_THRESHOLD})."
            )
            warnings.append(msg)
            score += 15


    if polyphen_score is not None:
        if polyphen_score >= POLYPHEN_PROBABLY_DAMAGING:
            msg = (
                f"UYARI: PolyPhen-2 'Probably Damaging' ({polyphen_score:.3f}). "
                f"Protein yapısı büyük ihtimalle zarar görüyor."
            )
            warnings.append(msg)
            score += 15
        elif polyphen_score >= POLYPHEN_POSSIBLY_DAMAGING:
            msg = (
                f"BİLGİ: PolyPhen-2 'Possibly Damaging' ({polyphen_score:.3f}). "
                f"Orta düzey endişe."
            )
            warnings.append(msg)
            score += 8



    combo_triggered = (
        cadd_score >= CADD_PATHOGENIC_THRESHOLD
        and af_esp < AF_ESP_VERY_RARE_THRESHOLD
        and sift_score is not None
        and sift_score <= SIFT_DAMAGING_THRESHOLD
    )
    if combo_triggered:
        msg = (
            "ACİL: CADD + Nadir AF + SIFT üçlü riski aynı anda tetikledi! "
            "Klinik genetikçiye yönlendirme şiddetle önerilir."
        )
        warnings.append(msg)
        result.override_ml = True
        score += 30

    score = max(0, min(100, score))   

    if score >= 70:
        result.risk_level = "CRITICAL"
    elif score >= 40:
        result.risk_level = "HIGH"
    elif score >= 20:
        result.risk_level = "MODERATE"
    else:
        result.risk_level = "LOW"


    if result.override_ml:
        result.explanation = (
            f"Kural motoru ML kararını eziyor. Risk seviyesi: {result.risk_level}. "
            f"Tetiklenen kural sayısı: {len(warnings)}. Lütfen kliniğe danışın."
        )
    elif warnings:
        result.explanation = (
            f"Kural motoru {len(warnings)} uyarı üretti. "
            f"Risk seviyesi: {result.risk_level}. ML kararı korunuyor, ancak bulgular izlenmeli."
        )
    else:
        result.explanation = (
            "Tüm medikal sınır kontrolleri geçildi. Risk seviyesi: LOW. "
            "ML kararı güvenle kullanılabilir."
        )

    result.warnings = warnings
    return result





if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    
    r = check_medical_rules(cadd_score=12.0, af_esp=0.03)
    print("=== Senaryo A: Normal Varyant ===")
    print(f"  override_ml : {r.override_ml}")
    print(f"  risk_level  : {r.risk_level}")
    print(f"  warnings    : {r.warnings}\n")

    
    r = check_medical_rules(cadd_score=34.7, af_esp=0.00003,
                            sift_score=0.02, polyphen_score=0.95)
    print("=== Senaryo B: Yüksek Riskli Varyant ===")
    print(f"  override_ml : {r.override_ml}")
    print(f"  risk_level  : {r.risk_level}")
    for w in r.warnings:
        print(f"  ⚠ {w}")
    print(f"\n  explanation : {r.explanation}\n")

    r = check_medical_rules(cadd_score=9.5, af_esp=0.05)
    print("=== Senaryo C: Yaygın Polimorfizm (ML Patojenik dediyse şüphelenilmeli) ===")
    print(f"  override_ml : {r.override_ml}")
    print(f"  risk_level  : {r.risk_level}")
    for w in r.warnings:
        print(f"  ℹ {w}")
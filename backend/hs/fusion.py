"""Transparent rule-based late fusion of the ECG-photo result and the clinical-form risk.
This is a design HEURISTIC (no dataset pairs ECG photos with the form values, so the
weights cannot be learned). It is not clinically validated."""

LEVELS = ["low", "medium", "high"]
EMERGENCY_NOTE = ("If you have chest pain, trouble breathing, fainting or other severe symptoms, "
                  "seek emergency care now, whatever this result says.")
DISCLAIMER = ("Research prototype for academic demonstration. Not a medical device; "
              "not for diagnosis or treatment decisions.")


def level_from_prob(p, medium=0.30, high=0.60):
    if p is None:
        return None
    return "high" if p >= high else "medium" if p >= medium else "low"


def fuse(photo_result=None, clinical_prob=None, medium=0.30, high=0.60):
    """photo_result: dict from PhotoECGPredictor (status, p_abnormal) or None.
    clinical_prob: probability (0-1) from the clinical-form model, or None.
    Rule: overall level = the HIGHER of the two available levels (conservative).
    If the photo was rejected or abstained, only the clinical level is used and the user is told."""
    ecg_level, notes = None, []
    if photo_result is not None:
        if photo_result.get("status") == "ok":
            ecg_level = level_from_prob(photo_result.get("p_abnormal"), medium, high)
        elif photo_result.get("status") == "abstain":
            notes.append("The ECG photo result was too uncertain to use. Please retake the photo or see a doctor.")
        else:
            reasons = "; ".join(photo_result.get("reasons", [])) or "unusable photo"
            notes.append(f"The ECG photo could not be used ({reasons}). Please retake it.")
    clin_level = level_from_prob(clinical_prob, medium, high)
    avail = [l for l in (ecg_level, clin_level) if l is not None]
    overall = max(avail, key=LEVELS.index) if avail else "unknown"
    message = {
        "low": "No major flag found by this screening aid. This does not rule out heart disease.",
        "medium": "Some risk indicators were found. It is better to see a doctor.",
        "high": "Higher-risk indicators were found. Please see a doctor soon.",
        "unknown": "Not enough usable input to give a result. Retake the photo and/or fill the form.",
    }[overall]
    return dict(overall=overall, ecg_level=ecg_level, clinical_level=clin_level, message=message,
                notes=notes, rule="overall = higher of ECG-photo and clinical level (heuristic)",
                emergency_note=EMERGENCY_NOTE, disclaimer=DISCLAIMER)

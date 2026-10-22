"""Rule-based fallback NLU.

Used when no LLM API key is set, or when an LLM call fails mid-call,
so a call never dies because the LLM is unavailable. It is deliberately
simple: keyword intent detection + regex slot extraction.
"""
import re
import time

from . import i18n
from .taxonomy import CATEGORIES, SLOT_LABELS, INTENT_CONFIDENCE_MIN

YES = ["haan", "han", "haa", "ha ji", "ji haan", "yes", "ji", "theek", "thik", "ok", "okay", "kar do", "kardo",
       "yeah", "sure", "right", "please do", "go ahead",
       "sahi", "correct", "bilkul", "हाँ", "हां", "जी", "ठीक", "सही", "कर दो", "बिल्कुल"]
NO = ["nahi", "nahin", "no", "galat", "mat", "नहीं", "गलत", "मत", "wrong", "don't", "not correct"]
BYE = ["bas", "bus itna", "thank", "dhanyavad", "dhanyawad", "shukriya", "bye", "kuch nahi", "बस", "धन्यवाद", "शुक्रिया"]
DONT_KNOW = ["nahi pata", "pata nahi", "yaad nahi", "malum nahi", "nahi maloom", "don't know", "dont know",
             "पता नहीं", "याद नहीं", "मालूम नहीं"]

MONTHS = {
    "jan": "January", "january": "January", "janvari": "January", "जनवरी": "January",
    "feb": "February", "february": "February", "farvari": "February", "फरवरी": "February",
    "march": "March", "मार्च": "March", "april": "April", "अप्रैल": "April", "may": "May", "मई": "May",
    "june": "June", "जून": "June", "july": "July", "जुलाई": "July", "august": "August", "अगस्त": "August",
    "september": "September", "sept": "September", "सितंबर": "September", "october": "October",
    "oct": "October", "अक्टूबर": "October", "november": "November", "nov": "November", "नवंबर": "November",
    "december": "December", "dec": "December", "दिसंबर": "December",
    "pichle mahine": "Last month", "last month": "Last month", "पिछले महीने": "Last month",
    "is mahine": "This month", "this month": "This month", "इस महीने": "This month",
}
BANKS = ["sbi", "state bank", "pnb", "punjab national", "bank of baroda", "bob", "canara", "hdfc", "icici",
         "axis", "union bank", "central bank", "bank of india", "indian bank", "gramin", "kotak", "airtel payments",
         "india post", "ippb", "स्टेट बैंक", "ग्रामीण"]
PAY_MODES = {"upi": "UPI", "neft": "NEFT", "imps": "IMPS", "rtgs": "RTGS", "cash": "Cash deposit",
             "cdm": "Cash deposit", "नकद": "Cash deposit", "कैश": "Cash deposit"}
PRODUCTS = {"dmt": "DMT", "money transfer": "DMT", "aeps": "AePS", "aadhaar": "AePS", "aadhar": "AePS",
            "bill": "BBPS", "bbps": "BBPS", "recharge": "Recharge", "pan": "PAN", "insurance": "Insurance"}
BILLERS = ["bijli", "electricity", "jio", "airtel", "vodafone", "vi ", "bsnl", "tata play", "dish tv",
           "fastag", "gas", "water", "pani", "बिजली"]
DEVICES = {"scanner": "Biometric device", "device": "Biometric device", "morpho": "Morpho device",
           "mantra": "Mantra device", "startek": "Startek device", "rd service": "RD service",
           "app": "Eko app", "login": "App login", "password": "App login", "otp": "OTP",
           "ऐप": "Eko app", "स्कैनर": "Biometric device"}

FREE_TEXT_SLOTS = {"issue_detail", "error_message", "bank_name", "biller", "payment_mode", "device_or_screen",
                   "month", "product", "txn_date"}

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _has(text: str, words) -> bool:
    return any(w in text for w in words)


def detect_intent(text: str):
    t = f" {text.lower()} "
    scores = {}
    for cat, cfg in CATEGORIES.items():
        s = 0.0
        for kw in cfg.get("keywords", []):
            if kw in t:
                s += 1 + 0.15 * len(kw.split())
        if s:
            scores[cat] = s
    if not scores:
        return "unknown", 0.25
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    margin = (top - second) / top
    conf = min(0.95, 0.55 + 0.12 * top) * (0.75 + 0.25 * margin)
    return best, round(conf, 2)


def extract_slots(text: str, asked_slot: str | None = None) -> dict:
    raw = text.translate(DEVANAGARI_DIGITS)
    t = raw.lower()
    out = {}

    m = re.search(r"(?:₹|rs\.?|inr|rupees?|rupaye|rupay)\s*([\d,]{2,9})", t) or \
        re.search(r"([\d,]{2,9})\s*(?:₹|rs\b|rupees?|rupaye|rupay|रुपये|रुपए|रूपए|ka transaction|ke)", t)
    if m:
        out["amount"] = m.group(1).replace(",", "")

    ids = [w for w in re.findall(r"\b[A-Za-z0-9]{8,22}\b", raw) if re.search(r"\d", w)]
    ids = [w for w in ids if w != out.get("amount") and not re.fullmatch(r"[6-9]\d{9}", w)]
    if ids:
        tid = ids[0].upper()
        if tid.startswith("CSP"):
            out["csp_code"] = tid
        elif asked_slot == "utr_or_ref" or _has(t, ["utr", "reference", "ref"]):
            out["utr_or_ref"] = tid
        else:
            out["txn_id"] = tid
    if asked_slot in ("txn_id", "utr_or_ref") and asked_slot not in out and _has(t, DONT_KNOW):
        out[asked_slot] = "not known"

    if _has(t, ["aaj", "today", "आज"]):
        out["txn_date"] = "Today"
    elif _has(t, ["parso", "परसों"]):
        out["txn_date"] = "Day before yesterday"
    elif _has(t, ["kal", "yesterday", "कल"]):
        out["txn_date"] = "Yesterday"
    else:
        d = re.search(r"\b(\d{1,2})\s*(?:[/-]\s*(\d{1,2})|\s*(?:tarikh|taarikh|तारीख))", t)
        if d:
            out["txn_date"] = d.group(0)

    for k, v in MONTHS.items():
        if k in t:
            out["month"] = v
            break
    for b in BANKS:
        if b in t:
            out["bank_name"] = b.upper() if len(b) <= 4 else b.title()
            break
    for k, v in PAY_MODES.items():
        if re.search(rf"\b{re.escape(k)}\b", t):
            out["payment_mode"] = v
            break
    for k, v in PRODUCTS.items():
        if k in t:
            out["product"] = v
            break
    for b in BILLERS:
        if b in t:
            out["biller"] = b.strip().title()
            break
    for k, v in DEVICES.items():
        if k in t:
            out["device_or_screen"] = v
            break
    em = re.search(r"(?:error|एरर)[^a-z0-9]*(.{3,60})", t)
    if em:
        out["error_message"] = em.group(1).strip(" .,'\"")

    # If we asked for a specific free-text slot and nothing matched, take the answer as-is.
    if asked_slot in FREE_TEXT_SLOTS and asked_slot not in out and len(t.split()) >= 1 \
            and not _has(t, NO + YES):
        out[asked_slot] = text.strip()[:200]
    return out


class RulesEngine:
    name = "rules"

    def decide(self, state: dict, user_text: str) -> dict:
        t0 = time.perf_counter()
        t = user_text.lower()
        intent, conf = state.get("intent"), state.get("intent_confidence") or 0
        slots = dict(state.get("slots") or {})
        asked = slots.get("_asked")
        lang = i18n.norm(state.get("lang"))
        decision = {"engine": self.name, "sentiment": "neutral", "understood": True,
                    "language": "en" if lang == "en" else "hi" if re.search(r"[ऀ-ॿ]", user_text) else "hinglish"}

        # Confirmation step
        if asked == "_confirm":
            if _has(t, YES) and not _has(t, NO):
                decision.update(action="create_ticket", reply="", intent=intent, intent_confidence=conf)
            else:
                decision.update(action="escalate", escalation_reason="CSP did not confirm the issue summary",
                                reply=i18n.t("confirm_declined", lang), intent=intent, intent_confidence=conf)
            return self._finish(decision, slots, t0)

        if _has(t, BYE) and len(t.split()) <= 4 and not intent:
            decision.update(action="end_call", reply=i18n.t("bye", lang),
                            intent="general_query", intent_confidence=0.9)
            return self._finish(decision, slots, t0)

        new_intent, new_conf = detect_intent(user_text)
        if not intent or intent in ("unknown", "general_query") or (new_conf > conf + 0.2 and new_intent != "unknown"):
            if new_intent != "unknown" or not intent:
                intent, conf = new_intent, new_conf

        extracted = extract_slots(user_text, asked if asked and not asked.startswith("_") else None)
        slots.update({k: v for k, v in extracted.items() if v})
        decision.update(intent=intent, intent_confidence=conf)

        if intent == "unknown" or conf < INTENT_CONFIDENCE_MIN:
            decision.update(understood=False, action="continue",
                            reply=i18n.t("unclear_intent", lang))
            return self._finish(decision, slots, t0)

        cfg = CATEGORIES[intent]
        if cfg.get("self_serve"):
            decision.update(action="resolved_by_agent", reply="")  # orchestrator fills ticket status
            return self._finish(decision, slots, t0)
        if cfg.get("always_escalate"):
            slots.setdefault("issue_detail", user_text[:200])
            decision.update(action="escalate", escalation_reason="Suspected fraud - mandatory RM escalation",
                            reply=i18n.t("fraud", lang))
            return self._finish(decision, slots, t0)

        missing = [s for s in cfg["slots"] if not slots.get(s)]
        if missing:
            slots["_asked"] = missing[0]
            prefix = i18n.t("understood_prefix", lang, spoken=i18n.spoken(intent, cfg["spoken"], lang)) \
                if not state.get("slots") else ""
            decision.update(action="continue", reply=prefix + i18n.slot_question(missing[0], lang))
            return self._finish(decision, slots, t0)

        slots["_asked"] = "_confirm"
        details = ", ".join(f"{SLOT_LABELS.get(k, k)}: {slots[k]}" for k in cfg["slots"])
        decision.update(action="continue",
                        reply=i18n.t("confirm", lang, spoken=i18n.spoken(intent, cfg["spoken"], lang), details=details))
        return self._finish(decision, slots, t0)

    @staticmethod
    def _finish(decision, slots, t0):
        decision["slots"] = slots
        decision["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        decision.setdefault("input_tokens", 0)
        decision.setdefault("output_tokens", 0)
        return decision

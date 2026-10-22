"""Everything the agent says outside the LLM, in Hindi and English, plus language detection.

Hindi is written in Devanagari so the Hindi TTS voice pronounces it naturally (Roman-script
Hinglish gets read with English phonetics). Banking / app terms stay in Latin script
(DMT, transaction ID, UTR, OTP) because that is how CSPs actually say them.

A call's language is "ask" (caller hasn't chosen yet), "hi" or "en".
"""
import re

LANGS = ("hi", "en")
DEFAULT = "hi"

PHRASES = {
    # Bilingual opener when the caller picks the language
    "choose_lang": {
        "hi": "नमस्ते! Eko सहायक में आपका स्वागत है। हिंदी में बात करने के लिए 'हिंदी' बोलिए। "
              "For English, please say 'English'.",
    },
    "greet_unknown": {
        "hi": "नमस्ते, मैं Eko सहायक हूँ, Eko की virtual assistant। कृपया अपना CSP ID बताइए, जैसे C S P 1 0 2 3 1।",
        "en": "Hello, I am Eko Sahayak, Eko's virtual assistant. Please tell me your CSP ID, for example C S P 1 0 2 3 1.",
    },
    "greet_inbound": {
        "hi": "नमस्ते {name} जी, मैं Eko सहायक हूँ, {rm} जी की virtual assistant। बताइए, आज मैं आपकी क्या मदद कर सकती हूँ?",
        "en": "Hello {name} ji, I am Eko Sahayak, {rm}'s virtual assistant. How can I help you today?",
    },
    "greet_outbound": {
        "hi": "नमस्ते {name} जी, मैं Eko सहायक, {rm} जी की तरफ़ से बोल रही हूँ। क्या आपको Eko की किसी service में कोई दिक्कत आ रही है?",
        "en": "Hello {name} ji, this is Eko Sahayak calling on behalf of {rm}. Are you facing any problem with an Eko service?",
    },
    "your_rm": {"hi": "आपके RM", "en": "your RM"},
    "rm_team": {"hi": "RM", "en": "the RM"},
    "lang_switched": {
        "hi": "ठीक है, अब हम हिंदी में बात करेंगे। बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
        "en": "Sure, let's continue in English. Please go ahead.",
    },
    "identified": {
        "hi": "धन्यवाद {name} जी, आपकी पहचान हो गई। बताइए क्या समस्या है?",
        "en": "Thank you {name} ji, I have found your account. Please tell me the problem.",
    },
    "human_escalate": {
        "hi": "बिल्कुल, मैं आपकी बात आपके RM तक पहुँचा रही हूँ। वे आपको जल्दी call करेंगे।",
        "en": "Of course, I am passing this to your RM. They will call you back shortly.",
    },
    "angry_escalate": {
        "hi": "माफ़ कीजिए, आपको इतनी परेशानी हुई। मैं यह call अभी आपके RM को दे रही हूँ, वे आपसे जल्दी बात करेंगे।",
        "en": "I am sorry for the trouble you have had. I am handing this to your RM right now, they will speak with you soon.",
    },
    "max_turns": {
        "hi": "मैं यह call आपके RM को forward कर रही हूँ, वे आपसे जल्दी बात करेंगे।",
        "en": "I am forwarding this call to your RM, they will speak with you soon.",
    },
    "not_understood_final": {
        "hi": "माफ़ कीजिए, मुझे आपकी बात ठीक से समझ नहीं आ रही। मैं आपके RM को बता रही हूँ, वे आपको जल्द call back करेंगे।",
        "en": "Sorry, I am not able to understand you clearly. I am informing your RM, they will call you back soon.",
    },
    "not_understood_retry": {
        "hi": "माफ़ कीजिए, आवाज़ साफ़ नहीं आई। क्या आप एक बार फिर से बता सकते हैं?",
        "en": "Sorry, I couldn't hear that clearly. Could you please say it again?",
    },
    "ticket_raised": {
        "hi": "आपका ticket raise हो गया है, number {no}। {rm} जी की team {sla} घंटे के अंदर इसे देखेगी। धन्यवाद!",
        "en": "Your ticket has been raised, number {no}. {rm}'s team will look into it within {sla} hours. Thank you!",
    },
    "reference": {"hi": "Reference number {no}।", "en": "Reference number {no}."},
    "bye": {"hi": "धन्यवाद! आपका दिन शुभ हो।", "en": "Thank you! Have a good day."},
    "no_tickets": {
        "hi": "आपके नाम पर अभी कोई complaint दर्ज नहीं है। अगर कोई नई समस्या है तो बताइए।",
        "en": "There is no complaint registered in your name right now. Please tell me if you have a new problem.",
    },
    "ticket_status": {
        "hi": "आपका {cat} वाला ticket, number {no}, {st}। धन्यवाद!",
        "en": "Your {cat} ticket, number {no}, {st}. Thank you!",
    },
    "st_open": {"hi": "अभी खुला है और team इसे देख रही है", "en": "is open and the team is looking into it"},
    "st_in_progress": {"hi": "पर काम चल रहा है", "en": "is being worked on"},
    "st_resolved": {"hi": "resolve हो चुका है", "en": "has been resolved"},
    "st_closed": {"hi": "बंद हो चुका है", "en": "has been closed"},
    # Rules engine
    "confirm_declined": {
        "hi": "ठीक है, मैं आपके RM को यह call forward कर रही हूँ, वे आपसे जल्दी संपर्क करेंगे।",
        "en": "Alright, I am forwarding this call to your RM, they will contact you soon.",
    },
    "unclear_intent": {
        "hi": "माफ़ कीजिए, मैं समझ नहीं पाई। क्या आपकी समस्या money transfer, AePS, wallet, commission, bill payment या app से जुड़ी है?",
        "en": "Sorry, I didn't understand. Is your problem about money transfer, AePS, wallet, commission, bill payment or the app?",
    },
    "fraud": {
        "hi": "यह गंभीर मामला है। मैं इसे तुरंत आपके RM को urgent priority पर भेज रही हूँ। कृपया किसी के साथ OTP या PIN share न करें।",
        "en": "This is a serious matter. I am sending it to your RM right away on urgent priority. Please do not share your OTP or PIN with anyone.",
    },
    "understood_prefix": {"hi": "समझ गई, {spoken} की समस्या है। ", "en": "I understand, this is a {spoken} problem. "},
    "confirm": {
        "hi": "मैं confirm कर लेती हूँ - {spoken} की समस्या। {details}। क्या मैं ticket raise कर दूँ?",
        "en": "Let me confirm - a {spoken} problem. {details}. Shall I raise a ticket?",
    },
}

SLOT_QUESTIONS = {
    "hi": {
        "txn_id": "कृपया transaction ID बताइए।",
        "amount": "कितने रुपये का transaction था?",
        "txn_date": "यह transaction किस दिन हुआ था?",
        "bank_name": "Customer का bank कौन सा है?",
        "utr_or_ref": "Payment का UTR या reference number बताइए।",
        "payment_mode": "पैसा किस तरीके से भेजा था - UPI, NEFT या cash deposit?",
        "month": "किस महीने का commission नहीं आया?",
        "product": "किस service का commission - DMT, AePS या bill payment?",
        "issue_detail": "थोड़ा detail में बताइए, समस्या क्या है?",
        "device_or_screen": "समस्या app में है या biometric device में?",
        "error_message": "Screen पर कोई error message आ रहा है? क्या लिखा है?",
        "biller": "किस biller या operator का payment था?",
    },
    "en": {
        "txn_id": "Please tell me the transaction ID.",
        "amount": "What was the transaction amount in rupees?",
        "txn_date": "On which day did this transaction happen?",
        "bank_name": "Which bank does the customer have?",
        "utr_or_ref": "Please tell me the UTR or reference number of the payment.",
        "payment_mode": "How was the money sent - UPI, NEFT or cash deposit?",
        "month": "For which month was the commission not received?",
        "product": "Commission for which service - DMT, AePS or bill payment?",
        "issue_detail": "Please describe the problem in a little more detail.",
        "device_or_screen": "Is the problem in the app or the biometric device?",
        "error_message": "Is any error message showing on the screen? What does it say?",
        "biller": "Which biller or operator was the payment for?",
    },
}

# How each category is named in a spoken sentence
SPOKEN = {
    "dmt_failed": {"hi": "money transfer", "en": "money transfer"},
    "aeps_issue": {"hi": "AePS", "en": "AePS"},
    "wallet_topup": {"hi": "wallet load", "en": "wallet load"},
    "commission": {"hi": "commission", "en": "commission"},
    "kyc_onboarding": {"hi": "KYC", "en": "KYC"},
    "app_device": {"hi": "app या device", "en": "app or device"},
    "bbps_recharge": {"hi": "bill payment", "en": "bill payment"},
    "refund_status": {"hi": "refund", "en": "refund"},
    "fraud_report": {"hi": "fraud", "en": "fraud"},
    "ticket_status": {"hi": "complaint status", "en": "complaint status"},
    "general_query": {"hi": "general", "en": "general"},
}


def norm(lang: str | None) -> str:
    return lang if lang in LANGS else DEFAULT


def t(key: str, lang: str | None, **kw) -> str:
    variants = PHRASES[key]
    text = variants.get(norm(lang)) or variants[DEFAULT]
    return text.format(**kw) if kw else text


def slot_question(slot: str, lang: str | None) -> str:
    return SLOT_QUESTIONS[norm(lang)][slot]


def spoken(cat: str, fallback: str, lang: str | None) -> str:
    return SPOKEN.get(cat, {}).get(norm(lang), fallback)


# ---------------------------------------------------------------- language detection
_EN_WORDS = {"english", "inglish", "angrezi", "angreji", "इंग्लिश", "अंग्रेज़ी", "अंग्रेजी", "انگریزی", "انگلش"}
_HI_WORDS = {"hindi", "हिंदी", "हिन्दी", "ہندی"}
_SWITCH_EN = ["english please", "in english", "speak english", "speak in english", "english mein", "english me ",
              "english main", "angrezi mein", "angreji mein", "इंग्लिश में", "अंग्रेज़ी में", "अंग्रेजी में"]
_SWITCH_HI = ["hindi please", "in hindi", "speak hindi", "speak in hindi", "hindi mein", "hindi me ", "hindi main",
              "हिंदी में", "हिन्दी में"]
# Common words that only show up when someone is speaking English
_EN_HINTS = {"the", "is", "my", "i", "am", "was", "has", "have", "not", "please", "money", "failed", "problem",
             "with", "and", "account", "customer", "it", "did", "want", "to", "from", "what", "how", "hello", "yes"}
_HI_HINTS = {"hai", "hain", "mera", "meri", "nahi", "nahin", "kya", "ho", "gaya", "gayi", "ka", "ki", "ke", "se",
             "mein", "paisa", "paise", "aap", "ji", "haan", "kar", "tha", "thi", "kaise"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+|[ऀ-ॿ]+|[؀-ۿ]+", text.lower())


def choice(text: str) -> str | None:
    """The caller's answer to 'Hindi or English?'."""
    w = set(_words(text))
    if w & _EN_WORDS:
        return "en"
    if w & _HI_WORDS:
        return "hi"
    return None


def switch_request(text: str) -> str | None:
    """Mid-call 'English please' / 'Hindi mein baat kijiye'."""
    t_ = f" {text.lower()} "
    if any(p in t_ for p in _SWITCH_EN):
        return "en"
    if any(p in t_ for p in _SWITCH_HI):
        return "hi"
    return None


def guess(text: str) -> str:
    """Best guess at the language of an utterance (used if the caller skips the language question)."""
    if re.search(r"[ऀ-ॿ؀-ۿ]", text):
        return "hi"
    w = _words(text)
    en, hi = sum(x in _EN_HINTS for x in w), sum(x in _HI_HINTS for x in w)
    return "en" if en > hi else "hi"

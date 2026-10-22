"""Issue taxonomy for CSP calls: categories, required details, priority and SLA.

Edit this file to add/rename categories. Both the Claude engine and the
rule-based fallback read from here, so they always stay in sync.
"""

CATEGORIES = {
    "dmt_failed": {
        "spoken": "money transfer",
        "label": "Money transfer (DMT) failed / pending",
        "slots": ["txn_id", "amount", "txn_date"],
        "priority": "high",
        "sla_hours": 24,
        "keywords": ["dmt", "money transfer", "transfer", "paisa nahi pahuncha", "paise nahi pahunche",
                     "beneficiary", "imps", "neft", "pending", "fail", "failed", "atak", "atka",
                     "ट्रांसफर", "पैसा नहीं पहुंचा", "पैसे नहीं पहुंचे", "पेंडिंग", "फेल"],
    },
    "aeps_issue": {
        "spoken": "AePS",
        "label": "AePS withdrawal / biometric issue",
        "slots": ["txn_id", "amount", "txn_date", "bank_name"],
        "priority": "high",
        "sla_hours": 24,
        "keywords": ["aeps", "aadhaar", "aadhar", "fingerprint", "finger", "thumb", "anguutha",
                     "angootha", "withdrawal", "nikasi", "cash nikala", "आधार", "अंगूठा", "फिंगरप्रिंट", "निकासी"],
    },
    "wallet_topup": {
        "spoken": "wallet load",
        "label": "Wallet load / top-up not credited",
        "slots": ["amount", "utr_or_ref", "txn_date", "payment_mode"],
        "priority": "high",
        "sla_hours": 12,
        "keywords": ["wallet", "balance", "top up", "topup", "load", "e-value", "evalue", "utr",
                     "वॉलेट", "बैलेंस", "लोड"],
    },
    "commission": {
        "spoken": "commission",
        "label": "Commission not received / mismatch",
        "slots": ["month", "product"],
        "priority": "medium",
        "sla_hours": 72,
        "keywords": ["commission", "kamission", "incentive", "payout", "earning", "kamai",
                     "कमीशन", "इंसेंटिव", "कमाई"],
    },
    "kyc_onboarding": {
        "spoken": "KYC",
        "label": "KYC / onboarding / account activation",
        "slots": ["issue_detail"],
        "priority": "medium",
        "sla_hours": 48,
        "keywords": ["kyc", "onboarding", "activate", "activation", "document", "pan card", "verification",
                     "account band", "id band", "केवाईसी", "एक्टिवेट", "डॉक्यूमेंट", "आईडी बंद"],
    },
    "app_device": {
        "spoken": "app ya device",
        "label": "App / login / device problem",
        "slots": ["device_or_screen", "error_message"],
        "priority": "medium",
        "sla_hours": 24,
        "keywords": ["app", "login", "password", "otp", "device", "scanner", "morpho", "mantra",
                     "rd service", "error", "crash", "khul nahi", "nahi khul", "ऐप", "लॉगिन", "पासवर्ड",
                     "स्कैनर", "एरर", "खुल नहीं"],
    },
    "bbps_recharge": {
        "spoken": "bill payment",
        "label": "Bill payment / recharge failed",
        "slots": ["txn_id", "amount", "biller"],
        "priority": "medium",
        "sla_hours": 24,
        "keywords": ["bill", "bijli", "electricity", "recharge", "dth", "bbps", "fastag", "gas",
                     "बिल", "बिजली", "रिचार्ज"],
    },
    "refund_status": {
        "spoken": "refund",
        "label": "Refund status enquiry",
        "slots": ["txn_id"],
        "priority": "medium",
        "sla_hours": 24,
        "keywords": ["refund", "wapas", "vapas", "return", "रिफंड", "वापस"],
    },
    "fraud_report": {
        "spoken": "fraud",
        "label": "Suspected fraud / unauthorised activity",
        "slots": ["issue_detail"],
        "priority": "critical",
        "sla_hours": 2,
        "always_escalate": True,
        "keywords": ["fraud", "dhokha", "scam", "unauthorised", "unauthorized", "hack", "chori",
                     "फ्रॉड", "धोखा", "चोरी"],
    },
    "ticket_status": {
        "spoken": "complaint status",
        "label": "Status of an existing complaint",
        "slots": [],
        "priority": "low",
        "sla_hours": 0,
        "self_serve": True,   # agent answers from the ticket DB, no new ticket needed
        "keywords": ["ticket", "complaint ka", "shikayat", "kya hua", "status", "शिकायत", "टिकट", "क्या हुआ"],
    },
    "general_query": {
        "spoken": "general",
        "label": "General query / other",
        "slots": ["issue_detail"],
        "priority": "low",
        "sla_hours": 72,
        "keywords": [],
    },
}

SLOT_LABELS = {
    "txn_id": "Transaction ID", "amount": "Amount (₹)", "txn_date": "Date", "bank_name": "Bank",
    "utr_or_ref": "UTR / Ref", "payment_mode": "Payment mode", "month": "Month", "product": "Product",
    "issue_detail": "Details", "device_or_screen": "Device / screen", "error_message": "Error message",
    "biller": "Biller",
}

# Phrases meaning "connect me to a human / my RM"
HUMAN_REQUEST = ["rm se baat", "rm sir", "rm madam", "insaan", "human", "manager se", "agent se baat",
                 "kisi aadmi", "real person", "आरएम", "मैनेजर से", "इंसान",
                 "talk to my rm", "speak to my rm", "talk to rm", "speak to rm", "connect me to", "talk to a person"]

ANGER = ["bakwas", "pagal", "gussa", "bekaar", "worst", "complaint karunga", "बकवास", "गुस्सा", "बेकार",
         "useless", "nonsense", "rubbish", "very angry", "fed up"]

# Thresholds
STT_CONFIDENCE_MIN = 0.45      # below this, a turn counts as "not understood"
INTENT_CONFIDENCE_MIN = 0.55   # below this, the agent asks a clarifying question
MAX_MISUNDERSTOOD = 2          # consecutive not-understood turns before escalation
MAX_TURNS = 12                 # safety cap on conversation length


def label(cat: str) -> str:
    return CATEGORIES.get(cat, {}).get("label", cat or "Unknown")

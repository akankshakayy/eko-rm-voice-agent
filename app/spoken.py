"""What a caller said -> what the agent can read reliably: numbers as digits, banks and payment modes by name.

Speech-to-text writes amounts the way people say them ("बारह हज़ार पाँच सौ", "साढ़े 12,000", "ढाई हज़ार",
"paanch hazaar") and spells English names in Devanagari ("केनरा बैंक", "आईएमपीएस"). The synthetic phone calls
(evals/synth_audio.py) showed the agent then gets amounts wrong (साढ़े 12,000 read as 12000) and misses banks.
normalize() rewrites those into digits and standard names before the agent sees the text; the call log keeps what
was really said. Ordinary words are left alone: a lone "एक" stays "एक", and "एक दो दिन" is not turned into a number.

canonical_bank() / canonical_mode() map near-misses that speech-to-text produces ("FBI" for SBI, "NAFT" for NEFT,
"EMS" for IMPS) onto the known names, for the details the agent extracts.
"""
import re
from difflib import get_close_matches

# ---------------------------------------------------------------- number words
_HI = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस इक्कीस "
       "बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस "
       "अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास "
       "इक्यावन बावन तिरेपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरेसठ चौंसठ पैंसठ छियासठ सड़सठ "
       "अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी "
       "बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे "
       "छियानवे सत्तानवे अट्ठानवे निन्यानवे").split()
WORDS = {w: i for i, w in enumerate(_HI)}
WORDS.update({  # spellings speech-to-text also writes, and Roman Hinglish
    "ज़ीरो": 0, "जीरो": 0, "zero": 0, "पांच": 5, "छः": 6, "छे": 6, "छै": 6, "नो": 9, "दश": 10, "ग्यारा": 11, "बारा": 12,
    "तेरा": 13, "पन्द्रह": 15, "सत्तरह": 17, "पच्चिस": 25, "पचीस": 25, "बयालिस": 42, "चालिस": 40,
    "ek": 1, "do": 2, "teen": 3, "tin": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "paach": 5, "chhe": 6,
    "che": 6, "saat": 7, "sat": 7, "aath": 8, "aat": 8, "nau": 9, "das": 10, "gyarah": 11, "gyara": 11, "barah": 12,
    "bara": 12, "terah": 13, "chaudah": 14, "pandrah": 15, "solah": 16, "satrah": 17, "atharah": 18, "unnis": 19,
    "bees": 20, "bis": 20, "pachchis": 25, "pachis": 25, "tees": 30, "tis": 30, "chalis": 40, "chaalis": 40,
    "pachas": 50, "pachaas": 50, "saath": 60, "sattar": 70, "assi": 80, "nabbe": 90})
# "सो" is how Whisper often writes सौ; it only counts next to a number, so "सो गया" (slept) is left alone
UNITS = {"सौ": 100, "सो": 100, "sau": 100, "हज़ार": 1000, "हजार": 1000, "hazaar": 1000, "hazar": 1000, "hajar": 1000,
         "लाख": 100000, "lakh": 100000, "lac": 100000, "करोड़": 10 ** 7, "करोड": 10 ** 7, "crore": 10 ** 7}
WHOLE = {"डेढ़": 1.5, "डेढ": 1.5, "डेड़": 1.5, "derh": 1.5, "dedh": 1.5,          # डेढ़ हज़ार = 1500
         "ढाई": 2.5, "धाई": 2.5, "ढाइ": 2.5, "dhai": 2.5, "dhaai": 2.5}             # ढाई हज़ार = 2500
SHIFT = {"साढ़े": 0.5, "साढे": 0.5, "साड़े": 0.5, "साडे": 0.5, "सारे": 0.5, "sadhe": 0.5, "saadhe": 0.5, "saade": 0.5,
         "सवा": 0.25, "sawa": 0.25, "सव्वा": 0.25,                                  # सवा दो हज़ार = 2250
         "पौने": -0.25, "पोने": -0.25, "paune": -0.25}                               # पौने दो हज़ार = 1750
CURRENCY = re.compile(r"^(?:रुपये|रुपए|रूपए|रुपय|रुपई|रुपे|रूपये|rupees?|rupaye|rupay|rs\.?|₹)$", re.I)
_TOKEN = re.compile(r"\d[\d,]*\d|\d|[^\s\d,।?!.]+|[,।?!.]")


def _num(tok: str):
    t = tok.lower()
    if re.fullmatch(r"\d[\d,]*", t):
        return int(t.replace(",", ""))
    return WORDS.get(t)


def _value(tokens: list[str]) -> int | None:
    """One spoken number ("बारह हज़ार पाँच सौ", "साढ़े 12,000", "ढाई हज़ार") -> its value; None if not one number."""
    total, current, shift, bare = 0.0, 0.0, 0.0, 0
    for t in tokens:
        low = t.lower()
        if low in SHIFT:
            shift = SHIFT[low]
        elif low in WHOLE:
            current = WHOLE[low]
        elif low in UNITS:
            unit = UNITS[low]
            current = (current or 1) + shift
            shift = 0.0
            if unit == 100:
                current *= 100                      # "पाँच सौ" stays in the running part ("बारह हज़ार पाँच सौ")
            else:
                total += current * unit
                current = 0.0
            bare = 0
        else:
            n = _num(t)
            if n is None:
                return None
            if bare and n >= 10 and current and current % 100 != 0:
                return None                          # "बीस पाँच" is two numbers, not one
            current += n
            bare += 1
    if shift:                                        # "साढ़े 12,000" (Whisper already multiplied): half a step more
        step = 1000 if current >= 1000 and current % 1000 == 0 else 100 if current % 100 == 0 else 0
        if not step:
            return None
        current += shift * step
    value = total + current
    return int(value) if value == int(value) else None


def _is_numberish(t: str) -> bool:
    low = t.lower()
    return _num(t) is not None or low in UNITS or low in WHOLE or low in SHIFT


def _single_digit(t: str) -> bool:
    n = _num(t)
    return n is not None and 0 <= n <= 9 and not re.fullmatch(r"\d{2,}", t)


def numbers_to_digits(text: str) -> str:
    """Rewrite only the stretches that are numbers; every other character of the text stays exactly as it was."""
    text = text or ""
    spans = [(m.start(), m.end(), m.group()) for m in _TOKEN.finditer(text)]
    out, pos, i = [], 0, 0
    while i < len(spans):
        if not _is_numberish(spans[i][2]):
            i += 1
            continue
        j = i
        while j < len(spans) and _is_numberish(spans[j][2]):
            j += 1
        run = [t for _, _, t in spans[i:j]]
        nxt = spans[j][2] if j < len(spans) else ""
        has_unit = any(t.lower() in UNITS or t.lower() in WHOLE or t.lower() in SHIFT for t in run)
        has_number = any(_num(t) is not None or t.lower() in WHOLE for t in run)
        new = None
        if len(run) >= 4 and all(_single_digit(t) for t in run):
            new = "".join(str(_num(t)) for t in run)                # an ID read digit by digit
        elif has_number and (has_unit or (CURRENCY.match(nxt) and not all(re.fullmatch(r"[\d,]+", t) for t in run))):
            v = _value(run)
            new = str(v) if v is not None else None
        if new is not None:                                          # else: "एक मनी ट्रांसफर", "सो गया": leave it
            out.append(text[pos:spans[i][0]] + new)
            pos = spans[j - 1][1]
        i = j
    return "".join(out) + text[pos:]


# ---------------------------------------------------------------- banks and payment modes
BANKS = {
    "SBI": ["sbi", "एसबीआई", "एस बी आई", "स्टेट बैंक", "state bank", "state bank of india"],
    "PNB": ["pnb", "पीएनबी", "पंजाब नेशनल बैंक", "punjab national bank"],
    "Bank of Baroda": ["bank of baroda", "bob", "बैंक ऑफ़ बड़ौदा", "बैंक ऑफ बड़ौदा", "बड़ौदा बैंक"],
    "Canara Bank": ["canara", "canara bank", "केनरा बैंक", "कैनरा बैंक", "केनरा"],
    "HDFC": ["hdfc", "एचडीएफसी", "hdfc bank"],
    "ICICI": ["icici", "आईसीआईसीआई", "icici bank"],
    "Axis Bank": ["axis", "axis bank", "एक्सिस बैंक", "एक्सिस"],
    "Union Bank": ["union bank", "यूनियन बैंक", "union bank of india"],
    "Bank of India": ["bank of india", "boi", "बैंक ऑफ़ इंडिया", "बैंक ऑफ इंडिया"],
    "Central Bank of India": ["central bank", "सेंट्रल बैंक"],
    "Indian Bank": ["indian bank", "इंडियन बैंक"],
    "Kotak": ["kotak", "कोटक", "kotak mahindra"],
    "IDBI": ["idbi", "आईडीबीआई"],
    "UCO Bank": ["uco", "uco bank", "यूको बैंक", "यूको"],
    "Airtel Payments Bank": ["airtel payments bank", "एयरटेल पेमेंट्स बैंक", "airtel bank", "एयरटेल बैंक"],
    "India Post Payments Bank": ["ippb", "india post payments bank", "पोस्ट ऑफिस", "इंडिया पोस्ट"],
    "Yes Bank": ["yes bank", "यस बैंक"],
    "IndusInd Bank": ["indusind", "इंडसइंड"],
    "Federal Bank": ["federal bank", "फेडरल बैंक"],
    "Bank of Maharashtra": ["bank of maharashtra", "बैंक ऑफ़ महाराष्ट्र"],
    "Indian Overseas Bank": ["iob", "indian overseas bank"],
    "Fino Payments Bank": ["fino", "फिनो"],
    "Paytm Payments Bank": ["paytm", "पेटीएम"],
}
MODES = {
    "NEFT": ["neft", "एनईएफटी", "नेफ्ट", "naft", "nift"],
    "IMPS": ["imps", "आईएमपीएस", "आई एम पी एस", "ems", "imtsa", "iempa", "imbs", "ipms"],
    "RTGS": ["rtgs", "आरटीजीएस"],
    "UPI": ["upi", "यूपीआई", "यू पी आई"],
    "Cash": ["cash", "कैश", "नकद"],
    "Cheque": ["cheque", "check", "चेक"],
}


def _key(s: str) -> str:
    return re.sub(r"[\W_]+", "", (s or "").lower())


def _canonical(value: str, table: dict, cutoff: float) -> str:
    """The standard name for a value speech-to-text bent out of shape; the value itself if nothing is close."""
    k = _key(value)
    if not k:
        return value
    aliases = {_key(a): name for name, names in table.items() for a in [name, *names]}
    if k in aliases:
        return aliases[k]
    hit = get_close_matches(k, list(aliases), n=1, cutoff=cutoff)
    return aliases[hit[0]] if hit else value.strip()


def canonical_bank(value: str) -> str:
    return _canonical(value, BANKS, 0.6)


def canonical_mode(value: str) -> str:
    return _canonical(value, MODES, 0.5)


_NAMES = sorted(((a, name) for table in (BANKS, MODES) for name, names in table.items() for a in names
                 if re.search(r"[ऀ-ॿ]", a)), key=lambda x: -len(x[0]))


def names_to_english(text: str) -> str:
    """Bank and payment-mode names written in Devanagari ("केनरा बैंक", "आईएमपीएस") -> their usual English names."""
    for alias, name in _NAMES:
        text = text.replace(alias, name)
    return text


def normalize(text: str) -> str:
    """Everything the agent reads goes through here first."""
    return numbers_to_digits(names_to_english(text or ""))

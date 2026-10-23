"""Synthetic phone-call audio for the scorecard: CSPs reading out amounts and IDs, the way real calls sound.

Misheard numbers and IDs are the agent's biggest open problem, and clean text-to-speech is far easier for Whisper
than a real call. So each case here is a CSP's opening complaint with the hard parts in it, spoken and then made to
sound like a phone call:
- numbers as people say them: Hindi words ("बारह हज़ार पाँच सौ"), the everyday forms ("डेढ़ हज़ार", "ढाई हज़ार",
  "साढ़े तीन हज़ार"), English words, and IDs read digit by digit, in groups ("चार सौ बयासी"), or with the letters
  spelt out ("टी एक्स एन")
- four voices (Hindi and Indian-English, male and female) at different speeds and pitches
- six line conditions: clean, phone line (300-3400 Hz, 8 kHz, G.711), GSM mobile codec, phone with shop noise at
  15 dB and at 5 dB, and phone with dropouts (short gaps)
Every case keeps its right answers (text, issue type, amount, IDs), so the scorecard measures what the agent caught.
Nothing in it comes from real CSPs or calls. The same seed always gives the same set.

    python -m evals.synth_audio                      # 48 cases -> evals/synthetic_cases.csv + evals/audio/synth/
    python -m evals.synth_audio --n 96 --seed 3
    python -m evals.run --cases evals/synthetic_cases.csv --audio-only
    python -m evals.synth_audio --report evals/results/<run>/results.csv   # scores by condition and reading style
"""
import argparse
import asyncio
import csv
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np

EVALS = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(EVALS, "audio", "synth")          # evals/audio/ is git-ignored
CASES_CSV = os.path.join(EVALS, "synthetic_cases.csv")
RATE = 16000
COLUMNS = ["id", "lang", "audio", "reference_text", "expected_intent", "expected_action", "amount", "txn_id",
           "utr_or_ref", "bank_name", "payment_mode", "month", "product", "notes"]
VOICES = {"hi": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"], "en": ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural"]}
CONDITIONS = ["clean", "phone", "gsm", "phone_noise15", "phone_noise5", "phone_dropouts"]

# ---------------------------------------------------------------- numbers as people say them
HI_0_99 = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस इक्कीस "
           "बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस "
           "अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास "
           "इक्यावन बावन तिरेपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरेसठ चौंसठ पैंसठ छियासठ सड़सठ "
           "अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी "
           "बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे "
           "छियानवे सत्तानवे अट्ठानवे निन्यानवे").split()
EN_0_19 = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
           "seventeen eighteen nineteen").split()
EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
HI_LETTERS = {"T": "टी", "X": "एक्स", "N": "एन", "R": "आर", "U": "यू", "E": "ई", "K": "के", "O": "ओ"}
assert len(HI_0_99) == 100


def hindi_words(n: int) -> str:
    """Hindi number words, Indian style: करोड़, लाख, हज़ार, सौ."""
    if n < 100:
        return HI_0_99[n]
    out = []
    for size, word in ((10 ** 7, "करोड़"), (10 ** 5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if n >= size:
            out.append(f"{hindi_words(n // size)} {word}")
            n %= size
    if n:
        out.append(HI_0_99[n])
    return " ".join(out)


def hindi_everyday(n: int) -> str:
    """The forms people really use for amounts like 1,500 / 2,500 / 3,500 / 12,500; else plain words."""
    k, rest = divmod(n, 1000)
    if rest == 500 and k >= 1:
        return {1: "डेढ़ हज़ार", 2: "ढाई हज़ार"}.get(k, f"साढ़े {hindi_words(k)} हज़ार")
    if n % 100 == 50 and 100 < n < 1000:                          # 350 -> साढ़े तीन सौ
        return f"साढ़े {hindi_words(n // 100)} सौ"
    return hindi_words(n)


def english_words(n: int) -> str:
    if n < 20:
        return EN_0_19[n]
    if n < 100:
        return EN_TENS[n // 10] + ("" if n % 10 == 0 else " " + EN_0_19[n % 10])
    out = []
    for size, word in ((10 ** 7, "crore"), (10 ** 5, "lakh"), (1000, "thousand"), (100, "hundred")):
        if n >= size:
            out.append(f"{english_words(n // size)} {word}")
            n %= size
    if n:
        out.append(english_words(n))
    return " ".join(out)


def say_amount(n: int, lang: str, style: str) -> str:
    if lang == "en":
        return english_words(n) + " rupees"
    return (hindi_everyday(n) if style == "everyday" else hindi_words(n)) + " रुपये"


def _hi_digit(d: str) -> str:
    return "ज़ीरो" if d == "0" else HI_0_99[int(d)]         # in an ID, Hindi speakers say "ज़ीरो", not "शून्य"


def say_id(value: str, lang: str, style: str) -> str:
    """An ID read out: digit by digit, in groups of three ("चार सौ बयासी"), letters spelt out."""
    letters = "".join(c for c in value if c.isalpha())
    digits = "".join(c for c in value if c.isdigit())
    if lang == "en":
        lead = " ".join(letters)
        if style == "groups":
            body = ", ".join(" ".join(EN_0_19[int(d)] for d in digits[i:i + 3]) for i in range(0, len(digits), 3))
        else:
            body = " ".join(EN_0_19[int(d)] for d in digits)
    else:
        lead = " ".join(HI_LETTERS.get(c, c) for c in letters)
        if style == "groups":
            groups = [digits[i:i + 3] for i in range(0, len(digits), 3)]
            body = ", ".join(hindi_words(int(g)) if g[0] != "0" and len(g) == 3 else " ".join(_hi_digit(d) for d in g)
                             for g in groups)
        else:
            body = " ".join(_hi_digit(d) for d in digits)
    return f"{lead} {body}".strip()


# ---------------------------------------------------------------- what callers say
BANKS = ["SBI", "PNB", "Bank of Baroda", "Canara Bank", "HDFC", "ICICI", "Union Bank"]
BANKS_HI = {"SBI": "एसबीआई", "PNB": "पीएनबी", "Bank of Baroda": "बैंक ऑफ़ बड़ौदा", "Canara Bank": "केनरा बैंक",
            "HDFC": "एचडीएफसी", "ICICI": "आईसीआईसीआई", "Union Bank": "यूनियन बैंक"}
MODES = [("NEFT", "एनईएफटी"), ("IMPS", "आईएमपीएस"), ("UPI", "यूपीआई")]

# (intent, details it carries, Hindi template, English template). {a} amount, {t} txn ID, {u} UTR, {b} bank, {m} mode
TEMPLATES = [
    ("dmt_failed", ("amount", "txn_id"),
     "सर, मेरा DMT फेल हो गया, {a} कट गए, ट्रांज़ैक्शन आईडी {t} है।",
     "Sir, my money transfer failed, {a} got deducted, the transaction ID is {t}."),
    ("dmt_failed", ("amount", "txn_id"),
     "कल एक मनी ट्रांसफर किया था {a} का, अभी तक नहीं पहुँचा, आईडी है {t}।",
     "I did a transfer of {a} yesterday and it has not reached, the ID is {t}."),
    ("aeps_issue", ("amount", "bank_name"),
     "AePS से {a} निकाले थे, {b} का खाता था, पैसा कट गया पर ग्राहक को नहीं मिला।",
     "I did an AePS withdrawal of {a} from a {b} account, the money got deducted but the customer did not get it."),
    ("wallet_topup", ("amount", "utr_or_ref", "payment_mode"),
     "मैंने {m} से वॉलेट में {a} डाले थे, यूटीआर नंबर {u} है, अभी तक बैलेंस नहीं आया।",
     "I loaded {a} into my wallet by {m}, the UTR number is {u}, but the balance has not come."),
    ("bbps_recharge", ("amount", "txn_id"),
     "बिजली का बिल भरा था {a} का, ट्रांज़ैक्शन आईडी {t}, वो पेंडिंग दिखा रहा है।",
     "I paid an electricity bill of {a}, transaction ID {t}, and it is showing pending."),
    ("refund_status", ("txn_id",),
     "मेरा रिफ़ंड कब आएगा? ट्रांज़ैक्शन आईडी {t} वाला।",
     "When will my refund come? It is for transaction ID {t}."),
]
OPENERS_HI = ["", "हेलो, ", "हाँ जी नमस्ते, ", "जी सर, "]
OPENERS_EN = ["", "Hello, ", "Good morning, "]


def make_case(i: int, rng: random.Random) -> dict:
    intent, slots, hi_t, en_t = TEMPLATES[i % len(TEMPLATES)]
    lang = "en" if i % 4 == 3 else "hi"                           # one in four in English
    cond = CONDITIONS[(i // 2) % len(CONDITIONS)]
    amount = rng.choice([500, 1500, 2500, 3500, 5000, 7200, 12500, 18000, 25000, 49500, 125000, rng.randrange(200, 99999)])
    txn = rng.choice(["TXN", ""]) + "".join(str(rng.randrange(10)) for _ in range(rng.choice([9, 10])))
    utr = "".join(str(rng.randrange(10)) for _ in range(12))
    bank = rng.choice(BANKS)
    mode_en, mode_hi = rng.choice(MODES)
    amount_style = rng.choice(["everyday", "words"]) if lang == "hi" else "words"
    id_style = rng.choice(["digits", "groups"])
    say = dict(a=say_amount(amount, lang, amount_style), t=say_id(txn, lang, id_style), u=say_id(utr, lang, id_style),
               b=bank if lang == "en" else BANKS_HI[bank], m=mode_en if lang == "en" else mode_hi)
    write = dict(a=f"{amount} रुपये" if lang == "hi" else f"{amount} rupees", t=txn, u=utr, b=bank,
                 m=mode_en)
    template = en_t if lang == "en" else hi_t
    opener = rng.choice(OPENERS_EN if lang == "en" else OPENERS_HI)
    voice = rng.choice(VOICES[lang])
    case = dict.fromkeys(COLUMNS, "")
    case.update(id=f"synth_{i:03d}_{intent}", lang=lang, audio=f"synth/synth_{i:03d}.wav",
                reference_text=opener + template.format(**write), expected_intent=intent)
    vals = {"amount": str(amount), "txn_id": txn, "utr_or_ref": utr, "bank_name": bank, "payment_mode": mode_en}
    for s in slots:
        case[s] = vals[s]
    case["notes"] = (f"condition={cond}; voice={voice}; amount_style={amount_style}; id_style={id_style}; "
                     f"spoken={opener + template.format(**say)}")
    case["_spoken"], case["_voice"], case["_cond"] = opener + template.format(**say), voice, cond
    case["_rate"], case["_pitch"] = f"{rng.randint(-10, 15):+d}%", f"{rng.randint(-8, 8):+d}Hz"
    return case


# ---------------------------------------------------------------- sound like a phone call
async def _tts(text: str, voice: str, rate: str, pitch: str, path: str):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(path)


def _load(path: str) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(RATE),
                          "-f", "f32le", "pipe:1"], capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    spec /= np.sqrt(np.arange(1, len(spec) + 1))
    x = np.fft.irfft(spec, n)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    p_s, p_n = np.mean(speech ** 2) + 1e-12, np.mean(noise ** 2) + 1e-12
    return speech + noise * np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))


def degrade(x: np.ndarray, cond: str, babble: list[np.ndarray], rng: np.random.Generator, out_path: str):
    """Make clean speech sound like it came down a phone line, then save it as 16 kHz WAV for the scorecard."""
    x = np.concatenate([np.zeros(int(0.3 * RATE), np.float32), x, np.zeros(int(0.4 * RATE), np.float32)])
    if cond.startswith("phone_noise"):
        shop = np.zeros_like(x)
        for b in rng.choice(len(babble), size=2, replace=False):     # two other voices talking in the shop
            clip = np.tile(babble[b], int(np.ceil(len(x) / len(babble[b]))))[: len(x)]
            shop += np.roll(clip, rng.integers(len(x)))
        shop += 0.3 * _pink(len(x), rng) * np.abs(shop).max()      # plus street hiss
        x = _mix_at_snr(x, shop, 15 if cond == "phone_noise15" else 5)
    if cond == "phone_dropouts":
        for _ in range(rng.integers(3, 7)):                         # short gaps, as on a weak signal
            start, length = rng.integers(len(x)), int(rng.uniform(0.06, 0.15) * RATE)
            x[start:start + length] = 0
    x = (x / (np.abs(x).max() + 1e-9) * 0.9).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "in.f32")
        x.tofile(raw)
        src = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(RATE), "-ac", "1", "-i", raw]
        if cond == "clean":
            subprocess.run(src + ["-c:a", "pcm_s16le", out_path], check=True)
            return
        band = ["-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000"]
        line = os.path.join(tmp, "line.gsm" if cond == "gsm" else "line.wav")
        codec = ["-c:a", "libgsm", "-f", "gsm"] if cond == "gsm" else ["-c:a", "pcm_mulaw"]
        subprocess.run(src + band + codec + [line], check=True)
        back = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + (["-f", "gsm", "-ar", "8000"] if cond == "gsm" else [])
        subprocess.run(back + ["-i", line, "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", out_path], check=True)


def generate(n: int, seed: int):
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    cases = [make_case(i, rng) for i in range(n)]
    with tempfile.TemporaryDirectory() as tmp:
        babble = []                                                  # background voices for the shop noise
        for j, (text, voice) in enumerate([("अरे भैया एक रिचार्ज करना है, कितने का पैक है?", "hi-IN-MadhurNeural"),
                                           ("हाँ हाँ, कल आना, आज सिस्टम धीरे चल रहा है।", "hi-IN-SwaraNeural"),
                                           ("बीस रुपये छुट्टे दो ना, पाँच सौ का नोट है।", "hi-IN-MadhurNeural"),
                                           ("Madam, ek photocopy aur ek print chahiye.", "en-IN-NeerjaNeural")]):
            p = os.path.join(tmp, f"babble{j}.mp3")
            asyncio.run(_tts(text, voice, "+0%", "+0Hz", p))
            babble.append(_load(p))
        for c in cases:
            mp3 = os.path.join(tmp, "speech.mp3")
            asyncio.run(_tts(c["_spoken"], c["_voice"], c["_rate"], c["_pitch"], mp3))
            degrade(_load(mp3), c["_cond"], babble, nrng, os.path.join(EVALS, "audio", c["audio"]))
            print(f"  {c['id']:28} {c['_cond']:15} {c['_voice']:20} {c['_spoken'][:60]}", flush=True)
    with open(CASES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(cases)
    print(f"\n{n} cases -> {os.path.relpath(CASES_CSV)}, audio in {os.path.relpath(AUDIO_DIR)}/")


# ---------------------------------------------------------------- results by condition
def report(results_csv: str, cases_csv: str = CASES_CSV):
    """The scorecard's results, split by line condition and by how the numbers were read out."""
    meta = {}
    with open(cases_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["id"]] = dict(kv.split("=", 1) for kv in r["notes"].split("; ") if "=" in kv and not kv.startswith("spoken"))
            meta[r["id"]]["lang"] = r["lang"]
    with open(results_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["id"] in meta]
    checks = sorted({k for r in rows for k in r if k.startswith("heard_slot_") or k == "heard_intent_ok"})
    for dim in ("condition", "amount_style", "id_style", "lang"):
        groups = defaultdict(list)
        for r in rows:
            groups[meta[r["id"]].get(dim, "-")].append(r)
        print(f"\nBy {dim}:")
        print(f"  {'':16}{'cases':>6}" + "".join(f"{c.replace('heard_slot_', '').replace('heard_', ''):>12}" for c in checks))
        for g, rs in sorted(groups.items()):
            cells = []
            for c in checks:
                vals = [r[c] for r in rs if r.get(c) not in ("", None)]
                cells.append(f"{100 * sum(v == 'True' for v in vals) / len(vals):>11.0f}%" if vals else f"{'–':>12}")
            print(f"  {g:16}{len(rs):>6}" + "".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m evals.synth_audio", description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=48, help="how many cases (default 48: every condition 8 times)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--report", metavar="RESULTS_CSV", help="split a scorecard run's results by condition")
    a = ap.parse_args()
    if a.report:
        report(a.report)
    else:
        generate(a.n, a.seed)
    sys.exit(0)

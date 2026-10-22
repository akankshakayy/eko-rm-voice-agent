# Scorecard: how well does Eko Sahayak hear and understand real calls?

```bash
python -m evals.run                 # score every case in cases.csv (uses Groq; ~3 min for 22 cases)
python -m evals.run --synthesize    # also make clean Edge-voice audio for cases with no recording
python -m evals.run --engine rules  # score the offline rules engine instead
python -m evals.run --only dmt_english,angry
```

Results go to `evals/results/<timestamp>/`: `report.md` (the scorecard and every miss), `results.csv` (one row
per case, everything measured) and `summary.json`. The run uses a throwaway database, so it never adds calls or
tickets to the real one. If Groq's free tier rate-limits a turn, the case waits and retries rather than scoring the
fallback as a mistake.

## What it measures

| | How |
|---|---|
| Hearing | Whisper's transcript vs `reference_text`: word and character error rate, stray `<\|..\|>` tokens |
| Understanding | Issue type, next action, and details (amount, transaction ID, UTR, bank, payment mode, month, product) after the caller's first sentence |
| Damage from mishearing | The same scoring on `reference_text`; the gap between "from audio" and "from correct text" is caused by speech-to-text alone |
| Speed | Speech-to-text, LLM, and voice generation per turn, plus end of speech to agent voice (including the 1.2 s silence wait) |

## Adding real recordings

1. **Get consent first.** Recordings contain the CSP's voice and possibly customer details. Only record people who
   have agreed, and keep the files inside Eko. `evals/audio/` is git-ignored so they are never committed.
2. Record the CSP's **opening sentence** describing a real problem, the way they'd say it on a call: shop noise,
   accents, Hinglish, numbers and IDs read aloud. Phone recordings are ideal. Any format Whisper accepts works
   (`.mp3`, `.m4a`, `.wav`, `.ogg`, `.webm`), up to 25 MB.
3. Put the file in `evals/audio/` and add a row to `cases.csv`:

| Column | Fill with |
|---|---|
| `id` | Short unique name, e.g. `patna_dmt_03` |
| `lang` | `hi` or `en` |
| `audio` | File name in `evals/audio/`, e.g. `patna_dmt_03.m4a` |
| `reference_text` | What was actually said, word for word (optional but recommended). Write Hindi in Devanagari: Whisper does, so Roman Hinglish inflates the error rates |
| `expected_intent` | One of `dmt_failed`, `aeps_issue`, `wallet_topup`, `commission`, `kyc_onboarding`, `app_device`, `bbps_recharge`, `refund_status`, `fraud_report`, `ticket_status`, `general_query` (blank = don't score) |
| `expected_action` | After that one sentence: `continue` (asks a follow-up), `escalated`, `ticket_raised` or `resolved_by_agent` (blank = don't score) |
| `amount`, `txn_id`, `utr_or_ref`, `bank_name`, `payment_mode`, `month`, `product` | The details actually stated in the recording; leave blank if not said. Amounts as digits (`2500`) |

**From the recordings pilot:** `python -m app.recordings export` cuts the CSP's opening complaint out of every
processed call and adds a draft row to `evals/candidates.csv` (git-ignored). `reference_text` is left empty on purpose,
because Whisper's own text would score itself as perfect. Listen to the clip, fill in the row, and move it here.

Aim for 20 to 30 recordings across issue types, regions and noise levels. The 22 starter rows are written
phrases to check the pipeline; replace or extend them with real ones.

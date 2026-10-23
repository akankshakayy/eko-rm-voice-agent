# Eko Sahayak: voice agent for RMs (v0.1)

A voice agent that answers and makes calls to CSPs on behalf of Relationship Managers. It understands the
issue (in Hindi or Hinglish), collects the details, raises a ticket, and escalates calls it can't handle.
Everything shows up live on the dashboards.

| Page | URL | What it shows |
|---|---|---|
| Call console | `/call` | Talk to the agent as a CSP using your mic or by typing. Inbound or outbound. |
| RM dashboard | `/rm` | Escalation queue, tickets for the RM's CSPs, transcripts, status changes, category confirmation |
| Agent & model performance | `/ops` | Live calls, volume, intent, outcome, duration, ticket lifecycle, intent accuracy, confidence, latency, tokens and cost |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # add GROQ_API_KEY (free) to use a real LLM
python -m app.simulator --backfill 150 --days 7          # optional: demo history for the charts
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000/call and click **Start call**.

**Engines.** Without an API key the agent runs on the **rules fallback engine**, which is free and works offline.
With a key it uses an LLM:

| Provider | Cost | Key | Notes |
|---|---|---|---|
| **Groq** (`GROQ_API_KEY`) | free tier, rate-limited | https://console.groq.com/keys | Default chain `qwen/qwen3.8-27b` → `openai/gpt-oss-120b` → `openai/gpt-oss-20b`: each model has its own free limit (~7–8k tokens/min, roughly 4 turns), so the next one answers when one is limited. Groq retires models often; list yours at console.groq.com. Fast, good for voice. Also powers speech-to-text. |
| Gemini (`GEMINI_API_KEY`) | free tier, rate-limited | https://aistudio.google.com/apikey | Default `gemini-2.5-flash`, via Google's OpenAI-compatible endpoint. |
| Claude (`ANTHROPIC_API_KEY`) | paid | https://console.anthropic.com | Default `claude-haiku-4-5`. |

`LLM_PROVIDER=auto` picks the first one with a key (Groq, then Gemini, then Claude). If an LLM call fails
(rate limit, outage) that turn falls back to the rules engine, so the call carries on.

**Voice.** The agent speaks with a free Microsoft neural voice (`edge-tts`, Hindi `hi-IN-SwaraNeural`), generated
on the server, so no browser voice is needed. For listening, a Groq key enables Whisper speech-to-text so the
mic works in any browser; without it the mic uses the browser's Web Speech API (Chrome or Edge only).

**Login and roles.** Every page and API needs an account. Open http://localhost:8000 and click **Create an account**.

| Role | Can use |
|---|---|
| **Admin** | Everything, plus the **Users** page: approve signups, change roles, link RMs to profiles, remove accounts |
| **Ops** | Call console, RM dashboard for all RMs, agent & model performance, simulator |
| **RM** | RM dashboard for their own CSPs and tickets only |
| **Pending** | Nothing; new signups wait here until an admin approves them |

The **first account ever created becomes Admin**; everyone after that signs up as Pending. Roles are enforced on
the server for every page and API call (RMs asking for another RM's data get a 404), and the UI hides what a role
can't use. Passwords are hashed with scrypt; logins last 7 days (HttpOnly session cookie, stored hashed in the
`sessions` table); 5 wrong passwords lock an email for 15 minutes. Ticket changes are recorded under the signed-in
user's name. Set `COOKIE_SECURE=1` when serving over HTTPS.

**Abandoned calls.** Closing or reloading the call console tab hangs up the call. Any call with no activity for
10 minutes (`CALL_IDLE_TIMEOUT_MIN`) is closed automatically by a background check that runs every minute, exactly
like a hang-up: if part of an issue was captured, the RM still gets an escalated ticket.

**Languages.** The caller can talk in **Hindi or English**. In the call console, "Caller's language" either fixes the
language up front or (default) lets the agent ask: *"हिंदी में बात करने के लिए 'हिंदी' बोलिए। For English, please say 'English'."*
If the caller skips the question and just describes the problem, the language is guessed from what they said. They can
switch at any time ("English please", "हिंदी में बात कीजिए"). Hindi replies are written in Devanagari, with banking terms
like DMT, transaction ID and UTR kept in English letters, so the Hindi voice pronounces them naturally. All fixed phrases
and slot questions live in `app/i18n.py`.

**Scorecard.** `python -m evals.run` measures how well the agent hears and understands calls, and how fast it
answers, over the cases in `evals/cases.csv` (add real CSP recordings to `evals/audio/`). See `evals/README.md`.

## Call recordings pilot

Real RM–CSP calls, collected every day to teach and test the agent. RMs call CSPs on Eko-issued SIMs, so every
call on that SIM is a work call.

```
RM's phone: dialer records the call → Sahayak Recorder Sync app (uploader/) uploads each Eko-SIM call ~3 min after it ends
   → POST /api/uploader/recordings (upload token per phone) → data/recordings/ (git-ignored)
   → processed straight away: Whisper → mask personal numbers → LLM review
       (who spoke each line: RM or CSP, issue type, details, CSP mood, how it ended, how the RM handled it, summary)
   → python -m app.recordings export → draft scorecard cases in evals/candidates.csv
```

1. **Set up a phone:** turn on automatic call recording in the dialer, install the app (see `uploader/README.md`),
   and paste an upload token from **Users → Recorder uploads** (admin only), or make one with
   `python -m app.recordings token --rm 1 --label "Asha's phone"`. The app only uploads calls made on the chosen
   Eko SIM, never the RM's other SIM, and it never places calls itself.
2. **After every call:** the app uploads the recording about 3 minutes after the call ends (8 pm every day is a
   safety net), and the server processes it within seconds. If Groq's free tier runs out, the rest waits and is
   retried every 30 minutes; a recording that fails is retried after 10 and 20 minutes. To process once a day
   instead, set `RECORDINGS_DAILY_AT=21:00`. To run it now: `python -m app.recordings process`.
3. **See the results:** `python -m app.recordings list`, `GET /api/recordings` and `GET /api/recordings/{id}`
   (admin/ops) return the transcript with a speaker on every line, plus the review.
4. **Feed the scorecard:** `python -m app.recordings export` cuts each CSP's opening complaint into `evals/audio/`
   and adds a draft row to `evals/candidates.csv`. Listen, write `reference_text` and the expected answers, then
   move the row to `evals/cases.csv`.

Aadhaar, PAN, card, account, phone and OTP/PIN numbers are masked from transcripts before they are stored or sent to
the LLM. Transaction IDs, UTRs and amounts are kept. Masking is regex-based, so numbers spoken as Hindi words get
through, and the original audio is kept unmasked. CSPs should know their calls are recorded: Google Dialer
announces it, but other dialers don't.

## Connecting other systems

Other systems can call the API with a service API key, and get webhooks when tickets, calls or recordings change.
See **[INTEGRATION.md](INTEGRATION.md)** for keys, endpoints, events, payloads and signature checks.

## How a call works

```
CSP speaks → speech-to-text (browser now, telephony later) → POST /api/calls/{id}/turn
   → guards: unclear audio? asked for RM? angry?        → escalate
   → engine (LLM via forced tool call, or rules)       → reply + intent + details + action
   → guards: fraud must escalate, max turns, engine failure → rules fallback
   → action: continue | create_ticket | escalate | resolved_by_agent | end_call
   → ticket in SQLite → RM dashboard + ops metrics (dashboards poll every 4–5 s)
```

* **Not understood:** speech confidence below 0.45, or the engine saying `understood=false`, twice in a row.
  The call is marked `not_understood` and an escalated ticket goes to the RM.
* **Always escalated:** the CSP asks for the RM or a human, sounds angry, reports fraud (critical, 2h SLA), the call
  runs past 12 turns, or the CSP hangs up mid-issue. Escalated tickets get a 4h SLA.
* **Solved by the agent (first Phase-4 skill):** "mere complaint ka kya hua" is answered from the ticket DB, and no new ticket is raised.
* **Unknown caller number:** the agent asks for the CSP ID and links the call to the right CSP and RM.

## Metrics

**Agent workflow:** total, inbound and outbound calls, ongoing calls with live timers, average and median duration, intent mix,
outcome mix, duration histogram, ticket-raised %, escalation %, not-understood %, % handled without the RM, ticket
open/in-progress/closed counts, average time to close, SLA breaches, and a recent-calls table whose ticket status updates when the RM closes the ticket.

**Model:** intent accuracy (the agent's category vs the category the RM confirms in the ticket drawer, both overall and per category),
top mix-ups, average intent confidence and % of turns with low confidence, speech-to-text confidence, NLU latency p50 and p95 (Claude-only too),
how many required details were captured, turns per call, tokens, estimated cost in ₹ (`LLM_PRICE_*` in `.env`), and engine errors and fallbacks.
Click any call to see per-turn data: STT confidence, intent, confidence, latency and engine.

> The simulator's tickets are confirmed with each scenario's *true* category, so accuracy on the demo
> data is a real measurement of the engine. It is still only scripted data. Real accuracy comes from RMs
> confirming categories on live tickets.

## Project layout

```
app/
  taxonomy.py      issue categories, required details, priorities, SLAs, thresholds  ← edit first
  agent.py         call orchestrator: state, guards, escalation, ticketing
  llm_engine.py    Claude engine (forced `agent_turn` tool call → structured output)
  rules_engine.py  keyword/regex fallback engine
  i18n.py          Hindi + English phrases, language choice and detection
  auth.py          accounts, roles, password hashing, sessions
  metrics.py       dashboard aggregations
  simulator.py     synthetic CSP callers (live or backfill)
  seed.py          demo RMs and CSPs (replace with Eko CSP master sync)
  recordings.py    call recordings pilot: uploads, transcription + review as calls arrive, export to the scorecard
  integrations.py  service API keys and signed webhooks for other systems (see INTEGRATION.md)
  db.py, main.py   SQLite + FastAPI
static/            call console, RM dashboard, ops dashboard (vanilla JS + Chart.js, vendored)
tests/             pytest suite (incl. a mocked Claude engine test)
uploader/          Android app that uploads the RMs' Eko-SIM call recordings every day
```

## Simulator

```bash
python -m app.simulator --backfill 150 --days 7     # history
python -m app.simulator --live 5                     # calls you can watch as "ongoing"
python -m app.simulator --backfill 30 --engine auto  # use Claude for simulated calls (costs tokens)
```
The ops dashboard also has a **▶ Simulate 5 live calls** button.

## Next steps (roadmap)

1. **Real phone calls:** connect Exotel, Plivo or Twilio media streams to a streaming STT (Sarvam, Google Chirp, Deepgram) and
   TTS. The agent API (`start_call`, `handle_turn`, `hangup`) stays the same; only the audio layer changes.
   Pipecat or LiveKit Agents can host that loop.
2. **Real data:** sync CSPs and RMs from Eko's master data, and push tickets into the existing CRM or ticketing tool.
3. **More issues the agent can solve itself:** read-only tools for transaction status, refund ETA, wallet balance and commission statements,
   plus guided troubleshooting from Eko's SOPs (retrieval over the SOP documents).
4. **Outbound campaigns:** scheduled follow-ups after a ticket closes, nudges to inactive CSPs, and wallet low-balance reminders.
5. **Production hardening:** Postgres, auth and roles for RMs, PII masking in transcripts, call-recording consent,
   and data kept in India where required.

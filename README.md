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
python -m app.kb_import data/kb/inbox/<calling sheet>.xlsx   # the real RMs, CSPs and slabs (see below)
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000/call and click **Start call**. For a demo with made-up CSPs instead, use a separate
database: `EKO_DB_PATH=data/demo.db DEMO_DATA=1 uvicorn app.main:app --port 8001` (the simulator only runs there).

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

**Login and roles.** Every page and API needs an account, and only Eko employees can have one. Sign in with
**Google** (an Eko Google account), or with email and password.

| Role | Can use |
|---|---|
| **Admin** | Everything, plus the **Users** page: change roles, link RMs to profiles, remove accounts |
| **Ops** | Call console, RM dashboard for all RMs, agent & model performance, recordings (new accounts start here) |
| **RM** | RM dashboard, CSPs, tickets and recordings for their own CSPs only |
| **No access** | Nothing: an admin can set this to take someone's access away |

**No admin approval.** A new account works as soon as its email is proven to belong to the person: by Google
(straight away) or by the confirmation link in an email (password accounts). Then it is Ops, or **RM** when the email is
an RM's on the calling sheet (linked to that RM profile). The RM picked by a user is never trusted. Admins are the
addresses in `ADMIN_EMAILS`; without it, the very first account becomes admin (fine on a laptop; set `ADMIN_EMAILS`
on a server). Roles are enforced on the server for every page and API call (RMs asking for another RM's data get a
404). Ticket changes are recorded under the signed-in user's name.

**Sign in with Google**: in Google Cloud Console make an OAuth client (type *Web application*) with the redirect URI
`<APP_BASE_URL>/api/auth/google/callback`, e.g. `https://indev.eko.in/rm-voice-agent/api/auth/google/callback`, then set
`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` and `APP_BASE_URL`. Only verified addresses in `ALLOWED_EMAIL_DOMAINS` get in.
**Without email (SMTP) set up**, password accounts can't be confirmed, so new accounts are made with Google only.

Account security:
- Only work emails can sign up (`ALLOWED_EMAIL_DOMAINS`, default `eko.co.in,ekosupport.in`).
- Passwords: at least 10 characters, not a common password, not containing the user's name or email; hashed with scrypt.
- 5 wrong passwords lock an email for 15 minutes, and 20 failed sign-ins lock the address they come from (counted
  in the database, so a restart doesn't reset it). Signups and reset emails are limited per address too.
- The login cookie is sent only to Sahayak's own path (taken from `APP_BASE_URL`, e.g. `/rm-voice-agent`), so other
  apps on a shared host never receive it.
- Logins last 7 days but end after 12 hours unused (`SESSION_IDLE_HOURS`); cookie is HttpOnly, stored hashed.
- **Forgot password**: emails a one-time link valid for 30 minutes (max 3 an hour); using it signs the account out
  everywhere. Without email set up, an admin makes the link on the **Users** page and passes it on.
- **Email linking**: when a proven email (Google, or a confirmed link) is the one the calling sheet gives for an
  RM, the account is linked to that RM profile and gets the RM role. The signup page shows no RM names.
- A first Google sign-in to an account whose email was never confirmed removes that account's password and signs
  it out everywhere, so nobody can prepare an account in someone else's name in advance.
- **Account page** (click your name): change password (other devices are signed out), change email (only after the
  new address confirms), sign out everywhere, recent sign-in activity. Everything is logged in `auth_events`.
- Emails need `SMTP_HOST`, `SMTP_FROM` and `APP_BASE_URL` (links are built from it, never from the request).
  They contain only a link, never CSP data. Set `COOKIE_SECURE=1` over HTTPS and `TRUST_PROXY=1` behind nginx.

**Real CSP data (knowledge base).** Eko's sheets go in `data/kb/inbox/` (git-ignored, never committed: a
pre-commit hook in `deploy/pre-commit` blocks spreadsheets, databases and recordings; install it with
`cp deploy/pre-commit .git/hooks/`). `python -m app.kb_import <file>` reads the **Calling Sheet New** and **Slab**
tabs without changing the file: RMs (with their sheet email and phone), CSPs, and per CSP per month the targets,
month-to-date PMJDY/PMJJBY/PMSBY/APY counts, slab, transaction count and commission (`csp_monthly`). The same file
twice changes nothing; CSPs that leave the sheet are marked inactive, not deleted; a number shared by several CSPs
is kept but never used to recognise a caller. **Slab rule** (matches every month of the sheet): a CSP reaches the
highest slab where at least 3 of the 4 schemes meet that slab's count (`kb_import.slab_for`). The first import into
a database that still has the made-up data needs `--remove-demo`: it copies the database to `data/demo.db` and
deletes the demo RMs/CSPs and every simulated call and ticket (`--map-rm 1=Shiwani` moves logins and upload tokens
of a demo RM to a real one instead of unlinking them). The LLM only ever sees a CSP's code and first name.

**Test CSPs and the call console.** Real CSPs are never called while Sahayak is being tested: every call that would
involve one (by number, by picking it, or by saying its code mid-call) is refused until `REAL_CSP_CALLS=1`.
`python -m app.synthetic` gives each RM 20 made-up test CSPs (codes `TST…`, mobile numbers starting with 5, which no
Indian mobile does; names never match a real CSP). The first 11 are edge cases: every slab situation this month, a
number shared by two CSPs, no number, one that left the sheet. The rest have April-to-now history drawn from the real
sheet's per-category percentiles (5th to 95th only, so no single CSP's figures are reproduced). In the call
console, pick an RM to see their own CSPs and search by code or name: test CSPs can be called, real CSPs are listed
for reference only (without their numbers). Calls to test CSPs are marked as tests and send no webhook events.

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
   `python -m app.recordings token --rm 4 --label "Shiwani's phone"`. In the app's setup the RM picks **their
   name**, and every recording is filed under it. A personal token (made for one RM) fixes the name and refuses
   any other; a shared token (`token` without `--rm`, or "Any RM" on the Users page) lets each RM pick theirs.
   The app only uploads calls made on the chosen Eko SIM, never the RM's other SIM, and it never places calls.
2. **After every call:** the app uploads the recording about 3 minutes after the call ends (8 pm every day is a
   safety net), and the server processes it within seconds. If Groq's free tier runs out, the rest waits and is
   retried every 30 minutes; a recording that fails is retried after 10 and 20 minutes. To process once a day
   instead, set `RECORDINGS_DAILY_AT=21:00`. To run it now: `python -m app.recordings process`.
3. **See the results:** `python -m app.recordings list`, `GET /api/recordings` and `GET /api/recordings/{id}`
   (admin/ops) return the transcript with a speaker on every line, plus the review.
4. **Feed the scorecard:** `python -m app.recordings export` cuts each CSP's opening complaint into `evals/audio/`
   and adds a draft row to `evals/candidates.csv`. Listen, write `reference_text` and the expected answers, then
   move the row to `evals/cases.csv`.

Aadhaar, PAN, card, account, phone and OTP/PIN numbers are masked from the transcripts that are shown, sent out
by webhook or sent to the LLM. Transaction IDs, UTRs and amounts are kept. Masking is regex-based, so numbers spoken
as Hindi words get through. The original audio, and Whisper's raw output kept so "Process again" needn't redo it,
are not masked and never leave the server.

**Speakers are told apart by voice** (speaker diarization) with two small local models from sherpa-onnx, about
50 MB, no GPU or account needed. Fetch them once per machine with `python -m app.diarize download`. Every word Whisper
hears goes to the voice speaking at that moment, so a transcript line never mixes two people; the page shows them as
Speaker 1, Speaker 2, and on RM–CSP calls also says which is the RM. Without the models, lines are split by sentence. CSPs should know their calls are recorded: Google Dialer
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
  auth.py          accounts, roles, password hashing, sessions, reset/verification links, email linking
  mailer.py        account emails over SMTP (off until configured)
  kb_import.py     imports Eko's calling sheet + slab table; the slab rule
  synthetic.py     made-up test CSPs for each RM (the only CSPs the console can call while testing)
  metrics.py       dashboard aggregations
  simulator.py     synthetic CSP callers (live or backfill)
  seed.py          made-up RMs and CSPs for tests, scorecard and simulator (data/demo.db)
  recordings.py    call recordings pilot: uploads, transcription + review as calls arrive, export to the scorecard
  integrations.py  service API keys and signed webhooks for other systems (see INTEGRATION.md)
  db.py, main.py   SQLite + FastAPI
static/            call console, RM dashboard, ops dashboard (vanilla JS + Chart.js, vendored)
tests/             pytest suite (incl. a mocked Claude engine test)
uploader/          Android app that uploads the RMs' Eko-SIM call recordings every day
```

## Simulator

The simulator only runs on a database without real CSPs (`EKO_DB_PATH=data/demo.db DEMO_DATA=1`).

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

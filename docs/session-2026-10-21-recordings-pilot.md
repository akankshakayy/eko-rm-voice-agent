# Session notes: call recordings pilot (21 Oct 2026)

Resume this conversation with `claude --continue` (or `claude --resume` to pick it from a list).

## What we decided
- Collect real RM–CSP call recordings straight from the RMs' phones to train and test Sahayak.
- RMs call CSPs on **Eko-issued SIMs**, so every call on that SIM counts as work data. Calls are told apart by SIM,
  not with the calling sheet.
- **Automatic daily upload**, not weekly and not by hand over WhatsApp.
- Why an app that records calls itself doesn't work: since Android 9, third-party apps can't capture call audio,
  and Google Play has banned call recording through the Accessibility API since 2022. So the phone's own dialer
  records the call and our app only uploads the files it makes.

## What was built
**Android app, `uploader/`** ("Sahayak Recorder Sync")
- APK: `uploader/app/build/outputs/apk/debug/app-debug.apk` (Android 8 and later). 10 unit tests pass.
- Setup screen: server address, upload token, Eko SIM, recordings folder.
- Uploads every day around 20:00, and has an "Upload now" button.
- Matches each recording file to a call-log entry and uploads only calls on the Eko SIM.
- Can't make calls, send SMS or record audio itself.
- Setup steps for each phone brand: `uploader/README.md`.

**Server, `app/recordings.py` + routes in `app/main.py`**
- `POST /api/uploader/recordings` and `GET /api/uploader/ping` sign in with an upload token per phone.
  Admins create and revoke tokens under **Users → Recorder uploads**.
- Daily run at 21:00 IST (`RECORDINGS_DAILY_AT`): Whisper, then masking of personal numbers (Aadhaar, PAN, card,
  account, phone, OTP), then an LLM review: who spoke each line (RM or CSP), issue type, details, CSP mood,
  how the call ended, how the RM handled it, and a summary.
- Commands: `python -m app.recordings token --rm 1 --label "..."`, `process`, `list`, `export`.
- `export` cuts each CSP's opening complaint into `evals/audio/` and adds a draft row to `evals/candidates.csv`.
  Someone listens, fills in the row and moves it to `evals/cases.csv`.
- All 44 tests pass. A live Groq check worked: Whisper on the existing recording, and the review on a made-up
  transcript.

## Next steps
1. **Hosting:** phones can't reach a server on this laptop. Use an Eko server with HTTPS, or a temporary tunnel.
2. **Test on one phone:** install the app on an Eko-SIM phone and call a colleague (not a CSP). Check that the
   recording uploads and gets processed. This also shows whether SIM matching works on that phone brand.
3. **Telling CSPs:** only Google Dialer announces that the call is recorded. On other phones the RM should say it.
4. **Rebuilding the app:** the Android SDK was downloaded into a temporary folder, so install Android Studio or the
   SDK first.
5. Maybe later: a page in the ops dashboard to browse the processed recordings. For now they're only available
   through the API and the command line.

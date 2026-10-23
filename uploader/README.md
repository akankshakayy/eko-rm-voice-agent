# Sahayak Recorder Sync (Android)

A small Android app for the recordings pilot. A few minutes after every call it uploads the call recordings the phone's **own dialer**
already made, but **only for calls on the Eko-issued SIM**, to the Sahayak server. There they are transcribed and
summarized for training and scoring the agent.

| It does | It never does |
|---|---|
| Reads the call log to find which recording belongs to which call and SIM | Place calls, send SMS (no `CALL_PHONE` / `SEND_SMS` permission) |
| Reads the recordings folder you pick (Storage Access Framework, read-only) | Record audio itself (no `RECORD_AUDIO`), or read any other folder |
| Uploads each Eko-SIM call recording about 3 minutes after the call ends, checks again at 8 pm every day, plus an **Upload now** button | Upload recordings of calls on any other SIM, which stay on the phone |

## How it decides what to upload

1. Lists audio files (`.m4a .mp3 .aac .amr .3gp .ogg .opus .wav`) in the chosen folder and up to 3 levels of
   subfolders that changed since the last run. The first run looks back 7 days.
2. Matches each file to an answered call in the call log. A recording is closed when the call ends, so the file's
   modified time should fall within a few minutes of `DATE + DURATION`. If the file name has a `yyyyMMdd_HHmmss`
   timestamp, that is matched against the call's start time. If the file name has a phone number, only calls with
   that number count.
3. Uploads the file only if that call's `PHONE_ACCOUNT_ID` is the Eko SIM. Phones store this id differently: most
   use the subscription id, some use the SIM's ICCID (sometimes with a trailing `F` or cut short), and some use their
   own id. At setup the app records every id it can find for the chosen SIM: the subscription id, the ICCID where
   Android still exposes it (up to Android 9), and the Telecom phone-account id, which is the exact value the call
   log stores. A file with no matching call, or a call on another SIM, is skipped for good.
4. Files are never sent twice. The app keeps a ledger of handled files (URI + size + modified time), and the server
   also answers `duplicate: true` for a file it already has. A failed upload (network, 5xx) is retried on the next
   run. A file the server rejects (400, or 413 over 100 MB) is skipped and shown under **Last problem**.

## Server API it uses

| Call | Purpose |
|---|---|
| `GET /api/uploader/ping` with `Authorization: Bearer <token>` | Test connection: the RM a personal token belongs to, or the RMs to choose from |
| `POST /api/uploader/recordings` (multipart) | Fields: `audio`, `started_at` (epoch ms), `duration_sec`, `direction` (`incoming` / `outgoing` / `unknown`), `other_number`, `device_id` |

## Pilot setup (per RM phone)

1. **Get an upload token.** The Sahayak admin creates one on the **Users** page under **Recorder uploads**
   (or `POST /api/uploader/tokens`): either for one RM (that phone can only upload as them), or "Any RM", where
   the RM picks their name in the app. One token per phone is safest, so each can be revoked on its own.
2. **Turn on automatic call recording in the dialer:**

   | Phone | Where |
   |---|---|
   | Google Phone app (Pixel, Motorola, Nokia, many others) | Phone › ⋮ › Settings › Call recording › **Always record** (choose "Numbers not in your contacts" too, or both). The other side hears "This call is now being recorded". |
   | Xiaomi / Redmi / Poco (MIUI / HyperOS dialer) | Phone › ⚙ › Call recording › **Record calls automatically** › All numbers |
   | Samsung (One UI) | Phone › ⋮ › Settings › Record calls › **Auto record calls** › All calls |
   | OnePlus / Realme / Oppo (ColorOS dialer) | Phone › ⋮ › Settings › Call recording › **Auto-record** › All calls |

   Make one short test call on the **Eko SIM** and check that a recording appears.
3. **Install the app** (below), open it, and grant **Call logs** and **Phone** permissions. If Android shows
   "Restricted setting", go to App info › ⋮ › **Allow restricted settings**, then open the app again.
4. Enter the server address and paste the token, then tap **Test connection**. Under **2. Your name (RM)** pick
   the RM (a personal token shows its RM's name, locked). Every recording from the phone is filed under that name.
5. Pick the **Eko SIM**. Tap **Choose folder** and open the dialer's recordings folder (the app lists the usual
   locations), then tap **Use this folder › Allow**.
6. Tap **Save and start uploading**, then **Upload now** once. The status should show the test call as uploaded.
7. Turn off battery optimisation for the app (App info › Battery › **Unrestricted**). Xiaomi, Oppo and Realme
   otherwise stop background work, and that phone's uploads stop.

Server address: use `https://`. Plain `http://` is accepted only for a private LAN address (`192.168.x.x`, `10.x.x.x`,
`172.16–31.x.x`), for testing against a laptop on the same Wi-Fi.

## Build and install

Requires JDK 17+ and the Android SDK (platform 35).

```bash
cd uploader
echo "sdk.dir=/path/to/Android/sdk" > local.properties    # or set ANDROID_HOME
./gradlew test             # unit tests for the file ↔ call-log matching
./gradlew assembleDebug    # → app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Without `adb`, copy the APK to the phone and open it. Android asks once to allow installs from that source. For a
wider rollout, sign a release build or push it through Eko's MDM.

## Code

```
app/src/main/java/co/eko/sahayak/recsync/
  Matcher.kt       file ↔ call matching, Eko-SIM check, server URL rule (pure Kotlin, unit-tested)
  Device.kt        SIM list, call log, recordings folder listing
  UploadWorker.kt  upload after each call (call-log trigger), the 8 pm safety run and "Upload now"
  Api.kt           ping + multipart upload (OkHttp)
  Settings.kt      setup values (token in EncryptedSharedPreferences), handled-file ledger, last run status
  MainActivity.kt  the setup and status screen
```

# Integrating with Eko Sahayak

How other systems (Eko's CRM or ticketing tool, data pipelines, telephony, other teams' services) connect to
Sahayak. There are three ways in and out:

| | Direction | Use it to |
|---|---|---|
| **REST API + service API key** | them → Sahayak | read and update tickets, calls, recordings and metrics; run calls |
| **Webhooks** | Sahayak → them | be told the moment a ticket is raised or changed, a call ends, or a recording is processed |
| **Agent API** (`start_call` / `turn` / `hangup`) | telephony → Sahayak | put a phone line in front of the agent |

The full, always up-to-date list of endpoints and fields is the OpenAPI description at **`/openapi.json`**. Browse
it at **`/docs`** while signed in.

## Stability

- Everything under `/api/` and every webhook payload below counts as the public contract.
- Changes are additive only: new endpoints, new fields and new event types. Receivers must ignore fields they
  don't know.
- A breaking change (renaming or removing a field, or changing a meaning) would come as `/api/v2/` with the
  old version kept running alongside. This hasn't happened yet.
- Times are Unix epoch seconds (floats), in UTC. Amounts are in rupees. IDs are integers. Ticket numbers look
  like `EKO-261022-0042`.

## 1. Calling the API with a service API key

An admin makes one key per system:

```bash
python -m app.integrations key --name "Eko CRM"     # prints the key once: ekosk_...
python -m app.integrations keys                     # list; revoke-key <id> to cut it off
```

You can also use `POST /api/integrations/keys` with `{"name": "Eko CRM"}` while signed in as admin. Then send the
key on every request:

```bash
curl -H "Authorization: Bearer ekosk_..." https://sahayak.example/api/tickets?status=open
```

- A key acts with the **Ops** role. It can use the dashboards' API, but it can never manage users, keys or webhooks.
- Changes made with a key appear in the ticket history under the key's name, e.g. `Eko CRM (API)`.
- A wrong or revoked key gets `401`. A forbidden action gets `403`.

The endpoints most integrations need:

| Endpoint | What |
|---|---|
| `GET /api/tickets?status=&rm_id=&escalated=` | Tickets, newest first |
| `GET /api/tickets/{id}` | One ticket with its history and the call transcript |
| `PATCH /api/tickets/{id}` `{"status": "in_progress" \| "resolved" \| "closed" \| "open", "rm_category": "...", "note": "..."}` | Update a ticket (e.g. when it is closed in the CRM) |
| `GET /api/calls/{id}` | One call |
| `GET /api/recordings`, `GET /api/recordings/{id}` | Recorded RM–CSP calls from the pilot, with speakers and review |
| `GET /api/metrics/agent`, `GET /api/metrics/model` | The dashboard numbers |
| `GET /api/config` | Issue categories, the details each one needs, thresholds |

## 2. Receiving webhooks

An admin subscribes a URL to some or all events:

```bash
python -m app.integrations webhook --url https://crm.example/hooks/sahayak --events ticket.created,ticket.updated
python -m app.integrations test-webhook 1           # send a test event now
python -m app.integrations deliveries               # what was sent, what failed and why
```

You can also use `POST /api/integrations/webhooks` with `{"url": "...", "events": ["ticket.created"]}`, where
`["*"]` means every event. The signing secret (`whsec_...`) is shown once: give it to the receiving system.

### Events

| Event | When | `data` |
|---|---|---|
| `ticket.created` | The agent raises or escalates a ticket | ticket |
| `ticket.updated` | Status, confirmed category or a note changes | ticket, plus `changes` (e.g. `["status"]`) and `changed_by` |
| `call.ended` | A call with the agent ends, however it ends | call, with `transcript` |
| `recording.processed` | A recorded RM–CSP call has been processed (minutes after the call) | recording, with `transcript` (speakers labelled, personal numbers masked) |

Calls from the simulator (demo data) send no events.

### What arrives

```http
POST /hooks/sahayak
Content-Type: application/json
X-Sahayak-Event: ticket.created
X-Sahayak-Delivery: 1842
X-Sahayak-Timestamp: 1792656000
X-Sahayak-Signature: sha256=6f1c...

{"event": "ticket.created", "created_at": 1792656000.2, "data": {
  "id": 42, "ticket_no": "EKO-261022-0042", "call_id": 311, "category": "dmt_failed", "rm_category": null,
  "priority": "high", "status": "open", "escalated": false, "escalation_reason": null,
  "summary": "DMT of 5000 failed; money debited but not credited",
  "details": {"txn_id": "TXN482913577", "amount": "5000", "txn_date": "yesterday"}, "slot_completeness": 1.0,
  "created_at": 1792656000.1, "updated_at": 1792656000.1, "closed_at": null, "sla_due_at": 1792742400.1,
  "csp": {"id": 1, "code": "CSP10231", "name": "Ramesh Kumar", "shop": "Kumar Mobile Point",
          "phone": "9000000101", "district": "Patna"},
  "rm": {"id": 1, "name": "Rahul Verma"}}}
```

### Rules for receivers

- **Check the signature.** Compute HMAC-SHA256 over `<X-Sahayak-Timestamp>.<raw body>` with the secret, and
  reject the request if it doesn't match or the timestamp is more than 5 minutes old:
  ```python
  import hashlib, hmac, time
  def valid(secret: str, headers, body: bytes) -> bool:
      ts = headers["X-Sahayak-Timestamp"]
      want = "sha256=" + hmac.new(secret.encode(), ts.encode() + b"." + body, hashlib.sha256).hexdigest()
      return hmac.compare_digest(want, headers["X-Sahayak-Signature"]) and abs(time.time() - int(ts)) < 300
  ```
- **Answer 2xx quickly** (within 10 s), and do slow work afterwards. Any other answer, or a timeout, is retried
  after 30 s, 2 min, 10 min, 30 min, 1 h, 3 h, 6 h and 12 h. After that the delivery is marked failed.
- **Expect duplicates and odd ordering.** A retry can arrive after a newer event. Use `X-Sahayak-Delivery` to
  skip deliveries you've already seen, and the ticket's `updated_at` to keep the newest state.

Events are stored before they're sent, so a receiver that is down, or a Sahayak restart, doesn't lose them.

## 3. Telephony and other plug-in points

| To plug in | Where |
|---|---|
| A phone line (Exotel, Plivo, Twilio, LiveKit, Pipecat…) | Drive the agent API: `POST /api/calls` → `POST /api/calls/{id}/turn` per caller utterance → `POST /api/calls/{id}/hangup`. Speech-to-text and voice stay on the telephony side, or use `/api/stt` and `/api/tts`. |
| Another LLM provider | `app/llm_engine.py`: add an entry to `PROVIDERS` (any OpenAI-compatible API) |
| Another speech-to-text | `app/stt.py` |
| New issue types, required details, SLAs | `app/taxonomy.py` |
| Eko's RM and CSP master data | **Not yet.** `app/seed.py` holds demo data. Real CSP data must be kept separate from the tables the simulator and tests write to, so this needs a decision before it's built. |
| Another database | `app/db.py` is the only module that talks to SQLite |

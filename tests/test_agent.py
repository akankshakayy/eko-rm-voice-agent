import json
import time
import os
import tempfile

import pytest

os.environ["AGENT_ENGINE"] = "rules"
os.environ["ALLOWED_EMAIL_DOMAINS"] = "*"   # tests use @eko.test addresses
for _k in ("SMTP_HOST", "SMTP_FROM", "APP_BASE_URL", "DEMO_DATA"):
    os.environ.pop(_k, None)


@pytest.fixture(autouse=True)
def fresh_db():
    from app import db
    from app.seed import seed
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    db.reset_connection(path)
    seed()
    yield


def talk(lines, phone="9000000101", direction="inbound"):
    from app import agent
    r = agent.start_call(direction, phone)
    cid = r["call_id"]
    for text, conf in lines:
        r = agent.handle_turn(cid, text, conf)
        if r["ended"]:
            break
    return r


def test_dmt_ticket_flow():
    r = talk([("Mera DMT transfer fail ho gaya", .9), ("TXN123456789", .9), ("5000 rupaye", .9), ("kal", .9),
              ("haan ji kar dijiye", .9)])
    assert r["ended"] and r["outcome"] == "ticket_raised"
    assert r["ticket"]["category"] == "dmt_failed" and r["ticket"]["slot_completeness"] == 1.0
    assert r["ticket"]["rm_id"] == 1


def test_not_understood_escalates_after_two_unclear_turns():
    r = talk([("hmm haan", .2), ("kya hello", .3)])
    assert r["outcome"] == "not_understood"
    assert r["ticket"]["escalated"] == 1


def test_human_request_escalates():
    r = talk([("Mujhe RM se baat karni hai", .9)])
    assert r["outcome"] == "escalated" and "RM" in r["ticket"]["escalation_reason"]


def test_fraud_always_escalates_critical():
    r = talk([("Kisi ne fraud transaction kar diya", .9)])
    assert r["outcome"] == "escalated" and r["ticket"]["priority"] == "critical"


def test_ticket_status_self_serve():
    talk([("Mera DMT transfer fail ho gaya", .9), ("TXN123456789", .9), ("5000 rupaye", .9), ("kal", .9), ("haan", .9)])
    r = talk([("Mere complaint ka kya hua, ticket status bataiye", .9)])
    assert r["outcome"] == "resolved_by_agent" and "खुला" in r["reply"]


def test_unknown_caller_identified_by_csp_code():
    from app import agent
    r = agent.start_call("inbound", "9999999999")
    r = agent.handle_turn(r["call_id"], "Mera CSP ID hai CSP 20311", .9)
    assert r["call"]["csp_code"] == "CSP20311" and r["call"]["rm_id"] == 2


def test_hangup_mid_issue_creates_escalation():
    from app import agent
    r = agent.start_call("inbound", "9000000102")
    agent.handle_turn(r["call_id"], "AePS se paisa kat gaya", .9)
    c = agent.hangup(r["call_id"])
    assert c["outcome"] == "dropped" and c["ticket_no"]


def test_rm_closes_ticket_and_metrics_reflect_it():
    from app import agent, metrics
    r = talk([("Mujhe RM se baat karni hai", .9)])
    agent.update_ticket(r["ticket"]["id"], status="closed", rm_category="general_query")
    m = metrics.agent_metrics(24)
    assert m["tickets"]["closed"] == 1 and m["recent_calls"][0]["ticket_status"] == "closed"
    mm = metrics.model_metrics(24)
    assert mm["intent"]["labelled_tickets"] == 1


def test_claude_failure_falls_back_to_rules(monkeypatch):
    from app import agent

    class Boom:
        name = "claude"

        def decide(self, *a):
            raise TimeoutError("simulated outage")
    monkeypatch.setattr(agent, "_engine", lambda: Boom())
    r = agent.start_call("inbound", "9000000101")
    r = agent.handle_turn(r["call_id"], "Mera DMT transfer fail ho gaya", .9)
    assert not r["ended"] and "transaction ID" in r["reply"]


def _client(signed_in=True, **signup):
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    if signed_in:
        body = {"email": "rahul@eko.test", "password": "correct horse", "name": "Rahul Verma", "rm_id": 1, **signup}
        assert c.post("/api/auth/signup", json=body).status_code == 200
    return c


def test_api_endpoints():
    c = _client()
    r = c.post("/api/calls", json={"direction": "inbound", "phone": "9000000101"}).json()
    r = c.post(f"/api/calls/{r['call_id']}/turn", json={"text": "Mujhe RM se baat karni hai", "stt_confidence": .9}).json()
    tid = r["ticket"]["id"]
    assert c.patch(f"/api/tickets/{tid}", json={"status": "in_progress"}).json()["status"] == "in_progress"
    assert c.get("/api/metrics/agent").status_code == 200
    assert c.get("/api/metrics/model").status_code == 200
    assert c.get(f"/api/tickets/{tid}").json()["transcript"]


def test_claude_engine_parsing_with_mock_client():
    """Exercises ClaudeEngine request building + tool_use parsing without a real API call."""
    from types import SimpleNamespace as NS
    from app import agent
    from app.llm_engine import ClaudeEngine
    captured = {}

    class FakeMessages:
        def create(self, **kw):
            captured.update(kw)
            return NS(content=[NS(type="tool_use", input={
                "reply": "Samajh gayi. Transaction ID bataiye.", "understood": True, "language": "hinglish",
                "intent": "dmt_failed", "intent_confidence": 0.92, "slots": {"amount": "5000"},
                "sentiment": "neutral", "action": "continue"})],
                usage=NS(input_tokens=900, output_tokens=60))

    eng = ClaudeEngine.__new__(ClaudeEngine)
    eng.client, eng.model = NS(messages=FakeMessages()), "test-model"
    r = agent.start_call("inbound", "9000000101")
    call = agent._call(r["call_id"])
    d = eng.decide(agent._state(call), "5000 ka DMT fail ho gaya")
    assert d["intent"] == "dmt_failed" and d["slots"]["amount"] == "5000" and d["input_tokens"] == 900
    msgs = captured["messages"]
    assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "user"
    assert all(msgs[i]["role"] != msgs[i + 1]["role"] for i in range(len(msgs) - 1))
    assert captured["tool_choice"] == {"type": "tool", "name": "agent_turn"}


def test_openai_compat_engine_parsing_with_mock_client():
    """Groq/Gemini path: forced function call is parsed into the same decision shape as Claude."""
    import json
    from types import SimpleNamespace as NS
    from app import agent
    from app.llm_engine import OpenAICompatEngine, _compat_tool
    captured = {}

    class FakeCompletions:
        def create(self, **kw):
            captured.update(kw)
            args = {"reply": "Ji, transaction ID bataiye.", "understood": True, "language": "hinglish",
                    "intent": "dmt_failed", "intent_confidence": 0.9, "slots": {"amount": 5000},
                    "sentiment": "neutral", "action": "continue"}
            msg = NS(tool_calls=[NS(function=NS(name="agent_turn", arguments=json.dumps(args)))])
            return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=800, completion_tokens=50))

    eng = OpenAICompatEngine.__new__(OpenAICompatEngine)
    eng.name, eng.model, eng.models, eng.tool, eng._cooldown = "groq", "test-model", ["test-model"], _compat_tool(), {}
    eng.client = NS(chat=NS(completions=FakeCompletions()))
    r = agent.start_call("inbound", "9000000101")
    d = eng.decide(agent._state(agent._call(r["call_id"])), "5000 ka DMT fail ho gaya")
    assert d["engine"] == "groq" and d["intent"] == "dmt_failed" and d["slots"]["amount"] == "5000"
    assert captured["messages"][0]["role"] == "system"
    assert captured["tool_choice"]["function"]["name"] == "agent_turn"
    assert captured["tools"][0]["function"]["parameters"]["properties"]["slots"]["properties"]


def test_provider_selection(monkeypatch):
    from app.llm_engine import provider
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    assert provider() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert provider() == "claude"
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert provider() == "groq"            # free provider preferred in auto mode
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    assert provider() == "claude"
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert provider() is None              # asked for Gemini but no Gemini key


def test_rate_limited_model_falls_through_to_next():
    import json
    import httpx
    from types import SimpleNamespace as NS
    from openai import RateLimitError
    from app import agent
    from app.llm_engine import OpenAICompatEngine, _compat_tool
    tried = []

    class FakeCompletions:
        def create(self, **kw):
            tried.append(kw["model"])
            if kw["model"] == "a":
                resp = httpx.Response(429, headers={"retry-after": "30"}, request=httpx.Request("POST", "http://x"))
                raise RateLimitError("limited", response=resp, body=None)
            args = {"reply": "Ji bataiye.", "understood": True, "intent": "unknown", "intent_confidence": 0.3,
                    "slots": {}, "sentiment": "neutral", "action": "continue"}
            return NS(choices=[NS(message=NS(tool_calls=[NS(function=NS(arguments=json.dumps(args)))]))], usage=None)

    eng = OpenAICompatEngine.__new__(OpenAICompatEngine)
    eng.name, eng.models, eng.model, eng.tool, eng._cooldown = "groq", ["a", "b"], "a", _compat_tool(), {}
    eng.client = NS(chat=NS(completions=FakeCompletions()))
    st = agent._state(agent._call(agent.start_call("inbound", "9000000101")["call_id"]))
    assert eng.decide(st, "hello")["reply"] == "Ji bataiye."
    assert eng.decide(st, "hello again")["reply"] == "Ji bataiye."
    assert tried == ["a", "b", "b"]   # "a" is skipped while it cools down


def test_caller_picks_english_and_gets_english_replies():
    from app import agent
    r = agent.start_call("inbound", "9000000101", lang="ask")
    assert "English" in r["reply"] and "हिंदी" in r["reply"] and r["call"]["lang"] == "ask"
    cid = r["call_id"]
    r = agent.handle_turn(cid, "English please", .9)
    assert r["call"]["lang"] == "en" and "How can I help" in r["reply"]
    r = agent.handle_turn(cid, "My DMT money transfer failed", .9)
    assert "transaction ID" in r["reply"] and not any("ऀ" <= ch <= "ॿ" for ch in r["reply"])


def test_caller_picks_hindi_gets_devanagari():
    from app import agent
    cid = agent.start_call("inbound", "9000000101", lang="ask")["call_id"]
    r = agent.handle_turn(cid, "हिंदी", .9)
    assert r["call"]["lang"] == "hi" and "नमस्ते" in r["reply"]
    r = agent.handle_turn(cid, "Mera DMT transfer fail ho gaya", .9)
    assert "transaction ID" in r["reply"] and "कृपया" in r["reply"]


def test_skipping_language_question_guesses_from_speech():
    from app import agent
    cid = agent.start_call("inbound", "9000000101", lang="ask")["call_id"]
    r = agent.handle_turn(cid, "My money transfer failed and the customer lost money", .9)
    assert r["call"]["lang"] == "en" and "transaction ID" in r["reply"]


def test_switch_language_mid_call():
    from app import agent
    cid = agent.start_call("inbound", "9000000101", lang="hi")["call_id"]
    r = agent.handle_turn(cid, "Mera DMT transfer fail ho gaya", .9)
    assert "कृपया" in r["reply"]
    r = agent.handle_turn(cid, "Can you speak in English", .9)
    assert r["call"]["lang"] == "en" and "English" in r["reply"]
    r = agent.handle_turn(cid, "hindi mein baat kijiye", .9)
    assert r["call"]["lang"] == "hi" and "हिंदी" in r["reply"]


def test_everything_requires_login():
    c = _client(signed_in=False)
    assert c.get("/api/tickets").status_code == 401
    assert c.post("/api/calls", json={"phone": "9000000101"}).status_code == 401
    r = c.get("/rm", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "login?next=rm"
    assert c.get("/login").status_code == 200 and c.get("/static/common.css").status_code == 200


def test_signup_login_logout_flow():
    from app import db
    c = _client()
    me = c.get("/api/auth/me").json()
    assert me["email"] == "rahul@eko.test" and me["rm_id"] == 1 and me["rm_name"] == "Rahul Verma"
    stored = db.q("SELECT password_hash FROM users", one=True)["password_hash"]
    assert stored.startswith("scrypt$") and "correct horse" not in stored
    assert c.post("/api/auth/logout").status_code == 200
    assert c.get("/api/auth/me").status_code == 401
    assert c.post("/api/auth/login", json={"email": "RAHUL@eko.test", "password": "wrong pass"}).status_code == 401
    assert c.post("/api/auth/login", json={"email": "Rahul@Eko.test", "password": "correct horse"}).status_code == 200
    assert c.get("/api/auth/me").json()["name"] == "Rahul Verma"


def test_signup_validation():
    c = _client(signed_in=False)
    ok = {"email": "a@eko.test", "password": "longenough", "name": "A"}
    assert c.post("/api/auth/signup", json={**ok, "password": "short"}).status_code == 400
    assert c.post("/api/auth/signup", json={**ok, "email": "not-an-email"}).status_code == 400
    assert c.post("/api/auth/signup", json={**ok, "rm_id": 2}).status_code == 200
    assert c.post("/api/auth/signup", json={**ok, "name": "B"}).status_code == 409             # email taken
    assert c.post("/api/auth/signup", json={**ok, "email": "b@eko.test", "rm_id": 2}).status_code == 409  # RM taken
    assert c.get("/api/auth/rms").status_code == 404        # the public list of RM names is gone


def test_login_lockout_after_repeated_failures():
    from app import auth
    c = _client()
    c.post("/api/auth/logout")
    for _ in range(auth.MAX_FAILS):
        assert c.post("/api/auth/login", json={"email": "rahul@eko.test", "password": "nope nope"}).status_code == 401
    r = c.post("/api/auth/login", json={"email": "rahul@eko.test", "password": "correct horse"})
    assert r.status_code == 429


def test_ticket_changes_record_who_made_them():
    c = _client()
    r = c.post("/api/calls", json={"direction": "inbound", "phone": "9000000101"}).json()
    r = c.post(f"/api/calls/{r['call_id']}/turn", json={"text": "Mujhe RM se baat karni hai", "stt_confidence": .9}).json()
    tid = r["ticket"]["id"]
    c.patch(f"/api/tickets/{tid}", json={"status": "in_progress"})
    events = c.get(f"/api/tickets/{tid}").json()["events"]
    assert any(e["actor"] == "Rahul Verma" and e["action"] == "status:in_progress" for e in events)


def _second_user(email="second@eko.test", rm_id=None):
    """A separate browser session for a second account (signs up as pending)."""
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    r = c.post("/api/auth/signup", json={"email": email, "password": "tea at four always", "name": "Second", "rm_id": rm_id})
    assert r.status_code == 200
    return c, r.json()["id"]


def _escalated_ticket(c, phone):
    r = c.post("/api/calls", json={"direction": "inbound", "phone": phone}).json()
    return c.post(f"/api/calls/{r['call_id']}/turn", json={"text": "Mujhe RM se baat karni hai", "stt_confidence": .9}).json()


def test_first_account_is_admin_and_new_signups_wait_for_approval():
    admin = _client()
    assert admin.get("/api/auth/me").json()["role"] == "admin"
    pending, _ = _second_user()
    assert pending.get("/api/auth/me").json()["role"] == "pending"
    assert pending.get("/api/tickets").status_code == 403
    r = pending.get("/rm", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "pending"
    assert pending.get("/pending").status_code == 200


def test_rm_only_sees_and_changes_their_own_tickets():
    admin = _client()                                  # admin is linked to RM 1 (Rahul)
    t_rm1 = _escalated_ticket(admin, "9000000101")["ticket"]   # Ramesh -> RM 1
    t_rm2 = _escalated_ticket(admin, "9000000201")["ticket"]   # Anil -> RM 2
    rm2, uid = _second_user(rm_id=2)
    assert admin.patch(f"/api/users/{uid}", json={"role": "rm", "rm_id": 2}).status_code == 200
    assert rm2.get("/api/auth/me").json()["role"] == "rm"

    seen = {t["id"] for t in rm2.get("/api/tickets?rm_id=1").json()}   # asking for RM 1's queue is ignored
    assert t_rm2["id"] in seen and t_rm1["id"] not in seen
    assert rm2.get(f"/api/tickets/{t_rm2['id']}").status_code == 200
    assert rm2.get(f"/api/tickets/{t_rm1['id']}").status_code == 404
    assert rm2.patch(f"/api/tickets/{t_rm1['id']}", json={"status": "closed"}).status_code == 404
    assert rm2.patch(f"/api/tickets/{t_rm2['id']}", json={"status": "in_progress"}).status_code == 200
    assert rm2.get(f"/api/calls/{t_rm1['call_id']}").status_code == 404
    assert rm2.get("/api/metrics/agent?rm_id=1").json() == admin.get("/api/metrics/agent?rm_id=2").json()

    # No call console, performance dashboard or user management for RMs
    assert rm2.post("/api/calls", json={"phone": "9000000201"}).status_code == 403
    assert rm2.get("/api/metrics/model").status_code == 403
    assert rm2.get("/api/users").status_code == 403
    r = rm2.get("/call", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "./?denied=1"
    assert rm2.get("/rm").status_code == 200


def test_ops_has_everything_except_user_management():
    admin = _client()
    ops, uid = _second_user()
    admin.patch(f"/api/users/{uid}", json={"role": "ops"})
    assert ops.post("/api/calls", json={"phone": "9000000101"}).status_code == 200
    assert ops.get("/api/metrics/model").status_code == 200
    assert ops.get("/api/tickets").status_code == 200
    assert ops.get("/api/users").status_code == 403
    assert ops.patch(f"/api/users/{uid}", json={"role": "admin"}).status_code == 403   # can't promote yourself
    assert ops.get("/users", follow_redirects=False).headers["location"] == "./?denied=1"


def test_admin_guards():
    admin = _client()
    me = admin.get("/api/auth/me").json()["id"]
    assert admin.patch(f"/api/users/{me}", json={"role": "ops"}).status_code == 409   # last admin / self-demotion
    assert admin.delete(f"/api/users/{me}").status_code == 409
    other, uid = _second_user()
    assert admin.patch(f"/api/users/{uid}", json={"role": "rm"}).status_code == 400   # RM needs a profile
    assert admin.patch(f"/api/users/{uid}", json={"role": "rm", "rm_id": 1}).status_code == 409  # taken by admin
    assert admin.patch(f"/api/users/{uid}", json={"role": "superuser"}).status_code == 400
    assert admin.delete(f"/api/users/{uid}").status_code == 200
    assert other.get("/api/auth/me").status_code == 401          # removed user is signed out


def test_idle_calls_are_closed_like_a_hangup():
    import time
    from app import agent, db
    idle = agent.start_call("inbound", "9000000101")["call_id"]
    agent.handle_turn(idle, "Mera DMT transfer fail ho gaya", .9)          # issue partly captured
    quiet = agent.start_call("inbound", "9000000102")["call_id"]             # only the greeting
    active = agent.start_call("inbound", "9000000103")["call_id"]
    sim = agent.start_call("inbound", "9000000104", ts=time.time() - 5 * 86400)["call_id"]  # simulated, past ts
    for cid in (idle, quiet):
        db.update("calls", cid, touched_at=time.time() - 11 * 60)

    assert sorted(agent.close_stale_calls()) == sorted([idle, quiet])
    c = agent.public_call(idle)
    assert c["status"] == "ended" and c["outcome"] == "dropped" and c["ticket_no"]   # RM still gets the issue
    assert agent.public_call(quiet)["outcome"] == "dropped" and not agent.public_call(quiet)["ticket_no"]
    assert agent.public_call(active)["status"] == "ongoing"
    assert agent.public_call(sim)["status"] == "ongoing"     # old timestamps alone don't count as idle
    assert c["duration_sec"] < 60                            # ended at its last activity, not at cleanup time


def test_every_category_has_spoken_names_in_both_languages():
    from app.i18n import SPOKEN
    from app.taxonomy import CATEGORIES
    assert set(SPOKEN) == set(CATEGORIES)
    assert all(set(v) == {"hi", "en"} for v in SPOKEN.values())


def test_scorecard_scoring_rules():
    from evals.run import cer, normalize, slot_match, wer
    assert normalize("Haan ji, <|hi|> ₹5,000!") == "haan ji 5 000"
    assert wer("mera paisa kat gaya", "mera paisa kat gaya") == 0
    assert wer("mera paisa kat gaya", "mera paise kat gaya") == 0.25
    assert cer("abc", "abd") == round(1 / 3, 3)
    assert slot_match("amount", "5000", "₹5,000") and slot_match("amount", "2500", "२५००")
    assert not slot_match("amount", "2500", "400")
    assert slot_match("txn_id", "TXN482913577", "txn 4829 13577") and not slot_match("txn_id", "TXN482913577", "XN 482913 577")
    assert slot_match("month", "last month", "Last month") and slot_match("bank_name", "SBI", "SBI bank")


def test_scorecard_runs_offline_on_rules_engine(tmp_path):
    import csv
    import os
    from app import db
    from evals import run
    cases = tmp_path / "cases.csv"
    with open(cases, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "lang", "audio", "reference_text", "expected_intent", "expected_action", "amount"])
        w.writerow(["dmt", "hi", "", "Mera DMT transfer fail ho gaya 5000 rupaye", "dmt_failed", "continue", "5000"])
        w.writerow(["rm", "hi", "", "Mujhe RM se baat karni hai", "", "escalated", ""])
    path = db.DB_PATH
    summary = run.main(["--cases", str(cases), "--engine", "rules", "--no-stt", "--no-tts", "--out", str(tmp_path / "out")])
    db.reset_connection(path)
    t = summary["understanding_from_text"]
    assert t["cases"] == 2 and t["intent_pct"] == 100.0 and t["action_pct"] == 100.0 and t["details_pct"] == 100.0
    out = next((tmp_path / "out").iterdir())
    assert {"report.md", "results.csv", "summary.json"} <= set(os.listdir(out))


def test_fraud_always_escalates_even_if_engine_wants_to_continue(monkeypatch):
    from app import agent

    class AsksMore:
        name = "groq"

        def decide(self, state, text):
            return {"engine": "groq", "reply": "Kab hua yeh?", "understood": True, "intent": "fraud_report",
                    "intent_confidence": 0.95, "slots": {}, "action": "continue", "sentiment": "neutral"}
    monkeypatch.setattr(agent, "_engine", lambda: AsksMore())
    cid = agent.start_call("inbound", "9000000101", lang="hi")["call_id"]
    r = agent.handle_turn(cid, "Kisi ne mere account se paise nikal liye, fraud hai", .9)
    assert r["ended"] and r["outcome"] == "escalated" and "OTP" in r["reply"]
    assert r["ticket"]["escalated"] and r["ticket"]["priority"] == "critical"


def test_schema_validation_error_falls_through_to_next_model():
    import json
    import httpx
    from types import SimpleNamespace as NS
    from openai import BadRequestError
    from app import agent
    from app.llm_engine import OpenAICompatEngine, _compat_tool
    tried = []

    class FakeCompletions:
        def create(self, **kw):
            tried.append(kw["model"])
            if kw["model"] == "a":
                resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
                raise BadRequestError("Tool call validation failed: parameters for tool agent_turn did not match schema",
                                      response=resp, body=None)
            args = {"reply": "Ji.", "understood": True, "intent": "commission", "intent_confidence": 0.9,
                    "slots": {}, "sentiment": "neutral", "action": "continue"}
            return NS(choices=[NS(message=NS(tool_calls=[NS(function=NS(arguments=json.dumps(args)))]))], usage=None)

    eng = OpenAICompatEngine.__new__(OpenAICompatEngine)
    eng.name, eng.models, eng.model, eng.tool, eng._cooldown = "groq", ["a", "b"], "a", _compat_tool(), {}
    eng.client = NS(chat=NS(completions=FakeCompletions()))
    st = agent._state(agent._call(agent.start_call("inbound", "9000000101")["call_id"]))
    assert eng.decide(st, "My commission is low")["intent"] == "commission" and tried == ["a", "b"]


def test_garbled_slot_keys_are_repaired():
    from app.llm_engine import _clean_slots
    assert _clean_slots({"payment_mode NEFT": "NEFT", "amount": 10000, "txn_id": "", "customer_phone": "98765"}) == \
        {"payment_mode": "NEFT", "amount": "10000", "customer_phone": "98765"}
    assert _clean_slots(None) == {} and _clean_slots("oops") == {}


def test_angry_caller_escalates_even_without_keywords(monkeypatch):
    from app import agent

    class HearsAnger:
        name = "groq"

        def decide(self, state, text):
            return {"engine": "groq", "reply": "Aapka gussa samajhti hoon, kaunsi service?", "understood": True,
                    "intent": "unknown", "intent_confidence": 0.3, "slots": {}, "action": "continue",
                    "sentiment": "angry"}
    monkeypatch.setattr(agent, "_engine", lambda: HearsAnger())
    cid = agent.start_call("inbound", "9000000101", lang="en")["call_id"]
    r = agent.handle_turn(cid, "Teen din se paisa atka hai, bakwa service hai", .9)   # misheard keyword
    assert r["ended"] and r["outcome"] == "escalated" and "sorry" in r["reply"]
    assert r["call"]["escalation_reason"] == "CSP sounded angry (detected by agent)"


def test_keyword_anger_gets_an_apology_not_of_course():
    from app import agent
    cid = agent.start_call("inbound", "9000000101", lang="hi")["call_id"]
    r = agent.handle_turn(cid, "Bakwas service hai, teen din se paisa atka hai", .9)
    assert r["outcome"] == "escalated" and "माफ़" in r["reply"]


def test_unsupported_audio_is_converted_for_whisper():
    import shutil
    import subprocess
    import pytest
    from app.stt import to_supported
    assert to_supported(b"abc", "clip.webm") == (b"abc", "clip.webm")        # supported: untouched
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    aac = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                          "-c:a", "aac", "-f", "adts", "pipe:1"], capture_output=True).stdout
    data, name = to_supported(aac, "WhatsApp Audio.aac")
    assert name == "WhatsApp Audio.flac" and data[:4] == b"fLaC"
    with pytest.raises(ValueError):
        to_supported(b"\xef\xbf\xbd\xef\xbf\xbd not audio", "broken.aac")   # damaged files fail clearly


# ---------------------------------------------------------------- call recordings pilot
def _admin_token(c, rm_id=1):
    r = c.post("/api/uploader/tokens", json={"rm_id": rm_id, "label": "Rahul's phone"})
    assert r.status_code == 200
    return r.json()["token"]


def _upload(c, token, data=b"fake m4a bytes", name="Call_9000000101.m4a", **form):
    fields = {"started_at": "1792560000000", "duration_sec": "95", "direction": "incoming",
              "other_number": "+91 90000 00101", "device_id": "dev-1", **form}
    return c.post("/api/uploader/recordings", files={"audio": (name, data)}, data=fields,
                  headers={"Authorization": f"Bearer {token}"} if token else {})


def test_uploader_needs_a_valid_token_not_a_login():
    from fastapi.testclient import TestClient
    from app.main import app
    admin = _client()                                  # first account = admin
    token = _admin_token(admin)
    phone = TestClient(app)                            # no session cookie, like the app on the RM's phone
    assert phone.get("/api/uploader/ping", headers={"Authorization": f"Bearer {token}"}).json()["rm_id"] == 1
    assert phone.get("/api/uploader/ping").status_code == 401
    assert _upload(phone, "ekoup_wrong").status_code == 401
    first = _upload(phone, token).json()
    assert first["duplicate"] is False
    assert _upload(phone, token).json() == {"id": first["id"], "duplicate": True}   # same file twice: stored once
    assert _upload(phone, token, name="notes.pdf").status_code == 400
    assert phone.get("/api/recordings").status_code == 401                          # results need a staff login
    rec = admin.get(f"/api/recordings/{first['id']}").json()
    assert rec["status"] == "new" and rec["csp_name"] and rec["direction"] == "incoming"
    tid = admin.get("/api/uploader/tokens").json()[0]["id"]
    assert admin.delete(f"/api/uploader/tokens/{tid}").status_code == 200
    assert _upload(phone, token, data=b"other").status_code == 401                  # revoked
    rm = _client(email="rm@eko.test", rm_id=2)                                      # pending, not admin
    assert rm.get("/api/uploader/tokens").status_code == 403


def test_masking_keeps_ids_but_cuts_personal_numbers():
    from app.recordings import mask
    assert mask("UTR 412398765432 aaya") == "UTR 412398765432 aaya"
    assert mask("transaction id 9876543210 hai") == "transaction id 9876543210 hai"
    assert mask("mera number 9876543210 hai") == "mera number [PHONE] hai"
    assert mask("aadhaar 1234 5678 9012") == "aadhaar [AADHAAR]"
    assert mask("आधार नंबर १२३४५६७८९०१२") == "आधार नंबर [AADHAAR]"
    assert mask("OTP 482913 aaya") == "OTP [OTP] aaya"
    assert mask("PAN ABCDE1234F") == "PAN [PAN]"
    assert mask("account number 50100234567890") == "account number [ACCOUNT]"
    assert mask("5000 rupaye kat gaye") == "5000 rupaye kat gaye"


def _fake_pipeline(monkeypatch, segs, review=None, ask=None):
    from app import llm_engine, recordings, stt
    monkeypatch.setattr(stt, "available", lambda: True)
    monkeypatch.setattr(llm_engine, "provider", lambda: "groq")
    monkeypatch.setattr(recordings, "_compact", lambda path: (open(path, "rb").read(), "call.ogg"))
    monkeypatch.setattr(stt, "transcribe_detailed", lambda audio, name, lang="hi", model=None, words=False, prompt=None: ([dict(s) for s in segs], []))
    monkeypatch.setattr(recordings, "_ask", ask or (lambda system, user, tool: (review, "test-model")))


SEGS = [{"start": 0.0, "end": 2.0, "text": "Haan Ramesh ji, bataiye"},
        {"start": 2.0, "end": 6.5, "text": "Sir mera DMT 5000 ka fail ho gaya, mera number 9876543210 hai"},
        {"start": 6.5, "end": 9.0, "text": "Theek hai, main check karke batata hoon"}]
REVIEW = {"speakers": ["rm", "csp", "rm"], "is_csp_call": True, "language": "hinglish", "issue_segment": 1,
          "intent": "dmt_failed", "intent_confidence": .9, "slots": {"amount": "5000", "bogus": "x"},
          "sentiment": "frustrated", "resolution": "follow_up_promised",
          "rm_handling": "RM noted the amount and promised to check.", "summary": "Failed DMT of 5000."}


def test_daily_processing_transcribes_masks_and_labels_speakers(monkeypatch):
    from app import recordings
    admin = _client()
    _upload(admin, _admin_token(admin))
    seen = {}
    _fake_pipeline(monkeypatch, SEGS, ask=lambda s, user, t: (seen.setdefault("user", user), (REVIEW, "m"))[1])
    assert recordings.process_pending() == {"processed": 1, "failed": 0, "left": 0, "stopped": None}
    assert "[PHONE]" in seen["user"] and "9876543210" not in seen["user"]          # masked before the LLM sees it
    rec = admin.get("/api/recordings/1").json()
    assert rec["status"] == "done" and rec["intent"] == "dmt_failed" and rec["is_csp_call"] == 1
    assert [s["speaker"] for s in rec["transcript"]] == ["rm", "csp", "rm"]
    assert "9876543210" not in json.dumps(rec["transcript"])                        # and before it's stored
    assert rec["slots"] == {"amount": "5000"}                                       # unknown keys dropped
    assert rec["timing"]["total_s"] >= 0 and rec["timing"]["transcribe_reused"] is False


def test_rate_limit_leaves_recordings_for_the_next_run(monkeypatch):
    from app import recordings
    admin = _client()
    tok = _admin_token(admin)
    _upload(admin, tok, data=b"one")
    _upload(admin, tok, data=b"two")

    def limited(*a):
        raise recordings.RateLimited("Groq rate limit", 30)
    _fake_pipeline(monkeypatch, SEGS, ask=limited)
    waits = []
    res = recordings.process_pending(sleep=waits.append)
    assert res["processed"] == 0 and res["left"] == 2 and res["stopped"] == "Groq rate limit"
    assert waits == [30, 30, 30]
    assert {r["status"] for r in recordings.list_recordings()} == {"new"}


def test_failing_recording_gives_up_after_three_runs(monkeypatch):
    from app import recordings
    admin = _client()
    _upload(admin, _admin_token(admin))

    def broken(*a):
        raise ValueError("bad audio")
    _fake_pipeline(monkeypatch, SEGS)
    monkeypatch.setattr(recordings, "_compact", broken)
    now = time.time()
    recordings.process_pending(now=now)
    assert recordings.process_pending(now=now + 60)["processed"] == 0          # waits before retrying
    assert recordings.get_recording(1)["attempts"] == 1
    recordings.process_pending(now=now + 11 * 60)                              # 10 min later: 2nd try
    recordings.process_pending(now=now + 40 * 60)                              # 20 min after that: 3rd
    r = recordings.get_recording(1)
    assert r["status"] == "failed" and r["attempts"] == 3 and "bad audio" in r["error"]


def test_daily_mode_runs_once_a_day_after_the_set_time():
    from datetime import datetime
    from app.recordings import due
    assert not due(datetime(2026, 10, 21, 20, 59), None, "21:00")
    assert due(datetime(2026, 10, 21, 21, 0), None, "21:00")
    assert not due(datetime(2026, 10, 21, 23, 0), "2026-10-21", "21:00")
    assert due(datetime(2026, 10, 22, 21, 30), "2026-10-21", "21:00")


def test_each_upload_wakes_processing_straight_away():
    from app import recordings
    assert recordings.DAILY_AT == ""                                           # default: no waiting for 9 pm
    admin = _client()
    recordings._arrived.clear()
    _upload(admin, _admin_token(admin))
    assert recordings._arrived.is_set()


def test_export_cuts_the_csp_complaint_into_a_draft_case(monkeypatch, tmp_path):
    import csv
    import shutil
    import subprocess
    import pytest
    from app import recordings
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    wav = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=10",
                          "-f", "wav", "pipe:1"], capture_output=True).stdout
    admin = _client()
    _upload(admin, _admin_token(admin), data=wav, name="call.wav")
    _fake_pipeline(monkeypatch, SEGS, REVIEW)
    recordings.process_pending()
    out = tmp_path / "candidates.csv"
    assert recordings.export_candidates(str(out), str(tmp_path)) == 1
    assert recordings.export_candidates(str(out), str(tmp_path)) == 0                # not added twice
    row = next(csv.DictReader(open(out, encoding="utf-8")))
    assert row["id"] == "rec1" and row["audio"] == "rec1.ogg" and row["expected_intent"] == "dmt_failed"
    assert row["reference_text"] == "" and "DMT 5000" in row["notes"]
    assert (tmp_path / "rec1.ogg").stat().st_size > 0


# ---------------------------------------------------------------- integrations: API keys and webhooks
def _service(key):
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app, headers={"Authorization": f"Bearer {key}"})


def test_service_api_key_acts_as_ops_without_a_login():
    admin = _client()
    key = admin.post("/api/integrations/keys", json={"name": "Eko CRM"}).json()["key"]
    crm = _service(key)
    r = crm.post("/api/calls", json={"direction": "inbound", "phone": "9000000101"}).json()
    tid = crm.post(f"/api/calls/{r['call_id']}/turn", json={"text": "Mujhe RM se baat karni hai"}).json()["ticket"]["id"]
    assert crm.patch(f"/api/tickets/{tid}", json={"status": "closed"}).status_code == 200
    events = crm.get(f"/api/tickets/{tid}").json()["events"]
    assert events[-1]["actor"] == "Eko CRM (API)"                      # audit trail names the system
    assert crm.get("/openapi.json").status_code == 200                 # machine-readable API description
    assert crm.get("/api/users").status_code == 403                    # never admin
    assert crm.post("/api/integrations/keys", json={"name": "x"}).status_code == 403
    assert _service("ekosk_wrong").get("/api/tickets").status_code == 401
    kid = admin.get("/api/integrations/keys").json()[0]["id"]
    admin.delete(f"/api/integrations/keys/{kid}")
    assert crm.get("/api/tickets").status_code == 401                  # revoked


def _sent():
    calls = []

    def send(url, body, headers):
        calls.append((url, json.loads(body), headers, body))
        return 200
    return calls, send


def test_webhook_gets_signed_ticket_and_call_events():
    from app import integrations
    admin = _client()
    hook = admin.post("/api/integrations/webhooks", json={"url": "https://crm.example/hook",
                                                          "events": ["ticket.created", "call.ended"]}).json()
    talk([("Mujhe RM se baat karni hai", .9)])
    calls, send = _sent()
    assert integrations.deliver_pending(send)["delivered"] == 2
    by_event = {c[1]["event"]: c for c in calls}
    url, payload, headers, body = by_event["ticket.created"]
    t = payload["data"]
    assert url == "https://crm.example/hook" and t["escalated"] is True and t["csp"]["code"] == "CSP10231"
    assert t["rm"] == {"id": 1, "name": "Rahul Verma"}
    assert headers["X-Sahayak-Signature"] == integrations.sign(hook["secret"], headers["X-Sahayak-Timestamp"], body)
    assert by_event["call.ended"][1]["data"]["transcript"]
    assert integrations.deliver_pending(send)["due"] == 0              # each event sent once


def test_ticket_updates_send_what_changed_and_who_changed_it():
    from app import agent, integrations
    _client().post("/api/integrations/webhooks", json={"url": "https://crm.example/hook", "events": ["ticket.updated"]})
    tid = talk([("Mujhe RM se baat karni hai", .9)])["ticket"]["id"]
    agent.update_ticket(tid, "Rahul Verma", status="in_progress", note="Called the bank")
    calls, send = _sent()
    integrations.deliver_pending(send)
    data = calls[0][1]["data"]
    assert data["changes"] == ["status", "note"] and data["changed_by"] == "Rahul Verma"


def test_failed_webhook_retries_with_backoff_then_gives_up():
    from app import integrations
    _client().post("/api/integrations/webhooks", json={"url": "https://down.example/hook"})
    talk([("Mujhe RM se baat karni hai", .9)])
    now = time.time()
    res = integrations.deliver_pending(lambda *a: 503, now=now)
    assert res["delivered"] == 0 and res["gave_up"] == 0
    assert integrations.deliver_pending(lambda *a: 200, now=now + 5)["due"] == 0        # waits before retrying
    t = now
    for wait in integrations.RETRY_AFTER:
        t += wait + 1
        integrations.deliver_pending(lambda *a: 503, now=t)
    assert {d["status"] for d in integrations.list_deliveries()} == {"failed"}


def test_simulated_calls_send_no_events():
    from app import integrations
    from app.simulator import run_backfill
    _client().post("/api/integrations/webhooks", json={"url": "https://crm.example/hook"})
    run_backfill(5, days=0)
    assert integrations.list_deliveries() == []


def test_recording_processed_event(monkeypatch):
    from app import integrations, recordings
    admin = _client()
    admin.post("/api/integrations/webhooks", json={"url": "https://data.example/hook", "events": ["recording.processed"]})
    _upload(admin, _admin_token(admin))
    _fake_pipeline(monkeypatch, SEGS, REVIEW)
    recordings.process_pending()
    calls, send = _sent()
    integrations.deliver_pending(send)
    data = calls[0][1]["data"]
    assert data["intent"] == "dmt_failed" and [s["speaker"] for s in data["transcript"]] == ["rm", "csp", "rm"]
    assert "file_path" not in data


def test_webhook_validation():
    admin = _client()
    assert admin.post("/api/integrations/webhooks", json={"url": "ftp://x"}).status_code == 400
    assert admin.post("/api/integrations/webhooks", json={"url": "https://x.example", "events": ["nope"]}).status_code == 400


def test_pages_work_under_an_address_prefix():
    """Sahayak runs at https://indev.eko.in/rm-voice-agent: every link, API call and redirect must be relative."""
    import glob
    import re
    from urllib.parse import urljoin
    for f in glob.glob("static/*.html") + ["static/common.js"]:
        text = open(f, encoding="utf-8").read()
        bad = re.findall(r"""["'`(]/(?:api|static|login|signup|forgot|reset|verify-email|account|rm|ops|call|users|pending|favicon)\b""", text)
        assert not bad, f"{f} has absolute paths that break under a prefix: {bad[:3]}"
    c = _client(signed_in=False)
    loc = c.get("/rm", follow_redirects=False).headers["location"]
    assert urljoin("https://indev.eko.in/rm-voice-agent/rm", loc) == "https://indev.eko.in/rm-voice-agent/login?next=rm"
    loc = c.get("/", follow_redirects=False).headers["location"]
    assert urljoin("https://indev.eko.in/rm-voice-agent/", loc).startswith("https://indev.eko.in/rm-voice-agent/login")


def test_recordings_page_manual_upload_audio_and_rm_scope(monkeypatch):
    from app import recordings
    admin = _client()
    files = [("audio", ("call one.m4a", b"first fake audio")), ("audio", ("call two.opus", b"second fake audio")),
             ("audio", ("notes.pdf", b"not audio"))]
    r = admin.post("/api/recordings/upload", files=files,
                   data={"rm_id": "2", "started_at": "1792560000000,,", "direction": "outgoing", "other_number": ""})
    res = r.json()["results"]
    assert [bool(x.get("error")) for x in res] == [False, False, True]
    assert admin.get("/recordings").status_code == 200
    rows = admin.get("/api/recordings").json()["recordings"]
    assert {x["source"] for x in rows} == {"manual"} and {x["rm_id"] for x in rows} == {2}
    first = next(x for x in rows if x["orig_name"] == "call one.m4a")
    assert first["started_at"] == 1792560000
    assert admin.get(f"/api/recordings/{first['id']}/audio").content == b"first fake audio"
    # An RM sees only their own recordings, and can't upload or reprocess
    rm1 = _client(email="rm3@eko.test", rm_id=3)
    uid = next(u["id"] for u in admin.get("/api/users").json() if u["email"] == "rm3@eko.test")
    admin.patch(f"/api/users/{uid}", json={"role": "rm", "rm_id": 3})
    assert rm1.get("/api/recordings").json()["recordings"] == []
    assert rm1.get(f"/api/recordings/{first['id']}").status_code == 404
    assert rm1.get(f"/api/recordings/{first['id']}/audio").status_code == 404
    assert rm1.post(f"/api/recordings/{first['id']}/reprocess").status_code == 403
    # Process, then process again; admin deletes
    _fake_pipeline(monkeypatch, SEGS, REVIEW)
    recordings.process_pending()
    assert admin.get(f"/api/recordings/{first['id']}").json()["status"] == "done"
    assert admin.post(f"/api/recordings/{first['id']}/reprocess").status_code == 200
    assert admin.get(f"/api/recordings/{first['id']}").json()["status"] == "new"
    assert admin.delete(f"/api/recordings/{first['id']}").status_code == 200
    assert admin.get(f"/api/recordings/{first['id']}").status_code == 404


def test_segments_are_split_into_sentences_before_speakers_are_labelled():
    from app.recordings import split_sentences
    segs = split_sentences([{"start": 0.0, "end": 12.0, "text": "बताइए क्या हुआ। सर मेरा DMT फेल हो गया।"},
                            {"start": 12.0, "end": 14.0, "text": "Rs. 2.5 lakh"}])
    assert [s["text"] for s in segs] == ["बताइए क्या हुआ।", "सर मेरा DMT फेल हो गया।", "Rs. 2.5 lakh"]
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == segs[1]["start"] and segs[1]["end"] == 12.0


def test_ids_read_out_in_pieces_are_joined_up():
    from app.llm_engine import _clean_slots, normalize_id
    assert normalize_id("TXN 482-913-577") == "TXN482913577"
    assert normalize_id("४१२ ३९८ ७६५ ४३२") == "412398765432"
    assert normalize_id("not known") == "not known"
    assert _clean_slots({"utr_or_ref": "4123 9876 5432", "amount": "5 000"}) == {"utr_or_ref": "412398765432", "amount": "5 000"}


def test_words_go_to_the_voice_speaking_and_lines_never_mix_two_voices():
    from app.diarize import assign_words, punctuate, voice_at
    turns = [{"start": 0.0, "end": 2.0, "voice": "A"}, {"start": 2.3, "end": 6.0, "voice": "B"},
             {"start": 6.2, "end": 7.0, "voice": "A"}]
    assert voice_at(turns, 1.0) == "A" and voice_at(turns, 2.2) == "B" and voice_at(turns, 9.0) == "A"
    words = [{"word": w, "start": s, "end": s + 0.4} for w, s in
             [("बताइए", 0.2), ("क्या", 0.7), ("हुआ", 1.2), ("सर", 2.4), ("DMT", 2.9), ("फेल", 3.4), ("हुआ", 3.9), ("अच्छा", 6.3)]]
    segs = [{"start": 0.0, "end": 4.4, "text": "बताइए क्या हुआ? सर DMT फेल हुआ।"}, {"start": 6.2, "end": 7.0, "text": "अच्छा।"}]
    lines = assign_words(punctuate(words, segs), turns)
    assert [(l["voice"], l["text"]) for l in lines] == [("A", "बताइए क्या हुआ?"), ("B", "सर DMT फेल हुआ।"), ("A", "अच्छा।")]


def test_one_voice_gets_one_role_on_every_line(monkeypatch):
    from app import recordings
    lines = [{"start": 0, "end": 1, "text": "Haan ji bataiye", "voice": "A"},
             {"start": 1, "end": 2, "text": "Sir DMT fail", "voice": "B"},
             {"start": 2, "end": 3, "text": "Theek hai", "voice": "A"}]
    out = {"speakers": ["rm", "csp", "csp"], "voice_roles": {"A": "rm", "B": "csp"}, "is_csp_call": True,
           "language": "hinglish", "issue_segment": 1, "intent": "dmt_failed", "intent_confidence": .9,
           "sentiment": "neutral", "resolution": "unclear", "rm_handling": "", "summary": ""}
    monkeypatch.setattr(recordings, "_ask", lambda *a: (dict(out), "m"))
    review, _ = recordings.review_call(lines)
    assert review["speakers"] == ["rm", "csp", "rm"]          # line 3 follows its voice, not the per-line guess
    assert "(voice A)" in recordings._review_input(lines)


def test_speaker_change_mid_sentence_moves_to_the_sentence_end():
    from app.diarize import assign_words
    # The voice model switches two words early: "super" | "site है।" belongs to the first speaker's sentence
    w = lambda text, s: {"word": text, "start": s, "end": s + 0.3}
    words = [w("लेकिन", 0.0), w("यह", 0.35), w("बहुत", 0.7), w("बड़ा", 1.05), w("super", 1.4), w("site", 1.75),
             w("है।", 2.1), w("आप", 3.0), w("लोग", 3.35), w("hardly", 3.7), w("एक", 4.05), w("page", 4.4)]
    turns = [{"start": 0.0, "end": 1.7, "voice": "A"}, {"start": 1.72, "end": 5.0, "voice": "B"}]
    lines = assign_words(words, turns)
    assert [(l["voice"], l["text"]) for l in lines] == [("A", "लेकिन यह बहुत बड़ा super site है।"), ("B", "आप लोग hardly एक page")]


def test_short_complete_replies_keep_their_own_speaker():
    from app.diarize import assign_words
    w = lambda text, s, d=0.3: {"word": text, "start": s, "end": s + d}
    words = [w("visible", 0.0), w("है?", 0.4), w("हाँ", 1.2), w("जी।", 1.5), w("ठीक", 2.5), w("है", 2.85), w("तो", 3.2)]
    turns = [{"start": 0.0, "end": 0.8, "voice": "A"}, {"start": 1.1, "end": 1.9, "voice": "B"},
             {"start": 2.4, "end": 3.6, "voice": "A"}]
    assert [l["voice"] for l in assign_words(words, turns)] == ["A", "B", "A"]


class _FakeSarvam:
    """Plays Sarvam's batch API: create job, upload URL, PUT, start, status (Running, then Completed), download."""
    RESULT = {"transcript": "...", "diarized_transcript": {"entries": [
        {"transcript": "हम लोग बात कर रहे हैं kiosk website के improvement की।", "start_time_seconds": 0.0,
         "end_time_seconds": 6.5, "speaker_id": "1"},
        {"transcript": "Yes.", "start_time_seconds": 6.8, "end_time_seconds": 7.2, "speaker_id": "0"},
        {"transcript": "तो docs folder में plan है।", "start_time_seconds": 7.5, "end_time_seconds": 10.0, "speaker_id": "1"}]}}

    def __init__(self):
        self.calls, self.polls = [], 0

    def __enter__(self): return self
    def __exit__(self, *a): pass

    def _r(self, body, code=200):
        import httpx
        return httpx.Response(code, json=body)

    def post(self, url, json=None):
        self.calls.append(("POST", url, json))
        if url.endswith("/job/v1"): return self._r({"job_id": "j1"})
        if url.endswith("/upload-files"): return self._r({"upload_urls": {json["files"][0]: {"file_url": "https://blob/up"}}})
        if url.endswith("/start"): return self._r({})
        if url.endswith("/download-files"): return self._r({"download_urls": {"0.json": {"file_url": "https://blob/down"}}})

    def put(self, url, content=None, headers=None, timeout=None):
        self.calls.append(("PUT", url, len(content)))
        return self._r({})

    def get(self, url):
        self.polls += 1
        state = "Running" if self.polls < 2 else "Completed"
        return self._r({"job_state": state, "job_details": [{"outputs": [{"file_name": "0.json"}]}]})


def test_sarvam_transcribes_and_splits_speakers(monkeypatch):
    import httpx
    from app import recordings, sarvam
    fake = _FakeSarvam()
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.setattr(sarvam, "_client", lambda: fake)
    monkeypatch.setattr(sarvam, "_as_mp3", lambda path: path)
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: httpx.Response(200, json=_FakeSarvam.RESULT))
    monkeypatch.setattr(recordings, "_ask", lambda *a: ({"speakers": [], "is_csp_call": False, "language": "hinglish",
        "issue_segment": -1, "intent": "unknown", "intent_confidence": 0, "sentiment": "neutral",
        "resolution": "no_issue", "rm_handling": "", "summary": "A meeting about the kiosk website."}, "m"))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(sarvam, "POLL_S", 0)
    admin = _client()
    admin.post("/api/recordings/upload", files=[("audio", ("meeting.m4a", b"fake audio"))], data={"rm_id": "0"})
    from app import llm_engine, stt
    monkeypatch.setattr(stt, "available", lambda: True)
    monkeypatch.setattr(llm_engine, "provider", lambda: "groq")
    assert recordings.process_pending()["processed"] == 1
    params = fake.calls[0][2]["job_parameters"]
    assert params["with_diarization"] is True and params["mode"] == "codemix" and params["model"] == "saaras:v4"
    rec = admin.get("/api/recordings/1").json()
    assert [(l["voice"], l["text"]) for l in rec["transcript"]] == [
        ("A", "हम लोग बात कर रहे हैं kiosk website के improvement की।"), ("B", "Yes."), ("A", "तो docs folder में plan है।")]
    assert rec["timing"]["stt"].startswith("Sarvam") and rec["rm_id"] is None
    # Process again: Sarvam's saved result is reused, nothing is sent twice
    n = len(fake.calls)
    admin.post("/api/recordings/1/reprocess")
    recordings.process_pending()
    assert len(fake.calls) == n and admin.get("/api/recordings/1").json()["timing"]["transcribe_reused"] is True


# ---------------------------------------------------------------- accounts: passwords, reset, email linking
@pytest.fixture
def mail(monkeypatch):
    """Email 'sent' by the app, captured instead of going out."""
    from app import mailer
    sent = []
    monkeypatch.setenv("SMTP_HOST", "smtp.test")
    monkeypatch.setenv("SMTP_FROM", "sahayak@eko.test")
    monkeypatch.setenv("APP_BASE_URL", "https://indev.eko.in/rm-voice-agent/")
    monkeypatch.setattr(mailer, "_deliver", lambda to, subject, text: sent.append((to, subject, text)))
    monkeypatch.setattr(mailer, "_spawn", lambda fn: fn())           # send straight away, no thread
    return sent


def _link_token(sent, page):
    import re
    text = sent[-1][2]
    m = re.search(rf"https://indev\.eko\.in/rm-voice-agent/{page}\?token=([\w-]+)", text)
    assert m, text
    return m.group(1)


def test_password_rules():
    from app import auth
    for bad in ("short", "aaaaaaaaaaaa", "password123", "rahul-secret-1"):
        with pytest.raises(auth.AuthError):
            auth.check_password(bad, "rahul@eko.test", "Rahul Verma")
    auth.check_password("tea at four, always", "rahul@eko.test", "Rahul Verma")


def test_only_work_emails_can_sign_up(monkeypatch):
    monkeypatch.setenv("ALLOWED_EMAIL_DOMAINS", "eko.co.in,ekosupport.in")
    c = _client(signed_in=False)
    body = {"password": "correct horse", "name": "A"}
    r = c.post("/api/auth/signup", json={**body, "email": "a@gmail.com"})
    assert r.status_code == 400 and "work email" in r.json()["detail"]
    assert c.post("/api/auth/signup", json={**body, "email": "a@ekosupport.in"}).status_code == 200


def test_forgot_password_emails_a_one_time_link_and_signs_out_everywhere(mail):
    from app import db
    c = _client()
    other_device = _client(signed_in=False)
    assert other_device.post("/api/auth/login", json={"email": "rahul@eko.test", "password": "correct horse"}).status_code == 200
    stranger = _client(signed_in=False)
    # Same reply whether or not the account exists
    assert stranger.post("/api/auth/forgot", json={"email": "nobody@eko.test"}).json() == {"ok": True}
    assert stranger.post("/api/auth/forgot", json={"email": "Rahul@eko.test"}).json() == {"ok": True}
    resets = [m for m in mail if m[1].startswith("Reset")]
    assert len(resets) == 1 and resets[0][0] == "rahul@eko.test"
    token = _link_token(mail, "reset")
    assert token not in str(db.q("SELECT * FROM auth_tokens"))          # only the hash is stored
    assert stranger.post("/api/auth/reset", json={"token": token, "password": "short"}).status_code == 400
    assert stranger.post("/api/auth/reset", json={"token": token, "password": "a brand new phrase"}).status_code == 200
    assert stranger.post("/api/auth/reset", json={"token": token, "password": "another new phrase"}).status_code == 400
    assert c.get("/api/auth/me").status_code == 401 and other_device.get("/api/auth/me").status_code == 401
    assert stranger.post("/api/auth/login", json={"email": "rahul@eko.test", "password": "correct horse"}).status_code == 401
    assert stranger.post("/api/auth/login", json={"email": "rahul@eko.test", "password": "a brand new phrase"}).status_code == 200
    assert stranger.get("/api/auth/me").json()["email_verified"]       # the link proved the inbox


def test_reset_links_expire_and_are_rate_limited(mail):
    from app import auth, db
    c = _client()
    for _ in range(5):
        c.post("/api/auth/forgot", json={"email": "rahul@eko.test"})
    assert sum(m[1].startswith("Reset") for m in mail) == auth.MAX_RESETS_PER_HOUR
    token = _link_token(mail, "reset")
    db.x("UPDATE auth_tokens SET expires_at=?", (time.time() - 1,))
    r = c.post("/api/auth/reset", json={"token": token, "password": "a brand new phrase"})
    assert r.status_code == 400 and "expired" in r.json()["detail"]


def test_without_email_admin_makes_a_reset_link():
    admin = _client()
    pending, uid = _second_user()
    r = admin.post(f"/api/users/{uid}/reset-link").json()
    assert r["path"].startswith("reset?token=")
    assert pending.post(f"/api/users/{uid}/reset-link").status_code == 403
    token = r["path"].split("=", 1)[1]
    assert pending.post("/api/auth/reset", json={"token": token, "password": "fresh start please"}).status_code == 200
    assert pending.post("/api/auth/login", json={"email": "second@eko.test", "password": "fresh start please"}).status_code == 200
    assert _client(signed_in=False).get("/api/auth/options").json()["email"] is False


def test_verified_email_links_the_rm_profile(mail):
    from app import db
    db.x("UPDATE rms SET email='priya.singh@eko.test' WHERE id=2")
    _client()                                                            # admin
    c, uid = _second_user(email="priya.singh@eko.test")
    assert c.get("/api/auth/me").json()["rm_id"] is None                # not until the email is proven
    r = c.post("/api/auth/verify", json={"token": _link_token(mail, "verify-email")}).json()
    assert r["email_verified"] and r["rm_id"] == 2 and r["rm_name"] == "Priya Singh" and r["role"] == "pending"
    assert db.q("SELECT event FROM auth_events WHERE user_id=? AND event='rm_linked'", (uid,), one=True)


def test_email_change_needs_password_and_the_new_inbox(mail):
    c = _client()
    assert c.post("/api/auth/email", json={"email": "new@eko.test", "password": "wrong one!"}).status_code == 403
    assert c.post("/api/auth/email", json={"email": "new@eko.test", "password": "correct horse"}).json()["sent"]
    assert c.get("/api/auth/me").json()["email"] == "rahul@eko.test"     # unchanged until the link is opened
    assert mail[-2][0] == "rahul@eko.test" and mail[-1][0] == "new@eko.test"   # old address is told
    c.post("/api/auth/verify", json={"token": _link_token(mail, "verify-email")})
    assert c.get("/api/auth/me").json()["email"] == "new@eko.test"


def test_change_password_keeps_this_session_only():
    c = _client()
    other = _client(signed_in=False)
    other.post("/api/auth/login", json={"email": "rahul@eko.test", "password": "correct horse"})
    assert c.post("/api/auth/password", json={"current": "nope nope", "new": "a whole new phrase"}).status_code == 403
    assert c.post("/api/auth/password", json={"current": "correct horse", "new": "a whole new phrase"}).status_code == 200
    assert c.get("/api/auth/me").status_code == 200 and other.get("/api/auth/me").status_code == 401
    events = [e["event"] for e in c.get("/api/auth/activity").json()["events"]]
    assert "password_changed" in events and "login" in events
    assert c.post("/api/auth/logout-all").status_code == 200 and c.get("/api/auth/me").status_code == 401


def test_idle_sessions_expire():
    from app import auth, db
    c = _client()
    db.x("UPDATE sessions SET last_seen_at=?", (time.time() - auth.SESSION_IDLE_HOURS * 3600 - 5,))
    assert c.get("/api/auth/me").status_code == 401


def test_pending_user_can_manage_their_own_account():
    _client()
    pending, _ = _second_user()
    assert pending.get("/account").status_code == 200
    assert pending.get("/api/auth/activity").status_code == 200
    assert pending.post("/api/auth/password", json={"current": "tea at four always", "new": "my own phrase now"}).status_code == 200


def test_security_headers():
    c = _client(signed_in=False)
    for r in (c.get("/login"), c.get("/rm", follow_redirects=False), c.get("/api/tickets")):
        assert r.headers["referrer-policy"] == "no-referrer" and r.headers["x-frame-options"] == "DENY"
    assert c.get("/api/tickets").headers["cache-control"] == "no-store"


# ---------------------------------------------------------------- knowledge base import (synthetic workbook only)
def _fake_calling_sheet(path, rows):
    """A tiny workbook shaped like Eko's calling sheet, filled with made-up CSPs."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Calling Sheet New"
    profile = ["CSP ID ", "CSP Name ", "CSP Mobile number", "CSP Category", "Relationship Manager ", "State",
               "District", "State ", "Circle (LHO)", "Email of RM", "Mobile No Of RM", "Terminal Status",
               "Total Transaction Count October'26"]
    groups = [("MTD Achievement October'26", ["PMJJBY", "PMSBY", "APY", "PMJDY", "Slab"]),
              ("Targets Oct 2026", ["PMJJBY\n Monthly", "PMSBY\n Monthly", "PMJDY\n Monthly", "APY\n Monthly",
                                    "DFS Slab\n Target"])]
    top, names = [None] * len(profile), list(profile)
    for g, subs in groups:
        top += [g] + [None] * (len(subs) - 1)
        names += subs
    ws.append(top)
    ws.append(names)
    for r in rows:
        ws.append(r)
    slab = wb.create_sheet("Slab")
    for r in [("Scheme", "Incentive for CSPs Per Month"), (None, "Slab-1", "Slab-2", "Slab-3", "Slab-4", "Super Slab"),
              ("PMJDY", 8, 10, 12, 14, 30), ("PMJJBY", 12, 14, 16, 18, 50), ("PMSBY", 20, 25, 30, 35, 100),
              ("APY", 3, 4, 5, 6, 15), ("Monthly extra incentive in Rs.", "1200/-", "1500/-", "1800/-", "2200/-", "5000/-")]:
        slab.append(r)
    wb.save(path)


def _row(code, name, phone, rm, mtd, slab, target="Slab-1 (₹1,200)"):
    return [code, name, phone, "Silver CSP", rm, "Circle Head", "Patna", "Bihar", "Patna", f"{rm.lower()}@eko.test",
            9810000099.0, "Active", 120, *mtd, slab, 12, 20, 8, 3, target]


def test_slab_rule_is_three_of_four_schemes():
    from app.kb_import import next_slab_gap, slab_for
    slabs = [{"slab": "Slab-1", "rank": 1, "pmjdy": 8, "pmjjby": 12, "pmsby": 20, "apy": 3, "incentive_rs": 1200},
             {"slab": "Slab-2", "rank": 2, "pmjdy": 10, "pmjjby": 14, "pmsby": 25, "apy": 4, "incentive_rs": 1500}]
    assert slab_for({"pmjdy": 0, "pmjjby": 12, "pmsby": 20, "apy": 3}, slabs) == "Slab-1"
    assert slab_for({"pmjdy": 8, "pmjjby": 12, "pmsby": 0, "apy": 0}, slabs) == "No Slab"
    assert slab_for({"pmjdy": 10, "pmjjby": 14, "pmsby": 25, "apy": 0}, slabs) == "Slab-2"
    gap = next_slab_gap({"pmjdy": 8, "pmjjby": 13, "pmsby": 20, "apy": 3}, slabs)
    assert gap["slab"] == "Slab-2" and gap["needed"] == {"pmjjby": 1, "apy": 1, "pmjdy": 2}


def test_import_calling_sheet_replaces_demo_data(tmp_path):
    from app import db, kb_import
    from app.seed import has_real_data
    path = str(tmp_path / "calling.xlsx")
    _fake_calling_sheet(path, [
        _row("1A859001", "Test Csp One", 9123400001.0, "Asha", (12, 20, 3, 1), "Slab-1"),
        _row("1A859002", "Test Csp Two", "91234 00002", "Asha", (0, 0, 0, 0), "No Slab", "—"),
        _row("1A859003", "Test Csp Three", 9123400003.0, "Bina", (18, 35, 6, 14), "Slab-4"),
        _row("1A859004", "Test Csp Four", 9123400003.0, "Bina", (1, 1, 1, 1), "No Slab"),   # shares a number
    ])
    before = open(path, "rb").read()
    _client()                                            # an admin linked to demo RM 1
    r = kb_import.import_file(path)
    assert open(path, "rb").read() == before             # the sheet is only read
    assert r["csps"] == 4 and r["rms"] == {"Asha": 2, "Bina": 2} and r["months"] == ["2026-10"]
    assert any("matches the 3-of-4 rule for all 4" in n for n in r["notes"])
    assert any("share a mobile number" in n for n in r["notes"])
    assert kb_import.import_file(path)["skipped"]        # same file twice changes nothing
    c = db.q("SELECT * FROM csps WHERE csp_code='1A859002'", one=True)
    assert c["phone"] == "9123400002" and c["source"] == "kb" and c["state"] == "Bihar"
    assert db.q("SELECT phone, shared_phone FROM csps WHERE csp_code='1A859004'", one=True) == \
        {"phone": None, "shared_phone": "9123400003"}
    m = db.q("SELECT * FROM csp_monthly m JOIN csps c ON c.id=m.csp_id WHERE c.csp_code='1A859003'", one=True)
    assert (m["month"], m["slab"], m["pmsby"], m["target_slab"], m["txn_count"]) == ("2026-10", "Slab-4", 35, "Slab-1", 120)
    assert db.q("SELECT email FROM rms WHERE name='Asha'", one=True)["email"] == "asha@eko.test"
    assert db.q("SELECT incentive_rs FROM slabs WHERE slab='Slab-4'", one=True)["incentive_rs"] == 2200

    out = kb_import.remove_demo()
    assert out["rms_deleted"] == 3 and out["csps_deleted"] == 15 and os.path.exists(out["demo_copy"])
    assert not db.q("SELECT 1 FROM csps WHERE source='demo'") and has_real_data()
    assert db.q("SELECT rm_id, role FROM users", one=True) == {"rm_id": None, "role": "admin"}
    import sqlite3
    demo = sqlite3.connect(out["demo_copy"])
    assert demo.execute("SELECT COUNT(*) FROM csps WHERE source='kb'").fetchone()[0] == 0
    assert demo.execute("SELECT COUNT(*) FROM csps").fetchone()[0] == 15


def test_simulator_refuses_to_run_next_to_real_csps(tmp_path):
    from app import db, simulator
    db.x("INSERT INTO csps(csp_code,name,source) VALUES('1A859999','Real Person','kb')")
    with pytest.raises(simulator.RealDataError):
        simulator.run_backfill(1)
    assert _client().post("/api/simulate", json={"calls": 1, "live": False}).status_code == 409


def test_server_start_adds_demo_data_only_when_asked(monkeypatch):
    from app import db, seed
    db.x("DELETE FROM csps"); db.x("DELETE FROM rms")
    seed.seed_if_demo()
    assert not db.q("SELECT 1 FROM rms")
    monkeypatch.setenv("DEMO_DATA", "1")
    seed.seed_if_demo()
    assert len(db.q("SELECT 1 FROM rms")) == 3


def test_detailed_summary_reads_the_whole_transcript_and_downloads(monkeypatch):
    from app import recordings, report
    admin = _client()
    _upload(admin, _admin_token(admin))
    long_segs = [{"start": float(i * 10), "end": float(i * 10 + 9), "text": f"विषय {i} के बारे में बात हुई, नंबर {1000 + i}।"}
                 for i in range(400)]                                   # long enough to need several parts
    _fake_pipeline(monkeypatch, long_segs, {**REVIEW, "speakers": [], "is_csp_call": False, "issue_segment": -1})
    seen = []

    def chat(system, user, max_tokens=3000):
        seen.append((system[:30], len(user)))
        if system.startswith("You write one part"):
            n = sum(1 for s, _ in seen if s.startswith("You write one part"))
            return (f"TOPICS\n### Topics part {n} [00:00–66:39]\n- [00:10] **Speaker 1** listed them\n\nEXTRACTS\n"
                    f"- Number: {1000 + n} is topic {n}", "m-part", False)
        assert "Number: 1001 is topic 1" in user                        # every part's extracts reach the final step
        return "## In short\nThey discussed 400 topics.\n\n## Decisions\n- Keep going.", "m-final", False
    monkeypatch.setattr(report, "_chat", chat)
    recordings.process_pending()
    rec = admin.get("/api/recordings/1").json()
    parts = sum(1 for s, _ in seen if s.startswith("You write one part"))
    assert parts >= 3 and all(n <= report.CHUNK_CHARS + 200 for s, n in seen if s.startswith("You write one part"))
    md = rec["report_md"]
    assert md.startswith("## In short") and rec["report_model"] == "m-final" and rec["report_s"] >= 0
    # every part's topic sections are in the summary, unchanged and in order, then the final sections
    assert all(f"### Topics part {i} [00:00–66:39]" in md for i in range(1, parts + 1))
    assert md.index("## Topic by topic") < md.index("### Topics part 1") < md.index("## Decisions")
    t = admin.get("/api/recordings/1/download/transcript.txt")
    assert t.status_code == 200 and "attachment" in t.headers["content-disposition"]
    assert "[00:00] Speaker" in t.text or "[00:00]" in t.text
    md = admin.get("/api/recordings/1/download/summary.md").text
    assert md.startswith("# Summary") and "They discussed 400 topics." in md
    doc = admin.get("/api/recordings/1/download/report.docx")
    assert doc.headers["content-type"].startswith("application/vnd.openxmlformats") and ".docx" in doc.headers["content-disposition"]
    import io
    import zipfile
    from xml.etree import ElementTree
    z = zipfile.ZipFile(io.BytesIO(doc.content))
    assert {"[Content_Types].xml", "_rels/.rels", "word/document.xml", "word/styles.xml"} <= set(z.namelist())
    body = ElementTree.fromstring(z.read("word/document.xml"))          # well-formed XML
    text = "".join(t.text or "" for t in body.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
    assert "Topics part 1 [00:00–66:39]" in text and "Transcript" in text and "विषय" in text
    assert admin.get("/api/recordings/1/download/report.doc").content[:2] == b"PK"   # old link, same .docx
    assert "[1:06:30] Speaker: विषय 399" in text
    assert admin.get("/api/recordings/1/download/secret.py").status_code == 404


def test_a_failed_summary_never_fails_the_recording(monkeypatch):
    from app import recordings, report
    admin = _client()
    _upload(admin, _admin_token(admin))
    _fake_pipeline(monkeypatch, SEGS, REVIEW)

    def broken(*a, **k):
        raise ValueError("model down")
    monkeypatch.setattr(report, "_chat", broken)
    recordings.process_pending()
    rec = admin.get("/api/recordings/1").json()
    assert rec["status"] == "done" and "model down" in rec["report_error"] and rec["report_md"] is None
    monkeypatch.setattr(report, "_chat", lambda *a, **k: ("TOPICS\n### Fixed [00:00]\nEXTRACTS\nNone" if a[0].startswith("You write")
                                                         else "## In short\nFixed.", "m", False))
    assert admin.post("/api/recordings/1/summarize").status_code == 200
    for _ in range(50):
        if admin.get("/api/recordings/1").json()["report_md"]:
            break
        time.sleep(0.05)
    assert admin.get("/api/recordings/1").json()["report_md"].startswith("## In short\nFixed.")


def test_a_leaked_whisper_hint_is_dropped_and_long_lines_are_broken():
    from app.diarize import assign_words
    from app.recordings import _drop_leaks
    hint = "हाँ जी, ठीक है। तो ये website का project है, उसमें docs folder है।"
    segs = [{"start": 0, "end": 3, "text": "हम kiosk website की बात कर रहे हैं।"},
            {"start": 3, "end": 6, "text": "तो ये website का project है, उसमें docs folder है।"},   # the hint, over silence
            {"start": 6, "end": 9, "text": "Testing बहुत ज़रूरी है।"}]
    words = [{"word": "हम", "start": 0.1, "end": 0.4}, {"word": "तो", "start": 3.2, "end": 3.5}, {"word": "Testing", "start": 6.1, "end": 6.6}]
    s2, w2 = _drop_leaks(segs, words, hint)
    assert [s["start"] for s in s2] == [0, 6] and [w["word"] for w in w2] == ["हम", "Testing"]
    many = [{"word": f"शब्द{i}", "start": i * 0.5, "end": i * 0.5 + 0.2} for i in range(200)]   # one voice, no punctuation
    lines = assign_words(many, [{"start": 0, "end": 200, "voice": "A"}])
    assert len(lines) > 1 and all(len(l["text"]) <= 700 for l in lines)


# ---------------------------------------------------------------- test CSPs, and no calls to real CSPs while testing
def _real_csp(code="CSP99999", phone="9123400777", rm_id=1, name="Real Person"):
    from app import db
    return db.x("INSERT INTO csps(csp_code,name,phone,rm_id,source) VALUES(?,?,?,?,'kb')", (code, name, phone, rm_id))


def test_synthetic_csps_cover_slab_cases_and_never_look_real():
    from app import db, synthetic
    _real_csp(name="Sunita Devi")                                        # a real name the generator must avoid
    r = synthetic.generate(per_rm=8)
    rows = db.q("SELECT * FROM csps WHERE source='synthetic'")
    assert r["written"] == 24 and len(rows) == 24                        # 3 demo RMs x 8
    assert all(c["csp_code"].startswith("TST") and synthetic.is_synthetic_phone(c["phone"]) for c in rows)
    assert "sunita devi" not in {c["name"].lower() for c in rows}
    assert len({c["phone"] for c in rows}) == 24
    slabs = {m["slab"] for m in db.q("SELECT slab FROM csp_monthly m JOIN csps c ON c.id=m.csp_id WHERE c.source='synthetic'")}
    assert {"No Slab", "Slab-1", "Slab-2", "Super Slab"} <= slabs
    synthetic.generate(per_rm=8)                                         # again: updated in place
    assert len(db.q("SELECT 1 FROM csps WHERE source='synthetic'")) == 24


def test_real_csps_cannot_be_called_while_testing(monkeypatch):
    from app import agent, db
    cid = _real_csp()
    for kw in ({"phone": "9123400777"}, {"csp_id": cid}):
        with pytest.raises(agent.RealCspBlocked):
            agent.start_call("outbound", **kw)
    assert not db.q("SELECT 1 FROM calls")
    c = _client()
    r = c.post("/api/calls", json={"direction": "outbound", "csp_id": cid})
    assert r.status_code == 403 and "Real CSPs" in r.json()["detail"]
    # Saying a real CSP's code during a test call doesn't attach it either
    call = agent.start_call("inbound", "5000000000")
    r = agent.handle_turn(call["call_id"], "Mera CSP code CSP 99999 hai", .9)
    assert db.q("SELECT csp_id FROM calls WHERE id=?", (call["call_id"],), one=True)["csp_id"] is None
    monkeypatch.setenv("REAL_CSP_CALLS", "1")                            # only once deployed for real
    assert agent.start_call("outbound", csp_id=cid)["call_id"]


def test_test_csp_calls_are_marked_as_tests():
    from app import agent, db, synthetic
    synthetic.generate(per_rm=2)
    csp = db.q("SELECT * FROM csps WHERE source='synthetic' LIMIT 1", one=True)
    r = agent.start_call("inbound", csp_id=csp["id"])
    call = db.q("SELECT * FROM calls WHERE id=?", (r["call_id"],), one=True)
    assert call["csp_id"] == csp["id"] and call["rm_id"] == csp["rm_id"] and call["simulated"] == 1


def test_console_lists_each_rms_csps_and_searches_by_code_or_name():
    from app import synthetic
    synthetic.generate(per_rm=3)
    _real_csp(code="1A859100", phone="9123400100", rm_id=2, name="Kamla Prasad")
    c = _client()
    d = c.get("/api/console/csps?rm_id=2").json()
    assert d["real_calls"] is False
    assert {x["rm_id"] for x in d["test"]} == {2} and [x["csp_code"] for x in d["real"]] == ["1A859100"]
    assert all(x["source"] in ("synthetic", "demo") for x in d["test"])
    assert "phone" not in d["real"][0]                                   # the console never needs a real number
    assert sorted(x["csp_code"] for x in c.get("/api/console/csps?q=tst02").json()["test"]) == ["TST02001", "TST02002", "TST02003"]
    assert c.get("/api/console/csps?q=kamla").json()["real"][0]["name"] == "Kamla Prasad"
    assert c.get("/api/console/csps?q=%25").json()["test"] == []         # % is searched for literally
    assert c.get("/api/console/csps?rm_id=1").json()["real"] == []

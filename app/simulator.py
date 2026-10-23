"""Synthetic CSP caller for demos and load testing.

The simulated CSP answers whatever the agent asks (it reads the agent's
question and replies with the matching fact), so it works with both the
rules engine and Claude. Each scenario carries its TRUE category, which the
simulated RM uses when confirming tickets - so the intent accuracy on the
dashboard is a real measurement of the engine against ground truth.

CLI:
  python -m app.simulator --backfill 120 --days 7     # history for charts
  python -m app.simulator --live 5                     # live calls you can watch
"""
import argparse
import random
import threading
import time
from datetime import datetime, timedelta

from . import agent, db
from .agent import IST
from .seed import has_real_data, seed

R = random.Random()


class RealDataError(RuntimeError):
    pass


def _demo_only():
    """Simulated calls and tickets must never be attached to real CSPs."""
    if has_real_data():
        raise RealDataError("This database holds real CSPs, so the simulator is off. "
                            "Run it on the demo database: EKO_DB_PATH=data/demo.db DEMO_DATA=1")
    seed()


def _txn():
    return f"TXN{R.randint(10**8, 10**9 - 1)}"


def _scenarios():
    return [
        # weight, true category, direction, opener, facts{slot: answer}, extra
        (14, "dmt_failed", "inbound", R.choice([
            "Namaste, mera ek DMT transfer fail ho gaya, customer ka paisa nahi pahuncha",
            "Customer ne beneficiary ko paisa bheja tha, abhi tak pending dikha raha hai",
            "Customer ka paisa kat gaya lekin uske bhai ke khaate mein nahi gaya"]),
         {"txn_id": f"Transaction ID hai {_txn()}", "amount": f"{R.choice([2000, 5000, 9500, 15000])} rupaye",
          "txn_date": R.choice(["kal hua tha", "aaj subah"])}, {}),
        (12, "aeps_issue", "inbound", R.choice([
            "AePS se customer ka paisa kat gaya lekin cash nahi nikla",
            "Aadhaar se nikasi ki, bank se paisa kat gaya par transaction fail dikha raha hai",
            "Customer ka khaata se paisa kat gaya, mujhe cash dena pada, ab kya karun"]),
         {"txn_id": f"ID {_txn()}", "amount": f"{R.choice([1000, 2000, 3000])} rupaye", "txn_date": "aaj",
          "bank_name": R.choice(["SBI", "PNB", "Bank of Baroda", "Gramin bank"])}, {}),
        (10, "wallet_topup", "inbound", "Maine wallet load kiya tha par balance abhi tak nahi aaya",
         {"amount": f"{R.choice([10000, 25000, 50000])} rupaye", "utr_or_ref": f"UTR {R.randint(10**11, 10**12-1)}",
          "txn_date": "aaj", "payment_mode": "UPI se bheja tha"}, {}),
        (8, "commission", "inbound", "Pichle mahine ka commission account mein nahi aaya",
         {"month": "September", "product": "DMT ka commission"}, {}),
        (6, "kyc_onboarding", "inbound", "Mera KYC pending dikha raha hai, ID activate nahi hui",
         {"issue_detail": "Documents paanch din pehle upload kiye the, verification abhi bhi pending hai"}, {}),
        (8, "app_device", "inbound", R.choice([
            "App mein login nahi ho raha, OTP bhi nahi aa raha",
            "Fingerprint scanner connect nahi ho raha, RD service ka error aa raha hai",
            "Mantra device se finger lagate hain to kuch nahi hota"]),
         {"device_or_screen": "app mein", "error_message": "error aa raha hai session expired"}, {}),
        (7, "bbps_recharge", "inbound", R.choice([
            "Bijli ka bill payment fail ho gaya", "Ek recharge kiya tha, pending mein atka hai"]),
         {"txn_id": f"Transaction {_txn()}", "amount": "1250 rupaye", "biller": "Bijli vibhag ka"}, {}),
        (5, "ticket_status", "inbound", "Mere pichle complaint ka kya hua, ticket status bataiye", {}, {}),
        (3, "fraud_report", "inbound", "Kisi ne mere account se fraud transaction kar diya hai", {}, {}),
        (4, "human", "inbound", "Mujhe seedha RM se baat karni hai", {}, {}),
        (3, "angry", "inbound", "Teen din se paisa atka hai, bakwas service hai aapki", {}, {}),
        (6, "garbled", "inbound", "haan... vo... hello", {}, {"garbled": True}),
        (4, "dropped", "inbound", "DMT transfer fail ho gaya hai", {}, {"hangup_after": 1}),
        (6, "wallet_topup", "outbound", "Haan ek problem hai, wallet mein paisa load nahi hua",
         {"amount": "20000 rupaye", "utr_or_ref": f"UTR {R.randint(10**11, 10**12-1)}", "txn_date": "kal",
          "payment_mode": "NEFT se"}, {"purpose": "follow_up"}),
        (4, "none", "outbound", "Nahi ji, sab theek chal raha hai, dhanyavaad", {}, {"purpose": "activation"}),
    ]


ASK_MAP = [  # phrases in the agent's question -> which fact to answer with
    (("utr", "reference"), "utr_or_ref"), (("transaction id", "txn id", "transaction number"), "txn_id"),
    (("kitne", "amount", "rupaye ka", "kitna"), "amount"), (("kis din", "date", "kab"), "txn_date"),
    (("bank",), "bank_name"), (("tarike", "upi", "neft", "mode"), "payment_mode"),
    (("mahine", "month"), "month"), (("service", "product"), "product"),
    (("error", "message"), "error_message"), (("app mein", "device", "app ya"), "device_or_screen"),
    (("biller", "operator"), "biller"), (("detail", "vistaar", "explain"), "issue_detail"),
]


def _answer(agent_text, facts, used):
    t = agent_text.lower()
    if any(w in t for w in ("confirm", "raise kar", "sahi hai", "theek hai na", "kar doon")):
        return "Haan ji, sahi hai, kar dijiye"
    for keys, slot in ASK_MAP:
        if slot in facts and slot not in used and any(k in t for k in keys):
            used.add(slot)
            return facts[slot]
    for slot, ans in facts.items():  # otherwise volunteer the next unknown fact
        if slot not in used:
            used.add(slot)
            return ans
    return "Haan ji, bas yahi problem hai"


def _pick():
    sc = _scenarios()
    return R.choices(sc, weights=[s[0] for s in sc])[0]


def play_call(ts=None, pace=None, engine="rules"):
    """Play one simulated call. ts=None -> real time. pace=(min,max) seconds between turns when live."""
    agent.force_engine("rules" if engine == "rules" else None)
    _, truth, direction, opener, facts, extra = _pick()
    csp = R.choice(db.q("SELECT * FROM csps WHERE source='demo'"))
    clock = [ts]

    def tick():
        if clock[0] is not None:
            clock[0] += R.uniform(9, 26)
            return clock[0]
        if pace:
            time.sleep(R.uniform(*pace))
        return None

    res = agent.start_call(direction, csp["phone"], extra.get("purpose", "support"), ts=clock[0])
    cid = res["call_id"]
    db.update("calls", cid, simulated=1)        # demo data: no webhook events
    used = set()
    utter, conf = opener, round(R.uniform(0.72, 0.97), 2)
    if extra.get("garbled"):
        conf = round(R.uniform(0.2, 0.4), 2)
    for i in range(14):
        res = agent.handle_turn(cid, utter, conf, ts=tick())
        if res["ended"]:
            break
        if extra.get("hangup_after") and i + 1 >= extra["hangup_after"]:
            agent.hangup(cid, ts=tick())
            break
        if extra.get("garbled"):
            utter, conf = R.choice(["...hm... ji", "kya? hello", "haan haan"]), round(R.uniform(0.15, 0.4), 2)
        else:
            utter, conf = _answer(res["reply"], facts, used), round(R.uniform(0.7, 0.97), 2)
    agent.force_engine(None)
    return cid, truth


def _rm_actions(cid, truth, created_ts, now):
    """Simulated RM work on the ticket: confirm the true category, move through statuses."""
    call = db.q("SELECT ticket_id FROM calls WHERE id=?", (cid,), one=True)
    if not call or not call["ticket_id"]:
        return
    tid = call["ticket_id"]
    from .taxonomy import CATEGORIES
    true_cat = truth if truth in CATEGORIES else None
    t = db.q("SELECT sla_due_at, created_at FROM tickets WHERE id=?", (tid,), one=True)
    sla_h = (t["sla_due_at"] - t["created_at"]) / 3600
    t1 = created_ts + R.uniform(0.05, 0.3) * sla_h * 3600
    if t1 > now:
        return
    agent.update_ticket(tid, "rm", status="in_progress", rm_category=true_cat, ts=t1)
    if R.random() < 0.8:
        t2 = created_ts + R.uniform(0.3, 1.25) * sla_h * 3600
        if t2 < now:
            agent.update_ticket(tid, "rm", status="closed", note="Resolved with ops team", ts=t2)


def run_backfill(n=120, days=7, engine="rules"):
    _demo_only()
    now = db.now()
    made = []
    for _ in range(n):
        d = R.randint(0, max(days - 1, 0))
        day = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=d)
        ts = (day + timedelta(hours=R.uniform(9, 20))).timestamp()
        if ts > now - 120:
            ts = now - R.uniform(300, 4 * 3600)
        cid, truth = play_call(ts=ts, engine=engine)
        _rm_actions(cid, truth, ts, now)
        made.append(cid)
    return {"created": len(made)}


def run_live(n=5, engine="rules"):
    _demo_only()
    threads = []
    for i in range(n):
        th = threading.Thread(target=play_call, kwargs={"pace": (2.5, 5.0), "engine": engine}, daemon=True)
        th.start()
        threads.append(th)
        time.sleep(R.uniform(1, 3))
    for th in threads:
        th.join()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=0)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--live", type=int, default=0)
    ap.add_argument("--engine", default="rules", choices=["rules", "auto"])
    a = ap.parse_args()
    if a.backfill:
        print(run_backfill(a.backfill, a.days, a.engine))
    if a.live:
        run_live(a.live, a.engine)
        print("live calls done")

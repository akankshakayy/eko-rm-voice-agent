"""Made-up test CSPs for each RM, for trying the agent in the call console.

Real CSPs (from Eko's sheets) are never called while Sahayak is being tested, so each RM gets a separate list of
synthetic CSPs to call instead. Nothing here is copied from a real CSP:
- codes start with TST (real ones look like 1A85xxxx)
- mobile numbers start with 5, which no Indian mobile number does, so they can never ring a real phone
- names are drawn from common first names and surnames, and never match a real CSP's full name
- the category mix and state come from counts over the real sheet, not from any one CSP
- this month's PMJDY/PMJJBY/PMSBY/APY counts are chosen to cover every slab case (no slab, one short of a slab,
  exactly on a slab, 3-of-4 schemes, Super Slab, no activity), so slab questions can be tested

Run again any time: the same RM gets the same CSPs (codes are fixed by RM and number), so it updates in place.
  python -m app.synthetic                 # 12 test CSPs per RM
  python -m app.synthetic --per-rm 20
"""
import argparse
import random
import time
from collections import Counter

from . import db

FIRST = ["Ramesh", "Sunita", "Arif", "Vikas", "Pankaj", "Anil", "Rekha", "Imran", "Deepak", "Neha", "Suresh",
         "Kailash", "Pooja", "Bhanwar", "Farida", "Manoj", "Geeta", "Rakesh", "Savita", "Dinesh", "Kavita", "Sanjay",
         "Anita", "Rajesh", "Meena", "Ajay", "Shabana", "Mukesh", "Lalita", "Naveen"]
LAST = ["Kumar", "Devi", "Ansari", "Gupta", "Mishra", "Chauhan", "Yadav", "Khan", "Tiwari", "Pal", "Meena", "Jat",
        "Sharma", "Lal", "Bano", "Singh", "Verma", "Prasad", "Paswan", "Thakur", "Saini", "Rawat", "Patel", "Joshi"]
SHOPS = ["Mobile Point", "Digital Seva", "Telecom", "General Store", "Cyber Cafe", "CSC Kendra", "Seva Kendra",
         "Recharge Centre", "Kirana & Mobile", "Communication"]
DISTRICTS = {"Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga", "Supaul"],
             "Uttar Pradesh": ["Lucknow", "Kanpur", "Barabanki", "Sitapur", "Unnao", "Gorakhpur"],
             "Haryana": ["Karnal", "Sirsa", "Hisar", "Rohtak"], "Punjab": ["Ludhiana", "Bathinda", "Patiala"],
             "Jharkhand": ["Giridih", "Deoghar", "Dhanbad"], "Himachal Pradesh": ["Kangra", "Mandi", "Shimla"],
             "Maharashtra": ["Nashik", "Nandurbar", "Pune"], "Nct Of Delhi": ["New Delhi"]}
# Share of each category over the real sheet (Basic 250, Silver 107, Bronze 103, Gold 59, Diamond 7 of 535)
CATEGORIES = [("Basic CSP", 47), ("Silver CSP", 20), ("Bronze CSP", 19), ("Gold CSP", 11), ("Diamond CSP", 3)]
DEFAULT_SLABS = [  # same shape as the "Slab" sheet; used when no sheet has been imported (tests, demo database)
    {"slab": "Slab-1", "rank": 1, "pmjdy": 8, "pmjjby": 12, "pmsby": 20, "apy": 3},
    {"slab": "Slab-2", "rank": 2, "pmjdy": 10, "pmjjby": 14, "pmsby": 25, "apy": 4},
    {"slab": "Slab-3", "rank": 3, "pmjdy": 12, "pmjjby": 16, "pmsby": 30, "apy": 5},
    {"slab": "Slab-4", "rank": 4, "pmjdy": 14, "pmjjby": 18, "pmsby": 35, "apy": 6},
    {"slab": "Super Slab", "rank": 5, "pmjdy": 30, "pmjjby": 50, "pmsby": 100, "apy": 15}]
SCHEMES = ("pmjdy", "pmjjby", "pmsby", "apy")


def is_synthetic_phone(phone: str | None) -> bool:
    return bool(phone) and len(phone) == 10 and phone[0] == "5"


def _slabs() -> list[dict]:
    rows = db.q("SELECT * FROM slabs ORDER BY rank")
    return rows or DEFAULT_SLABS


def _cases(slabs: list[dict]) -> list[tuple[str, dict]]:
    """Scheme counts for one month that hit each slab situation the agent may be asked about."""
    s = {x["slab"]: x for x in slabs}
    one, two, sup = s["Slab-1"], s["Slab-2"], s["Super Slab"]
    at = lambda slab, **less: {k: max(0, slab[k] - less.get(k, 0)) for k in SCHEMES}
    return [("no activity", dict.fromkeys(SCHEMES, 0)),
            ("one PMSBY short of Slab-1", at(one, pmsby=1, apy=one["apy"])),
            ("exactly Slab-1", at(one)),
            ("Slab-1 with only 3 of 4 schemes", at(one, pmjdy=one["pmjdy"])),
            ("two short of Slab-2", at(two, pmjjby=2, apy=two["apy"])),
            ("exactly Slab-2", at(two)),
            ("Super Slab", at(sup)),
            ("some activity, no slab", {"pmjdy": 3, "pmjjby": 4, "pmsby": 6, "apy": 1})]


def generate(per_rm: int = 12, seed: int = 7) -> dict:
    """Create or update `per_rm` test CSPs for every RM. Returns how many were written."""
    from .kb_import import slab_for
    rng = random.Random(seed)
    real = db.q("SELECT csp_code, phone, lower(name) name FROM csps WHERE source='kb'")
    real_names = {r["name"] for r in real}
    taken_phones = {r["phone"] for r in real if r["phone"]}
    slabs = _slabs()
    cases = _cases(slabs)
    month = time.strftime("%Y-%m")
    now = time.time()
    written = 0
    for rm in db.q("SELECT * FROM rms ORDER BY id"):
        states = Counter(r["state"] for r in db.q("SELECT state FROM csps WHERE rm_id=? AND source='kb' AND state "
                                                  "IS NOT NULL", (rm["id"],)))
        home = [s for s, _ in states.most_common(3) if s in DISTRICTS] or list(DISTRICTS)[:2]
        for i in range(1, per_rm + 1):
            code = f"TST{rm['id']:02d}{i:03d}"
            while True:
                name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
                if name.lower() not in real_names:
                    break
            phone = f"5{rm['id']:02d}{rng.randrange(10 ** 7):07d}"
            while phone in taken_phones:   # can't happen with a 5 in front, but never trust that
                phone = f"5{rm['id']:02d}{rng.randrange(10 ** 7):07d}"
            taken_phones.add(phone)
            state = rng.choice(home)
            label, counts = cases[(i - 1) % len(cases)]
            vals = dict(name=name, shop_name=f"{name.split()[1]} {rng.choice(SHOPS)}", phone=phone, rm_id=rm["id"],
                        district=rng.choice(DISTRICTS[state]), state=state,
                        category=rng.choices([c for c, _ in CATEGORIES], [w for _, w in CATEGORIES])[0],
                        terminal_status="Active" if label != "no activity" else "Inactive",
                        active=1, source="synthetic", updated_at=now)
            row = db.q("SELECT id, source FROM csps WHERE csp_code=?", (code,), one=True)
            if row and row["source"] != "synthetic":
                raise RuntimeError(f"{code} exists and is not synthetic; refusing to touch it")
            if row:
                db.update("csps", row["id"], **vals)
                csp_id = row["id"]
            else:
                db.x("UPDATE csps SET phone=NULL WHERE phone=? AND source='synthetic'", (phone,))
                csp_id = db.x(f"INSERT INTO csps(csp_code,{','.join(vals)}) VALUES(?{',?' * len(vals)})",
                              (code, *vals.values()))
            db.x("""INSERT INTO csp_monthly(csp_id,month,pmjdy,pmjjby,pmsby,apy,slab,target_slab,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(csp_id,month) DO UPDATE SET pmjdy=excluded.pmjdy,
                    pmjjby=excluded.pmjjby, pmsby=excluded.pmsby, apy=excluded.apy, slab=excluded.slab,
                    target_slab=excluded.target_slab, updated_at=excluded.updated_at""",
                 (csp_id, month, *(counts[k] for k in SCHEMES), slab_for(counts, slabs), "Slab-1", now))
            written += 1
    return {"written": written, "month": month}


def main():
    p = argparse.ArgumentParser(description="Make (or refresh) test CSPs for every RM")
    p.add_argument("--per-rm", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    print(generate(a.per_rm, a.seed))


if __name__ == "__main__":
    main()

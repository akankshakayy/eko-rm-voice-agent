"""Made-up test CSPs for each RM, for trying the agent in the call console.

Real CSPs (from Eko's sheets) are never called while Sahayak is being tested, so each RM gets a separate list of
synthetic CSPs to call instead. Nothing here is copied from a real CSP:
- codes start with TST (real ones look like 1A85xxxx)
- mobile numbers start with 5, which no Indian mobile number does, so they can never ring a real phone
- names are drawn from common first names and surnames, and never match a real CSP's full name
- figures follow the real data's overall shape only: `profile()` keeps per-category percentiles (5th to 95th, so
  no single CSP's extreme value is reproduced), the category mix and the inactive share, and nothing per CSP

Each RM gets two kinds of test CSP:
- edge cases (`EDGES`): every slab situation this month, a mobile number shared by two CSPs, no number, a CSP that
  left the sheet; these are the ones that catch mistakes
- realistic ones: a steady "activity level" per CSP, drawn through its category's percentiles month by month, so
  history, trends and slab changes look like the real sheet

Run again any time: codes are fixed by RM and number and the random seed is fixed, so it updates in place.
  python -m app.synthetic                 # 20 test CSPs per RM (the edge cases + realistic ones)
  python -m app.synthetic --per-rm 30
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
TIERS = ("Basic CSP", "Bronze CSP", "Silver CSP", "Gold CSP", "Diamond CSP")
SCHEMES = ("pmjdy", "pmjjby", "pmsby", "apy")
FIGURES = SCHEMES + ("txn_count", "commission")
DEFAULT_SLABS = [  # same shape as the "Slab" sheet; used when no sheet has been imported (tests, demo database)
    {"slab": "Slab-1", "rank": 1, "pmjdy": 8, "pmjjby": 12, "pmsby": 20, "apy": 3},
    {"slab": "Slab-2", "rank": 2, "pmjdy": 10, "pmjjby": 14, "pmsby": 25, "apy": 4},
    {"slab": "Slab-3", "rank": 3, "pmjdy": 12, "pmjjby": 16, "pmsby": 30, "apy": 5},
    {"slab": "Slab-4", "rank": 4, "pmjdy": 14, "pmjjby": 18, "pmsby": 35, "apy": 6},
    {"slab": "Super Slab", "rank": 5, "pmjdy": 30, "pmjjby": 50, "pmsby": 100, "apy": 15}]
PCTS = [5 + 5 * i for i in range(19)]   # 5th..95th percentile: the tails (single real CSPs) are left out
# Used when there is no real data (tests, demo database): rough shapes, nothing taken from Eko's sheets
_DEFAULT_SPREAD = {"pmjdy": (0, 12), "pmjjby": (0, 20), "pmsby": (0, 35), "apy": (0, 6), "txn_count": (0, 900),
                   "commission": (0, 40000)}
_TIER_SCALE = {"Basic CSP": .25, "Bronze CSP": .5, "Silver CSP": .75, "Gold CSP": 1.0, "Diamond CSP": 1.3}


def is_synthetic_phone(phone: str | None) -> bool:
    return bool(phone) and len(phone) == 10 and phone[0] == "5"


def _slabs() -> list[dict]:
    return db.q("SELECT * FROM slabs ORDER BY rank") or DEFAULT_SLABS


def _percentiles(values: list[float]) -> list[float]:
    v = sorted(values)
    return [v[round(p / 100 * (len(v) - 1))] for p in PCTS]


# ---------------------------------------------------------------- the real data's shape (summary numbers only)
def profile() -> dict:
    """Per-category percentiles of each monthly figure, the category mix, the inactive share and which months
    carry which figures. With no real data, a rough built-in shape."""
    real = db.q("SELECT id, category, terminal_status FROM csps WHERE source='kb' AND category IN "
                f"({','.join('?' * len(TIERS))})", TIERS)
    rows = db.q("SELECT m.*, c.category FROM csp_monthly m JOIN csps c ON c.id=m.csp_id WHERE c.source='kb'")
    if len(real) < 50 or not rows:
        now = time.localtime()
        months = [time.strftime("%Y-%m", time.localtime(time.mktime((now.tm_year, now.tm_mon - k, 1, 0, 0, 0, 0, 0, -1))))
                  for k in range(3, -1, -1)]
        return {"mix": {"Basic CSP": 47, "Bronze CSP": 19, "Silver CSP": 20, "Gold CSP": 11, "Diamond CSP": 3},
                "inactive": {t: .05 for t in TIERS}, "months": {f: months for f in FIGURES},
                "spread": {t: {f: [lo + (hi - lo) * _TIER_SCALE[t] * (p / 100) ** 2 for p in PCTS]
                               for f, (lo, hi) in _DEFAULT_SPREAD.items()} for t in TIERS}, "from_real": False}
    mix = Counter(r["category"] for r in real)
    inactive = {t: sum(r["terminal_status"] == "Inactive" for r in real if r["category"] == t) / max(1, mix[t])
                for t in TIERS}
    months = {f: sorted({r["month"] for r in rows if r[f] is not None}) for f in FIGURES}
    spread = {}
    for t in TIERS:
        mine = [r for r in rows if r["category"] == t] or rows      # a tier with no CSPs borrows the overall shape
        spread[t] = {f: _percentiles([r[f] for r in mine if r[f] is not None] or [0]) for f in FIGURES}
    return {"mix": dict(mix), "inactive": inactive, "months": months, "spread": spread, "from_real": True}


def _draw(pcts: list[float], u: float) -> float:
    """The value at position u (0..1) along the percentiles, interpolated."""
    x = min(max(u, 0), 1) * (len(pcts) - 1)
    i = int(x)
    return pcts[i] if i >= len(pcts) - 1 else pcts[i] + (pcts[i + 1] - pcts[i]) * (x - i)


# ---------------------------------------------------------------- edge cases
def _at(slab: dict, **less) -> dict:
    return {k: max(0, slab[k] - less.get(k, 0)) for k in SCHEMES}


def edges(slabs: list[dict]) -> list[dict]:
    """The situations that catch mistakes: every slab case this month, and awkward records."""
    s = {x["slab"]: x for x in slabs}
    one, two, sup = s["Slab-1"], s["Slab-2"], s["Super Slab"]
    return [{"case": "no activity this month", "now": dict.fromkeys(SCHEMES, 0), "terminal": "Inactive"},
            {"case": "one PMSBY short of Slab-1", "now": _at(one, pmsby=1, apy=one["apy"])},
            {"case": "exactly Slab-1", "now": _at(one)},
            {"case": "Slab-1 with only 3 of 4 schemes", "now": _at(one, pmjdy=one["pmjdy"])},
            {"case": "two PMJJBY short of Slab-2", "now": _at(two, pmjjby=2, apy=two["apy"])},
            {"case": "exactly Slab-2", "now": _at(two)},
            {"case": "Super Slab", "now": _at(sup)},
            {"case": "shares a mobile number with the next CSP", "phone": "shared"},
            {"case": "shares a mobile number with the CSP before", "phone": "shared"},
            {"case": "no mobile number", "phone": "none"},
            {"case": "left the sheet (inactive)", "active": 0}]


# ---------------------------------------------------------------- writing
def generate(per_rm: int = 20, seed: int = 7) -> dict:
    """Create or update `per_rm` test CSPs for every RM: the edge cases first, then realistic ones."""
    from .kb_import import slab_for
    rng = random.Random(seed)
    prof = profile()
    slabs = _slabs()
    cases = edges(slabs)
    real = db.q("SELECT phone, lower(name) name FROM csps WHERE source='kb'")
    real_names = {r["name"] for r in real}
    taken = {r["phone"] for r in real if r["phone"]}
    all_months = sorted({m for ms in prof["months"].values() for m in ms})
    current = all_months[-1] if all_months else time.strftime("%Y-%m")
    now = time.time()
    written, kinds = 0, Counter()
    tiers, weights = zip(*[(t, prof["mix"].get(t, 0) or 1) for t in TIERS])

    def phone_for(rm_id):
        while True:
            p = f"5{rm_id:02d}{rng.randrange(10 ** 7):07d}"
            if p not in taken:
                taken.add(p)
                return p

    for rm in db.q("SELECT * FROM rms ORDER BY id"):
        states = Counter(r["state"] for r in db.q("SELECT state FROM csps WHERE rm_id=? AND source='kb' AND state "
                                                  "IS NOT NULL", (rm["id"],)))
        home = [s for s, _ in states.most_common(3) if s in DISTRICTS] or list(DISTRICTS)[:2]
        shared = phone_for(rm["id"])
        for i in range(1, per_rm + 1):
            edge = cases[i - 1] if i <= len(cases) else None
            code = f"TST{rm['id']:02d}{i:03d}"
            while True:
                name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
                if name.lower() not in real_names:
                    break
            tier = rng.choices(tiers, weights)[0]
            level = rng.random()                            # this CSP's steady place within its category
            inactive = (edge or {}).get("terminal") == "Inactive" or \
                (not edge and rng.random() < prof["inactive"].get(tier, 0))
            kind = (edge or {}).get("phone")
            vals = dict(name=name, shop_name=f"{name.split()[1]} {rng.choice(SHOPS)}",
                        phone=None if kind else phone_for(rm["id"]), shared_phone=shared if kind == "shared" else None,
                        rm_id=rm["id"], state=(st := rng.choice(home)), district=rng.choice(DISTRICTS[st]),
                        category=tier, terminal_status="Inactive" if inactive else "Active",
                        active=(edge or {}).get("active", 1), source="synthetic", updated_at=now)
            row = db.q("SELECT id, source FROM csps WHERE csp_code=?", (code,), one=True)
            if row and row["source"] != "synthetic":
                raise RuntimeError(f"{code} exists and is not synthetic; refusing to touch it")
            if vals["phone"]:
                db.x("UPDATE csps SET phone=NULL WHERE phone=? AND source='synthetic' AND csp_code<>?",
                     (vals["phone"], code))
            if row:
                db.update("csps", row["id"], **vals)
                csp_id = row["id"]
            else:
                csp_id = db.x(f"INSERT INTO csps(csp_code,{','.join(vals)}) VALUES(?{',?' * len(vals)})",
                              (code, *vals.values()))
            for month in all_months:
                u = level + rng.gauss(0, .08)                # month-to-month wobble around the CSP's level
                fig = {f: (round(_draw(prof["spread"][tier][f], u + rng.gauss(0, .04)), 2 if f == "commission" else 0)
                           if month in prof["months"].get(f, []) else None) for f in FIGURES}
                if edge and month == current and "now" in edge:
                    fig.update(edge["now"])
                elif inactive:                               # nothing sold, in the months that carry the figure
                    fig.update({k: 0 for k in (*SCHEMES, "txn_count") if fig[k] is not None})
                has_schemes = all(fig[k] is not None for k in SCHEMES)
                slab = slab_for(fig, slabs) if has_schemes else None
                db.x("""INSERT INTO csp_monthly(csp_id,month,pmjdy,pmjjby,pmsby,apy,slab,txn_count,commission,updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(csp_id,month) DO UPDATE SET pmjdy=excluded.pmjdy,
                        pmjjby=excluded.pmjjby, pmsby=excluded.pmsby, apy=excluded.apy, slab=excluded.slab,
                        txn_count=excluded.txn_count, commission=excluded.commission, updated_at=excluded.updated_at""",
                     (csp_id, month, *(fig[k] for k in SCHEMES), slab, fig["txn_count"], fig["commission"], now))
            written += 1
            kinds["edge case" if edge else "realistic"] += 1
    return {"written": written, **kinds, "months": all_months, "shaped_on_real_data": prof["from_real"]}


def main():
    p = argparse.ArgumentParser(description="Make (or refresh) test CSPs for every RM")
    p.add_argument("--per-rm", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    print(generate(a.per_rm, a.seed))


if __name__ == "__main__":
    main()

"""Demo RMs and CSPs. Replace with a sync from Eko's CSP master data."""
from . import db

RMS = [
    (1, "Rahul Verma", "9810000001", "Bihar - Patna"),
    (2, "Priya Singh", "9810000002", "Uttar Pradesh - Lucknow"),
    (3, "Amit Yadav", "9810000003", "Rajasthan - Jaipur"),
]

CSPS = [
    # code, owner, shop, phone, district, rm_id
    ("CSP10231", "Ramesh Kumar", "Kumar Mobile Point", "9000000101", "Patna", 1),
    ("CSP10232", "Sunita Devi", "Sunita Digital Seva", "9000000102", "Gaya", 1),
    ("CSP10233", "Mohd. Arif", "Arif Telecom", "9000000103", "Muzaffarpur", 1),
    ("CSP10234", "Vikas Gupta", "Gupta General Store", "9000000104", "Bhagalpur", 1),
    ("CSP10235", "Pankaj Mishra", "Mishra Cyber Cafe", "9000000105", "Darbhanga", 1),
    ("CSP20311", "Anil Chauhan", "Chauhan Kirana", "9000000201", "Lucknow", 2),
    ("CSP20312", "Rekha Yadav", "Rekha CSC Kendra", "9000000202", "Kanpur", 2),
    ("CSP20313", "Imran Khan", "Khan Mobile Shop", "9000000203", "Barabanki", 2),
    ("CSP20314", "Deepak Tiwari", "Tiwari Seva Kendra", "9000000204", "Sitapur", 2),
    ("CSP20315", "Neha Pal", "Pal Digital Point", "9000000205", "Unnao", 2),
    ("CSP30411", "Suresh Meena", "Meena Recharge Centre", "9000000301", "Jaipur", 3),
    ("CSP30412", "Kailash Jat", "Jat Kirana & Mobile", "9000000302", "Ajmer", 3),
    ("CSP30413", "Pooja Sharma", "Sharma Digital Seva", "9000000303", "Tonk", 3),
    ("CSP30414", "Bhanwar Lal", "Lal Telecom", "9000000304", "Sikar", 3),
    ("CSP30415", "Farida Bano", "Bano Communication", "9000000305", "Alwar", 3),
]


def seed():
    if db.q("SELECT COUNT(*) n FROM rms", one=True)["n"]:
        return
    for r in RMS:
        db.x("INSERT INTO rms(id,name,phone,region) VALUES(?,?,?,?)", r)
    for c in CSPS:
        db.x("INSERT INTO csps(csp_code,name,shop_name,phone,district,rm_id) VALUES(?,?,?,?,?,?)", c)


def csp_by_phone(phone: str):
    phone = (phone or "")[-10:]
    return db.q("SELECT * FROM csps WHERE phone=?", (phone,), one=True)

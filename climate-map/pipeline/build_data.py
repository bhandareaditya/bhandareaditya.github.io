"""Build the climate map dataset: merge sources, filter, classify, embed, project, cluster."""
import csv, glob, json, re, sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import umap
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import geo

ROOT = sys.argv[1]
OUT = sys.argv[2]

SECTORS = {
    "solar": ("Solar, Wind & Storage", "#f5b82e"),
    "h2": ("Green Hydrogen", "#5ad1e6"),
    "bio": ("Bioenergy & Biofuels", "#9bd16a"),
    "evoem": ("EV Makers", "#4c8dff"),
    "evinfra": ("EV Charging, Swapping & Fleets", "#7b6cff"),
    "battery": ("Batteries & EV Components", "#b07cff"),
    "recycle": ("Battery & E-waste Recycling", "#e57cd8"),
    "waste": ("Waste & Circular Economy", "#ff7a7a"),
    "water": ("Water", "#2fb3ff"),
    "air": ("Air Quality", "#a9c3d9"),
    "carbon": ("Carbon, ESG & Climate Data", "#2ed19a"),
    "build": ("Green Buildings & Efficiency", "#ffa857"),
    "materials": ("Sustainable Materials & Packaging", "#d9a066"),
    "food": ("Alt Protein & Food", "#ff9ec4"),
    "agri": ("Climate-smart Agriculture", "#6fcf4f"),
    "finance": ("Climate Finance & Ecosystem", "#f2e86d"),
}

# keyword rules: (sector, regex, weight)
RULES = [
    ("h2", r"hydrogen|electroly[sz]er|fuel cell|green ammonia", 3),
    ("bio", r"biogas|bio-?cng|\bcbg\b|biofuel|bio-?diesel|ethanol|biomass|biochar|bio-?coal|pellet|briquet", 3),
    ("recycle", r"e-?waste|battery recycl|recycl\w* (of )?(lithium|batter)|black mass|urban mining", 4),
    ("waste", r"waste|recycl|circular|upcycl|compost|scrap|\bepr\b|landfill|segregat", 2),
    ("evinfra", r"charging|charger|swap|charge point|\bcpo\b|ocpp|fleet|ev rental|e-?rickshaw driver|battery[- ]as[- ]a[- ]service|baas|ride[- ]?hail|leasing", 2),
    ("evoem", r"electric (scooter|motorcycle|bike|two-?wheeler|three-?wheeler|truck|bus|vehicle manufactur)|e-?scooter|e-?bike|e-?rickshaw|\boem\b|electric cargo|\b2w\b|\b3w\b", 2),
    ("battery", r"battery pack|lithium|cell manufactur|\bbms\b|battery management|motor controller|powertrain|traction motor|li-?ion|sodium-?ion", 2),
    ("solar", r"solar|photovoltaic|\bpv\b|wind (energy|power|farm)|renewable|\bbess\b|energy storage|rooftop|open access|power producer|\bepc\b", 2),
    ("water", r"water|wastewater|sewage|\bstp\b|\betp\b|desalinat|irrigation water|rainwater|aquifer|groundwater|effluent", 2),
    ("air", r"air (quality|pollution|purif)|particulate|pm2\.?5|emission control|purifier|clean air|indoor air", 3),
    ("carbon", r"carbon (credit|accounting|footprint|market|offset|removal|capture|neutral)|\besg\b|net[- ]zero|decarboni[sz]ation software|\bbrsr\b|\bcbam\b|scope [123]|ghg|climate (risk|data|intelligence)|satellite|earth observation|\bmrv\b|sustainability reporting", 3),
    ("build", r"energy efficien|hvac|building automation|green building|smart building|heat pump|thermal (energy )?storage|waste heat|insulation|cooling as a service|energy audit|industrial decarboni[sz]|low[- ]carbon (cement|concrete)", 3),
    ("materials", r"packaging|biodegradable|compostable|bioplastic|plastic[- ]free|bamboo|mycelium|seaweed|natural fib|textile|fashion|apparel|eco-?friendly products|sustainable materials|jute|handloom", 2),
    ("food", r"plant[- ]based|alt(ernative)? protein|precision fermentation|meat alternative|dairy alternative|vegan|insect|millet|cultivated meat|food waste|mock meat", 3),
    ("agri", r"farm|agri|crop|soil|kisan|irrigation|horticulture|dairy|cold (chain|storage|room)|organic|fpo|harvest|seed|livestock|aquaculture|fisher", 2),
    ("finance", r"financ|loan|lend|nbfc|invest|fintech|insurance|credit|crowdfund|marketplace for investors|green bond", 1),
]
CLIMATE_GATE = re.compile(
    r"solar|renewable|clean energy|green|climate|carbon|emission|sustainab|electric vehicle|\bev\b|e-?mobility|battery|charging|recycl|waste|circular|water|air (quality|pollution)|hydrogen|biogas|biofuel|organic|regenerative|eco-?friendly|biodegradable|net[- ]zero|esg|energy efficien|compost|plant[- ]based|agri|farm|upcycl|decarboni",
    re.I,
)
NOISE = re.compile(
    r"media platform|news|magazine|journalism|\bexpo\b|conference|summit|exhibition|job portal|recruitment agency|training institute|coaching|youtube channel|podcast|blog\b|community for|association of|federation|chamber of",
    re.I,
)
BIG = {"5001-10000", "10001+"}


def norm_domain(d):
    d = (d or "").lower().strip()
    d = re.sub(r"^https?://", "", d)
    d = re.sub(r"^www\.", "", d)
    return d.split("/")[0]


def norm_name(n):
    n = (n or "").lower()
    n = re.sub(r"\(.*?\)", "", n)
    n = re.sub(r"\b(private|pvt|ltd|limited|llp|inc|technologies|technology|solutions|india|pvt\.|ltd\.)\b", "", n)
    return re.sub(r"[^a-z0-9]", "", n)


def clean_text(t):
    t = (t or "").replace("’", "'")
    t = re.sub(r"https?://\S+|\S+@\S+|#\w+", " ", t)
    t = re.sub(r"[\U00010000-\U0010ffff☀-➿←-⇿✀-➿⬀-⯿️]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # add spaces after sentence ends glued together ("solar.We")
    t = re.sub(r"([a-z0-9\)])([.!?])([A-Z])", r"\1\2 \3", t)
    return t


def short_desc(t, n=320):
    if len(t) <= n:
        return t
    cut = t[:n]
    p = max(cut.rfind(". "), cut.rfind("; "))
    if p > 140:
        return cut[: p + 1]
    return cut.rsplit(" ", 1)[0] + "…"


def classify(text, name=""):
    s = (name + " " + text).lower()
    scores = Counter()
    for sec, pat, w in RULES:
        hits = len(re.findall(pat, s))
        if hits:
            scores[sec] += w * min(hits, 4)
    if not scores:
        return None
    return scores.most_common(1)[0][0]


def city_of(loc):
    city = (loc.get("city") or "").strip()
    state = (loc.get("state") or "").strip()
    state = state.replace("ā", "a").replace("Ā", "A").replace("Mahārāshtra", "Maharashtra")
    state = re.sub(r"[āĀ]", "a", state)
    known = ["Bengaluru", "Bangalore", "Mumbai", "Navi Mumbai", "Pune", "Delhi", "New Delhi", "Gurugram", "Gurgaon", "Noida",
             "Greater Noida", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Indore", "Lucknow", "Chandigarh",
             "Kochi", "Coimbatore", "Vadodara", "Surat", "Bhubaneswar", "Nagpur", "Faridabad", "Ghaziabad", "Mohali"]
    for k in known:
        if k.lower() in city.lower():
            return {"Bangalore": "Bengaluru", "Gurgaon": "Gurugram", "New Delhi": "Delhi"}.get(k, k)
    if city and "," not in city and len(city) < 22 and city.lower() != "india":
        return city
    return state or "India"


HEAD_MULT = {"myself only": 0.6, "2-10": 0.8, "11-50": 1.0, "51-200": 1.35, "201-500": 1.7, "501-1000": 2.0,
             "1001-5000": 2.3, "5001-10000": 2.6, "10001+": 2.8}

# ---------- load crustdata ----------
rows = {}
files = sorted(glob.glob(f"{ROOT}/raw/*.json")) + sorted(glob.glob(f"{ROOT}/raw2/*.json"))
for f in files:
    fname = f.split("/")[-1]
    funded_file = fname.startswith("f")
    for c in json.load(open(f))["companies"]:
        b = c.get("basic_info", {})
        fu = c.get("funding") or {}
        cid = c["crustdata_company_id"]
        rec = rows.get(cid)
        if rec is None:
            desc = clean_text(b.get("description"))
            rec = rows[cid] = {
                "name": clean_text(b.get("name")), "domain": norm_domain(b.get("primary_domain")), "desc_full": desc,
                "year": b.get("year_founded"), "headcount": b.get("employee_count_range") or "",
                "linkedin": b.get("professional_network_url") or "", "ctype": b.get("company_type") or "",
                "city": city_of(c.get("locations", {})), "logo": b.get("logo_permalink") or "",
                "city_raw": (c.get("locations") or {}).get("city") or "", "state_raw": (c.get("locations") or {}).get("state") or "",
                "src": "crustdata", "funded": False, "round": "", "investors": "", "total": None,
            }
        if not rec["logo"] and b.get("logo_permalink"):
            rec["logo"] = b["logo_permalink"]
        if not rec["ctype"] and b.get("company_type"):
            rec["ctype"] = b["company_type"]
        if funded_file:
            rec["funded"] = True
        if fu.get("last_round_type"):
            rec["funded"] = True; rec["round"] = fu["last_round_type"]
            rec["investors"] = ", ".join((fu.get("investors") or [])[:6]); rec["total"] = fu.get("total_investment_usd")
print("crustdata unique", len(rows), "funded", sum(r["funded"] for r in rows.values()))

BACKED = re.compile(r"incubat|accelerat|\bdpiit\b|startup india|\bgrant\b|award|winner|backed by|\braised\b|funded by|seed fund|cohort|fellowship|y ?combinator|\byc [sw]\d|nasscom|birac|\bnidhi\b|atal incubation|t-hub|ksum|kitven|nsrcel|ciie|villgro|social alpha|sine iit|iit madras|iit bombay|iit delhi|iit kanpur|iisc|startup tn|elevate|shell e4|investors?\b", re.I)
SERVICE = re.compile(r"\bepc\b|install(er|ation services)|dealer|distributor|franchise|trading|traders|supplier|exporter|fabricat|pollution control board|authori[sz]ed (channel )?partner|channel partner|empanel|consultan|advisory|consulting|\bamc\b|o&m services|turnkey|contractor|manpower|recruit|training (institute|centre|center)|agency", re.I)
TECHY = re.compile(r"platform|software|saas|\bai\b|machine learning|\biot\b|patent|deep-?tech|proprietary|r&d|our technology|technology startup|app\b|marketplace|robot|sensor|satellite|we (design|develop|build)", re.I)
OLDCO = re.compile(r"establish\w* (in )?(19\d\d|200[0-7])|since (19\d\d|200[0-7])|\b([2-9]\d)\+? years (of|in|ago)|conglomerate|decades of (experience|excellence)", re.I)
ACAD = re.compile(r"universit|college|\biit\b|institute|department|\blab(oratory)?\b,|research group|foundation$|trust$|society$|association", re.I)
NAVJUNK = re.compile(r"HOMEABOUT|ProductsBlog|Contact\s*Us\s*Home", re.I)
NONPROFIT = {"Nonprofit", "Self-Employed", "Government Agency", "Educational", "Educational Institution"}

kept, dropped = [], Counter()
for r in rows.values():
    d = r["desc_full"]
    if len(d) < 50:
        dropped["short"] += 1; continue
    if NOISE.search(r["name"] + " " + d[:300]):
        dropped["media/events"] += 1; continue
    if not CLIMATE_GATE.search(d):
        dropped["not climate"] += 1; continue
    if ACAD.search(r["name"]) or NAVJUNK.search(d):
        dropped["university lab / junk profile"] += 1; continue
    if r["headcount"] in BIG or r["headcount"] == "1001-5000":
        dropped["large corporate"] += 1; continue
    if not r["funded"]:
        if r["headcount"] == "myself only" or r["ctype"] in NONPROFIT:
            dropped["solo/nonprofit"] += 1; continue
        if r["year"] and r["year"] < 2008:
            dropped["founded before 2008"] += 1; continue
        if SERVICE.search(d) and not TECHY.search(d):
            dropped["service firm (installer/dealer/consultancy)"] += 1; continue
        if OLDCO.search(d):
            dropped["legacy business (decades old)"] += 1; continue
    sec = classify(d, r["name"])
    if not sec:
        dropped["no sector"] += 1; continue
    r["sector"] = sec
    r["tier"] = "f" if r["funded"] else ("b" if BACKED.search(d) else "o")
    kept.append(r)
print("kept", len(kept), Counter(r["tier"] for r in kept), dict(dropped))
json.dump(dict(dropped), open(f"{ROOT}/dropped_counts.json", "w"))

# ---------- seed (research) + portfolios ----------
seed = list(csv.DictReader(open(f"{ROOT}/seed.tsv"), delimiter="\t"))
port = list(csv.DictReader(open(f"{ROOT}/portfolio.tsv"), delimiter="\t"))
backers = defaultdict(set)
for p_ in port:
    backers[norm_domain(p_["domain"]) or norm_name(p_["name"])].update(x.strip() for x in p_["backer"].split(","))
    backers[norm_name(p_["name"])].update(x.strip() for x in p_["backer"].split(","))
FUNDED_EV = re.compile(r"\$|₹|seed|series|pre-?seed|raised|angel|grant", re.I)
r2cities = {}
try:
    for x in json.load(open(f"{ROOT}/agent_cities.json")):
        r2cities[norm_name(x["name"])] = x
except FileNotFoundError:
    pass
extra_updates = []
seed_names0 = {norm_name(s_["name"]) for s_ in seed}
for r2 in csv.DictReader(open(f"{ROOT}/research2.tsv"), delimiter="\t"):
    k = norm_name(r2["name"])
    if k in seed_names0:
        extra_updates.append({"name": r2["name"], "status": "active", "update": r2["evidence"], "source": r2["source"]})
        continue
    ag = r2cities.get(k, {})
    city = r2["city"] or ag.get("city", "")
    seed.append({"name": r2["name"], "website": r2["website"] or ag.get("website", ""), "sector": r2["sector"], "hq": city,
                 "state": r2["state"] or ag.get("state", ""), "founded": "", "founders": "", "funding": r2["evidence"] if FUNDED_EV.search(r2["evidence"]) else "",
                 "hiring": "", "desc": r2["desc"], "evidence": r2["evidence"], "source": r2["source"],
                 "tier": "f" if FUNDED_EV.search(r2["evidence"]) else "b"})
seed_names = {norm_name(s_["name"]) for s_ in seed}
seed_domains = {norm_domain(s_["website"]) for s_ in seed if s_["website"]}
for p_ in port:
    if norm_name(p_["name"]) in seed_names or (p_["domain"] and norm_domain(p_["domain"]) in seed_domains):
        continue
    sec = classify(p_["desc"], p_["name"]) or "build"
    seed.append({"name": p_["name"], "website": p_["domain"], "sector": sec, "hq": "", "founded": "", "founders": "",
                 "funding": "", "hiring": "", "desc": p_["desc"]})

by_domain = {r["domain"]: r for r in kept if r["domain"]}
by_name = {norm_name(r["name"]): r for r in kept}
allrows_dom = {r["domain"]: r for r in rows.values() if r["domain"]}
allrows_name = {norm_name(r["name"]): r for r in rows.values()}
final = []
matched = 0
for s_ in seed:
    dom = norm_domain(s_["website"])
    hit = (allrows_dom.get(dom) if dom else None) or allrows_name.get(norm_name(s_["name"]))
    bk = sorted(backers.get(dom, set()) | backers.get(norm_name(s_["name"]), set()))
    rec = {
        "name": s_["name"], "domain": dom, "desc_full": s_["desc"], "year": int(s_["founded"]) if s_["founded"].isdigit() else None,
        "headcount": "", "linkedin": "", "city": s_["hq"] or "India", "city_raw": s_["hq"], "state_raw": s_.get("state", ""), "sector": s_["sector"], "src": "research",
        "founders": s_["founders"], "funding": s_["funding"], "hiring": s_["hiring"], "logo": "", "tier": s_.get("tier", "f"),
        "ev": s_.get("evidence", ""), "su": s_.get("source", ""),
        "backers": ", ".join(bk), "round": "", "investors": "",
    }
    ag = r2cities.get(norm_name(s_["name"]))
    if ag and (not s_["hq"] or s_["hq"] == "India") and ag.get("city"):
        rec["city"] = ag["city"]; rec["city_raw"] = ag["city"]; rec["state_raw"] = ag.get("state", "")
        rec["domain"] = rec["domain"] or norm_domain(ag.get("website", ""))
    if hit:
        matched += 1
        hit["_merged"] = True
        for k in ("headcount", "linkedin", "logo"):
            rec[k] = hit.get(k) or rec[k]
        rec["domain"] = rec["domain"] or hit["domain"]
        rec["year"] = rec["year"] or hit["year"]
        if rec["city"] in ("", "India"):
            rec["city"] = hit["city"]; rec["city_raw"] = hit["city_raw"]; rec["state_raw"] = hit["state_raw"]
        if hit.get("funded"):
            rec["tier"] = "f"
        rec["embed_extra"] = hit["desc_full"]
        rec["round"] = hit.get("round", ""); rec["investors"] = hit.get("investors", "")
        rec["src"] = "research+crustdata"
    final.append(rec)
print("seed+portfolio", len(seed), "matched to crustdata", matched)
for r in kept:
    if not r.get("_merged"):
        bk = sorted(backers.get(r["domain"], set()) | backers.get(norm_name(r["name"]), set()))
        if bk:
            r["backers"] = ", ".join(bk); r["tier"] = "f"
        final.append(r)

# de-dup crustdata by domain
seen, dedup = set(), []
for r in final:
    key = r["domain"] or norm_name(r["name"])
    nk = norm_name(r["name"])
    if key in seen or (nk and nk in seen):
        continue
    seen.add(key); seen.add(nk)
    dedup.append(r)
final = dedup
print("final", len(final), Counter(r["sector"] for r in final))

# ---------- status / funding updates ----------
upd = list(csv.DictReader(open(f"{ROOT}/updates.tsv"), delimiter="\t")) + extra_updates
byname = defaultdict(list)
for r in final:
    byname[norm_name(r["name"])].append(r)
applied = 0
for u in upd:
    for r in byname.get(norm_name(u["name"]), []):
        applied += 1
        r["up"] = (r.get("up", "") + " " + u["update"]).strip() if u["update"] not in r.get("up", "") else r.get("up", "")
        r["su"] = r.get("su") or u["source"]
        if u["status"] != "active":
            r["sx"] = u["status"]
        if FUNDED_EV.search(u["update"]) or "cohort" in u["update"].lower() or "incubator" in u["update"].lower():
            r["tier"] = "f" if FUNDED_EV.search(u["update"]) else (r.get("tier") if r.get("tier") == "f" else "b")
print("updates applied", applied)

# ---------- embed ----------
custom_stop = set(ENGLISH_STOP_WORDS) | {
    "startup", "startups", "india", "indian", "company", "companies", "solutions", "solution", "services", "service", "pvt",
    "ltd", "private", "limited", "llp", "mission", "vision", "we", "our", "us", "provide", "provides", "providing", "offer",
    "offers", "based", "founded", "team", "leading", "innovative", "help", "helps", "helping", "make", "making", "world",
    "future", "best", "new", "range", "quality", "reliable", "customers", "clients", "products", "product", "focused",
    "focus", "committed", "dedicated", "years", "experience", "join", "today", "global", "nation", "country", "across",
    "end", "including", "aim", "aims", "way", "deliver", "delivering", "growing", "growth", "smart", "technology", "tech",
}
texts = []
for r in final:
    t = (r["name"] + ". ") + r["desc_full"] + " " + r.get("embed_extra", "")
    texts.append(t[:1800])
vec = TfidfVectorizer(stop_words=list(custom_stop), ngram_range=(1, 2), min_df=2, max_df=0.35, sublinear_tf=True,
                      token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
X = vec.fit_transform(texts)
print("tfidf", X.shape)
svd = TruncatedSVD(n_components=110, random_state=7)
Z = normalize(svd.fit_transform(X))

# similar: cosine on Z
S = Z @ Z.T
np.fill_diagonal(S, -1)
similar = np.argsort(-S, axis=1)[:, :10]

reducer = umap.UMAP(n_neighbors=25, min_dist=0.35, spread=1.0, metric="cosine", random_state=42)
P = reducer.fit_transform(Z)
P = (P - P.mean(0)) / P.std(0) * 100

# ---------- cluster labels ----------
K = 32
LABEL_BLOCK = {"spare parts", "supply chain", "treatment plant", "sustainable living", "private limited", "environmental consultancy",
               "consultancy services", "pan india", "state art", "high quality", "long term", "real time", "make india"}
km = KMeans(n_clusters=K, n_init=10, random_state=3).fit(P)
labels = km.labels_
terms = np.array(vec.get_feature_names_out())
Xd = X.toarray() if X.shape[0] * X.shape[1] < 3e7 else None
glob_mean = np.asarray(X.mean(0)).ravel()
cluster_out = []
used = set()
for k in range(K):
    idx = np.where(labels == k)[0]
    if len(idx) < 8:
        continue
    m = np.asarray(X[idx].mean(0)).ravel()
    score = m / (glob_mean + 1e-3) * m
    order = np.argsort(-score)
    pick = None
    for j in order[:40]:
        t = terms[j]
        if " " in t and t not in used and t not in LABEL_BLOCK and t.replace(" ", "") not in {u.replace(" ", "") for u in used}:
            pick = t; break
    if pick is None:
        pick = terms[order[0]]
    used.add(pick)
    cx, cy = P[idx].mean(0)
    cluster_out.append({"name": pick.title().replace("Ev ", "EV ").replace(" Ev", " EV"), "x": round(float(cx), 2),
                        "y": round(float(cy), 2), "size": int(len(idx))})

# ---------- geocode ----------
geo_stats = Counter()
for r in final:
    g = geo.geocode(r.get("city_raw") or r.get("city"), r.get("state_raw", ""))
    if g:
        r["gcity"], r["gstate"], r["lon"], r["lat"], r["prec"] = g
        geo_stats[r["prec"]] += 1
    else:
        r["lon"] = r["lat"] = None; r["prec"] = None; geo_stats["none"] += 1
print("geocoded", dict(geo_stats))

LABEL_RENAME = {"Fruits Vegetables": "Fresh Produce", "Alternatives Everyday": "Bamboo & Eco Products", "Alternatives EVeryday": "Bamboo & Eco Products",
                "Artificial Intelligence": "Climate AI & Software", "Value Chain": "Agri Value Chains", "Iot Sensors": "IoT Sensors"}
for l in cluster_out:
    l["name"] = LABEL_RENAME.get(l["name"], l["name"])

# ---------- export ----------
pts = []
for i, r in enumerate(final):
    sec = r["sector"]
    pts.append({
        "n": r["name"],
        "x": round(float(P[i, 0]), 2), "y": round(float(P[i, 1]), 2),
        "s": sec,
        "d": short_desc(r["desc_full"]),
        "w": r["domain"],
        "c": (r.get("gcity") or (r["city"] if r["city"] not in ("India",) and len(r["city"]) < 30 else "")),
        "st": r.get("gstate") or "",
        "y0": r["year"],
        "h": r["headcount"],
        "li": r["linkedin"],
        "f": r.get("founders", ""),
        "fu": r.get("funding", ""),
        "hi": r.get("hiring", ""),
        "src": "r" if r["src"].startswith("research") else "c",
        "t": r.get("tier", "o"),
        "lg": r.get("logo", ""),
        "bk": r.get("backers", ""),
        "rd": r.get("round", ""),
        "iv": r.get("investors", ""),
        "ev": r.get("ev", ""), "su": r.get("su", ""), "up": r.get("up", ""), "sx": r.get("sx", ""),
        "sz": round(HEAD_MULT.get(r["headcount"], 1.0) * {"f": 1.3, "b": 1.05, "o": 0.85}[r.get("tier", "o")], 2),
        "sim": [int(j) for j in similar[i]],
        "lon": r.get("lon"), "lat": r.get("lat"), "prec": r.get("prec"),
    })
geo.spread(pts)
for p in pts:
    lon, lat = p.pop("glon", None), p.pop("glat", None)
    p.pop("lon"); p.pop("lat")
    if lon is not None:
        p["gx"], p["gy"] = geo.proj(lon, lat)
    p["gp"] = p.pop("prec") or ""
    if p["gp"] == "state" or p["c"] in ("India", "Bangalore Rural"):
        p["c"] = "" if p["c"] != "Bangalore Rural" else "Bengaluru"
json.dump(geo.india_geometry(), open(OUT.rsplit("/", 1)[0] + "/india.json", "w"), separators=(",", ":"))
meta = {"sectors": {k: {"name": v[0], "color": v[1]} for k, v in SECTORS.items()},
        "count": len(pts), "built": "2026-11-20", "tiers": dict(Counter(p["t"] for p in pts)), "geo": dict(geo_stats), "dropped": json.load(open(f"{ROOT}/dropped_counts.json"))}
json.dump({"meta": meta, "points": pts, "labels": cluster_out}, open(OUT, "w"), separators=(",", ":"), ensure_ascii=False)
print("wrote", OUT, len(pts), "points", len(cluster_out), "labels")
print([c["name"] for c in cluster_out])

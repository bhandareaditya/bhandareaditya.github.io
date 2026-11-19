"""Geocoding + India geometry for the climate startup map (offline: GeoNames cities + DataMeet boundaries)."""
import json, math, re, unicodedata
from collections import defaultdict

import geonamescache
from shapely.geometry import shape, Point

GEO_DIR = "/home/claude/geo"
LON0, LAT0, K = 82.8, 22.5, 16.0
COSL = math.cos(math.radians(LAT0))


def proj(lon, lat):
    return round((lon - LON0) * COSL * K, 2), round(-(lat - LAT0) * K, 2)


def deacc(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().strip()


STATE_ALIASES = {
    "national capital territory of delhi": "Delhi", "nct of delhi": "Delhi", "new delhi": "Delhi", "delhi": "Delhi",
    "orissa": "Odisha", "pondicherry": "Puducherry", "jammu and kashmir": "Jammu & Kashmir",
    "andaman and nicobar islands": "Andaman & Nicobar", "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu", "uttaranchal": "Uttarakhand",
}

# metro localities that GeoNames does not resolve well -> (display city, lat, lon)
METRO = [
    ("Bengaluru", 12.97, 77.59, r"bangalore|bengaluru|electronics city|koramangala|whitefield|bommasandra|indiranagar|\bhsr\b|jp nagar|yelahanka|peenya|mathikere|banashankari|padmanabhnagar|carmelram|hulsur|vimanapura|malleswaram|rajajinagar|hebbal|marathahalli|jayanagar|btm|bellandur|sarjapur|hoskote|devanahalli|msrit|chowdeshwari|dasarahalli|ashoknagar \(bangalore\)|bommanahalli|madivala|singasandra|thataguni|udaypura|vidyaranyapura|kengeri|nagarbhavi|basavanagudi|domlur|kr puram|hennur|tyagrajnagar"),
    ("Mumbai", 19.08, 72.88, r"mumbai|powai|andheri|nariman point|rajawadi|sakinaka|vikhroli|bandra|goregaon|malad|borivali|kurla|chembur|lower parel|worli|ghatkopar|mulund|kandivali|dadar|elephanta|bhandup|vile parle"),
    ("Navi Mumbai", 19.03, 73.03, r"navi mumbai|vashi|belapur|nerul|airoli|kharghar|panvel|turbhe"),
    ("Thane", 19.22, 72.98, r"\bthane\b|ambernath|kalyan|dombivli|bhiwandi"),
    ("Pune", 18.52, 73.86, r"\bpune\b|hadapsar|baner|aundh|hinjewadi|kharadi|n\.i\.a\.|bhosari|chinchwad|marunji|punawale|wakad|thathawade|deccan gymkhana|film institute|kothrud|viman nagar|magarpatta|pimpri|talegaon|chakan|sasanenagar|ganeshkhind|indrayaninagar|vidyanagari|ambethan|nanded city"),
    ("Delhi", 28.61, 77.21, r"\bdelhi\b|vasant kunj|okhla|dwarka sector|saket|connaught|nehru place|rohini|janakpuri|lajpat|mayur vihar|pitampura|masood pur|palam|bagdola|barthal|dhulsiras|raj nagar - ii"),
    ("Gurugram", 28.46, 77.03, r"gurgaon|gurugram|dundahera|samaspur|smaspur|sohna|badshahpur|kadipur|sikanderpur|sikohpur|khandsa|kherki daula|wazirabad|gwal pahari|jharsa|dlf ph|south city|manesar|teekli|fazilpur|paltra|narsinghpur|lakhnaula"),
    ("Noida", 28.54, 77.39, r"noida|nepz|nagla charandas|khora gaon|chhajarsi"),
    ("Greater Noida", 28.47, 77.50, r"greater noida"),
    ("Ghaziabad", 28.67, 77.45, r"ghaziabad|indirapuram|vaishali|sahibabad"),
    ("Faridabad", 28.41, 77.32, r"faridabad"),
    ("Hyderabad", 17.39, 78.49, r"hyderabad|secunderabad|serilingampalli|gachibowli|hitec|madhapur|kondapur|uppal|mallapur|nacharam|jubilee hills|hmt township|ramachandrapuram|dilsukhnagar|chaithanyapuri|p & t colony|banjara|kukatpally|begumpet|shamshabad|medchal|peerzadiguda|snehapuri|ameerpet|topgonda"),
    ("Chennai", 13.08, 80.27, r"chennai|taramani|tidel park|guindy|adambakkam|kattankulathur|nandambakkam|ramapuram|annamalaipuram|ekkaduthangal|nilamangai|koyambedu|nerkundram|velachery|perungudi|sholinganallur|tambaram|porur|ambattur|mylapore|anna nagar|t nagar|omr|siruseri"),
    ("Kolkata", 22.57, 88.36, r"kolkata|calcutta|bidhan nagar|salt lake|sech bhawan|nabadiganta|new town|rajarhat|howrah"),
    ("Ahmedabad", 23.02, 72.57, r"ahmedabad|navrangpura|ashram road|darpan society|gujarat university|gandhinagar|sanand|bopal|sg highway"),
    ("Kochi", 9.93, 76.27, r"kochi|cochin|ernakulam|kakkanad|kalamassery"),
    ("Coimbatore", 11.02, 76.96, r"coimbatore|peelamedu|gandhimaanagar"),
    ("Chandigarh", 30.73, 76.78, r"chandigarh|hallomajra|ram darbar"),
    ("Mohali", 30.70, 76.72, r"mohali|sahibzada ajit singh nagar"),
    ("Thiruvananthapuram", 8.52, 76.94, r"thiruvananthapuram|trivandrum|technopark|sreekaryam"),
    ("Jaipur", 26.91, 75.79, r"jaipur"),
    ("Indore", 22.72, 75.86, r"indore"),
    ("Bhopal", 23.26, 77.41, r"bhopal|govindpura"),
    ("Lucknow", 26.85, 80.95, r"lucknow"),
    ("Kanpur", 26.45, 80.33, r"kanpur"),
    ("Bhubaneswar", 20.30, 85.82, r"bhubaneswar|bhubaneshwar|sailashree vihar|suryanagar|baramunda|khorda"),
    ("Vadodara", 22.31, 73.18, r"vadodara|baroda|maneja"),
    ("Surat", 21.17, 72.83, r"surat"),
    ("Nagpur", 21.15, 79.09, r"nagpur"),
    ("Visakhapatnam", 17.69, 83.22, r"visakhapatnam|vizag"),
    ("Mangaluru", 12.91, 74.86, r"mangalore|mangaluru"),
    ("Mysuru", 12.30, 76.64, r"mysore|mysuru"),
    ("Patna", 25.59, 85.14, r"patna"),
    ("Guwahati", 26.14, 91.74, r"guwahati"),
    ("Dehradun", 30.32, 78.03, r"dehradun"),
    ("Goa", 15.49, 73.83, r"\bgoa\b|panaji|margao|verna|vasco"),
]
METRO_RE = [(n, la, lo, re.compile(p, re.I)) for n, la, lo, p in METRO]

_states = None
_cities = None


def load_states():
    global _states
    if _states is None:
        gj = json.load(open(f"{GEO_DIR}/states.json"))
        _states = [(f["properties"]["ST_NM"], shape(f["geometry"])) for f in gj["features"]]
    return _states


def state_at(lon, lat):
    p = Point(lon, lat)
    for n, g in load_states():
        if g.contains(p):
            return n
    best, bd = None, 1e9
    for n, g in load_states():
        d = g.distance(p)
        if d < bd:
            best, bd = n, d
    return best if bd < 0.6 else None


def norm_state(s):
    s = deacc(s).strip()
    if not s:
        return None
    k = s.lower()
    if k in STATE_ALIASES:
        return STATE_ALIASES[k]
    for n, _ in load_states():
        if n.lower() == k:
            return n
    for n, _ in load_states():
        if n.lower().replace("&", "and") == k.replace("&", "and"):
            return n
    return None


def load_cities():
    global _cities
    if _cities is None:
        gc = geonamescache.GeonamesCache()
        idx = defaultdict(list)
        for v in gc.get_cities().values():
            if v["countrycode"] != "IN":
                continue
            names = {v["name"]} | set(v.get("alternatenames") or [])
            for nm in names:
                nm = deacc(nm).lower()
                if 3 <= len(nm) <= 40:
                    idx[nm].append(v)
        _cities = idx
    return _cities


def state_centroid(st):
    for n, g in load_states():
        if n == st:
            c = g.representative_point()
            return c.x, c.y
    return None


def geocode(city_raw, state_raw=""):
    """Return (display_city, state, lon, lat, precision) or None."""
    text = deacc(f"{city_raw or ''}")
    st_hint = norm_state(state_raw) if state_raw else None
    full = f"{text} {deacc(state_raw or '')}"
    for n, la, lo, rx in METRO_RE:
        if rx.search(text):
            return n, state_at(lo, la), lo, la, "city"
    cities = load_cities()
    parts = [p.strip() for p in re.split(r"[,/|()]", text) if p.strip()]
    for part in parts:
        key = part.lower()
        key = re.sub(r"\b(h\.?o|s\.?o|p\.?o|sector \d+|ind area|i\.e\.?|phase [ivx\d]+)\b", "", key).strip(" .")
        cands = cities.get(key)
        if cands:
            if st_hint:
                same = [c for c in cands if state_at(c["longitude"], c["latitude"]) == st_hint]
                if same:
                    cands = same
                elif len(key) < 6:
                    continue
            c = max(cands, key=lambda v: v["population"])
            return c["name"], state_at(c["longitude"], c["latitude"]), c["longitude"], c["latitude"], "city"
    # state-level fallback (state field or a state written in the city field)
    st = st_hint or norm_state(text)
    if not st:
        for n, _ in load_states():
            if n.lower() in full.lower():
                st = n; break
    if st:
        c = state_centroid(st)
        if c:
            return "", st, c[0], c[1], "state"
    for n, la, lo, rx in METRO_RE:
        if rx.search(full):
            return n, state_at(lo, la), lo, la, "city"
    return None


def spread(points):
    """Sunflower jitter around each shared location so dots don't stack. points: list of dicts with lon/lat/prec."""
    groups = defaultdict(list)
    for i, p in enumerate(points):
        if p.get("lon") is None:
            continue
        groups[(round(p["lon"], 3), round(p["lat"], 3))].append(i)
    golden = math.radians(137.508)
    for (lon, lat), idxs in groups.items():
        step = 0.07 if points[idxs[0]]["prec"] == "state" else 0.045
        idxs.sort(key=lambda i: -points[i].get("sz", 1))
        for j, i in enumerate(idxs):
            r = step * math.sqrt(j + 0.5)
            a = j * golden
            points[i]["glon"] = lon + r * math.cos(a) / COSL
            points[i]["glat"] = lat + r * math.sin(a)


def india_geometry():
    out = {"states": [], "districts": []}
    for kind, fn, key in (("states", "states.json", "ST_NM"), ("districts", "districts.json", "DISTRICT")):
        gj = json.load(open(f"{GEO_DIR}/{fn}"))
        for f in gj["features"]:
            g = f["geometry"]
            if not g:
                continue
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            rings = []
            for poly in polys:
                for ring in poly[:1]:
                    flat = []
                    for lon, lat in ring:
                        x, y = proj(lon, lat)
                        flat += [round(x, 1), round(y, 1)]
                    if len(flat) >= 8:
                        rings.append(flat)
            rec = {"n": f["properties"][key], "r": rings}
            if kind == "states":
                c = shape(g).representative_point()
                rec["cx"], rec["cy"] = proj(c.x, c.y)
            else:
                rec["s"] = f["properties"].get("ST_NM", "")
            out[kind].append(rec)
    return out

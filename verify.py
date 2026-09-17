# 검증 스크립트: threads.json / items.jsonl / prompts.jsonl 정합성 + T2 육안 검증용 출력
import json, re, sys
from collections import defaultdict, Counter
sys.path.insert(0, "."); import pilot
data = json.load(open("threads.json", encoding="utf-8"))
items = [json.loads(l) for l in open("items.jsonl", encoding="utf-8")]
prom = defaultdict(dict)
for l in open("prompts.jsonl", encoding="utf-8"):
    r = json.loads(l); prom[r["item_id"]][r["arm"]] = r
TYPES = {"PERSON","CONTACT","ID_NUM","ORG","LOCATION","TIME","ATTRIBUTE","ASSET","OTHER"}
errs, warns = [], []
def E(m): errs.append(m)
def W(m): warns.append(m)

# 1) 스레드 수준
names_seen, emails_seen, pseudo_seen, orgs_seen = {}, {}, {}, {}
for th in data["threads"]:
    tid = th["thread_id"]; text, occ = pilot.parse_template(th["template"])
    ents = th["entities"]
    for eid, e in ents.items():
        if e["type"] not in TYPES: E(f"{tid} {eid} 타입 오류 {e['type']}")
        if e["type"] == "OTHER" and not e.get("note"): E(f"{tid} {eid} OTHER인데 note 없음")
        if e.get("parent") and e["parent"] not in ents: E(f"{tid} {eid} parent 없음")
        forms = {o["form"] for o in occ if o["eid"] == eid}
        miss = forms - set(e["pseudo"]); 
        if miss: E(f"{tid} {eid} pseudo 누락 {miss}")
        if not forms: E(f"{tid} {eid} 본문 등장 없음")
        for f, p in e["pseudo"].items():
            if e["type"] == "PERSON":
                if f in names_seen and names_seen[f] != tid: E(f"인명 중복 {f} {tid}/{names_seen[f]}")
                names_seen[f] = tid
                if p in pseudo_seen and pseudo_seen[p] != tid: E(f"가명 중복 {p}")
                pseudo_seen[p] = tid
                if len(p) != len(f): W(f"{tid} 가명 음절수 차이 {f}->{p}")
            if "@" in f:
                if f in emails_seen: E(f"이메일 중복 {f}")
                emails_seen[f] = tid
    # 표면형이 마커 밖에 남아있는지 (= 가려지지 않는 누수)
    cov = [False]*len(text)
    for o in occ:
        for i in range(o["start"], o["end"]): cov[i] = True
    for eid, e in ents.items():
        for f in e["pseudo"]:
            for m in re.finditer(re.escape(f), text):
                if not all(cov[m.start():m.end()]):
                    E(f"{tid} 마커 밖 표면형 '{f}' @{m.start()}")
    # 가명이 원문에 우연히 존재하는지 (복원 오염 위험)
    for eid, e in ents.items():
        for f, p in e["pseudo"].items():
            if p in text: W(f"{tid} 가명 '{p}'가 원문에 존재 -> 복원 오염 가능")
    # 메일 본문 길이
    blocks = th["template"].split("────────────────────────────")[1:]
    rblocks = [pilot.parse_template(b)[0] for b in blocks]
    if not 3 <= len(rblocks) <= 4: E(f"{tid} 메일 수 {len(rblocks)}")
    for b in rblocks:
        body = b.split("\n\n", 1)[1].strip()
        n = len(body)
        if not 100 <= n <= 200: W(f"{tid} 본문 길이 {n}자")

# 2) 아이템/프롬프트 수준
heur = defaultdict(list); rows_t2 = []
for it in items:
    iid, tid = it["item_id"], it["thread_id"]
    ents, rel = it["entities"], it["relevance"]
    persons = [e for e in ents if ents[e]["type"] == "PERSON"]
    ess = {e for e, r in rel.items() if r == "essential"}
    if it["task_type"] == "T1" and ess: E(f"{iid} T1인데 essential 존재")
    if it["task_type"] == "T3":
        if it["gold"]: E(f"{iid} T3 gold 비어있지 않음")
        if not 6 <= len(it["slots"]) <= 8: E(f"{iid} slot 수 {len(it['slots'])}")
        for s in it["slots"]:
            if not re.search(s, it["context"]) and not re.search(s, it["question"]):
                W(f"{iid} slot '{s}'가 원문에 없음(추론 값이면 정상)")
    # 원문 재현성: gold/slot이 A0 원문에서 뽑힐 수 있어야
    for arm, r in prom[iid].items():
        ctx_q = r["prompt"]
        if arm in ("A0", "A0p"):
            if r["masked_forms"]: E(f"{iid} {arm} 마스킹됨")
            continue
        tgt = pilot._targets(it, pilot.SCOPES[arm])
        tforms = {o["form"] for o in it["occ"] if o["eid"] in tgt}
        keep = {o["form"] for o in it["occ"] if o["eid"] not in tgt}
        for f in tforms:
            # 가려지지 않는 다른 표면형에 포함된 경우는 제외
            if f in ctx_q and not any(f in k for k in keep):
                E(f"{iid} {arm} 가려야 할 '{f}'가 프롬프트에 남음")
        if "가려" in ctx_q or "마스킹" in ctx_q or "PII" in ctx_q: E(f"{iid} {arm} 힌트 문구")
    if it["task_type"] != "T2": continue
    # T2 전용
    target = [e for e in persons if rel[e] == "essential"]
    if len(target) != 1: E(f"{iid} T2 essential PERSON {target}"); continue
    tgt = target[0]; full = [f for f in ents[tgt]["pseudo"] if len(f) == 3][0]
    q = it["question"]
    if full not in q: E(f"{iid} 질문에 대상 이름 없음")
    for g in it["gold"]:
        for e in ents:
            for f in ents[e]["pseudo"]:
                if f in g: E(f"{iid} gold에 PII '{f}'")
    p1a, p1b, p2a = prom[iid]["A1a"]["prompt"], prom[iid]["A1b"]["prompt"], prom[iid]["A2a"]["prompt"]
    q1a = p1a.split("[질문]\n")[1]; q1b = p1b.split("[질문]\n")[1]; q2a = p2a.split("[질문]\n")[1]
    idx = pilot._index_map(it, set(ents))[tgt]
    c1 = "[PERSON]" in q1a and full not in q1a
    c2 = idx in q1b and idx in p1b.split("[질문]")[0]
    others = [f for e in persons if e != tgt for f in ents[e]["pseudo"]]
    c3 = full in q2a and full in p2a.split("[질문]")[0] and not any(f in p2a for f in others)
    # 물리 위치 / 표시번호 / 시각 휴리스틱
    mails = re.findall(r"\[(\d)\] \S+ \(\S\) (\d\d:\d\d)\n보낸사람: (\S+)", it["context"])
    senders = [m[2] for m in mails]
    brs = [int(m[0]) for m in mails]; tms = [m[1] for m in mails]
    pos = senders.index(full)
    hits = {"물리_첫": pos == 0, "물리_끝": pos == len(mails)-1,
            "표시[1]": brs[pos] == 1, "표시_최대": brs[pos] == max(brs),
            "시각_최초": tms[pos] == min(tms), "시각_최후": tms[pos] == max(tms)}
    for k, v in hits.items(): heur[k].append(v)
    # 표시 순서가 시각 순서와 일치하면 함정 실패
    if brs == sorted(brs) and tms == sorted(tms): E(f"{iid} 표시와 시각 일치(함정 없음)")
    if sorted(tms) == tms and brs == sorted(brs): pass
    rows_t2.append((iid, full, pos+1, len(mails), brs, [sorted(tms).index(t)+1 for t in tms],
                    "OK" if c1 else "FAIL", "OK" if c2 else "FAIL", "OK" if c3 else "FAIL",
                    [k for k, v in hits.items() if v]))
    if not (c1 and c2 and c3): E(f"{iid} T2 육안검증 실패 A1a={c1} A1b={c2} A2a={c3}")

print("== T2 검증표 ==")
print("item      대상    물리위치  표시번호     시각순위    A1a  A1b  A2a  맞히는 휴리스틱")
for r in rows_t2:
    print(f"{r[0]:<9} {r[1]:<5} {r[2]}/{r[3]}      {str(r[4]):<12} {str(r[5]):<11} {r[6]:<4} {r[7]:<4} {r[8]:<4} {r[9]}")
print("\n== 휴리스틱별 적중 수 (10개 중) ==")
for k, v in heur.items(): print(f"  {k:<8} {sum(v)}/{len(v)}")
print("\n== T2 대상 물리 위치 분포 ==", Counter(r[2] for r in rows_t2))
print("\n== 스레드별 부모 없는 비인명 PII(=T3 A2에서 가려지는 것) ==")
for th in data["threads"]:
    orphan = [f"{e}:{v['type']}" for e, v in th["entities"].items() if not v.get("parent") and v["type"] != "PERSON"]
    extra = [f"{e}:{v['type']}" for e, v in th["entities"].items() if v["type"] in ("ID_NUM","ASSET","LOCATION") or (v["type"]=="CONTACT" and "-" in list(v["pseudo"])[0] and "@" not in list(v["pseudo"])[0])]
    print(f"  {th['thread_id']}  부모없음={orphan}  전화/사번/계좌/주소={len(extra)}")
# 선언 relevance와 build 후 실효 relevance 차이
print("\n== 선언 relevance ≠ build 실효 relevance ==")
for th in data["threads"]:
    for q in th["questions"]:
        it = next(i for i in items if i["item_id"] == q["question_id"])
        diff = {e: (q["relevance"].get(e), it["relevance"][e]) for e in it["relevance"] if q["relevance"].get(e) != it["relevance"][e]}
        if diff: print(f"  {q['question_id']}: {diff}")
print("\n== 경고 ==");  [print("  " + w) for w in warns]
print("\n== 오류 ==");  [print("  " + e) for e in errs]
print("\n결과:", "PASS" if not errs else f"FAIL ({len(errs)})")
# 육안 검증용 T2 프롬프트 덤프
with open("t2_visual_check.txt", "w", encoding="utf-8") as fo:
    for it in items:
        if it["task_type"] != "T2": continue
        for arm in ("A1a", "A1b", "A2a"):
            fo.write(f"\n######## {it['item_id']} / {arm} ########\n{prom[it['item_id']][arm]['prompt']}\n")

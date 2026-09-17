#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_extra.py — 사전등록 v0.3의 판정 규칙을 그대로 구현한 추가 분석
=====================================================================

pilot.py analyze가 다루지 않는 부분을 계산한다.

  1. 잡음 하한선 w           A0 vs A0p 부트스트랩 CI의 너비          (5절)
  2. 확증 대비 3개 + 판정     H1 -> H2 순서, |Δ| < w 해석 금지         (4절, 7절 1~4)
  3. 탐색적 비교 25쌍         부트스트랩 p값 + Holm 보정               (4절)
  4. 범위 × 표현 상호작용     I_i = (A1c − A1a) − (A2c − A2a)          (6.2절)
  5. 사람 짝비교 (T3)         3인 다수결(1-1-1은 동등), Fleiss' kappa,
                              사람↔slot recall Cohen's kappa, Spearman  (3.4절, 7절 5~6)

사용법
------
  python src/analysis_extra.py                       # runs/scored.jsonl
  python src/analysis_extra.py \
      --human rating/rater_a.json rating/rater_b.json rating/rater_c.json \
      --out runs/report.json

  --human 에는 compare.html이 내보낸 JSON 파일을 평가자 수만큼 넣는다.
  외부 패키지 없이 순수 파이썬으로 동작한다. 주 지표 정의는 pilot.py의 _primary를 그대로 쓴다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict

import pilot  # 같은 폴더의 pilot.py (ARMS, SCOPES, REPRS, _primary, _mean)
from pilot import RUNS

B = 5000            # 부트스트랩 반복 수 (사전등록 4절)
SEED = 7            # pilot._paired_boot와 같은 시드 -> 같은 CI
ALPHA = 0.05
KAPPA_MIN = 0.40    # 사전등록 7절 5·6번
TIE_WARN = 0.20     # 1-1-1 비율 경고 기준 (3.4절)
N_RATERS = 3
EPS = 1e-9         # 부동소수 오차 허용 (예: 평균이 -1e-17로 나와 "0 미포함"으로 오판하는 것 방지)

CONFIRMATORY = [("H1", "A0", "A1a"), ("H2", "A0", "A2a"), ("범위", "A1a", "A2a")]

LABEL_A0, LABEL_OTHER, LABEL_TIE = "A0 우세", "비교 arm 우세", "동등"
LABELS = [LABEL_A0, LABEL_OTHER, LABEL_TIE]
SCORE = {LABEL_A0: 1, LABEL_TIE: 0, LABEL_OTHER: -1}


def nan() -> float:
    return float("nan")


def includes0(lo, hi) -> bool:
    return lo <= EPS and hi >= -EPS


def isnum(x) -> bool:
    return isinstance(x, (int, float)) and x == x


def fmt(x, nd=3, sign=True) -> str:
    if not isnum(x):
        return "nan"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


# =============================================================================
# 부트스트랩
# =============================================================================

def boot(diffs: list[float]) -> dict:
    """아이템별 차이 d_i의 부트스트랩.
    pilot._paired_boot와 같은 난수 순서를 써서 CI가 똑같이 나온다.
    반환: 평균, 95% CI, 부트스트랩 p값, n"""
    d = [x for x in diffs if isnum(x)]
    if len(d) < 3:
        return {"n": len(d), "delta": nan(), "lo": nan(), "hi": nan(), "p": nan()}
    rng = random.Random(SEED)
    means = sorted(pilot._mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(B))
    le0 = sum(1 for m in means if m <= EPS) / B
    ge0 = sum(1 for m in means if m >= -EPS) / B
    return {"n": len(d), "delta": pilot._mean(d),
            "lo": means[int(.025 * B)], "hi": means[int(.975 * B)],
            "p": min(1.0, 2 * min(le0, ge0))}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm 보정 p값. 보정 p < α 이면 기각(유의)."""
    items = sorted(((p if isnum(p) else 1.0), k) for k, p in pvals.items())
    m, out, run = len(items), {}, 0.0
    for j, (p, k) in enumerate(items, 1):
        run = max(run, min(1.0, (m - j + 1) * p))
        out[k] = run
    return out


# =============================================================================
# 자동 지표 분석
# =============================================================================

def load_scored(path: str):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    by = defaultdict(dict)                    # item_id -> arm -> metrics
    for r in rows:
        by[r["item_id"]][r["arm"]] = r["metrics"]
    return by


def diffs(by, a: str, b: str, task: str | None = None) -> list[float]:
    out = []
    for m in by.values():
        if a in m and b in m and (task is None or m[a]["task"] == task):
            out.append(pilot._primary(m[a]) - pilot._primary(m[b]))
    return out


def auto_analysis(by) -> dict:
    rep = {}

    # ---- 1. 잡음 하한선 ----
    nf = boot(diffs(by, "A0", "A0p"))
    w = nf["hi"] - nf["lo"] if isnum(nf["lo"]) else nan()
    nf["w"] = w
    nf["warning"] = (None if (not isnum(nf["lo"]) or includes0(nf["lo"], nf["hi"]))
                     else "A0 vs A0p CI가 0을 포함하지 않음 -> 생성 안정성·캐시·gen_id 배선 점검 후 확증 분석")
    rep["noise_floor"] = nf

    # ---- 2. 확증 대비 + 판정 규칙 ----
    conf = {}
    for label, a, b in CONFIRMATORY:
        r = boot(diffs(by, a, b))
        r["pair"] = f"{a} vs {b}"
        r["ci_excludes_0"] = isnum(r["lo"]) and not includes0(r["lo"], r["hi"])
        r["below_noise"] = isnum(w) and isnum(r["delta"]) and abs(r["delta"]) < w
        conf[label] = r

    h1 = conf["H1"]
    h1_ok = isnum(h1["lo"]) and h1["lo"] > EPS and not h1["below_noise"]
    verdict = {"H1": "확인" if h1_ok else "미확인 -> 측정 도구 민감도 부족으로 기록, 지표 재설계 후 재등록"}

    h2 = conf["H2"]
    if not h1_ok:
        verdict["H2"] = "해석 보류 (H1 미확인). 수치만 보고"
    elif isnum(h2["lo"]) and includes0(h2["lo"], h2["hi"]):
        verdict["H2"] = (f"A0과 구분되지 않음. 배제 가능한 최대 하락 폭 = {fmt(-h2['lo'], sign=False)}"
                         f" (잡음 폭 w = {fmt(w, sign=False)})")
    else:
        verdict["H2"] = "A0과 차이 있음" + (" (단, |Δ| < w 이므로 해석하지 않음)" if h2["below_noise"] else "")

    sc = conf["범위"]
    # 대비 ③은 Δ = A1a − A2a 이므로, 기대 방향(A2a > A1a)이면 CI 상한 < 0
    if isnum(sc["hi"]) and sc["hi"] < -EPS and not sc["below_noise"]:
        verdict["범위"] = "확인 (A2a > A1a)"
    elif sc["below_noise"] and sc["ci_excludes_0"]:
        verdict["범위"] = "CI는 0을 벗어나지만 |Δ| < w 이므로 해석하지 않음"
    else:
        verdict["범위"] = "미확인"
    rep["confirmatory"] = conf
    rep["verdict"] = verdict

    # ---- 과제별 (탐색적) ----
    per_task = {}
    for t in ("T1", "T2", "T3"):
        per_task[t] = {label: boot(diffs(by, a, b, t)) for label, a, b in CONFIRMATORY}
    rep["confirmatory_by_task"] = per_task

    # ---- 3. 탐색적 25쌍 + Holm ----
    conf_pairs = {(a, b) for _, a, b in CONFIRMATORY}
    expl = {}
    for a, b in itertools.combinations(pilot.ARMS, 2):
        if (a, b) in conf_pairs:
            continue
        r = boot(diffs(by, a, b))
        r["below_noise"] = isnum(w) and isnum(r["delta"]) and abs(r["delta"]) < w
        expl[f"{a} vs {b}"] = r
    adj = holm({k: v["p"] for k, v in expl.items()})
    for k, v in expl.items():
        v["p_holm"] = adj[k]
        v["significant"] = adj[k] < ALPHA and not v["below_noise"]
    rep["exploratory"] = expl

    # ---- 4. 상호작용 ----
    inter = []
    for m in by.values():
        if all(x in m for x in ("A1a", "A1c", "A2a", "A2c")):
            q = {x: pilot._primary(m[x]) for x in ("A1a", "A1c", "A2a", "A2c")}
            inter.append((q["A1c"] - q["A1a"]) - (q["A2c"] - q["A2a"]))
    ir = boot(inter)
    ir["note"] = "사전 예측 방향: I > 0 (탐색적 검정)"
    rep["interaction"] = ir

    # ---- 표현 사다리 (점검용) ----
    ladder = {}
    for scope, arms in (("all", ("A1a", "A1b", "A1c")), ("ne", ("A2a", "A2b", "A2c"))):
        vals = [pilot._mean([pilot._primary(m[x]) for m in by.values() if x in m]) for x in arms]
        ok = all(isnum(v) for v in vals) and vals[0] <= vals[1] + 1e-9 and vals[1] <= vals[2] + 1e-9
        ladder[scope] = {"type": vals[0], "index": vals[1], "pseudonym": vals[2],
                         "ok": ok}
    rep["ladder"] = ladder
    return rep


# =============================================================================
# 사람 평가 분석
# =============================================================================

def to_label(rec: dict) -> str:
    """compare.html 기록 -> {A0 우세, 비교 arm 우세, 동등}"""
    _, base, other = rec["pair_id"].split("|")
    w = rec.get("winner_arm")
    if w is None:
        return LABEL_TIE
    if w == base:
        return LABEL_A0
    if w == other:
        return LABEL_OTHER
    raise ValueError(f"winner_arm이 쌍에 없는 arm: {rec}")


def majority(labels: list[str]) -> tuple[str, bool]:
    """규칙 A: 2인 이상이 고른 라벨. 전부 다르면 '동등'. (라벨, 1-1-1 여부)"""
    top, cnt = Counter(labels).most_common(1)[0]
    if cnt >= 2:
        return top, False
    return LABEL_TIE, True


def fleiss_kappa(table: list[dict[str, int]], n_raters: int) -> float:
    """table: 쌍마다 {라벨: 그 라벨을 고른 평가자 수}"""
    N = len(table)
    if N == 0:
        return nan()
    n = n_raters
    p_j = {c: sum(row.get(c, 0) for row in table) / (N * n) for c in LABELS}
    P_i = [(sum(row.get(c, 0) ** 2 for c in LABELS) - n) / (n * (n - 1)) for row in table]
    P_bar = sum(P_i) / N
    P_e = sum(v * v for v in p_j.values())
    if abs(1 - P_e) < 1e-12:
        return nan()          # 모두 한 라벨만 고름 -> 정의 불가
    return (P_bar - P_e) / (1 - P_e)


def cohen_kappa(x: list[str], y: list[str]) -> float:
    n = len(x)
    if n == 0:
        return nan()
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[c] * cy[c] for c in LABELS) / (n * n)
    if abs(1 - pe) < 1e-12:
        return nan()
    return (po - pe) / (1 - pe)


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1       # 동점은 평균 순위
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return nan()
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    syy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return sxy / (sxx * syy) if sxx and syy else nan()


def human_analysis(paths: list[str], by) -> dict:
    raters = {}
    for p in paths:
        recs = json.load(open(p, encoding="utf-8"))
        if isinstance(recs, dict):
            recs = [recs]
        ev = {r["evaluator"] for r in recs}
        if len(ev) != 1:
            raise ValueError(f"{p}: 한 파일에 평가자 ID가 여러 개 {ev}")
        name = ev.pop()
        if name in raters:
            raise ValueError(f"평가자 ID 중복: {name} (평가자마다 다른 ID를 써야 함)")
        raters[name] = {r["pair_id"]: to_label(r) for r in recs}

    rep = {"raters": {k: len(v) for k, v in raters.items()}, "warnings": []}
    if len(raters) != N_RATERS:
        rep["warnings"].append(f"평가자 수 {len(raters)}명 (사전등록은 {N_RATERS}명)")

    all_pairs = set().union(*[set(v) for v in raters.values()])
    common = sorted(p for p in all_pairs if all(p in v for v in raters.values()))
    missing = len(all_pairs) - len(common)
    if missing:
        rep["warnings"].append(f"일부 평가자가 빠뜨린 쌍 {missing}개 -> 모든 평가자가 평가한 {len(common)}쌍만 사용")
    if len(common) != 70:
        rep["warnings"].append(f"공통 평가 쌍 {len(common)}개 (사전등록은 70쌍)")

    # ---- 규칙 A: 다수결, 1-1-1은 동등 ----
    names = sorted(raters)
    pairs, table, n_split = [], [], 0
    for pid in common:
        labs = [raters[n][pid] for n in names]
        final, split = majority(labs)
        n_split += split
        table.append(Counter(labs))
        pairs.append({"pair_id": pid, "labels": dict(zip(names, labs)), "final": final,
                      "split_1_1_1": split,
                      "mean_score": sum(SCORE[l] for l in labs) / len(labs)})
    split_rate = n_split / len(common) if common else nan()
    rep["final_label_counts"] = dict(Counter(p["final"] for p in pairs))
    rep["split_1_1_1"] = {"n": n_split, "rate": split_rate,
                          "warning": (f"1-1-1 비율 {split_rate:.0%} ≥ {TIE_WARN:.0%}: '동등' 라벨이 부풀려졌을 수 있음"
                                      if isnum(split_rate) and split_rate >= TIE_WARN else None)}

    # ---- 7절 5번: 평가자 간 일치도 ----
    fk = fleiss_kappa(table, len(names))
    rep["fleiss_kappa"] = fk
    rater_ok = isnum(fk) and fk >= KAPPA_MIN
    rep["rater_agreement_pass"] = rater_ok

    # ---- 7절 6번: 사람 ↔ slot recall ----
    h, m, scores, sdiff, blind_leak = [], [], [], [], 0
    for p in pairs:
        item, base, other = p["pair_id"].split("|")
        mm = by.get(item, {})
        if base not in mm or other not in mm:
            rep["warnings"].append(f"scored에 없는 쌍: {p['pair_id']}")
            continue
        a, b = mm[base]["slot"], mm[other]["slot"]
        if not (isnum(a) and isnum(b)):
            continue
        d = a - b
        auto = LABEL_A0 if d > EPS else LABEL_OTHER if d < -EPS else LABEL_TIE
        p["auto"], p["slot_diff"] = auto, d
        h.append(p["final"]); m.append(auto); scores.append(p["mean_score"]); sdiff.append(d)
        if mm[base].get("unresolved", 0) > 0 or mm[other].get("unresolved", 0) > 0:
            blind_leak += 1

    ck = cohen_kappa(h, m)
    rep["cohen_kappa_human_vs_slot"] = ck
    rep["spearman_meanscore_vs_slotdiff"] = spearman(scores, sdiff)
    rep["blind_leak_pairs"] = blind_leak
    if not rater_ok:
        rep["slot_recall_decision"] = "보류 — 평가자 간 일치도 부족(Fleiss' κ < 0.40). 평가 기준 문구를 다듬고 재평가"
    elif isnum(ck) and ck >= KAPPA_MIN:
        rep["slot_recall_decision"] = "본실험 T3 지표로 사용 가능 (Cohen's κ ≥ 0.40)"
    else:
        rep["slot_recall_decision"] = "본실험 T3 지표로 사용하지 않음 (Cohen's κ < 0.40 또는 계산 불가)"
    rep["pairs"] = pairs
    return rep


# =============================================================================
# 출력
# =============================================================================

def print_auto(r: dict) -> None:
    nf = r["noise_floor"]
    print("\n[1] 잡음 하한선  A0 vs A0p")
    print(f"    Δ={fmt(nf['delta'])}  95%CI [{fmt(nf['lo'])}, {fmt(nf['hi'])}]  잡음 폭 w={fmt(nf['w'], sign=False)}")
    if nf["warning"]:
        print("    ⚠ " + nf["warning"])

    print("\n[2] 확증 대비 (주 지표: T1 cont / T2 slot_all / T3 slot, 30 아이템)")
    for label, v in r["confirmatory"].items():
        flag = "  |Δ|<w" if v["below_noise"] else ""
        print(f"    {label:<4} {v['pair']:<12} Δ={fmt(v['delta'])}  95%CI [{fmt(v['lo'])}, {fmt(v['hi'])}]  n={v['n']}{flag}")
    print("    판정")
    for k, v in r["verdict"].items():
        print(f"      {k:<4} {v}")

    print("\n    과제별 (탐색적, n=10)")
    for t, d in r["confirmatory_by_task"].items():
        s = "  ".join(f"{k} Δ={fmt(v['delta'])} [{fmt(v['lo'])},{fmt(v['hi'])}]" for k, v in d.items())
        print(f"      {t}  {s}")

    print("\n[3] 탐색적 비교 25쌍 (부트스트랩 p, Holm 보정 α=0.05)")
    print(f"    {'쌍':<14}{'Δ':>8}{'CI 하한':>9}{'CI 상한':>9}{'p':>8}{'p_holm':>8}  결과")
    for k, v in sorted(r["exploratory"].items(), key=lambda kv: kv[1]["p_holm"]):
        res = "유의" if v["significant"] else ("|Δ|<w" if v["below_noise"] and v["p_holm"] < ALPHA else "-")
        print(f"    {k:<14}{fmt(v['delta']):>8}{fmt(v['lo']):>9}{fmt(v['hi']):>9}"
              f"{fmt(v['p'], sign=False):>8}{fmt(v['p_holm'], sign=False):>8}  {res}")

    ir = r["interaction"]
    print("\n[4] 범위 × 표현 상호작용  I = (A1c−A1a) − (A2c−A2a)")
    print(f"    평균 I={fmt(ir['delta'])}  95%CI [{fmt(ir['lo'])}, {fmt(ir['hi'])}]  n={ir['n']}  ({ir['note']})")

    print("\n    표현 사다리 점검 (기대: type ≤ index ≤ pseudonym)")
    for s, v in r["ladder"].items():
        ok = "OK" if v["ok"] else "위반 -> 구현·측정 점검 후 해석"
        print(f"      {s:<4} type={fmt(v['type'], sign=False)} index={fmt(v['index'], sign=False)} "
              f"pseudo={fmt(v['pseudonym'], sign=False)}  {ok}")


def print_human(r: dict) -> None:
    print("\n[5] 사람 짝비교 (T3)")
    print(f"    평가자: {r['raters']}")
    for w in r["warnings"]:
        print("    ⚠ " + w)
    print(f"    최종 라벨(다수결, 1-1-1은 동등): {r['final_label_counts']}")
    s = r["split_1_1_1"]
    print(f"    1-1-1 쌍: {s['n']}개 ({fmt(s['rate'], 2, sign=False)})" + (f"  ⚠ {s['warning']}" if s["warning"] else ""))
    print(f"    ① 평가자 간 Fleiss' κ = {fmt(r['fleiss_kappa'], sign=False)}  "
          f"-> {'통과' if r['rater_agreement_pass'] else '미달(기준 0.40)'}")
    print(f"    ② 사람↔slot recall Cohen's κ = {fmt(r['cohen_kappa_human_vs_slot'], sign=False)}   "
          f"Spearman(평균 선호 점수, slot 차이) = {fmt(r['spearman_meanscore_vs_slotdiff'])}")
    print(f"    결정: {r['slot_recall_decision']}")
    print(f"    블라인드 누수 가능 쌍(어느 한쪽 unresolved>0): {r['blind_leak_pairs']}")


def main():
    ap = argparse.ArgumentParser(description="사전등록 v0.3 판정 규칙에 따른 추가 분석")
    ap.add_argument("--scored", default=str(RUNS / "scored.jsonl"))
    ap.add_argument("--human", nargs="*", default=[], help="compare.html 내보내기 JSON (평가자별 1개)")
    ap.add_argument("--out", default="", help="전체 결과를 JSON으로 저장할 경로")
    a = ap.parse_args()

    by = load_scored(a.scored)
    report = {"auto": auto_analysis(by)}
    print_auto(report["auto"])
    if a.human:
        report["human"] = human_analysis(a.human, by)
        print_human(report["human"])
    else:
        print("\n[5] 사람 짝비교: --human 파일이 없어 건너뜀")
    if a.out:
        json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2,
                  default=lambda x: None)
        print(f"\n-> {a.out}")
    print()


if __name__ == "__main__":
    main()

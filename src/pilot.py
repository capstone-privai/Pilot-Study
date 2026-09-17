#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pilot.py — PII 마스킹 파일럿 실험 파이프라인
============================================

확정된 설계
-----------
H1  전부 마스킹하면 품질이 유의하게 떨어진다        (양성 대조군)
H2  무관 PII만 마스킹하면 원본과 구분되지 않는다     (주 주장)

arm = 범위 2수준 × 표현 3수준 + 원본 + 원본재생성 = 8
  A0  none      / -          상한선
  A0p none      / -          잡음 하한선 (동일 프롬프트 재생성)
  A1a all       / type       [PERSON]        <- H1 주 조건
  A1b all       / index      [PERSON_1]
  A1c all       / pseudonym  랜덤 가명
  A2a essential 제외 / type                   <- H2 주 조건
  A2b essential 제외 / index
  A2c essential 제외 / pseudonym

N = 스레드 10 × 질문 3 = 30,  총 생성 = 240

단계
----
  build    data/threads.json -> runs/items.jsonl     (마커 파싱, 오프셋 자동 기록)
  prompts  runs/items.jsonl  -> runs/prompts.jsonl   (8 arm 전개)
  gen      runs/prompts.jsonl -> runs/raw.jsonl      (Groq 호출, 캐시, 429 백오프)
  restore  runs/raw.jsonl    -> runs/restored.jsonl  (복원. 유출은 raw에서 측정)
  score    runs/restored.jsonl -> runs/scored.jsonl
           주 지표: T1 containment / T2 slot_all / T3 slot recall
  analyze  runs/scored.jsonl -> 리포트

  demo     전 단계를 echo 백엔드로 한 번에 (배선 확인용)

경로 기본값은 저장소 루트 기준(data/, runs/)이며 어느 디렉터리에서 실행해도 된다.
API 키는 루트의 .env(GROQ_API_KEY=...) 또는 환경변수에서 읽는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

# 저장소 기준 경로. 어느 디렉터리에서 실행해도 같은 파일을 가리킨다.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)


def _load_env() -> None:
    """ROOT/.env 의 KEY=VALUE 를 환경변수로 등록. 이미 있는 값은 덮지 않는다."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_env()

MARKER = re.compile(r"\{\{([A-Za-z0-9_]+)\|(.+?)\}\}")

SCOPES = {"A0": "none", "A0p": "none",
          "A1a": "all", "A1b": "all", "A1c": "all",
          "A2a": "ne",  "A2b": "ne",  "A2c": "ne"}
REPRS = {"A0": "-", "A0p": "-",
         "A1a": "type", "A1b": "index", "A1c": "pseudonym",
         "A2a": "type", "A2b": "index", "A2c": "pseudonym"}
ARMS = list(SCOPES)


# =============================================================================
# 1. build — 마커 파싱
# =============================================================================

def parse_template(tpl: str) -> tuple[str, list[dict]]:
    """{{E2|최유진}} 마커를 제거하면서 문자 오프셋을 기록한다.
    손으로 span을 세다 틀리는 사고를 원천 차단하는 장치."""
    out, occ, pos, last = [], [], 0, 0
    for m in MARKER.finditer(tpl):
        out.append(tpl[last:m.start()])
        pos += m.start() - last
        form = m.group(2)
        occ.append({"eid": m.group(1), "form": form,
                    "start": pos, "end": pos + len(form)})
        out.append(form)
        pos += len(form)
        last = m.end()
    out.append(tpl[last:])
    return "".join(out), occ


def build(threads_path: str, out_path: str) -> None:
    data = json.load(open(threads_path, encoding="utf-8"))
    n = 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for th in data["threads"]:
            text, occ = parse_template(th["template"])
            # 검증: 기록한 오프셋이 실제 텍스트와 일치하는가
            for o in occ:
                assert text[o["start"]:o["end"]] == o["form"], \
                    f"offset mismatch {th['thread_id']} {o}"
            for q in th["questions"]:
                # 자식 개체는 부모의 relevance를 상속 (원칙: 한 개체의 모든 표면형을 함께 가림)
                rel = dict(q["relevance"])
                for eid, e in th["entities"].items():
                    if e.get("parent"):
                        rel[eid] = rel.get(e["parent"], rel.get(eid, "not_essential"))
                fo.write(json.dumps({
                    "item_id": q["question_id"], "thread_id": th["thread_id"],
                    "task_type": q["task_type"], "context": text,
                    "question": q["question"], "gold": q.get("gold", []),
                    "slots": q.get("slots", []),
                    "entities": th["entities"], "occ": occ, "relevance": rel,
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"build: {n} items -> {out_path}")


# =============================================================================
# 2. mask / restore
# =============================================================================

def _targets(item: dict, scope: str) -> set[str]:
    if scope == "none":
        return set()
    if scope == "all":
        return set(item["entities"])
    return {e for e, r in item["relevance"].items() if r == "not_essential"}


def _index_map(item: dict, targets: set[str]) -> dict[str, str]:
    """타입별로 첫 등장 순서대로 1,2,3... 부여 -> [PERSON_1]"""
    seen, counter, out = {}, defaultdict(int), {}
    for o in item["occ"]:
        eid = o["eid"]
        if eid in targets and eid not in seen:
            t = item["entities"][eid]["type"]
            counter[t] += 1
            seen[eid] = True
            out[eid] = f"[{t}_{counter[t]}]"
    return out


def mask(item: dict, scope: str, repr_: str) -> tuple[str, str, dict]:
    """(마스킹된 context, 마스킹된 question, 복원맵)을 돌려준다.

    복원맵의 키는 응답에 나타날 문자열, 값은 되돌릴 원본 문자열.
    repr='type'이면 타입당 개체가 1개일 때만 복원 가능하다(최선 복원)."""
    targets = _targets(item, scope)
    ctx, q = item["context"], item["question"]
    if not targets:
        return ctx, q, {}

    idx = _index_map(item, targets) if repr_ == "index" else {}
    # 타입당 대상 개체 수 -> type 표현의 복원 가능 여부 판정용
    per_type = defaultdict(set)
    for e in targets:
        per_type[item["entities"][e]["type"]].add(e)

    restore, repl_pairs = {}, []
    for o in item["occ"]:
        eid, form = o["eid"], o["form"]
        if eid not in targets:
            continue
        ent = item["entities"][eid]
        if repr_ == "type":
            new = f"[{ent['type']}]"
            if len(per_type[ent["type"]]) == 1:
                restore[new] = form              # 유일 개체 -> 복원 가능
        elif repr_ == "index":
            new = idx[eid]
            restore.setdefault(new, form)
        else:
            new = ent["pseudo"].get(form, f"[{ent['type']}]")
            restore[new] = form
        repl_pairs.append((o["start"], o["end"], new))

    # context는 오프셋 기반으로 뒤에서부터 치환 (앞에서 하면 오프셋이 밀림)
    chars = list(ctx)
    for s, e, new in sorted(repl_pairs, key=lambda x: -x[0]):
        chars[s:e] = list(new)
    ctx_m = "".join(chars)

    # question은 오프셋이 없으므로 긴 표면형부터 문자열 치환
    forms = sorted(
        {(o["form"], o["eid"]) for o in item["occ"] if o["eid"] in targets},
        key=lambda x: -len(x[0]))
    for form, eid in forms:
        ent = item["entities"][eid]
        if repr_ == "type":
            new = f"[{ent['type']}]"
        elif repr_ == "index":
            new = idx[eid]
        else:
            new = ent["pseudo"].get(form, f"[{ent['type']}]")
        q = q.replace(form, new)

    return ctx_m, q, restore


def restore_text(resp: str, restore: dict) -> tuple[str, int]:
    """복원 + 복원 못 한 자리표시자 개수를 함께 반환."""
    out = resp
    for k, v in sorted(restore.items(), key=lambda x: -len(x[0])):
        out = out.replace(k, v)
    unresolved = len(re.findall(r"\[[A-Z_]+(?:_\d+)?\]", out))
    return out, unresolved


PROMPT_TMPL = ("다음 이메일 스레드를 읽고 질문에 답하세요.\n\n"
               "[스레드]\n{ctx}\n\n[질문]\n{q}")
# 주의: "PII가 가려져 있습니다" 류의 힌트를 절대 넣지 말 것.
# 넣는 순간 '모델이 마스킹을 인지하고 대응하는' 다른 실험이 된다.


def prompts(in_path: str, out_path: str, seed: int = 20260917) -> None:
    items = [json.loads(l) for l in open(in_path, encoding="utf-8")]
    rows = []
    for it in items:
        for arm in ARMS:
            ctx, q, rest = mask(it, SCOPES[arm], REPRS[arm])
            rows.append({
                "gen_id": "", "item_id": it["item_id"], "arm": arm,
                "scope": SCOPES[arm], "repr": REPRS[arm],
                "prompt": PROMPT_TMPL.format(ctx=ctx, q=q),
                "restore": rest,
                "masked_forms": sorted({o["form"] for o in it["occ"]
                                        if o["eid"] in _targets(it, SCOPES[arm])}),
            })
    random.Random(seed).shuffle(rows)        # 순서 효과 방지
    with open(out_path, "w", encoding="utf-8") as fo:
        for i, r in enumerate(rows, 1):
            r["gen_id"] = f"gen_{i:04d}"
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"prompts: {len(rows)} -> {out_path}")


# =============================================================================
# 3. gen — Groq (OpenAI 호환)
# =============================================================================

class Cache:
    def __init__(self, path):
        self.path, self.mem = path, {}
        if os.path.exists(path):
            for l in open(path, encoding="utf-8"):
                d = json.loads(l)
                self.mem[d["k"]] = d["v"]

    def get(self, k): return self.mem.get(k)

    def put(self, k, v):
        self.mem[k] = v
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


# backend는 (응답 텍스트, 이번 호출이 쓴 총 토큰 수)를 돌려준다. 토큰 수는 TPM 페이싱용.
Backend = Callable[[str, str], tuple[str, int]]


def backend_echo(prompt: str, model: str) -> tuple[str, int]:
    """배선 확인 전용 더미. 추론 없이 정규식으로 긁어옴 -> 실험 결과 아님."""
    body = prompt.split("[스레드]", 1)[-1].split("[질문]", 1)[0]
    places = re.findall(r"(\d층\s*\S*회의실|본관\s*세미나실)", body)
    return ", ".join(dict.fromkeys(places)) or "정보 없음", 0


def _retry_after(e: Exception) -> float:
    """429 응답에서 재시도까지 남은 초. 헤더가 없으면 본문의 'try again in 3h47m12s'를 읽는다."""
    try:
        ra = e.response.headers.get("retry-after")          # openai.APIStatusError
        if ra:
            return float(ra)
    except Exception:
        pass
    m = re.search(r"try again in\s*(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", str(e))
    if m and any(m.groups()):
        h, mi, se = (float(x or 0) for x in m.groups())
        return h * 3600 + mi * 60 + se
    return 0.0


def backend_groq(api_key: str, base_url: str) -> Backend:
    from openai import OpenAI, RateLimitError      # pip install openai
    cli = OpenAI(api_key=api_key, base_url=base_url)

    def call(prompt: str, model: str) -> tuple[str, int]:
        attempt = 0
        while attempt < 6:
            try:
                # max_tokens는 실제 사용분만 TPM에 잡히므로(예약 아님) 넉넉히 둔다.
                # T3 회신이 low에서도 ~800까지 나와 800이면 절반 이상 잘렸다.
                r = cli.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000, reasoning_effort="low")
                return (r.choices[0].message.content or "",
                        r.usage.total_tokens if r.usage else 0)
            except RateLimitError as e:
                ra = _retry_after(e)
                if ra > 120:
                    # 분 단위(TPM/RPM)가 아니라 일일 한도(TPD/RPD). 리셋까지 자고 재시도.
                    # 시도 횟수를 소모하지 않으므로 그냥 틀어놓으면 다음날 이어서 돈다.
                    print(f"  일일 한도 도달. {ra / 3600:.1f}시간 대기 후 재개 "
                          f"({time.strftime('%m-%d %H:%M', time.localtime(time.time() + ra))})",
                          flush=True)
                    time.sleep(ra + 30)
                    continue
                wait = max(min(60, 2 ** attempt * 5), ra)
                print(f"  429, {wait:.0f}s 대기", flush=True)
                time.sleep(wait)
                attempt += 1
            except Exception as e:
                wait = min(60, 2 ** attempt * 5)
                print(f"  err, {wait}s 대기 ({e.__class__.__name__}: {str(e)[:80]})", flush=True)
                time.sleep(wait)
                attempt += 1
        raise RuntimeError("생성 실패")
    return call


def gen(in_path: str, out_path: str, call: Backend, model: str,
        sleep: float = 5.0, cache_path: str | os.PathLike = RUNS / "cache.jsonl",
        tpm: int = 8000) -> None:
    """호출마다 max(sleep, 이번 호출 토큰 / tpm * 60초) 만큼 쉰다.
    Groq 무료 TPM 8,000 기준 T1/T2(~650토큰)는 5초, T3(~1,400토큰)는 ~10초.
    240회에 약 30분. 일일 한도(TPD 200K)에 걸리면 다음날 같은 명령을 다시 실행하면
    캐시된 건은 건너뛰고 이어서 생성한다."""
    cache = Cache(str(cache_path))
    rows = [json.loads(l) for l in open(in_path, encoding="utf-8")]
    with open(out_path, "w", encoding="utf-8") as fo:
        for i, r in enumerate(rows, 1):
            k = hashlib.sha256(f"{model}||{r['gen_id']}||{r['prompt']}"
                               .encode()).hexdigest()[:24]
            v = cache.get(k)
            if v is None:
                v, used = call(r["prompt"], model)
                cache.put(k, v)
                time.sleep(max(sleep, used / tpm * 60) if sleep > 0 else 0)
            r["raw"] = v
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    print(f"gen: {len(rows)} -> {out_path}")


# =============================================================================
# 4. restore — 복원 (유출은 raw에서 측정하므로 둘 다 보관)
# =============================================================================

def restore_stage(in_path: str, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as fo:
        for l in open(in_path, encoding="utf-8"):
            r = json.loads(l)
            r["restored"], r["unresolved"] = restore_text(r["raw"], r["restore"])
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"restore -> {out_path}")


# =============================================================================
# 5. score
# =============================================================================

_JOSA = ("에서는", "으로는", "께서는", "이라고", "으로", "에서", "에게", "까지",
         "부터", "은", "는", "이", "가", "을", "를", "의", "에", "도", "로", "과", "와")


def norm_ko(s: str) -> str:
    s = re.sub(r"[^\w\s가-힣]", " ", s.strip().lower())
    out = []
    for t in s.split():
        for j in _JOSA:
            if t.endswith(j) and len(t) > len(j):
                t = t[:-len(j)]
                break
        out.append(t)
    return " ".join(out)


def containment(pred: str, golds: list[str]) -> float:
    """정답 문자열이 응답 안에 들어 있는가. 엄밀한 EM이 아님(이름 주의)."""
    if not golds:
        return float("nan")
    p = norm_ko(pred)
    return float(any(norm_ko(g) in p for g in golds))


def token_f1(pred: str, golds: list[str]) -> float:
    if not golds:
        return float("nan")
    pt, best = norm_ko(pred).split(), 0.0
    for g in golds:
        gt = norm_ko(g).split()
        if not pt or not gt:
            continue
        pool, c = list(pt), 0
        for t in gt:
            if t in pool:
                pool.remove(t); c += 1
        if c:
            pr, rc = c / len(pt), c / len(gt)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def slot_recall(pred: str, slots: list[str]) -> float:
    if not slots:
        return float("nan")
    return sum(1 for s in slots if re.search(s, pred)) / len(slots)


def slot_all(pred: str, slots: list[str]) -> float:
    """slot을 전부 찾았으면 1, 하나라도 빠지면 0.  T2 주 지표.
    containment는 '11월 20일(금), 송도 회의센터' 같은 정답 표기 변형을 놓치므로
    정보 조각(날짜·시각·장소 등)별 정규식이 모두 매치되는지로 정답을 판정한다."""
    r = slot_recall(pred, slots)
    return r if r != r else float(r == 1.0)


def copy_rate(pred: str, ctx: str) -> float:
    """지문 베끼기 탐지. slot recall이 복사에 보상을 주는 걸 견제."""
    pt = set(norm_ko(pred).split())
    return len(pt & set(norm_ko(ctx).split())) / max(len(pt), 1)


def score(in_path: str, items_path: str, out_path: str) -> None:
    items = {json.loads(l)["item_id"]: json.loads(l)
             for l in open(items_path, encoding="utf-8")}
    with open(out_path, "w", encoding="utf-8") as fo:
        for l in open(in_path, encoding="utf-8"):
            r = json.loads(l)
            it = items[r["item_id"]]
            mf = r["masked_forms"]
            r["metrics"] = {
                # 품질: 복원 후 응답으로
                "cont": containment(r["restored"], it["gold"]),
                "f1":   token_f1(r["restored"], it["gold"]),
                "slot": slot_recall(r["restored"], it["slots"]),
                "slot_all": slot_all(r["restored"], it["slots"]),
                "copy": copy_rate(r["restored"], it["context"]),
                # 유출: 반드시 raw 응답으로 (복원하면 전부 유출로 잡힘)
                "leak": (sum(1 for x in mf if x in r["raw"]) / len(mf)) if mf else float("nan"),
                # 복원 실패: R1(type)의 원리적 한계를 수치화
                "unresolved": r["unresolved"],
                "task": it["task_type"],
            }
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"score -> {out_path}")


# =============================================================================
# 6. analyze
# =============================================================================

def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return statistics.fmean(xs) if xs else float("nan")


def _paired_boot(pairs: list[tuple[float, float]], B: int = 5000, seed: int = 7):
    """항목별 쌍 차이의 부트스트랩 95% 신뢰구간.
    scipy 없이 순수 파이썬. 0을 포함하면 '차이 없음'."""
    d = [a - b for a, b in pairs if a == a and b == b]
    if len(d) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(_mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(B))
    return _mean(d), means[int(.025 * B)], means[int(.975 * B)]


def _primary(m: dict) -> float:
    """T1은 containment, T2는 slot 전부 매치(slot_all), T3는 slot recall을 주 지표로.
    (2026-09-17 변경: T2를 containment -> slot_all. 사전등록 v0.2 참고)"""
    if m["task"] == "T3":
        return m["slot"]
    if m["task"] == "T2":
        return m["slot_all"]
    return m["cont"]


def analyze(in_path: str) -> None:
    rows = [json.loads(l) for l in open(in_path, encoding="utf-8")]
    by = defaultdict(dict)                       # item_id -> arm -> row
    for r in rows:
        by[r["item_id"]][r["arm"]] = r

    # (a) 잡음 하한선
    nf = _paired_boot([(_primary(v["A0"]["metrics"]), _primary(v["A0p"]["metrics"]))
                       for v in by.values() if "A0" in v and "A0p" in v])
    print(f"\n[잡음 하한선] A0 vs A0' 평균차 {nf[0]:+.3f}  95%CI [{nf[1]:+.3f}, {nf[2]:+.3f}]")
    print("  CI가 0을 포함해야 정상. 하한선 폭 = 이 CI의 너비.\n")

    # (b) arm × task 표
    print(f"{'arm':<6}{'scope':<10}{'repr':<11}{'task':<6}"
          + "".join(f"{k:>9}" for k in ("주지표", "leak", "copy", "unres")) + f"{'n':>5}")
    print("-" * 74)
    buck = defaultdict(list)
    for r in rows:
        buck[(r["arm"], r["metrics"]["task"])].append(r)
    for (arm, task) in sorted(buck, key=lambda x: (ARMS.index(x[0]), x[1])):
        g = buck[(arm, task)]
        print(f"{arm:<6}{SCOPES[arm]:<10}{REPRS[arm]:<11}{task:<6}"
              f"{_mean([_primary(x['metrics']) for x in g]):>9.3f}"
              f"{_mean([x['metrics']['leak'] for x in g]):>9.3f}"
              f"{_mean([x['metrics']['copy'] for x in g]):>9.3f}"
              f"{_mean([x['metrics']['unresolved'] for x in g]):>9.2f}{len(g):>5}")

    # (c) 사전 지정 대비 3개 (나머지 쌍별 비교는 탐색용으로만)
    print("\n[사전 지정 대비]")
    for label, a, b in (("H1  A0 vs A1a", "A0", "A1a"),
                        ("H2  A0 vs A2a", "A0", "A2a"),
                        ("범위 A1a vs A2a", "A1a", "A2a")):
        pairs = [(_primary(v[a]["metrics"]), _primary(v[b]["metrics"]))
                 for v in by.values() if a in v and b in v]
        d, lo, hi = _paired_boot(pairs)
        sig = "유의" if not (lo <= 0 <= hi) else "차이 없음"
        print(f"  {label:<18} Δ={d:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  {sig}")

    # (d) 표현 3종 사다리 (정보 보존량 R1<R2<R3 -> 품질도 같은 순서여야 함)
    print("\n[표현 사다리 점검] 기대: type ≤ index ≤ pseudonym")
    for scope, arms in (("all", ("A1a", "A1b", "A1c")), ("ne", ("A2a", "A2b", "A2c"))):
        vals = [_mean([_primary(r["metrics"]) for r in rows if r["arm"] == x]) for x in arms]
        ok = "OK" if vals[0] <= vals[1] + 1e-9 <= vals[2] + 1e-9 else "위반 - 구현/측정 점검"
        print(f"  {scope:<4} type={vals[0]:.3f} index={vals[1]:.3f} "
              f"pseudo={vals[2]:.3f}  {ok}")
    print()


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)

    D, R = str(DATA), str(RUNS)
    p = s.add_parser("build");   p.add_argument("--threads", default=f"{D}/threads.json"); p.add_argument("--out", default=f"{R}/items.jsonl")
    p = s.add_parser("prompts"); p.add_argument("--in", dest="i", default=f"{R}/items.jsonl"); p.add_argument("--out", default=f"{R}/prompts.jsonl")
    p = s.add_parser("gen")
    p.add_argument("--in", dest="i", default=f"{R}/prompts.jsonl"); p.add_argument("--out", default=f"{R}/raw.jsonl")
    p.add_argument("--backend", choices=["echo", "groq"], default="echo")
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    p.add_argument("--api-key", default=os.environ.get("GROQ_API_KEY", ""))
    p.add_argument("--sleep", type=float, default=5.0, help="호출 간 최소 대기(초). 토큰 사용량에 따라 자동으로 늘어남")
    p.add_argument("--cache", default=f"{R}/cache.jsonl")
    p = s.add_parser("restore");  p.add_argument("--in", dest="i", default=f"{R}/raw.jsonl"); p.add_argument("--out", default=f"{R}/restored.jsonl")
    p = s.add_parser("score")
    p.add_argument("--in", dest="i", default=f"{R}/restored.jsonl"); p.add_argument("--items", default=f"{R}/items.jsonl"); p.add_argument("--out", default=f"{R}/scored.jsonl")
    p = s.add_parser("analyze"); p.add_argument("--in", dest="i", default=f"{R}/scored.jsonl")
    s.add_parser("demo")

    a = ap.parse_args()
    if a.cmd == "build":
        build(a.threads, a.out)
    elif a.cmd == "prompts":
        prompts(a.i, a.out)
    elif a.cmd == "gen":
        call = backend_echo if a.backend == "echo" else backend_groq(a.api_key, a.base_url)
        gen(a.i, a.out, call, a.model, 0.0 if a.backend == "echo" else a.sleep, a.cache)
    elif a.cmd == "restore":
        restore_stage(a.i, a.out)
    elif a.cmd == "score":
        score(a.i, a.items, a.out)
    elif a.cmd == "analyze":
        analyze(a.i)
    elif a.cmd == "demo":
        build(f"{D}/threads.json", f"{R}/items.jsonl")
        prompts(f"{R}/items.jsonl", f"{R}/prompts.jsonl")
        # 실제 실행 산출물(raw/restored/scored)을 덮어쓰지 않도록 demo_ 접두사를 쓴다
        gen(f"{R}/prompts.jsonl", f"{R}/demo_raw.jsonl", backend_echo, "echo", 0.0, f"{R}/demo_cache.jsonl")
        restore_stage(f"{R}/demo_raw.jsonl", f"{R}/demo_restored.jsonl")
        score(f"{R}/demo_restored.jsonl", f"{R}/items.jsonl", f"{R}/demo_scored.jsonl")
        analyze(f"{R}/demo_scored.jsonl")


if __name__ == "__main__":
    main()

# PII 범위 마스킹 파일럿

프롬프트 속 PII를 "질문에 필요한 것만 남기고" 가렸을 때 LLM 응답 품질이 유지되는지 보는 파일럿 실험.
설계·가설·판정 규칙은 [docs/preregistration.md](docs/preregistration.md)에 고정되어 있다.

## 구조

```
src/        pilot.py(파이프라인)  analysis_extra.py(추가 분석)  verify.py(데이터 정합성 검사)
data/       threads.json(스레드 10개)  archive/(이전 버전)
docs/       preregistration.md
rating/     compare.html(T3 블라인드 짝비교 도구)  rater_*.json(평가자 결과, 여기에 저장)
runs/       생성물 전부. git 제외. build부터 다시 만들 수 있음
```

## 준비

```bash
pip install -r requirements.txt
echo 'GROQ_API_KEY=gsk_...' > .env        # 또는 환경변수
```

## 실행

어느 디렉터리에서 실행해도 `data/`, `runs/` 기준으로 동작한다.

```bash
python src/pilot.py build          # data/threads.json -> runs/items.jsonl
python src/pilot.py prompts        # runs/items.jsonl  -> runs/prompts.jsonl (240건)
python src/verify.py               # 정합성 검사 + runs/t2_visual_check.txt

python src/pilot.py gen --backend groq       # runs/prompts.jsonl -> runs/raw.jsonl (약 30분)
python src/pilot.py restore
python src/pilot.py score
python src/pilot.py analyze
python src/analysis_extra.py --human rating/rater_a.json rating/rater_b.json rating/rater_c.json
```

배선 확인만 하려면 `python src/pilot.py demo` (echo 백엔드, API 호출 없음).

### gen 메모 (Groq 무료 티어)

- 한도: 30 RPM / 8K TPM / 1K RPD / **200K TPD**. 240건 전체가 약 210K 토큰이라 하루에 못 끝날 수 있다.
  일일 한도에 걸리면 스크립트가 리셋 시각까지 자동으로 기다렸다가 이어간다. 그냥 틀어놓으면 된다.
  ```bash
  nohup python src/pilot.py gen --backend groq > runs/gen.log 2>&1 &
  tail -f runs/gen.log
  ```
  중간에 죽어도 다시 실행하면 캐시(`runs/cache.jsonl`)된 건은 건너뛰고 이어간다.
- 호출 간 대기는 직전 호출의 토큰 사용량에 맞춰 자동으로 늘어난다 (`--sleep`은 최소값).
- 일부만 돌리려면 `--in`/`--out`으로 별도 파일을 지정한다.
  ```bash
  python src/pilot.py gen --backend groq --in runs/smoke.jsonl --out runs/smoke_raw.jsonl
  ```

# regime-alloc

레짐 인식 포트폴리오 배분 연구 코드베이스. **STABLE** (Soun et al., ICLR 2026)을 baseline으로 재현하고, 그 위에 새 방법(proposed)을 얹는다. 모든 method가 같은 데이터·split·백테스터·지표를 공유한다.

## 문서
| 문서 | 용도 |
|---|---|
| `README.md` | 개요, 설치·실행·모니터링 방법, 코드 구조 |
| `docs/NAMING.md` | 명명 규칙과 저장 구조 (설정·데이터·결과 파일이 어디에 어떤 이름으로 저장되는지, git 규칙) |
| `docs/EXPERIMENTS.md` | 실험 기록 (목적, 설정, 커밋, 결과 위치, 결과 요약) |
| `docs/DECISIONS.md` | 논문→코드 대응, 논문에 없는 값의 선택과 근거, 범위, 알려진 문제 |
| `docs/CHANGELOG.md` | 커밋 단위 변경 이력 |

## 설치
```bash
/usr/local/bin/python3.11 -m venv .venv
.venv/bin/pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt     # torch 줄 제외
```

## 실행
```bash
.venv/bin/python scripts/download.py                                         # yfinance → data/raw/
.venv/bin/python scripts/run_experiment.py configs/experiments/paper_table3_4.yaml
.venv/bin/python scripts/run_method.py --market sp500 --method configs/methods/stable.yaml --exp X \
    [--set '{"train":{"epochs":1}}'] [--skip-train]                          # 작업 하나만
.venv/bin/python -m pytest
```
`run_experiment.py`는 (시장 × method)마다 subprocess를 띄운다. STABLE은 GPU를 하나씩 나눠 쓰고, 기본 GPU 목록은 `0,1,2,3,4,5,7`이다(이 서버의 GPU 6은 고장).

## 모니터링
```bash
.venv/bin/python scripts/status.py                    # 모든 실험의 상태를 한 번 출력
.venv/bin/python scripts/status.py paper_table3_4 --watch 60    # 60초마다 갱신
tail -f results/<exp>/logs/<market>_<method>.log      # 작업별 실시간 로그
```
`status.py`는 작업마다 상태(pending / training / evaluating / done / FAILED / stalled?), epoch 진행률, ETA, 최근 loss, 완료된 작업의 지표를 보여준다.

## 코드 구조
```
configs/          data/ (유니버스) · splits/ · methods/ · experiments/
regime_alloc/
  data/           yfinance 다운로드, 유니버스 필터, MarketData 번들 (dataset.py)
  features/       거시 m_τ, Kalman β_τ (종목 임베딩)
  diffusion/      noise schedule, DDIM 샘플러
  nn/             조건부 1-D UNet
  portfolio/      Black–Litterman + max-Sharpe MVO
  eval/           백테스트 (모든 method 공통), 지표 (ASR/RMDD/AVol/MSE/DTW)
  methods/        base.py (인터페이스) · registry.py · stable/ · classic/ (CRP, MVO, MOM) · proposed/
scripts/          download · run_method · run_experiment · status
tests/            BL / DDIM / Kalman / 지표 / 모델 sanity test
docs/             기록용 문서 (위 표)
```
**새 method 추가:** `methods/base.Method`를 상속해 `fit`, `load`, `allocate`(필요하면 `forecast`도)를 구현한다. 그다음 `methods/registry.py`에 등록하고 `configs/methods/`에 설정을 추가한다. method는 `MarketData`와 리밸런싱 시점 τ만 받으며, τ 이후 행을 읽으면 안 된다.

**누수 방지:**
- 달력은 `test_end`에서 끝난다.
- 정규화와 Kalman 필터는 인과적(causal)이다.
- 수익률 스케일은 학습구간에서만 구한다.
- 학습 타깃은 `train_end`를 넘지 않는다.
- 비미국 시장에서는 미국 거시지표를 1일 늦춰 쓴다.

**재현성:** 시드 0에 cuDNN/cuBLAS 결정적 모드를 켜서, 다시 돌려도 결과가 bit 단위로 같다.

## 현황
- 완료: 데이터 파이프라인, STABLE, CRP/MVO/MOM, smoke test, 단위 테스트, 모니터링
- 진행 중: `paper_table3_4` 본 학습 (`docs/EXPERIMENTS.md`)
- 예정: 딥러닝 baseline, proposed method

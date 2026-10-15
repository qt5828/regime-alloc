# 명명 규칙 · 저장 구조

새 파일·실험·결과를 만들 때 이 규칙을 따른다. 규칙을 바꾸면 이 문서를 먼저 고친다.

## 설정 파일 (`configs/`)
| 종류 | 경로 | 이름 규칙 | 예 |
|---|---|---|---|
| 유니버스 | `configs/data/universes.yaml` | 시장 키는 소문자 (`sp500`, `csi300`, `eurostoxx`, `kospi200`) | |
| split | `configs/splits/<split>.yaml` | `<용도>_<구분>`, 파일 안의 `name`과 파일명을 같게 | `paper_main.yaml` |
| method | `configs/methods/<method>.yaml` | registry 키와 같게. 여러 method를 한 파일에 둘 때는 `파일:키`로 참조 | `stable.yaml`, `classic.yaml:crp` |
| 실험 | `configs/experiments/<exp>.yaml` | 파일 안의 `name` = 파일명 = 결과 폴더명 | `paper_table3_4.yaml` |

- 실험 이름: `<목적>_<대상>`. 논문 재현은 `paper_*`, 새 방법은 `prop_*`, ablation은 `abl_*`, 빠른 점검은 `smoke`.
- 하이퍼파라미터 주석: `[paper]` 논문 명시, `[ref:X]` 인용 논문 X에서 가져옴, `[assume]` 직접 정함(→ `docs/DECISIONS.md`에 근거 기록).

## 데이터 (`data/`, git 제외)
| 경로 | 내용 |
|---|---|
| `data/raw/macro.parquet` | 거시 5종 원가격 (열: dollar, tnx, irx, vix, gold) |
| `data/raw/<market>_prices.parquet` | 시장지수 대용 + 종목 adjusted close, 열 = 티커 |
| `data/processed/<market>_<split>_nu<ν>_l<ℓ>.pkl` | `MarketData` 캐시. 특징 코드를 바꾸면 이 파일을 지워야 다시 생성됨 |

## 결과 (`results/`, git 제외)
```
results/<exp>/
  summary.md                       실험 전체 요약표 (논문 수치 병기)
  logs/<market>_<label>.log        작업별 stdout/stderr (실시간)
  <market>/<label>/
    config.json                    실제 사용된 method·split 설정, 종목 리스트, 제외 종목
    train_log.jsonl                epoch별 loss / mse / gate 평균 / 경과 시간 (학습형 method만)
    model.pt                       net + EMA 가중치 (학습형 method만)
    metrics.json                   portfolio(ASR, RMDD, AVol) + forecast(MSE, DTW)
    daily_returns.csv              테스트 구간 일간 포트폴리오 수익률
    weights.csv                    리밸런싱 시점(행) × 종목(열) 가중치
```
- `<label>` = method 이름. ablation 플래그가 꺼지면 접미사가 붙음: `stable-no_kfe`, `stable-no_mlg_bl`.
- 같은 실험 이름으로 다시 돌리면 덮어쓴다. 결과를 보존하려면 실험 이름을 새로 정한다(예: `paper_table3_4_seed1`).

## 코드
- 공통 인프라는 `regime_alloc/<영역>/`, method 구현은 `regime_alloc/methods/<method>/`.
- 새 method: 폴더명 = registry 키 = config의 `method` 값.
- 스크립트: 동사형 (`download.py`, `run_method.py`, `run_experiment.py`, `status.py`).

## Git
- 브랜치: `main`. 실험용 큰 변경은 `exp/<이름>`, 기능은 `feat/<이름>`.
- 커밋 메시지: 첫 줄에 영어 명령형 요약(≤ 60자). 필요하면 빈 줄 뒤에 `- ` 목록으로 핵심만 적는다.
- 커밋·push 전에 사용자 확인을 받는다. 커밋할 때마다 `docs/CHANGELOG.md`에 한 줄 추가한다.

# 실험 기록

실험을 돌릴 때마다 위에 새 항목을 추가한다. 결과 파일 구조는 `docs/NAMING.md`를 따른다.

형식:
```
## <exp> — <날짜>
- 목적 / 설정 / 커밋 / 명령
- 결과 위치
- 요약 / 메모
```

---

## paper_table3_4 — 2026-10-15 (진행 중)
- **목적:** 논문 Table 3·4 재현 (STABLE + CRP/MVO/MOM, 4개 시장)
- **설정:** `configs/experiments/paper_table3_4.yaml`, split `paper_main`, STABLE 200 epoch, k = 50
- **커밋:** `1abfa14` (GPU 결정적 모드 추가 전에 시작해서 bit 단위 재현은 보장 안 됨)
- **명령:** `python scripts/run_experiment.py configs/experiments/paper_table3_4.yaml`
- **결과 위치:** `results/paper_table3_4/` (`summary.md`, `<market>/<method>/metrics.json` …)
- **현황:** CRP/MVO/MOM 12개 완료. STABLE은 학습 중 (`python scripts/status.py paper_table3_4`)

| 시장 | CRP ASR | MVO ASR | MOM ASR | 논문 CRP / MVO / MOM |
|---|---|---|---|---|
| sp500 | 0.77 | 0.38 | -0.30 | 0.82 / 1.18 / 0.03 |
| csi300 | -0.54 | -0.16 | -0.52 | -0.70 / -0.66 / -0.47 |
| eurostoxx | 0.88 | 2.95 | 1.59 | 1.31 / 0.48 / 1.42 |
| kospi200 | -0.13 | 0.31 | 0.32 | 0.76 / 0.45 / 0.33 |

## smoke — 2026-10-15
- **목적:** 파이프라인 end-to-end 점검
- **설정:** `paper_table3_4`와 동일하되 `--set '{"train":{"epochs":1},"diffusion":{"n_paths":5}}'`
- **결과 위치:** `results/smoke/`
- **메모:** 모든 작업 정상 종료. STABLE은 1 epoch, k = 5라 BL 가중치가 폭주함 (EUROSTOXX AVol 1269%). k ≤ S에서 쓰는 pinv Ω 문제로, `docs/DECISIONS.md`의 "알려진 문제" 참고.

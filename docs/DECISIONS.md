# 설계 결정 · 가정

논문에 없는 값을 어떻게 채웠는지, 그리고 그 근거를 기록한다. 결정을 바꾸면 기존 줄은 지우지 말고 `~~취소선~~`과 날짜를 남긴다.

## 논문 → 코드 대응
| 논문 | 코드 |
|---|---|
| m_τ (Table 1): 시장지수, DXY, 10y−3m, VIX, 금 / ν-rolling z-score + 로그차분 | `features/macro.py` (d_m = 10) |
| β_τ^(s) Kalman 임베딩 (Das & Ghoshal 2010) | `features/kalman_beta.py` |
| c_τ^(s) = [β, 정규화 종가, 일간 로그수익률] | `methods/stable/method.py::StableFeatures` |
| CDG + MLG: ε̂ = ε̂_n + z(ε̂_n^(s) − ε̂_n), z = g_π(h_f) ∈ [0, z_max], Eq.2 손실 | `methods/stable/model.py` |
| DDIM (§3.3, A.2) | `diffusion/ddim.py` |
| view (Eq.3–4), BL 사후분포 (Eq.6–7), w* (Eq.8), μ_prior = Σ_prior w_eq (A.3) | `portfolio/black_litterman.py` |
| ablation (A.7) | `configs/methods/stable.yaml`의 `ablation.use_kfe / use_mlg / use_bl` |

## 논문에 없는 값
| 항목 | 선택 | 근거 |
|---|---|---|
| 데이터 소스 | yfinance adjusted close, 2012-01-01 ~ 2026-10-15 수집 | 사용자 결정 (2026-10-15) |
| CSI300 / KOSPI200 지수 | 510300.SS / 069500.KS ETF (yfinance에 000300.SS, ^KS200 이력 없음) | 가정 |
| 3개월 금리 | ^IRX (13주 T-bill) | 가정 |
| 유니버스 | GICS 섹터별 시총 상위(2025-03 기준) 수작업 선정, 전 구간 이력 없으면 제외 → S&P 55, CSI 55, EURO 42(논문 37), KOSPI 39(논문 44) | 가정 |
| 미국 거시지표 시차 | 비미국 시장은 1일 lag (아시아·유럽 장이 미국보다 먼저 마감) | 가정 (누수 방지) |
| noise schedule | linear 1e-4→0.02, 1000/T 배율 | Improved-DDPM |
| UNet | lucidrains `Unet1D`, attention 없음, dim 64, mults (1,2,4) | ref |
| optimizer | Adam lr 2e-4, batch 128, grad-clip 1 | DDPM |
| EMA | 0.995 | Diffusion-TS |
| Kalman Q, R | Q = δ/(1−δ)I, δ = 1e-4 / R은 120일 innovation으로 적응 추정 | Chan / Das & Ghoshal |
| c_τ의 과거 수익률 길이 | ℓ = 20일, 종목별 학습구간 std로 스케일 | 가정 |
| prior 공분산 | ν일 일간 공분산 × ℓ (view의 ℓ일 누적수익률과 맞춤) | 가정 |
| k ≤ S일 때 Ω | Σ_view가 특이행렬 → pseudo-inverse | 가정 |
| 백테스트 | ℓ일마다 리밸런싱, 그 사이 buy-and-hold, 거래비용 0 | 가정 |
| DTW | \|·\| 비용, ℓ로 정규화 | 가정 |
| MVO / MOM baseline | long-only max-Sharpe 120일 / 120일 모멘텀 상위 1/3 동일가중 | Markowitz / Jegadeesh–Titman |
| 재현성 | 시드 0 + cuDNN/cuBLAS 결정적 모드 | 사용자 요청 (2026-10-15) |

## 범위
- 하지 않음: Appendix A.4–A.6 추가 실험 (사용자 결정, 2026-10-15)
- 나중에: 딥러닝 baseline (DeepTrader, MetaTrader, AlphaMix, Diffusion-TS, AEC-GAN, KoVAE)

## 논문과 다른 점 (재현 한계)
| 항목 | 논문 | 이 코드 | 영향 |
|---|---|---|---|
| 종목 수 | 55 / 55 / 37 / 44 | 55 / 55 / 42 / 39 | 유니버스 구성 차이 → 모든 수치에 영향 |
| 종목 리스트 | 비공개 | GICS 시총 상위 수작업 선정 | 위와 같음 |
| CSI300·KOSPI200 지수 | 지수 자체로 추정 | ETF 대용 (510300.SS, 069500.KS) | m_τ 중 시장지수 항목 |
| 데이터 소스 | 비공개 | yfinance | 수정주가 계산 방식이 다를 수 있음 |
| 하드웨어 | 4×RTX 3080, 모델 1개를 4장으로 학습 | 시장별 A6000 1장 | 학습 시간만 다르고 결과와는 무관 |
| 학습 세부 (optimizer, lr, batch, EMA, UNet 폭, schedule) | 비공개 | 인용 논문 기본값 (위 표) | 예측 품질에 영향 가능 |
| Kalman β 세부 (Q, R, 입력) | 비공개 | Das & Ghoshal식 적응 R, δ = 1e-4 | 임베딩 품질 |
| view 경로의 종목 간 노이즈 | 비공개 | 종목마다 독립 노이즈 | BL 레버리지에 큰 영향 (로컬 이슈 문서 참고) |
| Σ_view 역행렬 (k ≤ S) | 언급 없음 | pinv | 위와 같음 |
| DTW 정의 | "normalized DTW" | \|·\| 비용 / ℓ | DTW 절대값이 다를 수 있음 |
| 백테스트 세부 | 비공개 | ℓ일 리밸런싱, buy-and-hold, 비용 0 | 포트폴리오 지표 |
| Appendix A.4–A.6 실험 | 있음 | 하지 않음 (사용자 결정) | — |
| 딥러닝 baseline 6종 | 있음 | 아직 없음 | Table 3·4 일부 행 없음 |

열린 이슈와 실험 결과는 로컬 전용 문서(`docs/private/`, git 제외)에 기록한다.

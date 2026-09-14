# 📖 《실무로 통하는 인과추론 with 파이썬》 스터디 & 세미나 발표 자료

- **도서명**: 실무로 통하는 인과추론 with 파이썬 (Causal Inference for The Brave and True)
- **저자**: 마테우스 파치우 (Matheus Facure)
- **발표자**: 이준형 (Junhyeong Lee)
- **진행 기간**: **2025.12 ~ 2026.02**
- **목적**: 상관관계가 아닌 인과관계(Causality)를 데이터와 머신러닝을 통해 규명하고, 의사결정 및 정책 평가에 적용하는 인과추론 프레임워크 학습

---

## 📑 챕터별 세미나 발표 자료 (PDF)

| Chapter | Topic | Presentation Slide (PDF) | Key Concepts |
|:---:|:---|:---:|:---|
| **01** | **인과추론 입문 (Introduction)** | [📄 Ch 01 Slide](./01_Introduction_to_Causal_Inference.pdf) | 상관관계 vs 인과관계, 잠재적 결과(Potential Outcomes), 개입(Treatment)과 효과 |
| **02** | **무작위 통제 시험 (Randomized Experiments)** | [📄 Ch 02 Slide](./02_Randomized_Experiments.pdf) | A/B 테스팅, 이상적인 실험 설계, ATE(평균 처치 효과), 교란 요인 제거 |
| **03** | **그래프 인과 모델 (Graphical Causal Models)** | [📄 Ch 03 Slide](./03_Graphical_Causal_Models.pdf) | DAG(Directed Acyclic Graphs), 체인(Chain), 포크(Fork), 콜라이더(Collider) |
| **04** | **그래프 모델과 편향 (Bias & Confounding)** | [📄 Ch 04 Slide](./04_Graphical_Models_and_Bias.pdf) | 백도어 기준(Backdoor Criterion), 교란 편향(Confounding Bias), 선택 편향(Selection Bias) |
| **05** | **매칭과 서브클래스화 (Matching & Subclassification)** | [📄 Ch 05 Slide](./05_Matching_and_Subclassification.pdf) | 성향 점수 매칭(PSM), 공변량 균형(Covariate Balance), 차원의 저주 해결 |
| **06** | **선형 회귀와 인과추론 (Linear Regression & Causality)** | [📄 Ch 06 Slide](./06_Linear_Regression_in_Causal_Inference.pdf) | OLS 회귀를 통한 조건부 효과 추정, Frisch-Waugh-Lovell 정리를 통한 직교화 |

---

## 💡 핵심 학습 내용 요약

1. **Potential Outcomes Framework**:
   - $Y_i(1) - Y_i(0)$를 통해 개별 처치 효과(ITE)를 정의하고, 관측 불가능한 반사실(Counterfactual)을 추정하기 위한 방법론 정립.
2. **Causal DAGs & d-Separation**:
   - 인과 그래프를 통해 통제해야 하는 변수(Confounder)와 통제하면 안 되는 변수(Collider / Mediator)를 명확히 식별.
3. **Quasi-Experiments & Machine Learning Integration**:
   - A/B 테스트가 불가능한 환경에서 매칭, 회귀 분석, 성향 점수를 통해 인과 효과를 엄밀하게 추정하는 실무 역량 배양.

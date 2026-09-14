# 📚 AI & Vision Research & Seminar Archive

이 저장소는 **Computer Vision, Vision-Language Models (VLM), Deep Learning** 최신 연구 논문과 **인과추론(Causal Inference) & 머신러닝** 세미나에서 발표한 **슬라이드(PDF) 및 요약 노트**를 체계적으로 아카이빙하는 공간입니다.

<p align="left">
  <img src="https://img.shields.io/badge/Author-Junhyeong%20Lee-0969da?style=flat-square&logo=github" alt="Author" />
  <img src="https://img.shields.io/badge/Domain-Computer%20Vision%20%7C%20VLM%20%7C%20Causality-10b981?style=flat-square" alt="Domain" />
  <img src="https://img.shields.io/badge/Status-Actively%20Updating-f59e0b?style=flat-square" alt="Status" />
</p>

---

## 📑 1. Research Paper Reviews (논문 리뷰)

| Category | Date | Paper Title | Venue / Year | Review Slide (PDF) | Summary Note |
|:---:|:---:|:---|:---:|:---:|:---:|
| 🤖 **VLM** | 2026.09 | **Visual Instruction Tuning (LLaVA)** | NeurIPS 2023 (Oral) | [📄 Slide 보기](./Vision-Language/LLaVA/LLaVA_Paper_Review_LeeJunHyeong.pdf) | [📝 Review Note](./Vision-Language/LLaVA/README.md) |

---

## 📖 2. Book & Theory Seminars (도서 및 이론 세미나)

### 📊 《실무로 통하는 인과추론 with 파이썬》 스터디
> **Causal Inference for The Brave and True** — 세미나 발표 슬라이드 모음 ([📝 세미나 상세 페이지](./Book-Seminars/Causal-Inference-Python/README.md))

| Chapter | Topic | Presentation Slide (PDF) | Key Takeaway |
|:---:|:---|:---:|:---|
| **Ch 01** | **인과추론 입문** | [📄 Ch 01 Slide](./Book-Seminars/Causal-Inference-Python/01_Introduction_to_Causal_Inference.pdf) | 상관관계 vs 인과관계, 잠재적 결과(Potential Outcomes) 프레임워크 |
| **Ch 02** | **무작위 통제 시험** | [📄 Ch 02 Slide](./Book-Seminars/Causal-Inference-Python/02_Randomized_Experiments.pdf) | A/B 테스트 원리와 ATE(평균 처치 효과) 추정 |
| **Ch 03** | **그래프 인과 모델** | [📄 Ch 03 Slide](./Book-Seminars/Causal-Inference-Python/03_Graphical_Causal_Models.pdf) | DAG(Directed Acyclic Graphs), 체인/포크/콜라이더 구조 |
| **Ch 04** | **그래프 모델과 편향** | [📄 Ch 04 Slide](./Book-Seminars/Causal-Inference-Python/04_Graphical_Models_and_Bias.pdf) | 백도어 기준(Backdoor Criterion), 교란 편향 및 선택 편향 제거 |
| **Ch 05** | **매칭과 서브클래스화** | [📄 Ch 05 Slide](./Book-Seminars/Causal-Inference-Python/05_Matching_and_Subclassification.pdf) | 성향 점수 매칭(PSM) 및 공변량 균형 검증 |
| **Ch 06** | **선형 회귀와 인과추론** | [📄 Ch 06 Slide](./Book-Seminars/Causal-Inference-Python/06_Linear_Regression_in_Causal_Inference.pdf) | OLS 회귀를 활용한 조건부 인과 효과 추정 및 FWL 정리 |

---

## 📂 Repository Structure

```text
Paper-Review/ (AI-Study-Archive)
├── README.md                                  # 전체 논문 리뷰 & 세미나 인덱스 허브
│
├── Vision-Language/                           # [논문] Vision-Language Models (VLM)
│   └── LLaVA/
│       ├── README.md                          # LLaVA 논문 분석 노트
│       └── LLaVA_Paper_Review_LeeJunHyeong.pdf # 발표 슬라이드 전문 (웹 뷰어 지원)
│
└── Book-Seminars/                             # [세미나] 도서 및 이론 스터디
    └── Causal-Inference-Python/               # 《실무로 통하는 인과추론 with 파이썬》
        ├── README.md                          # 인과추론 챕터별 정리 노트
        ├── 01_Introduction_to_Causal_Inference.pdf
        ├── 02_Randomized_Experiments.pdf
        ├── 03_Graphical_Causal_Models.pdf
        ├── 04_Graphical_Models_and_Bias.pdf
        ├── 05_Matching_and_Subclassification.pdf
        └── 06_Linear_Regression_in_Causal_Inference.pdf
```

---

## 👤 Author

- **이준형 (Junhyeong Lee)**
- **GitHub**: [@leejunhyeong-02](https://github.com/leejunhyeong-02)
- **Interest**: Computer Vision, Vision-Language-Action (VLA), Causal Inference, Multimodal Deep Learning

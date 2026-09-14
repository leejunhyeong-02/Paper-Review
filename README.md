# 📚 Paper Review & Research Archive

이 저장소는 **Computer Vision, Vision-Language Models (VLM), Deep Learning** 관련 핵심 연구 논문들을 분석하고 세미나/연구실에서 발표한 **슬라이드(PDF) 및 요약 노트**를 체계적으로 아카이빙하는 공간입니다.

<p align="left">
  <img src="https://img.shields.io/badge/Author-Junhyeong%20Lee-0969da?style=flat-square&logo=github" alt="Author" />
  <img src="https://img.shields.io/badge/Domain-Computer%20Vision%20%7C%20VLM-10b981?style=flat-square" alt="Domain" />
  <img src="https://img.shields.io/badge/Status-Actively%20Updating-f59e0b?style=flat-square" alt="Status" />
</p>

---

## 📑 Reviewed Papers Catalog

| Domain | Date | Paper Title | Venue / Year | Review Slide (PDF) | Summary & Notes |
|:---:|:---:|:---|:---:|:---:|:---:|
| 🤖 **VLM** | 2026.09 | **Visual Instruction Tuning (LLaVA)** | NeurIPS 2023 (Oral) | [📄 Slide 보기](./Vision-Language/LLaVA/LLaVA_Paper_Review_LeeJunHyeong.pdf) | [📝 Review Note](./Vision-Language/LLaVA/README.md) |

---

## 📂 Repository Structure

```text
Paper-Review/
├── README.md                               # 전체 논문 리뷰 인덱스 및 소개
├── Vision-Language/                        # Vision-Language Models (VLM) & Multimodal
│   └── LLaVA/
│       ├── README.md                       # LLaVA 논문 핵심 요약 노트
│       └── LLaVA_Paper_Review_LeeJunHyeong.pdf # 발표 슬라이드 전문 (웹에서 바로 열람 가능)
├── Object-Detection/                       # Object Detection & Pose Estimation (업데이트 예정)
└── Autonomous-Driving/                     # 자율주행 및 인지/제어 (업데이트 예정)
```

---

## 💡 Highlight: Visual Instruction Tuning (LLaVA)

> **"Large Language and Vision Assistant"** — NeurIPS 2023  
> *Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee*

- **핵심 아이디어**:
  1. **Visual Instruction Dataset 구축**: GPT-4를 활용하여 이미지-텍스트 쌍(Caption, Bounding Box)을 대화형(Conversation), 상세 묘사(Detailed Description), 복합 추론(Complex Reasoning) 형태의 멀티모달 인스트럭션 데이터로 확장.
  2. **간결하고 효율적인 아키텍처**: 사전 학습된 **CLIP ViT-L/14** 비전 인코더와 **Vicuna LLM**을 가벼운 선형 투영 레이어(Projection Matrix $W$)로 연결.
  3. **2단계 훈련 파이프라인 (Two-Stage Training)**:
     - **Stage 1 (Feature Alignment)**: Vision Encoder와 LLM을 Freeze하고 프로젝션 레이어만 학습 (CC3M 기반 595K 필터링 데이터).
     - **Stage 2 (End-to-End Fine-Tuning)**: LLM과 프로젝션 레이어를 함께 파인튜닝하여 멀티모달 대화 능력 극대화.
- **발표 슬라이드**: [LLaVA 발표자료 PDF 바로가기](./Vision-Language/LLaVA/LLaVA_Paper_Review_LeeJunHyeong.pdf)

---

## 👤 Author

- **이준형 (Junhyeong Lee)**
- **GitHub**: [@leejunhyeong-02](https://github.com/leejunhyeong-02)
- **Interest**: Computer Vision, Vision-Language-Action (VLA), Multimodal Deep Learning, Autonomous Systems

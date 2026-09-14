# Visual Instruction Tuning (LLaVA)

- **Authors**: Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
- **Affiliation**: University of Wisconsin-Madison, Microsoft Research, Columbia University
- **Publication**: **NeurIPS 2023 (Oral Presentation)**
- **Paper Link**: [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) / [Project Page](https://llava-vl.github.io/)
- **Review Slide**: [📄 LLaVA_Paper_Review_LeeJunHyeong.pdf](./LLaVA_Paper_Review_LeeJunHyeong.pdf)

---

## 📌 1. Motivation & Background

- 기존의 대형 언어 모델(LLM)은 뛰어난 언어 이해 및 인스트럭션 추종(Instruction-following) 능력을 보여주었으나, 텍스트 모달리티에 한정되어 있었습니다.
- 기존 멀티모달 모델들은 비전-언어 정렬(Alignment)에는 능하지만, 사람의 복잡한 멀티모달 지시를 이해하고 대화하는 능력(Visual Instruction Following)은 부족했습니다.
- **LLaVA의 목표**: GPT-4를 활용해 멀티모달 인스트럭션 데이터를 자동 생성하고, 오픈소스 LLM(Vicuna)과 비전 인코더(CLIP)를 효율적으로 결합하여 강력한 범용 비전-언어 어시스턴트를 구축하는 것.

---

## 🏗️ 2. Core Architecture

LLaVA는 모델 구조의 복잡성을 최소화하고 가벼운 투영 레이어로 비전과 언어를 연결합니다:

```
[ Input Image ] ──▶ [ Vision Encoder (CLIP ViT-L/14) ] ──▶ [ Linear Projection Layer (W) ] ──┐
                                                                                             ├──▶ [ LLM (Vicuna) ] ──▶ [ Response ]
[ Input Text  ] ──────────────────────────▶ [ Tokenizer / Word Embedding ] ──────────────────┘
```

1. **Vision Encoder**: Pretrained `CLIP ViT-L/14` (이미지 입력 $X_v$에 대해 visual grid feature $Z_v$ 추출)
2. **Projection Layer**: 가벼운 학습 가능한 선형 투영 행렬 $W$를 통해 visual feature를 텍스트 임베딩 공간으로 매핑 ($H_v = W \cdot Z_v$)
3. **Language Model**: Pretrained `Vicuna` (Llama 기반)를 활용하여 언어와 이미지 토큰을 순차적으로 받아 응답 생성

---

## 🚀 3. Visual Instruction Data Generation (GPT-4 활용)

이미지만 보고 직접 대화 데이터를 만들 수 없었던 순수 텍스트 GPT-4를 활용하기 위해, 이미지를 **Symbolic Representation(Bounding box, Caption)**으로 변환하여 프롬프트로 제공했습니다.

1. **Conversation**: 이미지 속 객체, 색상, 위치에 대한 자연스러운 질의응답 (58K)
2. **Detailed Description**: 이미지의 전체적인 장면과 세부 사항에 대한 심층 묘사 (23K)
3. **Complex Reasoning**: 인과 관계, 배경 지식, 맥락 추론을 요구하는 고난도 질문 (77K)
- **총 158K 고품질 멀티모달 인스트럭션 튜닝 데이터셋 구축**

---

## ⚙️ 4. Two-Stage Training Paradigm

### Stage 1: Pre-training for Feature Alignment
- **데이터**: CC3M 필터링 595K Image-Text pairs (단순 대화 형태 변환)
- **학습 파라미터**: **Projection Layer $W$만 학습**, Vision Encoder와 LLM은 Freeze
- **목적**: 비전 특징 벡터를 LLM의 단어 임베딩 공간과 정렬

### Stage 2: Fine-Tuning End-to-End
- **데이터**: 158K Visual Instruction Dataset + ScienceQA (13K)
- **학습 파라미터**: **Projection Layer $W$ + LLM (Vicuna)** 동시 학습 (Vision Encoder만 Freeze)
- **목적**: 복잡한 멀티모달 대화, 추론 및 질의응답 능력 극대화

---

## 📊 5. Key Results & Takeaways

- **ScienceQA SOTA**: ScienceQA 벤치마크에서 **92.53% Accuracy** 달성 (당시 SOTA였던 GPT-4(82.69%) 및 LLaMA-Adapter 대비 월등한 성능).
- **Multimodal Chat Performance**: GPT-4 대비 85.1% 수준의 멀티모달 대화 및 복합 추론 성능 달성.
- **의의**: 복잡한 멀티모달 모델 구조 없이도 가벼운 Projection과 고품질 Instruction Tuning만으로 강력한 Vision-Language Assistant를 구현할 수 있음을 입증.

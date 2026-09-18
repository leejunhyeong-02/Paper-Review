# SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning

- **Authors**: Jian Zhang*, Shijie Zhou*, Bangya Liu*, Achuta Kadambi, Zhiwen Fan (*Equal Contribution)
- **Affiliation**: University of California, Los Angeles (UCLA) & UT Austin
- **Publication**: **CVPR 2026**
- **Presenter**: 이준형 (Junhyeong Lee)
- **Paper Link**: [arXiv:2603.27437](https://arxiv.org/abs/2603.27437)
- **Review Slide**: [📄 SpatialStack_Paper_Review_LeeJunHyeong.pdf](./SpatialStack_Paper_Review_LeeJunHyeong.pdf)

---

## 📌 1. Motivation & Background

- **기존 2D VLM의 한계**: CLIP이나 기존 비전 백본 기반의 2D VLM은 의미론적(Semantic) 인식에는 뛰어나지만, 물리적 세계의 3D 공간 기하 구조(Depth, Spatial Relation, Viewpoint Translation 등)를 정밀하게 추론하는 데 구조적 한계가 존재합니다.
- **기존 3D VLM 기법의 병목 현상**: 최근 연구들은 Multi-view Geometry Transformer(예: VGGT 등)나 사전학습된 지오메트리 피처를 결합하기 시작했으나, 대부분 **최상위 심층 레이어(Deep Layer)의 피처만을 LLM에 융합(Late-stage single fusion)**하는 구조를 채택했습니다.
- **문제점**: 이로 인해 얕은 레이어(Shallow layer)에 풍부하게 존재하는 미세한 기하학적 세부 정보(Fine-grained geometric details)가 소실되어 고난도 3D 공간 질의응답 및 위치 추론에서 정보 병목(Information Bottleneck)이 발생합니다.
- **SpatialStack의 핵심 목표**: 비전, 3D 기하 구조, 언어 표현을 계층적으로 정렬(Progressive Layered Alignment)하여 얕은 레이어의 기하 지각 능력과 깊은 레이어의 고차원 문맥 의미를 모두 활용할 수 있는 범용 3D VLM 아키텍처 구축.

---

## 🏗️ 2. Core Architecture: Layered Geometry-Language Fusion

SpatialStack은 Multi-level Geometric Feature를 Language Backbone의 여러 레이어에 걸쳐 단계적으로 동기화 및 주입하는 **Hierarchical / Layered Fusion Framework**를 제안합니다:

```text
[ Multi-View Images ] ──▶ [ Geometry Transformer / Encoder ]
                                │            │            │
                         (Shallow Feat) (Mid Feat)   (Deep Feat)
                                │            │            │
                                ▼            ▼            ▼
                         [ Stack Layer 1 ] [ Layer k ] [ Layer N ]  <-- Geometry-Language Alignment
                                │            │            │
[ Text / Instruction ] ──▶ [ Language Backbone (LLM) Transformer Blocks ] ──▶ [ 3D Spatial Reasoning Output ]
```

1. **Multi-level Geometric Feature Extraction**:
   - 기하 인코더(Geometry Transformer)의 서로 다른 계층(Shallow, Middle, Deep)에서 계층별 피처 맵을 보존 및 추출.
   - **Shallow Features**: 정밀한 공간 좌표, 깊이 경계, 국소적 3D 지각 정보 보존.
   - **Deep Features**: 광역적 3D 씬 구조 및 고수준 공간 관계 의미 정보 보존.

2. **Progressive Cross-Attention Stacking (SpatialStack Fusion)**:
   - LLM의 디코더 블록 중간중간에 Layered Adapter/Cross-Attention 모듈을 배치.
   - 텍스트 쿼리가 점진적으로 미세 공간 정보부터 고차원 씬 문맥까지 순차적으로 정렬하며 참조할 수 있도록 설계.

3. **Unified Spatial Representation**:
   - 3D Bounding Box 추론, 카메라 포즈 예측, 3D 질의응답(3D VQA) 등 다양한 공간 태스크를 통합 프롬프트 형식으로 수행.

---

## 🚀 3. Key Technical Contributions

1. **Information Bottleneck 해소**:
   - 단일 레이어 융합(Single-stage late fusion) 대신 계층적 피처 스택(Hierarchical Feature Stacking)을 도입하여 공간 기하 정보의 손실을 방지.
2. **효율적인 연산 및 유연한 모듈성**:
   - 다양한 언어 모델 백본(Llama, Qwen, Vicuna 등) 및 기하 백본에 플러그앤플레이 방식으로 결합 가능.
3. **Multi-View & Multi-Task 3D Reasoning 최적화**:
   - 단일 뷰 및 다중 뷰 입력 환경 모두에서 일관된 3D 공간 좌표계 추론 지원.

---

## 📊 4. Experimental Results & Benchmarks

- **3D Spatial VQA Benchmarks**:
  - ScanQA, SQA3D 등 대표적인 3D 공간 질의응답 벤치마크에서 기존 SOTA 3D VLM 모델 대비 유의미한 성능 향상 달성.
- **3D Grounding & Localization**:
  - 3D 객체 검출 및 공간 위치 지칭(Grounding) 태스크에서 정밀한 Bounding Box 예측 정확도 기록.
- **Ablation Study**:
  - Layered Fusion 유무에 따른 성능 비교 결과, 계층별 피처를 점진적으로 융합했을 때 공간 거리/방향 추론 오차가 대폭 감소함을 검증.

---

## 💡 5. Key Takeaways & Summary

- **멀티모달 3D 공간 추론의 패러다임 전환**: 2D 이미지의 단순 캡셔닝을 넘어, 기하학적 깊이 정보(Geometry)를 LLM 계층 전반에 유기적으로 연결하는 설계의 중요성을 증명.
- **VLA(Vision-Language-Action) 및 로보틱스 확장성**: 정밀한 3D 공간 좌표 이해가 필수적인 로봇 조작(Manipulation) 및 자율주행, 에이전트 내비게이션 분야로의 높은 확장 잠재력을 지님.

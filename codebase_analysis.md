# 코드베이스 분석 보고서: 컨테이너 선박 적부 계획 (Container Ship Stowage Planning)

이 문서는 컨테이너 선박 적부 계획 문제를 해결하기 위한 코드베이스의 구조와 알고리즘에 대해 상세히 설명합니다. 이 프로젝트는 불확실한 수요 하에서 선박의 적부 계획을 최적화하기 위해 심층 강화학습(Deep Reinforcement Learning, DRL)과 시나리오 트리 기반의 혼합 정수 계획법(Stochastic Mixed-Integer Programming, SMIP)을 사용합니다.

## 1. 리포지토리 구조 (Repository Structure)

프로젝트는 다음과 같은 주요 디렉토리와 파일로 구성되어 있습니다.

```
📂 project_root
├── 📂 environment/          # 강화학습 환경 정의
│   ├── env.py               # MasterPlanningEnv 및 BlockMasterPlanningEnv 클래스 (핵심)
│   ├── generator.py         # 수요 및 시나리오 생성기
│   └── utils.py             # 유틸리티 함수 (제약 조건 계산 등)
├── 📂 models/               # 신경망 모델 아키텍처
│   ├── autoencoder.py       # Actor 모델 (Encoder-Decoder 구조)
│   ├── critic.py            # Critic 모델 (Value & Dual Head)
│   ├── encoder.py           # Attention 및 MLP Encoder
│   ├── decoder.py           # Attention 및 MLP Decoder
│   └── embeddings.py        # 다양한 임베딩 레이어 (Cargo, Context, Dynamic 등)
├── 📂 rl_algorithms/        # 강화학습 알고리즘 및 손실 함수
│   ├── train.py             # 학습 루프 (PPO, SAC, DDPG 지원)
│   ├── loss.py              # Feasibility-aware 손실 함수 정의
│   ├── projection.py        # 제약 조건 만족을 위한 투영(Projection) 레이어
│   └── ...
├── 📄 main.py               # 메인 실행 스크립트 (학습 및 테스트 진입점)
├── 📄 scenario_tree_mip.py  # SMIP 베이스라인 (CPLEX 사용)
├── 📄 config.yaml           # 하이퍼파라미터 설정
└── 📄 README.md             # 프로젝트 개요
```

## 2. 문제 정의 (Problem Formulation)

이 코드는 **Master Planning Problem (MPP)**를 다룹니다. 이는 다중 기항지(Multi-port)를 운항하는 컨테이너 선박에서 각 항구마다 어떤 화물을 어디에 적재할지 결정하는 문제입니다.

### 주요 특징
*   **불확실성 (Uncertainty)**: 화물 수요(Demand)가 불확실하며, 선박이 항해하는 동안 단계적으로 실현됩니다.
*   **제약 조건 (Constraints)**:
    *   **용량 (Capacity)**: 각 Bay/Deck의 TEU 용량 제한.
    *   **안정성 (Stability)**: 선박의 무게 중심(LCG, VCG), 트림(Trim), 힐(Heel) 등의 안정성 조건.
    *   **운영 효율성**: 크레인 작업 횟수 최소화, 해치 오버스토리지(Hatch Overstowage) 최소화.
*   **Block Stowage**: `BlockMasterPlanningEnv`에서는 화물을 블록 단위로 관리하여 하역 효율성을 높이는 방식을 모델링합니다.

## 3. 알고리즘 (Algorithms)

이 프로젝트는 크게 두 가지 접근 방식을 구현하고 있습니다.

### 3.1. 심층 강화학습 (Deep Reinforcement Learning)
`rl_algorithms/` 및 `models/`에 구현되어 있으며, 불확실한 환경에서 적응형 정책을 학습합니다.

*   **알고리즘**: PPO (Proximal Policy Optimization) 또는 SAC (Soft Actor-Critic)를 기반으로 합니다.
*   **Feasibility-aware Loss**: 제약 조건 위반을 최소화하기 위해 손실 함수에 페널티 항을 추가하거나, 라그랑주 승수(Lagrangian Multiplier)를 학습하여 제약 조건을 만족시키도록 유도합니다 (`rl_algorithms/loss.py`).
    *   **Primal-Dual Method**: Critic 네트워크에 Dual Head를 추가하여 라그랑주 승수를 동적으로 조정합니다.
*   **모델 아키텍처 (`models/`)**:
    *   **Encoder**: 현재 상태(선박 적재 상태, 수요 등)를 임베딩합니다. Attention 메커니즘을 사용하여 화물 간의 관계를 포착합니다.
    *   **Decoder**: 각 위치(Bay/Deck)에 적재할 화물의 양을 결정하는 Action을 생성합니다.
    *   **Projection Layer**: 신경망이 출력한 Action이 물리적 제약(용량 등)을 위반하지 않도록 보정합니다 (`rl_algorithms/projection.py`).

### 3.2. 시나리오 트리 MIP (Scenario Tree MIP)
`scenario_tree_mip.py`에 구현되어 있으며, 성능 비교를 위한 베이스라인으로 사용됩니다.

*   **접근 방식**: 불확실한 미래 수요를 시나리오 트리(Scenario Tree)로 모델링하고, 이를 혼합 정수 계획법(MIP)으로 풉니다.
*   **Solver**: IBM CPLEX Optimizer를 사용하여 최적해(또는 근사해)를 구합니다.
*   **특징**: 이론적으로 최적에 가까운 해를 제공할 수 있지만, 시나리오 수가 늘어날수록 계산 복잡도가 기하급수적으로 증가합니다.

## 4. 핵심 컴포넌트 상세 (Key Components)

### 환경 (Environment) - `environment/env.py`
*   `TensorDict`를 사용하여 상태(Observation), 행동(Action), 보상(Reward)을 효율적으로 관리합니다.
*   `_step()` 함수에서 화물 적재/하역, 안정성 계산, 비용 계산(Overstowage, Crane moves)을 수행합니다.
*   `_compute_violation()` 함수를 통해 제약 조건 위반 여부를 판단하고 이를 학습에 반영합니다.

### 학습 루프 (Training Loop) - `rl_algorithms/train.py`
*   `TorchRL` 라이브러리를 활용하여 데이터 수집(Collector)과 리플레이 버퍼(Replay Buffer)를 관리합니다.
*   `EarlyStopping` 기능을 통해 학습이 수렴하거나 성능이 저하될 때 조기에 종료합니다.
*   검증(Validation) 단계에서 정책을 평가하고 모델을 저장합니다.

### 모델 (Models) - `models/autoencoder.py`, `models/critic.py`
*   **Actor (Autoencoder)**: Encoder가 상태를 압축하고, Decoder가 이를 바탕으로 확률적 정책(Truncated Normal 분포 등)을 출력합니다.
*   **Critic**: 상태 가치(Value)를 추정하며, Primal-Dual 설정 시 제약 조건 위반에 대한 페널티(Lagrangian Multiplier)도 함께 예측합니다.

이 코드베이스는 복잡한 제약 조건이 있는 조합 최적화 문제를 강화학습으로 해결하려는 고급 연구용 프로젝트로, 특히 **제약 조건 처리(Feasibility)**와 **불확실성 대응(Uncertainty)**에 중점을 두고 있습니다.

# Weights & Biases (wandb) 통합 가이드

## 개요

이 프로젝트는 실험 추적을 위해 **Weights & Biases (wandb)**를 통합했습니다.
- ✅ **tqdm** 진행 표시줄로 실시간 학습 진행 상황 확인
- ✅ **wandb** 로 loss, hyperparameters 등을 자동 로깅
- ✅ 선택적 사용: config에서 쉽게 활성화/비활성화

## 설치

```bash
# tqdm과 wandb 설치
pip install tqdm wandb

# 또는 requirements.txt로 일괄 설치
pip install -r requirements.txt
```

## Wandb 설정

### 1. Wandb 계정 생성 및 로그인

```bash
# wandb 로그인 (최초 1회만)
wandb login

# API 키 입력 (https://wandb.ai/authorize 에서 확인)
```

### 2. Config 파일에서 활성화

```yaml
# configs/config.yaml
experiment:
  use_wandb: true  # wandb 활성화
  wandb_project: patch-detection  # 프로젝트 이름
  wandb_entity: your-username  # 팀/사용자 이름 (선택)
```

### 3. 커맨드 라인에서 활성화

```bash
# wandb 사용
python test.py experiment.use_wandb=true

# 프로젝트 이름 변경
python test.py experiment.use_wandb=true experiment.wandb_project=my-project

# 엔티티 지정
python test.py experiment.use_wandb=true experiment.wandb_entity=my-team
```

## 로깅되는 정보

### Phase 1: 모델 학습

**Metrics:**
- `epoch`: 현재 에폭 번호
- `train_loss`: 에폭 평균 학습 loss

**Config (자동 기록):**
- `model_type`: 모델 타입 (autoencoder/vae/transformer)
- `num_epochs`: 총 에폭 수
- `hidden_dim`: 은닉층 차원
- `latent_dim`: 잠재 공간 차원

**Run Name:** `train-{model_type}` (예: `train-autoencoder`)

### Phase 2: LoRA 도메인 적응

**Metrics:**
- `epoch`: 현재 에폭 번호
- `lora_loss`: 에폭 평균 LoRA loss

**Config (자동 기록):**
- `model_type`: 모델 타입
- `num_epochs`: 총 에폭 수
- `lora_rank`: LoRA rank
- `lora_alpha`: LoRA alpha
- `lora_lr`: LoRA learning rate

**Run Name:** `lora-{model_type}` (예: `lora-autoencoder`)

## 사용 예제

### 예제 1: 기본 사용 (wandb 비활성화)

```bash
# wandb 없이 실행 (기본값)
python test.py
```

출력:
```
[Phase 1: Model Training (Streaming)]
Training autoencoder in streaming mode...
Epochs: 10
  Epoch 1/10: 100%|████████| 8/8 [00:02<00:00, loss=0.0234]
    Epoch 1/10: Loss = 0.023456
```

### 예제 2: wandb 활성화

```bash
# wandb 활성화
python test.py experiment.use_wandb=true
```

출력:
```
[Phase 1: Model Training (Streaming)]
Training autoencoder in streaming mode...
Epochs: 10
wandb: Tracking run with wandb version 0.16.0
wandb: Run data is saved locally in ./wandb/run-20231215_123456
wandb: Run `wandb sync` to sync local data to cloud
wandb: View run at https://wandb.ai/your-user/patch-detection/runs/abc123
  Epoch 1/10: 100%|████████| 8/8 [00:02<00:00, loss=0.0234]
    Epoch 1/10: Loss = 0.023456
```

### 예제 3: 커스텀 프로젝트

```bash
# 다른 프로젝트로 로깅
python test.py \
  experiment.use_wandb=true \
  experiment.wandb_project=my-experiment \
  experiment.wandb_entity=my-team
```

### 예제 4: 여러 모델 비교

```bash
# Autoencoder 실험
python test.py \
  model.type=autoencoder \
  experiment.use_wandb=true

# VAE 실험  
python test.py \
  model.type=vae \
  experiment.use_wandb=true

# Transformer 실험
python test.py \
  model.type=transformer \
  experiment.use_wandb=true
```

Wandb 대시보드에서 세 실험을 비교할 수 있습니다!

## tqdm 진행 표시줄

Wandb와 별도로, **tqdm**이 항상 활성화되어 실시간 진행 상황을 표시합니다:

```
[Phase 1: Model Training (Streaming)]
  Epoch 1/10: 100%|██████████████████| 8/8 [00:02<00:00, loss=0.0234]
  Epoch 2/10:  75%|███████████▌      | 6/8 [00:01<00:00, loss=0.0198]
```

**표시 정보:**
- 진행률 (%)
- 진행바
- 현재/전체 배치 수
- 경과 시간
- 예상 남은 시간
- 현재 배치의 loss

## Wandb 대시보드 활용

### 1. Loss 그래프

브라우저에서 자동으로 생성되는 loss 그래프를 확인할 수 있습니다:
- X축: epoch
- Y축: train_loss / lora_loss

### 2. 여러 실험 비교

같은 프로젝트의 여러 run을 선택하여 비교:
- 다른 모델 타입 비교
- 다른 하이퍼파라미터 비교
- 학습 추이 비교

### 3. Config 추적

각 실험의 모든 설정이 자동으로 기록됩니다:
- 모델 구조
- 학습 설정
- 데이터 설정

### 4. 실험 노트

Wandb 웹에서 각 실험에 메모를 추가할 수 있습니다.

## 트러블슈팅

### wandb 미설치 시

```python
Warning: wandb not installed, skipping logging
```

해결:
```bash
pip install wandb
```

### 로그인 안됨

```python
wandb: ERROR Not authenticated
```

해결:
```bash
wandb login
```

### 프로젝트 접근 권한 없음

```python
wandb: ERROR Permission denied
```

해결:
- `wandb_entity` 설정 확인
- 프로젝트 권한 확인

## 오프라인 모드

인터넷 연결 없이 로컬에서만 로깅:

```bash
# 환경 변수로 오프라인 설정
export WANDB_MODE=offline
python test.py experiment.use_wandb=true

# 나중에 동기화
wandb sync ./wandb/run-*
```

## Best Practices

### 1. 프로젝트 이름 규칙

```yaml
experiment:
  wandb_project: patch-detection-v2  # 버전 명시
```

### 2. Run 이름 규칙

자동 생성되는 run 이름:
- Phase 1: `train-{model_type}`
- Phase 2: `lora-{model_type}`

### 3. 태그 추가 (선택)

코드에서 커스텀 태그 추가 가능:
```python
wandb.init(
    project='patch-detection',
    tags=['baseline', 'autoencoder', 'imagenet-1k']
)
```

### 4. 실험 그룹화

관련 실험을 그룹으로 묶기:
```python
wandb.init(
    project='patch-detection',
    group='model-comparison',
    job_type='train'
)
```

## 결론

✅ **tqdm + wandb** 통합으로:
- 실시간 학습 진행 상황 확인 (tqdm)
- 자동 실험 추적 및 비교 (wandb)
- 선택적 사용 (config에서 on/off)
- 추가 코드 변경 없이 사용 가능

Happy experimenting! 🚀

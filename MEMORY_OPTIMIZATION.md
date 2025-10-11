# GPU 메모리 최적화 가이드

## 문제: 메모리 누적 (Memory Accumulation)

### 이전 구현의 문제점

```python
# ❌ BAD: 모든 임베딩을 리스트에 누적
train_embeddings = []
for imgs, _ in dataloader:
    embeddings_batch = extract_embeddings(imgs)
    for b in range(embeddings_batch.shape[0]):
        train_embeddings.append(embeddings_batch[b])  # 메모리 누적!

# 이후 한번에 학습
model.train(train_embeddings)  # OOM 발생!
```

**문제점:**
- 1000개 이미지 × 49 위치 (7×7) × 16 레이어 × 128 차원 = **~98M elements**
- GPU 메모리에 모든 데이터를 쌓음
- 배치마다 메모리가 계속 누적되어 OOM (Out Of Memory) 발생

## 해결: 스트리밍 학습 (Streaming Training)

### 새로운 구현

```python
# ✅ GOOD: 배치마다 즉시 학습하고 메모리 해제
for imgs, _ in dataloader:
    embeddings_batch = extract_embeddings(imgs)
    
    # 즉시 학습
    loss = model.train_on_batch(embeddings_batch)
    
    # 메모리 해제
    del embeddings_batch
    torch.cuda.empty_cache()
```

**장점:**
- ✅ 메모리 사용량이 배치 크기에만 비례 (일정)
- ✅ 임베딩을 리스트에 누적하지 않음
- ✅ 배치 처리 후 즉시 메모리 해제
- ✅ 대용량 데이터셋도 처리 가능

## 구현 세부사항

### 1. ModelTrainer에 스트리밍 메서드 추가

#### `train_streaming()` - Phase 1 학습

```python
def train_streaming(self, dataloader, extractor, num_epochs=10):
    """배치마다 즉시 학습 (메모리 누적 없음)"""
    for epoch in range(num_epochs):
        for imgs, _ in dataloader:
            imgs_gpu = imgs.to(self.device, non_blocking=True)
            
            # 임베딩 추출
            with torch.no_grad():
                activations = extractor(imgs_gpu)
                embeddings_batch = stack_trajectory(activations)
            
            # 즉시 학습
            loss = self.train_on_batch(embeddings_batch)
            
            # CRITICAL: 메모리 해제
            del imgs_gpu, activations, embeddings_batch
            torch.cuda.empty_cache()
```

#### `adapt_with_lora_streaming()` - Phase 2 LoRA 적응

```python
def adapt_with_lora_streaming(self, dataloader, extractor, lora_cfg, num_epochs=5):
    """LoRA 적응도 스트리밍 방식"""
    # LoRA 설정 및 optimizer 생성
    self.model, lora_params = apply_lora_to_model(self.model, ...)
    lora_optimizer = torch.optim.Adam(lora_params, ...)
    
    for epoch in range(num_epochs):
        for imgs, _ in dataloader:
            imgs_gpu = imgs.to(self.device, non_blocking=True)
            
            # 임베딩 추출
            with torch.no_grad():
                activations = extractor(imgs_gpu)
                embeddings_batch = stack_trajectory(activations)
            
            # LoRA 학습
            trajectories = embeddings_batch.reshape(-1, L, D)
            reconstruction, mu, logvar = self.model(trajectories)
            loss = compute_loss(...)
            loss.backward()
            lora_optimizer.step()
            
            # CRITICAL: 메모리 해제
            del imgs_gpu, activations, embeddings_batch, trajectories
            del reconstruction, loss
            if vae:
                del mu, logvar
            torch.cuda.empty_cache()
```

### 2. test.py 수정

#### Phase 1 (모델 학습)

```python
# ❌ Before: 모든 임베딩 수집 후 학습
train_embeddings = []
for imgs, _ in imagenet_loader:
    embeddings = extract_and_stack(imgs)
    train_embeddings.append(embeddings)  # 메모리 누적!
model_trainer.train(train_embeddings)  # OOM!

# ✅ After: 스트리밍 학습
model_trainer.train_streaming(
    imagenet_loader,
    extractor,
    num_epochs=10
)  # 메모리 일정!
```

#### Phase 2 (LoRA 적응)

```python
# ❌ Before: 도메인 임베딩 수집 후 적응
domain_embeddings = []
for imgs, _ in domain_loader:
    embeddings = extract_and_stack(imgs)
    domain_embeddings.append(embeddings)  # 메모리 누적!
model_trainer.adapt_with_lora(domain_embeddings)

# ✅ After: 스트리밍 적응
model_trainer.adapt_with_lora_streaming(
    domain_loader,
    extractor,
    lora_cfg=cfg.domain_adaptation.lora,
    num_epochs=5
)  # 메모리 일정!
```

## 메모리 사용량 비교

### 이전 방식 (누적)

```
배치 1:  1 GB
배치 2:  2 GB  ⬆️
배치 3:  3 GB  ⬆️⬆️
배치 4:  4 GB  ⬆️⬆️⬆️
...
배치 N:  OOM! 💥  (메모리 부족)
```

### 새로운 방식 (스트리밍)

```
배치 1:  1 GB  →  해제  →  0 GB
배치 2:  1 GB  →  해제  →  0 GB
배치 3:  1 GB  →  해제  →  0 GB
배치 4:  1 GB  →  해제  →  0 GB
...
배치 N:  1 GB  →  해제  →  0 GB  ✅ (일정)
```

## 추가 메모리 절약 팁

### 1. Gradient Accumulation 사용

배치 크기를 줄이고 gradient accumulation으로 효과적인 배치 크기 증가:

```yaml
# config.yaml
data:
  imagenet:
    batch_size: 64  # 128 → 64로 감소
    gradient_accumulation_steps: 2  # 효과적으로 128과 동일
```

### 2. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = model.train_on_batch(embeddings)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 작은 모델 사용

```yaml
# config.yaml
model:
  type: autoencoder  # transformer 대신 사용 (더 가벼움)
  hidden_dim: 64     # 128 → 64로 감소
  latent_dim: 32     # 64 → 32로 감소
  num_layers: 1      # 2 → 1로 감소
```

### 4. 낮은 spatial resolution

```yaml
model:
  spatial_resolution: 7  # 14 → 7로 감소 (49 vs 196 위치)
```

## 사용법

### 기본 사용 (자동으로 스트리밍 방식 적용)

```bash
python test.py
```

### 메모리가 부족한 경우 추가 옵션

```bash
# 배치 크기 감소
python test.py data.imagenet.batch_size=32 data.domain.batch_size=16

# 더 작은 모델 사용
python test.py model.hidden_dim=64 model.latent_dim=32

# 낮은 해상도 사용
python test.py model.spatial_resolution=7

# 샘플 수 감소
python test.py data.imagenet.num_samples=500
```

## 성능 비교

| 방식 | 메모리 사용 | 학습 속도 | 확장성 |
|------|------------|----------|--------|
| **이전 (누적)** | O(N) - 증가 | 빠름 | 제한적 (작은 데이터셋만) |
| **새로운 (스트리밍)** | O(1) - 일정 | 약간 느림 | 무제한 (대용량 가능) |

## 결론

✅ **스트리밍 학습 방식**으로 변경하여:
- 메모리 누적 문제 완전 해결
- 대용량 데이터셋 처리 가능
- GPU 메모리 사용량 일정하게 유지
- OOM 오류 방지

이제 안전하게 대용량 데이터로 학습할 수 있습니다! 🚀

class Config:
    """전역 설정 - 여기서 모든 파라미터를 조절하세요"""
    
    # GPU 설정
    DEVICE = 'cuda:2'  # 'cuda:0', 'cuda:1', 'cuda:2', 'cpu' 등
    NUM_WORKERS = 8    # DataLoader workers (CPU 코어 수에 맞게 조절)
    
    # 배치 크기 설정
    BATCH_SIZE_TRAIN = 128      # ImageNet 학습 배치 (클수록 빠르지만 메모리 많이 사용)
    BATCH_SIZE_CLEAN = 128      # Clean baseline 측정 배치
    BATCH_SIZE_TEST = 128       # 패치된 이미지 테스트 배치
    
    # 공간 해상도 설정 (높을수록 정밀하지만 느림)
    SPATIAL_RESOLUTION = 7    # 7, 14, 28, 56 중 선택
    FEATURE_DIM = 128          # Channel dimension
    
    # Takens embedding 파라미터
    EMBEDDING_M = 3            # Embedding dimension
    EMBEDDING_TAU = 1          # Time delay
    
    # Attractor learning
    PCA_COMPONENTS = 32        # PCA 차원
    N_CLEAN_IMAGES = 1000       # ImageNet에서 사용할 clean 이미지 수
    
    # Detection 설정
    THRESHOLD_MULTIPLIER = 2   # Mean + k*std (3=기본, 5=강함, 6=매우 강함)
    DETECTION_PIXEL_THRESHOLD = 0  # 이 값 이상의 픽셀이 감지되면 anomaly로 판단
    
    # KDE 설정
    KDE_BANDWIDTH = 0.5        # KDE bandwidth
    
    # PatchDetector 설정
    CHUNK_SIZE = 100           # Hausdorff 계산시 chunk 크기 (메모리에 맞게 조절)
    
    # 경로 설정
    IMAGENET_PATH = '/data/ImageNet/train'
    CLEAN_TEST_PATH = 'images_without_patches'
    PATCH_TEST_PATH = 'images_with_patches'
    OUTPUT_DIR = 'detection_results'
    
    @classmethod
    def print_config(cls):
        """설정 출력"""
        print("="*70)
        print("CONFIGURATION (100% GPU, No Numpy!)")
        print("="*70)
        print(f"GPU Settings:")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Num Workers: {cls.NUM_WORKERS}")
        print(f"\nBatch Sizes:")
        print(f"  Training (ImageNet): {cls.BATCH_SIZE_TRAIN}")
        print(f"  Clean Baseline: {cls.BATCH_SIZE_CLEAN}")
        print(f"  Patch Testing: {cls.BATCH_SIZE_TEST}")
        print(f"\nModel Settings:")
        print(f"  Spatial Resolution: {cls.SPATIAL_RESOLUTION}x{cls.SPATIAL_RESOLUTION} = {cls.SPATIAL_RESOLUTION**2} pixels")
        print(f"  Feature Dimension: {cls.FEATURE_DIM}")
        print(f"  Takens (m, tau): ({cls.EMBEDDING_M}, {cls.EMBEDDING_TAU})")
        print(f"  PCA Components: {cls.PCA_COMPONENTS}")
        print(f"\nDetection Settings:")
        print(f"  Threshold: Mean + {cls.THRESHOLD_MULTIPLIER}*std")
        print(f"  Detection Threshold: {cls.DETECTION_PIXEL_THRESHOLD} pixels")
        print("="*70 + "\n")
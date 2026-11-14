import os
import shutil
import random

random.seed(100)

SRC_DIR_K = "n02105505-komondor"
SRC_DIR_M = "whitemop"
EVAL_SET_SIZE = 30

OUTPUT_ROOT = "dataset_final"
for sub in ["test/komondor", "test/white_mop", "train_pool/komondor", "train_pool/white_mop"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)

def partition_images(source_path, label_name):
    """원본 이미지 폴더를 공통 테스트셋과 학습 풀(Pool)로 분할"""
    if not os.path.exists(source_path):
        return False
    
    file_list = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    random.shuffle(file_list)
    
    test_files = file_list[:EVAL_SET_SIZE]
    pool_files = file_list[EVAL_SET_SIZE:]
    
    for item in test_files:
        shutil.copy(os.path.join(source_path, item), os.path.join(OUTPUT_ROOT, "test", label_name, item))
    for item in pool_files:
        shutil.copy(os.path.join(source_path, item), os.path.join(OUTPUT_ROOT, "train_pool", label_name, item))
        
    return True

if partition_images(SRC_DIR_K, "komondor") and partition_images(SRC_DIR_M, "white_mop"):
    experiment_specs = {
        "model_1_balanced_small": {"komondor": 10, "white_mop": 10},
        "model_2_balanced_med": {"komondor": 20, "white_mop": 20},
        "model_3_imbalanced": {"komondor": 20, "white_mop": 2},
        "model_4_balanced_large": {"komondor": 100, "white_mop": 100}
    }
    
    for exp_key, quotas in experiment_specs.items():
        for category, limit in quotas.items():
            target_dir = os.path.join(OUTPUT_ROOT, exp_key, category)
            os.makedirs(target_dir, exist_ok=True)
            
            source_pool = os.path.join(OUTPUT_ROOT, "train_pool", category)
            pool_contents = os.listdir(source_pool)
            random.shuffle(pool_contents)
            
            allocated = pool_contents[:min(limit, len(pool_contents))]
            for img_name in allocated:
                shutil.copy(os.path.join(source_pool, img_name), os.path.join(target_dir, img_name))
                
    print("\n완료, 모든 실험 조건별 데이터셋이 성공적으로 구축")
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

TARGET_SIZE = (224, 224)
BATCH_SZ = 16
NUM_EPOCHS = 5

ROOT_DIR = "dataset_final"
TEST_DATA_DIR = os.path.join(ROOT_DIR, "test")

# 공통 테스트셋 로드 (모든 모델이 동일한 기준 사용해 평가)
eval_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DATA_DIR,
    image_size=TARGET_SIZE,
    batch_size=BATCH_SZ,
    shuffle=False
)
target_classes = eval_dataset.class_names

def construct_model():
    """MobileNetV2 백본을 활용한 이진 분류 모델 구조 생성"""
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    backbone.trainable = False
    
    net = models.Sequential([
        layers.Rescaling(1./127.5, offset=-1, input_shape=(224, 224, 3)),
        backbone,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    net.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return net

scenarios = [
    "model_1_balanced_small",
    "model_2_balanced_med",
    "model_3_imbalanced",
    "model_4_balanced_large"
]

performance_records = {}

for scenario in scenarios:
    print(f"\n=============\n 실험 진행 중: {scenario}\n==============")
    train_path = os.path.join(ROOT_DIR, scenario)
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=TARGET_SIZE,
        batch_size=BATCH_SZ,
        shuffle=True
    )
    
    model_instance = construct_model()
    model_instance.fit(train_dataset, epochs=NUM_EPOCHS, verbose=1)
    
    #학습된 모델 가중치 파일로 저장
    model_instance.save(f"{scenario}_model.h5")
    print(f"[{scenario}] 모델 가중치 저장 완료: {scenario}_model.h5")

    _, test_acc = model_instance.evaluate(eval_dataset, verbose=0)
    print(f"[{scenario}] Test Accuracy: {test_acc * 100:.2f}%")
    
    actuals, predictions = [], []
    for batch_imgs, batch_lbls in eval_dataset:
        probs = model_instance.predict(batch_imgs, verbose=0)
        bin_preds = (probs > 0.5).astype(int).flatten()
        actuals.extend(batch_lbls.numpy())
        predictions.extend(bin_preds)
        
    print("\n상세 분류 리포트")
    print(classification_report(actuals, predictions, target_names=target_classes, zero_division=0))
    
    performance_records[scenario] = test_acc

print("\n=========\n 최종 실험 결과 요약 (Accuracy)\n==========")
for name, score in performance_records.items():
    print(f"- {name}: {score * 100:.2f}%")
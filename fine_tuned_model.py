import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# ==========================================
# 1. VERİYİ YÜKLE
# ==========================================
print("📂 Etiketli veri yükleniyor...")
with open("etiketli_veri.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Toplam {len(data)} yorum yüklendi.\n")

# ==========================================
# 2. VERİYİ HAZIRLA (Aspect Bazında Parçala)
# ==========================================
# Her aspect-sentiment çifti için ayrı eğitim örneği oluştur
prepared_data = []
label_map = {"negative": 0, "neutral": 1, "positive": 2}

for item in data:
    text = item.get("CleanReview", item.get("Review", ""))
    aspects = item.get("Aspects", {})
    
    # Her aspect için ayrı örnek oluştur
    for aspect, sentiment in aspects.items():
        if sentiment in label_map:  # Sadece geçerli sentiment'leri al
            # Input formatı: [CLS] text [SEP] aspect: aspect [SEP]
            input_text = f"[CLS] {text} [SEP] aspect: {aspect} [SEP]"
            
            prepared_data.append({
                "text": input_text,
                "label": label_map[sentiment]
            })

print(f" Toplam {len(prepared_data)} aspect-sentiment çifti hazırlandı.\n")

# ==========================================
# 3. TRAIN/TEST SPLIT
# ==========================================
train_data, test_data = train_test_split(prepared_data, test_size=0.2, random_state=42)
print(f"📊 Eğitim seti: {len(train_data)} örnek")
print(f"📊 Test seti: {len(test_data)} örnek\n")

# ==========================================
# 4. HUGGING FACE DATASET'E ÇEVİR
# ==========================================
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# ==========================================
# 5. TOKENIZER İLE TOKENIZE ET
# ==========================================
MODEL_NAME = "Sengil/ABSA-Turkish-bert-based-small"
print(f"🔹 Model yükleniyor: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("⏳ Tokenization yapılıyor...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
print("✅ Tokenization tamamlandı.\n")

# ==========================================
# 6. MODELİ YÜKLE
# ==========================================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# ==========================================
# 7. TRAINING ARGUMENTS
# ==========================================
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=3,  # 1700 veri için 3 epoch yeterli
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# ==========================================
# 8. TRAINER'I BAŞLAT
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# ==========================================
# 9. EĞİTİMİ BAŞLAT
# ==========================================
print("🚀 Fine-tuning başlıyor...\n")
trainer.train()

# ==========================================
# 10. MODELİ KAYDET
# ==========================================
print("\n Model kaydediliyor...")
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("✅ Fine-tuning tamamlandı! Model './fine_tuned_model' klasörüne kaydedildi.\n")

# ==========================================
# 11. TEST SETİ ÜZERİNDE DEĞERLENDİRME
# ==========================================
print("📊 Test seti üzerinde değerlendirme yapılıyor...")
results = trainer.evaluate()
print(f"📈 Test Sonuçları:")
print(f"   - Eval Loss: {results['eval_loss']:.4f}")
print(f"   - Eval Runtime: {results['eval_runtime']:.2f} saniye")
print(f"   - Samples per Second: {results['eval_samples_per_second']:.2f}")
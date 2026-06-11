import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import re

# ==========================================
# 1. HAZIRLIK: ABSA MODELİ YÜKLEME
# ==========================================
MODEL_NAME = "Sengil/ABSA-Turkish-bert-based-small"
print("🔹 ABSA modeli yükleniyor... (İlk seferde ~500MB indirecek)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # Tahmin modu

id2label = model.config.id2label
print(f"✅ Model Hazır. Sınıflar: {id2label}\n")

#label mapping = {
#    0: 'negative',
#    1: 'neutral',
#    2: 'positive'
#}

# Modelin LABEL_X → anlamlı etiket eşlemesi (manuel override)
label_mapping = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral", 
    "LABEL_2": "positive"
}

# ==========================================
# 2. GELİŞMİŞ ASPECT EXTRACTION (Compound Öncelikli)
# ==========================================
# Temel aspect kelimeleri
ASPECT_KEYWORDS = {
    "kargo", "teslimat", "gönderi", "paket", "kutu",
    "fiyat", "ücret", "para", "indirim", "pahalı", "ucuz",
    "kalite", "malzeme", "sağlamlık", "dayanıklılık",
    "renk", "boyut", "beden", "model", "tasarım",
    "müşteri hizmetleri", "destek", "iletişim", "iade", "değişim",
    "ürün", "sipariş", "marka", "satıcı", "mağaza"
}

# Türkçe stopword'ler
STOPWORDS = {
    "ve", "veya", "ama", "fakat", "ile", "için", "bir", "bu", "şu", "o",
    "de", "da", "mi", "mı", "mu", "mü", "çok", "daha", "en", "az",
    "gibi", "kadar", "sonra", "önce", "şimdi", "bugün", "yarın"
}

# Sık geçen BİRLEŞİK (Compound) Aspect'ler
COMPOUND_ASPECTS = {
    "ürün kutusu", "kargo kutusu", "paket kutusu", "ambalaj hasarı",
    "müşteri hizmetleri", "fiyat performans", "kalite fiyat",
    "teslimat süresi", "iade süreci", "değişim hakkı", "kargo takip",
    "satıcı puanı", "mağaza güvenilirlik"
}

def extract_aspects(text):
    text_clean = re.sub(r'[^\w\sğüşıöç]', ' ', text.lower())
    words = text_clean.split()
    aspects = set()
    used_words = set()  # Cümlede compound olarak yakalanan kelimeleri burada tutacağız

    # 1. Önce compound aspect'leri bul
    for i in range(len(words)-1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram in COMPOUND_ASPECTS:
            aspects.add(bigram)
            used_words.update([words[i], words[i+1]])  # Bu kelimeleri "kullanıldı" olarak işaretle

    # 2. Tek kelimelik aspect'leri ekle (sadece compound'larda kullanılmayanları)
    for word in words:
        if word in ASPECT_KEYWORDS and word not in STOPWORDS:
            if word not in used_words:  # Sadece bu cümlede compound'a girmemişse ekle
                aspects.add(word)

    return list(aspects) if aspects else ["ürün"]

# ==========================================
# 3. GELİŞMİŞ ABSA ANALİZİ (Threshold + 1-5 Yıldız)
# ==========================================
CONFIDENCE_THRESHOLD = 0.65  # %65'in altındaki tahminler "belirsiz" sayılır

def analyze_sentiment(text, aspect):
    # Input formatı: [CLS] yorum [SEP] aspect: aspect [SEP]
    input_text = f"[CLS] {text} [SEP] aspect: {aspect} [SEP]"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Softmax ile 0-1 arasına çevir
    probs = F.softmax(outputs.logits, dim=-1)[0]
    predicted_id = torch.argmax(probs).item()
    confidence = probs[predicted_id].item()
    
    # Olasılıkları indekslere göre ayır (Genelde 0:Neg, 1:Neu, 2:Pos)
    p_neg = probs[0].item()
    p_neu = probs[1].item()
    p_pos = probs[2].item()
    
    # 📊 1-5 Yıldız Hesaplama (Ağırlıklı Ortalama)
    # Negatif -> 1, Nötr -> 3, Pozitif -> 5 puan ağırlığı
    star_rating = (p_neg * 1.0) + (p_neu * 3.0) + (p_pos * 5.0)
    star_rating = max(1.0, min(5.0, round(star_rating, 2))) # 1 ile 5 arasında sınırla
    
    # Etiket eşlemesi
    raw_label = id2label[predicted_id]
    sentiment = label_mapping.get(raw_label, raw_label)
    
    # 🔻 Threshold Kontrolü
    if confidence < CONFIDENCE_THRESHOLD:
        sentiment = "uncertain"
        status = "LOW_CONFIDENCE"
    else:
        status = "OK"
        
    return {
        "aspect": aspect,
        "sentiment": sentiment,
        "star_rating": star_rating,
        "confidence": round(confidence, 3),
        "status": status,
        "probabilities": {
            "negative": round(p_neg, 3),
            "neutral": round(p_neu, 3),
            "positive": round(p_pos, 3)
        }
    }

# ==========================================
# 4. ANA PIPELINE
# ==========================================
def run_absa_pipeline(review_text):
    print(f"\n🔍 Analiz: '{review_text}'")
    
    aspects = extract_aspects(review_text)
    print(f"📌 Aspect'ler: {aspects}")
    
    results = []
    for asp in aspects:
        res = analyze_sentiment(review_text, asp)
        results.append(res)
        print(f"  ✅ {asp}: {res['sentiment'].upper()} ({res['confidence']:.1%})")
        
    return {
        "original_text": review_text,
        "aspects_analyzed": results
    }

# ==========================================
# 5. TEST
# ==========================================
if __name__ == "__main__":
    test_reviews = [
        "Kargo çok hızlı geldi ama ürün kutusu ezikti.",
        "Fiyat performans olarak harika, sadece renk biraz farklıydı.",
        "Müşteri hizmetleri çok ilgili, kesinlikle tekrar alışveriş yaparım."
    ]
    
    final_output = []
    for review in test_reviews:
        result = run_absa_pipeline(review)
        final_output.append(result)
        
    with open("absa_results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print("\n💾 Sonuçlar 'absa_results.json' dosyasına kaydedildi.")
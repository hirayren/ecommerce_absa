import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import re

# ==========================================
# 1. HAZIRLIK: ABSA MODELİ YÜKLEME
# ==========================================
MODEL_NAME = "Sengil/ABSA-Turkish-bert-based-small"
print("🔹 ABSA modeli yükleniyor...")

# 🌟 SİHİRLİ SATIRLAR BURADA:
# local_files_only=True demek: "İnternete bakma, sadece bilgisayarımdaki kayıtlı dosyayı kullan" demektir.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)

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
# 2. NİHAİ VE GÜVENLİ ASPECT EXTRACTION
# ==========================================
GENERAL_ASPECTS = {
    "kargo", "teslimat", "gönderi", "paket", "paketleme", "kutu", "ambalaj",
    "fiyat", "ücret", "para", "indirim", "değer", "maliyet",
    "müşteri hizmetleri", "destek", "iletişim", "iade", "değişim",
    "satıcı", "mağaza", "ürün", "kalite", "performans", "dayanıklılık", "tasarım",
    "kullanım", "kullanışlılık", "rahatlık", "garanti"
}

# ==========================================
# TÜM KATEGORİLER İÇİN GENİŞLETİLMİŞ ASPECT LİSTELERİ
# ==========================================
CATEGORY_ASPECTS = {
    "laptop": {
        "pil", "şarj", "batarya", "ısınma", "ısın", 
        "ekran", "görüntü", "kamera", "ses", "hoparlör",
        "işlemci", "performans", "kasma", "kas", "donma", "don", "hız", "yavaşlama", "yavaş",
        "hafıza", "ram", "depolama", "ssd", "disk", "klavye", "touchpad", "mouse"
    },
    "telefon": {
        "pil", "şarj", "batarya", "ısınma", "ısın",
        "ekran", "görüntü", "kamera", "ses", "hoparlör",
        "işlemci", "performans", "kasma", "kas", "donma", "don", "hız", "yavaşlama", "yavaş",
        "hafıza", "ram", "depolama", "ssd", "disk",
        "klavye", "touchpad", "mouse"
    },
    "kozmetik": {
        "koku", "cilt", "içerik", "etki", "doku", "leke", "nem", "gözenek", "yağlanma", "yağlan",
        "tahriş", "tahriş et",     # "tahriş etti" -> "tahriş"
        "kuruma", "kurut",         # "cildimi kuruttu" -> "kurut"
        "yağlanma", "yağlan",      # "yağlanıyor" -> "yağlan"
        "sivilce", "akne"
    },
    "ev_yasam": {
        "montaj", "sağlamlık", "temizlik", "kurulum", "boyut", "ağırlık", "tasarım", "malzeme",
        "kırılma", "kırıl",        # "kırıldı" -> "kırıl"
        "çizilme", "çizil",        # "çizildi" -> "çizil"
        "paslanma", "paslan"       # "paslandı" -> "paslan"
    },
    "ses_sistemleri": {
        "ses", "hoparlör", "mikrofon", "bas", "tiz", "gürültü", "bozulma", "bozul", "kayıp", "kayıp ses",
        "bağlantı", "bluetooth", "kablo", "uzaklık", "menzil", "pil", "şarj", "batarya"
    },
    "mouse": {
        "ergonomi", "tasarım", "hassasiyet", "kablo", "kablosuz", "pil", "şarj", "batarya",
        "tıklama", "scroll", "tekerlek", "buton", "yazılım", "sürücü", "uyumluluk"
    },
    "temizlik_ürünleri": {
        "temizlik", "koku", "etki", "leke", "renk", "doku", "hassaslık", "tahriş", "tahriş et",     # "tahriş etti" -> "tahriş"
        "kuruma", "kurut",         # "cildimi kuruttu" -> "kurut"
        "yağlanma", "yağlan",      # "yağlanıyor
    }
}

COMPOUND_ASPECTS = {
    "ürün kutusu", "kargo kutusu", "paket kutusu", "ambalaj hasarı",
    "müşteri hizmetleri", "fiyat performans", "kalite fiyat",
    "teslimat süresi", "iade süreci", "değişim hakkı", "kargo takip"
}

STOPWORDS = {
    "ve", "veya", "ama", "fakat", "ile", "için", "bir", "bu", "şu", "o",
    "de", "da", "mi", "mı", "mu", "mü", "çok", "daha", "en", "az",
    "gibi", "kadar", "sonra", "önce", "şimdi", "bugün", "yarın"
}

def smart_aspect_matcher(word, active_keywords):
    # 1. Önce tam eşleşme var mı?
    if word in active_keywords:
        return normalize_aspect(word) # Normalizasyon fonksiyonunu çağır
    
    # 2. Yaygın Türkçe ekleri dene
    suffixes = ['lerinden', 'larından', 'lerimiz', 'larımız', 
                'inden', 'ından', 'unden', 'ünden',
                'den', 'dan', 'ten', 'tan', 'de', 'da', 'te', 'ta',
                'nin', 'nın', 'nun', 'nün', 'in', 'ın', 'un', 'ün',
                'yi', 'yı', 'yu', 'yü',
                'ler', 'lar', 'm', 'n', 'miz', 'niz', 'muz', 'nuz',
                'yor', 'ıyor', 'uyor', 'üyor', # Fiil ekleri de eklendi!
                'sı', 'si', 'su', 'sü', #garanti-si gibi
                'i', 'ı', 'u', 'ü']
    
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            potential_stem = word[:-len(suffix)]
            if potential_stem in active_keywords:
                return normalize_aspect(potential_stem) # Yakalanan kökü normalize et
                
    return None

"""def normalize_aspect(aspect):
    #Yakalanan ham kökleri, JSON çıktısında daha anlamlı ve standart isimlere dönüştürür.
    normalization_map = {
        "kas": "kasma",
        "kasma": "kasma",
        "don": "donma",
        "donma": "donma",
        "ısın": "ısınma",
        "ısınma": "ısınma",
        "yavaş": "yavaşlama"
    }
    # Eğer listede varsa normalize et, yoksa olduğu gibi döndür
    return normalization_map.get(aspect, aspect)"""

# ==========================================
# GENİŞLETİLMİŞ NORMALİZASYON HARİTASI
# ==========================================
def normalize_aspect(aspect):
    """
    Yakalanan ham kökleri, JSON çıktısında daha anlamlı ve standart isimlere dönüştürür.
    """
    normalization_map = {
        # Elektronik
        "kas": "kasma", "kasma": "kasma",
        "don": "donma", "donma": "donma",
        "ısın": "ısınma", "ısınma": "ısınma",
        "yavaş": "yavaşlama", "yavaşlama": "yavaşlama",
        
        # Giyim
        "yırtıl": "yırtılma", "yırtılma": "yırtılma",
        "sökül": "sökülme", "sökülme": "sökülme",
        "daral": "daralma", "daralma": "daralma",
        "tüylen": "tüylenme", "tüylenme": "tüylenme",
        "sol": "solma", "solma": "solma",
        
        # Kozmetik
        "kurut": "kuruma", "kuruma": "kuruma",
        "yağlan": "yağlanma", "yağlanma": "yağlanma",
        
        # Ev Yaşam
        "kırıl": "kırılma", "kırılma": "kırılma",
        "çizil": "çizilme", "çizilme": "çizilme",
        "paslan": "paslanma", "paslanma": "paslanma"
    }
    return normalization_map.get(aspect, aspect)

def extract_aspects(text, category="genel"):
    text_clean = re.sub(r'[^\w\sğüşıöç]', ' ', text.lower())
    words = text_clean.split()
    aspects = set()
    used_words = set()

    active_keywords = GENERAL_ASPECTS.union(CATEGORY_ASPECTS.get(category, set()))

    # 1. Compound aspect'leri bul
    for i in range(len(words)-1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram in COMPOUND_ASPECTS:
            aspects.add(bigram)
            used_words.update([words[i], words[i+1]])

    # 2. Tek kelimelik aspect'leri akıllı eşleştirici ile bul
    for word in words:
        if word not in STOPWORDS:
            matched_aspect = smart_aspect_matcher(word, active_keywords)
            if matched_aspect:
                aspects.add(matched_aspect)
                used_words.add(word)

    # KRİTİK DEĞİŞİKLİK: Hiçbir şey bulunamazsa ["ürün"] DAYATMA, boş liste dön.
    return list(aspects)

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
# 4. ANA PIPELINE (Business Rule Entegrasyonlu)
# ==========================================
def apply_domain_rules(text, aspect, result):
    """
    Modelin bağlamı kaçırabileceği durumlar için iş kuralları (Business Rules) uygular.
    """
    # KURAL 1: Paket/Kutu Hasar Kontrolü
    if aspect in ["paket", "kutu", "ambalaj"]:
        negative_packaging_words = ["açık", "açılmış", "leke", "ezik", "yırtık", "hasarlı", "kırık", "darbe", "yırtılmış"]
        if any(word in text for word in negative_packaging_words):
            # Model pozitif veya nötr dediyse, bunu negatife çek
            if result["sentiment"] != "negative":
                result["sentiment"] = "negative"
                result["confidence"] = max(result["confidence"], 0.85) # Güveni de düzelt
                result["status"] = "RULE_OVERRIDE" # Jüriye göstermek için özel statü
                result["star_rating"] = 1.0 # Yıldızı da düşür
                
    return result

def run_absa_pipeline(review_text, category="genel"):
    # 1. Aspect'leri çıkar
    aspects = extract_aspects(review_text, category)
    
    # KRİTİK DEĞİŞİKLİK: Aspect bulunamadıysa, zorla "ürün" yapma.
    if not aspects:
        return {
            "original_text": review_text,
            "aspects_analyzed": [],
            "message": "NO_ASPECT_FOUND", # Frontend bunu "Genel Yorum / Analiz Edilmedi" olarak gösterebilir
            "status": "IRRELEVANT_FOR_ABSA"
        }
    
    results = []
    for asp in aspects:
        res = analyze_sentiment(review_text, asp)
        
        # 🛡️ Business Rule Uygula (Paket/kutu hatalarını düzelt)
        res = apply_domain_rules(review_text, asp, res)
        
        results.append(res)
        
    return {
        "original_text": review_text,
        "aspects_analyzed": results,
        "status": "OK"
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
import json
import os
from collections import defaultdict
from absa_pipeline import run_absa_pipeline  # Daha önce yazdığın ve test ettiğin fonksiyon

# ==========================================
# ÇOK KAYNAKLI (MULTI-SOURCE) GRUP LAMA VE ANALİZ
# ==========================================
def process_multi_source_data(raw_data_list):
    # Hiyerarşik yapı: ProductName -> Site -> Seller
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "reviews": [],
        "total_reviews": 0,
        "total_aspects": 0,
        "star_rating_sum": 0.0,
        "sentiment_counts": {"positive": 0, "neutral": 0, "negative": 0, "uncertain": 0}
    })))
    
    print(f"⏳ Toplam {len(raw_data_list)} yorum çok kaynaklı olarak işleniyor...\n")
    
    for i, item in enumerate(raw_data_list):
        product = item.get("ProductName", "Bilinmeyen Ürün")
        site = item.get("Site", "Bilinmeyen Site")
        seller = item.get("Seller", "Bilinmeyen Satıcı")
        
        # Model için temizlenmiş metni kullan (Daha yüksek başarı sağlar!)
        text_to_analyze = item.get("CleanReview", item.get("Review", ""))
        
        # İlerleme çubuğu efekti (her 10 yorumda bir yazdır ki terminal dolmasın)
        if i % 10 == 0 or i == len(raw_data_list) - 1:
            print(f"[{i+1}/{len(raw_data_list)}] {product[:30]}... | {site} | {seller}")
        
        # 1. ABSA Analizini Yap
        absa_result = run_absa_pipeline(text_to_analyze)
        
        # 2. Orijinal veriyi ve analiz sonucunu birleştir
        enriched_review = {
            "original_data": {
                "Rating": item.get("Rating"),
                "Date": item.get("Date"),
                "WordCount": item.get("WordCount")
            },
            "absa_analysis": absa_result
        }
        
        # 3. Hiyerarşik yapıya ekle ve istatistikleri güncelle
        node = hierarchy[product][site][seller]
        node["reviews"].append(enriched_review)
        node["total_reviews"] += 1
        
        for aspect in absa_result["aspects_analyzed"]:
            node["total_aspects"] += 1
            node["star_rating_sum"] += aspect["star_rating"]
            
            sentiment = aspect["sentiment"]
            if sentiment in node["sentiment_counts"]:
                node["sentiment_counts"][sentiment] += 1

    # 4. İstatistikleri (Ortalama Puan vb.) Hesapla ve Temizle
    final_output = {}
    for product, sites in hierarchy.items():
        final_output[product] = {}
        for site, sellers in sites.items():
            final_output[product][site] = {}
            for seller, data in sellers.items():
                
                # Ortalama yıldız hesapla (aspect başına)
                avg_star = round(data["star_rating_sum"] / data["total_aspects"], 2) if data["total_aspects"] > 0 else 0.0
                
                final_output[product][site][seller] = {
                    "summary": {
                        "total_reviews": data["total_reviews"],
                        "total_aspects_analyzed": data["total_aspects"],
                        "average_absa_star_rating": avg_star,
                        "sentiment_distribution": data["sentiment_counts"]
                    },
                    "detailed_reviews": data["reviews"]
                }

    return final_output

# ==========================================
# DOSYADAN OKUMA VE ÇALIŞTIRMA
# ==========================================
if __name__ == "__main__":
    # 1. Giriş dosyasının adı (Arkadaşından alacağın dosya ile aynı olmalı)
    INPUT_FILE = "veri.json" 
    
    # 2. Çıkış dosyasının adı
    OUTPUT_FILE = "final_multi_source_analysis.json"
    
    print("🔄 Veri dosyası okunuyor...")
    
    try:
        # Dosyayı oku
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        print(f"✅ {len(raw_data)} adet yorum başarıyla okundu.\n")
        
        # İşlemi başlat
        result = process_multi_source_data(raw_data)
        
        # Sonucu kaydet
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        print("\n" + "="*70)
        print(f"🎉 BAŞARILI! Sonuç '{OUTPUT_FILE}' dosyasına kaydedildi.")
        print("💡 Frontend arkadaşın bu dosyayı alıp Ürün -> Site -> Satıcı ağacını çizebilir.")
        print("="*70)
        
    except FileNotFoundError:
        print(f"\n❌ HATA: '{INPUT_FILE}' dosyası bulunamadı!")
        print("Lütfen arkadaşından aldığın JSON dosyasının adını 'veri.json' olarak değiştirip bu klasöre koyduğundan emin ol.")
    except json.JSONDecodeError:
        print(f"\n❌ HATA: '{INPUT_FILE}' geçerli bir JSON dosyası değil. Dosya formatını kontrol et.")
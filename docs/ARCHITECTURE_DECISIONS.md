# Mimari Kararlar — Coiltech RAG Platform

> CSE 496 bitirme projesi. İkinci sunumdan (Nisan 2026) bu yana mimaride
> yapılan büyük değişikliklerin gerekçeleri. Her karar: önceki durum →
> yeni durum → neden.

---

## 1. ChromaDB → PostgreSQL + pgvector

**Önce:** Department-namespace'li ChromaDB koleksiyonları, ayrı vektör süreci.
**Sonra:** Tek PostgreSQL 16 örneği; `chunks` tablosunda pgvector kolonu,
HNSW ANN indeksi, `tsvector` + GIN ile Türkçe BM25.

**Neden:** Agent tanımları, sohbet geçmişi, servis kayıtları ve cevap
önbelleği zaten ilişkisel veri — vektörleri de aynı veritabanına almak ayrı
bir vektör servisinin operasyon yükünü (yedekleme, senkronizasyon, ikinci
bağlantı havuzu) ortadan kaldırdı. Hibrit arama (vektör + BM25) tek SQL
sorgusunda birleşiyor; ChromaDB'de BM25 için ayrıca Elasticsearch benzeri
bir bileşen gerekecekti.

## 2. SQLite → PostgreSQL

**Önce:** SQLite üzerinde Coiltech şeması.
**Sonra:** PostgreSQL 16 (pgvector ile aynı örnek).

**Neden:** Çok kiracılı mimari (tenant/user scoping) eşzamanlı yazma
gerektiriyor; SQLite'ın tek-yazar kilidi FastAPI'nin async havuzuyla
çelişiyordu. pgvector kararı PostgreSQL'i zaten zorunlu kılmıştı.

## 3. all-MiniLM-L6-v2 → nomic-embed-text

**Önce:** sentence-transformers all-MiniLM-L6-v2 (384-dim, İngilizce ağırlıklı).
**Sonra:** nomic-embed-text (768-dim), yerel Ollama üzerinde.

**Neden:** Soru ve dokümanların tamamı Türkçe; MiniLM Türkçe'de zayıf.
nomic-embed-text çok dilli performansı daha iyi, Ollama üzerinden GPU'suz
makinede de hızlı çalışıyor ve `/api/embed` ile toplu embedding destekliyor.
Embedding'in yerel kalması: her chunk ve her soru için bulut API maliyeti
ve gecikmesi yok.

## 4. Ollama (üretim) → Gemini 2.5 Flash (üretim) + Ollama (fallback)

**Önce:** Tüm cevap üretimi yerel Ollama gemma3 (CPU).
**Sonra:** Cevap üretimi Gemini 2.5 Flash; Gemini 429 dönerse otomatik
Ollama fallback.

**Neden:** Ölçüm sonucu (bkz. `eval/charts/1_ollama_vs_gemini.png`):
CPU'da gemma3 ortalama **~137 s/cevap**, Gemini **~9.7 s** — ~14× fark.
Demo ve gerçek kullanım için 2 dakikalık cevap süresi kabul edilemezdi.
Gemini'nin OpenAI uyumlu endpoint'i sayesinde mevcut DeepSeek istemci kodu
yeniden kullanıldı (`llm_providers.py`). Free tier 10 istek/dk limiti için:
tek hızlı retry (5 s) → sonra yerel gemma3'e şeffaf fallback. Sistem hiçbir
durumda kullanıcıya hata göstermez; en kötü durumda yavaş ama doğru cevap verir.

## 5. LLM tabanlı intent routing → embedding tabanlı routing

**Önce:** RouterAgent her mesaj için LLM'e "13 kategoriden hangisi?" diye
soruyordu (≥0.55 güven eşiği).
**Sonra:** Soru embedding'i × agent routing-description embedding'leri,
cosine benzerliği; en yüksek skor kazanır. LLM çağrısı yok.

**Neden:** Her mesajda fazladan bir LLM çağrısı hem gecikme (CPU'da +100 s,
Gemini'de +2 s ve kota) hem de belirsizlik getiriyordu. Embedding routing
deterministik, ~0.35 s ve ücretsiz. Routing description'ları "Örnek
sorular:" kalıbıyla soru-biçimli yazılınca doğruluk belirgin arttı
(`eval/eval_routing.py` ile ölçülüyor). Description'lar koddan her
startup'ta yeniden embed edilip DB'ye yazılır — kod ile DB ayrışmaz.

## 6. 15 agent → 8 agent

**Önce:** 15+ agent (Application Engineering, Configuration, Digital Twin,
Installed Base, Management Reporting...).
**Sonra:** 8 agent: Servis Kayıtları (SQL), Ürün Kataloğu, Yedek Parça,
Kurulum ve Devreye Alma, Kalite ve Uyumluluk, Satış ve Teklifler,
Özel Projeler, Yönetim Raporu (SQL istatistik).

**Neden:** Gerçek veri hangi alanda varsa agent orada: 781 gerçek
arıza/çözüm kaydı (Excel) ve departman dokümanları. Veri olmayan agent'lar
(ör. canlı sensör verisi gerektiren Digital Twin) boş cevap üretip routing
doğruluğunu düşürüyordu. Az ama tam kapsanan agent, hem routing isabetini
hem cevap kalitesini artırdı. Yönetim Raporu agent'ı, Management Reporting
vaadinin gerçek veriyle karşılanabilen kısmını (arıza istatistikleri,
kategori dağılımları) SQL agent altyapısı üzerinde sunar.

## 7. NL-to-SQL servis kayıtları (RAG yerine)

**Önce:** Excel kayıtları yalnızca RAG koleksiyonunda aranıyordu.
**Sonra:** 781 kayıt gerçek `service_records` tablosunda; sorular LLM ile
güvenli SELECT'e çevrilir (yalnızca SELECT, tek tablo, LIMIT ≤ 20, DML/yorum
yasak). Sonuç yoksa AND→OR gevşetme → tek anahtar kelime → RAG fallback.

**Neden:** "Kaç kez yaşandı?", "en sık arıza nedir?" gibi sayısal sorular
vektör aramayla cevaplanamaz. SQL yolu sayma/filtreleme/istatistiği kesin
yapar; Türkçe karakter sorunu generated-column normalize kolonlarla
(`tespit_norm`, `islem_norm`) çözüldü.

## 8. Semantik cevap önbelleği (CAG)

**Karar:** Soru embedding'i ile önceki sorular cosine ≥ 0.95 eşleşirse cevap
LLM'e gitmeden döner (PostgreSQL `answer_cache` tablosu, 24 saat TTL).

**Neden:** Aynı/benzer sorular tekrar soruluyor (demo ve gerçek kullanımda).
Ölçülen isabet süresi **~0.05 s** — Gemini'den ~190×, Ollama'dan ~2700× hızlı
(bkz. `eval/charts/4_quality_vs_speed.png`). Doküman ingest'inde önbellek
otomatik temizlenir, bayat cevap riski yönetilir.

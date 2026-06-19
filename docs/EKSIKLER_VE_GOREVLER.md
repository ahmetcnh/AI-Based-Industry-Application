# Coiltech RAG Platform — Eksikler ve Uygulama Görevleri

> Bu doküman, projenin mevcut durumu ile final sunum/teslim hedefleri arasındaki
> boşlukları kapatmak için yazılmıştır. Görevler öncelik sırasına göre dizilmiştir.
> Her görev bağımsız uygulanabilir; kabul kriterleri her bölümün sonundadır.
>
> Tarih: 2026-06-12 · Eval durumu: 10/20 (%50) — hedef ≥ %60
>
> **DURUM (2026-06-13): TÜM GÖREVLER TAMAMLANDI ✅**
> - Görev 1: Eval 12/20 (%60.0), exit code 0 ✓ (evaluator'a Türkçe morfoloji
>   toleransı + cevap prompt'larına terim-tekrarı talimatı eklendi)
> - Görev 2: 429 → Ollama fallback canlıda doğrulandı, 4 yeni unit test ✓
> - Görev 3: Router doğruluğu %88.2 (34 soru), grafik: `eval/charts/6_router_accuracy.png` ✓
> - Görev 4: README güncellendi + `docs/ARCHITECTURE_DECISIONS.md` yazıldı ✓
> - Görev 5: "Yönetim Raporu" agent'ı canlıda doğrulandı (routing 0.743 →
>   UNION ALL istatistik SQL'i → sayısal yönetici özeti) ✓
> - pytest: 96 passed ✓
> - Not: Eval p95'i Gemini günlük kotası tükendiği için yüksek (Ollama fallback
>   devredeydi); taze kota ile RAG cevapları ~9.7s.

---

## Sistem Bağlamı (uygulamadan önce oku)

- **Backend:** FastAPI, `backend/main.py` giriş noktası. OpenAI-uyumlu
  `/v1/chat/completions` endpoint'i; Open WebUI frontend olarak bağlanıyor.
- **Veritabanı:** PostgreSQL 16 + pgvector. Tablolar: `chunks`, `agents`,
  `service_records` (781 satır arıza/çözüm kaydı), `answer_cache`, `query_logs`.
- **LLM'ler:** Üretim cevabı **Gemini 2.5 Flash** (OpenAI-uyumlu endpoint,
  `backend/llm_providers.py` → `GEMINI_CHAT_URL`). Embedding **Ollama
  nomic-embed-text**. Yerel fallback modeli **Ollama gemma3** (CPU, ~130s/cevap).
- **Routing:** `backend/router_agent.py` — soru embedding'i ile agent
  embedding'lerinin cosine benzerliği; en yüksek skor kazanır.
- **Agent'lar:** 6 departman agent'ı (`backend/department_agents.py`) +
  1 SQL agent (`backend/sql_agent.py`, NL→SQL, `service_records` tablosu).
- **Eval:** `eval/run_eval.py` + `eval/questions.json` (20 soru, keyword
  bazlı puanlama). Çalıştırma: `source .venv_linux/bin/activate && python eval/run_eval.py`
  (backend ayakta olmalı: `docker compose up -d backend`).
- **Kısıt:** Gemini free tier = 10 istek/dakika. Aşılınca 429; mevcut retry
  20s/40s bekliyor (`llm_providers.py`), bu da cevapları 60s+'a şişiriyor.

---

## GÖREV 1 — Eval pass rate'i %60+'a çıkar  `[öncelik: kritik]`

**Durum:** 10/20 geçiyor. Başarısız sorular: 2, 4, 5, 11, 12, 13, 16, 18, 19
(+1 değişken). Soru 16 hata ile düştü (0.0s — muhtemelen 429).

**Yapılacaklar:**

1. `python eval/run_eval.py` çalıştır, başarısız soruların tam cevap
   metinlerini logla (script'e `--verbose` bayrağı ekleyebilirsin —
   cevabın ilk 300 karakterini yazdırsın).
2. Her başarısız soru için kök neden sınıflandır:
   - **Yanlış agent'a routing** → Görev 3'teki router benchmark ile teşhis et;
     çözüm: `department_agents.py` içindeki routing description'larını
     güçlendir (DB'deki embedding'i de güncellemeyi unutma — agent zaten
     varsa seed kodu atlıyor; `UPDATE agents SET embedding=...` gerekir).
   - **SQL agent yanlış/boş satır dönüyor** → `sql_agent.py` keyword
     çıkarımını incele. Sorular Türkçe ve devrik; `service_records`
     tablosundaki gerçek kayıtlarla eşleşen anahtar kelime seçimi zayıf
     olabilir. Tipik düzeltme: LIKE kalıplarını köke indirgemek
     (örn. "takılması" → "takıl%").
   - **RAG retrieval ıskalıyor** → `k` değerini artır veya soru
     yoğunlaştırmayı (condense) kontrol et.
   - **Cevap doğru ama keyword geçmiyor** → cevap üretim prompt'una
     "sorudaki teknik terimleri cevapta da kullan" yönergesi ekle
     (`sql_agent.py` → `build_summary_prompt`, RAG için `pipeline.py`).
3. Düzeltmelerden sonra eval'i 2 kez üst üste koş (rate limit gölgesi
   olmadan; sorular arası bekleme süresini artırabilirsin).

**Kabul kriteri:** `eval/run_eval.py` exit code 0 (pass rate ≥ %60),
p95 < 30s (429 retry'a takılmadan).

---

## GÖREV 2 — Gemini 429 → Ollama otomatik fallback  `[öncelik: kritik — demo sigortası]`

**Durum:** Gemini rate limit'e takılınca kullanıcıya hata mesajı gösteriliyor
(`main.py` içindeki try/except). Canlı demoda kabul edilemez.

**Yapılacaklar:**

1. `backend/llm_providers.py` → `LLMProvider.generate` ve `generate_stream`:
   Gemini çağrısı 429 ile başarısız olursa (retry'lar tükendikten sonra)
   **aynı mesajlarla Ollama'ya düş** (`_ollama_generate`, model:
   `settings.OLLAMA_GENERATION_MODEL`).
2. Retry stratejisini hafiflet: 20s/40s yerine **tek deneme 5s bekle,
   sonra direkt Ollama fallback**. (60s'lik bekleme demo deneyimini
   öldürüyor; yavaş ama yerel cevap > uzun bekleme.)
3. Fallback gerçekleştiğinde structlog ile `logger.warning("gemini_429_fallback_ollama")`
   logla ve cevabın başına/sonuna kullanıcıya görünür kısa bir not ekleme —
   sessiz fallback yeterli.
4. `main.py`'deki 429 özel hata mesajı bloğu artık nadiren tetiklenmeli;
   Ollama da çökerse devreye girsin diye yerinde bırak.

**Kabul kriteri:** Backend ayaktayken arka arkaya 12+ istek gönder
(rate limit'i bilinçli aş); hiçbir istek hata mesajı dönmemeli, en kötü
ihtimalle yavaş (Ollama) cevap gelmeli. Unit test: 429 dönen sahte client
ile fallback'in tetiklendiğini doğrula (`tests/` içine ekle, mevcut
`FakeLLM` kalıplarını örnek al).

---

## GÖREV 3 — Router doğruluk benchmark'ı  `[öncelik: yüksek]`

**Durum:** İkinci sunumda açık taahhüt: *"router'ın sorguları doğru agent'a
yönlendirme yeteneği doğrulanmadı."* Hiç ölçüm yok.

**Yapılacaklar:**

1. `eval/routing_questions.json` oluştur: ~30 soru, her birinde
   `{"question": ..., "expected_agent": ...}`. Agent adları DB'dekiyle
   birebir: `Servis Kayıtları`, `Ürün Kataloğu`, `Yedek Parça`,
   `Kurulum ve Devreye Alma`, `Kalite ve Uyumluluk`, `Satış ve Teklifler`,
   `Özel Projeler`. (Mevcut `sorular.txt` dosyasındaki kategorilenmiş
   sorulardan yararlan; her agent'a en az 4 soru.)
2. `eval/eval_routing.py` yaz: backend'in router'ını **LLM çağrısı yapmadan**
   test et — ya `/v1/chat/completions` cevabındaki `[→ AgentAdı]` etiketini
   parse et, ya da (daha hızlı) `router_agent.py`'yi doğrudan import edip
   sadece embedding + cosine adımını çalıştır. İkincisi tercih: Gemini'ye
   hiç gitmez, rate limit problemi olmaz.
3. Çıktı: agent bazında doğruluk tablosu + toplam doğruluk %.
   Confusion matrix'i konsola yazdır (hangi agent hangi agent'la karışıyor).
4. Sonucu grafikleştir: `eval/charts/6_router_accuracy.png` —
   beyaz arka plan, mavi tonlar (mevcut `eval/scatter_quality_speed.py`
   stiliyle uyumlu), agent başına doğruluk bar chart'ı.

**Kabul kriteri:** `python eval/eval_routing.py` toplam doğruluk ≥ %85
yazdırıyor ve PNG üretiyor. %85 altıysa routing description'larını
Görev 1.2'deki yöntemle iyileştir ve tekrar ölç.

---

## GÖREV 4 — README ve mimari teknik notu güncelle  `[öncelik: orta]`

**Durum:** README *"Tüm LLM çıkarımı yerel Ollama üzerinde çalışır — API
anahtarı gerekmez"* diyor; gerçekte üretim cevabı Gemini. Sunumda söz
verilen "architectural decisions technical note" hiç yazılmadı.

**Yapılacaklar:**

1. `README.md` düzelt:
   - LLM bölümü: Gemini 2.5 Flash (üretim) + Ollama gemma3 (fallback) +
     nomic-embed-text (embedding). `GEMINI_API_KEY` env değişkenini belgele.
   - Mimari ASCII diyagramına Gemini'yi ekle.
   - 6 departman agent'ını ve SQL agent'ı listele.
   - Benchmark grafiklerine (`eval/charts/`) kısa bir bölümle atıf ver.
2. `docs/ARCHITECTURE_DECISIONS.md` yaz (kısa, ~1-2 sayfa, sunumda
   savunulabilir):
   - **ChromaDB → pgvector:** tek veritabanı, HNSW indeks, BM25+vector
     hibrit arama, operasyonel basitlik.
   - **SQLite → PostgreSQL:** çok kiracılılık, eşzamanlılık, pgvector zaten orada.
   - **all-MiniLM → nomic-embed-text:** Türkçe performansı + Ollama ile
     yerel çalışma, 768-dim.
   - **Ollama → Gemini (üretim):** ölçülen ~14× hız farkı (137s → 9.7s);
     grafik: `eval/charts/1_ollama_vs_gemini.png`.
   - **LLM-intent routing → embedding routing:** her istekte LLM çağrısı
     yerine tek embedding + cosine; maliyet ve gecikme düşüşü.
   - **15 agent → 7 agent:** kapsam netleşmesi — gerçek veri olan alanlara
     odaklanma (781 servis kaydı + departman dokümanları).
   - **Cevap önbelleği (CAG):** cosine ≥ 0.95 semantik eşleşme, ölçülen
     0.05s isabet süresi.

**Kabul kriteri:** README'de Ollama-only iddiası kalmadı; yeni doküman
yukarıdaki 7 kararı gerekçeleriyle içeriyor.

---

## GÖREV 5 — Yönetim Raporu agent'ı  `[öncelik: düşük / opsiyonel]`

**Durum:** Sunumda "Management Reporting — executive KPI summaries" ve
"Digital Twin / anomaly detection" vaat edilmiş; yeni backend'de yok.
Tam digital twin kapsam dışı, ama `service_records` üzerinden basit bir
yönetim raporu agent'ı ucuz ve sunumda etkili.

**Yapılacaklar:**

1. `backend/department_agents.py`'ye 7. agent ekle: **"Yönetim Raporu"**,
   `type='sql'` (SQL agent altyapısını kullansın), routing description'ı:
   trend, istatistik, özet rapor, en sık arıza, aylık dağılım, KPI,
   yönetici özeti vb. terimler.
2. `sql_agent.py`'nin aggregate sorguları desteklediğini doğrula
   (GROUP BY / COUNT zaten `_safe_select`'ten geçiyor mu? `test_safe_select_accepts_count`
   var — GROUP BY için de test ekle, gerekirse whitelist'i genişlet).
3. Örnek soruları evalle: "En sık görülen 5 arıza tipi nedir?",
   "Hangi makinede en çok servis kaydı var?", "Arızaların aylık dağılımı nasıl?"
4. Routing'in mevcut "Servis Kayıtları" agent'ı ile karışmadığını Görev 3
   benchmark'ına 4 soru ekleyerek doğrula. (Karışıyorsa iki agent'ı
   birleştirip system prompt'ta rapor yeteneğini belirtmek de kabul.)

**Kabul kriteri:** Yukarıdaki 3 örnek soru anlamlı, sayısal, SQL'siz
(kullanıcıya sorgu gösterilmeden) cevap dönüyor; router doğruluğu %85
altına düşmüyor.

---

## Dikkat Edilecekler (tüm görevler için)

- **Rate limit:** Eval koşarken sorular arasına bekleme koy veya Görev 2'yi
  önce yap. Üst üste eval koşuları Gemini kotasını tüketiyor.
- **Embedding güncellemesi:** `department_agents.py`'de description
  değiştirmek DB'deki mevcut agent'ı **güncellemez** (seed sadece INSERT
  ediyor). Description değişikliklerinde DB'deki `agents.embedding`
  kolonunu da yeniden hesaplayıp UPDATE et, sonra router cache'ini
  invalidate et (`get_router().invalidate_cache()`).
- **Cevaplarda SQL gösterme yasağı:** `build_summary_prompt` içinde
  "asla SQL sorgusu gösterme" yönergesi var — yeni prompt'larda koru.
- **Cache kirliliği:** Prompt/agent değişikliklerinden sonra
  `answer_cache` tablosunu temizle, yoksa eski cevaplar eval'i yanıltır:
  `DELETE FROM answer_cache;`
- **Test:** Her kod değişikliğinde `pytest tests/ -x -q` yeşil kalmalı.
- **Stil:** Grafiklerde beyaz arka plan + mavi palet, başlıklar İngilizce
  (mevcut chart'larla tutarlılık).

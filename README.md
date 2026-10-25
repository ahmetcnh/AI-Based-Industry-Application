# Coiltech Multi-Agent RAG Platform

Coiltech endüstriyel otomasyon şirketi için geliştirilmiş çok-ajanlı bilgi sistemi. Doğal dil sorguları dokümanlara (PDF), yapısal verilere (SQL) ve gerçek zamanlı makine telemetrisine (Dijital İkiz) yönlendirilir; tüm cevaplar Türkçe üretilir.

**Routing accuracy: 86.9%** — 500 soruluk benchmark (hedef ≥ %85) ✓

---

## Mimari

```
┌──────────────────────────────────────────────────────────────┐
│                     Open WebUI :3000                         │
│              (Chat arayüzü — değiştirilmedi)                 │
└─────────────────────┬────────────────────────────────────────┘
                      │ /v1/chat/completions  (OpenAI uyumlu)
┌─────────────────────▼────────────────────────────────────────┐
│                  FastAPI Backend :8000                        │
│                                                              │
│  Kullanıcı mesajı                                            │
│    ├─ 1. "agent oluştur" niyeti? → agent_creation.py         │
│    ├─ 2. Semantik önbellek (cosine ≥ 0.95) → anında cevap   │
│    ├─ 3. Semantic Router → doğru agent seçimi                │
│    │     ├─ type='chatbot' → Hibrit RAG (pgvector + BM25)    │
│    │     ├─ type='sql'     → NL→SQL (service_records)        │
│    │     └─ Dijital İkiz  → gerçek zamanlı telemetri API     │
│    └─ 4. SSE stream + kaynak citation'ları                   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │            PostgreSQL 16 + pgvector                    │  │
│  │  chunks (HNSW) · agents · service_records ·            │  │
│  │  answer_cache · pending_agent_requests · query_logs    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────┬──────────────────────┬────────────────────────────┘
           │                      │
┌──────────▼──────────┐  ┌────────▼────────────┐
│  Gemini 2.0 Flash   │  │   Ollama (host)     │
│  (cevap üretimi)    │  │ nomic-embed-text    │
│  429 → Ollama       │  │ (embedding, 768-dim)│
│  otomatik fallback  │  │ gemma3 (fallback)   │
└─────────────────────┘  └─────────────────────┘
```

---

## Öne Çıkan Özellikler

- **9 uzman agent** — Servis Kayıtları (SQL), Yönetim Raporu (SQL), Ürün Kataloğu, Yedek Parça, Kurulum ve Devreye Alma, Kalite ve Uyumluluk, Satış ve Teklifler, Özel Projeler, Dijital İkiz; startup'ta otomatik seed
- **Semantik router** — embedding cosine benzerliği + Türkçe keyword boost kuralları + LLM classifier fallback (gap < 0.07); **86.9% doğruluk**
- **Hibrit RAG** — pgvector HNSW (dense) + PostgreSQL BM25 GIN (sparse), α=0.7/0.3; CrossEncoder reranker
- **NL-to-SQL** — 781 satır service_records üzerinde güvenli SELECT üretimi; sonuç yoksa RAG fallback
- **Dijital İkiz** — gerçek zamanlı OEE, alarm ve üretim sayacı sorguları (CTL-01/02, ACK-01/02, DST-01, PTR-01, SBL-01, LSD-01)
- **Sohbetten agent oluşturma** — PDF yükle + "agent oluştur" yaz; LLM içeriği okuyarak doğru koleksiyonu saptar
- **Agent izolasyonu** — departman agent'ları paylaşımlı (user_id=1); kullanıcının oluşturduğu agent'lar yalnızca o kullanıcıya görünür
- **Semantik cevap önbelleği** — benzer soru (cosine ≥ 0.95) varsa LLM'e gitmeden anında yanıt (~0.05 s)
- **Gemini → Ollama otomatik fallback** — rate limit (429) durumunda kesintisiz çalışma

---

## Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| Backend | FastAPI · Python 3.11 · asyncpg |
| Veritabanı | PostgreSQL 16 + pgvector |
| Vektör arama | HNSW indeksi (pgvector) |
| BM25 arama | PostgreSQL tsvector + GIN indeksi (Türkçe) |
| Reranking | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 |
| LLM (üretim) | Gemini 2.0 Flash |
| LLM (fallback) | Ollama — gemma3 |
| Embedding | nomic-embed-text (768-dim, Ollama) |
| Konteyner | Docker + Docker Compose |

---

## Proje Yapısı

```
bitirme2/
├── backend/                     # FastAPI uygulaması
│   ├── main.py                  # Tüm endpoint'ler, chat akışı
│   ├── router_agent.py          # Semantic router (cosine + keyword boosts + LLM fallback)
│   ├── department_agents.py     # 9 departman agent'ı seed + embedding refresh
│   ├── agent_creation.py        # Sohbetten agent oluşturma (içerik tabanlı koleksiyon tespiti)
│   ├── pipeline.py              # RAGPipeline: ingest + query orchestration
│   ├── sql_agent.py             # NL→SQL üretimi + güvenlik denetimi
│   ├── digital_twin.py          # Gerçek zamanlı makine telemetrisi
│   ├── answer_cache.py          # Semantik cevap önbelleği
│   ├── llm_providers.py         # Gemini + Ollama, embedding, fallback
│   ├── tenancy.py               # X-Tenant-ID / X-User-ID middleware
│   ├── agents.py                # Agent CRUD (paylaşımlı + kullanıcıya özel)
│   ├── db.py                    # asyncpg pool, pgvector şema init
│   ├── settings.py              # Pydantic BaseSettings
│   ├── file_watcher.py          # Otomatik ingest (dizin izleyici)
│   ├── chats.py                 # Sohbet thread + mesaj yönetimi
│   ├── Dockerfile
│   └── rag/
│       ├── store/               # insert_chunk, vector_search, bm25_search
│       ├── ingestion/           # pdf_ingest.py, excel_seed.py
│       └── retrieval/           # hybrid.py, query_engine.py, reranker.py
├── eval/
│   ├── eval_routing.py          # Router doğruluk benchmark'ı (Gemini kotası harcamaz)
│   ├── routing_questions_500.json  # 500 soruluk benchmark seti
│   ├── benchmark_response_time.py  # Agent bazında yanıt süresi grafikleri
│   ├── scatter_quality_speed.py    # Kalite-hız scatter plot
│   └── charts/                  # Üretilen PNG grafikler
├── data/
│   └── demo_docs/               # Sentetik demo dökümanları (PDF + TXT)
├── tests/
│   ├── test_backend.py
│   └── test_agent_features.py
├── issue_solution_final.xlsx    # 781 sentetik arıza/çözüm kaydı (service_records seed)
├── soru.md                      # Demo soruları (agent başına 5 soru)
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Kurulum

### Ön Gereksinimler

- Docker Desktop
- Ollama (host makinede)

```bash
ollama pull nomic-embed-text
ollama pull gemma3
```

### 1. Ortam Değişkenleri

```bash
cp .env.example .env
```

`.env` içinde düzenle:

```env
GEMINI_API_KEY=...
DATABASE_URL=postgresql://postgres:test1234@postgres:5432/ragdb
OLLAMA_BASE_URL=http://host.docker.internal:11434
DEFAULT_MODEL=gemini-2.0-flash
```

### 2. Başlat

```bash
docker compose up -d
```

İlk başlatmada PostgreSQL şeması oluşturulur, `issue_solution_final.xlsx` embed edilir ve 9 departman agent'ı otomatik seed edilir.

Depodaki Excel ve `data/demo_docs/` içerikleri tamamen kurgusal örneklerdir; gerçek müşteri, servis veya fiyat verisi içermez. Örnek dosyaları yeniden üretmek için `python -m pip install -r requirements-demo.txt` ardından `python scripts/generate_public_demo_data.py` çalıştırın. Mevcut veritabanında eski kayıtlar varsa yeni örnekler kendiliğinden üzerine yazılmaz; yeni bir demo veritabanı ile başlatın.

### 3. Erişim

| Servis | URL |
|---|---|
| Open WebUI (chat) | http://localhost:3000 |
| Backend API + Swagger | http://localhost:8000/docs |

Open WebUI'de model olarak **`coiltech-router`** seçin.

---

## Agent Listesi

| Agent | Tür | Koleksiyon |
|---|---|---|
| Dijital İkiz | Telemetri | gerçek zamanlı API |
| Servis Kayıtları | SQL | service_records (781 kayıt) |
| Yönetim Raporu | SQL | service_records (istatistik) |
| Ürün Kataloğu | RAG | products_* (9 koleksiyon) |
| Yedek Parça | RAG | spare_parts |
| Kurulum ve Devreye Alma | RAG | installation_commissioning |
| Kalite ve Uyumluluk | RAG | quality_documents |
| Satış ve Teklifler | RAG | sales_and_quotations |
| Özel Projeler | RAG | special_projects |

---

## Agent İzolasyonu

Departman agent'ları (`user_id=1`) tüm kullanıcılara görünür.  
"agent oluştur" ile oluşturulan agent'lar yalnızca oluşturan kullanıcıya aittir.

Router sorgusu: `WHERE tenant_id=$1 AND (user_id=$2 OR user_id=1)`

Kullanıcı kimliği HTTP header'dan gelir:
```
X-Tenant-ID: 1
X-User-ID: 42
```
Header yoksa varsayılan `tenant=1 / user=1` kullanılır.

---

## Sohbetten Agent Oluşturma

1. PDF dökümanı yükle
2. `agent oluştur` yaz
3. LLM PDF içeriğini okuyarak koleksiyonu saptar (içerik tabanlı, dosya adına değil)
4. Agent oluşturulur, router embedding'i yenilenir

Niyet algılama Türkçe/İngilizce destekler: *"agent oluştur"*, *"ajan yarat"*, *"create agent"*...

---

## Evaluation

```bash
# Router doğruluk benchmark'ı (LLM kotası harcamaz)
.venv_linux/bin/python eval/eval_routing.py --questions eval/routing_questions_500.json
```

**Sonuç: 86.9% (472/543)** — hedef ≥ %85 ✓

| Agent | Doğruluk |
|---|---|
| Yedek Parça | 100% |
| Servis Kayıtları | 95% |
| Dijital İkiz | 92% |
| Satış ve Teklifler | 90% |
| Yönetim Raporu | 89% |
| Kurulum ve Devreye Alma | 82% |
| Ürün Kataloğu | 80% |
| Özel Projeler | 76% |
| Kalite ve Uyumluluk | 75% |

```bash
# Birim testleri
python -m pytest tests -q
```

---

## API Referansı

### OpenAI Uyumlu

| Method | Path | Açıklama |
|---|---|---|
| GET | `/v1/models` | Model listesi |
| POST | `/v1/chat/completions` | Streaming chat + citation |

### RAG & İzleme

| Method | Path | Açıklama |
|---|---|---|
| POST | `/upload` | Döküman yükle ve ingest et |
| GET | `/collections` | Koleksiyonlar ve chunk sayıları |
| GET | `/metrics?hours=24` | Sorgu sayısı, gecikme, önbellek oranı |

### Agent'lar

| Method | Path | Açıklama |
|---|---|---|
| GET | `/agents` | Kullanıcının agent listesi (paylaşımlı + kişisel) |
| POST | `/agents` | Yeni agent oluştur |
| PUT | `/agents/{id}` | Agent güncelle (sadece kendi agent'ı) |
| DELETE | `/agents/{id}` | Agent sil (sadece kendi agent'ı) |

---

## Ortam Değişkenleri

```env
# Veritabanı
DATABASE_URL=postgresql://postgres:test1234@postgres:5432/ragdb

# LLM
GEMINI_API_KEY=...
DEFAULT_MODEL=gemini-2.0-flash
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_GENERATION_MODEL=gemma3
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# RAG
DEFAULT_ANSWER_LANGUAGE=tr
RAG_TOP_K=5
RAG_ALPHA=0.7                   # dense/sparse oranı
RAG_SCORE_THRESHOLD=0.3
RERANKER_MODEL=                 # boş = kapalı

# Önbellek
ANSWER_CACHE_ENABLED=true
ANSWER_CACHE_TTL_SECONDS=86400

# Agent
AGENTS_STORE_DIR=agents_store
PENDING_AGENT_TTL_MINUTES=30

# Seed
EXCEL_SEED_PATH=/app/issue_solution_final.xlsx
```

---

## Sorun Giderme

**Backend başlamıyor** — `.env` içinde `DATABASE_URL` host'unun `localhost` değil `postgres` olduğunu kontrol et.

**Excel seed başarısız** — Ollama çalışmıyor. Host makinede `ollama serve` çalıştır.

**Yanıtlar yavaş** — Gemini'ye ulaşılamıyor olabilir, Ollama fallback'e düşüyor (~137 s). `GEMINI_API_KEY` değerini kontrol et.

**"agent oluştur" yanlış koleksiyon seçti** — PDF yeniden yüklendiğinde içerik hash'i aynıysa yeni chunk eklenmez; sistem UUID prefix'i soyarak mevcut chunk'ları bulur. Sorun devam ederse farklı içerikli bir PDF ile dene.

# Coiltech RAG Platform

Coiltech endüstriyel otomasyon platformu için doküman tabanlı soru-cevap, çok kiracılı (multi-tenant) AI asistan ve bilgi yönetimi sistemi. Open WebUI frontend olarak kullanılır; backend OpenAI uyumlu API sunar.

---

## Mimari

```
┌─────────────────────────────────────────────────────────┐
│                     Open WebUI :3000                    │
│            (Chat arayüzü — değiştirilmedi)              │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP  /v1/chat/completions
                     │       /v1/models
┌────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend :8000                   │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ RAG Pipeline│  │ Agent CRUD   │  │  Multi-Tenant │  │
│  │ hybrid.py   │  │ agents.py    │  │  tenancy.py   │  │
│  └──────┬──────┘  └──────────────┘  └───────────────┘  │
│         │                                               │
│  ┌──────▼──────────────────────────────────────────┐   │
│  │              PostgreSQL + pgvector               │   │
│  │  chunks (768-dim embeddings) · agents · chats   │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   Ollama (host)      │
          │  gemma3 · llama3     │
          │  nomic-embed-text    │
          └─────────────────────┘
```

---

## Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| Backend framework | FastAPI (Python 3.11) |
| Veritabanı | PostgreSQL 16 + pgvector |
| Vektör arama | asyncpg ile ham SQL — IVFFlat ANN indeksi |
| BM25 arama | PostgreSQL `tsvector` / `ts_rank` (Türkçe) |
| Reranking | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (lazy load) |
| LLM çıkarım | Ollama (local) — Gemma3, Llama3 |
| Embedding | `nomic-embed-text` 768 boyut (Ollama üzerinden) |
| Alternatif LLM | DeepSeek API (opsiyonel) |
| Graf DB | Neo4j 5.15 (opsiyonel, `RAG_USE_GRAPH=1`) |
| Frontend | Open WebUI (değiştirilmedi) |
| Konteyner | Docker + Docker Compose |

---

## Proje Yapısı

```
bitirme2/
├── backend/                     # Üretim kodu (FastAPI + PostgreSQL + pgvector)
│   ├── main.py                  # FastAPI uygulaması, tüm endpoint'ler
│   ├── settings.py              # Pydantic BaseSettings (env vars)
│   ├── db.py                    # asyncpg pool, pgvector şema init
│   ├── llm_providers.py         # Ollama + DeepSeek LLM factory
│   ├── pipeline.py              # RAGPipeline: ingest + query orchestration
│   ├── answer_cache.py          # Semantik cevap önbelleği (pgvector cosine)
│   ├── router_agent.py          # Embedding tabanlı agent yönlendirici
│   ├── tenancy.py               # X-Tenant-ID / X-User-ID middleware
│   ├── tenants.py               # Tenant & kullanıcı yönetim API'leri
│   ├── agents.py                # Agent CRUD (PostgreSQL)
│   ├── chats.py                 # Sohbet thread + mesaj yönetimi
│   ├── file_watcher.py          # Dizin izleyici, otomatik ingest
│   ├── Dockerfile
│   └── rag/
│       ├── store/               # insert_chunk, vector_search, bm25_search
│       ├── ingestion/           # excel_seed.py, pdf_ingest.py (toplu embedding)
│       ├── retrieval/           # hybrid.py, query_engine.py, reranker.py
│       └── graph/               # Neo4j entegrasyonu (isteğe bağlı)
├── eval/
│   ├── questions.json           # ~20 Türkçe soru-anahtar kelime çifti
│   └── run_eval.py              # Değerlendirme scripti (canlı stack gerekir)
├── tests/                       # Yeni birim testleri (servis gerektirmez)
│   └── test_backend.py
├── legacy/                      # Eski ChromaDB/SQLite mimarisi (arşiv)
│   ├── agents/
│   ├── api/
│   ├── config/
│   ├── database/
│   ├── demo/
│   ├── llm/
│   ├── rag/
│   ├── full_system_test.py
│   └── tests/
│       └── test_platform.py
├── openwebui_functions/         # Open WebUI Function plugin
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── issue_solution_final.xlsx    # Çekirdek bilgi tabanı (Türkçe)
```

---

## Kurulum ve Çalıştırma

### Ön Gereksinimler

1. **Docker Desktop** kurulu ve çalışıyor olmalı
2. **Ollama** host makinede kurulu olmalı
3. Gerekli modelleri çek:

```powershell
ollama pull nomic-embed-text
ollama pull gemma3
```

### 1. Ortam Değişkenlerini Ayarla

```powershell
copy .env.example .env
```

`.env` içinde kontrol et / düzenle:

```env
DATABASE_URL=postgresql://postgres:test1234@postgres:5432/ragdb
OLLAMA_BASE_URL=http://host.docker.internal:11434
DEFAULT_MODEL=ollama:gemma3
EXCEL_SEED_PATH=/app/issue_solution_final.xlsx
```

### 2. Servisleri Başlat

```powershell
docker compose up -d
```

İlk başlatmada:
- PostgreSQL şeması otomatik oluşturulur
- `issue_solution_final.xlsx` dosyasındaki bakım kayıtları otomatik embed edilip veritabanına yazılır
- Open WebUI model dosyalarını indirir (ilk kez ~3-5 dakika)

### 3. Erişim

| Servis | URL |
|---|---|
| Open WebUI (chat) | http://localhost:3000 |
| Backend API + Swagger | http://localhost:8000/docs |
| Neo4j Browser | http://localhost:7474 |

---

## RAG Pipeline'ı Nasıl Çalışır

### Doküman Yükleme

```
POST /upload
  ↓
Dosya tipi algıla (PDF / DOCX / XLSX / TXT)
  ↓
Metin çıkar → chunk'lara böl (~512 token, 64 token örtüşme)
  ↓
nomic-embed-text ile 768-dim embedding üret
  ↓
PostgreSQL chunks tablosuna kaydet (content hash ile dedup)
```

### Sorgu & Cevap Üretimi

```
Kullanıcı sorusu (Open WebUI → /v1/chat/completions)
  ↓
[Opsiyonel] Sorgu genişletme — LLM 2 alternatif üretir
  ↓
Hibrit arama:
  • Vektör: pgvector cosine similarity
  • BM25: PostgreSQL ts_rank (Türkçe)
  • Harmanlama: score = 0.3×BM25 + 0.7×vector
  ↓
CrossEncoder reranking (top-K adayları yeniden sırala)
  ↓
[Eşik altıysa] ILIKE fallback arama
  ↓
[Opsiyonel] Neo4j graf bağlamı ekle
  ↓
LLM prompt oluştur + cevap üret (Gemma3 / Llama3 / DeepSeek)
  ↓
SSE stream olarak Open WebUI'ye gönder
```

---

## API Referansı

### OpenAI Uyumlu (Open WebUI için)

| Method | Path | Açıklama |
|---|---|---|
| GET | `/v1/models` | Mevcut modeller (Ollama + DeepSeek + agent'lar) |
| POST | `/v1/chat/completions` | Streaming chat, OpenAI formatında |

### RAG

| Method | Path | Açıklama |
|---|---|---|
| POST | `/upload` | Doküman yükle ve ingest et |
| GET | `/collections` | Koleksiyonlar ve chunk sayıları |
| GET | `/v1/llm-status` | LLM bağlantı durumu |

### Agent'lar

| Method | Path | Açıklama |
|---|---|---|
| GET | `/agents` | Kullanıcının agent listesi |
| POST | `/agents` | Yeni agent oluştur |
| GET | `/agents/{id}` | Agent detayı |
| PUT | `/agents/{id}` | Agent güncelle |
| DELETE | `/agents/{id}` | Agent sil |

### Sohbet Thread'leri

| Method | Path | Açıklama |
|---|---|---|
| GET | `/chats` | Sohbet listesi |
| POST | `/chats` | Yeni sohbet oluştur |
| PATCH | `/chats/{id}` | Başlık güncelle |
| DELETE | `/chats/{id}` | Sohbet sil |
| GET | `/chats/{id}/messages` | Mesaj geçmişi |

### Çok Kiracılı Yönetim

| Method | Path | Açıklama |
|---|---|---|
| GET | `/tenants` | Tenant listesi |
| POST | `/tenants` | Yeni tenant |
| GET | `/tenants/{id}/users` | Kullanıcı listesi |
| POST | `/tenants/{id}/users` | Kullanıcı ekle |
| GET | `/me` | Mevcut tenant/kullanıcı bağlamı |

---

## Çok Kiracılı Mimari

Her HTTP isteğinde `X-Tenant-ID` ve `X-User-ID` başlıkları okunur:

```
X-Tenant-ID: 1
X-User-ID: 1
```

- Başlık yoksa varsayılan: tenant=1, user=1
- Tüm chunk, agent, sohbet sorguları tenant+user kapsamıyla filtrelenir
- Open WebUI varsayılan olarak başlık göndermez → tenant 1 / user 1 kullanılır

---

## Agent Oluşturma

### Yol A — Open WebUI Native Editor

Workspace → Models → New Model → İsim + system prompt gir → Kaydet.
Arka planda kayıtlı olmaz, sadece prompt iletilir.

### Yol B — API ile

```powershell
curl -X POST http://localhost:8000/agents `
  -H "Content-Type: application/json" `
  -d '{
    "name": "ServisBot",
    "system_prompt": "Sen Coiltech servis uzmanısın. Sadece belgelerden cevap ver.",
    "model": "ollama:gemma3",
    "collection": "service_maintenance"
  }'
```

Oluşturulan agent `/v1/models`'da `agent-1-servisbot` olarak görünür ve Open WebUI'de model olarak seçilebilir.

### Yol C — Open WebUI Function Plugin

`openwebui_functions/agent_manager.py` dosyasını Open WebUI'ye yükle:
Admin → Functions → New Function → dosya içeriğini yapıştır.

Sohbet içinde komutlar:
```
/list-agents
/create-agent name=ServisBot prompt=Sen bir servis uzmanısın model=ollama:gemma3 collection=service_maintenance
/delete-agent id=1
```

---

## Bilgi Tabanı Koleksiyonları

| Koleksiyon | İçerik |
|---|---|
| `service_maintenance` | Bakım, arıza, alarm kayıtları (Excel'den) |
| `products_compact_drives` | Kompakt sürücü ürün belgeleri |
| `products_loop_drives` | Loop tipi sürücüler |
| `products_decoilers` | Açıcı (decoiler) belgeleri |
| `products_ctl_lines` | Kesme hattı belgeleri |
| `installation_commissioning` | Kurulum ve devreye alma |
| `spare_parts` | Yedek parça bilgileri |
| `quality_documents` | Kalite, CE, sertifika belgeleri |
| `sales_and_quotations` | Teklif ve satış belgeleri |

Yeni doküman yüklerken `collection_name` parametresiyle koleksiyon seçilir. Seçilmezse dosya adından otomatik tespit edilir.

---

## Ortam Değişkenleri

```env
# Uygulama
APP_NAME=Coiltech RAG Platform
APP_VERSION=2.0.0

# Veritabanı
DATABASE_URL=postgresql://postgres:test1234@postgres:5432/ragdb

# LLM
OLLAMA_BASE_URL=http://host.docker.internal:11434
DEFAULT_MODEL=ollama:gemma3
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
DEEPSEEK_API_KEY=               # Opsiyonel

# RAG Ayarları
DEFAULT_ANSWER_LANGUAGE=tr
RAG_TOP_K=10
RAG_ALPHA=0.3                   # BM25/vektör harmanlama oranı
RAG_SCORE_THRESHOLD=0.3         # Fallback eşiği
RAG_MULTI_QUERY=0               # 1=sorgu genişletme aktif
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

# Graf (Opsiyonel)
RAG_USE_GRAPH=0                 # 1=Neo4j aktif
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=test1234

# Excel Çekirdek Veri
EXCEL_SEED_PATH=/app/issue_solution_final.xlsx
OPENWEBUI_DEFAULT_COLLECTION=service_maintenance
```

---

## Sorun Giderme

### Backend başlamıyor — `Connection refused`
`.env` dosyasında `DATABASE_URL` içindeki host adının `localhost` değil `postgres` olduğunu kontrol et.

### Excel seed başarısız — `All connection attempts failed`
Ollama çalışmıyor. Host makinede Ollama'yı başlat:
```powershell
ollama serve
```

### Open WebUI açılmıyor
Port eşlemesini kontrol et — OpenWebUI içeride **8080** portunda çalışır:
```yaml
ports:
  - "3000:8080"
```

### Cevaplar boş geliyor veya "No relevant documents found"
Koleksiyona doküman yüklenmemiş olabilir. Backend'i yeniden başlat (Excel seed için):
```powershell
docker compose restart backend
```

---

## Değerlendirme (Eval)

`eval/questions.json` dosyasında ~20 Türkçe soru-anahtar kelime çifti bulunur. Bu çiftler `issue_solution_final.xlsx` verisinden türetilmiştir.

### Kullanım

Canlı stack çalışırken (bkz. **Hızlı Başlangıç**):

```bash
# Varsayılan ayarlarla (model: coiltech-rag, url: http://localhost:8000)
python eval/run_eval.py

# Farklı model veya URL ile
python eval/run_eval.py --model coiltech-rag --base-url http://localhost:8000
```

Çıktı her soru için `PASS/FAIL`, eksik anahtar kelimeler ve yanıt süresini tablo olarak gösterir. Geçme oranı %60'ın altındaysa çıkış kodu 1 döner.

---

## Geliştirme Notları

- **Yeni koleksiyon eklemek**: `backend/pipeline.py` içindeki `_KEYWORD_MAP` sözlüğüne anahtar kelime-koleksiyon eşlemesi ekle
- **Reranker'ı devre dışı bırakmak**: `.env`'de `RERANKER_MODEL=` (boş bırak)
- **Sorgu genişletmeyi aktifleştirmek**: `RAG_MULTI_QUERY=1`
- **Neo4j'yi aktifleştirmek**: `RAG_USE_GRAPH=1` ve Neo4j servisinin çalıştığından emin ol
- **Yeni LLM provider eklemek**: `backend/llm_providers.py` içindeki `LLMProvider` sınıfını genişlet

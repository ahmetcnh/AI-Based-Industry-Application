# Coiltech RAG Platform

Coiltech endüstriyel otomasyon platformu için doküman tabanlı soru-cevap, çok kiracılı (multi-tenant) AI asistan ve bilgi yönetimi sistemi. Open WebUI frontend olarak kullanılır; backend OpenAI uyumlu API sunar. Tüm LLM çıkarımı **yerel Ollama** üzerinde çalışır — API anahtarı gerekmez.

---

## Öne Çıkan Özellikler

- **Akıllı yönlendirme (`coiltech-router`)** — her mesaj, embedding benzerliğiyle en uygun uzman agent'a yönlendirilir; aynı sohbet içinde konu değişince agent da değişir
- **Sohbetten agent oluşturma** — "Bir agent oluştur" yazmak yeterli; dosya yüklenmemişse sistem uyarır, dosya gelince agent'ı otomatik oluşturur
- **NL-to-SQL servis kayıtları** — `issue_solution_final.xlsx` (781 arıza/çözüm kaydı) gerçek bir SQL tablosunda; bakım soruları RAG yerine güvenli SELECT sorgularıyla yanıtlanır
- **Semantik cevap önbelleği** — benzer sorular (cosine ≥ 0.95) LLM'e gitmeden anında yanıtlanır
- **Sohbet geçmişi desteği** — "Peki bunun yedek parçası var mı?" gibi takip soruları, geçmişten bağımsız soruya dönüştürülerek aranır
- **Kaynak gösterimi** — cevaplar hangi dokümandan geldiğini SSE citation olarak bildirir
- **`/metrics`** — koleksiyon bazında p95 gecikme, güven skoru ve önbellek isabet oranı

---

## Mimari

```
┌──────────────────────────────────────────────────────────────┐
│                       Open WebUI :3000                       │
│              (Chat arayüzü — değiştirilmedi)                 │
└─────────────────────┬────────────────────────────────────────┘
                      │ HTTP  /v1/chat/completions · /v1/models
┌─────────────────────▼────────────────────────────────────────┐
│                   FastAPI Backend :8000                      │
│                                                              │
│  Kullanıcı mesajı                                            │
│    ├─ 1. "agent oluştur" niyeti? → agent_creation.py         │
│    │     (dosya yok → uyarı + bekleyen istek;                │
│    │      dosya gelince otomatik oluşturma)                  │
│    ├─ 2. Cevap önbelleği (answer_cache.py, cosine ≥ 0.95)    │
│    ├─ 3. Router → en uygun agent (router_agent.py)           │
│    │     ├─ type='chatbot' → RAG pipeline (hibrit arama)     │
│    │     └─ type='sql'     → NL→SQL (sql_agent.py)           │
│    │                          └─ 0 satır → RAG fallback      │
│    └─ 4. [→ AgentAdı] etiketi + SSE stream + kaynaklar       │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              PostgreSQL 16 + pgvector                  │  │
│  │  chunks (HNSW) · agents · service_records ·            │  │
│  │  answer_cache · pending_agent_requests · query_logs    │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────┬────────────────────────────────────────┘
                      │
           ┌──────────▼──────────┐         agents_store/
           │   Ollama (host)     │         (agent tanım dosyaları,
           │ gemma3 · nomic-embed│          host'a mount edilir)
           └─────────────────────┘
```

---

## Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| Backend framework | FastAPI (Python 3.11) |
| Veritabanı | PostgreSQL 16 + pgvector |
| Vektör arama | asyncpg ile ham SQL — **HNSW** ANN indeksi |
| BM25 arama | PostgreSQL `tsvector` + **GIN indeksi** (Türkçe) |
| NL-to-SQL | LLM ile SELECT üretimi + güvenlik denetimi (`sql_agent.py`) |
| Reranking | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (opsiyonel, lazy load) |
| LLM çıkarım | Ollama (yerel) — Gemma3 |
| Embedding | `nomic-embed-text` 768 boyut (toplu `/api/embed`) |
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
│   ├── llm_providers.py         # Ollama + DeepSeek; embedding LRU cache, keep_alive
│   ├── pipeline.py              # RAGPipeline: ingest + query orchestration
│   ├── router_agent.py          # Embedding tabanlı agent yönlendirici (mesaj başına)
│   ├── agent_creation.py        # Sohbetten agent oluşturma + agents_store tanım dosyaları
│   ├── sql_agent.py             # NL→SQL: service_records sorgu üretimi + güvenlik
│   ├── answer_cache.py          # Semantik cevap önbelleği (pgvector cosine)
│   ├── tenancy.py               # X-Tenant-ID / X-User-ID middleware
│   ├── tenants.py               # Tenant & kullanıcı yönetim API'leri
│   ├── agents.py                # Agent CRUD (PostgreSQL)
│   ├── chats.py                 # Sohbet thread + mesaj yönetimi
│   ├── file_watcher.py          # Dizin izleyici, otomatik ingest (+ anlık tarama)
│   ├── Dockerfile
│   └── rag/
│       ├── store/               # insert_chunk, vector_search, bm25_search
│       ├── ingestion/           # excel_seed.py, pdf_ingest.py (toplu embedding)
│       ├── retrieval/           # hybrid.py, query_engine.py, reranker.py
│       └── graph/               # Neo4j entegrasyonu (isteğe bağlı)
├── agents_store/                # Oluşturulan agent tanım dosyaları (JSON, host'ta görünür)
├── eval/
│   ├── questions.json           # ~20 Türkçe soru-anahtar kelime çifti
│   └── run_eval.py              # Değerlendirme scripti (canlı stack gerekir)
├── tests/                       # Birim testleri (servis gerektirmez)
│   ├── test_backend.py
│   └── test_agent_features.py   # Niyet algılama, SQL güvenlik, tanım dosyaları
├── legacy/                      # Eski ChromaDB/SQLite mimarisi (arşiv)
├── openwebui_functions/         # Open WebUI Function plugin
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── issue_solution_final.xlsx    # Çekirdek bilgi tabanı (781 arıza/çözüm kaydı)
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

### 1. Ollama Performans Ayarları (önerilir)

Modellerin VRAM'de sabit kalması yanıt süresini ciddi iyileştirir (model takası 5-20 sn sürebilir). Kullanıcı ortam değişkenlerini ayarlayıp Ollama'yı yeniden başlat:

```powershell
[Environment]::SetEnvironmentVariable('OLLAMA_KEEP_ALIVE','-1','User')
[Environment]::SetEnvironmentVariable('OLLAMA_MAX_LOADED_MODELS','3','User')
[Environment]::SetEnvironmentVariable('OLLAMA_FLASH_ATTENTION','1','User')
[Environment]::SetEnvironmentVariable('OLLAMA_KV_CACHE_TYPE','q8_0','User')
```

### 2. Ortam Değişkenlerini Ayarla

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

### 3. Servisleri Başlat

```powershell
docker compose up -d
```

İlk başlatmada:
- PostgreSQL şeması otomatik oluşturulur (HNSW + GIN indeksleri dahil)
- `issue_solution_final.xlsx` hem RAG koleksiyonuna embed edilir hem de `service_records` SQL tablosuna yüklenir
- "Servis Kayıtları" SQL agent'ı otomatik oluşturulur
- Open WebUI model dosyalarını indirir (ilk kez ~3-5 dakika)

### 4. Erişim

| Servis | URL |
|---|---|
| Open WebUI (chat) | http://localhost:3000 |
| Backend API + Swagger | http://localhost:8000/docs |
| Neo4j Browser | http://localhost:7474 |

Open WebUI'de model olarak **`coiltech-router`** seçilmesi önerilir — sorular otomatik olarak doğru agent'a gider.

---

## Sorgu Akışı Nasıl Çalışır

```
Kullanıcı mesajı (Open WebUI → /v1/chat/completions)
  ↓
"Agent oluştur" niyeti mi? → agent oluşturma akışı (aşağıda)
  ↓
Takip sorusu mu? → geçmişle bağımsız soruya dönüştür (tek LLM çağrısı)
  ↓
Cevap önbelleği: benzer soru (cosine ≥ 0.95) varsa → anında cevap (~0.1 sn)
  ↓
Router: soru embedding'i × agent embedding'leri → en uygun agent
  ↓
┌─ Agent type='sql' (Servis Kayıtları):
│    LLM → SELECT üretimi → güvenlik denetimi → service_records sorgusu
│    Sonuç yoksa: AND→OR gevşetme → tek anahtar kelime → yine yoksa RAG fallback
│    Satırlar → LLM özetleme → cevap
│
└─ Agent type='chatbot':
     Hibrit arama (pgvector HNSW + BM25 GIN, eşzamanlı)
     → [Eşik altıysa] ILIKE fallback
     → LLM prompt + cevap üretimi
  ↓
[→ AgentAdı] etiketi + SSE stream + kaynak citation'ları
  ↓
Cevap önbelleğine yaz
```

---

## Sohbetten Agent Oluşturma

Open WebUI'de `coiltech-router` modeliyle:

1. **"Bir agent oluştur, adı SatışBot olsun, satış teklifleri sorularına cevap versin"** yaz
   - Yakın zamanda dosya yüklendiyse → agent hemen oluşturulur
   - Dosya yoksa → ⚠️ *"Agent oluşturmak için bir doküman gerekli..."* uyarısı gelir, istek 30 dk bekler
2. **Dosyayı yükle** (sadece dosya, metin yazmana gerek yok)
   - FileWatcher dosyayı algılar → agent, ilk mesajındaki ad/görev bilgileriyle oluşturulur
3. Sonraki mesajında **"✅ SatışBot agent'ı oluşturuldu"** onayını görürsün

Oluşturulan her agent:
- `agents` tablosuna kaydedilir (router'ın seçebilmesi için embedding'iyle)
- `agents_store/<isim>.json` tanım dosyası olarak host'a yazılır
- `/v1/models` listesinde görünür

Niyet algılama Türkçe/İngilizce ve ASCII yazımları destekler: *"agent oluştur"*, *"ajan oluştur"*, *"yeni agent"*, *"create agent"*, *"I want to create an agent"*...

### Diğer Yollar

**API ile:**
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

**Otomatik:** Yeni bir koleksiyona ilk doküman yüklendiğinde o koleksiyon için agent otomatik oluşturulur.

**Open WebUI Function Plugin:** `openwebui_functions/agent_manager.py` → Admin → Functions. Sohbette `/list-agents`, `/create-agent ...`, `/delete-agent id=1` komutları.

---

## NL-to-SQL Servis Kayıtları

`issue_solution_final.xlsx` içindeki 781 arıza/çözüm kaydı `service_records` tablosuna yüklenir
(`tespit_edilenler`, `yapilan_islem` + Türkçe karakter normalize edilmiş arama kolonları).

Bakım/arıza soruları router tarafından **Servis Kayıtları** agent'ına (type=`sql`) yönlendirilir:

1. LLM, soruyu güvenli bir `SELECT` sorgusuna çevirir (few-shot Türkçe örneklerle)
2. Güvenlik denetimi: yalnızca `SELECT`, yalnızca `service_records`, mutasyon/yorum/`;` yasak, `LIMIT ≤ 20`
3. Türkçe karakter katlama: `hatası` ↔ `hatasi` farkı sorun çıkarmaz
4. Sonuç yoksa otomatik gevşetme: `AND → OR` → tek anahtar kelime → yine yoksa **RAG fallback**
5. Bulunan satırlar LLM ile Türkçe özetlenir

Örnek: *"Toprak hatası alarmı nasıl çözülmüş?"* → `WHERE tespit_norm LIKE '%toprak%' OR ...` → 17 kayıt → özet cevap.

---

## API Referansı

### OpenAI Uyumlu (Open WebUI için)

| Method | Path | Açıklama |
|---|---|---|
| GET | `/v1/models` | Mevcut modeller (`coiltech-router`, `coiltech-rag`, Ollama, agent'lar) |
| POST | `/v1/chat/completions` | Streaming chat, OpenAI formatında, citation destekli |

### RAG & İzleme

| Method | Path | Açıklama |
|---|---|---|
| POST | `/upload` | Doküman yükle ve ingest et |
| GET | `/collections` | Koleksiyonlar ve chunk sayıları |
| GET | `/metrics?hours=24` | Sorgu sayısı, ort/p95 gecikme, güven, önbellek isabet oranı |
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
- Tüm chunk, agent, sohbet ve önbellek sorguları tenant+user kapsamıyla filtrelenir
- Open WebUI varsayılan olarak başlık göndermez → tenant 1 / user 1 kullanılır

---

## Bilgi Tabanı Koleksiyonları

| Koleksiyon | İçerik |
|---|---|
| `service_maintenance` | Bakım, arıza, alarm kayıtları (Excel'den, RAG fallback) |
| `service_records_sql` | SQL agent'ının sanal koleksiyonu (gerçek tablo: `service_records`) |
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
OLLAMA_GENERATION_MODEL=gemma3
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=300
DEFAULT_MODEL=ollama:gemma3
DEEPSEEK_API_KEY=               # Opsiyonel

# RAG Ayarları
DEFAULT_ANSWER_LANGUAGE=tr
RAG_TOP_K=5
RAG_ALPHA=0.3                   # BM25/vektör harmanlama oranı
RAG_SCORE_THRESHOLD=0.3         # Fallback eşiği
RAG_MULTI_QUERY=0               # 1=sorgu genişletme aktif
RERANKER_MODEL=                 # Boş = reranker kapalı (önerilen, hız için)

# Cevap Önbelleği
ANSWER_CACHE_ENABLED=true
ANSWER_CACHE_TTL_SECONDS=86400

# Agent Oluşturma
AGENTS_STORE_DIR=agents_store   # Tanım dosyaları klasörü
PENDING_AGENT_TTL_MINUTES=30    # Bekleyen isteğin geçerlilik süresi

# Graf (Opsiyonel)
RAG_USE_GRAPH=0                 # 1=Neo4j aktif
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=test1234

# Excel Çekirdek Veri & Open WebUI Entegrasyonu
EXCEL_SEED_PATH=/app/issue_solution_final.xlsx
OPENWEBUI_DEFAULT_COLLECTION=service_maintenance
OPENWEBUI_WATCH_ENABLED=true
OPENWEBUI_UPLOADS_DIR=/app/openwebui_data/uploads
OPENWEBUI_SCAN_INTERVAL=30
```

---

## Değerlendirme (Eval)

`eval/questions.json` dosyasında ~20 Türkçe soru-anahtar kelime çifti bulunur. Bu çiftler `issue_solution_final.xlsx` verisinden türetilmiştir.

Canlı stack çalışırken:

```bash
# Varsayılan ayarlarla (model: coiltech-rag, url: http://localhost:8000)
python eval/run_eval.py

# Farklı model veya URL ile
python eval/run_eval.py --model coiltech-router --base-url http://localhost:8000
```

Çıktı her soru için `PASS/FAIL`, eksik anahtar kelimeler ve yanıt süresini tablo olarak gösterir. Geçme oranı %60'ın altındaysa çıkış kodu 1 döner.

Birim testleri (servis gerektirmez):

```bash
python -m pytest tests -q          # backend birim testleri
python -m pytest legacy/tests -q   # eski mimarinin testleri (arşiv)
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

### Yanıtlar çok yavaş
- Ollama performans ortam değişkenlerinin ayarlı olduğundan ve Ollama'nın **yeniden başlatıldığından** emin ol (bkz. Kurulum 1. adım)
- `/metrics` ile p95 gecikmeyi ve önbellek isabet oranını izle

### "Agent oluştur" dedim ama yanlış/eski bir dosyaya bağlandı
Son 2 dakika içinde yüklenmiş dosya "yeni" sayılır. Önce mesajı yaz, uyarıyı al, sonra dosyayı yükle — istek dosya gelene kadar (30 dk) bekler.

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

### Doküman güncelledim ama eski cevap geliyor
Cevap önbelleği ingest'te otomatik temizlenir; gerekirse `ANSWER_CACHE_ENABLED=false` yapıp yeniden başlat.

---

## Geliştirme Notları

- **Yeni koleksiyon eklemek**: `backend/pipeline.py` içindeki `_KEYWORD_MAP` sözlüğüne anahtar kelime-koleksiyon eşlemesi ekle
- **Router eşiğini değiştirmek**: `backend/router_agent.py` → `CONFIDENCE_THRESHOLD` (0.55); loglardaki `top3` skorlarıyla ayarla
- **SQL agent few-shot örnekleri**: `backend/sql_agent.py` → `_SQL_PROMPT`
- **Agent oluşturma niyet kalıpları**: `backend/agent_creation.py` → `_INTENT_PATTERNS`
- **Reranker'ı açmak**: `.env`'de `RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Sorgu genişletmeyi aktifleştirmek**: `RAG_MULTI_QUERY=1`
- **Neo4j'yi aktifleştirmek**: `RAG_USE_GRAPH=1` ve Neo4j servisinin çalıştığından emin ol
- **Yeni LLM provider eklemek**: `backend/llm_providers.py` içindeki `LLMProvider` sınıfını genişlet

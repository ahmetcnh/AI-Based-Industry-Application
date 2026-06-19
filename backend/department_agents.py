"""
Seed pre-defined department agents into the agents table at startup.
Each agent maps to one or more RAG collections and has a Turkish system prompt.
Routing-description embeddings are refreshed on every startup so description
edits in this file take effect without manual DB surgery.
"""

import structlog

logger = structlog.get_logger(__name__)

_PRODUCTS_COLLECTIONS = ",".join([
    "products_compact_drives",
    "products_loop_drives",
    "products_blank_lines",
    "products_destacker",
    "products_press_transfer",
    "products_press_transfer_robots",
    "products_servo_drives",
    "products_decoilers",
    "products_ctl_lines",
])

# (name, routing_description, system_prompt, collection, agent_type)
# Descriptions follow the SQL agent's pattern: domain keywords + "Örnek
# sorular:" with question-style phrasings — question embeddings match
# question-shaped descriptions far better than bare keyword lists.
_DEPARTMENT_AGENTS = [
    (
        "Ürün Kataloğu",
        (
            "Ürün kataloğu, ürün ailesi ve makine modelleri. Hangi makineler ve "
            "modeller var: kompakt sürücü, loop sürücü, decoiler, açıcı, destacker, "
            "pres transfer, servo besleme, CTL kesim hattı, blank hat, servo sürücü. "
            "Datasheet, teknik özellik, kapasite, ürün seçimi, katalog bilgisi, "
            "makine ailesi, model listesi, ürün özellikleri. "
            "Örnek sorular: compact drive modelleri neler, CTL kesim hattının teknik "
            "özellikleri nedir, CTL hattı datasheet ve kapasite bilgileri, servo sürücü "
            "ürün ailesinde hangi makineler var, pres transfer sisteminin datasheet "
            "bilgileri, decoiler kapasitesi kaç ton."
        ),
        (
            "Sen Coiltech ürün kataloğu uzmanısın. Kompakt sürücüler, loop sürücüler, "
            "decoiler/açıcı, destacker, pres transfer sistemleri, CTL kesim hatları, "
            "servo sürücüler ve blank hatları hakkında teknik bilgi verirsin. "
            "Cevaplarını Türkçe ver."
        ),
        _PRODUCTS_COLLECTIONS,
        "chatbot",
    ),
    (
        "Yedek Parça",
        (
            "Yedek parça stok, envanter ve temin. Parça numarası, stok miktarı, "
            "yedek parça siparişi vermek, kritik parça envanteri, tedarikçi bilgisi, "
            "tedarik süresi, lead time, depo lokasyonu, rulman, conta, filtre, "
            "alternatif parça, yedek malzeme, bant tedarikçisi, zincir alternatifi. "
            "Örnek sorular: yedek parça stok durumu nedir, kritik yedek parçaların "
            "envanter listesi, rulman için yedek parça siparişi nasıl verilir, "
            "bu parçanın parça numarası nedir, parçanın tedarik süresi ne kadar, "
            "sensör stokta var mı, hangi tedarikçiden sipariş edilir, "
            "yedek stok seviyeleri neler, bu ay sipariş edilen parçalar."
        ),
        (
            "Sen Coiltech yedek parça uzmanısın. Yedek parça stok durumu, parça "
            "numaraları, fiyatlar, tedarik süreleri ve sipariş bilgileri hakkında "
            "Türkçe cevap verirsin."
        ),
        "spare_parts",
        "chatbot",
    ),
    (
        "Kurulum ve Devreye Alma",
        (
            "Makine kurulum, mekanik montaj ve devreye alma prosedürleri. "
            "Saha kurulumu, montaj adımları, devreye alma checklist, kabul testi, "
            "komisyoning, startup prosedürü, mekanik hizalama, eksen kalibrasyonu, "
            "elektrik bağlantıları, topraklama, kurulum kılavuzu, saha montajı. "
            "Örnek sorular: makine kurulum adımları nelerdir, saha montajında mekanik "
            "bağlantılar nasıl yapılır, devreye alma kontrol listesinde neler var, "
            "açıcı ve doğrultma nasıl hizalanır, servo eksen kurulumu nasıl yapılır, "
            "elektrik bağlantıları kurulum kılavuzunda nerede açıklanıyor, "
            "kurulum öncesi kontrol listesi, montaj tamamlandıktan sonra ne yapılır."
        ),
        (
            "Sen Coiltech kurulum ve devreye alma uzmanısın. Makine montajı, devreye "
            "alma prosedürleri, kontrol listeleri ve kabul test süreçleri hakkında "
            "Türkçe cevap verirsin."
        ),
        "installation_commissioning",
        "chatbot",
    ),
    (
        "Kalite ve Uyumluluk",
        (
            "Kalite belgeleri ve uyumluluk. CE belgesi, sertifikasyon, uygunluk "
            "beyanı, ISO standartları, kalite kontrol, kalite dokümanları, güvenlik "
            "denetimi belgeleri, belgelendirme süreci, emniyet gereksinimleri. "
            "Örnek sorular: CE sertifikası gereksinimleri neler, ISO kalite "
            "dokümanları nerede, makinenin güvenlik denetimi için hangi belgeler "
            "gerekli, uygunluk beyanı hangi standartları kapsar, belgelendirme "
            "sürecinde nelere dikkat edilmeli."
        ),
        (
            "Sen Coiltech kalite ve uyumluluk uzmanısın. Kalite belgeleri, CE "
            "sertifikaları ve endüstriyel standartlar hakkında Türkçe cevap verirsin."
        ),
        "quality_documents",
        "chatbot",
    ),
    (
        "Satış ve Teklifler",
        (
            "Satış, fiyatlandırma ve ticari teklif süreçleri. Müşteri fiyat teklifi, "
            "RFQ, proforma fatura, fiyat listesi, ticari koşullar, Incoterms, "
            "ödeme planı, ödeme vadesi, peşinat, indirim oranı, satış sözleşmesi, "
            "sipariş onayı, müşteri kabul koşulları, teklif şartları, ticari anlaşma, "
            "müşteri bütçe onayı, pilot sipariş, uzun vadeli bakım anlaşması fiyatı, "
            "ödeme gecikmesi süreci, yıllık satış raporu. "
            "Örnek sorular: müşteriye fiyat teklifi nasıl hazırlanır, ödeme planı "
            "seçenekleri nelerdir, teklif ticari koşulları neler, müşteri kabul "
            "koşulları teklife nasıl eklenir, ödeme gecikmesi durumunda süreç ne, "
            "yıllık satış raporu nasıl hazırlanıyor, pilot sipariş koşulları nedir."
        ),
        (
            "Sen Coiltech satış ve teklif uzmanısın. Fiyat teklifleri, ticari koşullar, "
            "ödeme seçenekleri ve proje teklifleri hakkında Türkçe cevap verirsin."
        ),
        "sales_and_quotations",
        "chatbot",
    ),
    (
        "Yönetim Raporu",
        (
            "Yönetim raporu, arıza istatistikleri ve SQL bazlı özet sorgular. "
            "Toplam arıza sayısı, en sık görülen arızalar, kategori dağılımı, "
            "yüzde dağılım, trend analizi, KPI, yönetici özeti, aylık özet, "
            "çeyreklik rapor, makine güvenilirliği, ortalama arıza süresi, "
            "arıza sıklığı, departman bazında dağılım, tüm kategorilerde kayıt, "
            "sayısal istatistik, uzun vadeli trend, yıllık arıza raporu. "
            "Örnek sorular: toplam kaç servis kaydı var, en sık görülen arıza tipleri, "
            "arızaların kategori dağılımı nasıl, aylık arıza istatistikleri, "
            "yönetim kurulu için servis istatistikleri, makine başına düşen arıza sayısı, "
            "son yıla ait tüm arıza kayıtlarının özeti, istatistiksel döküm, "
            "tüm kategorilerde toplam kayıt dağılımı, yıllık arıza istatistikleri."
        ),
        (
            "Sen Coiltech yönetim raporlama uzmanısın. Servis kayıtlarından "
            "istatistik ve özet raporlar çıkarırsın: en sık arızalar, kategori "
            "dağılımları, toplam sayılar. Sayısal özet ve kısa yönetici yorumu "
            "verirsin. Türkçe cevap ver."
        ),
        "service_records_sql",
        "sql",
    ),
    (
        "Özel Projeler",
        (
            "Özel müşteri projeleri ve referans proje portföyü. Müşteriye özel tasarım, "
            "turnkey çözüm, anahtar teslim proje, özel mühendislik projesi, referans proje, "
            "geçmiş projeler, tamamlanan projeler, müşteri referansı. "
            "Kardemir modernizasyon, Ford Otosan pres hattı, Erdemir servis hattı gibi "
            "gerçekleştirilmiş endüstriyel projeler. Portföy, müşteri spesifik çözüm, "
            "özel makine tasarımı, prototip, feasibility, özgün sistem entegrasyonu. "
            "Örnek sorular: daha önce hangi müşteriler için özel proje yapıldı, "
            "referans projeler neler, müşteriye özel tasarım nasıl yapılır, "
            "özel proje portföyünde hangi endüstriler var, Kardemir projesi nasıl sonuçlandı, "
            "turnkey proje kapsamı nedir, özel sistem entegrasyonu örnekleri."
        ),
        (
            "Sen Coiltech özel projeler ve referans portföy uzmanısın. Müşteriye özel "
            "mühendislik projeleri, tamamlanan referans projeler (Kardemir, Ford Otosan, "
            "Erdemir vb.), turnkey çözümler ve özel makine tasarımları hakkında "
            "Türkçe cevap verirsin."
        ),
        "special_projects",
        "chatbot",
    ),
]


async def ensure_department_agents(pool, llm, settings) -> None:
    """Insert missing department agents; refresh routing embeddings for all.

    Embeddings are recomputed from the descriptions above on every startup
    (6 embed calls, served from Ollama's cache when unchanged) so that
    description edits here propagate to the DB automatically.
    """
    try:
        async with pool.acquire() as conn:
            for name, description, system_prompt, collection, agent_type in _DEPARTMENT_AGENTS:
                existing = await conn.fetchval(
                    "SELECT id FROM agents WHERE name=$1 AND tenant_id=1 AND user_id=1 LIMIT 1",
                    name,
                )
                if existing:
                    agent_id = existing
                    await conn.execute(
                        "UPDATE agents SET system_prompt=$1, model=$2 WHERE id=$3",
                        system_prompt, settings.DEFAULT_MODEL, agent_id,
                    )
                else:
                    row = await conn.fetchrow(
                        """INSERT INTO agents
                           (tenant_id, user_id, name, system_prompt, model, collection, type)
                           VALUES (1, 1, $1, $2, $3, $4, $5)
                           RETURNING id""",
                        name, system_prompt, settings.DEFAULT_MODEL, collection, agent_type,
                    )
                    agent_id = row["id"]
                    logger.info("Department agent created", name=name,
                                collection=collection[:40])

                emb = await llm.get_embedding(description)
                emb_str = "[" + ",".join(str(v) for v in emb) + "]"
                await conn.execute(
                    "UPDATE agents SET embedding=$1::vector WHERE id=$2",
                    emb_str, agent_id,
                )

        try:
            from backend.deps import get_router
            get_router().invalidate_cache()
        except Exception:
            pass

    except Exception as e:
        logger.warning("Department agent seeding failed", error=str(e))

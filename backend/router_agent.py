import time
from typing import Any, Dict, List, Optional
import structlog

logger = structlog.get_logger(__name__)

CONFIDENCE_THRESHOLD = 0.55
CACHE_TTL = 300  # seconds
# When rank-1 and rank-2 cosine scores are within this gap, the embedding
# alone is considered ambiguous and the LLM classifier is invoked.
AMBIGUITY_THRESHOLD = 0.07

_TR_MAP = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")

# Keyword-based score boosts applied after cosine scoring.
# Catches systematic embedding-space blind spots where a distinctive keyword
# (milestone, CTL+datasheet, yönetim özeti) anchors the question to the right
# agent but embedding similarity alone falls short.
# Rules only fire when the target agent is already in the scored list.
_MACHINE_NAMES = [
    "ctl", "decoiler", "destacker", "blank hat", "pres transfer",
    "servo besleme", "loop surucü", "loop surücü", "kompakt surucü",
    "kompakt surücü", "servo surucü", "servo surücü", "acici", "açici",
    "dogrultma", "doğrultma", "loop surucu", "kompakt surucu", "servo surucu",
]
_SPEC_WORDS = [
    "teknik ozellik", "teknik özelli", "datasheet", "spesifikasyon",
    "konfigurasyonlar", "konfigürasyonlar", "veri sayfas", "voltaji",
    "voltajı", "ailesinde", "modelleri", "hangi makineler", "kapasitesi",
    "maksimum kapasite", "ailesi neler",
]

_DT_WORDS = [
    "dijital ikiz", "makine durumu", "ctl-01", "ctl-02", "ack-01", "ack-02",
    "dst-01", "ptr-01", "sbl-01", "lsd-01", "oee", "sensor", "sensör",
    "canli durum", "canlı durum", "hat durumu", "alarm veriyor", "hangi makine calis",
    "hangi makine çalış", "sicaklik alarmı", "sıcaklık alarmı", "basinc alarmi",
    "basınç alarmı", "rulo agirlik", "rulo ağırlık", "canli izle", "canlı izle",
    "fabrika durumu", "uretim sayaci", "üretim sayacı", "vardiya uretim",
    "vardiya üretim", "makine verisi", "makine verileri",
    "fabrikada kac makine", "fabrikada kaç makine", "kac makine aktif",
    "kaç makine aktif", "en yuksek oee", "en yüksek oee", "fabrika oee",
    "hat oee", "hangi hat en", "son alarm", "en son alarm",
    "aktif mi", "calisıyor mu", "çalışıyor mu", "alarm durumu",
]

_SERVIS_WORDS = [
    "nasil giderilmis", "nasıl giderilmiş", "daha once ne yapilmis", "daha önce ne yapılmış",
    "onceden", "önceden", "daha once", "daha önce", "ariza kaydi", "arıza kaydı",
    "bakim kaydi", "bakım kaydı", "servis kaydi", "servis kaydı", "servis gecmisi",
    "servis geçmişi", "ariza gecmisi", "arıza geçmişi", "bakim gecmisi", "bakım geçmişi",
    "nasil cozulmus", "nasıl çözülmüş", "bu ariza daha", "bu sorun daha", "kayit var mi",
    "kayıt var mı", "tespit edilenler", "yapilan islem", "yapılan işlem",
    "toprak hatasi", "toprak hatası", "enkoder hatasi", "enkoder hatası",
    "vakum hatasi", "vakum hatası", "servo alarmi", "servo alarmı",
    "hava kacagi", "hava kaçağı", "kablo kopu", "kablo kopuğu",
    "nasil mudahale", "nasıl müdahale", "kayitlarda ne var", "kayıtlarda ne var",
    "daha kac kez", "daha kaç kez", "kac kez yasand", "kaç kez yaşand",
    "kapasitor ariza", "kapasitör arıza", "guc modulu ariza", "güç modülü arıza",
]

_YP_WORDS = [
    "stok", "stogu", "stoku", "stoga", "stoğu", "stoğa",
    "parca numarasi", "parça numarası", "tedarik suresi", "tedarik süresi",
    "envanter", "yedek parca", "yedek parça", "lead time", "depo lokasyon",
    "rulman", "conta", "filtre", "siparis ver", "sipariş ver", "yedek stok",
    "kritik parca", "kritik parça", "parca siparis", "parça sipariş",
    "tedarikci", "tedarikçi", "stok miktari", "stok miktarı",
    "alternatif parca", "alternatif parça", "alternatifi var mi", "alternatifi var mı",
    "bant tedarik", "zincir alternati", "stogu tukendi", "stoku tukendi",
    "hangi parcalar mevcut", "hangi parçalar mevcut",
    "inverter alternati", "sigorta siparis", "kontaktor siparis",
    "v-kayis", "v-kayış", "selenoid", "hidrolik pompa stok",
]

_KUR_WORDS = [
    "kurulum", "montaj", "devreye alma", "komisyoning", "kabul testi",
    "hizalama", "konumlandirma", "konumlandırma", "startup", "montaj adim",
    "montaj adım", "mekanik baglanti", "mekanik bağlantı", "kurulum adimlari",
    "kurulum adımları", "kurulum prosedur", "kurulum prosedür", "devreye alma kontrol",
    "devreye alma checklist", "kurulumu nasil", "kurulumu nasıl",
    "saha kurulum", "saha montaj", "kurulum kilavuz", "kurulum kılavuz",
    "elektrik baglantilari", "elektrik bağlantıları", "topraklama prosedur",
    "montaj tamamland", "servo eksen kurul", "devreye alma oncesi",
    "devreye alma öncesi", "kabul testi prosedur",
]

_KALITE_WORDS = [
    "ce belgesi", "ce sertifika", "iso", "uygunluk beyani", "uygunluk beyanı",
    "kalite dokuman", "kalite doküman", "guvenlik denetim", "güvenlik denetim",
    "belgelendirme", "sertifikasyon", "emniyet belgesi", "kalite kontrol belgesi",
    "uyumluluk belgesi", "standart belgesi", "ce uygunluk", "iso standart",
    "kalite belgesi", "guvenlik belgesi", "güvenlik belgesi",
    "ohsas", "en iso", "sertifika gecerli", "sertifika geçerli",
]

_SATIS_WORDS = [
    "fiyat teklifi", "rfq", "proforma", "incoterms", "odeme plani", "ödeme planı",
    "pesinat", "peşinat", "siparis onay", "sipariş onay", "satis sozlesme",
    "satış sözleşme", "teklif hazirla", "teklif hazırla", "satis teklif",
    "satış teklif", "fiyat listesi", "odeme vadesi", "ödeme vadesi",
    "indirim orani", "indirim oranı", "ticari kosul", "ticari koşul",
    "musteri teklif", "müşteri teklif",
    "kabul kosul", "kabul koşul", "musteri sartname", "müşteri şartname",
    "teklif sart", "teklif şart", "odeme gecikmesi", "ödeme gecikmesi",
    "butce onay", "bütçe onay", "pilot siparis", "pilot sipariş",
    "yillik satis rapor", "yıllık satış rapor", "satis raporu", "satış raporu",
    "uzun vadeli bakim anlasma", "uzun vadeli bakım anlaşma",
    "ticari anlasma", "ticari anlaşma", "musteri bilgilendirme",
    "teklif sablonu", "teklif gecerli", "teklif geçerli",
    "teklif onay sureci", "teklif iptali", "teklif revizyonu", "teklif sunumu",
    "sozlesme imzala", "sözleşme imzala", "fiyat gecerli", "siparis formu", "sipariş formu",
    "ihale teklif", "musteri iskonto", "müşteri iskonto",
    "satis temsilcisi", "satış temsilcisi", "komisyon hesab",
    "on odeme", "ön ödeme", "satis hedef", "satış hedef",
    "yurt disi musteri", "yurt dışı müşteri", "teklif karsila", "teklif karşıla",
]

_KEYWORD_BOOSTS = [
    {
        "agent": "Dijital İkiz",
        "any": _DT_WORDS,
        "boost": 0.18,
    },
    {
        "agent": "Servis Kayıtları",
        "any": _SERVIS_WORDS,
        "boost": 0.13,
    },
    {
        "agent": "Yedek Parça",
        "any": _YP_WORDS,
        "boost": 0.16,
    },
    {
        "agent": "Kurulum ve Devreye Alma",
        "any": _KUR_WORDS,
        "boost": 0.13,
    },
    {
        "agent": "Kalite ve Uyumluluk",
        "any": _KALITE_WORDS,
        "boost": 0.14,
    },
    {
        "agent": "Satış ve Teklifler",
        "any": _SATIS_WORDS,
        "boost": 0.17,
    },
    {
        "agent": "Yönetim Raporu",
        "any": [
            "istatistik", "ozet rapor", "aylik ariza ozeti", "toplam kac",
            "kac servis", "yonetim icin", " kpi", "kategori dagilim",
            "aylik ozet", "sayisal ozet", "son ceyrekte", "bu yil kac",
            "ortalama ariza", "ariza sayisi", "makine basina", "yonetici ozeti",
            "trend analiz", "ceyreklik", "istatistiksel dokum", "uzun vadeli ariza",
            "ariza frekan", "ariza sikligi", "ariza orani", "bakim maliyeti",
            "bakım maliyeti", "servis istatistik", "makine guvenirlili",
            "tum kategorilerde", "tüm kategorilerde", "dagilim grafigi",
            "dagilim grafiği", "dağılım grafiği", "yonetim kurulu", "yönetim kurulu",
            "son yila ait", "son yıla ait", "ariza dagilim", "arıza dağılım",
            "ariza rapor", "arıza rapor", "servis ozet", "servis özet",
            "departman bazinda", "departman bazında", "yillik ariza",
            "yıllık arıza", "ariza trend", "bakım kpi",
            # aggregate-only signals
            "ozet nedir", "özet nedir", "ozeti nedir", "özeti nedir",
            "toplam kayit", "toplam kayıt", "toplam ariza", "toplam arıza",
            "kac farkli", "kaç farklı", "kac tip", "kaç tip",
            "dagilimi nasil", "dağılımı nasıl", "dagilim nedir", "dağılım nedir",
            "en sik", "en sık", "en fazla gorule", "en fazla görüle",
            "kategorilerde toplam", "kayit dagilim", "kayıt dağılım",
            "yil icerisinde", "yıl içerisinde", "ariza ozetini",
            # management summary specific
            "ozet hazirla", "özet hazırla", "sayisal dokum", "sayısal döküm",
            "yonetici icin", "yönetici için", "genel mudur", "genel müdür",
            "haftalik ozet", "haftalık özet", "haftalik rapor", "haftalık rapor",
            "kategori bazinda", "kategori bazında", "yuzde dagilim", "yüzde dağılım",
            "buyume orani", "büyüme oranı", "servis sayisi", "servis sayısı",
            "kac tekrar", "kaç tekrar", "en cok tekrar", "en çok tekrar",
            "cozum suresi", "çözüm süresi", "mudahale suresi", "müdahale süresi",
            "kategori bazinda toplam", "kayit sayisi nasil", "kayit ozeti",
            "kayıt özeti", "servis kayitlari ozeti", "servis kayıtları özeti",
        ],
        "boost": 0.20,
    },
    {
        "agent": "Özel Projeler",
        "any": [
            "milestone", "kilometre tasi", "proje takvim", "gantt",
            "proje zaman", "teslim takvim", "wbs", "proje tutanak",
            "proje ilerles", "proje kapsam", "proje fizibilite",
            "proje muhendis", "proje kapani", "proje deliverable",
            "proje charter", "proje tamamlan", "proje ekib",
            "proje butce", "proje bütce", "proje paydas",
            "proje risk", "proje aktivite", "proje basa", "proje ilerleme",
            # extra project signals
            "ozel proje", "özel proje", "referans proje", "turnkey",
            "anahtar teslim", "musteri proje", "müşteri proje",
            "proje basari", "proje başarı", "proje durumu nasil",
            "proje durumu nasıl", "proje onaylan", "proje kickoff",
            "proje son", "proje tamamlanma", "fabrikasyon takvim",
            "proje revize", "proje teklif",
        ],
        "boost": 0.16,
    },
    {
        "agent": "Ürün Kataloğu",
        "any": _MACHINE_NAMES,
        "also": _SPEC_WORDS,
        "boost": 0.15,
    },
]


def _norm(text: str) -> str:
    return text.translate(_TR_MAP).lower()


def _apply_boosts(scored: List[tuple], question_norm: str) -> List[tuple]:
    """Return a new scored list with keyword boosts applied."""
    result = list(scored)
    for rule in _KEYWORD_BOOSTS:
        target = rule["agent"]
        idx = next(
            (i for i, (_, a) in enumerate(result) if a.get("name") == target), None
        )
        if idx is None:
            continue
        any_hit = any(kw in question_norm for kw in rule["any"])
        if not any_hit:
            continue
        also = rule.get("also")
        if also and not any(kw in question_norm for kw in also):
            continue
        score, agent = result[idx]
        result[idx] = (score + rule["boost"], agent)
        logger.debug("keyword_boost_applied", agent=target, boost=rule["boost"])

    return result


_CLASSIFIER_PROMPT = """\
Sen bir Türkçe soru sınıflandırıcısısın. Aşağıdaki agent'lardan hangisinin soruyu cevaplaması gerektiğini belirle.

Agent'lar:
- Servis Kayıtları: geçmiş arıza kaydı arama — "X hatası nasıl çözülmüş?", "daha önce ne yapılmış?"
- Ürün Kataloğu: makine modelleri ve teknik özellikler — CTL, decoiler, servo sürücü, datasheet, kapasite
- Yedek Parça: stok, parça numarası, tedarik süresi, sipariş
- Kurulum ve Devreye Alma: makine kurulumu, montaj, devreye alma, kabul testi, komisyoning
- Kalite ve Uyumluluk: CE belgesi, ISO standartları, uygunluk beyanı, güvenlik denetimi
- Satış ve Teklifler: fiyat teklifi, RFQ, ödeme planı, sipariş, ticari koşullar
- Yönetim Raporu: istatistik, KPI, aylık özet, kategori dağılımı, yönetici raporu
- Özel Projeler: proje durumu, milestone, gantt, WBS, proje takvimi, iş planı
- Dijital İkiz: gerçek zamanlı makine durumu, OEE, sensör verileri, alarm, canlı hat izleme, CTL-01, ACK, PTR, SBL

Soru: {question}

Sadece agent adını yaz, başka hiçbir şey yazma.\
"""


async def _llm_classify(question: str, candidates: List[Dict], llm) -> Optional[Dict]:
    """Ask the LLM to pick the correct agent from the candidates list."""
    prompt = _CLASSIFIER_PROMPT.format(question=question)
    try:
        raw = await llm.generate(prompt)
        raw = raw.strip()
        # Fuzzy match: find the candidate whose name appears in the response
        for _, agent in candidates:
            if agent["name"].lower() in raw.lower():
                return agent
        # Broader: check if any agent name word appears
        for _, agent in candidates:
            first_word = agent["name"].split()[0].lower()
            if first_word in raw.lower():
                return agent
    except Exception as e:
        logger.warning("llm_classifier_failed", error=str(e))
    return None


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RouterAgent:
    def __init__(self, pool, llm, settings):
        self.pool = pool
        self.llm = llm
        self.settings = settings
        self._cache: Dict[str, tuple] = {}  # key -> (agent_row, expires_at)

    def invalidate_cache(self) -> None:
        self._cache.clear()
        logger.info("Routing cache invalidated")

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        agent_row, expires_at = entry
        if time.time() > expires_at:
            del self._cache[key]
            return None
        return agent_row

    def _cache_set(self, key: str, agent_row: Optional[Dict[str, Any]]) -> None:
        self._cache[key] = (agent_row, time.time() + CACHE_TTL)

    async def route(
        self,
        question: str,
        tenant_id: int = 1,
        user_id: int = 1,
    ) -> Optional[Dict[str, Any]]:
        cache_key = question.strip().lower()
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.info("Routing cache hit", agent=cached.get("name"))
            return cached

        question_emb = await self.llm.get_embedding(question)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, system_prompt, model, collection, type, embedding "
                "FROM agents WHERE tenant_id=$1 AND (user_id=$2 OR user_id=1) AND embedding IS NOT NULL",
                tenant_id, user_id,
            )

        if not rows:
            self._cache_set(cache_key, None)
            return None

        scored = []
        for row in rows:
            raw_emb = row["embedding"]
            if raw_emb is None:
                continue
            # asyncpg returns the pgvector column as a string like "[0.1,0.2,...]"
            if isinstance(raw_emb, str):
                agent_emb = [float(x) for x in raw_emb.strip("[]").split(",")]
            else:
                agent_emb = list(raw_emb)
            score = _cosine(question_emb, agent_emb)
            scored.append((score, dict(row)))

        scored = _apply_boosts(scored, _norm(question))
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            logger.warning("No agent embeddings found after scoring")
            self._cache_set(cache_key, None)
            return None

        best_score, best_agent = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        gap = best_score - second_score
        ambiguous = gap < AMBIGUITY_THRESHOLD

        if ambiguous:
            logger.info(
                "Routing ambiguous — invoking LLM classifier",
                top2=[(a["name"], round(s, 3)) for s, a in scored[:2]],
                gap=round(gap, 4),
            )
            llm_pick = await _llm_classify(question, scored, self.llm)
            if llm_pick is not None and llm_pick["name"] != best_agent["name"]:
                logger.info(
                    "LLM classifier overrode embedding",
                    embedding_pick=best_agent["name"],
                    llm_pick=llm_pick["name"],
                )
                best_agent = llm_pick

        logger.info(
            "Routing decision",
            agent=best_agent["name"],
            score=round(best_score, 3),
            gap=round(gap, 4),
            llm_fallback=ambiguous,
            top3=[(a["name"], round(s, 3)) for s, a in scored[:3]],
        )
        self._cache_set(cache_key, best_agent)
        return best_agent

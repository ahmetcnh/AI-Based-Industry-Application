"""
Router accuracy benchmark for the Coiltech RAG Platform.

Replicates RouterAgent's scoring (embed question via Ollama nomic-embed-text,
cosine-match against agent embeddings in Postgres) WITHOUT calling the
generation LLM — no Gemini quota is consumed.

Usage (host machine, stack running):
    python eval/eval_routing.py
    python eval/eval_routing.py --dsn postgresql://postgres:test1234@localhost:5432/ragdb

Outputs total + per-agent accuracy, a confusion table, and
eval/charts/6_router_accuracy.png. Exit code 0 if accuracy >= 85%.
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import asyncpg
import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

QUESTIONS_FILE = Path(__file__).parent / "routing_questions.json"
OUT_DIR = Path(__file__).parent / "charts"
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_DSN = "postgresql://postgres:test1234@localhost:5432/ragdb"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

_TR_MAP = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")

# Must stay in sync with backend/router_agent.py _KEYWORD_BOOSTS
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

_KEYWORD_BOOSTS = [
    {
        "agent": "Yönetim Raporu",
        "any": [
            "istatistik", "ozet rapor", "aylik ariza ozeti", "toplam kac",
            "kac servis", "yonetim icin", " kpi", "kategori dagilim",
            "aylik ozet", "sayisal ozet", "son ceyrekte", "bu yil kac",
            "ortalama ariza", "ariza sayisi", "makine basina", "yonetici ozeti",
            "trend analiz", "ceyreklik", "istatistiksel dokum", "uzun vadeli ariza",
            "ariza frekan", "ariza sikligi", "ariza orani", "bakim maliyeti",
            "servis istatistik", "makine guvenirlili",
            "tum kategorilerde", "tüm kategorilerde", "dagilim grafigi",
            "yonetim kurulu", "yönetim kurulu",
            "son yila ait", "son yıla ait", "ariza dagilim", "arıza dağılım",
            "ariza rapor", "arıza rapor", "servis ozet", "servis özet",
            "departman bazinda", "departman bazında", "yillik ariza",
            "yıllık arıza", "ariza trend", "bakim kpi",
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
        "agent": "Servis Kayıtları",
        "any": [
            "nasil giderilmis", "nasıl giderilmiş",
            "nasil cozulmus", "nasıl çözülmüş",
            "kayitlarda ne var", "kayıtlarda ne var",
            "enkoder hatasi", "enkoder hatası",
            "vakum hatasi", "vakum hatası",
            "hava kacagi", "hava kaçağı", "kablo kopu", "kablo kopuğu",
            "nasil mudahale", "nasıl müdahale",
            "kac kez yasand", "kaç kez yaşand", "daha kac kez", "daha kaç kez",
            "kapasitor ariza", "kapasitör arıza", "guc modulu ariza",
            "yapilan islem", "yapılan işlem",
        ],
        "boost": 0.13,
    },
    {
        "agent": "Satış ve Teklifler",
        "any": [
            "fiyat teklifi", "rfq", "proforma", "incoterms", "odeme plani",
            "pesinat", "peşinat", "satis sozlesme", "satış sözleşme",
            "teklif hazirla", "fiyat listesi", "odeme vadesi", "ödeme vadesi",
            "indirim orani", "ticari kosul", "ticari koşul", "musteri teklif",
            "kabul kosul", "kabul koşul", "odeme gecikmesi", "ödeme gecikmesi",
            "butce onay", "bütçe onay", "pilot siparis", "pilot sipariş",
            "yillik satis rapor", "yıllık satış rapor", "satis raporu",
            "uzun vadeli bakim anlasma", "ticari anlasma", "ticari anlaşma",
            "teklif sablonu", "teklif gecerli", "teklif geçerli",
            "teklif onay sureci", "teklif iptali", "teklif revizyonu", "teklif sunumu",
            "sozlesme imzala", "sözleşme imzala", "sozlesme degisik", "sözleşme değişik",
            "fiyat gecerli", "siparis formu", "sipariş formu",
            "ihale teklif", "musteri iskonto", "müşteri iskonto",
            "satis temsilcisi", "satış temsilcisi", "komisyon hesab",
            "on odeme", "ön ödeme", "satis hedef", "satış hedef",
            "yurt disi musteri", "yurt dışı müşteri", "yurt disi satis",
            "teklif karsila", "teklif karşıla",
        ],
        "boost": 0.17,
    },
    {
        "agent": "Dijital İkiz",
        "any": [
            "dijital ikiz", "ctl-01", "ctl-02", "ack-01", "ack-02",
            "dst-01", "ptr-01", "sbl-01", "lsd-01", "oee",
            "canli durum", "canlı durum", "hat durumu", "alarm veriyor",
            "sicaklik alarmi", "sıcaklık alarmı", "basinc alarmi", "basınç alarmı",
            "fabrika durumu", "uretim sayaci", "üretim sayacı", "makine verileri",
            "fabrikada kac makine", "fabrikada kaç makine", "kac makine aktif",
            "en yuksek oee", "en yüksek oee", "fabrika oee", "hangi hat en",
            "son alarm", "en son alarm", "aktif mi", "alarm durumu",
        ],
        "boost": 0.18,
    },
    {
        "agent": "Yedek Parça",
        "any": [
            "stok", "stogu", "stoku", "stoga", "stoğu", "stoğa",
            "parca numarasi", "parça numarası", "tedarik suresi", "tedarik süresi",
            "envanter", "yedek parca", "yedek parça", "lead time",
            "rulman", "conta", "filtre",
            "siparis ver", "sipariş ver", "siparisi nasil", "siparis nasil",
            "siparis verilebilir", "siparis edilir", "sipariş edilir",
            "kritik parca", "kritik parça", "tedarikci", "tedarikçi",
            "stok miktari", "stok miktarı", "alternatif parca", "alternatif parça",
            "alternatifi var mi", "alternatifi var mı", "bant tedarik",
            "zincir alternati", "stogu tukendi", "stoku tukendi",
            "depo lokasyon", "hangi parcalar mevcut", "hangi parçalar mevcut",
            "inverter alternati", "sigorta siparis", "kontaktor siparis",
            "v-kayis", "v-kayış", "selenoid", "hidrolik pompa stok",
        ],
        "boost": 0.16,
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
            "proje deliverable", "proje son", "proje tamamlanma",
            "fabrikasyon takvim", "proje revize", "proje teklif",
        ],
        "boost": 0.16,
    },
    {
        "agent": "Kurulum ve Devreye Alma",
        "any": [
            "kurulum", "montaj", "devreye alma", "komisyoning", "kabul testi",
            "hizalama", "konumlandirma", "konumlandırma", "startup", "montaj adim",
            "montaj adım", "mekanik baglanti", "mekanik bağlantı", "kurulum adimlari",
            "kurulum adımları", "kurulum prosedur", "devreye alma kontrol",
            "devreye alma checklist", "kurulumu nasil", "kurulumu nasıl",
            "saha kurulum", "saha montaj", "kurulum kilavuz", "kurulum kılavuz",
            "elektrik baglantilari", "elektrik bağlantıları", "topraklama prosedur",
            "montaj tamamland", "servo eksen kurul", "devreye alma oncesi",
            "devreye alma öncesi", "kabul testi prosedur",
        ],
        "boost": 0.13,
    },
    {
        "agent": "Kalite ve Uyumluluk",
        "any": [
            "ce belgesi", "ce sertifika", "iso", "uygunluk beyani", "uygunluk beyanı",
            "kalite dokuman", "kalite doküman", "guvenlik denetim", "güvenlik denetim",
            "belgelendirme", "sertifikasyon", "emniyet belgesi", "kalite kontrol belgesi",
            "uyumluluk belgesi", "standart belgesi", "ce uygunluk", "iso standart",
            "kalite belgesi", "guvenlik belgesi", "güvenlik belgesi", "ohsas",
            "en iso", "sertifika gecerli", "sertifika geçerli",
        ],
        "boost": 0.14,
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


def _apply_boosts(scored: list, question_norm: str) -> list:
    result = list(scored)
    for rule in _KEYWORD_BOOSTS:
        target = rule["agent"]
        idx = next((i for i, (_, n) in enumerate(result) if n == target), None)
        if idx is None:
            continue
        if not any(kw in question_norm for kw in rule["any"]):
            continue
        also = rule.get("also")
        if also and not any(kw in question_norm for kw in also):
            continue
        s, name = result[idx]
        result[idx] = (s + rule["boost"], name)
    return result


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


async def _embed(client: httpx.AsyncClient, text: str):
    resp = await client.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text, "keep_alive": -1},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


async def load_agents(dsn: str):
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            "SELECT name, embedding FROM agents "
            "WHERE tenant_id=1 AND user_id=1 AND embedding IS NOT NULL"
        )
    finally:
        await conn.close()
    agents = []
    for row in rows:
        raw = row["embedding"]
        emb = [float(x) for x in raw.strip("[]").split(",")] if isinstance(raw, str) else list(raw)
        agents.append((row["name"], emb))
    return agents


async def main(dsn: str, questions_path: Path = QUESTIONS_FILE) -> None:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    agents = await load_agents(dsn)
    print(f"Agents loaded: {[n for n, _ in agents]}")
    print(f"Questions: {len(questions)}\n")

    per_agent = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))
    correct = 0

    async with httpx.AsyncClient() as client:
        for q in questions:
            q_emb = await _embed(client, q["question"])
            scored = list((_cosine(q_emb, emb), name) for name, emb in agents)
            scored = _apply_boosts(scored, _norm(q["question"]))
            scored.sort(reverse=True)
            top_score, predicted = scored[0]
            expected = q["expected_agent"]
            ok = predicted == expected
            correct += ok
            per_agent[expected]["total"] += 1
            per_agent[expected]["correct"] += ok
            confusion[expected][predicted] += 1
            mark = "✓" if ok else "✗"
            extra = "" if ok else f"  (expected {expected}, score gap "\
                f"{top_score - next(s for s, n in scored if n == expected):.3f})"
            print(f"  {mark} [{predicted:<24}] {q['question'][:55]}{extra}")

    total = len(questions)
    accuracy = correct / total * 100

    print("\n── Per-agent accuracy ──")
    for agent, st in sorted(per_agent.items()):
        pct = st["correct"] / st["total"] * 100
        print(f"  {agent:<26} {st['correct']}/{st['total']}  ({pct:.0f}%)")

    print("\n── Confusion (expected → predicted, errors only) ──")
    any_err = False
    for exp, preds in confusion.items():
        for pred, n in preds.items():
            if pred != exp:
                any_err = True
                print(f"  {exp}  →  {pred}  ×{n}")
    if not any_err:
        print("  (none)")

    print(f"\nTotal accuracy: {correct}/{total}  ({accuracy:.1f}%)")

    chart(per_agent, accuracy)

    raise SystemExit(0 if accuracy >= 85 else 1)


def chart(per_agent, accuracy):
    names = sorted(per_agent.keys())
    pcts = [per_agent[n]["correct"] / per_agent[n]["total"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(names))
    bars = ax.bar(x, pcts, width=0.55, color="#1976D2",
                  edgecolor="white", linewidth=1.2)
    for bar, p in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{p:.0f}%", ha="center", fontsize=11, fontweight="bold",
                color="#1976D2")

    ax.axhline(85, color="#90CAF9", linewidth=1.2, linestyle="--")
    ax.text(len(names) - 0.5, 86.5, "target 85%", fontsize=9,
            color="#64B5F6", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ve ", " &\n").replace(" ", "\n", 1)
                        for n in names], fontsize=9.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Routing Accuracy (%)", fontsize=11, color="#444444")
    ax.set_title(
        f"Semantic Router Accuracy by Agent  —  Overall: {accuracy:.1f}%",
        fontsize=13, fontweight="bold", color="#111111", pad=12,
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")
    ax.tick_params(colors="#333333")

    plt.tight_layout()
    out = OUT_DIR / "6_router_accuracy.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  ✔ Chart saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=os.environ.get("ROUTER_EVAL_DSN", DEFAULT_DSN))
    parser.add_argument("--questions", default=None, help="path to questions JSON (default: routing_questions.json)")
    args = parser.parse_args()
    questions_path = Path(args.questions) if args.questions else QUESTIONS_FILE
    asyncio.run(main(args.dsn, questions_path))

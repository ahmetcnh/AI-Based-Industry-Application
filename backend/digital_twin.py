"""
Digital Twin — synthetic real-time machine telemetry for Coiltech.

Data is generated deterministically from wall-clock time so every caller
in the same second sees the same values, yet readings evolve naturally over
time without any external state or database.
"""
import math
import socket
import time
import urllib.request
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)


def _detect_public_ip(port: int = 8000) -> str:
    """Try AWS instance metadata, then fall back to outbound-interface IP."""
    # AWS EC2 metadata (IMDSv1 — works without token on most instances)
    for url in [
        "http://169.254.169.254/latest/meta-data/public-ipv4",
        "http://checkip.amazonaws.com",
    ]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "coiltech/1.0"})
            with urllib.request.urlopen(req, timeout=1) as r:
                ip = r.read().decode().strip().splitlines()[0].strip()
                if ip:
                    return f"http://{ip}:{port}"
        except Exception:
            pass
    # Fallback: IP of the interface used for outbound traffic
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return f"http://{s.getsockname()[0]}:{port}"
    except Exception:
        pass
    return f"http://localhost:{port}"


_public_base_url: str = ""  # set by resolve_public_base_url() at startup


def resolve_public_base_url(settings) -> None:
    """Called once at startup to determine the base URL for dashboard links."""
    global _public_base_url
    if settings.PUBLIC_BASE_URL:
        _public_base_url = settings.PUBLIC_BASE_URL.rstrip("/")
    else:
        _public_base_url = _detect_public_ip(settings.API_PORT)
    logger.info("Digital twin dashboard URL", url=_public_base_url + "/dashboard")


def get_dashboard_url() -> str:
    return (_public_base_url or "http://localhost:8000") + "/dashboard"

DIGITAL_TWIN_AGENT_NAME = "Dijital İkiz"
DIGITAL_TWIN_COLLECTION = "digital_twin"
DASHBOARD_PATH = "/dashboard"

_ROUTING_DESCRIPTION = (
    "Dijital ikiz ve makine durum izleme. Gerçek zamanlı sensör verileri, OEE, "
    "sıcaklık, hız, basınç, akım, üretim sayacı ve alarm durumları. "
    "Örnek sorular: CTL-01 hattının durumu nedir, hangi makineler alarm veriyor, "
    "fabrika geneli OEE kaç, açıcı rulo ağırlığı ne kadar, "
    "pres transfer kaç vuruş yapıyor, toplam üretim kaç parça."
)

_SYSTEM_PROMPT_TEMPLATE = (
    "Sen Coiltech fabrikasının dijital ikiz izleme uzmanısın. "
    "Sana verilen anlık sensör verilerine dayanarak makine durumu, OEE, "
    "üretim ve alarm bilgilerini Türkçe açıklarsın. "
    "Cevapların kısa ve net olsun; sayısal değerleri birimleriyle ver. "
    "Cevabının sonuna şu notu ekle: 📊 Canlı dashboard için: {dashboard_url}"
)


def _SYSTEM_PROMPT() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(dashboard_url=get_dashboard_url())


# ─── Machine baseline definitions ────────────────────────────────────────────

class _Machine:
    """Deterministic time-varying sensor readings for one machine."""
    def __init__(self, mid, name, mtype, status, **kw):
        self.id = mid
        self.name = name
        self.type = mtype
        self.base_status = status  # "running" | "idle" | "maintenance"
        self.params = kw           # baseline values + variation

    def _wave(self, t, period, amplitude, phase=0.0):
        return amplitude * math.sin(2 * math.pi * t / period + phase)

    def _noise(self, t, scale, seed):
        # Pseudo-random noise from time + seed
        v = math.sin(t * 127.1 + seed * 311.7) * math.cos(t * 311.7 + seed * 127.1)
        return v * scale

    def reading(self, t: float) -> Dict[str, Any]:
        p = self.params
        r: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "status": self.base_status,
        }

        if self.base_status == "idle":
            r.update({"oee": 0.0, "alarms": [], "notes": "Boşta — rulo değişimi bekleniyor"})
            return r

        if self.base_status == "maintenance":
            r.update({"oee": 0.0, "alarms": ["Planlı bakım"], "notes": "Bakımda"})
            return r

        # Speed / line speed
        if "speed_base" in p:
            spd = p["speed_base"] + self._wave(t, p.get("speed_T", 120), p.get("speed_A", 1.2)) \
                  + self._noise(t, 0.3, 1)
            r["speed_mpm"] = round(max(0, spd), 1)

        # Temperature
        if "temp_base" in p:
            tmp = p["temp_base"] + self._wave(t, 240, p.get("temp_A", 2.5), 1.1) \
                  + self._noise(t, 0.4, 2)
            r["temp_c"] = round(tmp, 1)

        # Hydraulic pressure
        if "hyd_base" in p:
            hyd = p["hyd_base"] + self._wave(t, 180, p.get("hyd_A", 4), 0.8) \
                  + self._noise(t, 0.8, 3)
            r["hyd_bar"] = round(hyd, 1)

        # Motor current
        if "current_base" in p:
            amp = p["current_base"] + self._wave(t, 90, p.get("current_A", 2.5), 2.3) \
                  + self._noise(t, 0.5, 4)
            r["current_a"] = round(max(0, amp), 1)

        # Stroke / cycle rate
        if "stroke_base" in p:
            stk = p["stroke_base"] + self._wave(t, 150, p.get("stroke_A", 2), 0.4) \
                  + self._noise(t, 0.5, 5)
            r["strokes_pm"] = round(max(0, stk), 1)

        # Vacuum
        if "vacuum_base" in p:
            vac = p["vacuum_base"] + self._wave(t, 200, 2.5, 3.0) + self._noise(t, 0.5, 6)
            r["vacuum_kpa"] = round(max(0, vac), 1)

        # Tension
        if "tension_base" in p:
            ten = p["tension_base"] + self._wave(t, 300, p.get("tension_A", 0.8), 1.7) \
                  + self._noise(t, 0.2, 7)
            r["tension_kn"] = round(max(0, ten), 1)

        # Coil diameter (decreasing over shift)
        if "coil_diam_base" in p:
            # Decreases ~300 mm over 8-hour shift
            elapsed_s = t % (8 * 3600)
            diam = p["coil_diam_base"] - (elapsed_s / (8 * 3600)) * 300 \
                   + self._noise(t, 5, 8)
            r["coil_diam_mm"] = round(max(200, diam))
            r["coil_weight_t"] = round(
                max(0.5, p.get("coil_weight_base", 12.0) * (diam / p["coil_diam_base"])), 1
            )

        # Feed length / accuracy
        if "feed_mm_base" in p:
            feed = p["feed_mm_base"] + self._noise(t, 0.05, 9)
            r["feed_mm"] = round(feed, 2)
            r["feed_accuracy_mm"] = round(abs(self._noise(t, 0.04, 10)) + 0.04, 3)

        # Loop height
        if "loop_h_base" in p:
            lh = p["loop_h_base"] + self._wave(t, 60, 30, 0.9) + self._noise(t, 5, 11)
            r["loop_h_mm"] = round(max(50, lh))

        # Production count (grows over shift)
        shift_s = t % (8 * 3600)
        rate = p.get("pcs_per_hour", 300)
        r["production_shift"] = int(shift_s / 3600 * rate + p.get("pcs_offset", 0))

        # OEE (varies ±3%)
        oee = p["oee_base"] + self._wave(t, 600, 1.5, 0.6) + self._noise(t, 0.5, 12)
        r["oee"] = round(min(99, max(60, oee)), 1)

        # Alarms
        alarms: List[str] = []
        temp = r.get("temp_c", 0)
        if "temp_warn" in p and temp > p["temp_warn"]:
            alarms.append(f"Yüksek sıcaklık: {temp}°C (limit {p['temp_warn']}°C)")
        hyd2 = r.get("hyd_bar", 0)
        if "hyd_warn" in p and hyd2 > p["hyd_warn"]:
            alarms.append(f"Yüksek basınç: {hyd2} bar (limit {p['hyd_warn']} bar)")
        r["alarms"] = alarms

        return r


# ─── Machine fleet ────────────────────────────────────────────────────────────

_MACHINES = [
    _Machine("CTL-01", "CTL Kesim Hattı #1", "CTL Hattı", "running",
             speed_base=45.3, speed_T=120, speed_A=1.5,
             temp_base=68.4, temp_warn=78,
             hyd_base=182, hyd_A=5,
             current_base=42.3, current_A=3,
             pcs_per_hour=380, pcs_offset=120, oee_base=91.2),

    _Machine("CTL-02", "CTL Kesim Hattı #2", "CTL Hattı", "running",
             speed_base=38.7, speed_T=140, speed_A=1.2,
             temp_base=74.8, temp_warn=76,
             hyd_base=178, hyd_A=4,
             current_base=38.9, current_A=2.5,
             pcs_per_hour=340, pcs_offset=80, oee_base=87.4),

    _Machine("ACK-01", "Açıcı / Decoiler #1", "Decoiler", "running",
             coil_diam_base=1540, coil_weight_base=14.2,
             speed_base=18.3, speed_T=200, speed_A=0.8,
             tension_base=12.4, tension_A=1.2,
             current_base=22.1, current_A=2,
             pcs_per_hour=0, pcs_offset=0, oee_base=94.1),

    _Machine("ACK-02", "Açıcı / Decoiler #2", "Decoiler", "idle"),

    _Machine("DST-01", "Destacker #1", "Destacker", "running",
             stroke_base=8.3, stroke_A=0.4,
             vacuum_base=85.2,
             temp_base=52.1, temp_warn=70,
             current_base=18.7, current_A=1.5,
             pcs_per_hour=500, pcs_offset=210, oee_base=88.9),

    _Machine("PTR-01", "Pres Transfer #1", "Pres Transfer", "running",
             stroke_base=62, stroke_A=2.5,
             hyd_base=195, hyd_A=6, hyd_warn=210,
             temp_base=74.1, temp_warn=82,
             current_base=55.8, current_A=4,
             pcs_per_hour=420, pcs_offset=60, oee_base=92.7),

    _Machine("SBL-01", "Servo Besleme #1", "Servo Besleme", "running",
             feed_mm_base=320.0,
             speed_base=85.2, speed_T=180, speed_A=1.0,
             current_base=28.4, current_A=2,
             temp_base=48.3, temp_warn=65,
             pcs_per_hour=0, pcs_offset=0, oee_base=96.3),

    _Machine("LSD-01", "Loop Sürücü #1", "Loop Sürücü", "running",
             tension_base=8.5, tension_A=0.6,
             speed_base=42.1, speed_T=160, speed_A=1.1,
             loop_h_base=245,
             temp_base=61.3, temp_warn=75,
             current_base=31.2, current_A=2.5,
             pcs_per_hour=0, pcs_offset=0, oee_base=93.8),
]


def get_status(t: float = None) -> Dict[str, Any]:
    """Return current snapshot of all machines."""
    if t is None:
        t = time.time()
    machines = [m.reading(t) for m in _MACHINES]

    active = [m for m in machines if m["status"] == "running"]
    avg_oee = round(sum(m["oee"] for m in active) / len(active), 1) if active else 0
    all_alarms = [(m["id"], a) for m in machines for a in m.get("alarms", [])]
    total_prod = sum(m.get("production_shift", 0) for m in machines)

    return {
        "timestamp": t,
        "machines": machines,
        "summary": {
            "active_count": len(active),
            "total_count": len(machines),
            "oee_avg": avg_oee,
            "total_production": total_prod,
            "alarm_count": len(all_alarms),
            "alarms": all_alarms,
        },
    }


def build_agent_context(t: float = None) -> str:
    """Build a text block injected into the Digital Twin agent's system prompt."""
    status = get_status(t)
    s = status["summary"]
    lines = [
        f"=== ANLIK MAKİNE VERİLERİ ===",
        f"Aktif makine: {s['active_count']}/{s['total_count']} | "
        f"Ortalama OEE: {s['oee_avg']}% | "
        f"Vardiya üretimi: {s['total_production']:,} parça | "
        f"Aktif alarm: {s['alarm_count']}",
        "",
    ]
    for m in status["machines"]:
        st = {"running": "ÇALIŞIYOR", "idle": "BEKLEMEDE", "maintenance": "BAKIMDA"}.get(
            m["status"], m["status"].upper()
        )
        row = [f"[{m['id']}] {m['name']} — {st}"]
        if m["status"] == "running":
            details = []
            if "speed_mpm" in m:
                details.append(f"Hız: {m['speed_mpm']} m/dak")
            if "temp_c" in m:
                details.append(f"Sıcaklık: {m['temp_c']}°C")
            if "hyd_bar" in m:
                details.append(f"Hidrolik: {m['hyd_bar']} bar")
            if "current_a" in m:
                details.append(f"Akım: {m['current_a']} A")
            if "strokes_pm" in m:
                details.append(f"Vuruş: {m['strokes_pm']}/dak")
            if "vacuum_kpa" in m:
                details.append(f"Vakum: {m['vacuum_kpa']} kPa")
            if "tension_kn" in m:
                details.append(f"Gerginlik: {m['tension_kn']} kN")
            if "coil_diam_mm" in m:
                details.append(f"Rulo Ø: {m['coil_diam_mm']} mm ({m['coil_weight_t']} ton)")
            if "feed_mm" in m:
                details.append(f"Besleme: {m['feed_mm']} mm (±{m['feed_accuracy_mm']} mm)")
            if "loop_h_mm" in m:
                details.append(f"Loop yüksekliği: {m['loop_h_mm']} mm")
            if "oee" in m:
                details.append(f"OEE: {m['oee']}%")
            if "production_shift" in m and m["production_shift"]:
                details.append(f"Vardiya üretimi: {m['production_shift']:,} parça")
            if m.get("alarms"):
                details.append("⚠ ALARM: " + "; ".join(m["alarms"]))
            row.append("  " + " | ".join(details))
        else:
            row.append(f"  {m.get('notes', '')}")
        lines.extend(row)

    if s["alarms"]:
        lines.append("")
        lines.append("AKTIF ALARMLAR:")
        for mid, alarm in s["alarms"]:
            lines.append(f"  ⚠ {mid}: {alarm}")

    return "\n".join(lines)


async def ensure_digital_twin_agent(pool, llm, settings) -> None:
    """Upsert the Digital Twin agent and refresh its routing embedding."""
    try:
        async with pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT id FROM agents WHERE name=$1 AND tenant_id=1 AND user_id=1 LIMIT 1",
                DIGITAL_TWIN_AGENT_NAME,
            )
            if existing:
                agent_id = existing
                # Always refresh type, system_prompt, and embedding so changes
                # in this file propagate without manual DB surgery.
                await conn.execute(
                    """UPDATE agents
                       SET type='digital_twin', system_prompt=$1, model=$2
                       WHERE id=$3""",
                    _SYSTEM_PROMPT(), settings.DEFAULT_MODEL, agent_id,
                )
            else:
                row = await conn.fetchrow(
                    """INSERT INTO agents
                       (tenant_id, user_id, name, system_prompt, model, collection, type)
                       VALUES (1, 1, $1, $2, $3, $4, 'digital_twin') RETURNING id""",
                    DIGITAL_TWIN_AGENT_NAME,
                    _SYSTEM_PROMPT(),
                    settings.DEFAULT_MODEL,
                    DIGITAL_TWIN_COLLECTION,
                )
                agent_id = row["id"]
                logger.info("Digital twin agent created")

            emb = await llm.get_embedding(_ROUTING_DESCRIPTION)
            emb_str = "[" + ",".join(str(v) for v in emb) + "]"
            await conn.execute(
                "UPDATE agents SET embedding=$1::vector WHERE id=$2",
                emb_str, agent_id,
            )
    except Exception as e:
        logger.warning("Digital twin agent setup failed", error=str(e))


# ─── LLM-free direct answer (used when LLM is unavailable) ──────────────────

def _machine_by_id(mid: str, machines: list) -> dict | None:
    mid_low = mid.lower().replace("-", "").replace(" ", "")
    for m in machines:
        if m["id"].lower().replace("-", "") == mid_low:
            return m
    return None


def build_direct_response(question: str, t: float = None) -> str:
    """Generate a structured answer from live data without an LLM."""
    status = get_status(t)
    s = status["summary"]
    machines = status["machines"]
    q = question.lower()

    lines: List[str] = []

    # Which machine is asked about?
    target = None
    for m in machines:
        if m["id"].lower() in q or m["id"].lower().replace("-", "") in q:
            target = m
            break

    if target:
        st = {"running": "Çalışıyor ✅", "idle": "Beklemede ⏸", "maintenance": "Bakımda 🔧"}.get(
            target["status"], target["status"]
        )
        lines.append(f"**{target['id']} — {target['name']}**")
        lines.append(f"Durum: {st}")
        if target["status"] == "running":
            metrics = []
            if "oee" in target:
                metrics.append(f"OEE: **{target['oee']}%**")
            if "speed_mpm" in target:
                metrics.append(f"Hız: {target['speed_mpm']} m/dak")
            if "temp_c" in target:
                metrics.append(f"Sıcaklık: {target['temp_c']}°C")
            if "hyd_bar" in target:
                metrics.append(f"Hidrolik: {target['hyd_bar']} bar")
            if "current_a" in target:
                metrics.append(f"Akım: {target['current_a']} A")
            if "strokes_pm" in target:
                metrics.append(f"Vuruş: {target['strokes_pm']}/dak")
            if "tension_kn" in target:
                metrics.append(f"Gerginlik: {target['tension_kn']} kN")
            if "coil_diam_mm" in target:
                metrics.append(f"Rulo Ø: {target['coil_diam_mm']} mm ({target.get('coil_weight_t','?')} ton)")
            if "feed_mm" in target:
                metrics.append(f"Besleme: {target['feed_mm']} mm (±{target.get('feed_accuracy_mm','?')} mm)")
            if "loop_h_mm" in target:
                metrics.append(f"Loop: {target['loop_h_mm']} mm")
            if "production_shift" in target and target["production_shift"]:
                metrics.append(f"Vardiya üretimi: {target['production_shift']:,} parça")
            lines.append(" | ".join(metrics))
            if target.get("alarms"):
                lines.append("⚠ **ALARM:** " + "; ".join(target["alarms"]))
    else:
        # General factory status
        lines.append(f"**Fabrika Genel Durumu**")
        lines.append(
            f"Aktif makine: {s['active_count']}/{s['total_count']} | "
            f"Ortalama OEE: **{s['oee_avg']}%** | "
            f"Vardiya üretimi: {s['total_production']:,} parça"
        )
        if s["alarms"]:
            lines.append(f"⚠ Aktif alarm: {s['alarm_count']}")
            for mid, alarm in s["alarms"]:
                lines.append(f"  • {mid}: {alarm}")
        else:
            lines.append("✅ Aktif alarm yok")

        # List all machines briefly
        lines.append("")
        for m in machines:
            st = {"running": "✅", "idle": "⏸", "maintenance": "🔧"}.get(m["status"], "?")
            oee_str = f" OEE:{m['oee']}%" if m.get("oee") else ""
            lines.append(f"  {st} **{m['id']}** {m['name']}{oee_str}")

    lines.append(f"\n📊 Canlı dashboard: {get_dashboard_url()}")
    return "\n".join(lines)

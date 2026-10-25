"""Create fictional, reproducible seed data for the public demo.

Requires openpyxl and reportlab. Every row and document is generated from
the constants below; no private workbook or document is read.
"""

from pathlib import Path

from openpyxl import Workbook
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "demo_docs"
COUNT = 781

CASES = [
    ("servo sürücü aşırı akım alarmı", "Sürücü akım sınırı ve mekanik yük kontrol edildi; test reçetesiyle yeniden devreye alındı."),
    ("enkoder geri besleme hatası", "Enkoder soketi ve kablo sürekliliği ölçüldü; referanslama işlemi tekrarlandı."),
    ("vakum tutma basıncı düşük", "Vakum hortumu ve valf kontrol edildi; filtre temizlenip tutma testi yapıldı."),
    ("hidrolik basınç kararsız", "Hidrolik filtre ve valf ayarı kontrol edildi; basınç sensörü kalibre edildi."),
    ("toprak hatası uyarısı", "Topraklama bağlantıları ve izolasyon ölçümü kontrol edildi; gevşek klemens sıkıldı."),
    ("sensör sinyali kesiliyor", "Sensör hizası ve kablo ekranlaması kontrol edildi; giriş sinyali test edildi."),
    ("kablo kopukluğu alarmı", "Kablo demeti sürekliliği ölçüldü; hasarlı örnek kablo değiştirildi."),
    ("motor sıcaklığı yüksek", "Motor fanı ve rulman kontrol edildi; yük sınırı test edildi."),
    ("servo besleme boyu sapması", "Besleme merdaneleri ve enkoder oranı kontrol edilip kalibrasyon yapıldı."),
    ("PLC haberleşme kesintisi", "Ağ bağlantısı ve cihaz adresi doğrulandı; haberleşme testi tekrarlandı."),
    ("acil durdurma zinciri açık", "Emniyet rölesi ve kapı anahtarı test edildi; güvenli başlatma doğrulandı."),
    ("kesme bıçağı konum hatası", "Bıçak referansı ve limit sensörü kontrol edilip deneme kesimi yapıldı."),
]
MODELS = ("CTL-01", "CTL-02", "ACK-01", "DST-01", "PTR-01", "SBL-01")
CONDITIONS = (
    "ilk devreye alma", "periyodik bakım", "reçete değişimi", "vardiya başlangıcı",
    "bobin değişimi", "deneme üretimi", "planlı duruş sonrası",
)

PDFS = {
    "kalite_uyumluluk_dokumani.pdf": (
        "Kalite ve Uyumluluk - Kurgusal Ornek",
        ["Bu belge yalnızca demo amaçlıdır; sertifika veya uygunluk beyanı değildir.",
         "Örnek pres hattı için risk değerlendirmesi, koruyucu kapak ve acil durdurma testi kaydedilir.",
         "Kabul dosyasında elektrik şeması, test tutanağı ve bakım yönergesi listelenir.",
         "Gerçek uygunluk değerlendirmesi yetkili uzman ve güncel mevzuatla yapılmalıdır."],
    ),
    "kurulum_devreye_alma_kilavuzu.pdf": (
        "Kurulum ve Devreye Alma - Kurgusal Ornek",
        ["Örnek CTL-01 hattında mekanik hizalama ve temel bağlantıları kontrol edilir.",
         "Enkoder yönü, sensör girişleri ve acil durdurma zinciri test edilir.",
         "Boşta deneme, düşük hızlı deneme ve reçete doğrulaması sırayla kaydedilir.",
         "Buradaki değerler gerçek bir makinenin kurulum talimatı değildir."],
    ),
    "ozel_projeler_portfoyu.pdf": (
        "Ozel Projeler Portfoyu - Kurgusal Ornek",
        ["Bu portföydeki bütün müşteriler ve projeler kurgusaldır.",
         "Örnek Müşteri A: bobin açıcı ile kesme hattı arasında örnek entegrasyon.",
         "Örnek Müşteri B: pres transfer hücresinde çift sac algılama senaryosu.",
         "Örnek Müşteri C: destacker ve robot besleme senkronizasyonu.",
         "Proje sonuçları, tarihler ve üretim sayıları gerçek referans değildir."],
    ),
    "satis_ve_teklifler.pdf": (
        "Satis ve Teklifler - Kurgusal Ornek",
        ["Bu belge fiyat listesi veya bağlayıcı ticari teklif değildir.",
         "Örnek teklif kapsamı: CTL-01 kesme hattı, operatör eğitimi ve kabul testi.",
         "Örnek ödeme planı: sipariş, sevkiyat ve kabul aşamalarında üç taksit.",
         "Teslim tarihi, garanti ve servis koşulları gerçek sözleşmede ayrıca belirlenir."],
    ),
    "urun_katalogu_servo_ctl.pdf": (
        "Urun Katalogu - Kurgusal Ornek",
        ["CTL-01, SBL-01 ve CD-300 demo model kodlarıdır.",
         "CTL-01: bobin açma, doğrultma ve boy kesme süreçlerini temsil eder.",
         "SBL-01: servo besleme ile reçete tabanlı adım kontrolünü temsil eder.",
         "Kapasite ve tolerans bilgileri gerçek ürün spesifikasyonu sayılmaz."],
    ),
    "yedek_parca_katalogu.pdf": (
        "Yedek Parca Katalogu - Kurgusal Ornek",
        ["Bu listedeki parça kodları ve stok değerleri yalnızca demo içindir.",
         "DEMO-ENC-01: örnek enkoder kiti; DEMO-VAC-01: örnek vakum valfi.",
         "DEMO-BRG-01: örnek rulman seti; DEMO-SEN-01: örnek yaklaşım sensörü.",
         "Sipariş ve uygunluk için gerçek makine seri numarası doğrulanmalıdır."],
    ),
}

TEXTS = {
    "ce_compliance_press_lines.txt": """Synthetic CE checklist example
This is not a certificate or legal declaration.
Demo press line: document the risk assessment, guard checks, emergency-stop test,
and electrical inspection before a fictional acceptance test.
""",
    "compact_drive_selection_guide.txt": """Synthetic Compact Drive Selection Guide
CD-300 and CD-600 are demo model codes. Compare strip width, material thickness,
feed speed, and servo accuracy before selecting a fictional line configuration.
All capacities here are illustrative and are not product specifications.
""",
    "ctl_commissioning_checklist.txt": """Synthetic Cut-to-Length Commissioning Checklist
Mechanical: check alignment, foundation fasteners, and shear clearance.
Electrical: test emergency stop, encoder direction, and PLC recipe transfer.
Process: run a low-speed dry cycle and record a fictional sample cut.
This is not an operating procedure for a real machine.
""",
    "quotation_terms_2026.txt": """Synthetic Commercial Terms Example
This is not a quotation, price list, or contract.
Demo quote: define scope, acceptance criteria, delivery milestone, and three
illustrative payment stages. No customer or actual price is represented.
""",
    "servo_drive_alarm_matrix.txt": """Synthetic Servo Drive Alarm Matrix
E-101 servo overload: inspect fictional test load, feed rolls, and encoder.
E-214 encoder drift: inspect coupling, reference mark, and feed calibration.
These codes and steps are demo examples, not a manufacturer fault manual.
""",
    "spare_parts_critical_stock.txt": """Synthetic Spare Parts Stock Example
DEMO-ENC-01 encoder kit: illustrative reorder level 2.
DEMO-VAC-01 vacuum valve: illustrative reorder level 2.
DEMO-BRG-01 bearing set: illustrative reorder level 4.
These are fictional codes and stock levels.
""",
    "special_project_destacker_layout.txt": """Synthetic Destacker Integration Concept
Demo destacker feeds a transfer conveyor and reports a blank counter to a PLC.
A fictional double-sheet sensor stops the simulated line for inspection.
No customer, site, or completed project is represented.
""",
}


def make_workbook() -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(("tespit_edilenler", "yapilan_islem"))
    for number in range(1, COUNT + 1):
        issue, action = CASES[(number - 1) % len(CASES)]
        model = MODELS[((number - 1) // len(CASES)) % len(MODELS)]
        condition = CONDITIONS[((number - 1) // (len(CASES) * len(MODELS))) % len(CONDITIONS)]
        sheet.append((
            f"Demo kayıt {number:03d}: {model} - {condition} sırasında {issue}.",
            f"Kurgusal servis işlemi {number:03d}: {action}",
        ))
    workbook.save(ROOT / "issue_solution_final.xlsx")


def pdf_font() -> str:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("DemoSans", str(path)))
            return "DemoSans"
    raise RuntimeError("A Unicode TTF font (Arial or DejaVu Sans) is required")


def make_pdfs() -> None:
    font = pdf_font()
    for filename, (title, paragraphs) in PDFS.items():
        document = canvas.Canvas(str(DOCS / filename), pagesize=(595, 842), invariant=1)
        document.setTitle(title)
        document.setAuthor("Public demo data generator")
        document.setFont(font, 16)
        document.drawString(48, 780, title)
        document.setFont(font, 10)
        y = 742
        for paragraph in paragraphs:
            for line in simpleSplit(paragraph, font, 10, 495):
                document.drawString(48, y, line)
                y -= 16
            y -= 20
        document.setFont(font, 8)
        document.drawString(48, 45, "Tamamen sentetik demo icerigi. Gercek musteri veya servis verisi icermez.")
        document.save()


def make_texts() -> None:
    for filename, content in TEXTS.items():
        with (DOCS / filename).open("w", encoding="utf-8", newline="\n") as output:
            output.write(content)
    with (ROOT / "data" / "demo_docs_test" / "compact_drive_selection_guide.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        output.write(TEXTS["compact_drive_selection_guide.txt"])


if __name__ == "__main__":
    make_workbook()
    make_pdfs()
    make_texts()

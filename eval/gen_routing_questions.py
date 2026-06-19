"""Generate ~500 routing questions from templates and save to routing_questions_500.json."""
import json, random
from pathlib import Path

random.seed(42)
OUT = Path(__file__).parent / "routing_questions_500.json"

# ── Servis Kayıtları ──────────────────────────────────────────────────────────
HATALAR = ["toprak hatası", "enkoder hatası", "servo alarmı", "vakum hatası",
           "ASI toprak hatası", "faz sırası hatası", "güç modülü arızası",
           "sensör hatası", "hava kaçağı", "kablo kopuğu", "motor arızası",
           "kapasitör arızası", "kontaktör arızası", "sigorta atması",
           "hidrolik kaçağı", "basınç düşüşü", "sıcaklık alarmı",
           "ekran donması", "PLC hatası", "inverter alarmı", "servo sürücü alarmı",
           "pozisyon hatası", "limit switch arızası", "selenoid valf hatası"]
MAKINE = ["destacker", "transfer hattı", "pres", "açıcı", "decoiler",
          "CTL hattı", "servo konveyör", "blank hat", "rollform",
          "doğrultma makinesi", "loop sürücü", "pres transfer"]
EYLEM = ["nasıl çözülmüş", "nasıl giderilmiş", "ne yapılmış",
         "için hangi işlem yapılmış", "nasıl müdahale edilmiş",
         "için kayıtlarda ne var", "daha önce nasıl düzeltilmiş"]

servis_qs = []
for h in HATALAR:
    for e in random.sample(EYLEM, 2):
        servis_qs.append(f"{h.capitalize()} {e}?")
for m in MAKINE:
    for h in random.sample(HATALAR, 3):
        servis_qs.append(f"{m.capitalize()}'da {h} {random.choice(EYLEM)}?")
servis_qs += [
    "DT35 sensörü takıldıktan sonra makine yeşile alındı fakat otomatik moda geçmiyor, kayıtlarda ne var?",
    "Reçete yükle butonuna basınca panel kapanıp açılıyor, bu sorun daha önce çözülmüş mü?",
    "Giyotin makas arızası geçmişte nasıl müdahale edilmiş?",
    "Transfer barında M8 valf soketi bağlantı sorunu için ne yapılmış?",
    "Pres ve rollform arasındaki loop hatası kayıtlarda var mı?",
]
servis_qs = list(dict.fromkeys(servis_qs))[:65]

# ── Ürün Kataloğu ─────────────────────────────────────────────────────────────
URUNLER = ["CTL kesim hattı", "decoiler", "açıcı", "destacker", "pres transfer",
           "servo besleme", "kompakt sürücü", "loop sürücü", "blank hat",
           "servo sürücü", "doğrultma makinesi", "koil aktarma arabası"]
OZELLIK = ["teknik özellikleri nedir", "datasheet bilgileri nerede",
           "kapasite değerleri neler", "modelleri nelerdir", "ailesi neler içeriyor",
           "maksimum kapasitesi kaç ton", "çalışma voltajı nedir",
           "hangi model seçilmeli", "konfigürasyonları neler"]

urun_qs = []
for u in URUNLER:
    for o in random.sample(OZELLIK, 4):
        urun_qs.append(f"{u.capitalize()} {o}?")
urun_qs += [
    "Compact drive ürün ailesinde hangi modeller var?",
    "Servo sürücü serisi hangi uygulamalar için uygundur?",
    "CTL hattı ile blank hat arasındaki fark nedir?",
    "Hangi makineler 10 ton üzeri kapasiteye sahip?",
    "Pres transfer robotlu sistemin katalog sayfası nerede?",
    "Loop sürücü teknik veri sayfası nasıl indirilir?",
    "Destacker modelleri ve kapasiteleri nelerdir?",
    "Doğrultma makinesi spesifikasyonları nedir?",
]
urun_qs = list(dict.fromkeys(urun_qs))[:63]

# ── Yedek Parça ───────────────────────────────────────────────────────────────
PARCALAR = ["rulman", "conta", "filtre", "sigorta", "selenoid valf",
            "servo motoru", "enkoder", "sensör", "kontaktör",
            "inverter", "kapasitör", "zincir", "bant", "v-kayış",
            "hidrolik pompa", "basınç valfi", "pnömatik silindir"]
STOK_EYLEM = ["stok durumu nedir", "parça numarası nedir", "tedarik süresi ne kadar",
              "siparişi nasıl verilir", "stoğu tükendi mi", "alternatifi var mı",
              "kritik stok seviyesi nedir", "tedarikçisi kimdir"]

yp_qs = []
for p in PARCALAR:
    for e in random.sample(STOK_EYLEM, 3):
        yp_qs.append(f"{p.capitalize()} {e}?")
yp_qs += [
    "Kritik yedek parçaların tam envanter listesi nedir?",
    "Acil durum için hangi parçaları stokta bulundurmak gerekir?",
    "Yedek parça siparişi için onay süreci nasıl işler?",
    "En uzun tedarik süresine sahip parça hangisi?",
    "Depo lokasyonlarında hangi parçalar mevcut?",
    "Bu ay sipariş edilen yedek parçalar neler?",
]
yp_qs = list(dict.fromkeys(yp_qs))[:63]

# ── Kurulum ve Devreye Alma ───────────────────────────────────────────────────
KUR_KONULAR = [
    "Makine kurulum adımları nelerdir?",
    "Saha montajında mekanik bağlantılar nasıl yapılır?",
    "Devreye alma checklist'inde neler var?",
    "Kabul testi prosedürü nasıl ilerler?",
    "Açıcı ve doğrultma hizalaması nasıl yapılır?",
    "Pres transfer kurulumunda dikkat edilmesi gerekenler neler?",
    "CTL hattı ilk devreye almada hangi adımlar izlenir?",
    "Servo sistem konfigürasyonu nasıl yapılır?",
    "Elektrik bağlantıları kurulum kılavuzunda nerede açıklanmış?",
    "Komisyoning sonrası kabul testleri neler?",
    "Temel hizalama toleransları ne kadar?",
    "Pnömatik sistem devreye alma prosedürü nedir?",
    "Hidrolik sistem ilk doldurma adımları neler?",
    "Güvenlik devresi testi kurulumda nasıl yapılır?",
    "Makine konumlandırma ve sabitlenme adımları neler?",
    "Kontrol paneli montajı hangi adımları içerir?",
    "İlk çalıştırmada yapılacak testler neler?",
    "Startup prosedürü sırasında hangi parametreler kontrol edilmeli?",
    "Makine nakliye sonrası kontrol listesi nedir?",
    "Saha kabulü için hangi belgeler hazırlanmalı?",
    "Devreye alma sırasında hangi güvenlik önlemleri alınmalı?",
    "Decoiler kurulum adımları nelerdir?",
    "Transfer hattı mekanik kurulum prosedürü nedir?",
    "Besleme ünitesi devreye alma kontrol listesi nerede?",
    "Destacker montaj kılavuzu hangi bölümleri içeriyor?",
    "Sistemin ilk testinde hangi alarm setleri kontrol edilmeli?",
    "Blank hat kurulum talimatları nasıl temin edilir?",
    "Kurulum sonrası garanti şartları nelerdir?",
    "Konveyör hız kalibrasyonu nasıl yapılır?",
    "Makine devreye almada müşteri onay süreci nedir?",
]
kur_qs = KUR_KONULAR[:]
kur_qs += [
    "Doğrultma ünitesi hizalama toleransları neler?",
    "Servo eksen kurulumu için adımlar nelerdir?",
    "Pres transfer robot kurulumunda kritik noktalar neler?",
    "Elektrik topraklama prosedürü kurulum kılavuzunda nerede?",
    "İlk çalışma öncesi güvenlik kontrolleri nelerdir?",
    "Makine komisyoning belgesi nasıl doldurulur?",
    "CTL hattı hız kalibrasyonu nasıl yapılır?",
    "Hydrolik sistem basınç ayarı devreye almada nasıl yapılır?",
    "Yükleme rampa kurulum adımları nelerdir?",
    "Temel montaj cıvataları ne kadar tork uygulanır?",
    "İlk devreye almada hangi yazılım ayarları yapılmalı?",
    "Pozisyon sensörleri kurulumda nasıl kalibre edilir?",
    "Makine teslim protokolü neleri içerir?",
    "Devreye alma öncesi elektrik testleri neler?",
    "Güvenlik kapıları kurulumda nasıl test edilir?",
    "Rollform hattı geometri ayarı nasıl yapılır?",
    "Servo sürücü parametre ayarı kurulumda nasıl yapılır?",
    "İlk test koşumunda nelere dikkat edilmeli?",
    "Pres kurs ayarı devreye almada nasıl yapılır?",
    "Makinenin kabul testini kim yapar?",
    "Devreye alma checklist onayı kim tarafından verilir?",
    "Saha mühendisi kurulum raporunu nereye gönderir?",
    "Montaj sonrası mekanik kontrol prosedürü nedir?",
    "Kurulum sırasında hangi ölçüm aletleri kullanılır?",
    "Makine satış sonrası kurulum desteği nasıl talep edilir?",
    "Kurulum hataları için garanti kapsamı nedir?",
    "Devreye alma protokolü müşteriye nasıl sunulur?",
    "Sistemin enerji analizatörüyle devreye alma testi nasıl yapılır?",
    "Kurulum tamamlanınca bakım ekibine handover nasıl yapılır?",
    "Remote devreye alma desteği nasıl alınır?",
    "CE kurulum uygunluğu için hangi belgeler gerekli?",
    "Makinenin zemin ankrajı nasıl yapılır?",
]
kur_qs = list(dict.fromkeys(kur_qs))[:63]

# ── Kalite ve Uyumluluk ───────────────────────────────────────────────────────
STANDARTLAR = ["ISO 9001", "ISO 13849", "EN 60204", "CE", "ATEX", "IEC 62061",
               "ISO 14001", "OHSAS 18001", "EN ISO 10218"]
kalite_qs = []
for st in STANDARTLAR:
    kalite_qs += [
        f"{st} standardı gereksinimleri nelerdir?",
        f"Makinenin {st} uygunluğu nasıl belgelenir?",
        f"{st} sertifikası için hangi testler yapılmalı?",
    ]
kalite_qs += [
    "CE belgesi için hangi belgeler gerekli?",
    "Uygunluk beyanı hangi standartları kapsar?",
    "Makinenin güvenlik denetimi için hangi belgeler hazırlanmalı?",
    "ISO kalite dokümanları nerede saklanıyor?",
    "Kalite kontrol prosedürü nasıl işler?",
    "Güvenlik risk analizi nasıl yapılır?",
    "Makinenin CE işaretlemesi süreci nasıl ilerler?",
    "Güvenlik fonksiyonları nasıl doğrulanır?",
    "Belgelendirme sürecinde teknik dosya neler içermeli?",
    "Periyodik güvenlik denetimi ne kadar sıklıkla yapılmalı?",
    "Makine direktifi gereksinimleri nelerdir?",
    "Acil durdurma devresi hangi standarda göre tasarlanmalı?",
    "Risk değerlendirmesi raporu nasıl hazırlanır?",
    "Makinenin ömür boyu güvenlik planı nedir?",
    "Kalite yönetim sistemi hangi prosedürleri içeriyor?",
    "Üretim sürecinde kalite kontrol noktaları neler?",
    "Uygunsuzluk raporu nasıl doldurulur?",
    "İç denetim prosedürü nasıl işler?",
    "Tedarikçi kalite gereksinimleri nelerdir?",
    "Sertifikasyon belgesi geçerlilik süresi ne kadar?",
    "Makine güvenlik kategorisi nasıl belirlenir?",
    "PL (Performance Level) hesabı nasıl yapılır?",
    "Güvenlik röleleri hangi standarda göre seçilmeli?",
    "Işık bariyeri kurulumu için hangi standartlar geçerli?",
]
kalite_qs = list(dict.fromkeys(kalite_qs))[:63]

# ── Satış ve Teklifler ────────────────────────────────────────────────────────
satis_qs = [
    "Müşteriye CTL hattı için fiyat teklifi nasıl hazırlanır?",
    "RFQ süreci nasıl işler?",
    "Proforma fatura ne zaman düzenlenir?",
    "Ödeme planı seçenekleri nelerdir?",
    "Peşinat oranı standart olarak ne kadar?",
    "İndirim oranı yetkisi kimin onayına bağlı?",
    "Satış sözleşmesi hangi maddeleri içermeli?",
    "Sipariş onayı alındıktan sonra süreç nasıl işler?",
    "Incoterms seçeneği müşteriye nasıl sunulur?",
    "Teklifin geçerlilik süresi ne kadar?",
    "Ticari koşullar nasıl belirleniyor?",
    "Müşteri fiyat listesine nasıl ulaşılır?",
    "Ödeme vadesi ne kadar uzatılabilir?",
    "Teklif hazırlama için hangi teknik bilgiler gerekli?",
    "Yurt dışı satışlarda döviz kuru nasıl belirleniyor?",
    "Teklif revizyonu nasıl yapılır?",
    "Müşteri kabul koşulları teklife nasıl eklenir?",
    "Satış hedefleri nasıl takip ediliyor?",
    "CRM sistemine teklif nasıl girilir?",
    "Müşteri iskonto talebi nasıl değerlendiriliyor?",
    "Spot teklif için süreç ne kadar sürüyor?",
    "Yedek parça satış teklifi nasıl hazırlanır?",
    "Servis sözleşmesi teklifi neleri kapsar?",
    "Müşteriye teknik teklif sunumu nasıl yapılır?",
    "Teklif onay süreci kaç aşamadan oluşuyor?",
    "Fiyat geçerliliği uzatma talepleri nasıl yönetiliyor?",
    "Sipariş formu nerede dolduruluyor?",
    "Müşteri ön ödeme nasıl alınıyor?",
    "Sözleşme değişikliği talepleri nasıl işleniyor?",
    "Yıllık satış raporu nasıl hazırlanıyor?",
    "Fiyat teklifi hazırlamak için mühendis desteği nasıl talep edilir?",
    "Müşteri özel koşulları teklife nasıl yansıtılır?",
    "Teklif şablonu nerede bulunuyor?",
    "Satış temsilcisi komisyon hesabı nasıl yapılıyor?",
    "Müşteri bütçe onayı alındıktan sonra ne yapılıyor?",
    "Toplu sipariş indirimi ne kadar?",
    "Ödeme gecikmesi durumunda süreç nasıl işler?",
    "Müşteriye katalog fiyat listesi nasıl gönderilir?",
    "Teklif karşılaştırma tablosu nasıl hazırlanır?",
    "Dövizli teklif nasıl TL'ye çevrilir?",
    "Müşteri sözleşme imzaladıktan sonra bilgilendirme nasıl yapılır?",
    "Satış sonrası destek paketi teklife eklenebilir mi?",
    "Rekabetçi teklif durumunda indirim politikası nedir?",
    "RFQ yanıt süresi kaç gündür?",
    "Müşteri onay vermezse teklif ne olur?",
    "Yurt dışı müşteri için özel belge gereksinimleri neler?",
    "Satış hedefi tutturulmazsa ne yapılır?",
    "Müşteri şikayeti satış sürecini nasıl etkiler?",
    "Teklif iptali durumunda süreç nedir?",
    "Uzun vadeli bakım anlaşması teklife dahil edilebilir mi?",
    "Satış ekibi yeni fiyat listesine nasıl erişir?",
    "Müşteri ön inceleme ziyareti nasıl planlanır?",
    "Teklif sunumunda hangi teknik belgeler eklenmeli?",
    "Satış süreci yönetim raporu nasıl hazırlanır?",
    "Müşteri referansı nasıl talep edilir?",
    "İhale tekliflerinde hangi belgeler gerekmektedir?",
    "Müşteri pilot sipariş koşulları nelerdir?",
    "Garanti şartları teklife nasıl ekleniyor?",
    "Teklif kabul oranı nasıl takip ediliyor?",
    "Sözleşme imzalandıktan sonra üretim süreci başlıyor mu?",
    "Teklif hazırlama maliyeti müşteriye yansıtılır mı?",
    "Müşteri özel baskı talebi teklife nasıl giriyor?",
    "Satış hedefi aşıldığında bonus politikası nedir?",
]
satis_qs = list(dict.fromkeys(satis_qs))[:63]

# ── Yönetim Raporu ────────────────────────────────────────────────────────────
yonetim_qs = [
    "En sık görülen 5 arıza tipi nedir?",
    "Toplam kaç servis kaydı var?",
    "Arızaların kategori dağılımı nasıl?",
    "Yönetim için aylık arıza özeti çıkarır mısın?",
    "Servo alarmı kayıtlarının istatistiksel özeti nedir?",
    "Bu yıl kaç arıza kaydı açıldı?",
    "Hangi makinede en çok servis kaydı var?",
    "Aylık arıza trend analizi yapabilir misin?",
    "En çok tekrarlanan 3 sorun tipi nedir?",
    "Arızaların çözüm süreleri ortalaması nedir?",
    "KPI raporu için arıza verileri lazım.",
    "Yönetim kuruluna sunmak için özet istatistik tablosu hazırla.",
    "Geçen ay kaç servis talebi geldi?",
    "Hidrolik arızaları toplam kayıtların yüzde kaçını oluşturuyor?",
    "Servo alarmları hangi makinelerde en çok görülüyor?",
    "Son çeyrekte arıza sayısı arttı mı azaldı mı?",
    "Makine bazında arıza frekans raporu oluştur.",
    "Genel müdür için haftalık arıza özeti lazım.",
    "Arıza tiplerine göre sayısal döküm tablosu çıkar.",
    "Bakım maliyetleri için arıza kategorisi dağılımı nedir?",
    "Toplam servis kayıtlarının yüzde dağılımı nedir?",
    "En uzun süren arıza tipleri neler?",
    "Aylık arıza yüzde dağılımı tablosu hazırla.",
    "Yönetim için istatistiksel arıza analizi raporunu çıkar.",
    "Hangi arıza tipi en çok tekrar ediyor?",
    "Çeyreklik arıza istatistikleri nelerdir?",
    "Kategori bazında toplam servis saatleri nedir?",
    "Bu ay en fazla kaydedilen sorun tipi hangisi?",
    "Arıza sıklığı aydan aya nasıl değişiyor?",
    "Servis kaydı sayısı son 6 ayda nasıl seyretmiş?",
    "Makine başına düşen ortalama arıza sayısı nedir?",
    "Kritik arıza oranı yüzde kaç?",
    "Arıza kayıtlarından yönetici özeti oluştur.",
    "Toplam servis kaydı sayısını kategori bazında göster.",
    "Genel arıza eğilimi raporunu hazırla.",
    "Yıllık arıza istatistikleri nasıl görünüyor?",
    "Bakım ekibi performans KPI'ları neler?",
    "İstatistiksel döküme göre hangi sistem en sorunlu?",
    "Arıza başına ortalama müdahale süresi nedir?",
    "Yönetim raporu için arıza sayım tablosu çıkar.",
    "Tüm kategorilerde toplam kayıt dağılımı nedir?",
    "Haftalık servis kayıtları özeti hazırlar mısın?",
    "Arızaların departman bazında dağılımı nedir?",
    "Uzun vadeli arıza trend analizi nedir?",
    "Makine verimliliği için arıza sıklığı istatistiği ver.",
    "Yönetim kurulu sunumu için servis istatistikleri lazım.",
    "Toplam kaç farklı arıza tipi kayıt altına alınmış?",
    "En kritik 5 arıza tipi için istatistik raporla.",
    "Önleyici bakım için arıza istatistikleri nelerdir?",
    "Son yıla ait tüm arıza kayıtlarının özeti nedir?",
    "Yönetici özetini arıza kategorilerine göre çıkar.",
    "Servis taleplerinin yıllık büyüme oranı nedir?",
    "Makine güvenilirlik endeksi için arıza verileri nedir?",
    "Aylık arıza istatistiği yöneticiye nasıl sunulur?",
    "Ortalama arıza süresi ve sıklığı istatistiği nedir?",
    "Haftalık KPI özet tablosu arıza verileriyle hazırla.",
    "Kategori dağılım grafiği için veri çıkarır mısın?",
    "Toplam arıza kaydı sayısını yıl bazında göster.",
    "Servis kayıtlarından yönetici için sayısal döküm çıkar.",
    "Yönetim için servis istatistikleri özetini çıkar.",
    "Geçen yıl en sık görülen 10 arıza tipi nedir?",
    "Arıza kategorilerinin yüzde dağılımını tablo olarak ver.",
    "KPI dashboard'u için hangi servis istatistikleri gerekli?",
]
yonetim_qs = list(dict.fromkeys(yonetim_qs))[:63]

# ── Özel Projeler ─────────────────────────────────────────────────────────────
ozel_qs = [
    "Proje durumu nasıl takip edilir?",
    "Milestone teslimat takvimi nedir?",
    "Özel makine projesinin iş planı hazır mı?",
    "Proje raporunda hangi kilometre taşları var?",
    "Gantt çizelgesi nasıl güncelleniyor?",
    "WBS şeması hazır mı?",
    "Proje takvimi revize edildi mi?",
    "Bu projenin kritik yolu nedir?",
    "Milestone gecikmesi nasıl yönetilir?",
    "Proje skop değişikliği nasıl onaylanır?",
    "Özel CTL hattı projesinin durumu nedir?",
    "Proje risk planı hazırlandı mı?",
    "Tasarım onay aşamasına geçildi mi?",
    "Fabrikasyon takvimi nasıl görünüyor?",
    "Proje ekibine kim dahil?",
    "Müşteri onay süreçleri proje planında nerede?",
    "Proje bütçesi güncel mi?",
    "Kilometre taşı toplantısı ne zaman?",
    "Proje teslim tarihi ertelendi mi?",
    "Özel makine teknik şartnamesi onaylandı mı?",
    "Müşteri proje tasarım revizyonu talep etti, süreç ne?",
    "Proje ilerlemesi yüzde kaç?",
    "Tedarik süreci proje planında nerede?",
    "Montaj takvimi hazırlandı mı?",
    "Proje dosyası nerede tutuluyor?",
    "Özel proje için kaynak planlaması yapıldı mı?",
    "Proje mühendisi kim?",
    "Müşteri kabul testleri proje planında ne zaman?",
    "Proje deliverable listesi tamamlandı mı?",
    "Proje kickoff toplantısı ne zaman yapıldı?",
    "Proje zaman çizelgesi müşteriye sunuldu mu?",
    "WBS üzerinde yapılan son değişiklik neydi?",
    "Projenin hangi milestone'u gecikiyor?",
    "Proje kalite planı hazırlandı mı?",
    "Özel makine proje charter'ı nerede?",
    "Gantt şeması üzerinde kritik görevler neler?",
    "Proje kapanış prosedürü nedir?",
    "Müşteri proje değişiklik talepleri nasıl yönetiliyor?",
    "Alt yüklenici proje takviminde ne zaman devreye giriyor?",
    "Proje toplantısı tutanakları nerede?",
    "Proje milestone'larından hangisi tamamlandı?",
    "Proje gecikmesi için kök neden analizi yapıldı mı?",
    "Proje iletişim planı hazır mı?",
    "Özel proje için onaylanmış bütçe nedir?",
    "Proje paydaşları kimler?",
    "Teslim takvimi müşteri onayına sunuldu mu?",
    "Proje kapsamı değişti mi?",
    "Proje fizibilite raporu nerede?",
    "Özel makine proje ekibi kimlerden oluşuyor?",
    "Proje aktivite listesi güncel mi?",
    "Proje yönetim aracında (Jira/MS Project) takip nasıl yapılıyor?",
    "Projenin kritik milestone'larına ulaşıldı mı?",
    "Özel proje için test ve validasyon planı hazır mı?",
    "Proje zaman çizelgesi değişiklikleri nasıl onaylanıyor?",
    "Proje haftalık ilerleme raporunu kim hazırlıyor?",
    "Proje kapsamı dışına çıkıldı mı?",
    "Özel makine projesinin son tahmini tamamlanma tarihi nedir?",
    "Proje başarı kriterleri nelerdir?",
    "Kilometre taşı raporunu müşteriye kim sunuyor?",
    "Proje kapat kararı kim tarafından veriliyor?",
    "Özel proje için kaynak kısıtlamaları var mı?",
    "Proje önceki versiyonunun gantt şemasına nasıl erişilir?",
    "Proje tamamlanma sertifikası hazır mı?",
]
ozel_qs = list(dict.fromkeys(ozel_qs))[:63]

# ── Dijital İkiz ──────────────────────────────────────────────────────────────
dijital_qs = [
    "CTL-01 hattının anlık durumu nedir?",
    "CTL-02'nin OEE değeri kaç?",
    "ACK-01 makinesinde aktif alarm var mı?",
    "ACK-02 bobin sıcaklığı kaç derece?",
    "DST-01'in vakum basıncı nedir?",
    "PTR-01 hidrolik basınç değeri nedir?",
    "SBL-01 besleme doğruluğu ne durumda?",
    "LSD-01 loop yüksekliği normal mi?",
    "Hangi makineler şu an çalışıyor?",
    "Fabrikada kaç makine aktif?",
    "Canlı makine durumunu göster",
    "Dijital ikiz dashboardunu aç",
    "CTL-01 sıcaklığı kaç?",
    "CTL-02 alarm veriyor mu?",
    "ACK-01'in OEE'si kaç?",
    "ACK-02 şu an çalışıyor mu?",
    "DST-01 vakum hatası var mı?",
    "PTR-01 sıcaklığı kaç derece?",
    "SBL-01 üretim sayacı ne gösteriyor?",
    "LSD-01 aktif mi pasif mi?",
    "Tüm hatların OEE ortalaması nedir?",
    "En son alarm hangi makineden geldi?",
    "CTL-01 ve CTL-02 karşılaştır",
    "Kaç makine alarm durumunda?",
    "Vardiya üretim sayıları nedir?",
    "Hangi hat en yüksek OEE'ye sahip?",
    "CTL-01 hattında sorun var mı?",
    "ACK hatlarının durumu nasıl?",
    "PTR-01 basınç alarmı veriyor mu?",
    "DST-01 vakum durumu normal mi?",
    "SBL-01 son üretim değerleri?",
    "LSD-01 sensör okumaları nedir?",
    "Makine verileri canlı olarak göster",
    "CTL-02 sıcaklık sensörü ne okuyor?",
    "ACK-01 bobin ağırlığı kaç?",
    "PTR-01 hidrolik sıcaklığı normal mı?",
    "Fabrika genel durumu nasıl?",
    "Kaç hat aktif, kaç hat pasif?",
    "OEE değerleri hangi makinelerde düşük?",
    "CTL-01 üretim sayacı kaçta?",
    "Alarm listesini göster",
    "Aktif alarmlar neler?",
    "DST-01 şu anki durumu nedir?",
    "SBL-01 OEE kaç?",
    "LSD-01 loop yüksekliği kaç mm?",
    "Tüm makinelerin sıcaklık değerleri nedir?",
    "CTL-01 hızı kaç?",
    "ACK-02 bobin sayacı ne gösteriyor?",
    "Canlı hat izleme verilerini göster",
    "PTR-01 basınç değeri nedir?",
    "Hangi makine en çok alarm veriyor?",
    "CTL-02 hızı ve OEE'si nedir?",
    "DST-01 sensör değerleri neler?",
    "ACK-01 aktif mi?",
    "SBL-01 besleme hızı kaç?",
    "Fabrika OEE ortalaması kaç?",
    "LSD-01 canlı sensör verisi?",
    "CTL-01 son alarm neydi?",
    "ACK-02 sıcaklık alarmı var mı?",
    "PTR-01 hidrolik alarm durumu?",
    "Makine durumu raporu göster",
    "Canlı dashboard'a bağlan",
    "CTL-01 normal mi çalışıyor?",
    "Tüm makine sensör okumaları nedir?",
]
dijital_qs = list(dict.fromkeys(dijital_qs))[:63]

# ── Assemble ──────────────────────────────────────────────────────────────────
AGENT_LISTS = {
    "Servis Kayıtları":       servis_qs,
    "Ürün Kataloğu":          urun_qs,
    "Yedek Parça":            yp_qs,
    "Kurulum ve Devreye Alma": kur_qs,
    "Kalite ve Uyumluluk":    kalite_qs,
    "Satış ve Teklifler":     satis_qs,
    "Yönetim Raporu":         yonetim_qs,
    "Özel Projeler":          ozel_qs,
    "Dijital İkiz":           dijital_qs,
}

questions = []
for agent, qs in AGENT_LISTS.items():
    for q in qs:
        questions.append({"question": q, "expected_agent": agent})

random.shuffle(questions)
OUT.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Generated {len(questions)} questions → {OUT}")
for agent, qs in AGENT_LISTS.items():
    print(f"  {agent}: {len(qs)}")

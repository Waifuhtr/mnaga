# Manga Çeviri — manga-image-translator + Hy-MT2-7B

Japonca / İngilizce manga sayfalarını **Türkçeye** çeviren, Hugging Face Spaces
üzerinde çalışan bir Docker uygulaması.

* **Pipeline:** [`zyddnys/manga-image-translator`](https://github.com/zyddnys/manga-image-translator) — algılama, OCR, maskeleme, metin silme (inpainting) ve yeniden yerleştirme upstream'den **değiştirilmeden** kullanılır.
* **Çeviri modeli:** [`tencent/Hy-MT2-7B-GGUF`](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) → `Hy-MT2-7B-Q4_K_M.gguf` (4.62 GB), llama.cpp `llama-server` ile OpenAI-uyumlu API olarak yayınlanır.
* **Bağlantı:** upstream'in kendi `custom_openai` translator'ı. Fork yok, patch yok.

---

## CPU build → T4 runtime

| Aşama | Donanım | Ne olur |
|---|---|---|
| **Build** | 8 vCPU / 32 GB (GPU yok) | Bütün pahalı işler burada biter: bağımlılıklar, llama.cpp ikilileri, M.I.T. modelleri ve 4.62 GB'lık Hy-MT2 image katmanlarına yazılır. |
| **Runtime** | T4 Small (4 vCPU / 15 GB RAM / 16 GB VRAM) | GPU görülürse Hy-MT2 tamamen GPU'ya offload edilir ve **runtime'da hiçbir indirme yapılmaz.** |

> **Donanım değiştirmek yeniden build tetikler.** Ölçüldü: CPU'dan T4'e geçişte
> HF Spaces image'ı sıfırdan yeniden kurdu, build günlüğünde tek bir `CACHED`
> katman yoktu ve 4.62 GB'lık model tekrar indi. Yani "donanım değişince model
> yeniden inmesin" hedefi HF Spaces'ta katman cache'i ile sağlanamıyor.
> Elde kalan ve asıl önemli olan kazanç şu: build **otomatik ve ~7 dakika**
> sürüyor (modelin payı ~72 sn), deterministik (sabit revision + SHA-256) ve
> **çalışma anında tek bir indirme bile yapılmıyor** — konteyner internet
> kapalıyken de açılır.

Build sırasında CUDA inference **başlatılmaz**; GPU yalnızca runtime'da,
`scripts/entrypoint.sh` içindeki `nvidia-smi` kontrolüyle devreye girer. GPU
yoksa uygulama CPU'ya düşer ve çalışmaya devam eder.

### Kurulum

1. Bu depoyu (`Waifuhtr/mnaga`) GitHub'da tutun — asıl kaynak burasıdır.
2. HF Space'e yalnızca iki dosya gider: `space/Dockerfile` → `Dockerfile`, `space/README.md` → `README.md`.
3. Space'i **CPU** donanımıyla build edin.
4. Build bittikten sonra donanımı **T4 Small**'a çevirin.
5. `/health` adresinden durumu doğrulayın.

### Kod güncelleme

Uygulama kaynağı image'ın **son katmanına** gömülür. Bu yüzden:

1. GitHub'a push edin.
2. Space'i **yeniden build edin** — *Settings → Factory rebuild*, ya da Space
   deposuna herhangi bir commit atın.

> **Önemli:** Space'i yalnızca **restart** etmek yetmez. Restart, mevcut image'ı
> yeniden başlatır ve içindeki eski kodu çalıştırır; yeni GitHub kodunu almaz.
> Bu doğrulandı: restart sonrası build günlüğünde yeni bir build görünmüyor.

Dockerfile'daki `ADD https://api.github.com/.../commits/<branch>` satırı, build
sırasında yalnızca **son katmanın** cache'ini kırmak içindir.

---

## BUILD TIME'da image'a gömülen büyük dosyalar

| Dosya / grup | Boyut | Nereden | Image içindeki yol |
|---|---|---|---|
| `Hy-MT2-7B-Q4_K_M.gguf` | **4.62 GB** | `tencent/Hy-MT2-7B-GGUF` (revision `ab847266`, SHA-256 doğrulanır) | `/opt/models/hy-mt2/Hy-MT2-7B-Q4_K_M.gguf` |
| llama.cpp CUDA ikilileri | ~170 MB | GitHub release `b11022` | `/opt/llama/` |
| PyTorch + torchvision (cu128) | ~2.5 GB | `download.pytorch.org` | site-packages |
| M.I.T. algılama / OCR / inpainting ağırlıkları | ~1.5 GB | `docker_prepare.py` | `/opt/manga-image-translator/models/` |
| manga-ocr ağırlıkları | ~450 MB | `kha-white/manga-ocr-base` | `/opt/hf-cache/` |
| CUDA 12.8 runtime + sistem paketleri | ~2.2 GB | temel image | — |

## RUNTIME'da ne nereden yüklenir

| Bileşen | Kaynak | Ağ gerekir mi |
|---|---|---|
| Hy-MT2 ağırlıkları | `/opt/models/hy-mt2/…gguf` (image) | hayır |
| llama-server | `/opt/llama/llama-server` (image) | hayır |
| Algılama / OCR / inpainting | `/opt/manga-image-translator/models/` (image) | hayır |
| manga-ocr | `/opt/hf-cache/` (image) | hayır |
| Yazı tipi | `/opt/manga-image-translator/fonts/comic shanns 2.ttf` (image) | hayır |

Ölçülen hız (T4 Small, 2 metin bloklu sayfa): ilk sayfa ~3.7 sn (modeller
VRAM'e yükleniyor), sonrası **~2.0 sn/sayfa**. Aynı sayfa CPU'da 32.1 sn
sürüyordu — yaklaşık **16× fark**.

`HF_HUB_OFFLINE=1` ve `TRANSFORMERS_OFFLINE=1` ayarlıdır: bir kütüphane runtime'da
gizlice indirme yapmaya kalkarsa sessizce beklemek yerine hata verir. **Uygulama
internet kapalıyken de açılır.**

## Cache / katman stratejisi

Dockerfile katmanları *kararlıdan değişkene* doğru sıralanmıştır:

```
1 sistem paketleri + Python 3.11 ── neredeyse hiç değişmez
2 CUDA kütüphaneleri (pip)       ── sürüm sabitlenmiş
3 PyTorch                        ── sürüm sabitlenmiş (2.9.1 / 0.24.1)
4 llama.cpp CUDA ikilileri       ── sürüm sabitlenmiş (b11022), burada test edilir
5 M.I.T. + pip bağımlılıkları    ── commit sabitlenmiş (95227a2b)
6 M.I.T. modelleri               ── build sırasında indirilir
7 manga-ocr ağırlıkları          ── build sırasında indirilir
8 Hy-MT2 GGUF  4.62 GB           ── EN PAHALI KATMAN
9 uygulama kaynağı               ── ⟵ normalde yalnızca burası yeniden kurulur
```

Uygulama kodu **en son** katmandadır; bu yüzden küçük bir kod değişikliği 4.62 GB'ı
yeniden indirmez. Model ayrıca `curl` ile doğrudan son yoluna indirilir:
`hf_hub_download` kullanılsaydı bir kopya da HF cache'inde kalır ve image
gereksiz yere iki katına çıkardı.

---

## Dosyalar

| Dosya | İşlevi |
|---|---|
| `space/Dockerfile` | **Asıl build dosyası.** Space'e aynen kopyalanır. Kök dizindeki `Dockerfile` buna bir symlink'tir (iki kopya arasında fark oluşmasın diye). |
| `space/README.md` | Space için gerekli YAML metadata (`sdk: docker`, `app_port: 7860`). |
| `scripts/entrypoint.sh` | llama-server ile web uygulamasını **paralel** başlatır, GPU'yu algılar, her satırı zaman damgalar, biri ölürse konteyneri indirir. |
| `app/server.py` | FastAPI uygulaması: yükleme, iş kuyruğu, ilerleme, sonuç ve ZIP indirme. M.I.T.'yi doğrudan kütüphane olarak çağırır. |
| `app/static/` | Arayüz (HTML + CSS + JS). Framework yok. |
| `assets/fonts/` | **Font bırakma klasörü.** Buraya konan her font arayüzdeki listede otomatik çıkar. Bkz. `assets/fonts/README.md`. |
| `config/gpt_config.yaml` | Hy-MT2 için prompt ve örnekleme ayarları. M.I.T. bunu `gpt_config` olarak yükler. |
| `.env.example` | Bütün ortam değişkenleri ve varsayılanları. |

Geçici dosyalar `/tmp/mnaga-work` altında tutulur: HF Spaces'ta `/data` yalnızca
çalışma anında (storage bucket bağlıysa) vardır ve konteyner UID 1000 ile çalışır.

---

## Yazı tipleri

Arayüzdeki "Yazı tipi" listesi **sabit değildir**: sunucu her açılışta
`assets/fonts/` klasörünü tarar ve orada bulduğu her `.ttf` / `.otf` / `.ttc`
dosyasını listeye koyar. Yeni bir font eklemek için klasöre dosyayı koyup push
etmek yeterli — kodda, Dockerfile'da veya arayüzde değişiklik gerekmez. Klasör
image'a zaten var olan `git clone` adımıyla (Dockerfile 9. katman) girer.

Listedeki ad dosya adıdır; seçilen font sayfanın tamamında kullanılır.
Ayrıntılar ve Türkçe harf uyarısı: **`assets/fonts/README.md`**.

Sunucu her fontu açıp `ç Ç ğ Ğ ı İ ö Ö ş Ş ü Ü` harflerini kontrol eder
(`freetype`, zaten render için kullanılan kütüphane) ve eksik harf varsa bunu
arayüzde fontun altında yazar. Eksik harf sayfayı bozmaz — `text_render` o
harfleri yedek fontla (Arial Unicode) çizer — ama kelime ortasında stil kayması
görünür. Şu an klasörde olanlar:

| Font | Türkçe harfler |
|---|---|
| `CCWildWords.ttf` | tamamı var |
| `anime_ace_3.ttf` | `ğ Ğ ı İ ş Ş` yok, yedekten gelir |

Hangi fontun önceden seçili geleceği `RENDER_FONT_KEY` ile ayarlanır; tanınmayan
bir değer verilirse uygulama var olan bir fonta düşer, fontsuz kalmaz.

---

## Metin algılama ayarları

Arayüzde iki kontrol var, ikisi de her iş için ayrı ayrı gönderilir — yani
karşılaştırma yapmak için yeniden derleme gerekmez:

* **Metin algılayıcı** — `DBNet` (varsayılan) veya `Paddle`. Paddle, upstream'in
  Rust PP-OCR ailesi algılayıcısıdır; `rusty-manga-image-translator` zaten bir
  gereklilik olduğu için image'a **ek yük getirmez** (ayrı model indirmesi yok).

  **Artık gerçek sayfa üzerinde ölçüldü.** Aynı 1280×1816 sayfada, aynı build'de:

  | Algılayıcı | Bölge | Küçük "Her ass" baloncuğu | Yerleşim |
  |---|---|---|---|
  | DBNet | 12-13 | **kaçırıyor** | düzgün |
  | Paddle | **16** | **yakalıyor** | bölgeler dar, yazı küçülür |

  Paddle daha çok metin buluyor ama bölgeleri daha dar çıkarıyor, bu yüzden
  yazı boyutu tabana dayanıp okunaksızlaşabiliyor. Çözümü aşağıdaki **en küçük
  yazı boyutu** ayarı. Varsayılan hâlâ DBNet, çünkü Paddle'ın yerleşim bedeli
  sayfaya göre değişiyor.
* **Algılama hassasiyeti** — DBNet'in `text_threshold` / `box_threshold`
  değerleri. Varsayılanımız `0.4 / 0.6`, upstream'in kendi varsayılanı
  `0.5 / 0.7`.

Varsayılanın neden değiştirildiği (ölçüm, gerçek 1280×1816 sayfa üzerinde):

| Eşikler | Bulunan bölge | Küçük baloncuğun kutusu | Kapsama |
|---|---|---|---|
| 0.5 / 0.7 (upstream) | 57 | ilk kelimenin yarısı | %10 |
| 0.4 / 0.6 (bizim) | 58 | ifadenin tamamı | %39 |

0.5/0.7'de kutu ifadenin sadece ilk kelimesini kapsıyor, bölge sonrasında
boru hattından düşüyor ve İngilizce metin sayfada **silinmemiş** kalıyordu —
kullanıcı testinde görülen kaçak buydu. 0.4/0.6'da kutu ifadenin tamamını
kapsıyor ve sayfadaki toplam bölge sayısı 57 → 58 kadar oynuyor, yani daha
gevşek eşik sayfayı sahte kutularla doldurmuyor.

---

## Hangi sürüm çalışıyor?

Arayüzün üst kısmında, durum göstergesinin hemen altında `sürüm 9443a29 · 24.09.2026`
gibi bir satır var. Bu, **image'ın hangi commit'ten derlendiğini** söyler.

Nereden geliyor: Dockerfile katman cache'ini kırmak için zaten
`ADD https://api.github.com/repos/.../commits/<branch> /tmp/app-commit.json`
yapıyor. Klon sonrası `.git` siliniyor (image küçük kalsın diye), dolayısıyla
konteyner içinde hangi commit'in çekildiğini söyleyen **tek kayıt** bu dosya.
Uygulama onu geri okuyup `/health` içinde ve arayüzde gösteriyor.

Bu, "Space gerçekten son push'umu mu çalıştırıyor?" sorusunu tek bakışta
cevaplar. Bir kez, bir haftalık eski build'in üzerinde hata aranarak
öğrenilmişti; bir daha gerekmesin diye kalıcı hale getirildi.

`/health` çıktısında da aynısı var:

```json
"build": { "commit": "9443a29", "committed_at": "2026-09-24T07:09:11Z", "subject": "..." }
```

---

## Başlangıç süresi

Konteyner açılışında iki ağır iş var ve artık **paralel** çalışıyorlar:

| | Ne yapar | Kim bekler |
|---|---|---|
| `llama-server` | 4.4 GB GGUF'u diskten okur, T4'e offload eder | çeviri isteği |
| `uvicorn` + app | torch ve manga-image-translator'ı import eder | — |

Eskiden `entrypoint.sh` llama-server `/health` yeşile dönene kadar bekleyip
**ondan sonra** uvicorn'u başlatıyordu, yani süreler toplanıyordu. Artık ikisi
birlikte başlıyor; toplam süre kabaca ikisinin **büyüğü** kadar.

Bunun bedeli: arayüz, çeviri motoru hazır olmadan erişilebilir oluyor. O yüzden
`/api/jobs` her işte llama-server'ı kontrol edip hazır değilse işi **başlatmadan**
reddediyor (503), arayüzdeki buton da o sırada "Çeviri modeli yükleniyor…"
yazıp pasif kalıyor. Yani yarım işlenmiş sayfa üretmesi mümkün değil.

Ölçüm için `entrypoint.sh`'ın her satırı artık zaman damgalı:

```
[entrypoint] 08:01:02 (+0s) starting web app on 0.0.0.0:7860 (in parallel with model load)
[entrypoint] 08:01:02 (+0s) waiting for llama-server on 127.0.0.1:8081 ...
[entrypoint] 08:04:11 (+189s) llama-server is healthy (model load took 189s)
[entrypoint] 08:04:11 (+189s) READY - both processes up
```

`(+Ns)` script başından beri geçen saniye. Böylece bir sonraki açılışta hangi
adımın ne kadar sürdüğü tahmin değil, ölçüm olur.

---

## Metin silme gücü (leke sorunu)

Silinen baloncukların kenarında iz kalıyorsa sebep büyük ihtimalle silme
maskesinin harflerin yeterince ötesine genişlememesi. İki parametre kontrol
ediyor ve upstream bunları **iki ayrı yerden** okuyor:

| Parametre | Upstream nereden okur | Bizde nereden gider |
|---|---|---|
| `mask_dilation_offset` | `config.mask_dilation_offset` | `build_config()` |
| `kernel_size` | `self.kernel_size` (constructor'da set edilir) | `run_job()`'ta örneğe atanır |

İkincisi upstream'de bilinen bir tuhaflık — kodda kendi `#todo: fix why is
kernel size loaded in the constructor` notu duruyor. Bu yüzden font gibi, her
iş için translator örneğine yazılıyor.

**Varsayılanları değiştirmedim** (20 / 3, upstream'in kendi değerleri): elimde
başka bir değerin daha iyi olduğunu söyleyen bir ölçüm yok, tahminle değiştirmek
sorunu sadece yer değiştirir. Onun yerine arayüze seçenek olarak kondu, böylece
üç ayarı **tek build üzerinde** karşılaştırabilirsin:

| Seçenek | dilation / kernel |
|---|---|
| Normal (varsayılan) | 20 / 3 |
| Güçlü (leke kalıyorsa) | 28 / 5 |
| Çok güçlü (deneysel) | 36 / 7 |

Hangisinin kullanıldığı iş durumunda `erase: "28/5"` olarak geri raporlanıyor.

---

## Yazı çok küçük çıkıyorsa

Upstream her bölge için yazı boyutunu o bölgenin geometrisinden hesaplıyor,
sonra `font_size_minimum` ile tabanlıyor. Taban `-1` iken otomatik:
`(en + boy) / 200` — 1280×1816 bir sayfada **~15px**. Dar bir bölge bu tabana
dayandığında metin okunaksız hâle geliyor, kelimeler ortadan bölünüyor.

Bölgenin dar olup olmaması **algılayıcıya** bağlı: Paddle aynı sayfada 16,
DBNet 13 bölge çıkarıyor; Paddle'ınkiler daha dar olduğu için tabana çok daha
sık dayanıyorlar. Paddle kullanıyorsan "En küçük yazı boyutu" ayarını
**26px**'e almak bunu düzeltir.

| Ayar | Ne yapar |
|---|---|
| Otomatik (varsayılan) | upstream'in `(en+boy)/200` değeri, ~15px |
| En az 20 / 26 / 32 px | sabit taban; 26px Paddle için başlangıç noktası |

---

## Tarayıcı cache'i (önemli)

`/static/*` dosyalarını `StaticFiles` ETag/Last-Modified ile sunuyor, yani
tarayıcı `app.js`'i rebuild'ler arasında **saklıyor**. `index.html` ise kendi
route'umuzdan, validator'sız geldiği için hep taze. Bu ikisi bir araya gelince
**yeni sayfa + eski script** çalışıyordu:

* font seçici "Yükleniyor…" yer tutucusunda takılı kalıyordu — cache'teki eski
  script onu dolduran kodu içermiyordu;
* seçici ölü olduğu için algılayıcı da seçilemiyordu, dolayısıyla o sekmede
  Paddle denenemiyordu.

Sonuç: aynı Space'in iki sekmesi bir test turu boyunca farklı davrandı.

Artık `index.html` sunulurken asset URL'lerine build commit'i ekleniyor
(`/static/app.js?v=135a1d4`) ve sayfanın kendisi `Cache-Control: no-store` ile
gidiyor. Rebuild olduğunda URL değişiyor, tarayıcı zorunlu olarak yeniden
indiriyor. Elle `Ctrl+F5` gerekmiyor.

---

## Font doğrulama (önemli)

Bir font, **her Türkçe harf için glif taşıyabilir**, o gliflerin **hepsi
birbirinden farklı olabilir**, ve yine de **yanlış şekilleri çizebilir**.
`assets/fonts/` içindeki yamalı bir yüz tam olarak bunu yaptı:

| Font | Örnek çıktı |
|---|---|
| `CCWildWords.ttf` | `ÖZÜR DİLERİM · KAÇ SAÇ ĞÜŞIÖÇ` ✅ |
| `anime_ace.ttf` | `UZbR DİLERİM · KA3 SA3 ğbŞIU3` ❌ |
| `anime_ace_3.ttf` | aynı bozukluk ❌ |

Bozuk olanlarda `Ğ Ş İ` doğru, ama `Ö→U`, `Ü→b`, `Ç→3`. Glifler mevcut ve
benzersiz olduğu için **hiçbir otomatik kontrol bunu yakalayamıyor** — varlık
kontrolü de, glif-indeksi karşılaştırması da, diakritik yüksekliği sezgiseli de
bu fontu "temiz" ilan etti. Yakalayan tek şey render edip bakmak.

Bu yüzden arayüzde font seçicinin altında **o fontla çizilmiş Türkçe örnek**
gösteriliyor (`/api/fonts/preview/<key>`). GPU harcamaz, çeviri çalıştırmaz;
bozuk bir font bir sayfa bile işlenmeden görünür.

---

## Font değiştirince eski fontla çizilmesi

Upstream glifleri şöyle önbelleğe alıyor:

```python
@functools.lru_cache(maxsize=1024, typed=True)
def get_char_glyph(cdpt: str, font_size: int, direction: int) -> Glyph:
    global FONT_SELECTION
```

Önbellek anahtarında **font yok**; yüz, `set_font()`'un değiştirdiği global
`FONT_SELECTION`'dan geliyor. Yani bir karakter bir kez çizildikten sonra,
sonradan hangi font seçilirse seçilsin aynı bitmap dönüyor — font değişimini
izleyen sayfa **bir önceki fontla** render ediliyor.

Gerçek yüzlerle doğrulandı: önbellek sıcakken `'ü'` 30px'te anime_ace ve
CCWildWords için **byte-byte aynı** bitmap'i döndürdü; önbellek temizlenince
iki farklı bitmap geldi.

İş başına font seçimi bizim eklediğimiz bir özellik olduğu için temizliği de
bize ait: `run_job()` her işte `text_render.get_char_glyph.cache_clear()`
çağırıyor.

---

## Üst üste binen baloncuklar

`render()` metin tuvalini `dst_points`'e homografi ile **warp ediyor**:

```python
M, _ = cv2.findHomography(src_points, dst_points, ...)
rgba_region = cv2.warpPerspective(box, M, ...)
```

Yani metin kutusunu asla taşmıyor, tam dolduruyor. Buradan çıkan sonuç net:
**iki metin çakışıyorsa kutuları çakışıyordur.** Sorun tamamen geometri.

Çeviri uzun geldiğinde upstream kutuyu büyütüyor ve sınır kırpmasını bilerek
kapatmış (`# 移除边界限制…`), komşu kontrolü ise hiç yok — kendi `dispatch()`
fonksiyonunda `# TODO: Maybe remove intersections` notu duruyor. Türkçe
İngilizce'den uzun olduğu için bu sürekli tetikleniyor.

Çözüm iki aşamalı, **sırası bilinçli**:

1. **İt.** Çakışan kutular birbirinden itiliyor. Boyut değişmiyor, yani satır
   sarması upstream'in seçtiği gibi kalıyor.
2. **Küçült.** Sadece itmenin ayıramadığı kadarı için kutu, dedektörün bulduğu
   kutuya doğru geri yürütülüyor.

İtme önce geliyor çünkü küçültmek okunabilirliğe mal oluyor: dar kutu = aynı
metnin daha dar bir sütuna sarılması, `homurdanıp` kelimesinin
`HOMU / RDANIP` diye bölünmesinin sebebi buydu.

**Eksen seçimi kritik.** "Daha küçük örtüşme" sezgiseli yanlış: yan yana duran
iki kutu tüm yüksekliklerini paylaşır, yani dikey örtüşme daha küçük sayıdır
ama tam da ayrılamayacakları yöndür — onu kapatmak satırı bir kutu boyu
baloncuğundan uzaklaştırmak demek. Onun yerine her eksenin gereksinimi kayma
sınırıyla karşılaştırılıp **gerçekten ayırabilen** eksen seçiliyor.

Kayma, kutunun kendi boyunun `MAX_BLOCK_SHIFT` katıyla sınırlı (varsayılan
0.45) — komşu baloncukları ayırmaya yeter, satırın baloncuğundan kopmasına
yetmez.

Yerel testler (hepsi çakışmayı 0'a indiriyor):

| senaryo | önce | sonra | boyutlar |
|---|---|---|---|
| yan yana, upstream sağa büyütmüş | 8000 px² | 0 | korundu |
| dedektör kutuları çakışıyor | 9800 px² | 0 | korundu |
| üst üste dizili | 2000 px² | 0 | korundu |
| neredeyse tam üst üste | 26450 px² | 0 | küçüldü (itme yetmiyor) |
| çakışma yok | 0 | 0 | dokunulmadı |

---

## Siyah baloncuklarda sönük metin

`fg_bg_compare()` beyaz kenarlığı yalnızca metinle arka plan CIE76'da 30'dan
az fark ediyorsa zorluyor. Gerçek OCR çıktısıyla ölçüldü:

| metin | fg | bg | fark | upstream kenarlık |
|---|---|---|---|---|
| `BEARD?!` | (11,14,17) | (34,37,34) | 11.2 | beyaz ✅ |
| `SHE WANTS` | (7,12,3) | (83,87,75) | **33.5** | koyu gri ❌ |

30–45 bandındakiler eşikten kaçıyor: neredeyse siyah metin, koyu gri kenarlık,
koyu baloncuk. Teoride okunur, pratikte sönük.

Artık **metin ve arka planın ikisi de koyuysa** kenarlık beyaza alınıyor.
Metnin kendisi siyah kalıyor — manga'da alışıldık hâli bu. Eşik
`DARK_BUBBLE_MAX` (varsayılan 100), Spaces'ta Settings → Variables'tan
rebuild'siz değiştirilebilir.

Sadece bozuk vaka değişiyor: beyaz baloncuklar, açık zeminler ve zaten beyaz
kenarlık alanlar aynen kalıyor.

---

## Bu düzeltmeler neden yama olarak duruyor

Dockerfile, manga-image-translator'ı her build'de sabit bir commit'ten
**yeniden klonluyor**. O ağaçta yapılan bir düzenleme build'i geçmez. Bu
yüzden ikisi de `app/server.py` içinden modül niteliği değiştirerek
kuruluyor (`_install_render_patches`), upstream dosyalarına dokunulmuyor.

---

## Toplu yükleme ve sayfa sırası

* **Tek görsel**, **çoklu görsel** ve **ZIP** yüklenebilir; hepsi aynı anda olabilir.
* Sayfalar **doğal sıraya** göre işlenir: `1, 2, 3, …, 10, 11` — sözlük sırasındaki `1, 10, 11, 2` değil.
* **Dosya adları değiştirilmez.** `3.jpg` çıktıda `3.png` olur; galeriden yüklenen adlar korunur.
* Sonuç ZIP'i aynı adları ve aynı sırayı taşır.

Aynı sıralama kuralı hem sunucuda (`app/server.py: natural_key`) hem arayüzde
(`app/static/app.js: naturalKey`) uygulanır; böylece başlatmadan önce gördüğünüz
liste, gerçek işlenme sırasıyla birebir aynıdır.

---

## Doğrulanmış teknik kararlar

Aşağıdakiler tahmin değil, gerçek model ve gerçek kaynak kod üzerinde test edildi:

* **STQ kernel'i gerekmiyor.** Model kartı, GGUF'ların llama.cpp PR #22836'daki
  STQ kernel'ine ihtiyaç duyduğunu söylüyor. O PR master'a **girmemiş** durumda.
  Ancak `Hy-MT2-7B-Q4_K_M.gguf` dosyasının tensör tipleri okundu: yalnızca
  `F32`, `Q4_K` ve `Q6_K` var — `STQ1_0` **yok**. Uyarı 1.25/2-bit sürümler için
  geçerli. Model, değiştirilmemiş llama.cpp `b11022` ile yüklenip çalıştırıldı.
* **Upstream'in `<|1|>` toplu çeviri protokolü Hy-MT2 ile çalışıyor.** Gerçek
  modele gerçek manga metni gönderildi; numaralandırma korundu, ses efektleri
  (`ドーン！！` → `GÜM!!!`), karakter adları (`タカシ先輩` → `Takashi senpai`) ve
  sayılar (`HP：120/500`) bozulmadı. Bu yüzden özel bir translator sınıfı
  yazılmadı — upstream `custom_openai` olduğu gibi kullanılıyor.
* **Dil kodu.** `TRK`, upstream tarafından prompt'a girmeden önce `Turkish`
  tam adına çevriliyor (`translators/common.py`), ki Hy-MT2 model kartı da tam
  dil adı istiyor.
* **Yazı tipi.** Upstream'in varsayılan manga fontları `anime_ace.ttf` ve
  `anime_ace_3.ttf`, Türkçe için gereken `ğ Ğ ı İ ş Ş` glifilerini
  **içermiyor** (fontTools ile doğrulandı) — bu fontlarla Türkçe çıktı bozuk
  görünürdü. Varsayılan bu yüzden `comic shanns 2.ttf`: hem çizgi roman görünümü
  var hem de Türkçe karakterlerin tamamını içeriyor.
* **Metin silme (inpainting) T4'te VRAM'i patlatabiliyor — asıl tuzak buydu.**
  `lama_large`, gerçek bir manga sayfasında (1280x1816) tek seferde **13.58 GiB**
  ayırmaya çalışıyor; T4'ün 14.74 GiB'ının ~5.5 GiB'ı zaten llama.cpp'de olduğu
  için CUDA OOM veriyor. Upstream bu hatayı yakalayınca
  `ctx.img_inpainted = ctx.img_rgb` yapıp **orijinal görsele geri düşüyor** ve
  çeviri, silinmemiş metnin üstüne basılıyor — iş "başarılı" görünüyor ama sayfa
  okunmaz oluyor. Üç şey değişti: varsayılan inpainting çözünürlüğü 2048'den
  **1024**'e indi (bellek çözünürlüğün karesiyle büyüyor), OOM'da çözünürlüğü
  kademeli düşüren bir **yeniden deneme** eklendi ve `ignore_errors` **False**
  yapıldı ki hata bir daha sessizce bozuk çıktıya dönüşmesin.
  Not: bu hata benim ilk testlerimde çıkmadı çünkü 700x500'lük sentetik sayfalarla
  test etmiştim; gerçek sayfa boyutuyla test etmek şarttı.
* **GPU algılaması yalnızca `nvidia-smi`'ye güvenmiyor.** Temel image artık
  `nvidia/cuda` değil düz `ubuntu` olduğu için `nvidia-smi` yalnızca konteyner
  çalışma zamanı enjekte ederse bulunur. Tek başına ona güvenmek, T4'te GPU
  varken sessizce CPU'ya düşmek demekti — kullanıcı T4 parası ödeyip CPU hızı
  alırdı. Bu yüzden sıralı kontrol var: `nvidia-smi` → `torch.cuda.is_available()`
  (libcuda ile doğrudan konuşur) → `/dev/nvidiactl`. Hangi yolla bulunduğu
  loglara yazılır.
* **Upstream'in `docker_prepare.py --models` filtresi Python 3.11'de sessizce
  hiçbir şey indirmiyor.** Filtre `f"detector.{k}"` yazıyor; `k` bir
  `(str, Enum)` üyesi ve Python 3.11, mixin enum'larda `__format__`
  davranışını değiştirip sınıf adını da ekliyor. Sonuç
  `"detector.Detector.default"` oluyor, hiçbir anahtar eşleşmiyor ve adım
  **başarılı görünerek** boş geçiyor. (Yalnızca `Translator` sınıfı `__str__`
  tanımlıyor; bu yüzden çalışma anındaki translator yolu etkilenmiyor.)
  Bu yüzden modeller enum üyesiyle doğrudan seçiliyor ve indirme sonrası
  ağırlıkların gerçekten diske indiği **kontrol ediliyor** — aksi halde image,
  ilk kullanımda model indirmeye çalışırdı.
* **Image boyutu, build'i düşürebiliyor.** İlk sürüm bütün adımları başarıyla
  tamamladıktan sonra `Pushing image` aşamasında `exit code 137 / OOMKilled`
  ile düştü: image çok büyüktü. Asıl sebep CUDA çalışma zamanının **iki kez**
  bulunmasıydı — bir kez `nvidia/cuda` temel image'ından, bir kez de PyTorch'un
  kendi `nvidia-*` paketlerinden (~5 GB fazlalık). Artık temel image düz
  `ubuntu:24.04` ve llama.cpp, `LD_LIBRARY_PATH` ile PyTorch'un CUDA
  kütüphanelerine yönlendiriliyor; tek kopya kalıyor. Ayrıca CUDA
  kütüphaneleri PyTorch'tan ayrı bir katmana alındı, böylece en büyük tek
  katman yarıya indi.
* **Temel image ve Python sürümü.** llama.cpp'nin resmi CUDA release ikilileri
  **GLIBC 2.38**'e bağlı; Ubuntu 22.04 yalnızca 2.35 veriyor ve `llama-server`
  açılmıyordu (yalnızca CPU tarball'ı 2.29 ile yetiniyor, bu yüzden sorun
  sadece CUDA yapısında çıkıyor). Bu yüzden temel image **Ubuntu 24.04**.
  24.04 Python 3.12 ile geliyor ama M.I.T. `<3.12` istiyor; Python 3.11
  deadsnakes'ten kuruluyor ve her şey bir venv içinde çalışıyor. Ayrıca
  24.04'te `libglib2.0-0` paketi `libglib2.0-0t64` olarak yeniden
  adlandırılmış durumda.
* **`libgomp1` gerekiyor.** CUDA runtime image'ı OpenMP çalışma zamanını
  içermiyor; ggml'in CPU backend'leri (`libggml-cpu-*.so`) buna bağlı.
  Eksikken `llama-server` çalışmıyordu. Dockerfile'daki
  `llama-server --version` adımı bunu **build sırasında** yakalayan sert bir
  kontroldür; runtime'a kadar beklemez.
* **Prompt.** Upstream'in ~1600 karakterlik üç adımlı prompt'u da denendi ve
  çalışıyor; ancak `config/gpt_config.yaml` içindeki 642 karakterlik sürüm aynı
  kaliteyi veriyor ve her istekten ~400 token siliyor. Bir bölüm boyunca bu
  ciddi bir hız farkı demek.

## Sağlık kontrolü

```bash
curl http://localhost:7860/health
```

Modelin diskte olup olmadığını, llama-server'ın ayakta olup olmadığını ve GPU'nun
görülüp görülmediğini döner. Arayüzdeki rozet de bunu gösterir. Hata ayıklama
modu açıkken `/api/debug/llama` llama-server günlüğünün son satırlarını verir —
T4'te GPU offload'un gerçekten olduğunu buradan doğrulayabilirsiniz.

## Lisans

Upstream `manga-image-translator` GPL-3.0 olduğu için bu depo da GPL-3.0'dır.
Hy-MT2 modeli Apache-2.0 ile dağıtılmaktadır.

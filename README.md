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
| **Runtime** | T4 Small (4 vCPU / 15 GB RAM / 16 GB VRAM) | Konteyner aynı katmanlardan açılır. **Hiçbir şey yeniden indirilmez.** GPU görülürse Hy-MT2 tamamen GPU'ya offload edilir. |

Build sırasında CUDA inference **başlatılmaz**; GPU yalnızca runtime'da,
`scripts/entrypoint.sh` içindeki `nvidia-smi` kontrolüyle devreye girer. GPU
yoksa uygulama CPU'ya düşer ve çalışmaya devam eder.

### Kurulum

1. Bu depoyu (`Waifuhtr/mnaga`) GitHub'da tutun — asıl kaynak burasıdır.
2. HF Space'e yalnızca iki dosya gider: `space/Dockerfile` → `Dockerfile`, `space/README.md` → `README.md`.
3. Space'i **CPU** donanımıyla build edin.
4. Build bittikten sonra donanımı **T4 Small**'a çevirin.
5. `/health` adresinden durumu doğrulayın.

Kod değiştirdiğinizde: GitHub'a push edin, sonra Space'i yeniden build edin.
Dockerfile'daki `ADD https://api.github.com/.../commits/<branch>` satırı yalnızca
**son katmanın** cache'ini kırar; 4.62 GB'lık model katmanı yerinde kalır.

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
| `scripts/entrypoint.sh` | llama-server'ı başlatır, GPU'yu algılar, sağlıklı olmasını bekler, sonra web uygulamasını açar. |
| `app/server.py` | FastAPI uygulaması: yükleme, iş kuyruğu, ilerleme, sonuç ve ZIP indirme. M.I.T.'yi doğrudan kütüphane olarak çağırır. |
| `app/static/` | Arayüz (HTML + CSS + JS). Framework yok. |
| `config/gpt_config.yaml` | Hy-MT2 için prompt ve örnekleme ayarları. M.I.T. bunu `gpt_config` olarak yükler. |
| `.env.example` | Bütün ortam değişkenleri ve varsayılanları. |

Geçici dosyalar `/tmp/mnaga-work` altında tutulur: HF Spaces'ta `/data` yalnızca
çalışma anında (storage bucket bağlıysa) vardır ve konteyner UID 1000 ile çalışır.

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

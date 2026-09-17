---
title: Manga Çeviri
emoji: 📖
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: gpl-3.0
short_description: Japonca/İngilizce manga sayfalarını Hy-MT2-7B ile Türkçeye çevirir
---

# Manga Çeviri — Hy-MT2-7B

Japonca / İngilizce manga sayfalarını Türkçeye çevirir.

- **Pipeline:** [`zyddnys/manga-image-translator`](https://github.com/zyddnys/manga-image-translator) (algılama → OCR → metin silme → yerleştirme)
- **Çeviri modeli:** [`tencent/Hy-MT2-7B-GGUF`](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) · `Hy-MT2-7B-Q4_K_M.gguf`, llama.cpp `llama-server` üzerinden OpenAI-uyumlu API
- **Kaynak kod:** <https://github.com/Waifuhtr/mnaga>

Bu Space deposu bilerek küçük tutulmuştur: yalnızca `Dockerfile` ve bu
`README.md` vardır. Uygulama kaynağı build sırasında GitHub'dan çekilir, model
dosyaları ise build sırasında image katmanlarına gömülür.

## Donanım

Önce **CPU (8 vCPU / 32 GB)** ile build edin — bütün ağır indirme ve kurulum
işleri burada biter. Sonra donanımı **T4 Small**'a çevirin; konteyner aynı image
katmanlarından açılır, hiçbir şey yeniden indirilmez ve GPU offload devreye girer.

Çalışırken `/health` uç noktası modelin, llama-server'ın ve GPU'nun durumunu döner.

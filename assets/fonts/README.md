# Yazı tipleri (fontlar)

Bu klasör **font bırakma klasörüdür**. Buraya koyduğun her font, web arayüzündeki
"Yazı tipi" listesinde otomatik olarak çıkar. Kaç tane koyarsan o kadar seçenek
olur — kod, Dockerfile ya da arayüz dosyalarında hiçbir değişiklik gerekmez.

## Nasıl eklenir

1. Font dosyasını bu klasöre koy: `assets/fonts/<istediğin-ad>.ttf`
   Desteklenen uzantılar: `.ttf`, `.otf`, `.ttc`, `.otc`
2. Commit + push et.
3. Space'i yeniden derle (kod değişikliği ile aynı — rebuild gerekir, restart yetmez).
4. Arayüzde "Yazı tipi" listesinden seç. Seçtiğin font sayfanın **tamamında**
   kullanılır.

## Listede nasıl görünür

Listedeki isim **dosya adıdır** (uzantısız). Yani dosyaya ne ad verirsen
arayüzde onu görürsün:

| Dosya | Listede görünen | İç anahtar (`RENDER_FONT_KEY` için) |
|---|---|---|
| `CCWildWords.ttf` | CCWildWords | `ccwildwords` |
| `CC Wild Words.ttf` | CC Wild Words | `cc_wild_words` |
| `anime_ace_3.ttf` | anime_ace_3 | `anime_ace_3` |

İç anahtar kuralı: küçük harfe çevrilir, harf/rakam dışındaki her şey `_` olur.

## Türkçe harf uyarısı

Sunucu her fontu açıp `ç Ç ğ Ğ ı İ ö Ö ş Ş ü Ü` harflerini arar ve sonucu
arayüzde fontun altında gösterir:

- **"Türkçe harflerin tamamı var."** → sorun yok.
- **"⚠ ğĞıİşŞ harfleri bu yazı tipinde yok, yedekten gelir."** → sayfa yine de
  düzgün çıkar (kutucuk/tofu olmaz), ama o harfler yedek fontla (Arial Unicode)
  çizilir; kelime ortasında hafif bir stil kayması görünür.

Bu uyarıyı almamak için fontu Türkçe harfleri ekleyerek yamalayıp aynı adla bu
klasöre geri koyabilirsin — kodda hiçbir şey değişmez.

## Varsayılan font

`RENDER_FONT_KEY` ortam değişkeni hangi fontun **önceden seçili** geleceğini
belirler (bkz. `.env.example`). Liste yine bu klasörün tamamını gösterir.
Değer tanınmazsa uygulama var olan bir fonta düşer, fontsuz kalmaz.

## Buraya konmayan fontlar

manga-image-translator'ın kendi `fonts/` klasöründeki `comic shanns 2.ttf`,
`anime_ace.ttf` ve `anime_ace_3.ttf` de listede çıkar; onlar image içinde zaten
var, buraya kopyalamaya gerek yok. Aynı adla buraya bir dosya koyarsan seninki
üsttekini gölgeler.

`Arial-Unicode-Regular.ttf`, `msyh.ttc`, `msgothic.ttc` listeye alınmaz: onlar
eksik harflerin çizildiği **yedek** fontlardır, sayfa yazı tipi olarak
tasarlanmamışlardır.

## Lisans

Buraya koyduğun font dosyalarının dağıtım hakkı sana ait değilse, repo public
olduğunda sorun çıkarabilir. Fontun lisansını kendin kontrol et — bu klasördeki
dosyalardan proje sorumlu değildir.

'use strict';

// --------------------------------------------------------------------------
// State
// --------------------------------------------------------------------------
let selected = [];        // File objects, in reading order
let jobId = null;
let poller = null;

const $ = (id) => document.getElementById(id);

// Same rule as the server (app/server.py: natural_key) so the list the user
// sees before starting matches the order pages are actually translated in.
function naturalKey(name) {
  const stem = name.replace(/\.[^.]+$/, '');
  return stem.split(/(\d+)/).filter(Boolean).map((part) =>
    /^\d+$/.test(part) ? [1, parseInt(part, 10), ''] : [0, 0, part.toLowerCase()]
  );
}

function naturalCompare(a, b) {
  const ka = naturalKey(a), kb = naturalKey(b);
  for (let i = 0; i < Math.max(ka.length, kb.length); i++) {
    const x = ka[i], y = kb[i];
    if (!x) return -1;
    if (!y) return 1;
    for (let j = 0; j < 3; j++) {
      if (x[j] < y[j]) return -1;
      if (x[j] > y[j]) return 1;
    }
  }
  return 0;
}

function humanSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
  return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

// --------------------------------------------------------------------------
// File selection
// --------------------------------------------------------------------------
function addFiles(list) {
  const incoming = Array.from(list).filter((f) =>
    f.name.toLowerCase().endsWith('.zip') || f.type.startsWith('image/') ||
    /\.(png|jpe?g|webp|bmp|gif|tiff?)$/i.test(f.name)
  );
  if (!incoming.length) {
    showError('Seçilen dosyalar arasında görsel veya ZIP yok.');
    return;
  }
  clearError();
  for (const f of incoming) {
    if (!selected.some((s) => s.name === f.name && s.size === f.size)) selected.push(f);
  }
  // ZIPs last; images sorted by page number.
  selected.sort((a, b) => {
    const az = a.name.toLowerCase().endsWith('.zip'), bz = b.name.toLowerCase().endsWith('.zip');
    if (az !== bz) return az ? 1 : -1;
    return naturalCompare(a.name, b.name);
  });
  renderFileList();
}

function renderFileList() {
  const box = $('file-list');
  if (!selected.length) {
    box.hidden = true;
    $('start').disabled = true;
    return;
  }
  box.hidden = false;
  box.innerHTML = selected.map((f, i) => `
    <div class="file-row">
      <span class="idx">${i + 1}</span>
      <span class="nm">${escapeHtml(f.name)}</span>
      <span class="sz">${humanSize(f.size)}</span>
    </div>`).join('');
  $('start').disabled = false;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------
function showError(msg) {
  const el = $('error');
  el.textContent = msg;
  el.hidden = false;
}
function clearError() { $('error').hidden = true; }

// --------------------------------------------------------------------------
// Fonts
// --------------------------------------------------------------------------
// The list is whatever the server found in assets/fonts/ - nothing is hardcoded
// here, so adding a font to the repo is enough to make it selectable.
let fontInfo = {};

async function loadFonts() {
  const sel = $('font');
  try {
    const r = await fetch('/api/fonts');
    const d = await r.json();
    if (!d.fonts?.length) throw new Error('yazı tipi bulunamadı');

    fontInfo = Object.fromEntries(d.fonts.map((f) => [f.key, f]));
    sel.innerHTML = d.fonts.map((f) =>
      `<option value="${escapeHtml(f.key)}"${f.key === d.default ? ' selected' : ''}>` +
      `${escapeHtml(f.label)}</option>`
    ).join('');
    sel.value = d.default || d.fonts[0].key;
  } catch (e) {
    sel.innerHTML = '<option value="">(varsayılan)</option>';
  }
  showFontNote();
}

function showFontNote() {
  const f = fontInfo[$('font').value];
  const note = $('font-note');
  if (!f) { note.textContent = ''; return; }
  if (f.turkish === 'partial') {
    note.textContent = `⚠ ${f.missing_glyphs} harfleri bu yazı tipinde yok, yedekten gelir.`;
  } else if (f.turkish === 'full') {
    note.textContent = 'Türkçe harflerin tamamı var.';
  } else {
    note.textContent = '';
  }
}

// --------------------------------------------------------------------------
// Health
// --------------------------------------------------------------------------
async function checkHealth() {
  const el = $('health');
  try {
    const r = await fetch('/health');
    const d = await r.json();
    if (d.status === 'ok') {
      el.className = 'health ok';
      el.textContent = d.gpu ? 'Hazır · GPU etkin' : 'Hazır · CPU modu (yavaş)';
    } else {
      el.className = 'health bad';
      const why = !d.llama_server?.reachable ? 'çeviri modeli yüklenmedi' : d.manga_image_translator;
      el.textContent = 'Hazır değil: ' + why;
    }
  } catch (e) {
    el.className = 'health bad';
    el.textContent = 'Sunucuya ulaşılamıyor';
  }
}

// --------------------------------------------------------------------------
// Job
// --------------------------------------------------------------------------
async function start() {
  if (!selected.length) return;
  clearError();
  $('start').disabled = true;
  $('progress-wrap').hidden = false;
  $('results-card').hidden = true;
  $('results').innerHTML = '';
  setProgress(0, 'Yükleniyor…', '');

  const fd = new FormData();
  for (const f of selected) fd.append('files', f, f.name);
  fd.append('target_lang', $('target_lang').value);
  fd.append('font', $('font').value);
  fd.append('ocr', $('ocr').value);
  fd.append('detector', $('detector').value);
  // One control drives both thresholds; they are only ever tuned together.
  const [textTh, boxTh] = $('sensitivity').value.split(',');
  fd.append('text_threshold', textTh);
  fd.append('box_threshold', boxTh);
  fd.append('inpainter', $('inpainter').value);
  fd.append('detection_size', $('detection_size').value);
  fd.append('inpainting_size', $('inpainting_size').value);
  fd.append('font_size_offset', $('font_size_offset').value);
  fd.append('debug', $('debug').checked ? 'true' : 'false');

  try {
    const r = await fetch('/api/jobs', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || ('HTTP ' + r.status));
    jobId = d.id;
    $('debug-card').hidden = !$('debug').checked;
    render(d);
    poller = setInterval(poll, 1500);
  } catch (e) {
    showError('Çeviri başlatılamadı: ' + e.message);
    $('start').disabled = false;
    $('progress-wrap').hidden = true;
  }
}

async function poll() {
  if (!jobId) return;
  try {
    const r = await fetch('/api/jobs/' + jobId);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    render(d);
    if (d.status === 'done' || d.status === 'error') {
      clearInterval(poller);
      poller = null;
      $('reset').hidden = false;
      if (d.status === 'error' && d.error) showError(d.error);
    }
  } catch (e) {
    clearInterval(poller);
    poller = null;
    showError('Durum alınamadı: ' + e.message);
    $('reset').hidden = false;
  }
}

const STAGE_TR = {
  upscaling: 'Görsel büyütülüyor',
  detection: 'Metin algılanıyor',
  ocr: 'Metin okunuyor (OCR)',
  'mask-generation': 'Maske hazırlanıyor',
  inpainting: 'Orijinal metin siliniyor',
  translating: 'Türkçeye çevriliyor',
  rendering: 'Metin yerleştiriliyor',
  colorizing: 'Renklendiriliyor',
  downscaling: 'Görsel küçültülüyor',
  loading: 'Modeller yükleniyor',
  finished: 'Tamamlandı',
  failed: 'Başarısız',
};

function setProgress(pct, text, stage) {
  $('bar-fill').style.width = pct + '%';
  $('progress-text').textContent = text;
  $('stage-text').textContent = stage || '';
}

function render(d) {
  const stage = STAGE_TR[d.stage] || d.stage || '';
  setProgress(d.percent, d.message || `${d.done}/${d.total} sayfa`, stage);

  const anyDone = d.pages.some((p) => p.status === 'done');
  $('results-card').hidden = !anyDone;

  $('results').innerHTML = d.pages.map((p) => {
    if (p.status === 'done') {
      return `<div class="result">
        <img loading="lazy" src="/api/jobs/${d.id}/page/${p.index}?t=${p.seconds}" alt="${escapeHtml(p.name)}"
             onclick="openViewer(this.src)">
        <div class="meta"><span class="nm">${escapeHtml(p.name)}</span><span class="st done">${p.regions} blok</span></div>
        ${p.warning ? `<div class="warn">${escapeHtml(p.warning)}</div>` : ''}
      </div>`;
    }
    const label = { pending: 'sırada', running: 'çevriliyor', error: 'hata' }[p.status] || p.status;
    return `<div class="result">
      <div class="meta"><span class="nm">${escapeHtml(p.name)}</span><span class="st ${p.status}">${label}</span></div>
      ${p.error ? `<div class="err">${escapeHtml(p.error)}</div>` : ''}
    </div>`;
  }).join('');
}

function openViewer(src) {
  let dlg = document.querySelector('dialog.viewer');
  if (!dlg) {
    dlg = document.createElement('dialog');
    dlg.className = 'viewer';
    dlg.innerHTML = '<img alt="">';
    dlg.addEventListener('click', () => dlg.close());
    document.body.appendChild(dlg);
  }
  dlg.querySelector('img').src = src;
  dlg.showModal();
}
window.openViewer = openViewer;

function reset() {
  if (poller) { clearInterval(poller); poller = null; }
  jobId = null;
  selected = [];
  $('pick-images').value = '';
  $('pick-zip').value = '';
  renderFileList();
  $('progress-wrap').hidden = true;
  $('results-card').hidden = true;
  $('reset').hidden = true;
  $('results').innerHTML = '';
  clearError();
}

// --------------------------------------------------------------------------
// Wiring
// --------------------------------------------------------------------------
$('pick-images').addEventListener('change', (e) => addFiles(e.target.files));
$('pick-zip').addEventListener('change', (e) => addFiles(e.target.files));
$('start').addEventListener('click', start);
$('reset').addEventListener('click', reset);

$('download').addEventListener('click', () => {
  if (jobId) window.location.href = `/api/jobs/${jobId}/download`;
});

$('debug-refresh').addEventListener('click', async () => {
  try {
    const r = await fetch('/api/debug/llama');
    $('debug-out').textContent = JSON.stringify(await r.json(), null, 2);
  } catch (e) {
    $('debug-out').textContent = 'Hata: ' + e.message;
  }
});

$('font').addEventListener('change', showFontNote);

$('debug').addEventListener('change', (e) => {
  $('debug-card').hidden = !e.target.checked;
});

const drop = $('drop');
['dragenter', 'dragover'].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => {
  if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
});

loadFonts();
checkHealth();
setInterval(checkHealth, 30000);

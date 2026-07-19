# Aset gambar

## Foto hero landing page

Taruh foto asli Kabupaten Batang di folder ini dengan nama:

- `hero-batang.webp` — utama. Resolusi sumber **2560×1440**, WebP quality ±72
  (target 200–350 KB).
- `hero-batang.jpg` — fallback untuk browser lama, JPEG quality ±85.

Tidak perlu mengubah kode apa pun: `js/app.js` mendeteksi file ini otomatis dan
CSS (`css/style.css`, selector `.hero.has-photo`) langsung memakainya di bawah
overlay gradien navy. Selama file belum ada, hero memakai gradien murni.

Sumber foto yang aman: foto sendiri, media resmi Pemkab Batang (dengan izin,
dicatat di skripsi), atau Wikimedia Commons berlisensi CC-BY (cantumkan
atribusi di footer).

## Logo

`logo-batang.png` (opsional) — saat ini header memakai lambang SVG generik
inline di `index.html`; ganti elemen `.brand-mark` bila logo resmi tersedia.

# Deploy ke Hugging Face Spaces (gratis, 16 GB RAM)

Hasil akhir: URL publik `https://huggingface.co/spaces/<username>/<nama-space>`
yang bisa dibuka siapa pun. Biaya: Rp 0 (free tier: 2 vCPU, 16 GB RAM).

## Sekali saja: siapkan akun & Space

1. **ROTATE API KEY DULU.** Buat `GEMINI_API_KEY` baru di
   https://aistudio.google.com/apikey (key lama pernah ke-commit di history
   git — jangan dipakai untuk deploy). Update juga `.env` lokalmu.
2. Daftar akun di https://huggingface.co (gratis).
3. Buat Space baru: **New Space** → beri nama (mis. `chatbot-batang`) →
   SDK pilih **Docker** → template **Blank** → visibility **Public**
   (atau Private + share ke akun penguji).
4. Di Space → **Settings → Variables and secrets** → **New secret**:
   - Name: `GEMINI_API_KEY`, Value: key BARU dari langkah 1.
   - (Opsional tapi disarankan) Name: `RAGTRIAL_ADMIN_TOKEN`, Value: string
     acak panjang buatanmu sendiri — dipakai untuk mengunduh traces (lihat
     bawah).

## Push kode ke Space

Space adalah repo git. Dari folder project:

```powershell
git remote add space https://huggingface.co/spaces/<username>/<nama-space>
git push space main
```

(Perlu login: `huggingface-cli login`, atau pakai token HF sebagai password
saat push — buat di https://huggingface.co/settings/tokens, role `write`.)

**Satu syarat HF**: bagian paling atas `README.md` di Space harus punya
frontmatter YAML. Tambahkan blok ini di baris PERTAMA `README.md` sebelum push
(aman — GitHub hanya menampilkannya sebagai tabel kecil):

```yaml
---
title: Chatbot Layanan Publik Kab. Batang
emoji: 🏛️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---
```

Setelah push, HF otomatis build Dockerfile (±10–20 menit pertama kali,
karena torch + model reranker ±1 GB ikut di-bake ke image) lalu app hidup.
Build berikutnya lebih cepat (layer di-cache).

## Update / revisi belakangan

```powershell
git push space main
```

Itu saja — HF rebuild dan restart otomatis. (Ini jawaban untuk "revisi
belakangan ribet nggak": tidak.)

## PENTING: traces (data penelitian)

Disk Spaces **ephemeral** — isi `data/traces/` HILANG setiap
restart/rebuild/sleep. Karena itu:

- **Unduh traces segera setelah tiap sesi pengujian**, lewat endpoint yang
  sudah disediakan (aktif hanya bila secret `RAGTRIAL_ADMIN_TOKEN` di-set):

  ```
  https://<username>-<nama-space>.hf.space/api/traces/export?token=<tokenmu>
  ```

  → mengunduh `traces.zip`. Simpan per sesi, gabungkan di lokal.
- Jangan jadwalkan sesi pengujian berdekatan dengan push revisi (push =
  rebuild = traces yang belum diunduh hilang).

## Perilaku yang normal (jangan panik)

| Gejala | Penjelasan |
|---|---|
| App "Sleeping" setelah ±48 jam tidak diakses | Normal di free tier — buka URL-nya, tunggu ±1–2 menit bangun |
| Setelah bangun/restart, jawaban pertama lambat | Warmup index + reranker ±30 detik (cek `/api/health` → `ready: true`) |
| Jawaban "Sistem sedang sibuk" | Kuota Gemini API tercapai sesaat — coba lagi |

## Kalau suatu saat pindah dari HF

`Dockerfile`-nya standar — bisa dipakai apa adanya di Railway/Fly/VPS mana
pun (ubah port lewat argumen `--port` di CMD bila perlu).

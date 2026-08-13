# Testset Curation — in-scope selection dari 202 kandidat

Dihasilkan oleh `eval/curate_testset.py` (read-only terhadap `eval/testset.json`). Setiap `gold_chunks` di-resolve balik ke teks chunk asli lewat pipeline chunking yang sama dengan `scripts/build_vectorstore.py`, jadi penilaian evidence-nya dilakukan terhadap teks yang benar-benar bisa di-retrieve, bukan terhadap label bawaan.

## Overall selection

| | n |
|---|---:|
| total candidates | 202 |
| in-scope candidates | 193 |
| selected in-scope | 110 |
| excluded / unused in-scope | 83 |
| OOS excluded from this selection | 9 |

OOS yang dikecualikan: NO001, NO002, NO003, NO004, NO006, NO007, NO008, NO009, NO010 — set OOS ~20 soal dibuat terpisah nanti.

## Domain distribution

| Domain | Target | Pool | Selected |
|---|---:|---:|---:|
| Sosial | ~25 | 61 | 25 |
| Dukcapil | ~25 | 60 | 25 |
| OPD | ~25 | 25 | 25 |
| Perizinan | ~25 | 31 | 25 |
| Cross-domain/ambiguous | ~10 | 16 | 10 |
| **Total** | **110** | **193** | **110** |

## Query-type distribution

Kategori ditentukan ulang dari evidence (bukan dari label `query_type` bawaan): **Complex** kalau butuh >1 gold chunk / lintas-store / sintesis banyak fakta; sisanya **Lexical** vs **Semantic** lewat overlap verbatim istilah paling distinktif dari pertanyaan terhadap gold chunk (ambang 0.70).

| Category | Target | Pool | Selected |
|---|---:|---:|---:|
| Lexical-oriented | ~45 | 53 | 39 |
| Semantic-oriented | ~45 | 79 | 45 |
| Complex | ~20 | 61 | 26 |

### Domain x query-type (selected)

Ada batas: satu query type maksimal ~60% dari jatah satu domain, kecuali pool domain itu emang ga nyediain pilihan lain (OPD isinya hampir semua semantic). Tanpa batas ini target lexical global gampang dipenuhin dengan numpukin satu domain penuh soal lookup gampang, dan capability coverage-nya jadi jelek.

| Domain | Lexical | Semantic | Complex |
|---|---:|---:|---:|
| Sosial | 13 | 5 | 7 |
| Dukcapil | 15 | 7 | 3 |
| OPD | 3 | 22 | 0 |
| Perizinan | 8 | 11 | 6 |
| Cross-domain/ambiguous | 0 | 0 | 10 |

## Quality summary

| Tier | Definisi | In-scope pool | Selected |
|---|---|---:|---:|
| high | evidence & fakta ter-grounding penuh, tanpa flag | 163 | 110 |
| acceptable | usable tapi ada flag lemah (mis. sebagian anchor jawaban ga ketemu) | 14 | 0 |
| problematic | ada cacat evidence/jawaban yang bikin ga layak dipakai apa adanya | 16 | 0 |
| unusable | gold_id ga resolve / ga ada gold / ga ada expected_facts | 0 | 0 |

### Alasan penolakan / flag (seluruh in-scope pool)

| Flag | n | Kenapa masalah |
|---|---:|---|
| `boilerplate_gold` | 11 | gold chunk-nya daftar sitasi doang (*Mengingat/Menimbang* atau daftar pustaka naskah akademik) — nyebutin banyak peraturan & judul tapi ga nyatain apa pun yang dibutuhin jawabannya |
| `unsupported_numbers` | 11 | ada angka di jawaban yang ga muncul di chunk yang disitir |
| `self_citing` | 9 | pertanyaannya udah nyebut Pasal + nomor peraturannya sendiri, jadi retriever dikasih lokasi jawabannya (masih valid sebagai kasus lexical, cuma lemah — dideprioritasin, bukan ditolak) |
| `english_gold` | 5 | gold-nya dokumen terjemahan Inggris — pertanyaan Indonesia vs evidence Inggris itu artefak korpus, bukan kemampuan retrieval yang mau diukur |
| `context_reference` | 4 | pertanyaannya nyebut "berdasarkan rujukan/informasi yang diberikan" — ngandaiin ada konteks yang user asli ga punya, jadi ga valid sebagai query eval |
| `metadata_only_answer` | 3 | jawabannya ngandelin status berlaku dokumen, yang adanya di metadata chunk dan ga pernah muncul di teks yang di-retrieve |
| `redundant` | 3 | nanya info / nembak evidence yang sama dengan kandidat lain |
| `facts_weak` | 1 | expected_facts cuma sebagian didukung chunk yang disitir |
| `answer_ungrounded` | 1 | baik kata-kata jawabannya maupun istilah kuncinya ga ada di evidence |

Dari 83 kandidat in-scope yang ga kepilih: **16 ditolak karena cacat evidence/jawaban** (tier `problematic`), **14 dikesampingin** karena ada flag lemah sementara masih ada kandidat `high` di domain yang sama, dan sisanya **53 kualitasnya oke tapi kepotong kuota domain** atau dihindari karena redundan / nembak chunk & topik yang udah diwakili soal lain.

### Kandidat bermasalah (semua dikeluarkan dari selection)

| ID | Domain | Flags | Gold |
|---|---|---|---|
| DK014 | dukcapil | `context_reference` | `dukcapil:page:278` |
| DK023 | dukcapil | `context_reference` | `dukcapil:page:270` |
| DK027 | dukcapil | `context_reference` | `dukcapil:page:35` |
| HD006 | cross | `boilerplate_gold` | `sosial:id:sosial-perbup-34-2024#narr:2` |
| HD011 | sosial | `boilerplate_gold` | `sosial:id:sosial-perbup-7-2016#pasal:12.5` |
| PZ023 | perizinan | `context_reference` | `perizinan:id:izin-praktik-elektromedis` |
| SO010 | sosial | `boilerplate_gold` | `sosial:id:sosial-perbup-58-2025#preamble` |
| SO019 | sosial | `boilerplate_gold`, `metadata_only_answer`, `unsupported_numbers` | `sosial:id:sosial-perbup-26-2021#narr:1` |
| SO022 | sosial | `facts_weak`, `answer_ungrounded`, `boilerplate_gold`, `metadata_only_answer`, `self_citing`, `unsupported_numbers` | `sosial:id:sosial-perbup-34-2024#narr:3` |
| SO029 | sosial | `boilerplate_gold`, `unsupported_numbers` | `sosial:id:sosial-abstrak-perda-2-2016#narr:1` |
| SO037 | sosial | `boilerplate_gold`, `self_citing` | `sosial:id:sosial-perda-10-2020#preamble.2` |
| SO039 | sosial | `metadata_only_answer` | `sosial:id:sosial-perbup-40-2018#narr:10` |
| SO042 | sosial | `boilerplate_gold` | `sosial:id:sosial-naskah-akademis-f4b90f06-2022#preamble.8` |
| SO049 | sosial | `boilerplate_gold` | `sosial:id:sosial-abstrak-perda-10-2020#narr:5` |
| SO057 | sosial | `boilerplate_gold` | `sosial:id:sosial-ranperbup-6e14ff06-2025#preamble` |
| SO059 | sosial | `boilerplate_gold`, `unsupported_numbers` | `sosial:id:sosial-abstrak-perda-10-2020#narr:2` |

## Feasibility

**Domain ~25/25/25/25 + 10: feasible, tapi OPD nol slack.**

| Domain | Usable (high+acceptable) | Butuh | Slack |
|---|---:|---:|---:|
| Sosial | 50 | 25 | +25 |
| Dukcapil | 57 | 25 | +32 |
| OPD | 25 | 25 | +0 |
| Perizinan | 30 | 25 | +5 |
| Cross-domain/ambiguous | 15 | 10 | +5 |

- **OPD** cuma punya 25 kandidat buat target 25 — jadi ga ada seleksi sama sekali di domain ini, semuanya masuk. Kalau nanti ada yang gugur pas manual audit, ga ada cadangan; harus generate soal OPD baru.
- **Perizinan** 31 kandidat buat 25 → slack 6.
- **Cross-domain** 16 kandidat buat 10 → slack cukup.
- **Sosial & Dukcapil** longgar (61 & 60), di situ seleksi kualitas & diversitas beneran kerja.

**Query-type ~45/45/20: mepet, dan bentuknya ditentukan sama isi domain.**

- OPD kepaksa nyumbang 22 soal semantic (isinya record kontak OPD yang terstruktur, hampir selalu ditanya pakai sinonim), dan cross-domain kepaksa nyumbang 10 complex. Dua blok ini udah ngunci ~32 dari 110 slot sebelum domain lain disentuh.

- Total kandidat lexical di seluruh pool cuma 53. Buat nembak 45 harus ambil ~85% dari semua kandidat lexical yang ada — artinya nyaris tanpa seleksi kualitas, dan praktiknya bikin satu domain (Dukcapil) ketimbun soal lookup gampang sampe 76% jatahnya. Makanya target lexical dilonggarin ke 39 dan complex kebawa naik ke 26.

- **Rekomendasi:** target domain dipertahankan; target query-type dianggap panduan. Hasil 39/45/26 (lexical/semantic/complex) masih dalam koridor dan ga ngorbanin kualitas.

## Difficulty & coverage

- difficulty: easy 51, medium 47, hard 12
- gold-set unik: 110/110 (ga ada dua soal terpilih yang nembak kombinasi chunk yang sama persis)
- topik/dokumen berbeda yang tersentuh: 102
- soal multi-chunk / cross-store: 20
- redundan yang ikut kepilih: 1

## Manual audit subset (25)

Random 5 per domain **di dalam** tiap grup (seed tetap), bukan 25 teratas — biar sampelnya representatif buat dicek balik ke dokumen sumber.

| ID | Domain | Query type | Difficulty | Pertanyaan |
|---|---|---|---|---|
| CR001 | Cross-domain/ambiguous | complex | medium | Jika saya ingin mengajukan Izin Praktik Bidan dan memiliki pertanyaan terkait prosedur atau per… |
| CR004 | Cross-domain/ambiguous | complex | medium | Jika seorang bidan yang menjadi korban bencana alam ingin memperpanjang Izin Praktik Bidan dan … |
| CR005 | Cross-domain/ambiguous | complex | medium | Jika seorang fisioterapis mengajukan Izin Praktik Fisioterapis yang memiliki masa berlaku 5 tah… |
| DK017 | Dukcapil | semantic | medium | Apabila seseorang yang sebelumnya tercatat sebagai warga nonpermanen berencana untuk mengubah a… |
| DK028 | Dukcapil | lexical | medium | Apabila seorang anak yang lahir dari perkawinan campuran di luar negeri belum didaftarkan ke Pe… |
| DK037 | Dukcapil | semantic | medium | Bagaimana cara menentukan tanggal pernikahan yang dicantumkan pada Kartu Keluarga, dan apa kons… |
| DK039 | Dukcapil | lexical | medium | Seorang warga negara asing yang memiliki izin tinggal tetap dan ingin berpindah domisili antar … |
| HD001 | Cross-domain/ambiguous | complex | hard | Perkawinan saya sudah diputuskan pengadilan dan berkekuatan hukum tetap, tapi istri saya sedang… |
| HD005 | Cross-domain/ambiguous | complex | hard | Pengadilan sudah menyatakan pembatalan perceraian saya berkekuatan hukum tetap dan saya ingin s… |
| HD007 | Dukcapil | complex | hard | Kalau saya pindah alamat antar-kelurahan dalam satu kabupaten tapi sudah terlanjur tinggal di a… |
| HD012 | Sosial | complex | hard | Dalam program Jaminan Sosial Ketenagakerjaan bagi Pekerja Rentan di Batang, mengapa seorang pek… |
| HD013 | Sosial | complex | hard | Setelah Dinas Sosial menerima tagihan iuran dari BPJS Ketenagakerjaan untuk penerima bantuan ya… |
| OP002 | OPD | semantic | easy | Berapakah nomor telepon yang dapat dihubungi untuk Inspektorat Kabupaten Batang? |
| OP006 | OPD | semantic | easy | Jika Anda ingin mengirimkan surat elektronik kepada kantor Kecamatan Pecalungan di Kabupaten Ba… |
| OP008 | OPD | semantic | easy | Jika Anda ingin menghubungi Badan Kepegawaian Daerah Kabupaten Batang melalui surat elektronik,… |
| OP016 | OPD | semantic | easy | Jika Anda ingin menghubungi Bagian Pengendalian Pembangunan di Kabupaten Batang melalui surat e… |
| OP023 | OPD | semantic | easy | Berapakah nomor telepon Kelurahan Sambong? |
| PZ007 | Perizinan | semantic | medium | Jika sebuah lembaga ingin mengajukan izin untuk operasional atau pendirian PAUD di Kabupaten Ba… |
| PZ013 | Perizinan | lexical | easy | Berapa lama perkiraan waktu yang dibutuhkan untuk menyelesaikan proses permohonan Izin Operasio… |
| PZ016 | Perizinan | semantic | easy | Untuk Izin Praktik Psikolog Klinis, mengapa pemohon tidak perlu khawatir soal biaya, dan berapa… |
| PZ021 | Perizinan | lexical | easy | Apa saja persyaratan yang dibutuhkan untuk mengajukan Izin Praktik Okupasi Terapis di Pemkab Ba… |
| PZ025 | Perizinan | complex | medium | Jika seorang dokter ingin mengajukan Izin Praktik Dokter di Kabupaten Batang, apa saja dokumen … |
| SO027 | Sosial | complex | medium | Apa perbedaan utama antara Sistem Jaringan Jalan Primer dan Sistem Jaringan Jalan Sekunder dala… |
| SO048 | Sosial | lexical | easy | Kepada siapa harus dilaporkan jika jenazah akan dibawa dari wilayah Kabupaten Batang ke luar wi… |
| SO065 | Sosial | complex | medium | Apa saja tugas Pokja Pengarusutamaan Gender (PUG) di Batang terkait peningkatan partisipasi sek… |

Artefak: `eval/selection_inscope_110.json` (selected_ids + manual_audit_subset), `eval/results/curation_audit.json` (skor & flag per kandidat).

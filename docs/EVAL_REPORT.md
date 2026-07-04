# Laporan Evaluasi — 3 Mode RAG (naive / enhanced / agentic)

_Tanggal: 2026-07-01 · Test set: `eval/testset.json` (198 soal synthetic, 4 domain, full ground-truth)_

## 1. Ringkasan Eksekutif

Evaluasi membandingkan tiga mode RAG pada corpus yang sama (dukcapil, opd, perizinan, sosial — 2.678 chunk) di empat sumbu: **retrieval**, **routing/intent**, **answer-quality** (LLM-as-judge), dan **latency**.

**Temuan utama:**
1. **Naive fan-out menang di retrieval** (recall@5 **0.81**) karena selalu mencari ke semua store. Enhanced (0.62) & agentic (0.65) kehilangan recall akibat *routing error* — kalau salah pilih store, retrieval = 0.
2. **Agentic menang di routing & answer-quality**: routing accuracy **0.92** (vs enhanced 0.72), faithfulness **0.95**, fact_recall **0.77**, dan latency total **terendah** (5.3s). Ini mode paling seimbang.
3. **Intent gate memisahkan naive dari dua lainnya**: naive **tidak pernah menolak** out-of-scope (recall_invalid **0.00**), sedangkan enhanced & agentic **1.00**. Untuk layanan publik (anti-halusinasi), ini krusial.
4. **Enhanced adalah titik terlemah di tengah**: routing 0.72 menekan recall, false-refusal tertinggi (0.35), dan latency tertinggi (14.6s) karena HyDE — sementara andalannya (reranker & hybrid) masih *stub/dormant* sehingga tak memberi gain. Lihat §6.
5. **Domain `sosial` paling sulit** di semua mode (recall@5 0.30–0.64) — granularitas chunk-precise (pasal vs penjelasan) + pasal near-duplicate antar-perda. Domain atomik (opd/perizinan) mendekati sempurna.

## 2. Tabel Perbandingan

| Metric | naive | enhanced | agentic |
|---|---|---|---|
| retrieval hit@5 | **0.851** | 0.654 | 0.693 |
| retrieval recall@5 | **0.811** | 0.621 | 0.647 |
| retrieval MRR | **0.718** | 0.568 | 0.579 |
| routing accuracy | — | 0.721 | **0.919** |
| routing store-correct | — | 0.716 | **0.919** |
| answer fact_recall | 0.649 | 0.638 | **0.765** |
| answer faithfulness | 0.855 | 0.872 | **0.954** |
| answer answer_relevance | 0.833 | 0.725 | **0.855** |
| refusal_correct (none) | 1.000 | 1.000 | 1.000 |
| **false_refusal_rate** ↓ | **0.182** | 0.353 | 0.200 |
| latency total mean (s) ↓ | 6.35 | 14.57 | **5.32** |
| latency total p95 (s) ↓ | 13.33 | 23.80 | **10.27** |

_Retrieval/routing/intent dihitung atas **198 soal penuh** (3 mode). Answer-quality (judge) atas **subset 60 soal** representatif (naive 60, enhanced 56, agentic 60 terjudge — sisanya rate-limit). Intent eval atas `intent_testset.json` (40 soal)._

## 3. Retrieval per Domain (recall@5)

| domain | n | naive | enhanced | agentic |
|---|---|---|---|---|
| opd | 25 | **1.00** | 1.00 | 0.96 |
| perizinan | 29 | **0.97** | 0.62 | 0.93 |
| dukcapil | 55 | **0.92** | 0.89 | 0.81 |
| sosial | 70 | **0.64** | 0.30 | 0.32 |
| both (cross) | 10 | 0.50 | 0.40 | 0.45 |

**Baca:** domain atomik (1 record = 1 chunk) hampir sempurna. `sosial` (90% index, chunk-precise) jadi pembeda kesulitan utama. Enhanced anjlok di `perizinan` (0.62) & `sosial` (0.30) karena routing sering salah arah; naive tak kena masalah ini karena fan-out. Cross-store (`both`) susah di recall karena harus dapat 2 chunk dari 2 domain sekaligus.

## 4. Routing & Intent

- **Routing** (enhanced vs agentic): agentic 0.92 jauh di atas enhanced 0.72. Semantic-router enhanced sering salah untuk query sosial (nyasar) & `both` (recall 0.60). Agentic (LLM tool-calling) memilih store jauh lebih akurat; confusion utamanya `none`→sosial (over-retrieve, precision none 0.53).
- **Intent gate** (VALID/INVALID, 40 soal):

  | mode | accuracy | recall_invalid | recall_valid |
  |---|---|---|---|
  | naive | 0.487 | **0.00** | 1.00 |
  | enhanced | **1.00** | **1.00** | 1.00 |
  | agentic | **1.00** | **1.00** | 1.00 |

  Naive tak punya gate → jawab semua (termasuk chit-chat & out-of-scope). Enhanced/agentic menolak sempurna pada subtype chitchat & oos.

## 5. Answer-Quality (subset 60, LLM-judge)

Agentic unggul di ketiga metrik: **fact_recall 0.77**, **faithfulness 0.95**, **answer_relevance 0.86**. Enhanced punya faithfulness bagus (0.87) tapi answer_relevance rendah (0.73) & **false_refusal 0.35** (paling sering nolak padahal info ada) — konsisten dengan recall retrieval-nya yang rendah (kalau retrieval kosong, model cenderung refuse). Naive relevance tinggi (0.83) tapi fact_recall & faithfulness lebih rendah dari agentic.

Breakdown by query_type (agentic) menandai titik sulit: `negation_edge` (hit 0.33), `multi_chunk` recall 0.36, `semantic` 0.55 — tipe non-lexical memang lebih menantang, sesuai desain test set.

## 6. Catatan Arsitektur (kenapa hasilnya begini)

- Enhanced mengaktifkan **HyDE + intent gate + semantic router**, tapi **reranker = stub** (`NotImplementedError`) dan **hybrid = dormant** (default dense). Jadi biaya HyDE (latency ↑↑) tak diimbangi gain re-ranking → enhanced kalah retrieval dari naive & lebih lambat. **Ini bukan bug eval — ini kondisi implementasi saat ini.**
- Naive unggul retrieval **hanya** karena fan-out; ia bayar mahal di intent (tak bisa refuse) & precision.
- Agentic paling seimbang: routing bagus → retrieval layak, answer-quality tertinggi, latency terendah, refuse benar.

**Implikasi untuk revisi berikutnya:** aktifkan reranker (low-effort, high-ROI) & hybrid untuk enhanced, dan tambah reflection/grading node di agentic — lalu re-run eval ini untuk mengukur delta.

## 7. Limitasi

- Answer-quality hanya subset 60/198 (rate-limit embedding free-tier). Retrieval/routing/intent penuh.
- Enhanced answer-quality atas 56 soal (4 rate-limit gagal); 1 retrieval-error tersisa (naive & enhanced masing-masing 1).
- Test set synthetic (LLM-generated, grounded ke chunk asli, terverifikasi gold-id existence + de-leakage). Bukan kurasi manusia penuh.
- LLM-judge (gemini-2.5-flash temp 0) punya noise; faithfulness/fact_recall indikatif, bukan absolut.
- Latency dari free-tier (termasuk retry rate-limit di `wall`) — angka relatif antar-mode valid, angka absolut akan beda di tier berbayar.

## 8. Cara Reproduksi

```bash
# 1. (bila chunk berubah) rebuild vector store dulu, lalu:
python -m eval.gen_testset --domain all --fresh --inter-call-delay 4   # generate test set
python -m eval.verify_testset                                          # validasi

# 2. retrieval + routing (3 mode, 198 soal)
python -m eval.run_eval --systems naive enhanced agentic --k 5 --no-judge --sleep 2

# 3. intent (3 mode, 40 soal)
python -m eval.run_intent_eval --systems naive enhanced agentic --sleep 4

# 4. answer-quality judge (subset ~60 soal via --ids, 3 mode)
python -m eval.run_eval --systems naive enhanced agentic --k 5 --sleep 4 --ids <60-id-subset>

# 5. agregasi + tabel perbandingan
python -m eval.analyze --systems naive enhanced agentic --k 5 \
    --breakdown expected_route difficulty query_type --save
```

## 9. Anggaran Waktu Evaluasi (patokan untuk re-run)

Diukur dari run ini (free-tier, rate-limit embedding ketat). "Kompute" = `sum(timings.wall)`, sudah termasuk retry rate-limit yang blocking; "real" menambah jeda `--sleep` antar-soal.

| Tahap | Cakupan | Kompute | Real (dgn sleep) |
|---|---|---|---|
| B1 retrieval | 3 mode × 198 | ~86 min | ~1.5 jam |
| — naive | 198 | 21 min | — |
| — enhanced | 198 | **48 min** (rate-limit ↑) | — |
| — agentic | 198 | 18 min | — |
| B3 intent | 3 mode × 40 | ~9 min | ~15 min |
| B2 judge | 3 mode × 60 (subset) | ~24 min | ~40 min |
| analyze + report | — | ~1 min | ~5 min |
| **TOTAL 1 siklus** | | **~2 jam** | **~2.5–3 jam** |

**Patokan praktis untuk re-generate setelah revisi arsitektur:**
- **Siklus penuh (judge subset):** sisihkan **~2.5–3 jam**, mostly bisa ditinggal (background).
- **Retrieval saja (cek delta cepat):** **~1.5 jam** (skip judge & intent).
- **Bottleneck = enhanced** (sensitif rate-limit embedding, ~2.5× lebih lama/soal dari agentic/naive). Kalau kuota longgar / tier berbayar, total bisa turun drastis (kompute murni non-enhanced ~40 menit).
- **Hemat:** naikkan `--sleep` untuk kurangi error (lebih sedikit re-run), atau judge subset lebih kecil (~30 soal) kalau cuma butuh sinyal answer-quality.

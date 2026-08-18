# Evaluasi Generation dengan RAGAS

Dokumen ini menjelaskan instrumen yang dipakai untuk menilai **kualitas jawaban** ketiga arsitektur
(Naive / Enhanced / Agentic), sebagai bahan penulisan Bab III skripsi.

Sejak 18 Agustus 2026, evaluasi generation memakai **RAGAS 0.4.3** dengan dua metric:
**Faithfulness** dan **Semantic Similarity**. Custom LLM-as-a-Judge yang dipakai sebelumnya
dipensiunkan dan diarsipkan di [`archive/eval_judge/`](../archive/eval_judge/).

---

## 1. Kenapa berpindah dari custom judge ke RAGAS

Instrumen lama adalah tiga judge holistik dengan prompt Bahasa Indonesia yang ditulis sendiri
(`judge_faithfulness`, `judge_answer_relevance`, `judge_refusal`), masing-masing satu panggilan LLM
yang mengembalikan skor kasar 0/1/2. Dua masalahnya:

1. **Tidak tervalidasi.** Validitas prompt sepenuhnya bersandar pada penilaian penulis sendiri.
   Sebagai alat ukur dalam skripsi, ini sulit dipertahankan — sudah tercatat sendiri sebagai
   limitation di `EVAL_REPORT.md` §11.
2. **Holistik, bukan statement-level.** Jawaban dengan 5 klaim yang 4 di antaranya didukung konteks
   hanya bisa mendarat di 0.0, 0.5, atau 1.0. Resolusinya terlalu kasar.

RAGAS memakai prompt dan algoritma yang sudah published, dan Faithfulness-nya bekerja per klaim.

**Yang ikut hilang dan tidak digantikan:** `answer_relevance` dan `refusal`. Kedua metric RAGAS yang
dipilih tidak mencakupnya. Angka keduanya di `EVAL_REPORT.md` §6 bersifat historis. Peringanannya,
`eval/main_testset.json` (n=115) sama sekali tidak memuat soal ber-`expected_route: "none"`, sehingga
cabang `refusal_correct` memang tidak akan pernah aktif pada test set final.

---

## 2. Faithfulness

**Definisi.** Proporsi klaim dalam jawaban yang didukung oleh konteks yang diambil. Mengukur
**halusinasi**: apakah sistem mengarang informasi di luar bukti yang dia punya.

| Aspek | Nilai |
|---|---|
| Input | `user_input` (pertanyaan), `response` (jawaban), `retrieved_contexts` (list chunk) |
| Ground truth | **TIDAK dipakai** |
| Evaluator | LLM — `gpt-4o-mini` (OpenAI), `temperature=0.0` |
| Prompt | Bawaan RAGAS: `StatementGeneratorPrompt` + `NLIStatementPrompt` |
| Rentang | 0–1, makin tinggi makin baik |
| Biaya | 2 panggilan LLM per sample |

Evaluator **sengaja vendor berbeda** dari generator produksi (`gemini-2.5-flash`, Google) — supaya
model yang menghasilkan jawaban tidak ikut menilai jawabannya sendiri. Alternatif tanpa API key baru
tersedia lewat `--judge-provider google` (masih Gemini, lihat §6), dengan trade-off: evaluator dan
generator kembali satu vendor.

**Cara skor diperoleh** (`ragas.metrics.collections.Faithfulness`):

1. **Statement extraction** — LLM memecah jawaban menjadi klaim-klaim atomik.
2. **NLI verification** — seluruh `retrieved_contexts` digabung (dipisah `\n`), lalu LLM memberi
   verdict didukung / tidak didukung untuk **tiap** klaim.
3. **Skor** = jumlah klaim didukung ÷ jumlah klaim total.

Kalau tidak ada klaim yang berhasil diekstrak, RAGAS mengembalikan NaN; runner mencatatnya sebagai
`nan_no_statements_extracted`, bukan sebagai 0.

**Interpretasi.**

| Skor | Arti |
|---|---|
| 1.0 | Semua klaim tertelusur ke konteks — tidak ada halusinasi |
| 0.5 | Separuh klaim tidak didukung konteks |
| 0.0 | Tidak satu pun klaim didukung konteks |

> **Penting untuk metodologi:** Faithfulness **tidak** mengukur kebenaran faktual. Jawaban yang setia
> mengutip konteks yang keliru tetap mendapat skor 1.0. Yang diukur adalah *konsistensi jawaban
> terhadap bukti yang diberikan kepadanya* — sehingga skor ini menilai komponen **generator**, bukan
> komponen retrieval. Kualitas retrieval diukur terpisah lewat hit@5 / recall@5 / MRR.

---

## 3. Semantic Similarity

**Definisi.** Kemiripan semantik antara jawaban yang dihasilkan dan jawaban acuan (ground truth),
diukur sebagai cosine similarity di ruang embedding.

| Aspek | Nilai |
|---|---|
| Input | `reference` (= `expected_answer`), `response` (jawaban) |
| Ground truth | **DIPAKAI** — `expected_answer` dari `eval/main_testset.json` |
| Embedding | `gemini-embedding-001` (SDK `google-genai`) |
| Rentang | 0–1, makin tinggi makin baik |
| Biaya | 0 panggilan LLM; 2 panggilan embedding per sample |

**Cara skor diperoleh** (`ragas.metrics.collections.SemanticSimilarity`): `reference` dan `response`
di-embed, lalu dihitung cosine similarity keduanya. Tidak ada LLM yang terlibat, sehingga metric ini
**sepenuhnya deterministik**.

**Interpretasi.** Nilai bersifat relatif, bukan absolut. Karena embedding kalimat Bahasa Indonesia
jarang mendekati 0 walau isinya berbeda, yang bermakna adalah **perbandingan antar arsitektur pada
soal yang sama**, bukan ambang mutlak. Sebagai kalibrasi dari uji plumbing:

| Kondisi | Skor teramati |
|---|---|
| Jawaban identik dengan reference | 1.000 |
| Jawaban benar, redaksi berbeda | ~0.95 |
| Jawaban menolak ("informasi tidak ditemukan") | ~0.57 |

> **Penting untuk metodologi:** metric ini menilai kemiripan **semantik**, bukan kebenaran. Jawaban
> yang mirip secara topik tapi salah angka bisa tetap mendapat skor tinggi. Karena itu ia dibaca
> berpasangan dengan Faithfulness, bukan sendirian.

---

## 4. Ringkasan perbedaan kedua metric

```
                    Faithfulness              Semantic Similarity
input               question                  —
                    generated answer          generated answer
                    retrieved contexts        expected answer
ground truth        TIDAK dipakai             DIPAKAI (expected_answer)
penilai             LLM (gpt-4o-mini)         embedding (gemini-embedding-001)
deterministik       tidak sepenuhnya          ya
menilai komponen    generator (halusinasi)    kedekatan ke jawaban acuan
```

---

## 5. Mapping data

| Field RAGAS | Sumber | Path |
|---|---|---|
| `user_input` | pertanyaan | `per_query_<sys>.json[].question` |
| `response` | jawaban yang dihasilkan | `per_query_<sys>.json[].answer` |
| `retrieved_contexts` | chunk yang diberikan ke generator | `per_query_<sys>.json[].retrieved_context` |
| `reference` | jawaban acuan | `main_testset.json.questions[].expected_answer` (join by `id`) |

`expected_facts` dan `gold_chunks` **tidak dipakai** oleh kedua metric ini. `gold_chunks` khususnya
tidak boleh menyentuh `retrieved_contexts` — itu ground-truth retrieval, bukan konteks aktual.

`retrieved_context` ditulis oleh `eval/run_eval.py` sebagai `[d.page_content for d in docs]`, yaitu
teks chunk yang benar-benar tersedia bagi generator pada arsitektur tersebut. Untuk Agentic, isinya
adalah gabungan hasil seluruh tool call (`RagResult.documents` yang sudah di-dedup), yakni seluruh
bukti yang dimiliki generator saat menyusun jawaban akhir.

---

## 6. Konfigurasi evaluator

Seluruhnya terpusat di [`eval/ragas_eval/config.py`](../eval/ragas_eval/config.py), dan ikut ditulis
ke setiap `summary_ragas_<system>.json` pada blok `config` supaya hasil membawa jejak konfigurasinya.

### Evaluator LLM

**Default: `gpt-4o-mini` (OpenAI)**, `temperature=0.0`, prompt sepenuhnya bawaan RAGAS. Butuh
`OPENAI_API_KEY` sendiri (lihat `.env.example`) — key dan billing yang **baru**, terpisah dari
`GEMINI_API_KEY` yang sudah ada di project.

**Kenapa vendor beda dari generator, bukan sekadar model beda.** Generator produksi memakai
`gemini-2.5-flash` (`src/ragtrial/llm.py: LLM_MODEL`). Kalau evaluator memakai model dari vendor yang
sama (apalagi model yang sama persis), muncul potensi *self-preference bias* — model cenderung
menilai keluaran "gaya"nya sendiri lebih baik. Memilih vendor lain sepenuhnya adalah bentuk mitigasi
yang lebih kuat daripada sekadar ganti model dalam satu vendor.

**Alternatif tanpa API key baru:** `--judge-provider google --judge-model <model-gemini-aktif>`
memakai Gemini lewat endpoint OpenAI-compatible-nya sendiri
(`https://generativelanguage.googleapis.com/v1beta/openai/`), masih dengan `GEMINI_API_KEY` yang
sudah ada. Trade-off-nya: evaluator kembali satu vendor dengan generator. Cek dulu model Gemini mana
yang masih aktif (nama model lama seperti `gemini-2.5-pro` bisa deprecated tanpa peringatan —
sudah pernah ditemukan langsung waktu implementasi).

**Catatan teknis untuk jalur `google`.** Wiring-nya juga lewat client `AsyncOpenAI` yang diarahkan ke
base URL Gemini — bukan berarti memakai model OpenAI, murni jalur teknis yang diverifikasi empiris
pada ragas 0.4.3:

| Jalur | Hasil |
|---|---|
| `llm_factory(provider="google", client=genai.Client(...))` | gagal — adapter `instructor`; ragas sendiri memperingatkan bug upstream instructor+google-genai (`HARM_CATEGORY_JAILBREAK`, instructor#1658) dan menyarankan endpoint OpenAI-compatible |
| `genai.Client(...).aio` | ditolak — adapter mewajibkan `google.genai.Client`, bukan `AsyncClient` |
| `Faithfulness.score()` (sinkron) | gagal — hanya membungkus `ascore()`, tetap butuh client async |
| **`AsyncOpenAI` + base_url Gemini** | **berhasil** |

Jalur default (`provider="openai"` + `AsyncOpenAI` asli ke API OpenAI) tidak butuh workaround ini
sama sekali — instructor adapter OpenAI bekerja langsung.

### Embedding evaluator

`gemini-embedding-001` lewat SDK `google-genai`.

**Sengaja berbeda** dari embedding retrieval (`models/gemini-embedding-2`, `task_type=retrieval_query`).
Kalau model yang sama dipakai untuk *memilih* bukti sekaligus *menilai* jawaban, muncul confound yang
sulit dibela. Embedding produksi/retrieval **tidak diubah sama sekali**.

### Environment

RAGAS dipasang di virtualenv **terpisah** (`.venv-ragas`), bukan di `.venv` utama, karena ia menarik
`langchain-core` 1.5.5 + `langchain-openai` + `datasets` + `instructor` yang akan mengubah dependency
tree pipeline RAG produksi. Runner evaluasi tidak meng-import `ragtrial` sama sekali, jadi pemisahan
ini tidak berbiaya apa pun.

> **Pin wajib:** `langchain-community==0.4.1`. ragas 0.4.3 melakukan import top-level
> `from langchain_community.chat_models.vertexai import ChatVertexAI`, dan modul itu **dihapus** di
> langchain-community 0.4.2 — sehingga `import ragas` langsung gagal. Jangan naikkan versinya.

---

## 7. Alur eksekusi

```
eval/main_testset.json (115 soal)
        │
        ├──────────────── expected_answer ─────────────────────┐
        ↓                                                      │
  eval/run_eval.py  (.venv utama — pipeline produksi)          │
        ↓                                                      │
  eval/results/main/per_query_{naive,enhanced,agentic}.json    │
        │   answer + retrieved_context                         │
        ↓                                                      │
  eval/run_ragas.py  (.venv-ragas — evaluator)  ◄──────────────┘
        ├── Faithfulness         (LLM)
        └── Semantic Similarity  (embedding)
        ↓
  eval/results/ragas/
      per_query_ragas_<system>.json
      summary_ragas_<system>.json
      SUMMARY_ragas.txt
```

Fase evaluasi murni JSON-in/JSON-out. Karena `eval/run_ragas.py` tidak meng-import `ragtrial`, secara
**struktural mustahil** ia memicu retrieval atau generation ulang — jaminan ini lebih kuat daripada
sekadar konvensi.

### Perintah

```powershell
# 1. generate (venv utama)
.\.venv\Scripts\python.exe -m eval.run_eval `
    --testset eval/main_testset.json --outdir eval/results/main `
    --systems naive enhanced agentic --sleep 1 --resume

# 2. estimasi biaya dulu — NOL panggilan API
.\.venv-ragas\Scripts\python.exe eval\run_ragas.py --dry-run

# 3. evaluasi RAGAS
.\.venv-ragas\Scripts\python.exe eval\run_ragas.py --sleep 1 --resume
```

### Biaya untuk test set penuh

115 soal × 3 arsitektur = 345 sample.

| | Jumlah |
|---|---|
| Panggilan LLM (Faithfulness, 2/sample) | **690** |
| Panggilan embedding (Semantic Similarity, 2/sample) | **690** |

`--dry-run` mencetak angka ini beserta coverage sebelum satu pun API call terjadi.

---

## 8. Keterbatasan dan sumber bias

1. **Evaluator ≠ generator (by design), tapi tidak sepenuhnya bebas bias.** Faithfulness dinilai
   `gpt-4o-mini` (OpenAI), sementara generator produksi memakai `gemini-2.5-flash` (Google) — vendor
   berbeda, dipilih khusus untuk menekan *self-preference bias*. Tugas yang diberikan ke evaluator
   juga bersifat **ekstraktif dan terstruktur** (pecah klaim, lalu cek entailment terhadap konteks
   yang diberikan), bukan penilaian preferensi bebas — ruang untuk bias jauh lebih sempit daripada
   judge holistik. Yang tersisa: kedua model tetap sama-sama LLM modern dan bisa berbagi bias sistemik
   yang tidak spesifik ke satu vendor (mis. kecenderungan menilai jawaban yang panjang/terstruktur
   lebih baik). Perlu dua API key aktif (`GEMINI_API_KEY` + `OPENAI_API_KEY`) — kalau butuh
   menghindari biaya/setup OpenAI, `--judge-provider google` tersedia dengan trade-off vendor sama
   lagi (lihat §6).
2. **Determinisme.** `temperature=0` menekan variasi tapi tidak menjaminnya nol; LLM tetap dapat
   menghasilkan pemecahan klaim yang sedikit berbeda antar run. Semantic Similarity sepenuhnya
   deterministik. Cache skor (`ragas_cache.json`) membuat run ulang atas jawaban yang sama
   mengembalikan angka identik tanpa memanggil API.
3. **Semantic Similarity bukan ukuran kebenaran.** Ia tidak menghukum kesalahan angka/nama selama
   kalimatnya mirip secara semantik. Karena itu dibaca berpasangan dengan Faithfulness.
4. **Faithfulness bukan ukuran kebenaran.** Jawaban yang setia pada konteks yang salah tetap
   bernilai 1.0. Ia menilai generator, bukan retrieval.
5. **Tidak ada metric relevansi jawaban.** Jawaban yang setia pada konteks tapi tidak menjawab
   pertanyaan tidak dihukum oleh kedua metric ini.
6. **Bahasa.** Prompt bawaan RAGAS berbahasa Inggris, sedangkan korpus, pertanyaan, dan jawaban
   berbahasa Indonesia. `gpt-4o-mini` multilingual, tapi kondisi lintas-bahasa ini perlu disebutkan
   sebagai keterbatasan.

---

## 9. Penanganan error dan reproducibility

- **Retry.** Exponential backoff pada error transient (429 / RESOURCE_EXHAUSTED / 503 / timeout),
  maksimal 5 percobaan, jeda awal 30 detik, faktor 1.5 — menyalin pola `_with_retry` di
  `src/ragtrial/llm.py`.
- **Kegagalan tidak menjatuhkan run.** Setiap sample dicatat dengan `status` ∈ `ok` / `partial` /
  `failed` beserta `errors` per metric. Kegagalan satu metric tidak membatalkan metric lain: sample
  tanpa konteks tetap mendapat Semantic Similarity, hanya Faithfulness-nya yang kosong dengan alasan
  `empty_retrieved_context`.
- **Tidak ada skip diam-diam.** Setiap alasan kegagalan tercatat per record dan direkap di
  `error_reasons` pada summary.
- **Guard artefak keliru.** Kalau ada `id` di `per_query_*.json` yang tidak dikenal test set, run
  **dibatalkan** dengan pesan jelas. Ini yang mencegah artefak lama (mis. 202-record candidate set)
  ikut terevaluasi tanpa disadari.
- **Coverage dilaporkan.** Soal yang belum punya hasil generation dihitung sebagai
  `n_missing_generation` + `missing_ids`, dipisahkan dari kegagalan evaluasi.
- **Cache.** Kunci = `id|system|metric|sha1(answer)|sha1(context|reference)|model`. Berbeda dari
  cache judge lama yang hanya mem-hash jawaban, di sini konteks **dan** identitas model ikut masuk
  kunci, sehingga mengganti evaluator otomatis meng-invalidate cache. Error transien tidak di-cache,
  supaya `--resume` bisa mencobanya lagi.
- **Checkpoint atomik** (temp+rename) tiap 15 sample yang memanggil API.

---

## 10. Skema output

`eval/results/ragas/per_query_ragas_<system>.json` — array, satu objek per soal:

```json
{
  "question_id": "OP001",
  "architecture": "naive",
  "question": "Berapa nomor telepon Inspektorat Kabupaten Batang?",
  "expected_answer": "Nomor telepon Inspektorat Kabupaten Batang adalah (0285) 391980.",
  "generated_answer": "...",
  "retrieved_contexts": ["chunk 1 ...", "chunk 2 ..."],
  "n_contexts": 2,
  "faithfulness": 1.0,
  "semantic_similarity": 0.946,
  "status": "ok",
  "errors": {}
}
```

Setiap baris dapat ditelusuri penuh ke question / expected_answer / generated_answer /
retrieved_contexts, sehingga failure case bisa dibaca langsung tanpa menggabungkan file lain.

`summary_ragas_<system>.json` — per metric: `mean`, `median`, `std`, `min`, `max`, `n_evaluated`,
`n_failed`; plus `status_counts`, `n_missing_generation`, `missing_ids`, `error_reasons`, dan blok
`config` berisi versi RAGAS, model evaluator, embedding, temperature, path input, dan timestamp.

`SUMMARY_ragas.txt` — tabel perbandingan ketiga arsitektur.

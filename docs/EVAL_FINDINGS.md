# Temuan Evaluasi — Klasifikasi, Kemungkinan Alasan, Kesimpulan

_Pendamping [`EVAL_REPORT.md`](EVAL_REPORT.md) (angka mentah). Dokumen ini = lapisan **interpretasi**: temuan dikelompokkan Expected / Menarik / Anomali, masing-masing dengan **kemungkinan penyebab** (ditandai **[hipotesis]** bila belum diverifikasi, **[terukur]** bila langsung dari data) dan **cara uji lanjut**. n=202 (retrieval/latency/generation full), intent n=40, ablation n=30 subset._

> **Aturan main tesis:** setiap "kenapa" di bawah adalah **kandidat penjelasan**, bukan fakta final. Yang **[terukur]** aman dinyatakan; yang **[hipotesis]** harus ditulis sebagai dugaan + rencana verifikasi, jangan di-over-claim.

---

## A. Temuan EXPECTED (sesuai teori/hipotesis desain)

Ini yang **memang diharapkan** — berguna sebagai validasi bahwa sistem & eval berperilaku benar.

| # | Temuan | Bukti | Kemungkinan alasan |
|---|---|---|---|
| E1 | **Enhanced > Naive di answer-quality** | faithfulness 0.870→0.927; relevance 0.832→0.912 | **[hipotesis]** Reranker menyaring chunk "mirip tapi salah" → konteks yang masuk ke LLM lebih bersih → jawaban lebih ter-*grounding*. *(Uji: korelasikan rerank_score vs faithfulness per soal.)* |
| E2 | **Intent gate memisahkan abstention** | recall_invalid: naive 0.00, enhanced/agentic 1.00 | **[terukur]** By design: naive tak punya gate → selalu menjawab; enhanced pakai semantic gate, agentic pakai keputusan tool-call implisit. |
| E3 | **Domain atomik ~sempurna, sosial tersulit** | opd 1.00, perizinan 0.94; sosial 0.40–0.58 | **[hipotesis]** Atomik = 1 record/chunk, self-contained, embedding distinct → mudah. Sosial = pasal chunk-precise + near-duplicate antar-perda → embedding tumpang-tindih, gold benar kalah ranking. |
| E4 | **Recall anjlok di soal hard** | easy ~0.84 → hard 0.48–0.55 | **[terukur]** 18 soal hard sengaja multi-hop/skenario/analitis → butuh >1 chunk atau reasoning; top-5 sering tak cukup. |
| E5 | **Agentic ~2× LLM call** | n_llm_calls 1.97 (vs 1.00) | **[terukur]** Loop agent: ≥1 turn orkestrasi (pilih tool) + 1 turn sintesis jawaban. |
| E6 | **Naive tercepat** | latency 4.64s (vs 7.36 / 6.61) | **[terukur]** Tanpa rerank, intent gate, atau loop — cuma retrieve + 1 generate. |

**Ringkas:** sistem berperilaku sesuai desain di sumbu-sumbu dasar. Ini fondasi yang bikin temuan lain kredibel.

---

## B. Temuan MENARIK (nuansa non-trivial, masuk akal tapi tak sepele)

Ini bahan **pembahasan** paling kaya — trade-off yang bersih.

### M1 — Non-monotonisitas: agentic recall TERENDAH (0.695) [terukur, pra-registrasi]
Agentic (0.695) < naive (0.781) < enhanced (0.792) di recall.
**Kenapa:** routing = filter 1 domain; **kalau salah domain → recall collapse** (bukan degradasi bertahap). Paling parah lintas-domain: `both` recall **0.38** (agent pilih 1 sisi, miss sisi lain — confusion matrix §5). Routing itu **high-reward, high-risk**: akurasi 0.881 bagus, tapi 12% error-nya fatal.
**Implikasi:** *inference-time control* tak otomatis menang; pada korpus yang mayoritas single-hop, *author-time control* (fan-out/global) lebih aman untuk recall.

### M2 — Latency terbalik per-domain: enhanced > agentic HANYA di 3/6 domain [terukur]
Enhanced lebih lambat di sosial/none/opd; agentic malah lebih lambat di perizinan/dukcapil/both.
**Kenapa [terukur + hipotesis]:** latency ≈ **ukuran konteks × jumlah call**.
- Sosial: konteks global gemuk (chunk hukum panjang) → generate enhanced 6.73s vs agentic 3.26s (routing → konteks kecil). Enhanced kalah.
- Atomik: konteks kecil → keunggulan enhanced hilang, tersisa **overhead 2-call** agentic → agentic kalah.
**Implikasi:** klaim "enhanced paling lambat" itu **agregat yang didominasi sosial**, bukan hukum universal.

### M3 — Rerank warm ~1.4s, bukan 12s [terukur]
12s hanya **cold-start** (load model 2GB, sekali per proses). Warm bge-base ~1.4s.
**Implikasi:** jangan laporkan 12s sebagai biaya rerank; itu one-time. Biaya efektif rerank ~1.4s/soal.

### M4 — Tail latency agentic TERBAIK (p95 11.16 < enhanced 15.46) [terukur]
**Kenapa [hipotesis]:** worst-case enhanced = generate gemuk pada soal sosial berkonteks besar (p95 generate 12.24s). Agentic memfilter → konteks selalu relatif kecil → tail lebih pendek. Agentic lebih *predictable*.

### M5 — Agentic over-refuse sosial (routing none precision 0.47) [terukur]
Agent 6× salah route sosial→none (nolak padahal ada jawaban).
**Kenapa [hipotesis]:** saat retrieval sosial lemah (rerank score rendah), sinyal self-correction + prompt "jawab jujur kalau tak yakin" bikin agent **menyerah** ketimbang memaksa. Ini menyumbang recall sosial 0.40. Trade-off jujur-vs-cakupan.

---

## C. Temuan ANOMALI (berlawanan ekspektasi — WAJIB dibahas jujur)

Ini yang **menohok narasi** dan paling penting tidak di-over-claim.

### A1 — Enhanced nyaris tak ungguli naive di recall; IDENTIK di sosial [terukur]
Recall +0.011 agregat (dalam noise, ~2 soal); sosial **0.58 = 0.58**.
**Kenapa [terukur + hipotesis]:** (a) hybrid menyeret recall (lihat A2); (b) **[hipotesis]** untuk sosial, gold entah tertangkap dense atau tidak — rerank hanya **mengurutkan ulang** kandidat, tak **memperluas cakupan** top-5; kalau gold tak ada di pool kandidat, rerank tak bisa memunculkannya.
**Implikasi:** **nilai enhanced = kualitas + abstention, BUKAN recall.** Naive dense global sudah ~98% jalan di recall. *Jangan tulis "enhanced unggul retrieval".*

### A2 — Hybrid (BM25) MENURUNKAN recall; reranker yang jadi value-add [terukur, ablation]
dense_only 0.750 → +hybrid 0.728 (turun); rerank_no_hybrid **0.850** → +hybrid 0.761 (turun). Rerank sendiri = lompatan terbesar (0.750→0.850).
**Kenapa [hipotesis, TESTABLE]:** korpus hukum Indonesia penuh **pasal near-duplicate berbunyi mirip**. BM25 (leksikal) menarik banyak kandidat yang **cocok kata tapi salah pasal/perda**, mendorong gold keluar top-k sebelum rerank; RRF mencampur noise ini ke ranking akhir. Dense + rerank lebih presisi tanpa banjir leksikal.
**Cara uji:** inspeksi manual kandidat BM25 vs dense pada 10 soal sosial — hitung berapa "mirip tapi salah" yang disuntik BM25. **Rekomendasi:** kandidat default Enhanced = **dense + rerank (buang hybrid)**; konfirmasi di full 202 dulu (n=30 kecil).

### A3 — Enhanced LEBIH BURUK dari naive di soal easy (0.84 vs 0.88) [terukur]
**Kenapa [hipotesis]:** pada soal mudah/pendek, dense sudah menaruh chunk benar di peringkat atas; **cross-encoder kadang salah menilai dan mendemosikannya** (rerank = overhead tanpa gain, kadang malah rugi). Enhanced menang justru di *medium* (0.80 vs 0.74).
**Cara uji:** ambil soal easy di mana enhanced miss tapi naive hit → cek apakah gold ada di pool tapi didemosi rerank.

### A4 — Agentic faithful (0.874) padahal recall terburuk (0.695) [terukur]
Setia-konteks tinggi + retrieval jelek = **"faithful ke konteks yang SALAH"**.
**Kenapa [terukur]:** faithfulness mengukur *grounding ke chunk yang di-retrieve*, bukan ke gold. Agent salah-route → jawaban tetap konsisten dengan chunk keliru → skor faithful tinggi meski jawabannya bisa salah.
**Implikasi KRITIS:** **faithfulness ≠ correctness.** Jangan pakai faithfulness sebagai proxy kebenaran; **correctness wajib ke evaluasi expert**.

### A5 — Agentic AMBRUK di analytical (recall 0.64, relevance 0.68 vs 0.95/0.95) [terukur]
Tier "paling pintar reasoning" justru **terlemah** di soal analitis.
**Kenapa [hipotesis]:** analitis butuh **sintesis lintas-konten**; routing + fokus 1-domain malah **menyempitkan** konteks yang justru dibutuhkan. Enhanced (global + rerank top-k lebih luas) memberi LLM bahan lebih banyak untuk menalar.
**Cara uji:** cek per-soal analytical agentic — apakah routing memfilter domain yang seharusnya multi, atau top-k terlalu sempit.

### A6 — Naive recall kompetitif (0.78) [terukur]
Floor yang "seharusnya lemah" ternyata dekat enhanced di retrieval.
**Kenapa [hipotesis]:** fan-out/global dense menangkap gold **tanpa risiko routing**, dan korpus cukup "dense-friendly" (embedding multilingual menangani sebagian besar query). Bedanya telak hanya di abstention (recall_invalid 0.00 vs 1.00) & precision.

---

## D. KESIMPULAN

1. **Tidak ada pemenang tunggal — bergantung sumbu evaluasi.**
   - **Enhanced** = kualitas jawaban (faithfulness 0.93) + abstention terbaik (false_refusal 0.13). Payoff-nya di **kualitas, bukan recall**.
   - **Naive** = recall murah & cepat (0.78, 4.6s), tapi **tak bisa menolak** (bahaya untuk layanan publik).
   - **Agentic** = adaptif + refuse OOS sempurna + tail latency terbaik, tapi **recall terendah** (routing loss) dan **terlemah di analytical**.

2. **Retrieval-stack: reranker terbukti, hybrid tidak.** Rerank = penyumbang utama recall (0.750→0.850); **BM25/hybrid justru menurunkan recall** di korpus hukum ini. → **Rekomendasi: Enhanced default = dense + rerank** (verifikasi full 202).

3. **Non-monotonisitas itu temuan sah, bukan kegagalan.** Pada korpus mayoritas single-hop, memindahkan kontrol ke LLM (agentic) **menukar recall demi** efisiensi-per-call, adaptivitas, & abstention — tidak otomatis lebih baik. Ini kontras arsitektural yang bersih untuk dibahas.

4. **Faithfulness bukan kebenaran.** Agentic bisa "faithful" ke chunk yang salah. **Evaluasi expert (blind, terpisah) wajib** untuk mengukur correctness — automated eval hanya mengukur grounding + relevansi + retrieval.

5. **Latency = fungsi (ukuran konteks × jumlah call), bukan sekadar jumlah call.** Enhanced membayar di generation-gemuk (konteks global sosial); agentic membayar di jumlah call. Tiap arsitektur menang di rezim domain berbeda.

6. **Jangan over-claim enhanced.** Kemenangan enhanced di retrieval recall = **dalam noise**; yang solid adalah kualitas & abstention. Klaim yang bisa dipertahankan: *"retrieval-stack (khususnya reranker) meningkatkan kualitas & precision jawaban dan kemampuan menolak, dengan biaya latency, tanpa banyak menggeser recall pada korpus yang sudah dense-friendly."*

---

## E. Tindak lanjut yang dimotivasi temuan (opsional)

- **[P1]** Konfirmasi anomali hybrid (A2) di **full 202** — kalau tahan, ubah default Enhanced ke dense+rerank.
- **[P1]** **Evaluasi expert** untuk correctness (menutup A4 — faithfulness≠kebenaran).
- **[P2]** Inspeksi manual kandidat BM25 sosial (uji hipotesis A2) & soal easy yang didemosi rerank (uji A3).
- **[P2]** Agentic-fanout ablation: jalankan agentic tanpa routing (selalu global) → isolasi "kontrol LLM" dari "routing" (jelaskan sumber recall loss).
- **[P3]** RAGAS faithfulness statement-level di subset sebagai cross-check judge holistik.

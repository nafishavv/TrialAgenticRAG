"""Intent gate — decide VALID (needs retrieval) vs INVALID (answer directly).

This is the Enhanced tier's *user intent handling* (paper: "Is Agentic RAG Worth
It?") and the ONLY decision Enhanced makes before retrieving. It runs FIRST in
the pipeline and answers exactly one question: retrieve, or answer directly?

There is deliberately NO domain routing in Enhanced — a VALID query always
retrieves globally over the whole unified index (same scope as naive; Enhanced
differs by hybrid search + cross-encoder rerank). Domain selection exists only in
the agentic tier, where the LLM picks a `search_<domain>` tool.

Implementation uses the `semantic-router` library (embedding-based classifier) so
the decision is a cheap vector comparison, not an LLM call. The library's encoder
is adapted to reuse our Gemini `embeddings` singleton (same model as retrieval).

NOTE: the library's main class is named `SemanticRouter` — aliased here as
`_SRRouter`. Despite the name it is a 2-class intent classifier here, not a
domain router.

Binary by design: INVALID covers both chit-chat/identity and out-of-scope. One
smart generation prompt (PROMPT_INVALID) tells those apart at answer time, so the
gate stays a simple 2-class problem (easy to evaluate F1/recall on).
"""

from __future__ import annotations

from typing import List, Optional

from semantic_router import Route
from semantic_router.encoders import DenseEncoder
from semantic_router.routers import SemanticRouter as _SRRouter

from ragtrial.llm import embeddings as default_embeddings
from ragtrial.pipeline.base import RagState, Stage

# ============ Example sets (the router is DEFINED by these) ============
# Same 20 VALID + 20 INVALID utterances as intent_reference_set.json — this IS
# now the production router index, not just the offline threshold-tuning
# reference. Keeping the two in sync means THRESHOLD (validate_intent_threshold.py,
# TOP_K=5 fixed) is valid as measured, with no train/serve skew between the index
# it was tuned against and the index actually queried at runtime.
# VALID = service questions needing retrieval, one domain per OPD/Dukcapil/
# Perizinan/Sosial (5 each, matching REF_VALID_OPD/DUK/PER/SOS_00X).
VALID_EXAMPLES: List[str] = [
    # OPD
    "Saya mau menanyakan soal bantuan sosial, kantor Dinas Sosial Batang ada di jalan apa?",
    "Dinas Perhubungan Kabupaten Batang kantornya di mana ya?",
    "Ada pembuangan limbah yang mencemari sungai dekat rumah saya. Instansi mana yang menangani dan bagaimana cara menghubunginya?",
    "Kecamatan Limpung punya alamat email resmi tidak? Saya mau konsultasi berkas dulu sebelum datang.",
    "Saya ada keperluan soal arsip daerah. Saya harus datang ke kantor mana dan di mana alamatnya?",
    # Dukcapil
    "Anak saya baru berusia 17 tahun. Apa saja syarat untuk membuat KTP pertama kalinya?",
    "Desa kami baru dimekarkan sehingga nama wilayahnya berubah. Apakah KK dan KTP kami harus diperbarui?",
    "Saya akan menetap di luar negeri. Bagaimana cara melaporkan kepindahan saya ke Disdukcapil?",
    "Akta kelahiran saya masih format lama tanpa QR code. Apakah bisa diganti dengan format yang baru?",
    "Apakah pengurusan akta kelahiran bisa dilakukan secara daring, dan apakah tetap perlu tanda tangan pelapor?",
    # Perizinan
    "Apakah pengurusan Izin Reklame di Batang dikenakan biaya?",
    "Saya mau memasang papan reklame. Apakah ada ketentuan minimal ketinggian tiangnya?",
    "Izin Kerja Perekam Medis harus diurus datang ke kantor atau bisa diajukan lewat website?",
    "Berapa lama proses Izin Praktik Psikolog Klinis dan apakah ada biayanya?",
    "Izin Penelitian Non Akademis saya hampir habis. Dokumen apa yang perlu dilampirkan untuk perpanjangannya?",
    # Sosial & Hukum
    "Ayah saya meninggal dan keluarga kami tergolong tidak mampu. Apa syarat mengajukan santunan kematian ke Pemkab Batang?",
    "Saya tidak sanggup membayar pengacara untuk perkara saya. Apakah Pemkab Batang menyediakan bantuan hukum gratis dan apa saja berkasnya?",
    "Untuk memakamkan keluarga saya di TPU, apakah perlu izin penggunaan tanah makam dan siapa yang mengajukannya?",
    "Kalau pengelola gedung membiarkan orang merokok di kawasan tanpa rokok, sanksi apa yang bisa dikenakan?",
    "Siapa yang menetapkan warga sebagai peserta jaminan kesehatan PBI APBD di Kabupaten Batang?",
]

# INVALID = no retrieval needed: chit-chat/identity + out-of-scope (matching
# REF_INVALID_00X categories: 6 completely_unrelated, 6 general_knowledge,
# 6 casual_conversational, 1 non_batang_public_service, 1 out_of_scope_local).
INVALID_EXAMPLES: List[str] = [
    # completely unrelated
    "Bisa bantu buatkan caption Instagram untuk foto liburan?",
    "Olahraga apa yang bagus untuk menurunkan berat badan dalam sebulan?",
    "Rekomendasi film yang cocok ditonton bareng keluarga apa ya?",
    "Kenapa kucing saya tiba-tiba nggak mau makan selama dua hari?",
    "Bagaimana cara merawat tanaman hias agar tidak mudah layu?",
    "Apa cara paling mudah mengatur file dan folder di laptop?",
    # general knowledge
    "Siapa penemu lampu pijar?",
    "Apa ibu kota negara Australia?",
    "Jelaskan proses terjadinya hujan secara singkat.",
    "Siapa penulis novel Bumi Manusia?",
    "Mengapa air laut terasa asin?",
    "Apa yang menyebabkan terjadinya pergantian musim?",
    # casual / identity
    "Kamu ini robot atau manusia sih sebenarnya?",
    "Kalau kamu bisa liburan ke mana saja, kamu pilih ke mana?",
    "Kamu suka makanan pedas nggak?",
    "Terima kasih ya, kamu membantu banget hari ini.",
    "Kalau sedang libur, biasanya kamu lebih suka melakukan apa?",
    "Menurutmu, makanan apa yang paling cocok untuk sarapan?",
    # non-Batang public service / out-of-scope local info
    "Berapa biaya pembuatan IMB di Kota Semarang?",
    "Kapan jadwal pemadaman listrik bergilir di wilayah Kabupaten Batang?",
]

VALID = "valid"
INVALID = "invalid"
# Optimal threshold from intent_validation_set.json tuning (Macro-F1=0.9374).
# Found via validate_intent_threshold.py on 80-query validation set (40 VALID + 40 INVALID).
THRESHOLD = 0.63
TOP_K = 5


# ============ Encoder adapter ============
class GeminiEncoder(DenseEncoder):
    """Adapt our langchain Gemini `embeddings` singleton to semantic-router.

    semantic-router calls `encoder(docs) -> list[list[float]]`. We delegate to
    `embeddings.embed_documents`, so the gate uses the exact same embedding model
    as the RAG retrieval pipeline (controlled variable across tiers).
    """

    name: str = "gemini-intent"

    def __init__(self, embedder=None, **kwargs):
        super().__init__(name=kwargs.pop("name", "gemini-intent"), **kwargs)
        # store outside pydantic fields to avoid schema friction
        object.__setattr__(self, "_embedder", embedder or default_embeddings)

    def __call__(self, docs: List[str]) -> List[List[float]]:
        return self._embedder.embed_documents(list(docs))


# ============ Intent gate stage ============
class IntentStage(Stage):
    """Classify the ORIGINAL question as VALID/INVALID; short-circuit if INVALID.

    Downstream stages (rewrite/retrieve) check `state.intent` and no-op when it is
    'invalid'; generate then uses PROMPT_INVALID for a direct answer. No-match from
    the router (winning class's mean similarity below THRESHOLD) falls back to
    'valid' — fail-OPEN, not the fail-closed rule tuned in validate_intent_threshold.py
    (that script's QueryScores.predict rejects below threshold regardless of winning
    class). Deliberate deviation: the retrieval ablation (eval/results/ablation/
    results.json) measured a 20% false-reject rate on undeniably in-scope
    main_testset.json questions (5/25: OP024, SO002, SO003, SO007, CR010) — an
    unretrieved query scores a guaranteed 0 on every retrieval metric, which was
    making Enhanced look worse than Naive for reasons having nothing to do with
    retrieval quality. Failing open trades this for a higher false-accept rate on
    genuinely out-of-scope queries (unmeasured after this change — re-check against
    intent_validation_set.json's INVALID half before relying on refusal-rate numbers).
    """

    name = "intent"

    def __init__(self, encoder=None, fallback: str = VALID):
        self._encoder = encoder or GeminiEncoder()
        self._fallback = fallback
        self._router: Optional[_SRRouter] = None  # lazy: avoid embedding at import

    def _ensure_router(self) -> _SRRouter:
        if self._router is None:
            # score_threshold set explicitly on each Route (not on the encoder):
            # visible at the construction site, doesn't depend on the library's
            # internal encoder -> router -> route propagation order
            # (routers/base.py: _set_score_threshold + the route-inheritance loop).
            routes = [
                Route(name=VALID, utterances=VALID_EXAMPLES, score_threshold=THRESHOLD),
                Route(name=INVALID, utterances=INVALID_EXAMPLES, score_threshold=THRESHOLD),
            ]
            self._router = _SRRouter(
                encoder=self._encoder, routes=routes, auto_sync="local", top_k=TOP_K
            )
        return self._router

    def run(self, state: RagState) -> RagState:
        router = self._ensure_router()
        choice = router(text=state.question)
        name = getattr(choice, "name", None)
        score = getattr(choice, "similarity_score", None)

        intent = name if name in (VALID, INVALID) else self._fallback
        state.intent = intent
        state.meta["intent"] = intent
        state.meta["intent_route"] = name
        if score is not None:
            state.meta["intent_score"] = round(float(score), 4)
        return state


INTENT_GATES: dict[str, Optional[type[Stage]]] = {
    "none": None,
    "semantic": IntentStage,
}

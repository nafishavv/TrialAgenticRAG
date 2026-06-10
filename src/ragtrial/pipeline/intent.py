"""Intent gate — decide VALID (needs retrieval) vs INVALID (answer directly).

This is the Enhanced tier's *user intent handling* (paper: "Is Agentic RAG Worth
It?"). It runs FIRST in the pipeline, BEFORE the domain router, and is orthogonal
to it: the domain router picks dukcapil/opd/both for VALID queries; this gate
decides whether to retrieve at all.

Implementation uses the `semantic-router` library (embedding-based classifier) so
the decision is a cheap vector comparison, not an LLM call. The library's encoder
is adapted to reuse our Gemini `embeddings` singleton (same model as retrieval).

NOTE: the library's main class is also named `SemanticRouter` — aliased here as
`_SRRouter` to avoid confusion with our in-house domain SemanticRouter (route.py).

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
# VALID = service questions needing retrieval, spanning the 5 service categories.
VALID_EXAMPLES: List[str] = [
    # Perizinan
    "syarat izin mendirikan bangunan apa saja?",
    "berapa biaya retribusi IMB?",
    "cara mengurus izin usaha mikro?",
    "bagaimana prosedur perpanjangan izin usaha?",
    # Kependudukan
    "cara daftar KTP elektronik?",
    "syarat membuat kartu keluarga baru?",
    "bagaimana mengurus surat pindah domisili?",
    "dokumen untuk akta kelahiran apa saja?",
    # Pajak Daerah
    "bagaimana cara bayar PBB?",
    "berapa tarif retribusi pasar?",
    # Hukum & Peraturan
    "apa isi perda tentang retribusi daerah?",
    "peraturan bupati soal pelayanan administrasi?",
    # Sosial & Kesejahteraan
    "bagaimana penyelenggaraan perlindungan anak?",
    "apa aturan kawasan tanpa rokok?",
    "bagaimana prosedur pengumpulan sumbangan sosial?",
    # Informasi & Komunikasi / OPD
    "alamat kantor Disdukcapil Batang di mana?",
    "nomor telepon dinas kesehatan Kabupaten Batang?",
    "bagaimana cara menyampaikan pengaduan layanan?",
]

# INVALID = no retrieval needed: chit-chat/identity + out-of-scope.
INVALID_EXAMPLES: List[str] = [
    # chit-chat / identity / metadata
    "halo",
    "selamat pagi",
    "siapa kamu?",
    "kamu bisa bantu apa saja?",
    "apa yang bisa dibantu?",
    "terima kasih ya",
    "oke sip",
    # out-of-scope
    "berapa harga beras hari ini?",
    "siapa presiden Indonesia?",
    "resep nasi goreng spesial?",
    "bagaimana cara belajar coding?",
    "jadwal pertandingan bola malam ini?",
    "cuaca besok hujan tidak?",
    "rekomendasi film bagus dong",
]

VALID = "valid"
INVALID = "invalid"


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
    the router falls back to 'valid' (fail-open to retrieval — safer for a public
    service bot than wrongly refusing a real service question).
    """

    name = "intent"

    def __init__(self, encoder=None, fallback: str = VALID):
        self._encoder = encoder or GeminiEncoder()
        self._fallback = fallback
        self._router: Optional[_SRRouter] = None  # lazy: avoid embedding at import

    def _ensure_router(self) -> _SRRouter:
        if self._router is None:
            routes = [
                Route(name=VALID, utterances=VALID_EXAMPLES),
                Route(name=INVALID, utterances=INVALID_EXAMPLES),
            ]
            self._router = _SRRouter(
                encoder=self._encoder, routes=routes, auto_sync="local"
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

"""Static UI copy served by GET /api/meta (Bahasa Indonesia, government tone).

Deliberately static — no imports from the heavy pipeline stack, so the frontend
can bootstrap instantly even while models are still warming up. The `domain`
keys mirror ragtrial.sources.SOURCES; they are navigation aids only — the
backend decides routing, users never pick a domain.
"""

from __future__ import annotations

DOMAINS = [
    {
        "domain": "dukcapil",
        "title": "Kependudukan & Pencatatan Sipil",
        "description": (
            "KTP elektronik, Kartu Keluarga, akta kelahiran & kematian, "
            "pindah domisili, dan layanan adminduk lainnya."
        ),
        "example": "Apa syarat membuat KTP elektronik?",
    },
    {
        "domain": "perizinan",
        "title": "Perizinan",
        "description": (
            "Syarat, prosedur, biaya, dan estimasi waktu pengurusan izin "
            "usaha serta perizinan lainnya (SIPUAS)."
        ),
        "example": "Bagaimana cara mengurus izin reklame?",
    },
    {
        "domain": "sosial",
        "title": "Bantuan & Kesejahteraan Sosial",
        "description": (
            "Peraturan daerah seputar bantuan sosial, perlindungan anak dan "
            "perempuan, jaminan sosial, dan kesejahteraan warga."
        ),
        "example": "Apa dasar hukum bantuan sosial bagi warga miskin?",
    },
    {
        "domain": "opd",
        "title": "Perangkat Daerah (OPD)",
        "description": (
            "Alamat kantor, nomor telepon, dan email dinas, badan, kecamatan, "
            "serta kelurahan di Kabupaten Batang."
        ),
        "example": "Alamat Dinas Kependudukan dan Pencatatan Sipil?",
    },
]

POPULAR_QUESTIONS = [
    "Apa syarat membuat KTP elektronik?",
    "Bagaimana cara mengurus izin reklame?",
    "Alamat Dinas Kependudukan dan Pencatatan Sipil?",
    "Apa dasar hukum bantuan sosial bagi warga miskin?",
    "Bagaimana cara mengurus akta kelahiran?",
]

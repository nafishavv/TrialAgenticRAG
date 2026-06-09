"""Perizinan domain — izin/perizinan layanan SIPUAS Kab. Batang.

Sumber: data/raw/perizinan/perizinan_data.json (34 jenis izin). Tiap perizinan
jadi satu Document atomik (lihat preprocess.py); chunking = identity.

NOTE: Source object dirakit penuh di Step 3 (butuh capability + chunk). Untuk
sekarang cukup export build_documents + path agar preprocess CLI bisa jalan.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from ragtrial.config import RAW_DIR
from ragtrial.sources.perizinan.preprocess import preprocess

# Step 2 memindahkan ini ke config sebagai PERIZINAN_RAW_DIR.
_PERIZINAN_RAW_DIR = RAW_DIR / "perizinan"
_DATA_FILE = "perizinan_data.json"


def build_documents() -> List[Document]:
    """Parse JSON perizinan di raw dir jadi Documents atomik."""
    return preprocess(_PERIZINAN_RAW_DIR / _DATA_FILE)

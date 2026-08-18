"""Evaluasi generation berbasis RAGAS (Faithfulness + Semantic Similarity).

Paket ini SENGAJA tidak meng-import apa pun dari `ragtrial` maupun `eval.eval_core`.
Ia murni JSON-in/JSON-out: membaca artefak `per_query_<system>.json` yang sudah
dihasilkan fase generation, lalu menilainya. Konsekuensinya:

  1. bisa dijalankan di virtualenv terpisah (`.venv-ragas`) tanpa menyentuh
     dependency produksi;
  2. secara struktural MUSTAHIL memicu retrieval atau generation ulang.

Lihat docs/EVAL_GENERATION_RAGAS.md untuk metodologinya.
"""

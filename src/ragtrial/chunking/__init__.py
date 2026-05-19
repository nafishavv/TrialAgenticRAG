"""Chunking strategies — turn cleaned source Documents into retrievable chunks.

Each source picks a strategy that matches its data nature:
  - dukcapil: Q&A regex split for BAB II + recursive narrative split for others
  - opd: identity (atomic records, no chunking)
"""

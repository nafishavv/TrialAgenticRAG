"""Synthetic test-set generation package.

Builds eval/testset.json by grounding LLM-generated questions on real chunks
(the exact embed-ready chunks produced by `SOURCES[domain].chunk(...)`), so every
gold_chunks entry resolves to a chunk the retriever can actually return.

Layout:
  schema.py     — enums, QRecord, gold-id stripping, serialization
  chunks.py     — load chunks, compute gold-ids, stratified sampling
  prompts.py    — LLM prompt skeletons + strict-JSON parsing
  llm_client.py — paced + rate-limit-retrying generation client
  generators.py — single / multi / cross / none question builders
  dedup.py      — de-leakage + near-duplicate rejection
"""

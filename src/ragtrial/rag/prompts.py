"""Prompt templates for the naive / enhanced / agentic pipelines.

Instruction scaffolding is written in ENGLISH (models follow English instructions
more reliably), but every answer-generating prompt explicitly directs the model to
RESPOND IN BAHASA INDONESIA — this is an Indonesian citizen-service bot. Literal
Indonesian response strings (e.g. the "not found" refusal) and domain content
(service categories, capability descriptions, few-shot examples) stay in Indonesian
on purpose: they are output text / classifier data, not instructions.
"""

from __future__ import annotations

PROMPT_COMBINED = """You are a public-service assistant for Kabupaten Batang. You have access to several context sources:
{sources_brief}

RULES:
1. Answer ONLY from the context. Do not add information from general knowledge.
2. Use only the sources relevant to the question; ignore irrelevant context (do not force it in).
3. If the question needs only one kind of information, answer directly WITHOUT forcing a multi-part structure.
4. If the question needs more than one source, combine them concisely.
5. If the information is not present in any context, reply exactly: "Maaf, informasi tidak ditemukan dalam sumber yang tersedia."
6. Respond in clear, concise Bahasa Indonesia.

CITATION:
{citation_rules}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


PROMPT_NAIVE = """You are a public-service assistant for Kabupaten Batang. Answer the question ONLY from the CONTEXT below.
If the answer is not in the context, say "Maaf, informasi tidak ditemukan." Respond concisely in Bahasa Indonesia.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


PROMPT_SINGLE = """You are a public-service assistant for Kabupaten Batang.
Source used: {source_description}

RULES:
1. Answer ONLY from the context. Do not add information from general knowledge.
2. If the answer is not in the context, reply: "Maaf, informasi tidak ditemukan di sumber yang tersedia."
3. Respond in clear, concise Bahasa Indonesia.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


PROMPT_HYDE = """Write ONE paragraph of a hypothetical answer to the following question, as if quoted from an official public-service document of Kabupaten Batang (a regulation, SOP, or guideline).

RULES:
- Write like formal document content: direct, fact-dense, in the style of government regulations/guidelines.
- Do NOT say you don't know or ask for clarification — fabricate a plausible, factual-sounding answer.
- Mention concrete relevant terms/entities (document names, requirements, procedures, agencies) where sensible.
- At most 4-5 sentences, one paragraph, in formal Bahasa Indonesia. No opening/closing lines.

QUESTION: {question}

HYPOTHETICAL PARAGRAPH:"""


REWRITE_PROMPT = """Given the conversation history and the user's latest question, rewrite the latest question into a STANDALONE question that is understandable without the prior conversation.

RULES:
- Resolve pronouns & references ("itu", "tadi", "dia", "yang sebelumnya") to explicit entities from the history.
- Keep the original question's language & style (do NOT translate it).
- If the question is already standalone (no reference to the history), return it AS-IS.
- Output: ONLY the rewritten question, one line, no explanation, no quotes, no prefix.

Conversation history:
{history}

Latest question: {question}

Standalone question:"""


PROMPT_NONE = """The user's question is outside the scope of the Kabupaten Batang public-service chatbot.
Politely reply that this question is out of scope, and offer help related to the available sources.
Respond briefly in Bahasa Indonesia.

QUESTION: {question}

ANSWER:"""


# Public-service categories the chatbot handles. Shared by the intent gate
# (PROMPT_INVALID) and the agentic system prompt so identity/refusal messages stay
# consistent. Kept in Indonesian — these are the actual service names shown to users.
SERVICE_CATEGORIES: list[str] = [
    "Perizinan",
    "Kependudukan",
    "Pajak Daerah",
    "Hukum & Peraturan",
    "Informasi & Komunikasi",
]


def service_categories_block() -> str:
    """Render service categories as a bullet list for prompts."""
    return "\n".join(f"- {c}" for c in SERVICE_CATEGORIES)


# Used for INVALID queries (intent gate skips retrieval). ONE prompt handles two
# cases at once — the LLM distinguishes chit-chat/identity vs out-of-scope.
PROMPT_INVALID = """You are a public-service chatbot assistant for Kabupaten Batang.

You can help with information about:
{categories}

The following question does NOT require document retrieval. Respond according to its type:

- If the user greets you, introduces themselves, or asks about your identity / what services you offer:
  briefly explain that you are a chatbot dedicated to Kabupaten Batang public services,
  list the service categories above, then offer help warmly.
- If the user asks about something outside public services (e.g. product prices, national figures,
  recipes, other general topics): politely decline, saying
  "Maaf, saya hanya bisa membantu dengan informasi layanan publik Kabupaten Batang.",
  then steer back to the service categories above.

Use polite, formal Bahasa Indonesia. Be concise; do not fabricate service information.

QUESTION: {question}

ANSWER:"""


ROUTER_PROMPT = """You are a router for the Kabupaten Batang public-service chatbot.
Your task: classify the user's question into ONE of the categories below.

CATEGORIES:
{categories_block}
- "both"      → a question that needs BOTH / a combination of sources (e.g. procedure + agency contact).
- "none"      → outside the scope of all sources above (recipes, sports, entertainment, etc.).

EXAMPLES:
{examples_block}
Q: "Resep nasi goreng?"  → {{"route":"none","reason":"off-topic"}}

OUTPUT RULES:
- Reply with ONLY one line of valid JSON: {{"route":"<category>","reason":"<short reason>"}}
- No other text, no markdown, no code fence.

Question: {question}
JSON:"""


def build_sources_brief(capabilities: dict) -> str:
    """Render bullet list of source name + description for PROMPT_COMBINED."""
    return "\n".join(
        f"- {cap.name.upper()} — {cap.description}" for cap in capabilities.values()
    )


def build_citation_rules(capabilities: dict) -> str:
    """Render per-source citation instructions derived from each capability's hint.

    Adding a source with a `citation` hint extends this block automatically.
    """
    lines = [
        f"- {cap.name.upper()}: {cap.citation_hint()}"
        for cap in capabilities.values()
        if cap.citation_hint()
    ]
    return "\n".join(lines) if lines else "- (no special citation format)"


def build_router_prompt(question: str, capabilities: dict) -> str:
    """Build ROUTER_PROMPT with categories and few-shot examples derived from registry."""
    cats = "\n".join(
        f'- "{cap.name}"  → {cap.description}' for cap in capabilities.values()
    )
    examples = []
    for cap in capabilities.values():
        for ex in cap.router_examples:
            examples.append(
                f'Q: "{ex}"  → {{"route":"{cap.name}","reason":"example {cap.name}"}}'
            )
    return ROUTER_PROMPT.format(
        categories_block=cats,
        examples_block="\n".join(examples) + "\n" if examples else "",
        question=question,
    )

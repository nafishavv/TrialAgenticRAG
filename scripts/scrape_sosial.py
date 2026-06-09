"""
Scraper untuk JDIH Batang - Sektor Sosial
Output:
  - data/raw/sosial/metadata.json  (metadata semua dokumen)
  - data/raw/sosial/{id}.pdf       (file PDF tiap dokumen)

Usage:
    uv run python scripts/scrape_sosial.py
"""

import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL   = "https://jdih.batangkab.go.id"
TAG_SLUG   = "sosial"
API_KEY    = "ZGhSdSt6U2NnQ3pvRXZWK2dUOTZLekkrZ0NHcngwOEZGcjNsZW9EZXpyaz0="
OUTPUT_DIR = Path("data/raw/sosial")
DELAY      = 3  # seconds between requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAGTrial-Scraper/1.0)",
    "Referer": BASE_URL,
}

# ── STEP 1: FETCH LIST (paginated AJAX) ───────────────────────────────────────
def fetch_list_page(page: int) -> list[dict]:
    url = f"{BASE_URL}/page/load_data_dokumen_tag/{page}"
    payload = {
        "action":    "fetch_data",
        "pencarian": "",
        "slug":      TAG_SLUG,
        "api_key":   API_KEY,
    }
    resp = requests.post(url, data=payload, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    html = data.get("dokumen", "")
    if not html or html.strip() == "":
        return []

    soup = BeautifulSoup(html, "html.parser")

    # All divs matching the outer-card pattern
    all_cards = soup.find_all(
        "div",
        class_=lambda c: c and "overflow-hidden" in c and "br-0" in c,
    )
    # Exclude nested inner cards (they carry border-0 + box-shadow-0)
    outer_cards = [
        card for card in all_cards
        if not (
            "border-0" in card.get("class", [])
            and "box-shadow-0" in card.get("class", [])
        )
    ]
    if not outer_cards:
        return []

    return [item for card in outer_cards if (item := parse_card(card))]


def parse_card(card) -> dict | None:
    h4 = card.find("h4")
    if not h4:
        return None
    judul_list = h4.get_text(strip=True)

    tipe_list = ""
    for span in card.find_all("span"):
        if span.find("i", class_=lambda c: c and "fa-file-text" in c):
            tipe_list = span.get_text(strip=True)
            break

    detail_url = ""
    pdf_url    = ""
    footer = card.find("div", class_="card-footer")
    if footer:
        for a in footer.find_all("a", href=True):
            href    = a["href"]
            classes = a.get("class", [])
            if "btn-info" in classes:
                detail_url = href if href.startswith("http") else BASE_URL + href
            elif "btn-success" in classes:
                pdf_url = href if href.startswith("http") else BASE_URL + href

    if not detail_url:
        return None

    return {
        "judul_list": judul_list,
        "tipe_list":  tipe_list,
        "detail_url": detail_url,
        "pdf_url":    pdf_url,
    }


def fetch_all_list() -> list[dict]:
    """Paginate sampai halaman kosong (max 15 halaman safety limit)."""
    all_items: list[dict] = []
    page = 1
    while page <= 15:
        print(f"  Fetching list page {page}...")
        items = fetch_list_page(page)
        if not items:
            print(f"  ✓ Halaman {page} kosong, selesai pagination.")
            break
        all_items.extend(items)
        print(f"  ✓ {len(items)} item di halaman {page} (total: {len(all_items)})")
        page += 1
        time.sleep(DELAY)
    return all_items


# ── STEP 2: FETCH DETAIL PAGE ─────────────────────────────────────────────────
LABEL_MAP = {
    "Nomor Peraturan":        "nomor",
    "Judul":                  "judul",
    "Tahun Terbit":           "tahun",
    "Jenis/Bentuk Peraturan": "tipe_dokumen",
    "Bidang":                 "bidang",
    "OPD Pemrakarsa":         "opd_pemrakarsa",
    "Status Peraturan":       "status",
}

def fetch_detail(detail_url: str) -> dict:
    resp = requests.get(detail_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    metadata: dict = {}

    table = soup.find("table", class_="tb-detail")
    if table:
        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) < 3:
                continue
            label = cols[0].get_text(strip=True)
            value = cols[2].get_text(strip=True)
            field = LABEL_MAP.get(label)
            if field:
                metadata[field] = value

    # PDF url dari download button di detail page
    pdf_url = ""
    pdf_btn = soup.find("a", class_="btn-success", href=re.compile(r"\.pdf"))
    if not pdf_btn:
        pdf_btn = soup.find("a", href=re.compile(r"/file/.*\.pdf"))
    if pdf_btn:
        href = pdf_btn["href"]
        pdf_url = href if href.startswith("http") else BASE_URL + href

    metadata["_pdf_url_detail"] = pdf_url  # temp key, popped in main
    return metadata


# ── STEP 3: BUILD ID ──────────────────────────────────────────────────────────
TIPE_SHORT = {
    "peraturan-daerah":           "perda",
    "peraturan-bupati":           "perbup",
    "keputusan-bupati":           "kepbup",
    "instruksi-bupati":           "instruksi",
    "rancangan-peraturan-daerah": "ranperda",
    "rancangan-peraturan-bupati": "ranperbup",
    "abstrak-peraturan-daerah":   "abstrak-perda",
    "abstrak-peraturan-bupati":   "abstrak-perbup",
    "naskah-akademis":            "naskah-akademis",
    "surat-edaran-bupati":        "se-bupati",
    "surat-edaran-sekda":         "se-sekda",
    # tipe non-regulasi (tidak punya nomor standar)
    "artikel-hukum":              "artikel",
    "monografi":                  "monografi",
    "peraturan-terjemahan-resmi": "terjemahan",
}

def build_id(doc: dict) -> str:
    """sosial-{tipe_slug}-{nomor_or_short_hash}-{tahun}"""
    import hashlib

    # tipe: prefer tipe_dokumen (detail page), fall back to tipe_list (list page)
    tipe_raw = doc.get("tipe_dokumen", "").strip() or doc.get("tipe_list", "dokumen").strip()
    tipe = re.sub(r"[^a-z0-9]+", "-", tipe_raw.lower()).strip("-")
    tipe_slug = TIPE_SHORT.get(tipe, tipe[:20])

    nomor_raw = str(doc.get("nomor", "")).strip()
    if nomor_raw:
        nomor = re.sub(r"[^0-9a-zA-Z]", "-", nomor_raw).strip("-")
    else:
        url_tail = doc.get("source_url", "").rstrip("/").split("/")[-1]
        # Try to extract nomor from URL slug: "...-no-{nomor}-tahun-..."
        m = re.search(r"-no-(\d[\d-]*)-tahun", url_tail)
        if m:
            nomor = m.group(1)
        else:
            # No numeric nomor in URL → short hash for uniqueness
            nomor = hashlib.md5(url_tail.encode()).hexdigest()[:8]

    tahun = str(doc.get("tahun", "xxxx"))
    return f"sosial-{tipe_slug}-{nomor}-{tahun}"


# ── STEP 4: DOWNLOAD PDF ──────────────────────────────────────────────────────
def download_pdf(pdf_url: str, dest_path: Path) -> bool:
    if not pdf_url:
        print("    ⚠  Tidak ada PDF URL")
        return False
    try:
        resp = requests.get(pdf_url, headers=HEADERS, timeout=30, stream=True)
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_kb = dest_path.stat().st_size / 1024
        print(f"    ✓ PDF saved ({size_kb:.0f} KB) → {dest_path.name}")
        return True
    except Exception as e:
        print(f"    ✗ PDF download failed: {e}")
        return False


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT_DIR / "metadata.json"

    # Resume: load existing jika ada, keyed by source_url (= detail_url)
    existing: dict[str, dict] = {}
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            for doc in json.load(f):
                url = doc.get("source_url")
                if url:  # skip any doc tanpa source_url (data corrupt lama)
                    existing[url] = doc
        print(f"Resume: {len(existing)} dokumen sudah ada.\n")

    # STEP 1: fetch list
    print("=== STEP 1: Fetch list ===")
    list_items = fetch_all_list()
    print(f"Total {len(list_items)} item ditemukan.\n")

    # STEP 2, 3, 4: per dokumen
    print("=== STEP 2: Fetch detail + download PDF ===")
    results: list[dict] = list(existing.values())
    processed_urls = set(existing.keys())  # same type as source_url (detail_url)

    for i, item in enumerate(list_items, 1):
        detail_url = item["detail_url"]
        if detail_url in processed_urls:
            print(f"[{i}/{len(list_items)}] SKIP: {item['judul_list'][:60]}")
            continue

        print(f"\n[{i}/{len(list_items)}] {item['judul_list'][:70]}")

        try:
            detail = fetch_detail(detail_url)
            time.sleep(DELAY)
        except Exception as e:
            print(f"  ✗ Detail fetch failed: {e}")
            continue

        # PDF URL: prefer detail page button, fall back to list-page link
        pdf_url = detail.pop("_pdf_url_detail") or item.get("pdf_url", "")

        doc: dict = {
            **detail,
            # list-page fields kept as fallback / extra context
            "judul_list":      item["judul_list"],
            "tipe_list":       item["tipe_list"],
            # resolved PDF url (after fallback logic)
            "pdf_url":         pdf_url,
            # provenance
            "sector":          "sosial",
            "source_url":      detail_url,
            "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Backfill tipe_dokumen dari tipe_list kalau detail page tidak punya
        if not doc.get("tipe_dokumen") and doc.get("tipe_list"):
            doc["tipe_dokumen"] = doc["tipe_list"]

        doc["id"]       = build_id(doc)
        doc["filename"] = f"{doc['id']}.pdf"
        pdf_path        = OUTPUT_DIR / doc["filename"]

        print(f"  PDF: {pdf_url}")
        if not pdf_path.exists():
            download_pdf(pdf_url, pdf_path)
            time.sleep(DELAY)
        else:
            print(f"  ✓ PDF sudah ada")

        doc["pdf_available"] = pdf_path.exists()
        results.append(doc)

        # Save progress setiap dokumen (resume-safe)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n=== SELESAI ===")
    print(f"Total dokumen: {len(results)}")
    print(f"Metadata: {metadata_path}")
    success = sum(1 for d in results if d.get("pdf_available"))
    print(f"PDF downloaded: {success}/{len(results)}")


if __name__ == "__main__":
    main()

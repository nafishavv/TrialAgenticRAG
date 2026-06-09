"""
Scraper untuk data perizinan SIPUAS Batang Kabupaten
Output: data/raw/perizinan/perizinan_data.json

Usage:
    uv run python scripts/scrape_perizinan.py
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from datetime import datetime, timezone
from pathlib import Path

# -- CONFIG -------------------------------------------------------------------
BASE_URL = "https://sipuas.batangkab.go.id"
LIST_ENDPOINT = f"{BASE_URL}/web/beranda/get_perizinan"
API_KEY = "OW41LzQzempPVGJNdVZ4aU5vcjBEcmdQTjViZWV2NDluOENpelJ1YW9uZz0="
DELAY_SECONDS = 3
OUTPUT_DIR = Path("data/raw/perizinan")
OUTPUT_FILE = OUTPUT_DIR / "perizinan_data.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": BASE_URL,
}


# -- HELPERS ------------------------------------------------------------------

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text


def fetch_list() -> list[dict]:
    print("Fetching daftar perizinan...")
    resp = requests.post(
        LIST_ENDPOINT,
        data={"pencarian": "", "api_key": API_KEY},
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    soup = BeautifulSoup(data["data"], "html.parser")
    items = []
    for a in soup.find_all("a", href=True):
        nama_tag = a.find("b")
        kategori_tag = a.find("p")
        if nama_tag:
            items.append({
                "nama": nama_tag.get_text(strip=True),
                "kategori": kategori_tag.get_text(strip=True) if kategori_tag else "",
                "url": a["href"],
            })

    print(f"   Ditemukan {len(items)} perizinan")
    return items


def parse_persyaratan(soup: BeautifulSoup) -> list[str]:
    section = soup.find("div", id="pills-syarat")
    if not section:
        return []
    return [li.get_text(strip=True) for li in section.find_all("li") if li.get_text(strip=True)]


def parse_mekanisme(soup: BeautifulSoup) -> list[dict]:
    section = soup.find("div", id="pills-mekanisme")
    if not section:
        return []
    rows = []
    for tr in section.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) >= 2:
            nomor_text = tds[0].get_text(strip=True).rstrip(".")
            deskripsi = tds[1].get_text(strip=True)
            if nomor_text.isdigit() and deskripsi:
                rows.append({"nomor": int(nomor_text), "deskripsi": deskripsi})
    return rows


def parse_dasar_hukum(soup: BeautifulSoup) -> list[dict]:
    section = soup.find("div", id="pills-hukum")
    if not section:
        return []
    rows = []
    for tr in section.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) >= 2:
            nomor_text = tds[0].get_text(strip=True).rstrip(".")
            referensi = tds[1].get_text(strip=True)
            if nomor_text.isdigit() and referensi:
                rows.append({"nomor": int(nomor_text), "referensi": referensi})
    return rows


def parse_keterangan(soup: BeautifulSoup) -> dict:
    section = soup.find("div", id="pills-keterangan")
    if not section:
        return {}
    result = {}
    key_map = {
        "estimasi selesai": "estimasi_selesai",
        "masa berlaku": "masa_berlaku",
        "biaya": "biaya",
    }
    for tr in section.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) >= 3:
            raw_key = tds[0].get_text(strip=True).lower()
            value = tds[2].get_text(strip=True)
            for k, field in key_map.items():
                if k in raw_key:
                    result[field] = value  # keep "-" as-is so LLM can interpret it
                    break
    return result


def scrape_detail(item: dict) -> dict | None:
    url = item["url"]
    print(f"   Scraping: {item['nama']} ...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"   GAGAL fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    return {
        "id": slugify(item["nama"]),
        "nama_perizinan": item["nama"],
        "kategori": item["kategori"],
        "url_sumber": url,
        "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
        "persyaratan": parse_persyaratan(soup),
        "mekanisme_pelayanan": parse_mekanisme(soup),
        "dasar_hukum": parse_dasar_hukum(soup),
        "keterangan": parse_keterangan(soup),
    }


# -- MAIN ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    perizinan_list = fetch_list()

    results = []
    failed = []

    for i, item in enumerate(perizinan_list, start=1):
        print(f"\n[{i}/{len(perizinan_list)}]")
        data = scrape_detail(item)
        if data:
            results.append(data)
        else:
            failed.append(item["nama"])

        if i < len(perizinan_list):
            time.sleep(DELAY_SECONDS)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Selesai! {len(results)}/{len(perizinan_list)} perizinan berhasil di-scrape")
    print(f"Output: {OUTPUT_FILE.resolve()}")
    if failed:
        print(f"\nGagal ({len(failed)}):")
        for name in failed:
            print(f"   - {name}")


if __name__ == "__main__":
    main()

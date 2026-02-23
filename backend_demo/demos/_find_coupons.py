"""
Great Clips Offer Scraper
Scrapes offer details from Great Clips offer pages and filters
for offers near target locations (San Joaquin, Fresno, Manteca, etc.)
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional


# =============================================================================
# CONFIG
# =============================================================================
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

# Areas you care about — offers matching any of these are saved separately
TARGET_AREAS = [
    "san joaquin", "fresno", "manteca", "mountain house", "tracy",
    "stockton", "modesto", "lodi", "turlock", "merced", "clovis",
    "visalia", "madera", "lathrop", "ripon", "escalon", "oakdale",
]


# =============================================================================
# DATA
# =============================================================================
@dataclass
class OfferData:
    url: str
    price: Optional[str] = None
    location: Optional[str] = None
    expiration_date: Optional[str] = None
    description: Optional[str] = None
    is_target_area: bool = False
    error: Optional[str] = None


# =============================================================================
# OFFER URLS
# =============================================================================
OFFER_URLS = [
    "https://offers.greatclips.com/bWp2H5l",
    "https://offers.greatclips.com/BmjxPaV",
    "https://offers.greatclips.com/zmlW78U",
    "https://offers.greatclips.com/4TlEsJP",
    "https://offers.greatclips.com/F2zEagt",
    "https://offers.greatclips.com/Dcezu58",
    "https://offers.greatclips.com/sO7Zl41",
    "https://offers.greatclips.com/dkIygSf",
    "https://offers.greatclips.com/WwsRrEB",
    "https://offers.greatclips.com/R12OToz",
    "https://offers.greatclips.com/4n896Ai",
    "https://offers.greatclips.com/tjVfvhA",
    "https://offers.greatclips.com/aCAtuVU",
    "https://offers.greatclips.com/4ECj50d",
    "https://offers.greatclips.com/gGmwxRu",
    "https://offers.greatclips.com/gS1yuyY",
    "https://offers.greatclips.com/8nLBx6P",
    "https://offers.greatclips.com/2SXfKr1",
    "https://offers.greatclips.com/vMaE8CK",
    "https://offers.greatclips.com/5ut1w9s",
    "https://offers.greatclips.com/rYMtUgT",
    "https://offers.greatclips.com/VRDfsr4",
    "https://offers.greatclips.com/aeCUWS6",
    "https://offers.greatclips.com/vmmOwot",
    "https://offers.greatclips.com/lTiQcBs",
    "https://offers.greatclips.com/DG4ITzd",
    "https://offers.greatclips.com/i2qf7cp",
    "https://offers.greatclips.com/mlK7Pdr",
    "https://offers.greatclips.com/LA9ZDU6",
    "https://offers.greatclips.com/krPavIr",
    "https://offers.greatclips.com/hOvIrtv",
    "https://offers.greatclips.com/Rn6l90i",
    "https://offers.greatclips.com/MWJbVmp",
    "https://offers.greatclips.com/IyYUH6e",
    "https://offers.greatclips.com/cSQyDJL",
    "https://offers.greatclips.com/bvEA9MY",
    "https://offers.greatclips.com/d5wf9AP",
    "https://offers.greatclips.com/1HAFLlw",
    "https://offers.greatclips.com/P7fgpL7",
    "https://offers.greatclips.com/NBBbbP5",
    "https://offers.greatclips.com/1Ps7to7",
    "https://offers.greatclips.com/aq5iyyz",
    "https://offers.greatclips.com/1dEClhW",
    "https://offers.greatclips.com/pls0QWQ",
    "https://offers.greatclips.com/aWkLplt",
    "https://offers.greatclips.com/dAGM47h",
    "https://offers.greatclips.com/JIiOv3l",
    "https://offers.greatclips.com/jkYTGLH",
    "https://offers.greatclips.com/t9lYQhH",
    "https://offers.greatclips.com/Z6DNy3n",
    "https://offers.greatclips.com/RHLFDRk",
    "https://offers.greatclips.com/Aa9kwvI",
    "https://offers.greatclips.com/afLFnor",
    "https://offers.greatclips.com/ejNwyLa",
    "https://offers.greatclips.com/pc8BPU6",
    "https://offers.greatclips.com/Twtsmjt",
    "https://offers.greatclips.com/ZuxSKnY",
    "https://offers.greatclips.com/J51Y26p",
    "https://offers.greatclips.com/BReupGI",
    "https://offers.greatclips.com/LSz8N8W",
    "https://offers.greatclips.com/7MVTNjx",
    "https://offers.greatclips.com/8yOY45H",
    "https://offers.greatclips.com/5aByDLM",
    "https://offers.greatclips.com/B0eOic7",
    "https://offers.greatclips.com/f6q4nxL",
    "https://offers.greatclips.com/uid4XIZ",
]


# =============================================================================
# SCRAPER
# =============================================================================

def scrape_offer(url: str) -> OfferData:
    """Scrape offer data from a Great Clips offer URL."""
    offer = OfferData(url=url)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        page_text = soup.get_text()

        # Price (e.g., $12.99)
        price_match = re.search(r'\$\d+\.\d{2}', page_text)
        if price_match:
            offer.price = price_match.group()

        # Location — try multiple patterns
        loc_match = re.search(
            r'participating\s+(.+?)\s+area\s+Great\s+Clips', page_text
        )
        if loc_match:
            offer.location = loc_match.group(1).strip()
        else:
            loc_match = re.search(
                r'Great\s+Clips\s+(.+?)\s+at\s+(.+?)\s+in\s+(\w+)', page_text
            )
            if loc_match:
                offer.location = (
                    f"{loc_match.group(1).strip()}, "
                    f"{loc_match.group(2).strip()}, "
                    f"{loc_match.group(3).strip()}"
                )
            else:
                loc_match = re.search(
                    r'Valid\s+at\s+Great\s+Clips\s+(.+?)\s+at\s+(.+?)\s+in\s+([^.]+)',
                    page_text,
                )
                if loc_match:
                    offer.location = loc_match.group(3).strip()

        # Expiration date
        date_match = re.search(
            r'expires?\s*(\d{2}/\d{2}/\d{4})', page_text, re.IGNORECASE
        )
        if date_match:
            offer.expiration_date = date_match.group(1)

        # Description
        desc_match = re.search(
            r'Description\s*(.+?)(?:Terms|$)', page_text, re.DOTALL
        )
        if desc_match:
            desc_text = re.sub(r'\s+', ' ', desc_match.group(1).strip())[:200]
            offer.description = desc_text.strip()

        # Check if this offer is in a target area
        check_text = (page_text + " " + (offer.location or "")).lower()
        for area in TARGET_AREAS:
            if area in check_text:
                offer.is_target_area = True
                break

    except requests.RequestException as e:
        offer.error = f"Request failed: {str(e)}"
    except Exception as e:
        offer.error = f"Parsing error: {str(e)}"

    return offer


def run_scraper():
    """Scrape all offers, filter for target areas, and save results."""
    total = len(OFFER_URLS)
    all_offers: list[OfferData] = []
    target_offers: list[OfferData] = []

    print("=" * 60)
    print("  GREAT CLIPS OFFER SCRAPER")
    print(f"  Scraping {total} offer URLs...")
    print(f"  Filtering for: {', '.join(a.title() for a in TARGET_AREAS[:6])}...")
    print("=" * 60)
    print()

    for i, url in enumerate(OFFER_URLS, 1):
        print(f"[{i}/{total}] {url}")
        offer = scrape_offer(url)
        all_offers.append(offer)

        status = []
        if offer.price:
            status.append(f"${offer.price}")
        if offer.location:
            status.append(offer.location)
        if offer.expiration_date:
            status.append(f"exp {offer.expiration_date}")
        if offer.is_target_area:
            status.append("*** YOUR AREA ***")
            target_offers.append(offer)
        if offer.error:
            status.append(f"ERROR: {offer.error}")

        print(f"  -> {' | '.join(status) if status else 'No data extracted'}")

        # Small delay to be polite
        if i < total:
            time.sleep(0.5)

    # ---- Save all results ----
    all_data = [asdict(o) for o in all_offers]
    with open("great_clips_all_offers.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    # ---- Save target area results ----
    target_data = [asdict(o) for o in target_offers]
    with open("great_clips_my_area.json", "w", encoding="utf-8") as f:
        json.dump(target_data, f, indent=2)

    # ---- Summary ----
    print()
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    valid = [o for o in all_offers if not o.error]
    print(f"  Total scraped:   {len(valid)}/{total}")
    print(f"  Your area:       {len(target_offers)} offers found")
    print()
    print(f"  Saved all    -> great_clips_all_offers.json")
    print(f"  Saved nearby -> great_clips_my_area.json")

    if target_offers:
        print()
        print("=" * 60)
        print("  OFFERS IN YOUR AREA")
        print("=" * 60)
        for o in target_offers:
            print(f"\n  URL:      {o.url}")
            print(f"  Price:    {o.price or 'N/A'}")
            print(f"  Location: {o.location or 'N/A'}")
            print(f"  Expires:  {o.expiration_date or 'N/A'}")
            if o.description:
                print(f"  Details:  {o.description[:100]}")
    else:
        print("\n  No offers found in your target areas.")

    print()


if __name__ == "__main__":
    run_scraper()
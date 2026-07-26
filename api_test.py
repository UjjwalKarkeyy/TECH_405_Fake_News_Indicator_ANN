import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

categories = ["true", "mostly-true", "false", "pants-fire"]
base_url = "https://api.politifact.com/factchecks/list/"

records = []

for ruling in categories:
    response = requests.get(
        base_url,
        params={
            "category": "environment",
            "ruling": ruling,
        },
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=20,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    items = soup.select(".o-listicle__list > li")

    print(f"{ruling}: {len(items)} items")

    for item in items:
        statement_element = item.select_one(".m-statement__quote a")

        if statement_element is None:
            continue

        records.append({
            "statement": statement_element.get_text(" ", strip=True),
            "label": ruling
        })

for record in records[:5]:
    print(record)
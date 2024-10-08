import re
import sys
import time

import requests
import tqdm
from bs4 import BeautifulSoup

special_things_to_not_include_if_in_href = [
    "identifier",
    "Special",
    "Portal",
    "Template",
    ".png",
    "Category",
    "Help",
    "Wikipedia",
    "File",
    "Main_Page",
    "Talk",
    "Special",
    "User",
]
special_things_to_not_include_if_in_title = ["Article", "Read"]


def extract_redirects(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    redirects = {
        "pages": [],
    }

    # Extract links to other Wikipedia pages
    for a in soup.find_all("a", href=re.compile("^/wiki/")):
        if not a.find_parent("figure"):
            if all(
                [x not in a["href"] for x in special_things_to_not_include_if_in_href]
            ) and all(
                [x not in a.text for x in special_things_to_not_include_if_in_title]
            ):
                redirects["pages"].append({"text": a.text, "href": a["href"]})

    return redirects


# Example usage
html_content = requests.get("https://en.wikipedia.org/wiki/Virology").text
result = extract_redirects(html_content)

result["pages"] += [{"text": "Virology", "href": "/wiki/Virology"}]


def get_wikipedia_text(url):
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "format": "json",
            "action": "query",
            "prop": "extracts",
            "exlimit": "max",
            "explaintext": "",
            "titles": url,
            "redirects": "",
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))["extract"]
    return page


full_text_of_all_pages = ""
for page in (pbar := tqdm.tqdm(result["pages"])):
    pbar.set_postfix({"page": page["text"]})
    try:
        full_text = get_wikipedia_text(f"{page['href'][6:]}")
        print(full_text)
    except Exception:
        print(f"Failed to get text for {page['text']}", file=sys.stderr)
    # flush stdout
    sys.stdout.flush()
    time.sleep(1.5)

# run with python get-virology-adjacent.py > full_text_of_virology_and_children.txt

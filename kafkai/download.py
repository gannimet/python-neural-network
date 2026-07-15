import requests
from bs4 import BeautifulSoup
import time
import re

HEADERS = {"User-Agent": "Mozilla/5.0 (privates Lernprojekt, NLP-Textkorpus)"}

def fetch_chapter_text(url):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Alle Absätze im Haupttext einsammeln
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if p]  # leere raus

    # Die Seiten enthalten den Kapiteltext oft doppelt (Normalansicht + "Lesemodus").
    # Dedupe unter Beibehaltung der Reihenfolge:
    seen = set()
    unique_paragraphs = []
    for p in paragraphs:
        if p not in seen:
            seen.add(p)
            unique_paragraphs.append(p)

    return "\n\n".join(unique_paragraphs)


def download_book(base_url_template, n_chapters, output_file):
    all_text = []
    for i in range(1, n_chapters + 1):
        url = base_url_template.format(i)
        print(f"Lade Kapitel {i}/{n_chapters}: {url}")
        try:
            text = fetch_chapter_text(url)
            all_text.append(text)
        except Exception as e:
            print(f"  Fehler bei Kapitel {i}: {e}")
        time.sleep(1)  # höflich sein, Server nicht bombardieren

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))
    print(f"Fertig: {output_file}")


# Das Schloss (20 Kapitel)
download_book(
    "https://projekt-gutenberg.org/authors/franz-kafka/books/das-schloss/chapter/{}",
    20,
    "das_schloss.txt"
)

# Amerika (39 Kapitel)
download_book(
    "https://projekt-gutenberg.org/authors/franz-kafka/books/franz-kafka-amerika/chapter/{}",
    39,
    "amerika.txt"
)
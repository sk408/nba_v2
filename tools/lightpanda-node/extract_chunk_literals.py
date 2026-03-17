import argparse
import pathlib
import re


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to JS chunk file")
    args = parser.parse_args()

    text = pathlib.Path(args.path).read_text(encoding="utf-8", errors="ignore")

    urls = sorted(set(re.findall(r'https?://[^"\\s)]+', text)))
    api_paths = sorted(set(re.findall(r"/api/[A-Za-z0-9_\-/{}\[\]?=&.]*", text)))

    print(f"urls {len(urls)}")
    for value in urls[:150]:
        print(value)

    print("--- api-like literals ---")
    for value in api_paths[:300]:
        print(value)

    keywords = [
        "stargate",
        "season",
        "episode",
        "search",
        "tmdb",
        "imdb",
        "trakt",
        "watchparty",
        "browse",
        "/tv/",
        "/movie/",
    ]

    print("--- keyword hits ---")
    for key in keywords:
        if key in text.lower():
            print(f"has {key}")

    print("--- keyword context ---")
    lower = text.lower()
    for key in keywords:
        idx = lower.find(key)
        if idx == -1:
            continue
        start = max(0, idx - 120)
        end = min(len(text), idx + 220)
        snippet = text[start:end].replace("\n", " ")
        print(f"[{key}] {snippet}")


if __name__ == "__main__":
    main()

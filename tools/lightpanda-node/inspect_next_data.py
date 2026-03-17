import argparse
import html
import json
import pathlib


def extract_pre(raw: str) -> str:
    start = raw.find("<pre>")
    end = raw.rfind("</pre>")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate <pre>...</pre> JSON payload")
    payload = raw[start + len("<pre>") : end]
    return html.unescape(payload)


def flatten_strings(obj, out):
    if isinstance(obj, str):
        out.append(obj)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            flatten_strings(value, out)
        return
    if isinstance(obj, list):
        for value in obj:
            flatten_strings(value, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to saved HTML with <pre> JSON")
    parser.add_argument("--keyword", default="", help="Optional keyword to search in all string values")
    args = parser.parse_args()

    raw = pathlib.Path(args.path).read_text(encoding="utf-8", errors="ignore")
    payload = extract_pre(raw)
    data = json.loads(payload)

    print("top_keys:", list(data.keys()))
    page_props = data.get("pageProps", {})
    print("pageProps_keys:", list(page_props.keys()))
    print("__N_SSG:", data.get("__N_SSG"))
    print("__N_SSP:", data.get("__N_SSP"))

    if args.keyword:
        bucket = []
        flatten_strings(data, bucket)
        needle = args.keyword.lower()
        matches = [s for s in bucket if needle in s.lower()]
        print(f"keyword_matches({args.keyword}):", len(matches))
        for value in matches[:50]:
            print(value)


if __name__ == "__main__":
    main()

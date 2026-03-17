import argparse
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to .js.map file")
    parser.add_argument(
        "--needle",
        action="append",
        default=[],
        help="Case-insensitive keyword to search for in sourcesContent",
    )
    args = parser.parse_args()

    needles = [n.lower() for n in args.needle] or [
        "api",
        "search",
        "season",
        "episode",
        "tmdb",
        "imdb",
        "stargate",
        "cineby",
    ]

    raw = pathlib.Path(args.path).read_text(encoding="utf-8", errors="ignore")
    data = json.loads(raw)
    sources = data.get("sources", [])
    contents = data.get("sourcesContent", [])

    print(f"sources={len(sources)}")
    total_hits = 0

    for idx, content in enumerate(contents):
        if not isinstance(content, str):
            continue
        lower = content.lower()
        matching_needles = [n for n in needles if n in lower]
        if not matching_needles:
            continue

        total_hits += 1
        source_name = sources[idx] if idx < len(sources) else f"source#{idx}"
        print(f"\n=== {source_name} ===")
        print("matched:", ", ".join(matching_needles))

        lines = content.splitlines()
        shown = 0
        for line in lines:
            low_line = line.lower()
            if any(n in low_line for n in matching_needles):
                print(line[:500])
                shown += 1
            if shown >= 30:
                break

    print(f"\nfiles_with_hits={total_hits}")


if __name__ == "__main__":
    main()

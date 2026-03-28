import argparse
from web_gui.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CARDIOTECT web GUI.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    serve(port=args.port, open_window=not args.no_browser)


if __name__ == "__main__":
    main()

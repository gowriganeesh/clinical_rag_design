"""Minimal CLI chat over the clinical RAG multi-agent pipeline.

Usage:
    python -m app.cli_chat --user EXAMINER_1192
"""
from __future__ import annotations

import argparse

from src.agents.orchestrator import respond


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical RAG chat")
    parser.add_argument(
        "--user",
        default="EXAMINER_1192",
        help="Acting user id; drives the ACL filter.",
    )
    args = parser.parse_args()

    print(f"Clinical RAG chat - acting as {args.user}. Ctrl-C to exit.\n")
    try:
        while True:
            query = input("you > ").strip()
            if not query:
                continue
            output = respond(query, user_id=args.user)
            print(f"bot > {output['answer']}")
            print(f"      [{output['meta']}]\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()

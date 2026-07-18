"""CLI entrypoint. Subcommands: run-once | poll-votes | calibrate | refresh-taste."""

from __future__ import annotations

import argparse
import logging
import sys

from .calibrate import run_calibrate
from .config import load_config
from .db import connect, init_schema
from .pipeline import SendPlan, persist_and_send, run_cycle
from .taste import refresh_taste
from .telegram import is_configured
from .votes import poll_votes

log = logging.getLogger("research-bot")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Silence chatty third-party libs (HF cache-metadata HEADs, device notices).
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_plan(plan: SendPlan) -> None:
    print(
        f"\nfetched {plan.fetched} · new {plan.new} · qualifiers {plan.qualifiers} "
        f"(after bar+age+dedup) · sent in 24h {plan.already_sent_24h} · slots {plan.slots}"
    )
    if not plan.to_send:
        print("\nno papers to send — quiet day (this is correct behavior).")
        return
    print(f"\nwould send {len(plan.to_send)}:\n")
    for p, r, _ in plan.to_send:
        print(f"  🔬 {p.title}")
        print(f"     {p.source} · similarity {r.score:.2f} · closest to: {r.nearest}")
        print(f"     {p.url}")


def cmd_run_once(args: argparse.Namespace) -> int:
    cfg = load_config()
    conn = connect(cfg.db_path)
    try:
        init_schema(conn)
        plan = run_cycle(cfg, conn)
        _print_plan(plan)
        if args.dry:
            print("\n(--dry: nothing sent or persisted.)")
        elif not is_configured(cfg):
            print(
                "\n(Telegram not configured — nothing sent or persisted. "
                "Set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env, or use --dry.)"
            )
        else:
            sent = persist_and_send(cfg, conn, plan)
            print(f"\nsent {sent} paper(s); {plan.new} new marked seen.")
    finally:
        conn.close()
    return 0


def cmd_poll_votes(args: argparse.Namespace) -> int:
    cfg = load_config()
    conn = connect(cfg.db_path)
    try:
        init_schema(conn)
        poll_votes(cfg, conn, once=args.once)
    finally:
        conn.close()
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = load_config()
    conn = connect(cfg.db_path)
    try:
        init_schema(conn)
        run_calibrate(cfg, conn, days=args.days)
    finally:
        conn.close()
    return 0


def cmd_refresh_taste(args: argparse.Namespace) -> int:
    cfg = load_config()
    conn = connect(cfg.db_path)
    try:
        init_schema(conn)
        counts = refresh_taste(cfg, conn)
    finally:
        conn.close()
    log.info(
        "taste profile: %d seeds, %d 👍 positives, %d 👎 negatives",
        counts.get("seed", 0),
        counts.get("pos", 0),
        counts.get("neg", 0),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-bot")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run-once", help="one poll cycle then exit")
    p_run.add_argument(
        "--dry", action="store_true", help="preview the send plan without sending (M7+)"
    )
    p_run.set_defaults(func=cmd_run_once)

    p_poll = sub.add_parser("poll-votes", help="Telegram 👍/👎 callback loop")
    p_poll.add_argument(
        "--once", action="store_true", help="drain pending votes and exit (for cron)"
    )
    p_poll.set_defaults(func=cmd_poll_votes)

    p_cal = sub.add_parser("calibrate", help="bar -> papers/day table on live data")
    p_cal.add_argument("--days", type=int, default=14)
    p_cal.set_defaults(func=cmd_calibrate)

    p_ref = sub.add_parser(
        "refresh-taste", help="(re)build the taste profile from seeds + votes"
    )
    p_ref.set_defaults(func=cmd_refresh_taste)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

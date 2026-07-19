"""CLI entrypoint. Subcommands: run-once | poll-votes | calibrate | refresh-taste."""

from __future__ import annotations

import argparse
import logging
import sys

import httpx

from .calibrate import run_calibrate
from .config import load_config
from .db import connect, init_schema
from .pipeline import (
    SendPlan,
    persist_and_send,
    run_cycle,
    select_unsent_qualifiers,
)
from .taste import refresh_taste
from .telegram import is_configured, send_text
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
        f"(after bar+age+dedup) · sent today {plan.already_sent_today} · slots {plan.slots}"
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
            _maybe_quiet_day_alert(cfg, plan.qualifiers)
    finally:
        conn.close()
    return 0


def _maybe_quiet_day_alert(cfg, qualifiers: int) -> None:
    """Ping when fewer than max_per_day papers cleared the bar, so a quiet day is
    never silent. Best-effort: a failed alert never fails the run."""
    if qualifiers >= cfg.max_per_day:
        return
    if qualifiers == 0:
        note = "📭 Quiet day — 0 papers cleared the bar today."
    else:
        s = "s" if qualifiers != 1 else ""
        note = f"📭 Quiet-ish day — only {qualifiers} paper{s} cleared the bar today."
    try:
        send_text(cfg, note)
        print(f"(quiet-day alert sent: {qualifiers}/{cfg.max_per_day} cleared the bar)")
    except httpx.HTTPError as exc:
        log.warning("quiet-day alert failed to send: %s", exc)


def cmd_send_now(args: argparse.Namespace) -> int:
    """Manually send the top-N unsent papers that clear the bar (feed seed / re-kick)."""
    cfg = load_config()
    conn = connect(cfg.db_path)
    try:
        init_schema(conn)
        picks = select_unsent_qualifiers(conn, cfg, ignore_age=args.ignore_age)[: args.count]
        if not picks:
            print("no qualifying unsent papers to send.")
            return 0
        print(f"selected {len(picks)} paper(s) to send now:")
        for p, r, _ in picks:
            print(f"  {r.score:.3f} {p.source} · {p.title}")
        if args.dry:
            print("\n(--dry: nothing sent.)")
        elif not is_configured(cfg):
            print("\n(Telegram not configured — nothing sent.)")
        else:
            sent = persist_and_send(cfg, conn, SendPlan(to_send=picks))
            print(f"\nsent {sent} paper(s).")
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

    p_now = sub.add_parser(
        "send-now", help="manually send the top-N unsent papers that clear the bar"
    )
    p_now.add_argument("--count", type=int, default=3, help="how many to send (default 3)")
    p_now.add_argument(
        "--ignore-age",
        action="store_true",
        help="relax the freshness window (for a one-off reset/seed send)",
    )
    p_now.add_argument("--dry", action="store_true", help="preview without sending")
    p_now.set_defaults(func=cmd_send_now)

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

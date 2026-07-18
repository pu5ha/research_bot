# Local Similarity Research Bot

Watches research-paper feeds (arXiv, bioRxiv/medRxiv, IACR ePrint, NBER), scores each new paper by
embedding **similarity to a taste profile** of papers you already love, and sends you — via Telegram,
with 👍/👎 buttons — only the few that clear a quality bar, capped at **3 per day**. Quiet days send
nothing, and that's correct. Every vote updates the taste profile so it sharpens over time.

**No API keys except Telegram.** Scoring is fully local via a small CPU embedding model
(`BAAI/bge-small-en-v1.5`). Each sent paper also gets a 3-sentence summary from a **local LLM**
via [Ollama](https://ollama.com) (`llama3.2`) — no cloud API. Summaries only run for the ≤3 papers
actually sent per day; if Ollama is unreachable the message falls back to a truncated abstract.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pulls torch + sentence-transformers
cp config.example.yaml config.yaml       # edit if desired
cp .env.example .env                      # fill in when you reach the Telegram milestone
```

The embedding model (~130 MB) downloads automatically on first use.

For summaries, install Ollama and pull the model (optional — falls back to abstracts if absent):

```bash
# https://ollama.com  — then:
ollama pull llama3.2
```

## Environment variables (`.env`)

| var | needed for |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Telegram send / vote loop (M7+) |
| `TELEGRAM_CHAT_ID`   | Telegram send / vote loop (M7+) |
| `CONTACT_EMAIL`      | polite `User-Agent` on source requests |

## CLI

```bash
python -m src.main run-once        # one poll cycle then exit (cron / GitHub Actions)
python -m src.main poll-votes      # Telegram callback loop (long-poll)
python -m src.main calibrate --days 14   # bar -> papers/day table on live data; sends nothing
python -m src.main refresh-taste   # fold new votes into the taste profile
```

## Tuning the bars (`config.yaml`)

`bars` sets the per-source similarity threshold; `windows` sets each source's daily
fetch look-back in days (tight for arXiv's firehose, wider for sparse RSS feeds).
Run `calibrate` to see the bar → papers/day distribution on real data and pick bars
that land you in the 3–10/day range.

> **Note on thresholds:** the spec's default bars (~0.60) assume a similarity
> distribution that `bge-small-en-v1.5` with title-only seed anchors does **not**
> reproduce — its embeddings cluster in a tight high-cosine cone, so the real knee
> sits near **0.80**. Use `calibrate` to set your own bars (and optionally supply
> real seed abstracts in `seeds/ground_truth.txt` to spread the distribution). This
> is exactly the per-field bias the spec anticipates tuning here.

## Tests

```bash
pytest -q
ruff check src tests
```

## Build status (incremental — one milestone at a time)

- [x] **M1** Scaffolding: config, models, DB schema, embedding primitive, CLI stubs, tests.
- [x] **M2** Taste profile from `seeds/ground_truth.txt`.
- [x] **M3** arXiv source + seen-dedup + scoring (`run-once` prints top 10).
- [x] **M4** bioRxiv/medRxiv, IACR, NBER sources (fixture-based tests).
- [x] **M5** `calibrate` mode + per-source fetch windows.
- [x] **M6** near-duplicate dedup + daily send logic (`run-once --dry`).
- [x] **M7** Telegram send (`run-once` pings the top qualifiers).
- [x] **M8** Telegram vote loop (`poll-votes`) + taste refresh.
- [x] **M9** Deployment (cron + GitHub Actions).

## Known limitations (by design)

- Similarity echoes your existing taste; it won't surprise you with fields none of your seeds touch.
  The vote loop widens it slowly, not instantly.
- The first ~2 weeks are the least accurate (the profile is just the seeds); it improves as you vote.
- Coverage is preprints + NBER, not full journals; social buzz (e.g. Hacker News) is intentionally ignored.
- Per-source bars are a blunt fix for embedding bias across fields (crypto scored low); revisit via `calibrate`.

## Deployment

`data/bot.db` **is** the bot's memory (seen papers, sends, votes, taste). Idempotency —
never pinging the same paper twice — depends entirely on that file persisting between runs.
So the deploy target must give the DB a stable home. A **$5 VPS with cron is the recommended
setup**; GitHub Actions works but its ephemeral runners make DB persistence fragile.

### (a) Cheap VPS with cron (recommended)

```bash
git clone <your-repo> ~/ai_podcast && cd ~/ai_podcast
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
cp .env.example .env            # fill in TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID / CONTACT_EMAIL
curl -fsSL https://ollama.com/install.sh | sh   # for summaries (optional)
ollama pull llama3.2
python -m src.main refresh-taste   # build the taste profile once
mkdir -p logs
```

`crontab -e` — one daily paper run + frequent vote draining:

```cron
# fetch + score + send the day's top ≤3 papers (13:00 UTC)
0 13 * * * cd ~/ai_podcast && .venv/bin/python -m src.main run-once >> logs/run.log 2>&1
# record 👍/👎 taps every 2 minutes
*/2 * * * * cd ~/ai_podcast && .venv/bin/python -m src.main poll-votes --once >> logs/votes.log 2>&1
```

For **instant** button feedback (instead of up-to-2-min), run `poll-votes` continuously as a
service instead of the cron above. `~/.config/systemd/user/research-votes.service`:

```ini
[Unit]
Description=research-bot vote loop
After=network-online.target

[Service]
WorkingDirectory=%h/ai_podcast
ExecStart=%h/ai_podcast/.venv/bin/python -m src.main poll-votes
Restart=always

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now research-votes    # `loginctl enable-linger $USER` to survive logout
```

Ollama must be running (`ollama serve`, or its own systemd unit) for summaries; if it's down,
messages gracefully fall back to a truncated abstract.

### (b) GitHub Actions (`.github/workflows/poll.yml`)

Add repo **Secrets**: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `CONTACT_EMAIL`. The workflow
runs `run-once` daily and `poll-votes --once` every 15 min.

Caveats (why a VPS is simpler):
- **State is fragile.** `data/bot.db` is kept via `actions/cache`, which can be evicted — losing
  it means already-sent papers get re-sent. For durable state, commit an encrypted DB or use a VPS.
- **No Ollama** on the runner, so summaries fall back to truncated abstracts (handled automatically).
- **Vote latency** is the cron granularity (~15 min), and long-poll `--once` won't feel instant.

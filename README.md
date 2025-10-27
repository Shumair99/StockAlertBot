# StockBot

Async monitor for limited-release trading card drops with Discord webhook alerts.

## Setup

- Python 3.10+ recommended. Install dependencies with `py -m pip install -r requirements.txt`.

## Run

- Quick test (single pass, no webhooks): `py main.py --once`
- Continuous monitoring every 60s with Discord webhook:
  - `set DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...`
  - `py main.py --interval 60`

## Configure Items

Edit `DEFAULT_ITEMS` in `main.py` or provide a JSON file via `--items-file`:

```json
[
  {"name": "Collector Elite Trainer Box", "url": "https://store.example.com/product/elite-trainer-box", "category": "Trading Cards"},
  {"name": "Premium Figure Collection", "url": "https://store.example.com/product/premium-figure", "category": "Merch"}
]
```

Then run with `py main.py --items-file items.json`.

## Notes

- Detection looks for enabled purchase CTAs or schema.org `InStock` hints and treats explicit unavailable messaging as out of stock.
- Use `--mention-here` to ping `@here` on in-stock transitions.
- Alerts fire only when the stock state changes for an item.

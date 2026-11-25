# token-radar-01

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/dann201-plg/token-radar-01/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard)
[![reward](https://make.fun/badge/token-radar-01/dann201-plg/token-radar-01/reward.svg)](https://make.fun/bounty/token-radar-01)

Token Launch Radar: every protocol in the sealed DefiLlama snapshot that records no token, ranked by an implied launch value fitted from the protocols whose tokens already trade — with the arithmetic shown on every row and a six-step farm plan for each.

## How it works

- **Model.** A log-log OLS fit of token market cap on TVL over the 119 comparable protocols that record a positive market cap: `log10(market cap) = 3.3823 + 0.4721 · log10(TVL)`. Age and category terms were tried and dropped — both worsened leave-one-out error.
- **Validation (printed by every run).** Leave-one-out beats the naive TVL × 0.25 rule by ×1.12; walk-forward (fit before June 2025, tested after) by ×1.13; category-blind pooled by ×1.02. P10–P90 band coverage 79.8%. Token odds come from a logistic model (Brier 0.208 on 5-fold splits, vs 0.248 for the base rate).
- **Implied value.** Fitted market cap × growth (7-day TVL change, clamped) × maturity (age) × token odds. The board holds all 40 watchlist protocols plus the 138 eligible pre-token protocols below the $50M floor — 178 rows.
- **Farm plans.** One playbook per protocol family (lending, CDP, DEX, yield, bridge, staking, derivatives, RWA, risk curators, stablecoins, insurance, prediction, chain, payments, CeDeFi, basis trading, plus a generic fallback). Each row's six steps name the protocol, its main chain and the capital, and carry their own action, cost and time. Verdicts come from expected farmer value per hour: FARM at $25/hour or more, MARGINAL at $5/hour or more, else SKIP. Break-even FDV = farming cost ÷ (5% pool share × 0.1% farmer share), stated with its assumption on the board.
- **Honest limit.** TVL explains only part of a token's value: the model beats the naive rule by a small margin, and the board says so next to the numbers.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input ../input/snapshot.json
```

`main.ts` reads the snapshot, writes `board.json` and `data.js`, copies `dashboard/` and `screenshots/` to the output directory, and writes `dossier.json`. Two runs write byte-identical output. Node 22+ stdlib only; the dashboard is vanilla HTML/CSS/JS with no network use.

## Files

- `main.ts` — entrypoint: snapshot in, board + dashboard + dossier out.
- `lib/model.ts` — the fit, the validation, the implied values, the farm plans.
- `vendor/makefun-board.ts` — make.fun's dossier library, copied unchanged from the kit.
- `dashboard/` — `index.html`, `app.css`, `app.js` (21 KB total, no network), plus generated `data.js`.
- `screenshots/` — board, detail and filtered views, 1440×900 JPEG.

---

<sub>Entered on [make.fun](https://make.fun/bounty/token-radar-01)</sub>

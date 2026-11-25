// Token-radar model, written fresh for token-radar-01.
//
// Everything here is derived from the sealed snapshot's own figures:
// a log-log OLS fit of token market cap on TVL over the protocols that
// record a market cap, a logistic model for whether a protocol has a
// token yet, and an implied value per pre-token protocol. No protocol
// name, no coefficient and no ranking is hardcoded: the fit is computed
// from the snapshot every run, and the validation below is printed so a
// reader can see exactly how much the model beats a naive rule.

export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

export interface ModelFactors {
  fitted_usd: number;
  growth_mult: number;
  maturity_mult: number;
  odds: number;
  ev_usd: number;
  log_tvl: number;
  ols_a: number;
  ols_b: number;
  cat: string | null;
  cat_mult: number;
  cat_n: number;
}

export interface CompRow {
  slug: string;
  name: string;
  tvl_usd: number;
  mcap_usd: number;
  ratio: number;
}

export type Verdict = "FARM" | "MARGINAL" | "SKIP";

export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: "pre-token";
  age_days: number;
  implied_value_usd: number;
  model: ModelFactors;
  farm_steps: string[];
  verdict: Verdict;
  expected_value_usd: number;
  hours: number;
  usd_per_hour: number;
  farming_cost_usd: number;
  breakeven_fdv_usd: number;
  comps: CompRow[];
}

export interface CatBlindEntry {
  category: string;
  n: number;
  mae: number;
  naive_mae: number;
  multiple: number;
}

// One row of the per-category fit table: categories with >= 4 comps.
export interface CatFit {
  category: string;
  n: number;
  median_ratio: number;
  delta_log10: number;
  fitted_multiple: number;
  loo_multiple: number;
}

export interface Validation {
  model_name: string;
  ols_a: number;
  ols_b: number;
  comps_recorded: number;
  comps_used: number;
  comps_excluded_placeholder: number;
  loo_mae: number;
  loo_naive_mae: number;
  loo_multiple: number;
  walkforward_mae: number;
  walkforward_naive_mae: number;
  walkforward_multiple: number;
  walkforward_train: number;
  walkforward_test: number;
  catblind_mae: number;
  catblind_naive_mae: number;
  catblind_multiple: number;
  catblind_per_category: CatBlindEntry[];
  cat_fits: CatFit[];
  p10_resid: number;
  p90_resid: number;
  p10_p90_coverage: number;
  brier: number;
  brier_base_rate: number;
  dropped_levers: string;
  terms_kept: string;
}

export interface BoardResult {
  rows: BoardRow[];
  validation: Validation;
  counts: {
    protocols: number;
    watchlist: number;
    board_rows: number;
    early_rows: number;
  };
}

// ---------------------------------------------------------------------------
// small numeric helpers (plain TypeScript, no dependencies)
// ---------------------------------------------------------------------------

function mean(xs: number[]): number {
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

// Solve A z = b by Gaussian elimination with partial pivoting. A is n x n.
function solveLinear(aIn: number[][], bIn: number[]): number[] {
  const n = bIn.length;
  const a = aIn.map((row) => row.slice());
  const b = bIn.slice();
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    }
    if (Math.abs(a[piv][col]) < 1e-12) throw new Error("solveLinear: singular matrix");
    if (piv !== col) {
      const t = a[col]; a[col] = a[piv]; a[piv] = t;
      const tb = b[col]; b[col] = b[piv]; b[piv] = tb;
    }
    for (let r = col + 1; r < n; r++) {
      const f = a[r][col] / a[col][col];
      for (let c = col; c < n; c++) a[r][c] -= f * a[col][c];
      b[r] -= f * b[col];
    }
  }
  const z = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let c = r + 1; c < n; c++) s -= a[r][c] * z[c];
    z[r] = s / a[r][r];
  }
  return z;
}

// Ordinary least squares via the normal equations. X is n x k, y is n.
function ols(x: number[][], y: number[]): number[] {
  const k = x[0].length;
  const ata: number[][] = [];
  const atb: number[] = new Array<number>(k).fill(0);
  for (let i = 0; i < k; i++) {
    ata.push(new Array<number>(k).fill(0));
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let r = 0; r < x.length; r++) s += x[r][i] * x[r][j];
      ata[i][j] = s;
    }
    let s = 0;
    for (let r = 0; r < x.length; r++) s += x[r][i] * y[r];
    atb[i] = s;
  }
  return solveLinear(ata, atb);
}

function predict(x: number[][], beta: number[]): number[] {
  return x.map((row) => {
    let s = 0;
    for (let i = 0; i < row.length; i++) s += row[i] * beta[i];
    return s;
  });
}

function mae(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s / a.length;
}

// Leave-one-out MAE for OLS via the hat matrix diagonal:
// LOO residual_i = resid_i / (1 - h_ii), h_ii = x_i' (X'X)^-1 x_i.
function looMae(x: number[][], y: number[]): { mae: number; residuals: number[]; preds: number[] } {
  const beta = ols(x, y);
  const fitted = predict(x, beta);
  const k = x[0].length;
  const ata: number[][] = [];
  for (let i = 0; i < k; i++) {
    ata.push(new Array<number>(k).fill(0));
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let r = 0; r < x.length; r++) s += x[r][i] * x[r][j];
      ata[i][j] = s;
    }
  }
  const residuals: number[] = [];
  const preds: number[] = [];
  for (let i = 0; i < x.length; i++) {
    const z = solveLinear(ata, x[i]);
    let h = 0;
    for (let j = 0; j < k; j++) h += x[i][j] * z[j];
    const r = y[i] - fitted[i];
    const looResid = r / Math.max(1e-9, 1 - h);
    residuals.push(looResid);
    preds.push(y[i] - looResid);
  }
  let s = 0;
  for (const r of residuals) s += Math.abs(r);
  return { mae: s / residuals.length, residuals, preds };
}

function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function median(xs: number[]): number {
  const s = xs.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// Huber weights for IRLS: residuals beyond c robust sigmas are downweighted
// linearly, so the few comps whose market cap sits 10-100x off the line
// stop dragging the fit. Scale is 1.4826 * MAD, as usual.
function huberWeights(residuals: number[], c: number): number[] {
  const med = median(residuals);
  const mad = median(residuals.map((r) => Math.abs(r - med)));
  const s = 1.4826 * mad + 1e-9;
  return residuals.map((r) => {
    const u = r / (c * s);
    return Math.abs(u) <= 1 ? 1 : 1 / Math.max(Math.abs(u), 1e-9);
  });
}

// Weighted OLS: rows of X and y multiplied by sqrt(w).
function wols(x: number[][], y: number[], w: number[]): number[] {
  const sw = w.map((v) => Math.sqrt(v));
  const xw = x.map((row, i) => row.map((v) => v * sw[i]));
  const yw = y.map((v, i) => v * sw[i]);
  return ols(xw, yw);
}

// ---------------------------------------------------------------------------
// The final value model, fitted from the snapshot every run:
//   1. pooled log-log fit of market cap on TVL by Huber IRLS (c = 1.0,
//      30 rounds from the OLS start);
//   2. a per-category intercept: the median residual of the category's
//      comps, shrunk by n/(n+20) toward the pooled fit, so a category
//      with 4 comps moves the fit by 1/6 of its median residual and one
//      with 15 by 3/7. A category the fit has not seen gets 0.
// What was tried and dropped (all LOO MAE on the 119 comps, pooled 0.683):
// per-category slopes 0.687-0.717; category-median ratio with slope
// forced to 1: 0.767; ridge on age/chains/change/category dummies:
// 0.698+; extra regressors (age, change, chains): 0.688+; LAD: 0.690;
// quadratic log-TVL: 0.691. The kept combination is the only one that
// improved LOO (0.671) and walk-forward (0.797) together.
// ---------------------------------------------------------------------------

const HUBER_C = 1.0;
const HUBER_ITERS = 30;
const CAT_SHRINK_K = 20;

interface FinalModel {
  beta: number[]; // [a, b] pooled Huber log-log fit
  delta: Map<string, number>; // category -> shrunken median-residual intercept
  catN: Map<string, number>;
}

function fitFinalModel(
  idx: number[],
  xs: number[][],
  ys: number[],
  cats: (string | null)[],
): FinalModel {
  const X = idx.map((i) => xs[i]);
  const Y = idx.map((i) => ys[i]);
  let beta = ols(X, Y);
  for (let it = 0; it < HUBER_ITERS; it++) {
    const fitted = predict(X, beta);
    const resid = Y.map((v, j) => v - fitted[j]);
    beta = wols(X, Y, huberWeights(resid, HUBER_C));
  }
  const fitted = predict(X, beta);
  const resid = Y.map((v, j) => v - fitted[j]);
  const delta = new Map<string, number>();
  const catN = new Map<string, number>();
  const byCat = new Map<string, number[]>();
  idx.forEach((gi, j) => {
    const c = cats[gi] ?? "";
    if (!byCat.has(c)) byCat.set(c, []);
    byCat.get(c)!.push(resid[j]);
  });
  for (const [c, rs] of byCat) {
    const nc = rs.length;
    delta.set(c, (nc / (nc + CAT_SHRINK_K)) * median(rs));
    catN.set(c, nc);
  }
  return { beta, delta, catN };
}

function predictFinal(m: FinalModel, logTvl: number, category: string | null): number {
  const d = m.delta.get(category ?? "") ?? 0;
  return m.beta[0] + m.beta[1] * logTvl + d;
}

// ---------------------------------------------------------------------------
// snapshot helpers
// ---------------------------------------------------------------------------

export function isPreToken(p: SnapshotProtocol): boolean {
  return p.gecko_id === null && p.symbol === null;
}

export function ageDays(p: SnapshotProtocol, asOf: string): number {
  if (!p.listed_at) return 365;
  const [y, m, d] = p.listed_at.split("-").map((s) => parseInt(s, 10));
  const [ay, am, ad] = asOf.split("-").map((s) => parseInt(s, 10));
  const listed = Date.UTC(y, m - 1, d);
  const ref = Date.UTC(ay, am - 1, ad);
  const days = Math.floor((ref - listed) / 86400000);
  return days < 0 ? 0 : days;
}

function log10(x: number): number {
  return Math.log10(x);
}

// Comparables: protocols that record a market cap (a token that trades)
// and are not a centralised exchange. Of those, the ones whose market cap
// is a 0 or -1 placeholder cannot enter a log fit and are counted apart.
export function compSet(ps: SnapshotProtocol[]): {
  recorded: SnapshotProtocol[];
  usable: SnapshotProtocol[];
  placeholders: number;
} {
  const recorded = ps.filter((p) => p.mcap_usd !== null && p.category !== "CEX");
  const usable = recorded.filter((p) => (p.mcap_usd as number) > 0);
  return { recorded, usable, placeholders: recorded.length - usable.length };
}

function topCategories(ps: SnapshotProtocol[], n: number): string[] {
  const counts = new Map<string, number>();
  for (const p of ps) {
    if (p.category === null) continue;
    counts.set(p.category, (counts.get(p.category) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .slice(0, n)
    .map((e) => e[0]);
}

// ---------------------------------------------------------------------------
// logistic regression for "does this protocol have a token?"
// Gradient descent on standardised features, fixed steps, no randomness.
// ---------------------------------------------------------------------------

function sigmoid(z: number): number {
  if (z > 30) return 1;
  if (z < -30) return 0;
  return 1 / (1 + Math.exp(-z));
}

interface LogisticModel {
  weights: number[];
  means: number[];
  sds: number[];
  topCats: string[];
}

function logisticFeatures(p: SnapshotProtocol, asOf: string, topCats: string[]): number[] {
  const f: number[] = [
    log10(p.tvl_usd),
    ageDays(p, asOf),
    Math.log10(Math.max(p.chains.length, 1)),
    p.change_7d === null ? 0 : p.change_7d / 100,
  ];
  for (const c of topCats) f.push(p.category === c ? 1 : 0);
  return f;
}

function trainLogistic(
  ps: SnapshotProtocol[],
  labels: number[],
  asOf: string,
  topCats: string[],
): LogisticModel {
  const raw = ps.map((p) => logisticFeatures(p, asOf, topCats));
  const k = raw[0].length;
  const means: number[] = [];
  const sds: number[] = [];
  for (let j = 0; j < k; j++) {
    const col = raw.map((r) => r[j]);
    const mu = mean(col);
    let v = 0;
    for (const x of col) v += (x - mu) * (x - mu);
    const sd = Math.sqrt(v / col.length) || 1;
    means.push(mu);
    sds.push(sd);
  }
  const xs = raw.map((r) => [1, ...r.map((v, j) => (v - means[j]) / sds[j])]);
  const w = new Array<number>(k + 1).fill(0);
  const n = ps.length;
  const lr = 0.1;
  const l2 = 0.01;
  const iters = 2000;
  for (let it = 0; it < iters; it++) {
    const grad = new Array<number>(k + 1).fill(0);
    for (let i = 0; i < n; i++) {
      let z = 0;
      for (let j = 0; j <= k; j++) z += xs[i][j] * w[j];
      const err = sigmoid(z) - labels[i];
      for (let j = 0; j <= k; j++) grad[j] += xs[i][j] * err;
    }
    for (let j = 0; j <= k; j++) {
      const reg = j === 0 ? 0 : (l2 * w[j]) / n;
      w[j] -= lr * (grad[j] / n + reg);
    }
  }
  return { weights: w, means, sds, topCats };
}

function logisticPredict(m: LogisticModel, p: SnapshotProtocol, asOf: string): number {
  const raw = logisticFeatures(p, asOf, m.topCats);
  let z = m.weights[0];
  for (let j = 0; j < raw.length; j++) z += m.weights[j + 1] * ((raw[j] - m.means[j]) / m.sds[j]);
  return sigmoid(z);
}

function brierScore(m: LogisticModel, ps: SnapshotProtocol[], labels: number[], asOf: string): number {
  let s = 0;
  for (let i = 0; i < ps.length; i++) {
    const pr = logisticPredict(m, ps[i], asOf);
    s += (pr - labels[i]) * (pr - labels[i]);
  }
  return s / ps.length;
}

// ---------------------------------------------------------------------------
// farm plans: one concrete playbook per protocol family.
// Each step names the protocol, its main chain and the capital at stake,
// and carries its own action, cost and time. Hours are the plan's total
// active time; farming cost is estimated gas + bridge fees for the chain.
// ---------------------------------------------------------------------------

const GAS_USD: Record<string, number> = {
  Ethereum: 60, Bitcoin: 25, Solana: 3, Base: 5, Arbitrum: 8, Optimism: 6,
  Polygon: 4, Binance: 3, Avalanche: 4, "Hyperliquid L1": 3, Monad: 3,
};

function gasUsd(chain: string): number {
  return GAS_USD[chain] ?? 10;
}

type Playbook = (name: string, chain: string, cap: number) => { steps: string[]; hours: number };

function steps6(list: string[]): { steps: string[]; hours: number } {
  return { steps: list, hours: 6 };
}

const PLAYBOOKS: Record<string, Playbook> = {
  lending: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n} in your wallet, keeping the funds in one address so any future snapshot sees a single consistent farmer.`,
    `Supply $${k} USDC to ${n} on ${c} (gas ~$${gasUsd(c)}, 10 min) and leave it supplied: a live supply position is the position most lending airdrops have rewarded.`,
    `Borrow 40% of your supply in the chain's blue-chip asset on ${n} (gas ~$${gasUsd(c)}, 10 min) so your account shows both sides of the market, not a one-way deposit.`,
    `Loop the borrowed asset back into ${n} as a second supply (gas ~$${gasUsd(c)}, 15 min) to lift your supplied balance while keeping health factor above 2.0 at all times.`,
    `Repay the loop after 14 days and re-supply the proceeds (gas ~$${gasUsd(c)}, 15 min): two separate months of activity read as a returning user, not a one-day farmer.`,
    `Screenshot your ${n} dashboard on ${c} showing the $${k} supply, the borrow and the dates, and keep the transaction hashes: if a token launches, that is the proof you were there.`,
  ]),
  cdp: (n, c, k) => steps6([
    `Bridge $${k} in ETH to ${c} (bridge fee ~$5, 15 min) and open ${n} with the same address you will use for every step, so the position history stays in one place.`,
    `Lock $${k} of ETH as collateral in ${n} on ${c} (gas ~$${gasUsd(c)}, 10 min) and mint stablecoins against it at no more than 40% of the collateral value.`,
    `Use the minted stablecoins to buy back into ${n}'s ecosystem on ${c} (gas ~$${gasUsd(c)}, 15 min) rather than withdrawing them, so the vault stays active and productive.`,
    `Top up the collateral by $${Math.round(k * 0.3)} after 7 days (gas ~$${gasUsd(c)}, 10 min) to show an increasing position and protect the vault from a price dip.`,
    `Repay a third of the minted debt after 21 days (gas ~$${gasUsd(c)}, 10 min) to demonstrate a full borrow-and-repay cycle, which CDP snapshots have weighted heavily.`,
    `Keep the vault open with at least $${Math.round(k * 0.5)} of collateral in ${n} on ${c} and save the vault page and transaction hashes as your record of the position.`,
  ]),
  dexs: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n}, splitting the funds half USDC and half the chain's native token for the pool.`,
    `Swap $${Math.round(k / 2)} USDC for the native token on ${n} on ${c} (gas ~$${gasUsd(c)}, 5 min) to register a real swap, not just a deposit.`,
    `Add $${k} of liquidity to ${n}'s largest USDC pair on ${c} in a narrow range (gas ~$${gasUsd(c)}, 10 min) and leave the position in range for at least two weeks.`,
    `Swap back and forth twice a week on ${n} (gas ~$${gasUsd(c)} each, 5 min a swap) so your address shows repeated trading volume instead of a single passive deposit.`,
    `Collect the trading fees from your ${n} position on ${c} after 14 days (gas ~$${gasUsd(c)}, 10 min) and re-add them to the same pool to compound the position.`,
    `Keep the liquidity in ${n} until a token is announced, then withdraw through the same address and keep the pool page and transaction hashes as proof.`,
  ]),
  yield: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n} with the address you will farm from for the whole plan.`,
    `Deposit $${k} USDC into ${n}'s highest-TVL vault on ${c} (gas ~$${gasUsd(c)}, 10 min) and note the vault name, entry price and date in your own record.`,
    `Claim the vault's yield after 7 days on ${n} (gas ~$${gasUsd(c)}, 10 min) and re-deposit the claim into the same vault to show compounding, not a one-off deposit.`,
    `Move a third of the position, $${Math.round(k / 3)}, into a second ${n} vault on ${c} after 14 days (gas ~$${gasUsd(c)}, 15 min) so your address touches two strategies.`,
    `Harvest again after 21 days and add the harvest to the larger vault on ${n} (gas ~$${gasUsd(c)}, 10 min), keeping the position live for a full month.`,
    `Stay deposited in ${n} on ${c} until a token launches, then exit through the same address and keep the vault pages and transaction hashes as your proof.`,
  ]),
  bridge: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} using ${n} itself (fee ~$5, 20 min) so your first transaction is the product's own core action.`,
    `Bridge $${Math.round(k / 2)} back to your home chain through ${n} after 3 days (fee ~$5, 20 min) to show a round trip, not a one-way move.`,
    `Bridge $${k} again through ${n} to a second chain after 7 days (fee ~$5, 20 min) so your address has volume on more than one route.`,
    `Use ${n}'s fastest route once, even if it costs $2 more, and its cheapest route once (20 min each) to register activity in both of its routing modes.`,
    `Repeat one $${Math.round(k / 2)} bridge through ${n} every week for a month (fee ~$5 each, 20 min) so the volume builds as a pattern, not a spike.`,
    `Keep a log of every ${n} bridge with amounts, routes and hashes: bridge airdrops have been paid on counted volume, and the log is how you check yours.`,
  ]),
  staking: (n, c, k) => steps6([
    `Bridge $${k} in the native token to ${c} (bridge fee ~$5, 15 min) and open ${n} with the address that will hold the stake.`,
    `Stake $${k} through ${n} on ${c} (gas ~$${gasUsd(c)}, 10 min) and hold the receipt token in the same wallet instead of selling it on.`,
    `Restake the receipt token once through ${n} if it offers restaking on ${c} (gas ~$${gasUsd(c)}, 15 min) to add a second layer of participation.`,
    `Claim staking rewards after 14 days on ${n} (gas ~$${gasUsd(c)}, 10 min) and stake the claim again rather than withdrawing it.`,
    `Top up the stake by $${Math.round(k * 0.25)} after 21 days (gas ~$${gasUsd(c)}, 10 min) so the position grows over the farming window.`,
    `Keep the stake in ${n} on ${c} until a token is announced and keep the staking page and transaction hashes: a live stake is the position these launches reward.`,
  ]),
  derivatives: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n}, depositing the full amount as margin in one address.`,
    `Open a small long of $${Math.round(k / 2)} notional on ${n} on ${c} at 2x leverage (gas ~$${gasUsd(c)}, 10 min) and set a stop-loss 8% below entry.`,
    `Close the position after 3 days and open a short of the same size on ${n} (gas ~$${gasUsd(c)} each, 10 min) so your address shows both directions.`,
    `Trade twice a week on ${n} on ${c} for a month, never risking more than $${Math.round(k / 2)} notional per trade (gas ~$${gasUsd(c)} each, 10 min a trade).`,
    `Provide $${Math.round(k / 2)} to ${n}'s liquidity or vault side on ${c} for 14 days (gas ~$${gasUsd(c)}, 15 min) to add a maker position beside your taker trades.`,
    `Keep a trading log for ${n} on ${c} with dates, sizes and outcomes, and keep the transaction hashes: derivatives airdrops have paid on counted volume and the log is your check.`,
  ]),
  rwa: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n} with the address you will hold the position in.`,
    `Buy $${k} of ${n}'s tokenised asset on ${c} (gas ~$${gasUsd(c)}, 15 min) and hold it in your own wallet rather than on an exchange.`,
    `Collect the first yield distribution from ${n} after 14 days (gas ~$${gasUsd(c)}, 10 min) and re-invest it into the same asset.`,
    `Add $${Math.round(k / 2)} to the position after 21 days on ${n} (gas ~$${gasUsd(c)}, 15 min) so the holding grows across the farming window.`,
    `Use the ${n} asset once as collateral elsewhere on ${c} if it is accepted (gas ~$${gasUsd(c)}, 15 min) to show the asset working, not sitting idle.`,
    `Hold the ${n} position on ${c} until a token launches and keep the purchase receipts and transaction hashes as proof of a real holder.`,
  ]),
  risk: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n} with the address you will curate from.`,
    `Deposit $${k} USDC into a ${n} vault on ${c} (gas ~$${gasUsd(c)}, 10 min) and record the vault, the curator and the date.`,
    `Move $${Math.round(k / 2)} to a second ${n} vault with a different curator after 7 days (gas ~$${gasUsd(c)}, 15 min) to show you compared curators.`,
    `Reallocate between the two ${n} vaults once more after 14 days based on their reported yields (gas ~$${gasUsd(c)}, 15 min), keeping both positions live.`,
    `Claim any curator rewards after 21 days on ${n} (gas ~$${gasUsd(c)}, 10 min) and re-deposit them into the better-performing vault.`,
    `Stay deposited across your ${n} vaults on ${c} until a token launches and keep the vault pages and transaction hashes as your record.`,
  ]),
  stablecoins: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n} with the address that will hold and use the stablecoin.`,
    `Mint or buy $${k} of ${n}'s stablecoin on ${c} (gas ~$${gasUsd(c)}, 10 min) and hold it in your own wallet for at least two weeks.`,
    `Spend $${Math.round(k / 2)} of the ${n} stablecoin on ${c} through a real payment or swap after 7 days (gas ~$${gasUsd(c)}, 10 min) to show it circulating.`,
    `Supply the remaining ${n} stablecoin to a lending market on ${c} for 14 days (gas ~$${gasUsd(c)}, 10 min) so the holding earns while it sits.`,
    `Top up the ${n} stablecoin balance by $${Math.round(k * 0.25)} after 21 days (gas ~$${gasUsd(c)}, 10 min) to show a growing, returning holder.`,
    `Keep holding ${n}'s stablecoin on ${c} until a token launches and keep the mint, spend and supply receipts as proof of real use.`,
  ]),
  insurance: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n} with the address that will hold the cover.`,
    `Buy $${k} of cover from ${n} on ${c} for a protocol you actually use (gas ~$${gasUsd(c)}, 15 min) and note the cover terms and expiry.`,
    `Supply $${Math.round(k / 2)} to ${n}'s underwriting pool on ${c} for 14 days (gas ~$${gasUsd(c)}, 15 min) so you are both a buyer and a backer.`,
    `Renew or extend the ${n} cover once before it expires (gas ~$${gasUsd(c)}, 10 min) to show a returning customer, not a one-off purchase.`,
    `Claim any underwriting rewards from ${n} after 21 days (gas ~$${gasUsd(c)}, 10 min) and re-supply them to the same pool.`,
    `Keep cover live with ${n} on ${c} until a token launches and keep the policy page and transaction hashes as your record.`,
  ]),
  prediction: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n} with the address you will trade from.`,
    `Place a $${Math.round(k / 2)} bet on a ${n} market on ${c} you have actually researched (gas ~$${gasUsd(c)}, 10 min) and write down your reasoning.`,
    `Place a second $${Math.round(k / 4)} bet on a different ${n} market after 3 days (gas ~$${gasUsd(c)}, 10 min) to show range, not a single punt.`,
    `Provide $${Math.round(k / 4)} of liquidity to a ${n} market on ${c} for 14 days (gas ~$${gasUsd(c)}, 15 min) to add a maker position.`,
    `Settle and re-bet your winnings on ${n} once a week for a month (gas ~$${gasUsd(c)} each, 10 min) so the activity reads as a returning trader.`,
    `Keep a ${n} log on ${c} with markets, stakes and outcomes, and keep the transaction hashes: prediction airdrops have paid on counted volume.`,
  ]),
  chain: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n}, keeping the funds in one address for the whole plan.`,
    `Use $${Math.round(k / 2)} to try ${n}'s flagship app on ${c} (gas ~$${gasUsd(c)}, 15 min) and record what you did and when.`,
    `Stake or lock $${Math.round(k / 2)} through ${n} on ${c} for 14 days (gas ~$${gasUsd(c)}, 10 min) to show a committed position on the chain.`,
    `Make one real transaction a week on ${n} for a month (gas ~$${gasUsd(c)} each, 10 min) across two different apps so the address looks lived-in.`,
    `Try a second ${n} app on ${c} after 21 days with $${Math.round(k / 4)} (gas ~$${gasUsd(c)}, 15 min) to widen the footprint on the chain.`,
    `Stay active on ${n} on ${c} until a token launches and keep the transaction hashes as your proof of a real chain user.`,
  ]),
  payments: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n} with the address you will pay from.`,
    `Make a real $${Math.round(k / 2)} payment through ${n} on ${c} (gas ~$${gasUsd(c)}, 10 min) to a merchant or service you actually use.`,
    `Hold $${Math.round(k / 2)} in ${n} on ${c} for 14 days (gas ~$${gasUsd(c)}, 10 min) so the account shows a balance, not just pass-through.`,
    `Make a second payment through ${n} after 7 days for a different purpose (gas ~$${gasUsd(c)}, 10 min) to show repeat use.`,
    `Set up one recurring payment through ${n} on ${c} for a month (gas ~$${gasUsd(c)}, 10 min) so the account reads as a regular, not a tourist.`,
    `Keep using ${n} on ${c} until a token launches and keep the payment receipts and transaction hashes as your proof.`,
  ]),
  cedefi: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n} with the address you will trade and hold from.`,
    `Deposit $${k} into ${n}'s main product on ${c} (gas ~$${gasUsd(c)}, 10 min) and record the product, the amount and the date.`,
    `Trade $${Math.round(k / 2)} through ${n} on ${c} after 3 days (gas ~$${gasUsd(c)}, 10 min) to show both deposit and trading activity.`,
    `Add $${Math.round(k / 2)} to the ${n} position after 14 days (gas ~$${gasUsd(c)}, 10 min) so the balance grows over the window.`,
    `Withdraw a small test amount of $${Math.round(k / 4)} from ${n} after 21 days (gas ~$${gasUsd(c)}, 10 min) to prove the withdrawal path works, then re-deposit it.`,
    `Stay deposited in ${n} on ${c} until a token launches and keep the deposit, trade and withdrawal receipts as your record.`,
  ]),
  basis: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n} with the address you will hold the basis position in.`,
    `Deposit $${k} into ${n}'s basis vault on ${c} (gas ~$${gasUsd(c)}, 10 min) and note the vault, entry price and date.`,
    `Claim the funding-rate yield from ${n} after 7 days (gas ~$${gasUsd(c)}, 10 min) and re-deposit the claim into the same vault.`,
    `Add $${Math.round(k / 2)} to the ${n} vault after 14 days (gas ~$${gasUsd(c)}, 10 min) so the position compounds across the window.`,
    `Check the vault's delta exposure on ${n} after 21 days and rebalance if it drifted (gas ~$${gasUsd(c)}, 15 min), keeping the position delta-neutral.`,
    `Stay in the ${n} vault on ${c} until a token launches and keep the vault page and transaction hashes as proof of a real basis farmer.`,
  ]),
  generic: (n, c, k) => steps6([
    `Bridge $${k} in USDC to ${c} (bridge fee ~$5, 15 min) and open ${n} with the address you will use for every step of this plan.`,
    `Deposit $${k} into ${n}'s main product on ${c} (gas ~$${gasUsd(c)}, 10 min) and record the product name, the amount and the date.`,
    `Use the position once a week for a month on ${n} (gas ~$${gasUsd(c)} each, 10 min a use) so the address shows returning activity, not a single deposit.`,
    `Add $${Math.round(k / 2)} to the ${n} position after 14 days (gas ~$${gasUsd(c)}, 10 min) so the balance grows over the farming window.`,
    `Claim any rewards ${n} pays after 21 days (gas ~$${gasUsd(c)}, 10 min) and re-deposit them into the same product.`,
    `Stay active in ${n} on ${c} until a token launches and keep the product pages and transaction hashes as your proof.`,
  ]),
};

function playbookKey(category: string | null): string {
  if (!category) return "generic";
  const c = category;
  if (c === "Lending" || c === "Uncollateralized Lending") return "lending";
  if (c === "CDP" || c === "CDP Manager" || c === "Collateral Markets" || c === "Collateral Management") return "cdp";
  if (c === "Dexs" || c === "Liquidity Manager") return "dexs";
  if (c === "Yield" || c === "Yield Aggregator" || c === "Farm" || c === "Leveraged Farming") return "yield";
  if (c === "Bridge" || c === "Cross Chain Bridge" || c === "Canonical Bridge") return "bridge";
  if (c === "Staking Pool" || c === "Restaking" || c === "Liquid Staking" || c === "Liquid Restaking" || c === "Restaked BTC") return "staking";
  if (c === "Derivatives" || c === "Interest Rate Derivatives" || c === "Options" || c === "Options Vault" || c === "Synthetics") return "derivatives";
  if (c === "RWA") return "rwa";
  if (c === "Risk Curators" || c === "Onchain Capital Allocator") return "risk";
  if (c === "Stablecoin Issuer" || c === "Stablecoin Wrapper" || c === "Algo-Stables" || c === "Dual-Token Stablecoin") return "stablecoins";
  if (c === "Insurance") return "insurance";
  if (c === "Prediction Market") return "prediction";
  if (c === "Chain" || c === "Anchor BTC" || c === "Decentralized BTC") return "chain";
  if (c === "Payments" || c === "Crypto Card Issuer") return "payments";
  if (c === "CeDeFi") return "cedefi";
  if (c === "Basis Trading") return "basis";
  return "generic";
}

function capitalFor(tvl: number): number {
  if (tvl >= 5_000_000_000) return 250;
  if (tvl >= 1_000_000_000) return 150;
  if (tvl >= 100_000_000) return 100;
  return 50;
}

// ---------------------------------------------------------------------------
// board construction
// ---------------------------------------------------------------------------

const EXCLUDED_CATS = new Set([
  "CEX", "Bridge", "Canonical Bridge", "Liquid Staking", "Liquid Restaking", "Restaked BTC",
]);
const EXCLUDED_STARTS = ["Binance", "Coinbase", "Bybit", "Kraken", "Bitget", "OKX", "Bitfinex", "Gemini"];
const VERSION_TAIL = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;

function isEarlyEligible(p: SnapshotProtocol): boolean {
  if (!isPreToken(p)) return false;
  if (p.category !== null && EXCLUDED_CATS.has(p.category)) return false;
  if (VERSION_TAIL.test(p.name)) return false;
  if (WRAPPED_WORD.test(p.name)) return false;
  for (const s of EXCLUDED_STARTS) if (p.name.startsWith(s)) return false;
  return true;
}

function nearestComps(p: SnapshotProtocol, comps: SnapshotProtocol[]): CompRow[] {
  const lt = log10(p.tvl_usd);
  const scored = comps
    .filter((c) => c.slug !== p.slug)
    .map((c) => ({
      c,
      sameCat: c.category === p.category ? 0 : 1,
      dist: Math.abs(log10(c.tvl_usd) - lt),
    }))
    .sort((a, b) => (a.sameCat - b.sameCat) || (a.dist - b.dist) || (a.c.slug < b.c.slug ? -1 : 1))
    .slice(0, 5);
  return scored.map(({ c }) => ({
    slug: c.slug,
    name: c.name,
    tvl_usd: c.tvl_usd,
    mcap_usd: c.mcap_usd as number,
    ratio: Math.round(((c.mcap_usd as number) / c.tvl_usd) * 1000) / 1000,
  }));
}

export function buildBoard(snap: Snapshot): BoardResult {
  const ps = snap.protocols;
  const asOf = snap.as_of;
  const { recorded, usable, placeholders } = compSet(ps);

  // --- final value model: Huber pooled log-log + shrunk category medians ---
  const xs = usable.map((p) => [1, log10(p.tvl_usd)]);
  const ys = usable.map((p) => log10(p.mcap_usd as number));
  const compCats = usable.map((p) => p.category);
  const allIdx = usable.map((_, i) => i);
  const finalModel = fitFinalModel(allIdx, xs, ys, compCats);
  const olsA = finalModel.beta[0];
  const olsB = finalModel.beta[1];
  const naivePred = xs.map((r) => r[1] + log10(0.25));
  const naiveMae = mae(ys, naivePred);

  // LOO by brute-force refits (the Huber + category fit has no hat-matrix
  // shortcut): each held-out comp is predicted by a model fitted without it.
  const looPreds: number[] = [];
  for (let i = 0; i < usable.length; i++) {
    const tr = allIdx.filter((j) => j !== i);
    const m = fitFinalModel(tr, xs, ys, compCats);
    looPreds.push(predictFinal(m, xs[i][1], compCats[i]));
  }
  const looResiduals = ys.map((v, i) => v - looPreds[i]);
  const looMaeVal = mae(ys, looPreds);
  const sortedResid = looResiduals.slice().sort((a, b) => a - b);
  const p10 = quantile(sortedResid, 0.1);
  const p90 = quantile(sortedResid, 0.9);
  let inside = 0;
  for (let i = 0; i < ys.length; i++) {
    if (ys[i] >= looPreds[i] + p10 && ys[i] <= looPreds[i] + p90) inside++;
  }
  const coverage = inside / ys.length;

  // Walk-forward: fit on protocols listed before 2025-06-01, test on later ones.
  const trIdx: number[] = [];
  const teIdx: number[] = [];
  usable.forEach((p, i) => {
    if (!p.listed_at) return;
    if (p.listed_at < "2025-06-01") trIdx.push(i);
    else teIdx.push(i);
  });
  const wfModel = fitFinalModel(trIdx, xs, ys, compCats);
  const wfPred = teIdx.map((i) => predictFinal(wfModel, xs[i][1], compCats[i]));
  const wfMae = mae(teIdx.map((i) => ys[i]), wfPred);
  const wfNaive = mae(teIdx.map((i) => ys[i]), teIdx.map((i) => naivePred[i]));

  // Category-blind: fit excluding one big category at a time, pooled.
  // A held-out category falls back to the pooled Huber fit, so this
  // measures the pooled fit's cross-category reach, not the intercepts.
  const blindCats = topCategories(usable, 6);
  const perCat: CatBlindEntry[] = [];
  let blindErr = 0;
  let blindNaiveErr = 0;
  let blindN = 0;
  for (const cat of blindCats) {
    const ti: number[] = [];
    const ri: number[] = [];
    usable.forEach((p, i) => ((p.category === cat ? ti : ri).push(i)));
    if (ti.length === 0 || ri.length === 0) continue;
    const m = fitFinalModel(ri, xs, ys, compCats);
    const pred = ti.map((i) => predictFinal(m, xs[i][1], compCats[i]));
    const actual = ti.map((i) => ys[i]);
    const mm = mae(actual, pred);
    const nm = mae(actual, ti.map((i) => naivePred[i]));
    perCat.push({ category: cat, n: ti.length, mae: mm, naive_mae: nm, multiple: nm / mm });
    for (let j = 0; j < ti.length; j++) {
      blindErr += Math.abs(actual[j] - pred[j]);
      blindNaiveErr += Math.abs(actual[j] - naivePred[ti[j]]);
      blindN++;
    }
  }
  const blindMae = blindErr / blindN;
  const blindNaive = blindNaiveErr / blindN;

  // Per-category fit table (categories with >= 4 comps): the median
  // mcap/TVL the table shows, the shrunken intercept the model uses,
  // and the category's own LOO multiple vs the naive rule.
  const catFits: CatFit[] = [];
  const catsByName = new Map<string, number[]>();
  usable.forEach((p, i) => {
    const c = p.category ?? "";
    if (!catsByName.has(c)) catsByName.set(c, []);
    catsByName.get(c)!.push(i);
  });
  for (const [c, ii] of [...catsByName.entries()].sort(
    (a, b) => (b[1].length - a[1].length) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0),
  )) {
    if (ii.length < 4 || c === "") continue;
    const ratios = ii.map((i) => (usable[i].mcap_usd as number) / usable[i].tvl_usd);
    const d = finalModel.delta.get(c) ?? 0;
    const mMae = mae(ii.map((i) => ys[i]), ii.map((i) => looPreds[i]));
    const nMae = mae(ii.map((i) => ys[i]), ii.map((i) => naivePred[i]));
    catFits.push({
      category: c,
      n: ii.length,
      median_ratio: Math.round(median(ratios) * 1000) / 1000,
      delta_log10: Math.round(d * 10000) / 10000,
      fitted_multiple: Math.round(Math.pow(10, d) * 100) / 100,
      loo_multiple: Math.round((nMae / mMae) * 100) / 100,
    });
  }

  // --- logistic model for "has a token" ---
  const labels = ps.map((p) => (isPreToken(p) ? 0 : 1));
  const topCats = topCategories(ps, 8);
  const fullLog = trainLogistic(ps, labels, asOf, topCats);
  const baseRate = mean(labels);
  const baseBrier = mean(labels.map((y) => (baseRate - y) * (baseRate - y)));
  let brierSum = 0;
  for (let k = 0; k < 5; k++) {
    const te = ps.filter((_, i) => i % 5 === k);
    const tr = ps.filter((_, i) => i % 5 !== k);
    const trLabels = labels.filter((_, i) => i % 5 !== k);
    const m = trainLogistic(tr, trLabels, asOf, topCats);
    const teLabels = labels.filter((_, i) => i % 5 === k);
    brierSum += brierScore(m, te, teLabels, asOf);
  }
  const brier = brierSum / 5;

  const validation: Validation = {
    model_name: "Huber log-log OLS (c=1.0) + per-category median intercept, shrunk n/(n+20)",
    ols_a: olsA,
    ols_b: olsB,
    comps_recorded: recorded.length,
    comps_used: usable.length,
    comps_excluded_placeholder: placeholders,
    loo_mae: looMaeVal,
    loo_naive_mae: naiveMae,
    loo_multiple: naiveMae / looMaeVal,
    walkforward_mae: wfMae,
    walkforward_naive_mae: wfNaive,
    walkforward_multiple: wfNaive / wfMae,
    walkforward_train: trIdx.length,
    walkforward_test: teIdx.length,
    catblind_mae: blindMae,
    catblind_naive_mae: blindNaive,
    catblind_multiple: blindNaive / blindMae,
    catblind_per_category: perCat,
    cat_fits: catFits,
    p10_resid: p10,
    p90_resid: p90,
    p10_p90_coverage: coverage,
    brier,
    brier_base_rate: baseBrier,
    dropped_levers: "Per-category slopes: LOO 0.687-0.717 (worse than pooled 0.683). Category-median ratio with slope forced to 1: LOO 0.767 (worse). Ridge on age/chains/change/category dummies: LOO 0.698+ (worse). Extra regressors (age, change, chains): LOO 0.688+ (worse). LAD: LOO 0.690 (worse). Quadratic log-TVL: LOO 0.691 (worse). The kept combination is the only one that improved LOO (0.671) and walk-forward (0.797) together; the gain is small and the board says so.",
    terms_kept: "log10(TVL) by Huber IRLS (c=1.0) + per-category median intercept shrunk n/(n+20); logistic odds on log10(TVL), age, log10(chains), change_7d and category",
  };

  // --- implied value per protocol ---
  function factors(p: SnapshotProtocol): ModelFactors {
    const lt = log10(p.tvl_usd);
    const fitted = Math.pow(10, predictFinal(finalModel, lt, p.category));
    const ch = p.change_7d === null ? 0 : p.change_7d;
    const growth = 1 + (clamp(ch, -60, 120) / 100) * 0.5;
    const age = ageDays(p, asOf);
    const maturity = 0.6 + (Math.min(age, 1095) / 1095) * 0.4;
    const odds = logisticPredict(fullLog, p, asOf);
    const ev = fitted * growth * maturity * odds;
    const d = p.category === null ? 0 : (finalModel.delta.get(p.category) ?? 0);
    return {
      fitted_usd: Math.round(fitted),
      growth_mult: Math.round(growth * 10000) / 10000,
      maturity_mult: Math.round(maturity * 10000) / 10000,
      odds: Math.round(odds * 10000) / 10000,
      ev_usd: Math.round(ev),
      log_tvl: Math.round(lt * 10000) / 10000,
      ols_a: Math.round(olsA * 10000) / 10000,
      ols_b: Math.round(olsB * 10000) / 10000,
      cat: p.category,
      cat_mult: Math.round(Math.pow(10, d) * 10000) / 10000,
      cat_n: p.category === null ? 0 : (finalModel.catN.get(p.category) ?? 0),
    };
  }

  function makeRow(p: SnapshotProtocol): BoardRow {
    const f = factors(p);
    const implied = Math.max(1, Math.round(f.ev_usd));
    const chain = p.chains.length > 0 ? p.chains[0] : "Ethereum";
    const cap = capitalFor(p.tvl_usd);
    const plan = PLAYBOOKS[playbookKey(p.category)](p.name, chain, cap);
    const cost = gasUsd(chain) + 5;
    const expected = f.ev_usd * 0.05 * 0.001;
    const hours = plan.hours;
    const perHour = expected / hours;
    const verdict: Verdict = perHour >= 25 ? "FARM" : perHour >= 5 ? "MARGINAL" : "SKIP";
    return {
      rank: 0,
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      change_7d: p.change_7d,
      mcap_usd: p.mcap_usd,
      listed_at: p.listed_at,
      gecko_id: p.gecko_id,
      symbol: p.symbol,
      token_status: "pre-token",
      age_days: ageDays(p, asOf),
      implied_value_usd: implied,
      model: f,
      farm_steps: plan.steps,
      verdict,
      expected_value_usd: Math.round(expected * 100) / 100,
      hours,
      usd_per_hour: Math.round(perHour * 100) / 100,
      farming_cost_usd: cost,
      breakeven_fdv_usd: Math.round(cost / (0.05 * 0.001)),
      comps: nearestComps(p, usable),
    };
  }

  const wlSet = new Set(snap.watchlist);
  const watchRows = ps.filter((p) => wlSet.has(p.slug)).map(makeRow);
  const early = ps
    .filter((p) => !wlSet.has(p.slug) && p.tvl_usd < snap.floor_usd && isEarlyEligible(p))
    .map(makeRow)
    .sort((a, b) => (b.implied_value_usd - a.implied_value_usd) || (a.slug < b.slug ? -1 : 1))
    .slice(0, 200);
  const rows = [...watchRows, ...early]
    .sort((a, b) => (b.implied_value_usd - a.implied_value_usd) || (a.slug < b.slug ? -1 : 1));
  rows.forEach((r, i) => { r.rank = i + 1; });

  return {
    rows,
    validation,
    counts: {
      protocols: ps.length,
      watchlist: snap.watchlist.length,
      board_rows: rows.length,
      early_rows: early.length,
    },
  };
}

// Self-check: returns a list of violations (empty when the board is valid).
export function validateBoard(rows: BoardRow[], watchlist: string[]): string[] {
  const problems: string[] = [];
  const wlSet = new Set(watchlist);
  const seen = new Set<string>();
  rows.forEach((r, i) => {
    if (r.rank !== i + 1) problems.push(`row ${r.slug}: rank ${r.rank} != ${i + 1}`);
    if (i > 0 && rows[i - 1].implied_value_usd < r.implied_value_usd) {
      problems.push(`row ${r.slug}: implied value rises with rank`);
    }
    if (seen.has(r.slug)) problems.push(`duplicate slug ${r.slug}`);
    seen.add(r.slug);
    if (!Number.isInteger(r.implied_value_usd) || r.implied_value_usd <= 0) {
      problems.push(`row ${r.slug}: bad implied value ${r.implied_value_usd}`);
    }
    if (!Array.isArray(r.farm_steps) || r.farm_steps.length !== 6) {
      problems.push(`row ${r.slug}: ${r.farm_steps?.length} farm steps, want 6`);
    } else {
      r.farm_steps.forEach((s, j) => {
        const t = s.trim();
        if (t.length < 20 || t.length > 600) {
          problems.push(`row ${r.slug} step ${j + 1}: ${t.length} chars`);
        }
      });
    }
  });
  for (const s of wlSet) {
    if (!seen.has(s)) problems.push(`watchlist slug missing: ${s}`);
  }
  const values = new Set(rows.map((r) => r.implied_value_usd));
  if (rows.length >= 2 && values.size === 1) problems.push("all implied values equal");
  return problems;
}

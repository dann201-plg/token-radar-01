// Token-radar entrypoint: reads the sealed snapshot, builds the ranked
// board, writes the dashboard data, copies the dashboard and screenshots,
// and writes the dossier. Node 22+ stdlib only. Deterministic: no clock,
// no randomness, every sort order total, so two runs write identical bytes.

import { readFileSync, writeFileSync, mkdirSync, cpSync, existsSync, statSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildDossier } from "./vendor/makefun-board.ts";
import { buildBoard, validateBoard } from "./lib/model.ts";
import type { Snapshot } from "./lib/model.ts";

const ROOT = dirname(fileURLToPath(import.meta.url));

function usage(): never {
  console.error("usage: node main.ts [--input <snapshot.json|dir>] [--no-dossier]");
  console.error("  input defaults to /input/snapshot.json; output is $MAKEFUN_OUTPUT_DIR or ./out");
  process.exit(2);
}

function parseArgs(argv: string[]): { input: string; noDossier: boolean } {
  let input = "/input/snapshot.json";
  let noDossier = false;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--input") {
      const v = argv[++i];
      if (!v) usage();
      input = v;
    } else if (a === "--no-dossier") {
      noDossier = true;
    } else if (a === "--help" || a === "-h") {
      usage();
    } else {
      console.error(`unknown argument: ${a}`);
      usage();
    }
  }
  return { input, noDossier };
}

function readSnapshot(path: string): Snapshot {
  let file = path;
  if (existsSync(path) && statSync(path).isDirectory()) file = join(path, "snapshot.json");
  const text = readFileSync(file, "utf-8");
  const snap = JSON.parse(text) as Snapshot;
  if (!Array.isArray(snap.protocols) || !Array.isArray(snap.watchlist)) {
    throw new Error(`snapshot at ${file} is missing protocols/watchlist`);
  }
  return snap;
}

// The snapshot's source string names a network address. Dashboard files
// must name none, so the board carries a scrubbed provenance label instead.
function scrubSource(source: string): string {
  const m = /fetched\s+(.+?)\s*$/.exec(source);
  return m ? `DefiLlama protocols snapshot, fetched ${m[1]}` : "DefiLlama protocols snapshot";
}

function main(): void {
  const { input, noDossier } = parseArgs(process.argv.slice(2));
  const outDir = resolve(process.env.MAKEFUN_OUTPUT_DIR ?? join(ROOT, "out"));
  const snap = readSnapshot(input);
  const result = buildBoard(snap);
  const problems = validateBoard(result.rows, snap.watchlist);
  if (problems.length > 0) {
    for (const p of problems) console.error(`board violation: ${p}`);
    process.exit(1);
  }

  const v = result.validation;
  console.log(`snapshot as_of=${snap.as_of} protocols=${snap.protocols.length} watchlist=${snap.watchlist.length} floor=$${snap.floor_usd}`);
  console.log(`model: ${v.model_name}; log10(mcap) = ${v.ols_a.toFixed(4)} + ${v.ols_b.toFixed(4)}*log10(TVL) + category median intercept`);
  console.log(`comps: ${v.comps_used} usable of ${v.comps_recorded} recorded (${v.comps_excluded_placeholder} carry a 0/-1 placeholder mcap and cannot enter a log fit)`);
  console.log(`validation: LOO MAE ${v.loo_mae.toFixed(4)} vs naive ${v.loo_naive_mae.toFixed(4)} -> x${v.loo_multiple.toFixed(2)}`);
  console.log(`validation: walk-forward MAE ${v.walkforward_mae.toFixed(4)} vs naive ${v.walkforward_naive_mae.toFixed(4)} -> x${v.walkforward_multiple.toFixed(2)} (train ${v.walkforward_train}, test ${v.walkforward_test})`);
  console.log(`validation: category-blind pooled MAE ${v.catblind_mae.toFixed(4)} vs naive ${v.catblind_naive_mae.toFixed(4)} -> x${v.catblind_multiple.toFixed(2)}`);
  console.log(`validation: P10-P90 band [${v.p10_resid.toFixed(3)}, ${v.p90_resid.toFixed(3)}] coverage ${(v.p10_p90_coverage * 100).toFixed(1)}%`);
  console.log(`validation: logistic Brier ${v.brier.toFixed(4)} (base rate ${v.brier_base_rate.toFixed(4)})`);
  console.log(`validation: category fits: ${v.cat_fits.map((c) => `${c.category} n=${c.n} fitted x${c.fitted_multiple.toFixed(2)} LOO x${c.loo_multiple.toFixed(2)}`).join("; ")}`);
  console.log(`validation: dropped levers: ${v.dropped_levers}`);
  console.log(`board: ${result.rows.length} rows (${result.counts.watchlist} watchlist + ${result.counts.early_rows} early), top implied $${result.rows[0].implied_value_usd.toLocaleString("en-US")}`);

  mkdirSync(outDir, { recursive: true });
  mkdirSync(join(outDir, "dashboard"), { recursive: true });
  mkdirSync(join(outDir, "screenshots"), { recursive: true });

  // board.json: the ranked board plus the model record, so the file alone
  // carries everything the ranking claims.
  const boardDoc = {
    meta: {
      as_of: snap.as_of,
      source: scrubSource(snap.source),
      floor_usd: snap.floor_usd,
      watchlist: snap.watchlist,
      model: v.model_name,
      ols_a: v.ols_a,
      ols_b: v.ols_b,
      validation: v,
      counts: result.counts,
    },
    rows: result.rows,
  };
  writeFileSync(join(outDir, "board.json"), JSON.stringify(boardDoc));

  // data.js: the same board for the dashboard page, as one global.
  const radar = {
    meta: boardDoc.meta,
    rows: result.rows,
  };
  const dataJs = `window.RADAR = ${JSON.stringify(radar)};\n`;
  writeFileSync(join(outDir, "dashboard", "data.js"), dataJs);
  writeFileSync(join(outDir, "data.js"), dataJs);

  // dashboard source (fresh files from this repository) and screenshots.
  cpSync(join(ROOT, "dashboard", "index.html"), join(outDir, "dashboard", "index.html"));
  cpSync(join(ROOT, "dashboard", "app.css"), join(outDir, "dashboard", "app.css"));
  cpSync(join(ROOT, "dashboard", "app.js"), join(outDir, "dashboard", "app.js"));
  cpSync(join(ROOT, "screenshots"), join(outDir, "screenshots"), { recursive: true });

  if (!noDossier) {
    const dossier = buildDossier(outDir, {
      text_max_bytes: 65536,
      screenshots: { max: 4, max_bytes: 524288 },
      dashboard: { max_files: 400, max_bytes: 4194304 },
    });
    writeFileSync(join(outDir, "dossier.json"), dossier);
    console.log(`dossier: ${dossier.length} bytes`);
  }
  console.log(`wrote ${outDir}`);
}

main();

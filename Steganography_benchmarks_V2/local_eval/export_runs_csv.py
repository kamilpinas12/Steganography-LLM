import csv
import json
from pathlib import Path

runs_root = Path(__file__).resolve().parent / "runs"
out_csv = runs_root / "local_runs_summary.csv"

columns = [
    "Benchmark",
    "Model",
    "threshold",
    "topN",
    "result",
    "avg_bits_per_token",
    "stego_activation_ratio",
    "avg_pool_size_stego_only",
]

rows = []

# szukaj runów z manifestem i wynikiem ewaluacji
for manifest_path in sorted(runs_root.rglob("manifest.json")):
    run_dir = manifest_path.parent
    eval_dir = run_dir / "evaluation"
    if not eval_dir.is_dir():
        continue

    result_files = sorted(eval_dir.glob("*_results.json"))
    if not result_files:
        continue

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(result_files[0].read_text(encoding="utf-8"))

    benchmark = manifest.get("test") or results.get("test", "")
    model = manifest.get("model_key", "")
    threshold = manifest.get("threshold", "")
    top_n = manifest.get("top_n", "")

    # główna metryka zależna od benchmarku
    if benchmark == "humaneval":
        result = results.get("passed_count", results.get("pass_at_1", ""))
    elif benchmark == "capacity":
        result = results.get("capacity", {}).get("embedding_rate", "")
    elif benchmark == "perplexity":
        result = results.get("perplexity", "")
    elif benchmark == "binoculars":
        result = results.get("binoculars_score", "")
    else:
        result = ""

    cap = results.get("capacity", {})
    avg_bpt = cap.get("avg_bits_per_token", "")
    pool_stego = cap.get("avg_pool_size_stego_only", "")

    stego_ratio = cap.get("stego_activation_ratio")
    if stego_ratio is None:
        total = cap.get("total_steps") or 0
        stego = cap.get("stego_applied_steps") or 0
        if total:
            stego_ratio = stego / total
        elif threshold == 0 or threshold == 0.0:
            stego_ratio = 0.0
        else:
            stego_ratio = ""

    rows.append(
        {
            "Benchmark": benchmark,
            "Model": model,
            "threshold": threshold,
            "topN": top_n,
            "result": result,
            "avg_bits_per_token": avg_bpt,
            "stego_activation_ratio": stego_ratio,
            "avg_pool_size_stego_only": pool_stego,
        }
    )

rows.sort(key=lambda r: (r["Benchmark"], r["Model"], float(r["threshold"] or 0)))

with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"Zapisano {len(rows)} wierszy -> {out_csv}")

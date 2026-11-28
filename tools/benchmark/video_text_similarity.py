"""Benchmark video/text filtering on the repository's MP4 fixtures."""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import av
import datasets
import numpy as np
import torch
import transformers

from data_juicer.core.data import NestedDataset
from data_juicer.ops.filter.video_frames_text_similarity_filter import (
    VideoFramesTextSimilarityFilter,
)
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import SpecialTokens


def fixture_rows(paths, count):
    captions = ("People are talking indoors.", "An animated scene with several animals.", "A landscape at night.")
    rows = []
    for i in range(count):
        caption = captions[(i // len(paths) + i) % len(captions)]
        text = f"{SpecialTokens.video} {caption}"
        media = [str(paths[i % len(paths)])]
        if i % 5 == 0:
            text += f"{SpecialTokens.eoc}{SpecialTokens.video} {captions[(i + 1) % len(captions)]}"
            media.append(str(paths[(i + 1) % len(paths)]))
        elif i % 7 == 0:
            text += SpecialTokens.video
            media.append(str(paths[(i + 1) % len(paths)]))
        rows.append({"id": i, "videos": media, "text": text})
    return rows


def run(op, rows, cuda):
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    dataset = NestedDataset.from_list(rows).add_column(Fields.stats, [{} for _ in rows])
    scored = dataset.map(op.compute_stats, with_rank=cuda, batch_size=op.batch_size)
    evaluated = time.perf_counter()
    kept = list(scored.filter(op.process, batch_size=op.batch_size)["id"])
    scores = [stat[StatsKeys.video_frames_text_similarity] for stat in scored[Fields.stats]]
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "seconds": elapsed,
        "scoring_seconds": evaluated - start,
        "rows_per_second": len(rows) / elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
        "scores": scores,
        "kept_ids": kept,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--rows", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 8, 16, 32])
    parser.add_argument("--frame-num", type=int, default=3)
    parser.add_argument("--sampling", choices=["uniform", "all_keyframes"], default="uniform")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if (
        min(args.rows, args.repeats, args.frame_num) < 1
        or min(args.budgets) < 0
        or len(set(args.budgets)) != len(args.budgets)
    ):
        parser.error("positive counts and distinct nonnegative budgets are required")
    cuda = args.device == "cuda"
    if cuda and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    datasets.disable_caching()
    datasets.disable_progress_bars()
    torch.set_num_threads(4)
    media_dir = Path(__file__).resolve().parents[2] / "tests" / "ops" / "data"
    unique = {}
    for path in sorted(media_dir.glob("video*.mp4")):
        unique.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), path)
    rows = fixture_rows(list(unique.values()), args.rows)
    manifest = [{**row, "videos": [Path(path).name for path in row["videos"]]} for row in rows]
    operators = {}
    for budget in args.budgets:
        op = VideoFramesTextSimilarityFilter(
            hf_clip=args.model,
            accelerator=args.device,
            auto_op_parallelism=False,
            batch_mode=bool(budget),
            batch_size=16,
            max_batch_frames=budget or 32,
            frame_sampling_method=args.sampling,
            frame_num=args.frame_num,
            min_score=0.20,
            max_score=0.30,
        )
        if operators:
            op.model_key = next(iter(operators.values())).model_key
        operators[budget] = op
        run(op, rows[: min(len(rows), 16)], cuda)

    trials = {budget: [] for budget in operators}
    for trial in range(args.repeats):
        offset = trial % len(args.budgets)
        for budget in args.budgets[offset:] + args.budgets[:offset]:
            result = run(operators[budget], rows, cuda)
            trials[budget].append(result)
            print(f"repeat {trial + 1}, frame budget {budget}: {result['rows_per_second']:.2f} rows/s", flush=True)
    baseline = trials[0][0] if 0 in trials else None
    summary = {}
    for budget, results in trials.items():
        case = {"median_rows_per_second": statistics.median(r["rows_per_second"] for r in results)}
        if cuda:
            for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
                case[field] = max(r[field] for r in results)
        if baseline:
            reference = np.concatenate(baseline["scores"])
            sizes = [len(row) for row in baseline["scores"]]
            for result in results:
                if [len(row) for row in result["scores"]] != sizes:
                    raise AssertionError("the number of scores per row changed")
            case["max_abs_score_delta"] = max(
                float(np.max(np.abs(np.concatenate(r["scores"]) - reference))) for r in results
            )
            case["kept_ids_match"] = all(r["kept_ids"] == baseline["kept_ids"] for r in results)
        summary[budget] = case
    report = {
        "runtime": {
            "platform": platform.platform(),
            "device": torch.cuda.get_device_name() if cuda else "cpu",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "pyav": av.__version__,
        },
        "model": args.model,
        "fixture_sha256": {path.name: sha for sha, path in unique.items()},
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "rows": len(rows),
        "video_references": sum(len(row["videos"]) for row in rows),
        "frame_num": args.frame_num,
        "sampling": args.sampling,
        "score_interval": [0.20, 0.30],
        "summary": summary,
        "trials": trials,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

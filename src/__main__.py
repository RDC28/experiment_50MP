"""Command line: ``python -m synthetic_raw_lab <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import environment, pipeline, test_images
from .config import load_config, parameter_table
from .resolution import compute_target_resolution


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m synthetic_raw_lab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run Phase 0/1: ingest, baselines, exports, metrics, report")
    run.add_argument("source", type=Path)
    run.add_argument("--config", type=Path, help="YAML configuration (defaults apply otherwise)")
    run.add_argument("--project-root", type=Path, default=Path.cwd())
    run.add_argument("--preset", choices=("quick", "standard", "research"), help="override runtime_preset")
    run.add_argument("--category", choices=("human", "night", "universal"), help="override input_category")
    run.add_argument("--benchmark-memory", action="store_true", help="also run the representation benchmark")

    image = commands.add_parser("make-test-image", help="write the deterministic synthetic test PNG")
    image.add_argument("path", type=Path)
    image.add_argument("--width", type=int, default=3840)
    image.add_argument("--height", type=int, default=2160)
    image.add_argument("--seed", type=int, default=42)
    image.add_argument("--embed-srgb-profile", action="store_true")

    target = commands.add_parser("target", help="print the target resolution for a source size")
    target.add_argument("width", type=int)
    target.add_argument("height", type=int)
    target.add_argument("--megapixels", type=float, default=50.0)
    target.add_argument("--mode", choices=("tolerance", "exact"), default="tolerance")

    bench = commands.add_parser("benchmark-memory", help="measure PRD 21.3 representations at a size")
    bench.add_argument("width", type=int)
    bench.add_argument("height", type=int)

    commands.add_parser("environment", help="print the runtime inspection")
    describe = commands.add_parser("describe-config", help="print every parameter with its documentation")
    describe.add_argument("--config", type=Path)

    args = parser.parse_args(argv)
    if args.command == "run":
        overrides = {k: v for k, v in (("runtime_preset", args.preset), ("input_category", args.category)) if v}
        config = load_config(args.config, **overrides)
        result = pipeline.run(
            args.source, config, args.project_root, benchmark_memory=True if args.benchmark_memory else None
        )
        print(
            json.dumps(
                {"run_id": result.run_id, "report": str(result.report_path), "masters": str(result.masters_dir)},
                indent=2,
            )
        )
    elif args.command == "make-test-image":
        path = test_images.write_test_image(
            args.path, args.width, args.height, args.seed, embed_srgb_profile=args.embed_srgb_profile
        )
        print(path)
    elif args.command == "target":
        print(
            json.dumps(
                compute_target_resolution(args.width, args.height, args.megapixels, mode=args.mode).to_dict(), indent=2
            )
        )
    elif args.command == "benchmark-memory":
        print(json.dumps(environment.benchmark_representations(args.width, args.height), indent=2))
    elif args.command == "environment":
        print(json.dumps(environment.inspect_runtime(), indent=2))
    elif args.command == "describe-config":
        for row in parameter_table(load_config(args.config)):
            print(f"{row['parameter']}: {row['current']!r} (default {row['default']!r}; {row['accepted']})")
            warning = f" WARNING: {row['warning']}" if row["warning"] else ""
            print(f"    {row['description']} {row['effect']}{warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command-line interface: ``pendulum-lab <command>``.

Commands
--------
``all``      run every study, write figures to ``docs/img`` and results to
             ``docs/results``, and refresh the generated README tables.
``report``   regenerate Markdown tables (and README sections) from saved JSON.
``swingup``  run the nominal swing-up and write its figure and GIF.
``analyze``  print the structural analysis (poles, controllability, observability).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pendulum_lab.experiments import RunSettings, run_all, study_structure, study_swingup
from pendulum_lab.params import LabConfig
from pendulum_lab.report import inject_readme, render_tables


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pendulum-lab", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all", help="run every study")
    a.add_argument("--img", type=Path, default=Path("docs/img"))
    a.add_argument("--results", type=Path, default=Path("docs/results"))
    a.add_argument("--readme", type=Path, default=Path("README.md"))
    a.add_argument("--quick", action="store_true", help="small sample sizes (smoke test)")
    a.add_argument("--no-gif", action="store_true")
    r = sub.add_parser("report", help="render tables from saved results")
    r.add_argument("--results", type=Path, default=Path("docs/results"))
    r.add_argument("--readme", type=Path, default=Path("README.md"))
    s = sub.add_parser("swingup", help="nominal swing-up figure and GIF")
    s.add_argument("--img", type=Path, default=Path("docs/img"))
    sub.add_parser("analyze", help="print structural analysis")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = _parser().parse_args(argv)
    config = LabConfig()
    if args.cmd == "all":
        settings = RunSettings.quick() if args.quick else RunSettings()
        results = run_all(args.img, settings, config, gif=not args.no_gif)
        args.results.mkdir(parents=True, exist_ok=True)
        path = args.results / "results.json"
        path.write_text(json.dumps(results, indent=1, default=float) + "\n")
        tables = render_tables(results)
        (args.results / "tables.md").write_text(
            "\n\n".join(f"### {k}\n\n{v}" for k, v in tables.items()) + "\n"
        )
        if args.readme.exists():
            inject_readme(args.readme, tables)
        print(f"results written to {path}")
    elif args.cmd == "report":
        results = json.loads((args.results / "results.json").read_text())
        tables = render_tables(results)
        (args.results / "tables.md").write_text(
            "\n\n".join(f"### {k}\n\n{v}" for k, v in tables.items()) + "\n"
        )
        if args.readme.exists():
            inject_readme(args.readme, tables)
    elif args.cmd == "swingup":
        print(json.dumps(study_swingup(config, args.img, seed=RunSettings().seed), indent=1))
    elif args.cmd == "analyze":
        print(json.dumps(study_structure(config), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

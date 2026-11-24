import argparse
import json
from .data import download
from .runner import Run, worker


def main():
    parser = argparse.ArgumentParser(description="Bike demand course project")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--profile", choices=["smoke", "full"], default="full")
    p = sub.add_parser("_worker")
    p.add_argument("job")
    sub.add_parser("doctor")
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(download(), ensure_ascii=False, indent=2))
    elif args.command == "doctor":
        from .models import cuda_info
        from .runner import environment
        print(json.dumps({**environment(), "gpu": cuda_info()}, ensure_ascii=False, indent=2))
    elif args.command == "_worker":
        worker(args.job)
    elif args.command == "run":
        folder = Run(args.profile).execute()
        from .report import build_report
        print(build_report(args.profile))
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        if not status["complete"]:
            raise SystemExit(2)
    else:
        from .report import build_report
        print(build_report(args.profile))


if __name__ == "__main__":
    main()

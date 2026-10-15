"""Verify the frozen code package before executing a saved development recipe."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    frozen = json.loads((root / "configs/main_source_freeze.json").read_text(encoding="utf-8"))
    files = {str(path.relative_to(root / "code").as_posix()): path for path in (root / "code").rglob("*")
             if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts
             and path.suffix not in (".pyc", ".pyo")}
    expected = frozen["code_files"]
    if set(files) != set(expected):
        raise ValueError("Frozen code package membership differs")
    for name, item in expected.items():
        if files[name].stat().st_size != item["bytes"] or sha(files[name]) != item["sha256"]:
            raise ValueError("Frozen source changed: " + name)
    for name, item in frozen["command_files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != item["sha256"]:
            raise ValueError("Frozen recipe changed: " + name)
    print(json.dumps({"status": "source_verified", "files": len(files),
                      "training_source_sha256": frozen["training_source_sha256"],
                      "package_sha256": frozen["code_package_sha256"]}), flush=True)


if __name__ == "__main__":
    main()

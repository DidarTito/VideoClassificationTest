"""Inventory, verify, and retrieve checkpoint artifacts kept outside Git.

The manifest deliberately separates local identity (path, size, SHA-256) from
provenance and redistribution metadata. Refresh never invents an upstream URL
or license; it preserves metadata already entered for a known path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import sys
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "checkpoints"
MANIFEST = CHECKPOINT_ROOT / "manifest.csv"
ARTIFACT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".json",
    ".pkl",
    ".pt",
    ".pth",
    ".pyth",
    ".safetensors",
    ".torch",
}
FIELDS = (
    "path",
    "bytes",
    "sha256",
    "model_family",
    "dataset_hint",
    "artifact_role",
    "source_url",
    "google_drive_id",
    "source_project",
    "license",
    "redistribution_status",
    "notes",
)
PRESERVED_FIELDS = (
    "source_url",
    "google_drive_id",
    "source_project",
    "license",
    "redistribution_status",
    "notes",
)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_files(checkpoint_root: Path = CHECKPOINT_ROOT) -> list[Path]:
    return sorted(
        (
            path
            for path in checkpoint_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in ARTIFACT_SUFFIXES
            and not path.name.endswith(".part")
        ),
        key=lambda path: path.relative_to(checkpoint_root).as_posix().lower(),
    )


def dataset_hint(relative: Path) -> str:
    value = relative.as_posix().lower()
    if "ssv2" in value or "sthv2" in value or "something" in value:
        return "ssv2"
    if "videomae-vit-" in value and "kinetics" not in value:
        return "kinetics400-pretraining"
    known_k400_families = (
        "dualformer/",
        "focalnet/",
        "mvit/",
        "omnivore/",
        "svt/",
        "timesformer/",
        "uniformer/",
        "videoswin/",
        "vivit/",
        "vtn/",
        "zeroi2v/",
    )
    if (
        "k400" in value
        or "kinetics" in value
        or value.startswith(known_k400_families)
    ):
        return "kinetics400"
    return "unknown"


def artifact_role(relative: Path) -> str:
    name = relative.name.lower()
    if relative.suffix.lower() == ".json":
        return "configuration-metadata"
    if "labels" in name:
        return "label-metadata"
    if "sims" in name or "similarity" in name:
        return "similarity-metadata"
    return "model-checkpoint"


def read_manifest(manifest_path: Path = MANIFEST) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        return []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{manifest_path} is missing columns: {', '.join(sorted(missing))}"
            )
        return [dict(row) for row in reader]


def safe_destination(
    manifest_value: str,
    root: Path = ROOT,
    checkpoint_root: Path = CHECKPOINT_ROOT,
) -> Path:
    raw = Path(manifest_value)
    if raw.is_absolute():
        raise ValueError(f"manifest path must be relative: {manifest_value}")
    destination = (root / raw).resolve()
    allowed = checkpoint_root.resolve()
    try:
        destination.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(
            f"manifest path escapes {checkpoint_root}: {manifest_value}"
        ) from exc
    return destination


def write_manifest(
    rows: list[dict[str, str]], manifest_path: Path = MANIFEST
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest_path)


def refresh(
    root: Path = ROOT,
    checkpoint_root: Path = CHECKPOINT_ROOT,
    manifest_path: Path = MANIFEST,
) -> int:
    previous = {
        row["path"]: row
        for row in read_manifest(manifest_path)
        if row.get("path")
    }
    rows: list[dict[str, str]] = []
    total_bytes = 0

    for index, path in enumerate(artifact_files(checkpoint_root), start=1):
        relative_checkpoint = path.relative_to(checkpoint_root)
        relative_root = path.relative_to(root).as_posix()
        before = path.stat()
        print(
            f"[{index:02d}] hashing {relative_root} "
            f"({before.st_size / (1024 ** 2):.1f} MiB)",
            flush=True,
        )
        checksum = sha256_file(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError(
                f"{relative_root} changed while hashing; wait for downloads to finish"
            )

        old = previous.get(relative_root, {})
        row = {
            "path": relative_root,
            "bytes": str(after.st_size),
            "sha256": checksum,
            "model_family": relative_checkpoint.parts[0],
            "dataset_hint": dataset_hint(relative_checkpoint),
            "artifact_role": artifact_role(relative_checkpoint),
            "source_url": "",
            "google_drive_id": "",
            "source_project": "",
            "license": "",
            "redistribution_status": "unverified-do-not-redistribute",
            "notes": "",
        }
        for field in PRESERVED_FIELDS:
            if old.get(field):
                row[field] = old[field]
        rows.append(row)
        total_bytes += after.st_size

    write_manifest(rows, manifest_path)
    print(
        f"Wrote {len(rows)} artifacts ({total_bytes / (1024 ** 3):.3f} GiB) "
        f"to {manifest_path}"
    )
    return 0


def selected_rows(
    rows: list[dict[str, str]], requested: list[str] | None
) -> list[dict[str, str]]:
    if not requested:
        return rows
    wanted = {value.replace("\\", "/") for value in requested}
    selected = [row for row in rows if row["path"].replace("\\", "/") in wanted]
    missing = wanted - {row["path"].replace("\\", "/") for row in selected}
    if missing:
        raise ValueError(f"paths not present in manifest: {', '.join(sorted(missing))}")
    return selected


def verify(
    quick: bool = False,
    requested: list[str] | None = None,
    root: Path = ROOT,
    checkpoint_root: Path = CHECKPOINT_ROOT,
    manifest_path: Path = MANIFEST,
) -> int:
    rows = selected_rows(read_manifest(manifest_path), requested)
    failures = 0
    for row in rows:
        path = safe_destination(row["path"], root, checkpoint_root)
        problem = ""
        if not path.is_file():
            problem = "missing"
        elif path.stat().st_size != int(row["bytes"]):
            problem = f"size mismatch ({path.stat().st_size} != {row['bytes']})"
        elif not quick and sha256_file(path).lower() != row["sha256"].lower():
            problem = "SHA-256 mismatch"

        if problem:
            failures += 1
            print(f"FAIL {row['path']}: {problem}")
        else:
            print(f"OK   {row['path']}")

    print(f"Verified {len(rows) - failures}/{len(rows)} artifacts")
    return 1 if failures else 0


def download_http(url: str, temporary: Path) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"only HTTP(S) source URLs are accepted: {url}")
    # Official TimeSformer checkpoint links are Dropbox share pages with
    # ``dl=0``.  Requesting that URL verbatim stores an HTML preview rather
    # than the checkpoint, so switch only Dropbox's documented download flag.
    if (parsed.hostname or "").lower() in {"dropbox.com", "www.dropbox.com"}:
        query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
        query["dl"] = "1"
        parsed = parsed._replace(query=urllib.parse.urlencode(query))
        url = urllib.parse.urlunparse(parsed)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "VideoClassificationTest-artifact-helper/1.0"},
    )
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)


def download_gdrive(file_id: str, temporary: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive retrieval requires: pip install -r "
            "requirements-artifacts.txt"
        ) from exc
    result = gdown.download(id=file_id, output=str(temporary), quiet=False)
    if not result:
        raise RuntimeError(f"gdown did not retrieve Google Drive file {file_id}")


def download(
    requested: list[str] | None = None,
    replace: bool = False,
    root: Path = ROOT,
    checkpoint_root: Path = CHECKPOINT_ROOT,
    manifest_path: Path = MANIFEST,
) -> int:
    rows = selected_rows(read_manifest(manifest_path), requested)
    downloaded = 0
    skipped = 0
    failures = 0

    for row in rows:
        destination = safe_destination(row["path"], root, checkpoint_root)
        if destination.is_file():
            size_matches = destination.stat().st_size == int(row["bytes"])
            hash_matches = size_matches and (
                sha256_file(destination).lower() == row["sha256"].lower()
            )
            if hash_matches:
                print(f"OK   {row['path']} already verified")
                skipped += 1
                continue
            if not replace:
                print(f"FAIL {row['path']} exists but does not match; pass --replace")
                failures += 1
                continue

        source_url = row.get("source_url", "").strip()
        drive_id = row.get("google_drive_id", "").strip()
        if bool(source_url) == bool(drive_id):
            print(
                f"SKIP {row['path']}: set exactly one source_url or google_drive_id"
            )
            skipped += 1
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        temporary.unlink(missing_ok=True)
        try:
            print(f"GET  {row['path']}", flush=True)
            if source_url:
                download_http(source_url, temporary)
            else:
                download_gdrive(drive_id, temporary)
            actual_size = temporary.stat().st_size
            if actual_size != int(row["bytes"]):
                raise RuntimeError(
                    f"size mismatch after download ({actual_size} != {row['bytes']})"
                )
            actual_hash = sha256_file(temporary)
            if actual_hash.lower() != row["sha256"].lower():
                raise RuntimeError("SHA-256 mismatch after download")
            os.replace(temporary, destination)
            downloaded += 1
            print(f"OK   {row['path']}")
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            failures += 1
            print(f"FAIL {row['path']}: {exc}")

    print(
        f"Downloaded {downloaded}; already present/skipped {skipped}; "
        f"failed {failures}"
    )
    return 1 if failures else 0


def list_artifacts(manifest_path: Path = MANIFEST) -> int:
    rows = read_manifest(manifest_path)
    for row in rows:
        source = row.get("source_url") or row.get("google_drive_id") or "-"
        print(
            f"{int(row['bytes']) / (1024 ** 2):9.1f} MiB  "
            f"{row['dataset_hint']:24s}  {row['path']}  source={source}"
        )
    print(f"{len(rows)} artifacts")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("refresh", help="hash local artifacts and rewrite manifest")

    verify_parser = subparsers.add_parser("verify", help="verify manifest artifacts")
    verify_parser.add_argument("--quick", action="store_true", help="check sizes only")
    verify_parser.add_argument("--path", action="append", help="verify one manifest path")

    download_parser = subparsers.add_parser(
        "download", help="download rows with a configured source"
    )
    download_parser.add_argument("--path", action="append", help="download one path")
    download_parser.add_argument(
        "--replace", action="store_true", help="replace a mismatched local file"
    )
    subparsers.add_parser("list", help="list inventory rows and configured sources")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "refresh":
        return refresh()
    if args.command == "verify":
        return verify(quick=args.quick, requested=args.path)
    if args.command == "download":
        return download(requested=args.path, replace=args.replace)
    if args.command == "list":
        return list_artifacts()
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

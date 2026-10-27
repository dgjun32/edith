"""Restore and verify the complete example HDF5 using only Python's standard library."""

import argparse
import hashlib
import json
import lzma
from pathlib import Path


def unpack(folder: Path, output: Path) -> Path:
    manifest = json.loads((folder / "manifest.json").read_text())
    if output.exists():
        raise FileExistsError(f"Will not overwrite {output}")
    # Check all parts before creating an output file.
    for part in manifest["archive_parts"]:
        path = folder / part["path"]
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        if size != part["bytes"] or digest.hexdigest() != part["sha256"]:
            raise ValueError(f"Archive part failed checksum: {path}")

    decoder = lzma.LZMADecompressor()
    archive_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    archive_bytes = output_bytes = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    destination = output.open("xb")
    try:
        with destination:
            for part in manifest["archive_parts"]:
                with (folder / part["path"]).open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        archive_hash.update(block)
                        archive_bytes += len(block)
                        decoded = decoder.decompress(block)
                        destination.write(decoded)
                        output_hash.update(decoded)
                        output_bytes += len(decoded)
        if not decoder.eof or decoder.unused_data:
            raise ValueError("Incomplete XZ stream or unexpected trailing data")
        if (
            archive_bytes != manifest["archive"]["bytes"]
            or archive_hash.hexdigest() != manifest["archive"]["sha256"]
            or output_bytes != manifest["uncompressed"]["bytes"]
            or output_hash.hexdigest() != manifest["uncompressed"]["sha256"]
        ):
            raise ValueError("Restored recording failed checksum")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=folder / "observations.h5")
    args = parser.parse_args()
    path = unpack(folder, args.output)
    print(f"Verified complete recording: {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

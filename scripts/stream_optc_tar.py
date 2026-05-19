"""
Stream-extract targeted folders from OpTC.tar.gz without full extraction.
Reads archive sequentially — never loads full 288 GB into memory or disk.
Writes only matching files (~10-20 GB) to the existing OpTCNCR directory.
"""
import tarfile
from pathlib import Path

ARCHIVE = Path("datasets/DARPA_OpTC/OpTC.tar.gz")
DEST    = Path("datasets/DARPA_OpTC/OpTCNCR-20260326T025141Z-1-006/OpTCNCR")

TARGET_PATTERNS = [
    "ecar-bro/evaluation/01Oct19-red/",
    "ecar-bro/evaluation/07Oct19-red/",
    "bro/2019-10-01/",
    "bro/2019-10-07/",
]

def matches_target(name: str) -> bool:
    return any(pattern in name for pattern in TARGET_PATTERNS)

def stream_extract():
    extracted, skipped = 0, 0

    # mode='r|gz': streaming (sequential), no seeking — works for gzip
    with tarfile.open(ARCHIVE, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not matches_target(member.name):
                skipped += 1
                continue

            # Compute destination path, strip archive root prefix
            # Typical path: OpTCNCR/ecar-bro/evaluation/01Oct19-red/HOST/ecarbro.json.gz
            rel = Path(member.name)
            
            # Simply remove the first component if it's "OpTCNCR"
            parts = rel.parts
            if len(parts) > 1 and parts[0] == "OpTCNCR":
                rel_path = Path(*parts[1:])   # strip the OpTCNCR prefix
            else:
                rel_path = rel                         # fallback: use as-is

            dest_file = DEST / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            f = tar.extractfile(member)
            if f:
                dest_file.write_bytes(f.read())
                extracted += 1
                if extracted % 500 == 0:
                    print(f"  Extracted {extracted:,} files | Last: {rel_path}", end="\r")

    print(f"\nDone: {extracted} files extracted, {skipped} skipped.")

if __name__ == "__main__":
    stream_extract()

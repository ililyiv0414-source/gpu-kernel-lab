#!/usr/bin/env python3
"""Create a source-only upload bundle without GPU tools or third-party packages."""
import datetime as dt
import hashlib
import json
import zipfile
from run_gpu import ROOT, snapshot


def main():
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    directory = ROOT / "results" / ("package-" + stamp)
    directory.mkdir(parents=True, exist_ok=False)
    snapshot(ROOT, directory)
    hashes = json.loads((directory / "source-sha256.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(directory / "source.zip") as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(hashes):
            raise RuntimeError("source archive integrity check failed")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise RuntimeError("source hash mismatch: " + name)
    (directory / "PACKAGE_ONLY.txt").write_text(
        "Source packaging only. No CUDA compilation, GPU execution or performance validation.\n",
        encoding="utf-8")
    print("Source bundle: " + str(directory / "source.zip"))
    print("Verified archive and SHA256 for " + str(len(hashes)) + " source files; not GPU execution.")


if __name__ == "__main__":
    main()

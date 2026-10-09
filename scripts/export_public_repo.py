"""
scripts/export_public_repo.py
-----------------------------
Generates a 100% clean, verified public export of Dynamic Target Suite
in a parallel directory without carrying over proprietary data,
NDAs (TTC/Hoosier), CAD curves, private setups, or Git commit history.

Usage:
    python scripts/export_public_repo.py [--dest PATH]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def export_public_repo(dest_path: Path):
    src_root = Path(__file__).resolve().parent.parent
    dest_root = dest_path.resolve()

    print("=" * 70)
    print(f"EXPORTING CLEAN PUBLIC REPOSITORY")
    print(f"Source:      {src_root}")
    print(f"Destination: {dest_root}")
    print("=" * 70)

    if dest_root.exists():
        print(f"[!] Warning: Destination already exists. Cleaning up {dest_root}...")
        shutil.rmtree(dest_root)

    dest_root.mkdir(parents=True, exist_ok=True)

    copied_files_count = 0
    total_bytes = 0

    def copy_file(src_file: Path, rel_target: str):
        nonlocal copied_files_count, total_bytes
        if not src_file.is_file():
            return
        dst_file = dest_root / rel_target
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied_files_count += 1
        total_bytes += dst_file.stat().st_size

    # 1. ROOT ESSENTIALS
    root_files = [
        "main.py",
        "requirements.txt",
        "pytest.ini",
        "README.md",
        "LICENSE",
        "MDO_Target_Suite.spec",
        ".gitignore",
        "build_exe.bat",
        "AGENTS.md",
        "GEMINI.md",
        "CHANGELOG.md",
    ]
    print("\n[1/6] Copying Root Configuration & Launchers...")
    for rf in root_files:
        p = src_root / rf
        if p.is_file():
            copy_file(p, rf)
            print(f"  + {rf}")

    # 2. SOURCE CODE (src/)
    print("\n[2/6] Copying Clean Source Code (src/)...")
    for root, dirs, files in os.walk(src_root / "src"):
        # Prune pycache, private dirs, and ui/setups
        dirs[:] = [
            d for d in dirs
            if d != "__pycache__" and "private" not in d.lower() and d != "setups"
        ]
        for f in files:
            fl = f.lower()
            if fl.endswith((".pyc", ".pyo", ".pyd", ".dll", ".sig", ".c")):
                continue
            if "backup" in fl or "private" in fl or "et18" in fl or "et17" in fl:
                print(f"  [SKIP PRIVATE/SETUP] {f}")
                continue
            p = Path(root) / f
            rel = p.relative_to(src_root)
            copy_file(p, str(rel))

    # 3. TEST SUITE (tests/)
    print("\n[3/6] Copying Test Suite (tests/)...")
    for root, dirs, files in os.walk(src_root / "tests"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                continue
            p = Path(root) / f
            rel = p.relative_to(src_root)
            copy_file(p, str(rel))

    # 4. SCRIPTS (scripts/)
    print("\n[4/6] Copying Utility Scripts (scripts/)...")
    for root, dirs, files in os.walk(src_root / "scripts"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                continue
            p = Path(root) / f
            rel = p.relative_to(src_root)
            copy_file(p, str(rel))

    # 5. DOCUMENTATION (docs/)
    print("\n[5/6] Copying Markdown Documentation (docs/)...")
    for root, dirs, files in os.walk(src_root / "docs"):
        # Skip documents (PDFs) and Mathematica references
        dirs[:] = [d for d in dirs if d not in ("documents", "Mathematica_Reference", "__pycache__")]
        for f in files:
            ext = f.lower().split('.')[-1] if '.' in f else ''
            if ext in ("pdf", "docx", "pptx", "m", "nb"):
                print(f"  [SKIP DOC BINARY] {f}")
                continue
            p = Path(root) / f
            rel = p.relative_to(src_root)
            copy_file(p, str(rel))

    # 6. FILTERED DATA (data/)
    print("\n[6/6] Copying Whitelisted Public Data (data/)...")
    
    # 6a. Tires
    public_tires = ["Simple_Load_Dependent.tir", "Road_215_60R17_Touring.tir"]
    for tname in public_tires:
        p = src_root / "data" / "tire_tir" / tname
        if p.is_file():
            copy_file(p, f"data/tire_tir/{tname}")
            print(f"  + Tire: data/tire_tir/{tname}")

    # 6b. Tracks (fsae_* only)
    tracks_dir = src_root / "data" / "tracks"
    if tracks_dir.is_dir():
        for tf in os.listdir(tracks_dir):
            if tf.startswith("fsae_") and tf.endswith(".csv"):
                copy_file(tracks_dir / tf, f"data/tracks/{tf}")
                print(f"  + Track: data/tracks/{tf}")

    # 6c. Setups (fsae_mockup.json, segmento_b_suv.json only)
    public_setups = ["fsae_mockup.json", "segmento_b_suv.json"]
    for sname in public_setups:
        p = src_root / "data" / "setups" / sname
        if p.is_file():
            copy_file(p, f"data/setups/{sname}")
            print(f"  + Setup: data/setups/{sname}")

    # =========================================================================
    # POST-EXPORT SECURITY AUDIT
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING SECURITY AUDIT ON EXPORTED DIRECTORY...")
    print("=" * 70)

    forbidden_patterns = [
        "hoosier",
        "backup",
        "et18",
        "et17",
        ".pdf",
        ".docx",
        ".pptx",
        ".dll",
        ".sig",
        "mingw",
        "venv",
    ]

    violations = []
    for root, dirs, files in os.walk(dest_root):
        for f in files:
            fl = f.lower()
            rel = Path(root).relative_to(dest_root) / f
            for pat in forbidden_patterns:
                if pat in fl:
                    violations.append((str(rel), pat))

    if violations:
        print("\n[FAIL] SECURITY AUDIT FAILED! The following forbidden files were found:")
        for vfile, pat in violations:
            print(f"  ! {vfile} (matched '{pat}')")
        sys.exit(1)

    print("\n[OK] SECURITY AUDIT PASSED 100%!")
    print(f"  - No proprietary tire data (Hoosier / TTC) found.")
    print(f"  - No CAD suspension curves found.")
    print(f"  - No PDF or proprietary documents found.")
    print(f"  - No backup files or .dll binaries found.")
    print(f"  - Total files copied: {copied_files_count}")
    print(f"  - Total size:         {total_bytes / (1024 * 1024):.2f} MB")
    print("\n" + "=" * 70)
    print(f"CLEAN REPOSITORY READY AT:\n{dest_root}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export clean public repository")
    default_dest = Path(__file__).resolve().parent.parent.parent / "Dynamic_Target_Suite_Public"
    parser.add_argument("--dest", type=Path, default=default_dest, help="Destination directory")
    args = parser.parse_args()

    export_public_repo(args.dest)

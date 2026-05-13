#!/usr/bin/env python3
"""Colorized wrapper for `snakemake --list`.

Usage:
    python scripts/smk_list.py
    python scripts/smk_list.py --list-rules   # pass extra snakemake flags
"""

import subprocess
import sys

BOLD_CYAN  = "\033[1;36m"
DIM        = "\033[2m"
RESET      = "\033[0m"


def colorize(lines):
    in_desc = False
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped and not stripped.startswith(" "):
            in_desc = False
            if " (" in stripped:
                name, rest = stripped.split(" (", 1)
                print(f"{BOLD_CYAN}{name}{RESET} {DIM}({rest}{RESET}")
                in_desc = not stripped.endswith(")")
            else:
                print(f"{BOLD_CYAN}{stripped}{RESET}")
        else:
            # continuation line of a multi-line description
            print(f"{DIM}{stripped}{RESET}")


def main():
    extra = sys.argv[1:]
    result = subprocess.run(
        ["snakemake", "--list"] + extra,
        capture_output=True, text=True,
    )
    colorize(result.stdout.splitlines(keepends=True))
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()

"""Create venvs sharing the current Python's packages without downloading Torch.

Usage: python create_envs.py .venv/bsa .venv/causal
"""

from pathlib import Path
import site
import subprocess
import sys
import venv

import torch


def main():
    targets = [Path(path).absolute() for path in sys.argv[1:]]
    if len(targets) != 2 or targets[0] == targets[1]:
        raise SystemExit("Usage: python create_envs.py NEW_BSA_ENV NEW_CAUSAL_ENV")
    for target in targets:
        if target.exists():
            raise SystemExit(f"Environment already exists; choose a new directory: {target}")

    print(f"Base Python: {sys.executable}", flush=True)
    print(f"Base Torch: {torch.__version__} ({torch.__file__})", flush=True)
    # addsitedir processes .pth files too, including CUTLASS package registrations.
    # New environments' own packages precede these shared base directories.
    registration = "import site; " + "; ".join(
        f"site.addsitedir({path!r})" for path in site.getsitepackages()
    )
    for target in targets:
        venv.EnvBuilder(with_pip=True).create(target)
        python = str(target / "bin/python")
        destination = subprocess.check_output(
            [python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
        Path(destination, "zz_base_packages.pth").write_text(registration + "\n")
        subprocess.run([
            python, "-c",
            "import sys, torch\n"
            "assert torch.__file__ == sys.argv[1], 'Torch must be reused from the base environment'\n"
            "print(sys.executable, torch.__version__, torch.__file__)",
            torch.__file__,
        ], check=True)


if __name__ == "__main__":
    main()

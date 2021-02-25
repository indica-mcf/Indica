import pathlib
import subprocess

try:
    git_hash = subprocess.check_output(
        ["git", "describe", "--always"], text=True
    ).strip()
    git_diff = subprocess.check_output(
        ["git", "diff", "HEAD", "--", "indica"], text=True
    ).strip()
    if len(git_diff) > 0:
        git_hash += "-dirty"
except FileNotFoundError:
    git_hash = "UNKNOWN"
with open(pathlib.Path("indica") / "git_version", "w") as f:
    f.write(git_hash)

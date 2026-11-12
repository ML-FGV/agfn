import os
import subprocess
import warnings
from io import BytesIO

import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in cast",
)


def support_graphics():
    is_kitty = os.environ.get("TERM", "").startswith("xterm-kitty")
    is_ghostty = os.environ.get("TERM", "").startswith("xterm-ghostty")
    return is_kitty or is_ghostty


def show_kitty():
    if not support_graphics():
        return

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Pipe directly to kitty
    subprocess.run(["kitty", "+kitten", "icat"], input=buf.getvalue())
    plt.close()

"""Build the static trace viewer from a run's trace files.

    python scripts/render_traces.py outputs/eval/traces outputs/eval/traces.html
"""

from __future__ import annotations

import argparse

from toolagent.viewer import write_viewer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_dir")
    ap.add_argument("out_html")
    a = ap.parse_args()
    print(f"wrote {write_viewer(a.trace_dir, a.out_html)} episodes to {a.out_html}")


if __name__ == "__main__":
    main()

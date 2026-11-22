"""Point the Hugging Face page's "Mark your own video" button at the live site.

    .venv/bin/python tools/publish_live_link.py https://xyz.trycloudflare.com   # online
    .venv/bin/python tools/publish_live_link.py ""                              # offline

Writes live.json ({"url": ..., "updated": ...}) to the Space; the read-only site
reads it (and pings the link) to show or hide the button. Called by
tools/share_site.sh on start and stop (user's request, 13 Nov 2026). Uses the
Hugging Face login already on this Mac (`hf auth login`).
"""

from __future__ import annotations

import datetime
import json
import os
import sys

from huggingface_hub import HfApi

SPACE = os.environ.get("MMPQUANT_SPACE", "Frankyharsh08/mmp-quant")


def main() -> None:
    url = (sys.argv[1] if len(sys.argv) > 1 else "").strip().rstrip("/") or None
    body = {"url": url, "updated": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}
    HfApi().upload_file(
        path_or_fileobj=json.dumps(body).encode(), path_in_repo="live.json",
        repo_id=SPACE, repo_type="space",
        commit_message=f"live site {'online' if url else 'offline'}",
    )
    print(f"Hugging Face button: {'-> ' + url if url else 'offline'}")


if __name__ == "__main__":
    main()

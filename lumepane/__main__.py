"""Start the local LumePane image examination desk."""

import os
from waitress import serve

from .web import create_app

serve(
    create_app(os.environ.get("LUMEPANE_DATA", "data")),
    host=os.environ.get("HOST", "127.0.0.1"),
    port=int(os.environ.get("PORT", 19953)),
    threads=int(os.environ.get("WAITRESS_THREADS", 4)),
)

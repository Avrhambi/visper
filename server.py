# Thin shim — allows `python server.py` from the repo root.
# The real implementation lives in local_stt_he/server.py.
from local_stt_he.server import main

if __name__ == "__main__":
    main()

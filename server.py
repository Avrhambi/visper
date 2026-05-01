# Thin shim — allows `python server.py` from the repo root.
# The real implementation lives in visper/server.py.
from visper.server import main

if __name__ == "__main__":
    main()

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admirl_server import create_app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("ADMIRL_MODEL_SERVER_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

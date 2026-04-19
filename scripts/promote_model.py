"""CLI to manage model stage transitions."""
import sys

sys.path.insert(0, "src")

from churn.models.registry import archive_version, list_versions, promote_to_production, promote_to_staging

CMD_MAP = {
    "staging": promote_to_staging,
    "production": promote_to_production,
    "archive": archive_version,
    "list": lambda _: list_versions(),
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: promote_model.py <list|staging|production|archive> [version]")
        sys.exit(1)
    cmd = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else None
    CMD_MAP[cmd](version)

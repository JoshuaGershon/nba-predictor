import os
from dotenv import load_dotenv
from datetime import datetime, date

load_dotenv()  # loads .env file

def get_env(name: str, default: str | None = None) -> str:
    """Fetch environment variable or raise error if missing."""
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def to_datestr(d: date | datetime | str | None) -> str:
    """Convert date/datetime to string (YYYY-MM-DD)."""
    if d is None:
        return date.today().isoformat()
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return d


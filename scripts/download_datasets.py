"""Download the original datasets. Run from the repository root."""

from pathlib import Path
from zipfile import ZipFile
import io

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
HILLSTROM_URL = (
    "https://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)


def download_retail() -> None:
    """Extract just the public UCI workbook, preserving its official file name."""
    destination = RAW / "online_retail_II.xlsx"
    if destination.exists():
        print(f"Already present: {destination}")
        return
    print("Downloading Online Retail II from UCI...")
    response = requests.get(UCI_URL, timeout=180)
    response.raise_for_status()
    with ZipFile(io.BytesIO(response.content)) as archive:
        filenames = [n for n in archive.namelist() if n.lower().endswith(".xlsx")]
        if len(filenames) != 1:
            raise ValueError(f"Expected one .xlsx in UCI archive, found {filenames}")
        with archive.open(filenames[0]) as infile, destination.open("wb") as outfile:
            # Stream extraction rather than loading an entire Excel workbook into memory.
            import shutil
            shutil.copyfileobj(infile, outfile)
    print(f"Saved: {destination}")


def download_hillstrom() -> None:
    """Download the experiment CSV; refuse an HTML error page masquerading as CSV."""
    destination = RAW / "hillstrom.csv"
    if destination.exists():
        print(f"Already present: {destination}")
        return
    print("Downloading Hillstrom randomized experiment from the author...")
    response = requests.get(HILLSTROM_URL, timeout=90)
    response.raise_for_status()
    first_line = response.text.splitlines()[0].lower()
    if "segment" not in first_line or "conversion" not in first_line:
        raise ValueError("Hillstrom download does not look like the expected CSV header")
    destination.write_bytes(response.content)
    print(f"Saved: {destination}")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, fetch in (("Retail II", download_retail), ("Hillstrom", download_hillstrom)):
        try:
            fetch()
        except (requests.RequestException, OSError, ValueError, RuntimeError) as exc:
            # Permit manual placement if either source server is temporarily unavailable.
            print(f"ERROR downloading {name}: {exc}")
            print("See the manual download instructions in README.md")

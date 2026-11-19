"""Download the original datasets. Run from the repository root."""

from pathlib import Path
from zipfile import ZipFile
import io
import truststore
truststore.inject_into_ssl()
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
    """Download Hillstrom, using GitHub as a fallback."""
    destination = RAW / "hillstrom.csv"

    # Skip downloading if the dataset already exists.
    if destination.exists():
        print(f"Already present: {destination}")
        return

    # Try the original website first, then the GitHub mirror.
    mirror_url = (
        "https://raw.githubusercontent.com/W-Tran/uplift-modelling/"
        "master/data/hillstrom/"
        "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    )

    for url in (HILLSTROM_URL, mirror_url):
        try:
            # Request the dataset.
            response = requests.get(url, timeout=90)
            response.raise_for_status()

            # Verify that the response looks like our expected CSV.
            first_line = response.text.splitlines()[0].lower()
            if "segment" not in first_line or "conversion" not in first_line:
                raise ValueError("Unexpected CSV header")

            # Save the validated dataset.
            destination.write_bytes(response.content)
            print(f"Saved: {destination}")
            return

        except (requests.RequestException, ValueError) as exc:
            # Try the next source if this one fails.
            print(f"Download failed: {exc}")

    raise RuntimeError("Both Hillstrom download sources failed")



if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, fetch in (("Retail II", download_retail), ("Hillstrom", download_hillstrom)):
        try:
            fetch()
        except (requests.RequestException, OSError, ValueError, RuntimeError) as exc:
            # Permit manual placement if either source server is temporarily unavailable.
            print(f"ERROR downloading {name}: {exc}")
            print("See the manual download instructions in README.md")

"""Download and verify the fixed upstream archive; never overwrite raw data."""
import _bootstrap
from blood_cell_detection.data import fetch

if __name__ == "__main__":
    fetch()

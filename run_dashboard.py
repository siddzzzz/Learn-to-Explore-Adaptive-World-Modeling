"""Launcher script for the HEx-WM Interactive Scientific Studio."""

import subprocess
import sys


def main():
    print("Launching HEx-WM Interactive Scientific Studio...")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStudio shutdown cleanly.")


if __name__ == "__main__":
    main()

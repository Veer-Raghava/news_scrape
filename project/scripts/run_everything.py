import time
import subprocess
import os

# Path to project root
BASE = os.path.dirname(os.path.dirname(__file__))

RSS_SCRIPT = os.path.join(BASE, "scripts", "rss_ingest.py")
SCRAPER_SCRIPT = os.path.join(BASE, "scripts", "scraper.py")

def run():
    while True:
        print("\n🚀 Running RSS ingest...\n")
        subprocess.run(["python", RSS_SCRIPT])

        print("\n📰 Running Scraper...\n")
        subprocess.run(["python", SCRAPER_SCRIPT])

        print("\n⏳ Sleeping for 3 hours...\n")
        time.sleep(3 * 60 * 60)  # 3 hours = 10800 seconds

if __name__ == "__main__":
    run()


#to run ts in background.. go to powersherll and paste this -->  start /min python -m project.scripts.run_everything
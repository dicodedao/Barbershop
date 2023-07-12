import os
import time
from pathlib import Path

def main():
    media = os.path.join(Path(__file__).resolve().parents[1], 'media')
    print(f"cronjob run at {time.time()}")
    remove_expired_files_in_dir(os.path.join(media, 'user_input'), 1)
    remove_expired_files_in_dir(os.path.join(media, 'user_output'), 1)

def remove_expired_files_in_dir(dir, minutes):
    cutoff = time.time() - minutes * 60
    for fn in os.listdir(dir):
        path = os.path.join(dir, fn)
        if os.path.isfile(path):
            if os.path.getmtime(path) < cutoff:
                os.remove(path)            

if __name__ == "__main__":
    main()
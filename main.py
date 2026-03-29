from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def run_step(module_name: str) -> None:
    cmd = [sys.executable, "-m", module_name]
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)

if __name__ == "__main__":
    run_step("src.models.train_classifier")
    run_step("src.models.segmentation")
    print("\nDone. Check the artifacts/ folder for outputs.")

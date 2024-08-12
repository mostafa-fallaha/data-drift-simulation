import subprocess
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run DVC and Git commands.")
parser.add_argument("commit_message", type=str, help="Commit message for Git")
args = parser.parse_args()

# Define variables
data_file = "new_data/Google-Playstore.csv"
dvc_file = "new_data/Google-Playstore.csv.dvc"
gitignore_file = "new_data/.gitignore"
# script_file = "version_data_apps.py"
commit_message = args.commit_message

# Run DVC and Git commands
subprocess.run(["dvc", "add", data_file], check=True)
subprocess.run(["git", "add", dvc_file, gitignore_file, 'run_versioning.ps1', '.gitignore', 'model_a.py',
                'test_data_b.py', 'version_new_data.py', 'README.md', 'new_data/new_data.ipynb'], check=True)
subprocess.run(["git", "commit", "-m", commit_message], check=True)
subprocess.run(["dvc", "push"], check=True)
subprocess.run(["git", "push"], check=True)

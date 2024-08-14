import subprocess
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run DVC and Git commands.")
parser.add_argument("commit_message", type=str, help="Commit message for Git")
args = parser.parse_args()

# Define variables
data_file = "data/new.csv"
dvc_file = "data/new.csv.dvc"
gitignore_file = "data/.gitignore"
# script_file = "version_data_apps.py"
commit_message = args.commit_message

# Run DVC and Git commands
subprocess.run(["dvc", "add", data_file], check=True)
subprocess.run(["git", "add", dvc_file, gitignore_file, 'run_versioning.ps1', '.gitignore', 'model_a.py',
                'test_data_b.py', 'train_on_data_b.py', 'version_new_data.py', 'README.md', 'images/',
                'new_data/new_data.ipynb', 'version_pipeline_data.py', 'online_train.py', 'diff_pipeline.py'], check=True)
subprocess.run(["git", "commit", "-m", commit_message], check=True)
subprocess.run(["dvc", "push"], check=True)
subprocess.run(["git", "push"], check=True)

subprocess.run(["python", "online_train.py"], check=True)
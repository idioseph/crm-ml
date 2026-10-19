import os
import subprocess
import shutil

# 1. Install uv in the Kaggle environment
os.system("curl -LsSf https://astral.sh/uv/install.sh | sh")
os.environ["PATH"] += os.pathsep + "/root/.local/bin"

# 2. Clone your GitHub repository to get the script and Excel file
# REPLACE THIS with your actual public GitHub URL
repo_url = "https://github.com/your-username/your-repo-name.git"
print("Downloading files from GitHub...")
subprocess.run(["git", "clone", repo_url, "project_files"], check=True)

# 3. Change directory into the downloaded folder
os.chdir("project_files")

# 4. Execute your pipeline
command = "uv run crm-ml-beta.py --data syne_field.xlsx --forecast 372 --ml xgb --out results_syne_beta"
print(f"Executing: {command}")
subprocess.run(command, shell=True, check=True)

# 5. Move the results to Kaggle's root working directory so it saves them for download
# Kaggle ONLY saves files that end up in /kaggle/working/
shutil.move("results_syne_beta", "/kaggle/working/results_syne_beta")
print("Pipeline finished. Results saved.")
from pathlib import Path
import subprocess
repo = Path(r"d:/Aayush_Acharya/Aayushweb/portfoliowebsite")
branches = []
for i in range(1, 11):
    branches.append(f"apr20-readme-{i:02d}-deployment-badges")
for i in range(1, 11):
    branches.append(f"apr21-readme-{i:02d}-update-achievements")
for i in range(1, 11):
    branches.append(f"apr23-readme-{i:02d}-fix-markdown")
found = []
for b in branches:
    try:
        out = subprocess.check_output(["git", "branch", "--list", b], cwd=repo, text=True).strip()
        if out:
            found.append(out)
    except subprocess.CalledProcessError:
        pass
Path(repo / 'branch_check.txt').write_text('\n'.join(found), encoding='utf-8')
print('done')

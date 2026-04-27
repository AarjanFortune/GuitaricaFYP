from pathlib import Path
import subprocess
repo = Path(r"d:/Aayush_Acharya/Aayushweb/portfoliowebsite")
branches = []
for i in range(1, 11):
    branches.append((f"apr20-readme-{i:02d}-deployment-badges", 20))
for i in range(1, 11):
    branches.append((f"apr21-readme-{i:02d}-update-achievements", 21))
for i in range(1, 11):
    branches.append((f"apr23-readme-{i:02d}-fix-markdown", 23))
marker = "\n---\n\n*Built with ❤️ and an unhealthy amount of late-night coding sessions*"
for branch, day in branches:
    subprocess.run(["git", "checkout", "main"], cwd=repo, check=True)
    subprocess.run(["git", "branch", "-D", branch], cwd=repo, check=False)
    subprocess.run(["git", "checkout", "-b", branch], cwd=repo, check=True)
    path = repo / "README.md"
    text = path.read_text(encoding="utf-8")
    if marker not in text:
        raise ValueError("Marker not found in README")
    insert = f"\n\n### Branch note: {branch}\n- Generated tiny README doc update for April {day}.\n"
    new_text = text.replace(marker, insert + marker, 1)
    path.write_text(new_text, encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", f"docs: add README branch note for {branch}"], cwd=repo, check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], cwd=repo, check=True)
subprocess.run(["git", "checkout", "main"], cwd=repo, check=True)
print("done")

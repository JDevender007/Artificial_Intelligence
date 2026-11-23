# Push this repo's project notebooks to Kaggle and run them.

The Kaggle CLI on this machine is already authenticated as **devenderj** via OAuth
(`kaggle config view`), so no token file is needed. If that ever expires:

```powershell
kaggle auth login        # opens the browser once; you sign in yourself
kaggle config view       # should print: username: devenderj
```

## One-time prerequisite: push the repo first

Every notebook starts with:

```python
!git clone --depth 1 https://github.com/JDevender007/Artificial_Intelligence.git
%cd Artificial_Intelligence/<project>
```

So the GitHub `main` branch must already contain the folder name the notebook
expects. After the `pN-*` -> `NN-*` rename, kernels will fail at `%cd` until the
rename commit is pushed. Commit + push first, then run this tool.

## Commands

```powershell
cd "D:\Projects\Artificial Intelligence"

# 1. See what would be pushed (writes metadata into tools/kaggle/_stage, no network)
.\tools\kaggle\push_kernels.ps1 -DryRun

# 2. Run one project (stage + push). Use the folder name.
.\tools\kaggle\push_kernels.ps1 -Only 04-intent-classification

# 3. Run everything (see quota note below)
.\tools\kaggle\push_kernels.ps1

# 4. Check status / download outputs
kaggle kernels status devenderj/p4-intent-runner
kaggle kernels output devenderj/p4-intent-runner -p 04-intent-classification/reports/kaggle_t4
```

`kernels.json` maps each project to its current Kaggle kernel ref and GPU flag.
Pushing a kernel whose `title` is the folder name makes Kaggle **re-slug** it, so
`devenderj/p2-dl-fundamentals-runner` becomes `devenderj/02-dl-fundamentals`.
That is intended (kernel names now match the folder rename), but it means the
`kernel_id` in `kernels.json` must be refreshed after each first push — a stale
ref makes Kaggle create a duplicate kernel instead of updating. Check the truth
with `kaggle kernels list --user devenderj`.

`11-credit-mlops` is intentionally absent: it is deployment work, not a training job.

## Quota and hardware

- Kaggle free tier: ~30 GPU hours/week, max 2 GPU sessions running at once.
  The script pushes sequentially, but sessions still queue.
- T4 x2 (16 GB each) is best for P07/P08/P09. P01/P05/P10 are CPU jobs and gain
  nothing from a GPU.
- Colab free T4 (16 GB, ~12 GB RAM) can run P02/P04/P06 smoke runs; 7B QLoRA
  (P07/P08) will OOM or crawl there. Use Kaggle for those.

## Colab (you open these yourself)

Colab reads the notebook straight from the public GitHub repo:

```
https://colab.research.google.com/github/JDevender007/Artificial_Intelligence/blob/main/07-rag-qlora/notebooks/kaggle_run.ipynb
```

Swap `07-rag-qlora` for any project folder. These links 404 until the rename is
pushed, and the notebook's `git clone` + `%cd` must match the folder on `main`.

## Why I do not drive your Chrome profile

Signing into Google/Kaggle with your account requires your password and 2FA, and
automating a signed-in Google account breaks Google's terms. The Kaggle CLI is
already authenticated, which is the supported path and needs no browser session.

# PAI Project Rules

- Source first: use the task DOCX, NeoRL paper, official NeoRL repository, and fixed official benchmark commits before inference or secondary material.
- Official code first: reuse or faithfully reproduce official environment, dataset, reward, and BC behavior. Label compatibility changes.
- Keep three modules separate: world-model construction, frozen-world-model strategy optimization, and simulator evaluation.
- Reuse the verified GPU environment. Do not modify the working base Conda environment or install a system CUDA toolkit without evidence that it is required.
- Distinguish `[TASK]`, `[PAPER]`, `[OFFICIAL CODE]`, `[VERIFIED LOCALLY]`, `[ENGINEERING CHOICE]`, and `[UNVERIFIED]` claims.
- Complete the IB-M-100 baseline pipeline before model comparisons or research extensions.
- Prefer module-boundary tests, reproducible configs, and stable Git checkpoints over fragmented experiments or frequent status-document edits.
- Never commit datasets, environments, checkpoints, third-party repositories, credentials, caches, or large logs.


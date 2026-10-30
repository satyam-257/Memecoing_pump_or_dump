# Model Training Git Sync Rule

Whenever model training is finished (such as executing `train_and_submit.py`, training models via CV, or running pipeline experiments):
1. **Verify Staging**: Ensure only code, configuration, experiment notes, and documentation are committed (keep dataset CSVs and output caches ignored via `.gitignore`).
2. **Commit**: Create a commit with a descriptive message detailing the model training run, updated parameters, or validation metrics.
3. **Push**: Automatically push the commit to the GitHub remote repository (`git push origin main`).

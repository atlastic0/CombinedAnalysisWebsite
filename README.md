# Combined Security Analysis Website

This project combines two separate tools behind one home page:

- **Smishing Detector** at `/smishing`: paste an SMS message and classify it using the supplied saved CatBoost model and matching TF-IDF vectorizer.
- **Dataset Analyzer** at `/dataset-analyzer`: upload a CSV to inspect data quality, missing values, repeated values, duplicate rows, recommendations, and a preview.

## Run in VS Code

1. Extract this folder.
2. Open the extracted `combined_security_analyzer` folder in VS Code.
3. Open the VS Code terminal and install dependencies:

   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

   On macOS/Linux, activate with `source .venv/bin/activate`.

4. Confirm these two model files remain beside `app.py`:
   - `smishing_model.cbm`
   - `tfidf_vectorizer.joblib`

   They are included in this project. The app loads them and does not retrain the model during startup.

5. Start the website:

   ```bash
   python app.py
   ```

6. Open http://127.0.0.1:5000.

## Pages

- `/` — Home page
- `/smishing` — Smishing Detector
- `/dataset-analyzer` — Dataset Analyzer

## Notes

- The dataset analyzer accepts CSV files up to 150 MB.
- Target suggestions are heuristics, not guaranteed identification of the correct target column.
- The Fix Dataset button currently removes exact duplicate rows only. Review the downloaded file before using it.
- Smishing predictions are not guarantees. Do not click suspicious links or share sensitive information based only on a prediction.

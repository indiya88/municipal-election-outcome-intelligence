# Municipal Election Outcome Intelligence

A binary classification project estimating whether a candidate will be elected in a contested Canadian municipal race, with a FastAPI prediction service.

## Purpose
An early estimate could help campaign teams identify candidates whose prospects need closer investigation. Estimates should be combined with current local evidence before considering additional campaign support. This model does not identify which campaign intervention will work, establish election fairness, or guarantee a win.

## Data
Canadian Municipal Elections Database, Borealis, Version 6. DOI: https://doi.org/10.5683/SP2/4MZJPQ. The source contains 123,416 candidate records and 19 variables spanning 1867–2022. After removing missing outcomes and filtering to contested races, 91,768 records remain. Final vote totals and acclamation are excluded from predictors.

## Approach and recorded results
Training uses 83,857 records before 2022. Evaluation uses 7,911 records from 2022. Random Forest tuning uses three-fold validation grouped by race. The 2022 results also informed model selection, so a separate final evaluation is needed before operational use.

| Model | Accuracy | Precision | Recall | F1 (elected) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority-class baseline | 0.590 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.705 | 0.717 | 0.463 | 0.563 | 0.722 |
| Random Forest | 0.701 | 0.637 | 0.629 | 0.633 | 0.738 |
| Tuned Random Forest | 0.701 | 0.636 | 0.635 | 0.635 | 0.733 |

The tuned forest uses 120 trees. Tuning improved F1 by 0.002. The original forest had higher ROC-AUC; Logistic Regression had higher accuracy. Selection prioritized F1.

## Files
- `Municipal_Election_Outcome_Intelligence.ipynb`: original submitted notebook, including saved outputs.
- `app.py`: API implementation exported from the notebook.
- `requirements.txt`: package versions recorded in the notebook; these pins have not been independently installed here.
- `Dockerfile`: prepared container definition, not yet verified by running Docker.
- `sample_request.json`: illustrative candidate request used for the recorded API test.
- `recorded_response.json`: saved response from that test, not a live endpoint.
- `requirements-notebook.txt`: adds notebook and API-test dependencies.

## Generate the model
The trained `municipal_election_model.joblib` is not included in this repository package. In Google Colab:
1. Download the dataset ZIP from the Borealis source above.
2. Upload it as `elections-dataset.zip` to `/content`.
3. Open the notebook and run cells in order. The extraction cell expects `/content/elections-dataset.zip` and the data loader searches for `elec.dta`.
4. The export cell saves `municipal_election_model.joblib`. Download it and place it beside `app.py`.
5. Use compatible Python and the training package versions when loading the saved pipeline. Retraining may yield different results.

For local Jupyter, install `requirements-notebook.txt`, adapt the Colab ZIP-extraction path to your downloaded dataset, and rerun the workflow. Only load joblib files from trusted sources.

## Run the API locally
From this repository folder, after placing the model file beside `app.py`:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to try the request in Swagger UI. `GET /health` checks model availability. `POST /predict` accepts the fields in `sample_request.json` and returns a class, outcome label, and elected probability.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @sample_request.json
```

## Docker (prepared, not verified)
With the model file present:

```bash
docker build -t election-api .
docker run --rm -p 127.0.0.1:8000:8000 election-api
```

A WSL service failure prevented the original container test. These commands describe intended execution; successful container deployment is not claimed.

## Validation and limitations
Recorded notebook TestClient checks returned HTTP 200 for `/health` and `/predict`. The illustrative response was Elected with probability 0.5232. This verifies application behavior in process, not public hosting or election accuracy for that candidate.

Campaign spending, polling, and local issues are absent. Historical coverage is uneven. Gender is an input; group fairness checks remain outstanding. Probability calibration and independent evaluation are needed before treating scores as dependable real-world chances.

## Portfolio
https://renee-ramjet.com/municipal-election.html

Developed by Renee Ramjet. AI assistance supported explanations and debugging; the notebook records the project workflow and results.

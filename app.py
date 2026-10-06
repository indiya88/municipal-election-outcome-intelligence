# Import the packages required for the deployment service
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Load the complete trained pipeline
deployment_model = joblib.load("municipal_election_model.joblib")

# Create the FastAPI application
app = FastAPI(
    title="Municipal Election Outcome Intelligence API",
    description="Estimates a candidate's probability of being elected.",
    version="1.0.0",
)

# Define and validate the information accepted by the API
class CandidateInput(BaseModel):
    incumbent: int = Field(ge=0, le=1)
    election_year: int = Field(ge=1867, le=2100)
    candidate_count: int = Field(ge=2)
    seats_available: int = Field(ge=1)
    candidate_to_seat_ratio: float = Field(gt=0)
    municipality: str
    party: str
    gender: str
    position: str
    ward: str
    province: str
    region: str


@app.get("/health")
def health_check():
    """Confirm that the API and model are available."""
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict_election_outcome(candidate: CandidateInput):
    """Return the predicted election outcome and probability."""
    try:
        # Convert the submitted JSON information into a one-row DataFrame
        candidate_data = pd.DataFrame([candidate.model_dump()])

        # Generate the predicted class and probability
        predicted_class = int(deployment_model.predict(candidate_data)[0])
        probability_elected = float(
            deployment_model.predict_proba(candidate_data)[0, 1]
        )

        return {
            "predicted_class": predicted_class,
            "predicted_outcome": (
                "Elected" if predicted_class == 1 else "Not elected"
            ),
            "probability_elected": round(probability_elected, 4),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction could not be completed: {error}",
        )

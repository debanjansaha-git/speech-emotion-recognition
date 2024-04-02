from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI()

mlflow_uri = "http://mlflow:5000"


class TrainRequest(BaseModel):
    param_key: str
    param_value: str


### ===========================
###       DEFINE THE ROUTES
### ===========================
@app.get("/test-connection")
async def test_connection():
    async with httpx.AsyncClient() as client:
        # Attempt to fetch MLFlow's health or main page just to test connectivity
        response = await client.get(mlflow_uri)
    return {"mlflow_response": response.text}


@app.get("/get-run")
async def get_run(run_id: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{mlflow_uri}/api/2.0/mlflow/runs/get", params={"run_id": run_id}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=str(exc.response.text)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/train")  # Note the change to a POST method to accept a body.
async def train_model(train_request: TrainRequest):
    try:
        async with httpx.AsyncClient() as client:
            # Start an MLflow run
            create_response = await client.post(
                f"{mlflow_uri}/api/2.0/mlflow/runs/create", json={}
            )
            create_response.raise_for_status()
            run_id = create_response.json()["run"]["info"]["run_id"]

            # Log a parameter from the request
            log_response = await client.post(
                f"{mlflow_uri}/api/2.0/mlflow/runs/log-parameter",
                json={
                    "run_id": run_id,
                    "key": train_request.param_key,
                    "value": train_request.param_value,
                },
            )
            log_response.raise_for_status()

        return {"message": "Training started and parameter logged"}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=str(exc.response.text)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict")
def predict():
    # Add prediction logic here
    return {"message": "This is a stub for the prediction endpoint"}

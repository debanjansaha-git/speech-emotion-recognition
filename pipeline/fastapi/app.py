from fastapi import FastAPI
import httpx

app = FastAPI()

mlflow_uri = "http://mlflow:5000"


### ===========================
###       DEFINE THE ROUTES
### ===========================
@app.get("/test-connection")
async def test_connection():
    async with httpx.AsyncClient() as client:
        # Attempt to fetch MLFlow's health or main page just to test connectivity
        response = await client.get(mlflow_uri)
    return {"mlflow_response": response.text}


@app.get("/train")
async def train_model():
    async with httpx.AsyncClient() as client:
        # Example: Start an MLflow run
        response = await client.post(
            f"{mlflow_uri}/api/2.0/mlflow/runs/create", json={}
        )
        run_id = response.json()["run"]["info"]["run_id"]

        # Example: Log a parameter
        await client.post(
            f"{mlflow_uri}/api/2.0/mlflow/runs/log-parameter",
            json={"run_id": run_id, "key": "example_param", "value": "value"},
        )

    return {"message": "Training started and parameters logged"}


@app.get("/predict")
def predict():
    # Add prediction logic here
    return {"message": "This is a stub for the prediction endpoint"}

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import traceback
import os
import pandas as pd
import joblib

app = FastAPI()

# Allow CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerData(BaseModel):
    total_spent: float
    avg_spent: float
    spending_variability: float
    Age: int
    Stay_In_Current_City_Years: int
    Marital_Status: int
    City_A: int
    City_B: int
    City_C: int
    Occupation_Freq: float
    unique_categories: int
    Category_1: int
    Category_5: int
    Category_8: int
    Category_Other: int



# Train pipeline route
@app.get("/train", status_code=status.HTTP_200_OK, include_in_schema=False)
def training():
    try:
        os.system("python main.py")  # Runs your main training script
        return {"message": "Training Successful!"}
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(traceback.format_exc()))

# Predict existing customer using clustered_data.csv
@app.post("/predict_existing_customer", status_code=status.HTTP_200_OK)
def result(input_customer_id: int):
    try:
        clustered_data = pd.read_csv("artifacts/model_trainer/clustered_data.csv")
        cluster = clustered_data.loc[clustered_data['User_ID'] == input_customer_id, 'Cluster']
        if not cluster.empty:
            return {"customer_id": input_customer_id, "cluster": int(cluster.values[0])}
        else:
            raise HTTPException(status_code=404, detail="Customer ID not found")
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(traceback.format_exc()))

# Predict new customer using model.joblib
@app.post("/predict_new_customer", status_code=status.HTTP_200_OK)
def predict_new_customer(customer_data: CustomerData):
    try:
        model = joblib.load("artifacts/model_trainer/model.joblib")
        input_data = pd.DataFrame([customer_data])  # Convert input data to DataFrame
        predicted_cluster = model.predict(input_data)
        return {"predicted_cluster": int(predicted_cluster[0])}
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(traceback.format_exc()))

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

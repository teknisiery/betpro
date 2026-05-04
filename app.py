from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pycaret.classification import *
import uvicorn
import shutil
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

model = None
target_col = ""

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/train")
async def train(file: UploadFile = File(...), target: str = Form(...)):
    global model, target_col
    target_col = target
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    df = pd.read_csv(file_path)
    exp = setup(df, target=target, verbose=False, session_id=42)
    best = compare_models(n_select=1, verbose=False)
    model = best
    save_model(model, "best_model")
    metrics = pull().iloc[0]['Accuracy']
    return JSONResponse({"accuracy": round(metrics, 4), "message": "Model siap!"})

@app.post("/predict")
async def predict(
    team_a: str = Form(...),
    team_b: str = Form(...),
    possession_a: float = Form(...),
    shots_a: int = Form(...),
    shots_b: int = Form(...),
    xg_a: float = Form(0.0),
    xg_b: float = Form(0.0)
):
    global model, target_col
    if model is None:
        return JSONResponse({"error": "Model belum dilatih"}, status_code=400)
    input_data = pd.DataFrame([{
        "team_a": team_a,
        "team_b": team_b,
        "possession_a": possession_a,
        "shots_a": shots_a,
        "shots_b": shots_b,
        "xg_a": xg_a,
        "xg_b": xg_b
    }])
    prediction = predict_model(model, data=input_data)
    result = prediction["Label"].iloc[0]
    return JSONResponse({"prediction": str(result)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

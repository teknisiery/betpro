from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import os
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

model = None
encoder_dict = {}
target_col = ""

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/train")
async def train(file: UploadFile = File(...), target: str = Form(...)):
    global model, target_col, encoder_dict
    target_col = target
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_csv(file_path)
    X = df.drop(columns=[target])
    y = df[target]

    X_encoded = X.copy()
    encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            encoders[col] = le
    encoder_dict = encoders

    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        encoder_dict['target'] = le_target

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    model = clf
    joblib.dump(model, "best_model.pkl")
    joblib.dump(encoder_dict, "encoders.pkl")

    return JSONResponse({"accuracy": round(acc, 4), "message": "Model siap!"})

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
    global model, encoder_dict, target_col
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

    for col, le in encoder_dict.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col])

    prediction = model.predict(input_data)[0]

    if 'target' in encoder_dict:
        prediction = encoder_dict['target'].inverse_transform([prediction])[0]

    return JSONResponse({"prediction": str(prediction)})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

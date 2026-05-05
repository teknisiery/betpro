from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import zipfile, os, shutil

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <h1>Debug ZIP</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="zip_file" accept=".zip">
      <button type="submit">Upload</button>
    </form>
    """

@app.post("/predict")
async def predict(zip_file: UploadFile = File(...)):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(temp_dir, "uploaded.zip")
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(zip_file.file, buffer)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
        shutil.rmtree(temp_dir)
        return JSONResponse({"files_in_zip": file_list})
    except Exception as e:
        shutil.rmtree(temp_dir)
        return JSONResponse({"error": str(e)}, status_code=400)

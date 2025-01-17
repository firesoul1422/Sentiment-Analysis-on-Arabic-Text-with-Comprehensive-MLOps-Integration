from fastapi import FastAPI
from inference import inference
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from config_loader import load_config

configs = load_config()


class Tweet(BaseModel):
    tweet: str
    
    

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def main():
    with open("static/index.html", "r") as file:
        return HTMLResponse(content=file.read())


@app.post("/predict")
def predict(tweet: Tweet):
    return inference(tweet.tweet)
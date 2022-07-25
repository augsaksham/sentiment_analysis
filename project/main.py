from fastapi import FastAPI
import predict as pre

app = FastAPI()

@app.get('/')
def hello():
    return {"Hello":"World"}


@app.post('/some/{text}')
def predict(text:str):

    result=pre.predict(text,r'saved_model/model.pth')
    return {"Input Text":text,"Result":result}    
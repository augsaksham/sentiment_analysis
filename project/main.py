#Main App File for the project
from fastapi import FastAPI
import predict as pre

app = FastAPI()

#Demo get function call
@app.get('/')
def hello():
    return {"Hello":"World"}


#Post api call to get the ML predicted label for a given text
@app.post('/some/{text}')
def predict(text:str):

    result=pre.predict(text,r'saved_model/model.pth')
    return {"Input Text":text,"Result":result}    
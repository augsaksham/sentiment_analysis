Sentiment App

This app offers a fullt trained BERT to allow sentiment analysis of a given text. The detection can be made using an API endpoint (Local) or (Cloud)

Installation

Local PC:

git clone
cd sentimentapp/project
pip install -r requirements.txt
Goto "saved_mode" dir and donload the pretrained model from readme
Open console and input : uvicorn main:app
This will start a local server at localhost:8000

Goto localhost:8000/docs to see the API Documentation

Docker

git clone
cd sentimentapp/project
docker build . -t dev_foundry
docker run -d --name dev_curr -p 80:80 dev_foundry
This will start a local server at localhost:80

Goto localhost:80/docs to see the API Documentation

Use the App Without Installation

This App is currently hosted on Google Cloud and hence can be used directly.

Goto: https://foundry-6nmnvycfcq-uc.a.run.app/docs#/ to see the API Documentation

Note : If you want to host your own image on cloud follow this:

Make a docker image of the project (Follow docker_instructions in the Instructions folder)
Follow "cloud_installation.txt" in the Instructions folder
Features

Robust text classification
Live cloud hosted module for 24x7 support
Documented API for easier usage

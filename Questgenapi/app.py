import os

from fastapi import FastAPI
from pprint import pprint
from Questgen import main

app = FastAPI()


try:
    os.rm('C:\\Users\\user\\AppData\\Roaming\\nltk_data\\corpora\\stopwords\\hinglish')
except:
    pass

def loadModel():
    qg = main.QGen()
    return qg

@app.get("/")
async def root():
    return {"message": "Welcome to QuestGen API"}


@app.get("/boolgen")
async def Boolgen(input_text: str):
    payload = {
        "input_text": input_text
    }
    qe = main.BoolQGen()
    output = qe.predict_boolq(payload)

    return output

@app.get("/anspred")
async def answerPredictor(input_text: str, input_question: str):
    payload = {
        "input_text": input_text, "input_question": input_question
    }
    answer = main.AnswerPredictor()
    output = answer.predict_answer(payload)

    return output


@app.get("/mcqquest")
async def mcgQuest(input_text: str):
    payload = {
        "input_text": input_text
    }
    qg = loadModel()
    output = qg.predict_mcq(payload)

    return output


@app.get("/paraquest")
async def paraQuest(input_text: str, max_questions: int):
    payload = {
        "input_text": input_text,
        "max_questions": max_questions
    }
    qg = loadModel()
    output = qg.paraphrase(payload)
    return output
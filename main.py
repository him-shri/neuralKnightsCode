from fastapi import FastAPI, HTTPException

from EnglishSpanishFrench.EnglishSpanishFrench import testLanguageProbability
from HappyOrAngry.HappyOrAngry import testEmotionalProbability
from HumanOrRobot.HumanOrRobot import testSystemProbability
from models.common import AnalysisResponse, AudioSample
description = """
"""
app = FastAPI(
title="neural_knigfts",
description=description,
version="0.0.1",
terms_of_service="",
contact={
        "name": "xyz",
    },
)

# Define a route
@app.post("/voice/analyze", response_model=AnalysisResponse)
def analyze_audio(audio_sample: AudioSample):

    # Here you would add the logic to analyze the audio file
    # For now, we'll just return a dummy response

    # if audio_sample.sample != "audio.wav":
    #     raise HTTPException(status_code=400, detail="Invalid audio file")

    result = testLanguageProbability(audio_sample.sample)
    print('val->',result)
    happyOrAngry = testEmotionalProbability(audio_sample.sample)
    humanOrRobot = testSystemProbability(audio_sample.sample)

    return {
        "status": "success",
        "analysis": {
            "detectedVoice": True,
            "voiceType": humanOrRobot[1],
            "confidenceScore": {
                'aiProbability':humanOrRobot[0].aiProbability, 'humanProbability':humanOrRobot[0].humanProbability, 'accuracy':result.accuracy, 'spanishProbability':result.spanishProbability, 'frenchProbability':result.frenchProbability, 'englishProbability':result.englishProbability, 'language':result.language
            },
            "additionalInfo": {
                "emotionalTone": happyOrAngry.emotionalTone,
                "backgroundNoiseLevel": "low"
            }
        },
        "responseTime": 200
    }
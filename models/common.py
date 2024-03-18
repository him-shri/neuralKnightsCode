from pydantic import BaseModel
from typing import Optional

import json

class AudioSample(BaseModel):
    sample: str

class ConfidenceScore(BaseModel):
    aiProbability: float
    humanProbability: float
    language: str
    accuracy: float
    spanishProbability: float
    englishProbability: float
    frenchProbability: float

    def to_json(self):
        return json.dump()

class AdditionalInfo(BaseModel):
    emotionalTone: str
    backgroundNoiseLevel: str
 
class Analysis(BaseModel):
    detectedVoice: bool
    voiceType: str
    confidenceScore: ConfidenceScore
    additionalInfo: AdditionalInfo

class AnalysisResponse(BaseModel):
    status: str
    analysis: Analysis
    responseTime: int





from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from run import run

app = FastAPI(title="Video Moment Retrieval API")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

origins = [
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceRequest(BaseModel):
    video_path: str
    query: str


class InferenceResponse(BaseModel):
    start_time: float
    end_time: float
    score : float



@app.post("/inference")
def inference(req: InferenceRequest):
    try:

        output = run(req.video_path, req.query)
        
        result = InferenceResponse(
            start_time=output[0],
            end_time=output[1],
            score=output[2]
        )

        return result
    except Exception as e:
        logger.error(f"추론 실패: {e}")
        return {"error": str(e)}





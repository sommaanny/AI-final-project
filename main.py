from fastapi import FastAPI
from pydantic import BaseModel
import logging

from run import run_example

app = FastAPI(title="Video Moment Retrieval API")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class InferenceRequest(BaseModel):
    video_path: str
    query: str


class InferenceResponse(BaseModel):
    query: str
    start_time: float
    end_time: float
    duration: float


@app.post("/inference")
def inference(req: InferenceRequest):
    try:

        run_example()

        return "ok"
    except Exception as e:
        logger.error(f"추론 실패: {e}")
        return {"error": str(e)}





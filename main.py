from fastapi import FastAPI
from pydantic import BaseModel
import logging

from inference import ModelLoader, run_inference

app = FastAPI(title="Video Moment Retrieval API")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 전역 모델 로드 (서버 시작 시 1회)
loader = ModelLoader()
clip_handler, qd_detr_model, adapter, device, args = loader.get_models()

class InferenceRequest(BaseModel):
    video_path: str
    query: str


class InferenceResponse(BaseModel):
    query: str
    start_time: float
    end_time: float
    duration: float


@app.post("/inference", response_model=InferenceResponse)
def inference(req: InferenceRequest):
    try:
        start, end = run_inference(
            clip_handler, qd_detr_model, adapter, args, req.video_path, req.query, device
        )

        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            logger.warning("추론 값 에러 Nan")
            start, end = 0.0, 0.0

        duration = round(end - start, 2)
        logger.info(f"결과: {start:.2f} ~ {end:.2f} ({duration:.2f}s)")


        return InferenceResponse(
            query=req.query,
            start_time=round(start, 2),
            end_time=round(end, 2),
            duration=duration,
        )
    except Exception as e:
        logger.error(f"추론 실패: {e}")
        return {"error": str(e)}





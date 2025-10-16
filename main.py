import torch
import numpy as np
import time
import cv2


from model.clip_model import ModelHandler as ClipHandler
from model.qd_detr_model import build_model
from mlp_adapter import MLPAdapter
from model_config import ModelConfig

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def get_video_duration(video_path: str) -> float:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration
    except Exception: return 0


def run_inference(clip_handler, qd_detr_model, video_path, query, device):
    #추론 함수
    print("CLIP 피처 추출 중...")
    text_features = clip_handler.get_text_feature([query])
    video_features_512 = clip_handler.get_video_feature(video_path)
    logger.info(f"텍스트 피처: {text_features.shape}, 비디오 피처:")

    #mlp adapter 적용하여 차원수 조절
    adapter = MLPAdapter(512, 2818).to(device)
    video_features = adapter(video_features_512)

    logger.info("2단계: QD-DETR 입력 데이터 가공 중...")
    src_vid = video_features.unsqueeze(0).to(device)
    src_txt = text_features.unsqueeze(1).to(device)

    # 마스크 생성: 모델은 0을 유효, 1을 패딩으로 처리합니다. 모든 부분이 유효하므로 0으로 채웁니다.
    src_vid_mask = torch.zeros(src_vid.shape[:2], dtype=torch.bool, device=device)
    src_txt_mask = torch.zeros(src_txt.shape[:2], dtype=torch.bool, device=device)
    
    logger.info("입력 데이터 및 마스크 준비 완료.")
    
    logger.info("QD-DETR 추론 중...")
    with torch.no_grad():
        outputs = qd_detr_model(src_txt=src_txt, src_txt_mask=src_txt_mask, src_vid=src_vid, src_vid_mask=src_vid_mask)

    # 결과 후처리
    pred_spans = outputs["pred_spans"]
    pred_logits = outputs["pred_logits"]
    prob_foreground = pred_logits.softmax(-1)[:, :, 0]
    best_query_idx = prob_foreground.argmax(dim=1)
    best_span = pred_spans[0, best_query_idx].squeeze().cpu().numpy()
    center, width = best_span[0], best_span[1]
    video_duration = get_video_duration(video_path)
    start_time = max(0, (center - width / 2) * video_duration)
    end_time = min(video_duration, (center + width / 2) * video_duration)
    logger.info("추론 및 후처리 완료.")
    return float(start_time), float(end_time)


def main():
    logger.info("애플리케이션 시작: 모델 로딩 중...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # CLIP 로드
    clip_handler = ClipHandler()
    
    # 모델 초기화에 필요한 파라미터가 너무 많은 관계로 클래스를 정의해서 함수에 넘겨줌
    args_for_build = ModelConfig()
    args_for_build.device = device
    
    # 모델 초기화
    qd_detr_model, _ = build_model(args_for_build)
    
    try:
        logger.info(f"'{args_for_build.CKPT_PATH}'에서 가중치 로딩 중...")
        checkpoint = torch.load(args_for_build.CKPT_PATH, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        qd_detr_model.load_state_dict(clean_state_dict, strict=False)
        qd_detr_model.to(device)
        qd_detr_model.eval()
        logger.info("모든 모델 로딩 완료!\n")
    except FileNotFoundError:
        logger.info(f"[오류] 가중치 파일을 찾을 수 없습니다: '{args_for_build.CKPT_PATH}'")
        return
    except Exception as e:
        logger.info(f"[오류] 모델 로딩 중 오류 발생: {e}")
        return

    # 추론 파이프라인 실행
    # 테스트를 위해 변수로 사용, 실제 구현시 파라미터화로 받아올 예정
    TEST_VIDEO_PATH = "test_video2.mp4"
    TEST_QUERY = "aaaa"

    try:
        start_time = time.time()
        start, end = run_inference(clip_handler, qd_detr_model, TEST_VIDEO_PATH, TEST_QUERY, device)
        end_time = time.time()
        
        logger.info("\n" + "="*50)
        logger.info("최종 예측 결과")
        logger.info(f"입력 텍스트: '{TEST_QUERY}'")
        logger.info(f"예측된 시간: {start:.2f}초 ~ {end:.2f}초")
        logger.info(f"(총 소요 시간: {end_time - start_time:.2f}초)")
        logger.info("="*50)

    except Exception as e:
        logger.info(f"파이프라인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



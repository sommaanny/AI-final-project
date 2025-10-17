import torch
import torch.nn.functional as F
import numpy as np
import time
import math
from typing import Tuple
import cv2

from qd_detr.span_utils import span_cxw_to_xx
from model.clip_model import ModelHandler as ClipHandler
from model.qd_detr_model import build_model
from mlp_adapter import MLPAdapter
from model_config import ModelConfig


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

#------------------------------------------------------------------------------#
TEXT_FEAT_PATH = "./features/clip_text_features/qid5263.npz"
CLIP_VID_FEAT_PATH = "./features/clip_video_features/qtyBzFV9yTs_60.0_210.0.npz"
SLOWFAST_VID_FEAT_PATH = "./features/slowfast_video_features/qtyBzFV9yTs_60.0_210.0.npz"
    
def load_npz_feature(file_path):
    data = np.load(file_path)
    if 'pooler_output' in data: return data['pooler_output']
    keys = ['features', 'feat', 'feature', 'data'];
    for key in keys:
        if key in data: return data[key]
    if data.files: return data[data.files[0]]
    raise ValueError(f"'{file_path}'에서 데이터를 찾을 수 없습니다.")
#------------------------------------------------------------------------------#

def get_video_duration(video_path: str) -> float:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0:
            return 0.0
        return float(frame_count) / float(fps)
    except Exception:
        return 0.0
    

@torch.no_grad()
def run_inference(clip_handler,
                  qd_detr_model,
                  adapter,
                  args,
                  video_path: str,
                  query: str,
                  device) -> Tuple[float, float]:
    """
    1) clip_handler로부터 (text_feature, video_feature) 추출
    2) adapter로 차원 맞추기 (필요 시)
    3) QD-DETR에 넣고 outputs 얻기
    4) span_loss_type에 따라 디코딩 (l1 또는 ce)
    5) NaN/inf 방어, 안전하게 start,end 반환
    """
    try:
        logger.info("CLIP 피처 추출 중...")
        # --- 텍스트 피처 (shape: [1, T_dim] or [T_dim])
        text_feat = clip_handler.get_text_feature([query])  # 기대: tensor [1, dim] 또는 [dim]
        if isinstance(text_feat, torch.Tensor):
            text_feat = text_feat.to(device)
        else:
            raise RuntimeError("clip_handler.get_text_feature did not return torch.Tensor")

        # --- 비디오 피처 (CLIP) (shape: [N_clips, 512])
        vid_feat_clip = clip_handler.get_video_feature(video_path)  # 기대: tensor [N, 512]
        if not isinstance(vid_feat_clip, torch.Tensor):
            raise RuntimeError("clip_handler.get_video_feature did not return torch.Tensor")
        vid_feat_clip = vid_feat_clip.to(device)

        # Optional: if model expects multi-modal vid feat (e.g., CLIP concat SlowFast),
        # you should extract slowfast features here too. For now we assume single-source clip features
        # If your QD-DETR expects larger v_feat_dim, use adapter to map [N,512] -> [N, v_dim]
        v_expected = getattr(args, "v_feat_dim", None)
        if v_expected is not None and vid_feat_clip.shape[-1] != v_expected:
            # use adapter path-by-path
            try:
                vid_feat = adapter(vid_feat_clip)  # adapter maps to correct dim
            except Exception:
                # if adapter expects batch dim etc, ensure shape
                vid_feat = adapter(vid_feat_clip.float())
        else:
            vid_feat = vid_feat_clip.float()

        logger.info(f"텍스트 피처: {tuple(text_feat.shape)}, 비디오 피처: {tuple(vid_feat.shape)}")

        #---------------피처 직접 테스트-----------------------------#
        text_feat = load_npz_feature(TEXT_FEAT_PATH)
        clip_video_feat = load_npz_feature(CLIP_VID_FEAT_PATH)
        slowfast_video_feat = load_npz_feature(SLOWFAST_VID_FEAT_PATH)

        # --- 2. 정규화
        if len(text_feat.shape) == 1:
            text_feat = np.expand_dims(text_feat, axis=0)

        clip_video_feat /= np.linalg.norm(clip_video_feat, axis=1, keepdims=True)
        slowfast_video_feat /= np.linalg.norm(slowfast_video_feat, axis=1, keepdims=True)
        text_feat /= np.linalg.norm(text_feat, axis=1, keepdims=True)

        # --- 3. 결합
        vid_feat = np.concatenate((clip_video_feat, slowfast_video_feat), axis=1)

        num_clips = vid_feat.shape[0]
        tef_st = np.linspace(0, 1, num_clips + 1)[:-1]
        tef_ed = np.linspace(0, 1, num_clips + 1)[1:]
        tef = np.stack([tef_st, tef_ed], axis=1)
        
        vid_feat = np.concatenate([vid_feat, tef], axis=1)

        # --- 4. Torch Tensor 변환 (★ 중요)
        vid_feat = torch.tensor(vid_feat, dtype=torch.float32)
        text_feat = torch.tensor(text_feat, dtype=torch.float32)

        #---------------피처 직접 테스트-----------------------------#


        # ----- Prepare src tensors expected by QD-DETR
        # src_vid: (batch=1, L_vid, D_vid)
        src_vid = vid_feat.unsqueeze(0) if vid_feat.dim() == 2 else vid_feat
        # src_txt: QD-DETR expects (batch=1, L_txt, D_txt)
        if text_feat.dim() == 1:
            src_txt = text_feat.unsqueeze(0).unsqueeze(0)
        elif text_feat.dim() == 2:
            # (1, D) -> (1, 1, D)
            if text_feat.shape[0] == 1:
                src_txt = text_feat.unsqueeze(1)  # (1, 1, D)
            else:
                src_txt = text_feat.unsqueeze(0)  # (1, L_txt, D)
        else:
            src_txt = text_feat

        src_vid = src_vid.to(device)
        src_txt = src_txt.to(device)
        src_vid_mask = torch.zeros(src_vid.shape[:2], dtype=torch.bool, device=device)
        src_txt_mask = torch.zeros(src_txt.shape[:2], dtype=torch.bool, device=device)

        # --- quick sanity checks on inputs
        if torch.isnan(src_vid).any() or torch.isinf(src_vid).any():
            logger.warning("입력 비디오 피처에 NaN/Inf가 포함되어 있습니다. nan->0으로 치환합니다.")
            src_vid = torch.nan_to_num(src_vid, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(src_txt).any() or torch.isinf(src_txt).any():
            logger.warning("입력 텍스트 피처에 NaN/Inf가 포함되어 있습니다. nan->0으로 치환합니다.")
            src_txt = torch.nan_to_num(src_txt, nan=0.0, posinf=0.0, neginf=0.0)

        # --- forward
        logger.info("QD-DETR 추론 수행...")
        outputs = qd_detr_model(src_txt=src_txt, src_txt_mask=src_txt_mask, src_vid=src_vid, src_vid_mask=src_vid_mask)

        # basic outputs exists?
        if "pred_logits" not in outputs or "pred_spans" not in outputs:
            raise RuntimeError("모델 출력에 pred_logits 또는 pred_spans가 없습니다.")

        # --- decode: two modes depending on args.span_loss_type
        span_type = getattr(args, "span_loss_type", "l1")
        video_duration = get_video_duration(video_path)
        if video_duration <= 0:
            logger.warning("video_duration이 0 또는 알 수 없음. 1.0 초로 대체합니다.")
            video_duration = 1.0

        # ensure tensors on CPU for conversion later
        pred_spans = None
        scores = None

        logits = outputs["pred_logits"]  # shape: (bsz, n_queries, 2)
        # clamp logits to avoid extreme values causing inf/nan in softmax
        logits = torch.clamp(logits, -50.0, 50.0)
        prob = F.softmax(logits, dim=-1)  # (bsz, n_queries, 2)

        if span_type == "l1":
            # outputs["pred_spans"]: (bsz, n_queries, 2) where each is (center, width) normalized in [0,1]
            scores = prob[..., 0]  # foreground prob
            pred_spans = outputs["pred_spans"]  # (bsz, n_q, 2)
            # nan guard
            if torch.isnan(pred_spans).any() or torch.isinf(pred_spans).any():
                logger.warning("pred_spans에 NaN/Inf 존재, nan->0으로 치환")
                pred_spans = torch.nan_to_num(pred_spans, nan=0.0, posinf=1.0, neginf=0.0)
            # convert cxw -> xx (start_ratio, end_ratio) and scale by duration
            pred_spans_abs = span_cxw_to_xx(pred_spans[0]) * float(video_duration)
        else:
            # span logits case: outputs["pred_spans"] shaped (bsz, n_q, 2*max_v_l) or (bsz,n_q,2,max_v_l)
            bsz, n_q = outputs["pred_spans"].shape[:2]
            # try to view as (bsz, n_q, 2, max_v_l)
            max_v_l = getattr(args, "max_v_l", outputs["pred_spans"].shape[-1] // 2)
            try:
                pred_spans_logits = outputs["pred_spans"].view(bsz, n_q, 2, max_v_l)
            except Exception:
                # fallback: try reshape by inferring max_v_l
                max_v_l = outputs["pred_spans"].shape[-1] // 2
                pred_spans_logits = outputs["pred_spans"].reshape(bsz, n_q, 2, max_v_l)

            # clamp logits and softmax across last dim
            pred_spans_logits = torch.clamp(pred_spans_logits, -50.0, 50.0)
            pred_span_scores, pred_spans_idx = F.softmax(pred_spans_logits, dim=-1).max(-1)  # scores and indices
            scores = torch.prod(pred_span_scores, dim=2)  # (bsz, n_q)
            # indices are discrete clip indices in [0, max_v_l-1]
            pred_spans_idx = pred_spans_idx.float()
            # make end exclusive by +1 (as original code)
            pred_spans_idx[:, :, 1] += 1.0
            clip_len = getattr(args, "clip_length", 2)  # number of seconds per clip (default 2)
            pred_spans_abs = pred_spans_idx * float(clip_len)

        # now pred_spans_abs: (n_q, 2) in seconds, scores: (1, n_q) or (bsz,n_q)
        # convert to cpu numpy and safe-clean
        pred_spans_abs = pred_spans_abs.detach().cpu()
        scores = scores.detach().cpu()

        # nan/inf defense
        if torch.isnan(pred_spans_abs).any() or torch.isinf(pred_spans_abs).any():
            logger.warning("pred_spans_abs에 NaN/Inf 존재, nan->0, inf->clamp(video_duration)")
            pred_spans_abs = torch.nan_to_num(pred_spans_abs, nan=0.0, posinf=video_duration, neginf=0.0)

        if torch.isnan(scores).any() or torch.isinf(scores).any():
            logger.warning("scores에 NaN/Inf 존재, nan->0으로 치환")
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

        # build results list
        results = []
        n_queries = pred_spans_abs.shape[0]
        for i in range(n_queries):
            st = float(pred_spans_abs[i, 0])
            ed = float(pred_spans_abs[i, 1])
            sc = float(scores[0, i]) if scores.dim() == 2 else float(scores[i])
            # clamp within [0, video_duration]
            st = max(0.0, min(st, video_duration))
            ed = max(0.0, min(ed, video_duration))
            if ed < st:
                # swap or set minimal width
                logger.debug(f"end < start for query {i} (st={st}, ed={ed}), swapping")
                st, ed = ed, st
            results.append({"start": st, "end": ed, "score": sc})

        # sort by score desc and choose best
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        if len(results_sorted) == 0:
            return 0.0, 0.0
        best = results_sorted[0]

        # final safety checks: numeric, not nan/inf
        def safe_float(x):
            try:
                v = float(x)
            except Exception:
                return 0.0
            if math.isnan(v) or math.isinf(v):
                return 0.0
            return v

        start = safe_float(best["start"])
        end = safe_float(best["end"])

        logger.info(f"예측 완료: start={start:.2f}, end={end:.2f}, score={best['score']:.4f}")
        return start, end

    except Exception as e:
        logger.exception(f"추론 중 예외 발생: {e}")
        return 0.0, 0.0



class ModelLoader:
    def __init__(self):
        logger.info("애플리케이션 시작: 모델 로딩 중...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        # CLIP 로드
        self.clip_handler = ClipHandler()
        logger.info("clip 모델 로딩완료")
    
        # QD-DETR 로드
        args = ModelConfig()
        args.device = self.device
        self.qd_detr_model, _ = build_model(args)
        self.args = args

        try:
            logger.info(f"'{args.CKPT_PATH}'에서 가중치 로딩 중...")
            checkpoint = torch.load(args.CKPT_PATH, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.qd_detr_model.load_state_dict(clean_state_dict, strict=False)
            self.qd_detr_model.to(self.device)
            self.qd_detr_model.eval()
            logger.info("QD-DETR 모델 로딩 완료\n")
        except FileNotFoundError:
            logger.info(f"[오류] 가중치 파일을 찾을 수 없습니다: '{args.CKPT_PATH}'")
            return
        except Exception as e:
            logger.info(f"[오류] 모델 로딩 중 오류 발생: {e}")
            return
        
        # MLP 어댑터 (512 → 2818)
        self.adapter = MLPAdapter(512, 2818).to(self.device)

    def get_models(self):
        return self.clip_handler, self.qd_detr_model, self.adapter, self.device, self.args



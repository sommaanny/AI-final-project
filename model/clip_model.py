import torch
import clip
from PIL import Image
import cv2

class ModelHandler:
    def __init__(self):
        #모델 로딩
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print("CLIP model loaded successfully.")

    def get_text_feature(self, text_list):
        #텍스트 피처 추출
        text = clip.tokenize(text_list).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features.cpu()
    
    def get_video_feature(self, video_path, segment_sec=2):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_per_segment = int(fps * segment_sec) #2초단위로 추출

        segment_features = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            #frame_idx가 2의 배수일 때만 피처 추출
            if (frame_idx + 1) % frames_per_segment == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feature = self.model.encode_image(tensor)
                segment_features.append(feature.cpu())

            frame_idx += 1
        
        cap.release()

        if not segment_features:
            return torch.empty(0)

        # shape: [추출된 프레임 수, 피처 차원]
        return torch.cat(segment_features, dim=0)

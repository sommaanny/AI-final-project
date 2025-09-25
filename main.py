from model.clip_model import ModelHandler
import torch



def run_clip_model():
    print("clip_model 시작")

    model = ModelHandler() #모델 로드

    TEST_VIDEO_PATH = "test_video.mp4"
    TEST_QUERIES = ["Ohtani homerun"]
    
    try:
        feature = model.get_text_feature(TEST_QUERIES)
        print("텍스트 피처:")
        print(feature)
        print("텍스트 피처 shape:")
        print(feature.shape)
    except Exception as e:
        print("텍스트 피처 추출 실패")

    
    try:
        feature = model.get_video_feature(TEST_VIDEO_PATH)
        print("비디오 피처:")
        print(feature)
        print("비디오 피처 shape:")
        print(feature.shape)
    except Exception as e:
        print("비디오 피처 추출 실패")


if __name__ == "__main__":
    run_clip_model()
    


class ModelConfig:
    # 피처 차원
    t_feat_dim = 512
    v_feat_dim = 2818
    a_feat_dir = None
    
    # Transformer 설정 (config.py의 default 값)
    hidden_dim = 256
    enc_layers = 2
    dec_layers = 2
    dim_feedforward = 1024
    dropout = 0.1
    nheads = 8
    pre_norm = False

     # 데이터 최대 길이 (config.py의 default 값)
    max_q_l = 32  # 텍스트 쿼리의 최대 토큰(단어) 길이
    max_v_l = 90  # 비디오의 최대 클립(세그먼트) 개수
    
    # 기타 모델 설정
    num_queries = 10
    input_dropout = 0.5
    n_input_proj = 2
    aux_loss = True
    position_embedding = 'sine'
    use_txt_pos = False

    # --- Matcher & Loss 설정 (build_matcher, SetCriterion에 필요) ---
    set_cost_span = 10
    set_cost_giou = 1
    set_cost_class = 4
    span_loss_coef = 10
    giou_loss_coef = 1
    label_loss_coef = 4
    eos_coef = 0.1
    span_loss_type = 'l1'

    # --- Contrastive & Saliency Loss 관련 설정 (형식적으로 필요) ---
    contrastive_align_loss = False
    contrastive_hdim = 64
    contrastive_align_loss_coef = 0.0
    lw_saliency = 1.0
    saliency_margin = 0.2
    temperature = 0.07
    dset_name = 'hl' # use_matcher=True로 설정하기 위한 값
    
    # 가중치 파일 경로
    CKPT_PATH = "./model/weights/model_best.ckpt"
# 하이브리드 분류 모델 정의
import torch
import torch.nn as nn
from transformers import RobertaModel

class RoBERTaHybridClassifier(nn.Module):
    """
    RoBERTa의 문맥 이해 능력과 DVL 패턴 수치를 결합한 하이브리드 분류기.
    구조: RoBERTa [CLS] 벡터 (768차원) + DVL flags (5차원) → MLP 레이어 → 가짜 뉴스 이진 분류
    """

    # 한국어 뉴스 데이터셋에 최적화된 KLUE-RoBERTa 모델 사용
    MODEL_NAME = "klue/roberta-base"

    def __init__(
        self,
        num_dvl_features: int = 5,   # 5대 DVL 패턴 지표
        num_labels: int = 2,         # 진짜(0)/가짜(1) 분류
        dropout: float = 0.3,        # 과적합 방지를 위한 드롭아웃 비율
        freeze_encoder_layers: int = 0, # 메모리 부족 시 하위 레이어 고정 옵션
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(self.MODEL_NAME)
        hidden_size = self.roberta.config.hidden_size  # 기본 768차원

        # 일부 레이어 동결 설정 (학습 속도 향상 및 자원 절약)
        if freeze_encoder_layers > 0:
            self._freeze_layers(freeze_encoder_layers)

        # 분류기(MLP): 텍스트 임베딩과 DVL 지표를 합쳐 최종 판단 수행
        self.classifier = nn.Sequential(
            # 입력 차원: 768 (RoBERTa) + 5 (DVL) = 773
            nn.Linear(hidden_size + num_dvl_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
        )

    def _freeze_layers(self, n: int) -> None:
        """모델의 하위 레이어 가중치를 고정하여 학습에서 제외."""
        # 임베딩 층 고정
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        # 사용자가 지정한 n개의 인코더 레이어 고정
        for layer in self.roberta.encoder.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dvl_flags: torch.Tensor,
    ) -> torch.Tensor:
        """데이터셋에서 받은 정보를 모델에 통과시키는 과정"""
        # 1. RoBERTa를 통해 텍스트 문맥 추출
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # 2. 문장 전체의 의미를 담고 있는 [CLS] 토큰의 출력값만 추출 
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # 3. 문맥 벡터와 우리가 정한 5개 수치 데이터(DVL)를 하나로 합침 
        combined = torch.cat([cls_output, dvl_flags], dim=-1)
        
        # 4. 최종 분류기를 통과하여 진짜/가짜 뉴스 점수 반환
        return self.classifier(combined)
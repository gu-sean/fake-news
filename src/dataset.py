import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 가짜 뉴스 판별을 위한 5대 DVL 패턴 컬럼 정의
DVL_COLS = [
    "stat_distortion",      # 통계 왜곡
    "causal_error",         # 인과 오류
    "emotional_provocation",# 감정 자극
    "source_lack",          # 출처 불명
    "img_mismatch",         # 이미지 불일치
]

# RoBERTa 모델의 표준 패딩 토큰 ID 
PAD_ID = 1

class FakeNewsDataset(Dataset):
    """
    36만 건의 대규모 데이터를 다루기 위해 'Lazy Tokenization' 방식을 사용.
    모든 데이터를 미리 토큰화하여 메모리에 올리지 않고, 학습 시 필요한 시점에만 변환하여 RAM 부족을 방지.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.texts = df["clean_message"].fillna("").tolist()
        self.dvl_flags = df[DVL_COLS].values.astype(np.float32)
        self.labels = df["label"].values.astype(np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """학습 루프에서 배치를 생성할 때 개별 샘플을 추출하는 핵심 메서드"""
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,        # 최대 길이 초과 시 절단
            return_tensors="pt",    # PyTorch 텐서 형태로 반환
        )
        
        return {
            # squeeze(0): 토크나이저가 생성한 불필요한 배치 차원 제거
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "dvl_flags": torch.from_numpy(self.dvl_flags[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Dynamic Padding: 배치 내에서 가장 긴 문장을 기준으로 패딩 길이를 결정.
    모든 배치를 무조건 512로 맞추는 고정 패딩보다 연산 속도가 훨씬 빠름.
    """
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,           # (batch, seq_len) 형태로 생성
        padding_value=PAD_ID,       # 텍스트 패딩 값
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,            # 마스크 패딩 값 (0: 무시)
    )
    
    # 리스트 형태의 텐서들을 하나의 배치 텐서로 병합
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "dvl_flags": torch.stack([item["dvl_flags"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }
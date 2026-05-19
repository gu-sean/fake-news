# 모델 학습 스크립트
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.dataset import FakeNewsDataset, collate_fn
from src.model import RoBERTaHybridClassifier

# 학습 과정을 터미널에 실시간으로 출력하기 위한 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def save_results(
    result_path: str,
    args: argparse.Namespace,
    epoch_results: list[dict],
    best_epoch: int,
    best_f1: float,
    elapsed_sec: float,
) -> None:
    """에포크별 성능 수치를 정리하여 txt 파일로 저장."""
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 55,
        "           모델 학습 성능 결과",
        "=" * 55,
        f"실행 일시   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"데이터 파일 : {args.data_path}",
        f"총 에포크   : {args.epochs}",
        f"배치 크기   : {args.batch_size}",
        f"최대 길이   : {args.max_length}",
        f"총 소요 시간: {elapsed_sec/60:.1f}분 ({elapsed_sec:.0f}초)",
        "-" * 55,
        f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Macro-F1':>8}",
        "-" * 55,
    ]

    for r in epoch_results:
        marker = " *" if r["epoch"] == best_epoch else ""
        lines.append(
            f"{r['epoch']:>5}  {r['train_loss']:>10.4f}  {r['train_acc']:>9.4f}"
            f"  {r['val_loss']:>8.4f}  {r['val_acc']:>7.4f}  {r['val_f1']:>8.4f}{marker}"
        )

    lines += [
        "-" * 55,
        f"최고 성능   : Epoch {best_epoch}  Macro-F1 = {best_f1:.4f}  (* 표시)",
        "=" * 55,
    ]

    Path(result_path).write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"결과 저장 완료: {result_path}")


# ── Optimizer 빌드: 부위별 차등 학습률 적용 ────────────────────────
def build_optimizer(
    model: RoBERTaHybridClassifier,
    roberta_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """
    이미 학습된 RoBERTa 몸체와 새로 만든 분류기 헤드에 
    서로 다른 학습률을 적용하여 모델의 안정성을 높임.
    """
    # 가중치 감쇄를 적용하지 않을 파라미터들 (주로 편향 및 정규화 층)
    no_decay = {"bias", "LayerNorm.weight"}

    def split(named_params):
        decay = [p for n, p in named_params if not any(nd in n for nd in no_decay)]
        no_dc = [p for n, p in named_params if any(nd in n for nd in no_decay)]
        return decay, no_dc

    # RoBERTa 파라미터와 분류기(MLP) 파라미터를 분리하여 리스트화
    rb_decay, rb_nodecay = split(model.roberta.named_parameters())
    head_params = list(model.classifier.parameters())

    return torch.optim.AdamW(
        [
            # RoBERTa 가중치: 낮은 학습률 적용
            {"params": rb_decay,   "lr": roberta_lr, "weight_decay": weight_decay},
            {"params": rb_nodecay, "lr": roberta_lr, "weight_decay": 0.0},
            # 새로 만든 분류기: 상대적으로 높은 학습률 적용
            {"params": head_params, "lr": head_lr,   "weight_decay": weight_decay},
        ]
    )


# ── Train Epoch: 1회 학습 루프 ──────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float]:
    model.train()  
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    # 진행률 표시바(tqdm)와 함께 배치 학습 진행
    for batch in tqdm(loader, desc="  train", leave=False, dynamic_ncols=True):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        dvl_flags      = batch["dvl_flags"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad() 
        
        # 모델 예측 (텍스트 + DVL 결합 데이터 입력)
        logits = model(input_ids, attention_mask, dvl_flags)
        loss = criterion(logits, labels) # 오차값 계산
        
        loss.backward() # 오차 역전파
        
        # Gradient Clipping: 그래디언트 폭주를 막아 학습 안정성 확보
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()  # 가중치 업데이트
        scheduler.step()  # 스케줄러에 따라 학습률 조정

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return float(total_loss / len(loader)), float(acc)


# ── Evaluate: 모델 검증 루프 (학습 반영 안 함) ────────────────────────────────────
@torch.no_grad() # 평가 시에는 그래디언트 계산을 꺼서 메모리 절약
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval() # 드롭아웃 등을 비활성화하여 일관된 결과 도출
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in tqdm(loader, desc="  val  ", leave=False, dynamic_ncols=True):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        dvl_flags      = batch["dvl_flags"].to(device)
        labels         = batch["label"].to(device)

        logits = model(input_ids, attention_mask, dvl_flags)
        total_loss += criterion(logits, labels).item()
        
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    # 가짜 뉴스와 진짜 뉴스 판별 성능의 균형을 위해 macro-F1을 지표로 사용
    f1  = f1_score(all_labels, all_preds, average="macro")
    return float(total_loss / len(loader)), float(acc), float(f1)


# ── Main: 전체 실행 프로세스 ────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    # 재현성을 위해 시드값 고정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # GPU 가용 여부에 따라 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 데이터 로딩 및 레이블 비율 유지하며 분할
    df = pd.read_csv(args.data_path, low_memory=False)
    train_df, val_df = train_test_split(
        df, test_size=args.val_ratio, random_state=args.seed, stratify=df["label"]
    )
    logger.info(f"Train: {len(train_df):,}  Val: {len(val_df):,}")

    # KLUE-RoBERTa 기반 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(RoBERTaHybridClassifier.MODEL_NAME)
    train_ds = FakeNewsDataset(train_df.reset_index(drop=True), tokenizer, args.max_length)
    val_ds   = FakeNewsDataset(val_df.reset_index(drop=True),   tokenizer, args.max_length)

    # DataLoader 구성: 동적 패딩(collate_fn) 적용
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # 하이브리드 모델 인스턴스 생성
    model = RoBERTaHybridClassifier(
        dropout=args.dropout,
        freeze_encoder_layers=args.freeze_layers,
    ).to(device)

    # 손실 함수 및 최적화 도구 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.roberta_lr, args.head_lr, args.weight_decay)

    # Warmup 설정: 초반에 학습률을 서서히 높여 모델이 급격히 변하는 것 방지
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 최적 모델(Checkpoint) 저장 디렉토리 생성
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    best_epoch = 1
    epoch_results: list[dict] = []

    start_time = time.time()

    # 실제 학습 루프 실행
    for epoch in range(1, args.epochs + 1):
        logger.info(f"── Epoch {epoch}/{args.epochs} ──────────────────────────")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, args.grad_clip)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"train  loss={train_loss:.4f}  acc={train_acc:.4f}\n"
            f"val    loss={val_loss:.4f}  acc={val_acc:.4f}  macro-f1={val_f1:.4f}"
        )

        epoch_results.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc, "val_f1": val_f1,
        })

        # F1 스코어가 가장 높은 시점의 모델 가중치 저장
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)
            logger.info(f"  Best Model Updated (val_f1={val_f1:.4f}) -> {ckpt_path}")

    elapsed = time.time() - start_time
    logger.info(f"학습 완료. Best val macro-F1: {best_f1:.4f}  소요 시간: {elapsed/60:.1f}분")

    save_results(args.result_path, args, epoch_results, best_epoch, best_f1, elapsed)


# ── 실행 파라미터 정의 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoBERTa Hybrid Classifier 학습 스크립트")

    # 기본 경로 설정
    parser.add_argument("--data_path",       default="data/processed/unified_news_refined.csv")
    parser.add_argument("--checkpoint_dir",  default="checkpoints")
    parser.add_argument("--result_path",     default="data/processed/training_results.txt",
                        help="학습 결과를 저장할 txt 파일 경로")

    # 모델 하이퍼파라미터
    parser.add_argument("--max_length",      type=int,   default=512)
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--freeze_layers",   type=int,   default=0,
                        help="하위 N개 인코더 레이어 고정 (메모리 절약용)")

    # 학습 하이퍼파라미터
    parser.add_argument("--epochs",          type=int,   default=5)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--roberta_lr",      type=float, default=2e-5)
    parser.add_argument("--head_lr",         type=float, default=1e-4)
    parser.add_argument("--weight_decay",    type=float, default=0.01)
    parser.add_argument("--warmup_ratio",    type=float, default=0.06)
    parser.add_argument("--grad_clip",       type=float, default=1.0)

    # 기타 옵션
    parser.add_argument("--val_ratio",       type=float, default=0.1)
    parser.add_argument("--num_workers",     type=int,   default=0)
    parser.add_argument("--seed",            type=int,   default=42)

    main(parser.parse_args())
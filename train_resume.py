import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 continue training helper (resume or finetune-from-ckpt)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path. Recommend last.pt for true resume.",
    )
    parser.add_argument(
        "--search_dir",
        type=str,
        default=r"G:\yolov8\runs\train_ct",
        help="Auto-search dir when --checkpoint is empty.",
    )
    parser.add_argument(
        "--fallback_best",
        action="store_true",
        help="Use best.pt if last.pt not found.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="resume",
        choices=["resume", "finetune"],
        help="resume: true checkpoint resume; finetune: load weights and train with new params.",
    )
    parser.add_argument("--data", type=str, default=r"G:\yolov8\dataste\data.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/train_ct")
    parser.add_argument("--name", type=str, default="lung_exp_resume")
    return parser.parse_args()


def resolve_checkpoint(checkpoint: str, search_dir: str, fallback_best: bool) -> Path:
    if checkpoint:
        ckpt = Path(checkpoint).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    root = Path(search_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Search directory not found: {root}")

    candidates = sorted(root.glob("**/weights/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates and fallback_best:
        candidates = sorted(root.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found. Provide --checkpoint or ensure runs/**/weights/last.pt exists."
        )
    return candidates[0]


def main():
    args = parse_args()
    torch.cuda.empty_cache()

    ckpt = resolve_checkpoint(args.checkpoint, args.search_dir, args.fallback_best)
    print(f"[Info] Checkpoint: {ckpt}")
    print(f"[Info] Mode: {args.mode}")

    model = YOLO(str(ckpt))

    if args.mode == "resume":
        # True resume: keep optimizer/EMA state from checkpoint.
        model.train(resume=True)
    else:
        # Finetune: start from checkpoint weights with user-defined params.
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=False,
        )

    print("[Done] Training finished.")


if __name__ == "__main__":
    main()

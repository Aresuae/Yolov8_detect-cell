import argparse
import os
import sys

import torch
from ultralytics import YOLO


# 默认路径（云端 AI Studio）；本地运行时可在命令行覆盖
DEFAULT_MODEL = "/home/aistudio/yolov8/ultralytics/models/v8/yolo_p2.yaml"
DEFAULT_DATA = "/home/aistudio/yolov8/data/data.yaml"
DEFAULT_PREDICT_SOURCE = "/home/aistudio/yolov8/data/valid/images"
DEFAULT_PROJECT = "runs/train_ct"
DEFAULT_NAME = "lung_exp"


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 CT Detection: train / resume / eval / predict"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="yaml 或 .pt（仅新训练）")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA, help="data.yaml 路径")
    parser.add_argument("--predict_source", type=str, default=DEFAULT_PREDICT_SOURCE, help="预测图片目录")

    parser.add_argument("--epochs", type=int, default=30, help="总训练轮数（续训时仍为总 epochs，非额外增加）")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2, help="2 核 CPU 建议 0~2")

    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--name", type=str, default=DEFAULT_NAME)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="从断点继续训练（默认使用 project/name/weights/last.pt）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="指定断点权重，例如 runs/train_ct/lung_exp/weights/last.pt",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="仅训练/续训，跳过 predict 与 val",
    )
    return parser.parse_args()


def get_run_dir(project: str, name: str) -> str:
    return os.path.join(project, name)


def get_last_checkpoint(project: str, name: str) -> str:
    return os.path.join(project, name, "weights", "last.pt")


def get_best_checkpoint(project: str, name: str) -> str:
    return os.path.join(project, name, "weights", "best.pt")


def resolve_resume_checkpoint(args) -> str:
    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        ckpt = get_last_checkpoint(args.project, args.name)

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            "未找到断点权重，无法续训。\n"
            f"期望路径: {ckpt}\n"
            "请先正常训练至少 1 个 epoch，或使用 --checkpoint 指定 last.pt。"
        )
    return ckpt


def build_train_kwargs(args) -> dict:
    return dict(
        data=args.data_dir,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=0,
        optimizer="Adam",
        lr0=0.001,
        cos_lr=True,
        box=10.0,
        cls=0.5,
        mosaic=0.0,
        mixup=0.0,
        project=args.project,
        name=args.name,
    )


def train_model(args):
    torch.cuda.empty_cache()
    train_kwargs = build_train_kwargs(args)

    if args.resume:
        ckpt = resolve_resume_checkpoint(args)
        print(f"\n🔁 断点续训: {ckpt}")
        model = YOLO(ckpt)
        # resume=True 会从上次 epoch、优化器状态继续；epochs 表示「总轮数」
        model.train(resume=True, **train_kwargs)
    else:
        print(f"\n🚀 新训练: {args.model}")
        model = YOLO(args.model)
        model.train(**train_kwargs)

    return model


def run_predict_and_val(args):
    best_path = get_best_checkpoint(args.project, args.name)
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"找不到 best 权重: {best_path}")

    print(f"\n📌 加载最佳模型: {best_path}")
    model = YOLO(best_path)

    model.predict(
        source=args.predict_source,
        save=True,
        save_txt=True,
        save_conf=True,
        imgsz=args.imgsz,
        conf=0.1,
        iou=0.6,
        device=0,
        project="runs/predict",
        name="lung_vis",
        line_width=2,
        show_labels=True,
        show_conf=True,
    )
    print("\n✅ 检测结果图已生成: runs/predict/lung_vis/")

    metrics = model.val(
        data=args.data_dir,
        imgsz=args.imgsz,
        conf=0.001,
        iou=0.6,
        save_json=True,
    )

    print("\n📊 模型评估指标:")
    print(metrics)
    print(f"\n📂 训练曲线目录: {get_run_dir(args.project, args.name)}/")
    print(f"Precision:      {metrics.box.mp:.4f}")
    print(f"Recall:         {metrics.box.mr:.4f}")
    print(f"mAP@0.5:        {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95:   {metrics.box.map:.4f}")


def main():
    args = parse_args()
    train_model(args)

    if args.train_only:
        print("\n⏭ 已跳过 predict / val（--train-only）")
        return

    run_predict_and_val(args)


if __name__ == "__main__":
    main()

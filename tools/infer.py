import argparse
import json
import os
import sys

import matplotlib
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")  # 確保在沒有圖形介面的 Server/SSH 上畫圖不會報錯
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# 確保能引用到專案根目錄的 src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.core import YAMLConfig


def plot_training_logs(log_path, output_dir):
    """讀取 log.txt 並畫出 Loss 與 mAP 的 PDF 曲線圖"""
    if not os.path.exists(log_path):
        print(f"⚠️ 找不到 Log 檔: {log_path}，跳過畫圖步驟。")
        return

    epochs = []
    train_losses = []
    val_mAP_50_95 = []
    val_mAP_50 = []

    print(f"📊 正在讀取訓練日誌並繪製圖表...")
    with open(log_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                epochs.append(data["epoch"])
                # 讀取訓練 Loss
                train_losses.append(data.get("train_loss", 0))

                # 讀取驗證集 mAP (COCO 格式的 stats 陣列，索引 0 是 0.5:0.95，索引 1 是 0.5)
                if "test_coco_eval_bbox" in data:
                    val_mAP_50_95.append(data["test_coco_eval_bbox"][0])
                    val_mAP_50.append(data["test_coco_eval_bbox"][1])
            except:
                continue

    # 1. 繪製 Training Loss 曲線
    plt.figure(figsize=(8, 6))
    plt.plot(
        epochs,
        train_losses,
        marker="o",
        markersize=4,
        linestyle="-",
        color="tab:blue",
        label="Train Loss",
    )
    plt.title("Training Loss per Epoch", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    loss_pdf_path = os.path.join(output_dir, "train_loss_curve.pdf")
    plt.savefig(loss_pdf_path, format="pdf", bbox_inches="tight")
    plt.close()

    # 2. 繪製 Validation mAP 曲線
    if val_mAP_50_95:
        plt.figure(figsize=(8, 6))
        plt.plot(
            epochs,
            val_mAP_50_95,
            marker="s",
            markersize=4,
            linestyle="-",
            color="tab:orange",
            label="mAP @ 0.5:0.95",
        )
        plt.plot(
            epochs,
            val_mAP_50,
            marker="^",
            markersize=4,
            linestyle="-",
            color="tab:green",
            label="mAP @ 0.5",
        )
        plt.title("Validation mAP per Epoch", fontsize=16)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("mAP", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=12)
        map_pdf_path = os.path.join(output_dir, "val_mAP_curve.pdf")
        plt.savefig(map_pdf_path, format="pdf", bbox_inches="tight")
        plt.close()

    print(f"✅ 圖表已儲存為 PDF: {loss_pdf_path} 與 {map_pdf_path}")


def main(args):
    # --- 模型初始化 ---
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        state = (
            checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        )
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model
            self.postprocessor = cfg.postprocessor

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)
    model.eval()

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    results = []
    image_filenames = [
        f for f in os.listdir(args.test_dir) if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    print(f"Found {len(image_filenames)} images in {args.test_dir}")
    print("Starting inference...")

    # --- 推論與視覺化 ---
    visualized_count = 0  # 紀錄已經視覺化了幾張圖片

    with torch.no_grad():
        for filename in tqdm(image_filenames):
            filepath = os.path.join(args.test_dir, filename)
            im_pil = Image.open(filepath).convert("RGB")
            w, h = im_pil.size

            orig_size = torch.tensor([[w, h]]).to(args.device)
            im_data = transforms(im_pil)[None].to(args.device)

            output = model(im_data, orig_size)

            res = output[0]
            labels = res["labels"].cpu().numpy()
            boxes = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()

            image_id_int = int(os.path.splitext(filename)[0])

            # 🔥 [新增] 視覺化前五張圖片
            vis_boxes = []
            vis_labels = []
            vis_scores = []

            for label, box, score in zip(labels, boxes, scores):
                if score < 0.25:
                    continue

                x_min, y_min, x_max, y_max = box
                width = float(x_max - x_min)
                height = float(y_max - y_min)
                category_id = int(label)

                prediction = {
                    "image_id": image_id_int,
                    "bbox": [float(x_min), float(y_min), width, height],
                    "score": float(score),
                    "category_id": category_id,
                }
                results.append(prediction)

                # 收集要畫在圖上的資訊
                vis_boxes.append([x_min, y_min, width, height])
                vis_labels.append(category_id)
                vis_scores.append(score)

            # 畫出前 5 張圖片並存檔
            if visualized_count < 5:
                fig, ax = plt.subplots(1, figsize=(6, 6))
                ax.imshow(im_pil)
                for (vx, vy, vw, vh), vlbl, vscr in zip(
                    vis_boxes, vis_labels, vis_scores
                ):
                    # 畫紅框
                    rect = patches.Rectangle(
                        (vx, vy), vw, vh, linewidth=2, edgecolor="red", facecolor="none"
                    )
                    ax.add_patch(rect)
                    # 寫標籤
                    text_str = f"ID:{vlbl} ({vscr:.2f})"
                    ax.text(
                        vx,
                        max(vy - 2, 0),
                        text_str,
                        color="white",
                        fontsize=10,
                        bbox=dict(facecolor="red", alpha=0.7, edgecolor="none", pad=1),
                    )

                plt.axis("off")
                # 將圖片存成 PDF 與 PNG 兩種格式，方便報告使用
                vis_pdf_path = os.path.join(
                    os.path.dirname(args.output), f"vis_test_{image_id_int}.pdf"
                )
                vis_png_path = os.path.join(
                    os.path.dirname(args.output), f"vis_test_{image_id_int}.png"
                )
                plt.savefig(
                    vis_pdf_path, format="pdf", bbox_inches="tight", pad_inches=0
                )
                plt.savefig(
                    vis_png_path,
                    format="png",
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=300,
                )
                plt.close()
                visualized_count += 1

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(
        f"✅ Submission file generated! Saved {len(results)} entries to {args.output}"
    )

    # --- 繪製訓練日誌圖表 ---
    # 自動推斷 log.txt 的路徑 (通常與你傳入的 checkpoint 在同一個目錄)
    ckpt_dir = os.path.dirname(args.resume)
    log_file_path = os.path.join(ckpt_dir, "log.txt")
    output_graph_dir = (
        os.path.dirname(args.output) if os.path.dirname(args.output) != "" else "."
    )

    plot_training_logs(log_file_path, output_graph_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-t", "--test-dir", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="pred.json")
    args = parser.parse_args()
    main(args)

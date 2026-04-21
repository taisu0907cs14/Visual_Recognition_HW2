"""
by lyuwenyu
"""

import datetime
import json
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes, make_grid  # [新增] 引入畫圖工具

from src.data import get_coco_api_from_dataset
from src.misc import dist

from .det_engine import evaluate, train_one_epoch
from .solver import BaseSolver


class DetSolver(BaseSolver):

    # ===== [新增] 視覺化驗證集推論結果 =====
    @torch.no_grad()
    def visualize_validation(self, epoch, writer, num_images=4):
        """抓取驗證集圖片並畫出預測結果，上傳到 TensorBoard"""
        self.model.eval()

        try:
            # 抓取一個 batch 的驗證集資料
            samples, targets = next(iter(self.val_dataloader))
            samples = samples.to(self.device)

            # 進行推論
            # (修改後的寫法)
            outputs = self.model(samples)

            # 取得當前 Tensor 畫布的實際長寬 (H, W)
            h, w = samples.shape[-2], samples.shape[-1]

            # 建立與畫布尺寸相同的 target_sizes，讓框直接對齊 Tensor
            target_sizes = torch.tensor([[h, w]] * len(samples), device=self.device)
            results = self.postprocessor(outputs, target_sizes)

            vis_images = []
            # 只取前 num_images 張圖來畫
            for i in range(min(num_images, len(samples))):
                img = samples[i].cpu()
                res = results[i]

                # 設定信心門檻，只畫出 score > 0.45 的預測框
                score_threshold = 0.45
                keep = res["scores"] > score_threshold
                boxes = res["boxes"][keep].cpu()
                labels = [
                    f"{int(l)-1}:{s:.2f}"
                    for l, s in zip(res["labels"][keep], res["scores"][keep])
                ]

                # 將 0~1 的影像數值轉換為 0~255 的 uint8 格式供繪圖
                img_uint8 = (img * 255).clamp(0, 255).to(torch.uint8)

                # 如果有偵測到物件，則畫上 Bounding Box
                if len(boxes) > 0:
                    img_with_boxes = draw_bounding_boxes(
                        img_uint8, boxes, labels=labels, colors="red", width=2
                    )
                else:
                    img_with_boxes = img_uint8

                vis_images.append(img_with_boxes)

            # 將這幾張圖片拼成網格並傳到 TensorBoard
            if vis_images:
                grid = make_grid(vis_images)
                writer.add_image("Val_Visualization/Predictions", grid, epoch)

        except Exception as e:
            print(f"Visualization skipped at epoch {epoch} due to error: {e}")

        # 切換回訓練模式
        self.model.train()

    # =======================================

    def fit(
        self,
    ):
        print("Start training")
        self.train()

        args = self.cfg

        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("number of params:", n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        best_stat = {
            "epoch": -1,
        }

        # ===== TensorBoard 初始化 =====
        writer = None
        if self.output_dir and dist.is_main_process():
            tb_log_dir = self.output_dir / "tb_logs"
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_log_dir))
            print(f"✅ TensorBoard is logging to: {tb_log_dir}")
        # ==================================

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            # 傳入 writer 給 engine
            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                args.clip_max_norm,
                print_freq=args.log_step,
                ema=self.ema,
                scaler=self.scaler,
                writer=writer,
            )

            # ===== 寫入過濾後的 Training Stats =====
            if writer is not None:
                allowed_keys = ["loss", "loss_bbox", "loss_giou", "loss_vfl", "lr"]
                for k, v in train_stats.items():
                    if k in allowed_keys:
                        writer.add_scalar(f"Train_Epoch/{k}", v, epoch)
            # ================================================

            self.lr_scheduler.step()

            if self.output_dir:
                checkpoint_paths = [self.output_dir / "checkpoint.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(
                        self.output_dir / f"checkpoint{epoch:04}.pth"
                    )
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                base_ds,
                self.device,
                self.output_dir,
            )

            # ===== 展開並寫入 Validation Stats =====
            if writer is not None:
                if "coco_eval_bbox" in test_stats:
                    stats = test_stats["coco_eval_bbox"]
                    writer.add_scalar("Val/AP_0.5_0.95(mAP)", stats[0], epoch)
                    writer.add_scalar("Val/AP_0.5", stats[1], epoch)
                    writer.add_scalar("Val/AP_0.75", stats[2], epoch)
                    writer.add_scalar("Val/AP_small", stats[3], epoch)
                    writer.add_scalar("Val/AP_medium", stats[4], epoch)
                    writer.add_scalar("Val/AP_large", stats[5], epoch)
                    writer.add_scalar("Val/AR_max_1", stats[6], epoch)
                    writer.add_scalar("Val/AR_max_10", stats[7], epoch)
                    writer.add_scalar("Val/AR_max_100", stats[8], epoch)
                    writer.add_scalar("Val/AR_small", stats[9], epoch)
                    writer.add_scalar("Val/AR_medium", stats[10], epoch)
                    writer.add_scalar("Val/AR_large", stats[11], epoch)

                # ===== [呼叫新增的視覺化函數] =====
                self.visualize_validation(epoch, writer)
                # ==================================
            # ===================================================

            for k in test_stats.keys():
                if k in best_stat:
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]
            print("best_stat: ", best_stat)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        if writer is not None:
            writer.close()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(
        self,
    ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            base_ds,
            self.device,
            self.output_dir,
        )

        if self.output_dir:
            dist.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return

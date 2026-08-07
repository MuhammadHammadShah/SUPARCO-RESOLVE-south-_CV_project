from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from soilreg.losses import build_loss
from soilreg.metrics import compute_metrics
from soilreg.utils import get_logger


class Trainer:

    def __init__(
        self,
        model,
        config,
        device,
    ):

        self.model = model.to(device)
        self.cfg = config
        self.device = device
        self.logger = get_logger()

        self.criterion = build_loss(
            self.cfg.loss.name
        )

        self.optimizer = self._build_optimizer()

        self.scheduler = self._build_scheduler()


        self.device_type = (
            "cuda"
            if "cuda" in str(device).lower()
            else "cpu"
        )


        self.scaler = GradScaler(
            device=self.device_type,
            enabled=(
                self.cfg.mixed_precision
                and self.device_type == "cuda"
            ),
        )


        # -----------------------------
        # Training state
        # -----------------------------

        self.best_loss = float("inf")

        self.current_epoch = 0


        # -----------------------------
        # Early stopping
        # -----------------------------

        self.patience = 10

        self.counter = 0



        # -----------------------------
        # Checkpoints
        # -----------------------------

        self.output_dir = Path(
            self.cfg.output_dir
        )

        self.checkpoint_dir = (
            self.output_dir
            / "checkpoints"
        )


        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )



    def _build_optimizer(self):

        name = self.cfg.optimizer.name.lower()


        if name == "adam":

            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.training.learning_rate,
                weight_decay=self.cfg.training.weight_decay,
            )


        if name == "adamw":

            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.training.learning_rate,
                weight_decay=self.cfg.training.weight_decay,
            )


        if name == "sgd":

            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.training.learning_rate,
                momentum=0.9,
                weight_decay=self.cfg.training.weight_decay,
            )


        raise ValueError(
            f"Unknown optimizer: {name}"
        )



    def _build_scheduler(self):

        name = self.cfg.scheduler.name.lower()



        if name == "none":

            return None



        if name == "cosine":

            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training.epochs,
            )



        if name == "step":

            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.1,
            )



        raise ValueError(
            f"Unknown scheduler: {name}"
        )



    def save_checkpoint(
        self,
        filename: str,
    ):


        checkpoint_path = (
            self.checkpoint_dir
            / filename
        )


        torch.save(

            {

                "epoch": self.current_epoch,

                "model":
                    self.model.state_dict(),

                "optimizer":
                    self.optimizer.state_dict(),

                "scheduler":
                    None
                    if self.scheduler is None
                    else self.scheduler.state_dict(),

                "best_loss":
                    self.best_loss,

            },

            checkpoint_path,

        )


        self.logger.info(
            f"Saved checkpoint to {checkpoint_path}"
        )



    def load_checkpoint(
        self,
        path: str,
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        if (
            self.scheduler is not None
            and checkpoint.get("scheduler") is not None
        ):
            self.scheduler.load_state_dict(
                checkpoint["scheduler"]
            )

        self.current_epoch = checkpoint.get(
            "epoch",
            0,
        )

        # compatibility with old checkpoints
        self.best_loss = checkpoint.get(
            "best_loss",
            float("inf"),
        )

        self.logger.info(
            f"Resumed from epoch {self.current_epoch}"
        )



    def train_one_epoch(
        self,
        train_loader,
    ):

        self.model.train()

        running_loss = 0.0



        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=False,
        )



        for images, targets in progress_bar:


            images = images.to(
                self.device,
                non_blocking=True,
            )


            targets = targets.to(
                self.device,
                non_blocking=True,
            )



            self.optimizer.zero_grad(
                set_to_none=True
            )



            with autocast(

                device_type=self.device_type,

                enabled=(
                    self.cfg.mixed_precision
                    and self.device_type == "cuda"
                ),

            ):


                predictions = self.model(
                    images
                )


                loss = self.criterion(
                    predictions,
                    targets,
                )



            self.scaler.scale(
                loss
            ).backward()



            self.scaler.unscale_(
                self.optimizer
            )



            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )



            self.scaler.step(
                self.optimizer
            )


            self.scaler.update()



            running_loss += loss.item()



            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )



        epoch_loss = (
            running_loss
            /
            len(train_loader)
        )



        if self.scheduler is not None:

            self.scheduler.step()



        return epoch_loss



    @torch.no_grad()
    def validate(
        self,
        val_loader,
    ):


        self.model.eval()


        running_loss = 0.0

        predictions = []

        targets = []



        for images, labels in tqdm(
            val_loader,
            desc=f"Epoch {self.current_epoch + 1} [Valid]",
            leave=False,
        ):


            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )



            with autocast(

                device_type=self.device_type,

                enabled=(
                    self.cfg.mixed_precision
                    and self.device_type == "cuda"
                ),

            ):


                outputs = self.model(
                    images
                )


                loss = self.criterion(
                    outputs,
                    labels,
                )



            running_loss += loss.item()



            predictions.append(
                outputs.cpu().numpy()
            )


            targets.append(
                labels.cpu().numpy()
            )



        epoch_loss = (
            running_loss
            /
            len(val_loader)
        )



        predictions = np.concatenate(
            predictions,
            axis=0,
        )


        targets = np.concatenate(
            targets,
            axis=0,
        )



        metrics = compute_metrics(
            targets,
            predictions,
        )


        metrics["loss"] = epoch_loss


        return metrics




    def fit(
        self,
        train_loader,
        val_loader,
    ):


        self.logger.info(
            "Starting training..."
        )



        for epoch in range(
            self.current_epoch,
            self.cfg.training.epochs
        ):


            self.current_epoch = epoch



            train_loss = self.train_one_epoch(
                train_loader
            )


            metrics = self.validate(
                val_loader
            )



            self.logger.info(

                f"Epoch {epoch+1:03d} | "

                f"train_loss={train_loss:.4f} | "

                f"val_loss={metrics['loss']:.4f} | "

                f"MAE={metrics['mae']:.4f} | "

                f"RMSE={metrics['rmse']:.4f} | "

                f"R2={metrics['r2']:.4f}"

            )



            self.save_checkpoint(
                "last.pt"
            )



            if metrics["loss"] < self.best_loss:


                self.best_loss = metrics["loss"]

                self.counter = 0


                self.save_checkpoint(
                    "best.pt"
                )


                self.logger.info(
                    "Best checkpoint updated."
                )



            else:


                self.counter += 1


                self.logger.info(

                    f"No improvement "
                    f"({self.counter}/{self.patience})"

                )



                if self.counter >= self.patience:


                    self.logger.info(
                        "Early stopping triggered."
                    )


                    break



        self.logger.info(
            "Training finished."
        )
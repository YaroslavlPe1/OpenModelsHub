import os
import json
import torch
import joblib
import tensorflow as tf
from typing import Any, Dict, Optional, Union


class ModelSaverLoader:
    @staticmethod
    def save_model(model: Any, save_path: str, optimizer: Optional[Any] = None, tokenizer: Optional[Dict] = None, epoch: Optional[int] = None, additional_info: Optional[Dict] = None):
        """Сохранение модели, оптимизатора и токенов (если применимо)."""
        os.makedirs(save_path, exist_ok=True)

        # Сохранение модели
        model_save_path = os.path.join(save_path, "model")
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), model_save_path + ".pth")
        elif isinstance(model, tf.keras.Model):
            model.save(model_save_path)
        else:
            joblib.dump(model, model_save_path + ".pkl")

        # Сохранение оптимизатора
        if optimizer:
            optimizer_save_path = os.path.join(save_path, "optimizer.pth")
            if isinstance(optimizer, torch.optim.Optimizer):
                torch.save(optimizer.state_dict(), optimizer_save_path)

        # Сохранение токенизатора
        if tokenizer:
            tokenizer_save_path = os.path.join(save_path, "tokenizer.json")
            with open(tokenizer_save_path, "w") as f:
                json.dump(tokenizer, f)

        # Сохранение гиперпараметров или других данных
        additional_info = additional_info or {}
        if epoch is not None:
            additional_info["epoch"] = epoch

        config_save_path = os.path.join(save_path, "config.json")
        with open(config_save_path, "w") as f:
            json.dump(additional_info, f)

        print(f"Model and files saved to {save_path}")

    @staticmethod
    def load_model(model: Optional[Any] = None, optimizer: Optional[Any] = None, tokenizer: Optional[Dict] = None, load_path: str = "model_checkpoint", device: str = 'cpu') -> Dict[str, Any]:
        """Загрузка модели, оптимизатора и токенов (если применимо)."""
        model_load_path = os.path.join(load_path, "model")
        config_load_path = os.path.join(load_path, "config.json")

        # Загрузка модели
        if os.path.exists(model_load_path + ".pth"):
            if model is not None and isinstance(model, torch.nn.Module):
                model.load_state_dict(torch.load(model_load_path + ".pth", map_location=device))
            else:
                raise ValueError("Provide a model instance for PyTorch.")
        elif os.path.exists(model_load_path + ".h5"):
            if model is not None and isinstance(model, tf.keras.Model):
                model = tf.keras.models.load_model(model_load_path)
            else:
                raise ValueError("Provide a model instance for TensorFlow.")
        elif os.path.exists(model_load_path + ".pkl"):
            model = joblib.load(model_load_path + ".pkl")
        else:
            raise FileNotFoundError("Model file not found.")

        # Загрузка оптимизатора
        if optimizer and os.path.exists(os.path.join(load_path, "optimizer.pth")):
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.load_state_dict(torch.load(os.path.join(load_path, "optimizer.pth"), map_location=device))
            else:
                raise ValueError("Optimizer must be a PyTorch optimizer.")

        # Загрузка токенизатора
        if tokenizer and os.path.exists(os.path.join(load_path, "tokenizer.json")):
            with open(os.path.join(load_path, "tokenizer.json"), "r") as f:
                tokenizer.update(json.load(f))

        # Загрузка дополнительных данных (например, гиперпараметров)
        additional_info = {}
        if os.path.exists(config_load_path):
            with open(config_load_path, "r") as f:
                additional_info = json.load(f)

        print(f"Model loaded from {load_path}")
        return additional_info

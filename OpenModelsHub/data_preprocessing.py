import pandas as pd
import json
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchaudio import load as load_audio
import cv2  # Для обработки видео

class DataPreprocessor:
    def __init__(self):
        self.transform = None

    def load_csv(self, file_path, delimiter=','):
        """Загрузка данных из CSV файла"""
        data = pd.read_csv(file_path, delimiter=delimiter)
        return data

    def load_json(self, file_path):
        """Загрузка данных из JSON файла"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def load_images(self, folder_path, image_size=(64, 64), normalize=True):
        """Загрузка и предобработка изображений из папки"""
        transform_list = [transforms.Resize(image_size), transforms.ToTensor()]
        if normalize:
            transform_list.append(transforms.Normalize([0.5], [0.5]))
        self.transform = transforms.Compose(transform_list)

        images = []
        for image_file in os.listdir(folder_path):
            if image_file.endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(folder_path, image_file))
                img = self.transform(img)
                images.append(img)
        return torch.stack(images)

    def load_audio(self, folder_path, sample_rate=16000):
        """Загрузка и предобработка аудио файлов"""
        audios = []
        for audio_file in os.listdir(folder_path):
            if audio_file.endswith('.wav'):
                waveform, sr = load_audio(os.path.join(folder_path, audio_file), normalize=True)
                if sr != sample_rate:
                    raise ValueError(f"Sample rate mismatch: expected {sample_rate}, got {sr}")
                audios.append(waveform)
        return torch.stack(audios)

    def load_video(self, folder_path, video_length=16, frame_size=(64, 64)):
        """Загрузка и предобработка видео файлов"""
        videos = []
        for video_file in os.listdir(folder_path):
            if video_file.endswith(('.mp4', '.avi')):
                frames = self.extract_frames(os.path.join(folder_path, video_file), video_length, frame_size)
                videos.append(frames)
        return torch.stack(videos)

    def extract_frames(self, video_file, num_frames, frame_size):
        """Извлечение кадров из видео файла с помощью OpenCV"""
        cap = cv2.VideoCapture(video_file)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, frame_count // num_frames)  # Интервал для извлечения кадров

        while len(frames) < num_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, frame_size)  # Изменение размера кадра
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Конвертация в RGB
            frames.append(torch.tensor(frame).permute(2, 0, 1))  # Преобразование в тензор

        cap.release()

        # Если кадров меньше, чем нужно, дублируем последние кадры
        while len(frames) < num_frames:
            frames.append(frames[-1])

        return torch.stack(frames)

    def preprocess_text(self, text_data, tokenizer):
        """Преобразование текста в числовое представление"""
        return tokenizer(text_data)

    def load_dataset(self, file_path, file_type='csv', **kwargs):
        """Универсальный метод для загрузки данных различных форматов"""
        if file_type == 'csv':
            return self.load_csv(file_path, **kwargs)
        elif file_type == 'json':
            return self.load_json(file_path)
        elif file_type == 'images':
            return self.load_images(file_path, **kwargs)
        elif file_type == 'audio':
            return self.load_audio(file_path, **kwargs)
        elif file_type == 'video':
            return self.load_video(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

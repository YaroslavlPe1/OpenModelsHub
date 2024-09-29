import torch
import torch.nn as nn
from collections import deque

# VGAN (Video GAN)
class VideoGANGenerator(nn.Module):
    def __init__(self, z_dim, video_length, img_channels):
        super(VideoGANGenerator, self).__init__()
        self.fc = nn.Linear(z_dim, video_length * img_channels * 64 * 64)

    def forward(self, z):
        video = self.fc(z).view(z.size(0), -1, 3, 64, 64)  # (batch_size, frames, channels, height, width)
        return video

class VideoGANDiscriminator(nn.Module):
    def __init__(self, video_length, img_channels):
        super(VideoGANDiscriminator, self).__init__()
        self.fc = nn.Linear(video_length * img_channels * 64 * 64, 1)

    def forward(self, video):
        video_flat = video.view(video.size(0), -1)
        return torch.sigmoid(self.fc(video_flat))

class VGAN:
    def __init__(self, z_dim=100, video_length=16, img_channels=3):
        self.generator = VideoGANGenerator(z_dim, video_length, img_channels).cuda()
        self.discriminator = VideoGANDiscriminator(video_length, img_channels).cuda()

    def train(self, dataloader, epochs=50, lr=0.0002):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        loss_function = nn.BCELoss()

        for epoch in range(epochs):
            for i, (videos, _) in enumerate(dataloader):
                batch_size = videos.size(0)

                # Генерация шума и меток
                z = torch.randn(batch_size, 100).cuda()
                fake_videos = self.generator(z)
                valid = torch.ones(batch_size, 1).cuda()
                fake = torch.zeros(batch_size, 1).cuda()

                # Тренировка дискриминатора
                real_loss = loss_function(self.discriminator(videos.cuda()), valid)
                fake_loss = loss_function(self.discriminator(fake_videos.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                # Тренировка генератора
                g_loss = loss_function(self.discriminator(fake_videos), valid)
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

                print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")

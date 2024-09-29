import torch
import torch.nn as nn
import torch.optim as optim

# WaveGAN для генерации аудио
class WaveGANGenerator(nn.Module):
    def __init__(self, z_dim, audio_length):
        super(WaveGANGenerator, self).__init__()
        self.fc = nn.Linear(z_dim, audio_length)

    def forward(self, z):
        return torch.tanh(self.fc(z))

class WaveGANDiscriminator(nn.Module):
    def __init__(self, audio_length):
        super(WaveGANDiscriminator, self).__init__()
        self.fc = nn.Linear(audio_length, 1)

    def forward(self, audio):
        return torch.sigmoid(self.fc(audio))

class WaveGAN:
    def __init__(self, z_dim=100, audio_length=16000, lr=0.0002):
        self.generator = WaveGANGenerator(z_dim, audio_length).cuda()
        self.discriminator = WaveGANDiscriminator(audio_length).cuda()
        self.adversarial_loss = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr)

    def train(self, dataloader, epochs=50):
        for epoch in range(epochs):
            for i, (audio, _) in enumerate(dataloader):
                batch_size = audio.size(0)

                valid = torch.ones(batch_size, 1).cuda()
                fake = torch.zeros(batch_size, 1).cuda()

                z = torch.randn(batch_size, 100).cuda()
                generated_audio = self.generator(z)

                real_loss = self.adversarial_loss(self.discriminator(audio.cuda()), valid)
                fake_loss = self.adversarial_loss(self.discriminator(generated_audio.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                self.optimizer_D.zero_grad()
                d_loss.backward()
                self.optimizer_D.step()

                g_loss = self.adversarial_loss(self.discriminator(generated_audio), valid)

                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()

                print(f"Epoch {epoch}/{epochs} Batch {i} Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")

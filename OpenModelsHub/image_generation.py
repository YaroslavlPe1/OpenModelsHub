import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from .model_saver_loader import ModelSaverLoader

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_channels * 64 * 64),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), 1, 64, 64)

class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_channels * 64 * 64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))

class GANImageGenerator(ModelSaverLoader):
    def __init__(self, z_dim=100, img_channels=1, lr=0.0002, optimizer='adam', loss_function='bce', betas=(0.5, 0.999)):
        self.generator = Generator(z_dim, img_channels).cuda()
        self.discriminator = Discriminator(img_channels).cuda()
        self.loss_function = self._get_loss_function(loss_function)
        self.optimizer_G = self._get_optimizer(optimizer, self.generator.parameters(), lr, betas)
        self.optimizer_D = self._get_optimizer(optimizer, self.discriminator.parameters(), lr, betas)
        self.z_dim = z_dim
        self.img_channels = img_channels

    def _get_optimizer(self, optimizer_name, params, lr, betas):
        if optimizer_name == 'adam':
            return optim.Adam(params, lr=lr, betas=betas)
        elif optimizer_name == 'sgd':
            return optim.SGD(params, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _get_loss_function(self, loss_function_name):
        if loss_function_name == 'bce':
            return nn.BCELoss()
        elif loss_function_name == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")

    def train(self, dataloader, epochs=50, save_interval=100, save_path="model_checkpoint", scheduler=None):
        for epoch in range(epochs):
            for i, (imgs, _) in enumerate(dataloader):
                batch_size = imgs.size(0)
                real_imgs = imgs.cuda()

                valid = torch.ones((batch_size, 1)).cuda()
                fake = torch.zeros((batch_size, 1)).cuda()

                z = torch.randn(batch_size, self.z_dim).cuda()

                gen_imgs = self.generator(z)
                g_loss = self.loss_function(self.discriminator(gen_imgs), valid)
                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()

                real_loss = self.loss_function(self.discriminator(real_imgs), valid)
                fake_loss = self.loss_function(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                self.optimizer_D.zero_grad()
                d_loss.backward()
                self.optimizer_D.step()

                print(f"Epoch {epoch}/{epochs} Batch {i} Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")

                if i % save_interval == 0:
                    save_image(gen_imgs.data[:25], f"images/{epoch}_{i}.png", nrow=5, normalize=True)

            self.save_model(self.generator, self.optimizer_G, save_path=f"{save_path}/epoch_{epoch}", epoch=epoch)

            if scheduler:
                scheduler.step()

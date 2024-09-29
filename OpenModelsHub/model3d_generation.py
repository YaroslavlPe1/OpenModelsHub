import torch
import torch.nn as nn

class VoxelGANGenerator(nn.Module):
    def __init__(self, z_dim, voxel_dim):
        super(VoxelGANGenerator, self).__init__()
        self.fc = nn.Linear(z_dim, voxel_dim ** 3)

    def forward(self, z):
        return torch.sigmoid(self.fc(z)).view(z.size(0), 1, voxel_dim, voxel_dim, voxel_dim)

class VoxelGANDiscriminator(nn.Module):
    def __init__(self, voxel_dim):
        super(VoxelGANDiscriminator, self).__init__()
        self.fc = nn.Linear(voxel_dim ** 3, 1)

    def forward(self, voxel):
        return torch.sigmoid(self.fc(voxel.view(voxel.size(0), -1)))

class VoxelGAN:
    def __init__(self, z_dim=100, voxel_dim=32):
        self.generator = VoxelGANGenerator(z_dim, voxel_dim).cuda()
        self.discriminator = VoxelGANDiscriminator(voxel_dim).cuda()

    def train(self, dataloader, epochs=50, lr=0.0002):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        loss_function = nn.BCELoss()

        for epoch in range(epochs):
            for i, voxel in enumerate(dataloader):
                batch_size = voxel.size(0)

                z = torch.randn(batch_size, 100).cuda()
                fake_voxels = self.generator(z)

                valid = torch.ones(batch_size, 1).cuda()
                fake = torch.zeros(batch_size, 1).cuda()

                real_loss = loss_function(self.discriminator(voxel.cuda()), valid)
                fake_loss = loss_function(self.discriminator(fake_voxels.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                g_loss = loss_function(self.discriminator(fake_voxels), valid)

                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

                print(f"Epoch [{epoch}/{epochs}] Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")

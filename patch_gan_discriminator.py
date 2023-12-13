import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels, num_patches=64):
        super(PatchGANDiscriminator, self).__init__()
        self.num_patches=num_patches
        
        #Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2,padding=1 )
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2,padding=1 )
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2,padding=1 )
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1,padding=1 )
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1,padding=1 )
    
    def forward(self, x):
        # Forward pass through convolutional layers
        x=torch.relu(self.conv1(x))
        x=torch.relu(self.conv2(x))
        x=torch.relu(self.conv3(x))
        x=torch.relu(self.conv4(x))
        x=self.conv5(x)
        
        #Average pooling to get the PatchGAN output
        # x=nn.functional.avg_pool2d(x,x.size()[2:])
        # return x.view(-1,1)
        return x

# # # Example usage:
# # # Initialize the PatchGAN discriminator
# discriminator = PatchGANDiscriminator(in_channels=5)

# # Create a random input image tensor (batch_size, channels, height, width)
# fake_image = torch.randn(2, 5, 256, 256)

# # Forward pass through the discriminator
# output = discriminator(fake_image)
# accuracy=0
# for adx in range(len(output)):
#     accuracy+=output[adx]
# accuracy/=len(output)
# # Print the output shape

# print("PatchGAN Output Shape:", output.shape)
# print("PatchGAN output is:", output)
# print("Accuracy of the batch is:", accuracy)
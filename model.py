import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3d(nn.Module):
    """
    A convolutional block that consists of a 3D convolutional layer followed by batch normalization and ReLU activation.
    Conv3d -> BatchNorm -> ReLU
    
    The documentation of torch.nn.Conv3d can be found at:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolution kernel.
        stride (int or tuple, optional): Stride of the convolution. Default is 1.
        padding (int or tuple, optional): Padding added to both sides of the input. Default is 0.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        # self.relu = nn.ReLU(inplace=True) # we can use F.relu instead of nn.ReLU to save memory
        
    def forward(self, x):
        assert x.dim() == 5, "Input tensor must be 5D (batch_size, channels, depth, height, width)"
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class ConvBlock3dPool(nn.Module):
    """
    A convolutional block that consists of a 3D convolutional layer followed by batch normalization, ReLU activation, and max pooling.
    Conv3d -> BatchNorm -> ReLU -> MaxPool3d

    The documentation of torch.nn.Conv3d can be found at:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolution kernel.
        stride (int or tuple, optional): Stride of the convolution. Default is 1.
        padding (int or tuple, optional): Padding added to both sides of the input. Default is 0.
        pool_size (int or tuple, optional): Size of the max pooling window. Default is 2.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pool_size=2):
        super(ConvBlock3dPool, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(kernel_size=pool_size)
        # self.relu = nn.ReLU(inplace=True) # we can use F.relu instead of nn.ReLU to save memory
        
    def forward(self, x):
        assert x.dim() == 5, "Input tensor must be 5D (batch_size, channels, depth, height, width)"
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

class SFCNN(nn.Module):
    """
    SFCNN model for Protein-Ligand Affinity Prediction.
    Paper: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04762-3
    Implementation in Tensorflow: https://github.com/bioinfocqupt/Sfcnn
    This model consists of several convolutional blocks followed by max pooling layers alternatively, and a final fully connected MLP.


    Original model building method:
    
    def build_model():
        model = tf.keras.Sequential([
        Conv3D(7,kernel_size=(1,1,1),input_shape=(20,20,20,28),strides=(1,1,1)),
        BatchNormalization(),  
        Activation(tf.nn.relu),
        Conv3D(7,kernel_size=(3,3,3)),
        BatchNormalization(),  
        Activation(tf.nn.relu),
        Conv3D(7,kernel_size=(3,3,3)),
        BatchNormalization(),
        Activation(tf.nn.relu),
        Conv3D(28,kernel_size=(1,1,1)),
        BatchNormalization(),  
        Activation(tf.nn.relu),
        Conv3D(56,kernel_size=(3,3,3),padding='same'),
        BatchNormalization(),  
        Activation(tf.nn.relu),
        MaxPooling3D(pool_size=2),
        Conv3D(112,kernel_size=(3,3,3),padding='same'),
        BatchNormalization(),  
        Activation(tf.nn.relu),
        MaxPooling3D(pool_size=2),
        Conv3D(224,kernel_size=(3,3,3),padding='same'),
        BatchNormalization(),  
        Activation(tf.nn.relu),
        MaxPooling3D(pool_size=2),
        Flatten(),
        Dense(256),
        BatchNormalization(),
        Activation(tf.nn.relu),
        Dense(1)])

        model.load_weights('weights_22_112-0.0083.h5')
        return model

    Args:
        nn (_type_): _description_
    """
    def __init__(self, in_channels=28):
        super(SFCNN, self).__init__()
        
        # the convolutional blocks
        # input: (batch_size, in_channels, depth, height, width)
        # in_channels = 28
        # depth, height, width = 20, 20, 20
        self.conv1 = ConvBlock3d(in_channels, 7, kernel_size=(1, 1, 1)) # (b, 28, 20, 20, 20) -> (b, 7, 20, 20, 20)
        self.conv2 = ConvBlock3d(7, 7, kernel_size=(3, 3, 3)) # (b, 7, 20, 20, 20) -> (b, 7, 18, 18, 18)
        self.conv3 = ConvBlock3d(7, 7, kernel_size=(3, 3, 3)) # (b, 7, 18, 18, 18) -> (b, 7, 16, 16, 16)
        self.conv4 = ConvBlock3d(7, 28, kernel_size=(1, 1, 1)) # (b, 7, 16, 16, 16) -> (b, 28, 16, 16, 16)
        self.conv5 = ConvBlock3dPool(28, 56, kernel_size=(3, 3, 3), padding=1) # (b, 28, 16, 16, 16) -> (b, 56, 16, 16, 16) -> (b, 56, 8, 8, 8)
        self.conv6 = ConvBlock3dPool(56, 112, kernel_size=(3, 3, 3), padding=1) # (b, 56, 8, 8, 8) -> (b, 112, 8, 8, 8) -> (b, 112, 4, 4, 4)
        self.conv7 = ConvBlock3dPool(112, 224, kernel_size=(3, 3, 3), padding=1) # (b, 112, 4, 4, 4) -> (b, 224, 4, 4, 4) -> (b, 224, 2, 2, 2)

        # flatten the output of the last convolutional block
        self.flatten = nn.Flatten() # (b, 224, 2, 2, 2) -> (b, 224 * 2 * 2 * 2)
        # the fully connected layers
        self.fc1 = nn.Linear(224 * 2 * 2 * 2, 256) # dense 256
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        assert x.dim() == 5, "Input tensor must be 5D (batch_size, channels, depth, height, width)"
        x = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x))))))) # (b, 28, 20, 20, 20) -> (b, 224, 2, 2, 2)
        x = self.flatten(x) # (b, 224, 2, 2, 2) -> (b, 224 * 2 * 2 * 2)
        x = self.fc1(x) # (b, 224 * 2 * 2 * 2) -> (b, 256)
        x = self.bn1(x) # (b, 256) -> (b, 256)
        x = F.relu(x) # (b, 256) -> (b, 256)
        x = self.fc2(x) # (b, 256) -> (b, 1)
        return x
    
# unit test
if __name__ == "__main__" and False:
    # test the model
    model = SFCNN()
    x = torch.randn(2, 28, 20, 20, 20) # (batch_size, channels, depth, height, width)
    y = model(x)
    print(y.shape) # should be (2, 1)
    # test the ConvBlock3d
    conv_block = ConvBlock3d(28, 7, kernel_size=(1, 1, 1))
    x = torch.randn(2, 28, 20, 20, 20) # (batch_size, channels, depth, height, width)
    y = conv_block(x)
    print(y.shape) # should be (2, 7, 20, 20, 20)
    # test the ConvBlock3dPool
    conv_block_pool = ConvBlock3dPool(28, 7, kernel_size=(1, 1, 1))
    x = torch.randn(2, 28, 20, 20, 20) # (batch_size, channels, depth, height, width)
    y = conv_block_pool(x)
    print(y.shape) # should be (2, 7, 20, 20, 20)
    
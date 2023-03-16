import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_layer(in_channels = 243, out_channels=256, kernel_size=None, padding=None, stride=None):
    """Conv2D Layer
    Args:
        in_channels: int, the input channels of the conv2d layer
        out_channels: int, the output channels of the conv2d layer
        kernel_size: list, the kernel size of the conv2d layer
        strides: list, the strides of the conv2d layer
    """
    if kernel_size is None:
        kernel_size = (1, 1)
    if padding is None:
        padding = (0, 1)
    if stride is None:
        stride = (1, 1)
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    return conv2d

class FormantLayer(nn.Module):
    """Formant Layer
    Args:
        padding_size: list, the padding size of each conv2d layer
        kernels_size: list, the kernel size of each conv2d layer
        inputs: list, the input channels of each conv2d layer    
        outputs: list, the output channels of each conv2d layer    
    """
    def __init__(self, padding_size=None, kernels_size=None, inputs=None, outputs=None):
        super(FormantLayer, self).__init__()

        if kernels_size is None:
            kernels_size = [(1, 3), (1, 3), (1, 3), (1, 3), (1, 2)]
        if padding_size is None:
            padding_size = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 0)]
        if inputs is None:
            inputs = [1, 72, 108, 162, 243]
        if outputs is None:
            outputs = [72, 108, 162, 243, 256]

        self.kernels_size = kernels_size
        self.inputs = inputs
        self.outputs = outputs

        self.formant_layers = nn.Sequential()

        for i in range(len(self.kernels_size)):
            self.formant_layers.add_module('conv{}'.format(i), conv2d_layer(in_channels=self.inputs[i],
                                                 out_channels=self.outputs[i],
                                                 kernel_size=self.kernels_size[i],
                                                 padding=padding_size[i],
                                                 stride=(1, 2)))
            self.formant_layers.add_module('relu{}'.format(i), nn.ReLU())

    def forward(self, x):
        x = self.formant_layers(x)
        return x

class ArticulationLayer(nn.Module):
    """Articulation Layer
    Args:
        kernels_size: list, the kernel size of each conv2d layer
        E: int, the channels of the emotion layer
        conv2d_strides: list, the strides of each conv2d layer
        emotion_strides: list, the strides of each emotion layer
    """
    def __init__(self, kernels_size=None, E=16, conv2d_strides=None, emotion_strides=None):
        super(ArticulationLayer, self).__init__()

        self.E = E 
        if kernels_size is None:
            kernels_size = [(3, 1), (3, 1), (3, 1), (3, 1), (4, 1)]
        if emotion_strides is None:
            emotion_strides = [(2, 1), (4, 1), (8, 1), (16, 1), (64, 1)]
        if conv2d_strides is None:
            conv2d_strides = [(2, 1), (2, 1), (2, 1), (2, 1), (4, 1)]

        self.kernels_size = kernels_size
        self.emotion_strides = emotion_strides
        self.conv2d_strides = conv2d_strides
        
        self.emotion = nn.Parameter(torch.normal(size=[1, self.E, 64, 1], mean=0.0, std=1.0))

        self.articulation_layer_1 = nn.ModuleList()
        self.articulation_layer_2 = nn.ModuleList()
        
        for i in range(len(self.kernels_size)):
            sub_layer_1 = nn.Sequential(conv2d_layer(256 if i == 0 else 256 + self.E, 256, kernel_size=self.kernels_size[i], padding=(1, 0), stride=self.conv2d_strides[i]),
                nn.LeakyReLU())
            self.articulation_layer_1.append(sub_layer_1)
            sub_layer_2 = nn.Sequential(conv2d_layer(self.E, self.E, kernel_size=self.kernels_size[i], padding=(1, 0), stride=self.emotion_strides[i]),
                nn.LeakyReLU())
            self.articulation_layer_2.append(sub_layer_2)

    def forward(self, x):
        emotion_input = torch.tile(self.emotion, (x.shape[0], 1, 1, 1))
        for i in range(len(self.kernels_size)):
            conv_x = self.articulation_layer_1[i](x)
            emotion_x = self.articulation_layer_2[i](emotion_input)
            mixed_x = torch.cat([conv_x, emotion_x], dim = 1)
            x = mixed_x
        return (x, emotion_input)

class OutputLayer(nn.Module):
    """Output Layer
    Args:
        input_size: int, the input size of the output layer
        output_size: int, the output size of the output layer
        keep_pro: float, the keep probability of the dropout layer
    """
    def __init__(self, input_size, output_size, keep_pro = 0.5):
        super(OutputLayer, self).__init__()
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 150),
            nn.Dropout(keep_pro),
            nn.Linear(150, output_size),
            # nn.Tanh()
        )
    def forward(self, x):
        return self.output_layer(x)

class Audio2Face(nn.Module):
    def __init__(self, output_size, keep_pro = 0.5):
        super(Audio2Face, self).__init__()
        self.output_size = output_size
        self.keep_pro = keep_pro
        self.FormantLayer = FormantLayer()
        self.ArticulationLayer = ArticulationLayer()
        self.OutputLayer = OutputLayer(self.ArticulationLayer.E + 256, self.output_size, self.keep_pro)
        
    def forward(self, x):
        x = x.transpose(1, 3)
        x = self.FormantLayer(x)
        x, emotion_input = self.ArticulationLayer(x)
        x = self.OutputLayer(x)
        return (x, emotion_input)

def losses(y, output):
    """Loss Function
    Args:
        y: tensor, the ground truth
        output: tensor,[pred , emotion_input] the output of the model
    """
    y_, emotion_input = output

    y = y.type(torch.float32)    # Cast the type of y to float32
    y_ = y_.type(torch.float32)   # Cast the type of y_ to float32
    emotion_input = emotion_input.type(torch.float32)  # Cast the type of emotion_input to float32

    loss_P = torch.mean(torch.square(y - y_)) # Calculate the loss_P

    # Calculate the loss_M
    split_y = torch.chunk(y, 2, 0)  # Parameter: tensor, split number, dimension
    split_y_ = torch.chunk(y_, 2, 0)

    y0 = split_y[0] # y0 is the first half of y
    y1 = split_y[1] # y1 is the second half of y
    y_0 = split_y_[0]   # y_0 is the first half of y_
    y_1 = split_y_[1]   # y_1 is the second half of y_
    loss_M = 2 * torch.mean(torch.square(y0 - y1 - y_0 + y_1)) # Calculate the loss_M

    # Calculate the loss_R
    split_emotion_input = torch.chunk(emotion_input, 2, 0)
    emotion_input0 = split_emotion_input[0] # emotion_input0 is the first half of emotion_input
    emotion_input1 = split_emotion_input[1] # emotion_input1 is the second half of emotion_input

    # Formula(3), Rx3 is R'(x)
    Rx0 = torch.square(emotion_input0 - emotion_input1)  # Calculate the m[·]
    Rx1 = torch.sum(Rx0, dim = 1)  # 4-dim, sum of the height
    Rx2 = torch.sum(Rx1, dim = 1)  # 3-dim, sum of the width
    Rx3 = 2 * torch.mean(Rx2, dim = 1)  # 2-dim, mean of the emotion

    # Formula(4), Rx is R(x), length is batch_size/2
    e_mean0 = torch.sum(torch.square(emotion_input0), dim = 2)  # 4-dim, sum of the width
    e_mean1 = torch.mean(torch.mean(e_mean0, dim = 1), dim = 1)  # 2-dim, mean of the emotion
    Rx = Rx3 / e_mean1  # R(x)

    # Formula(5)
    # beta = 0.99
    # R_vt = beta * R_vt_input + (1-beta) * tf.reduce_mean(tf.square(Rx)) # every epoch update
    # R_vt_ = R_vt/(1-tf.pow(beta, step))

    # Formula(6) Calculate the loss_R
    # loss_R = tf.reduce_mean(Rx)/(tf.sqrt(R_vt_)+epsilon)
    loss_R = torch.mean(Rx)
    # loss_R = tf.reduce_mean(tf.square(emotion_input1 - emotion_input0), name='loss_R')

    # Calculate the total loss
    loss = loss_P + loss_M + loss_R
    mse = F.mse_loss(y, y_)
    return loss, mse
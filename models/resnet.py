'''
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch

class ShallowResNetBlock(torch.nn.Module):
    '''
        input -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + ShortCut -> Relu
          |                                                |
          ------------- (Conv1x1 -> BN ->) -----------------
    '''
    def __init__(self, in_channel, out_channel, downsample=False):
        super(ShallowResNetBlock, self).__init__()
        
        # main branch
        # block1
        if downsample:
            self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channel)
        self.relu1 = torch.nn.ReLU(inplace=True)
        # block2
        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channel)

        # shortcut
        # if the main branch is downsampled the shortcut branch will be downsampled (use conv1x1) too
        if downsample:
            self.shortcut_conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False)
            self.shortcut_bn1 = torch.nn.BatchNorm2d(out_channel)
        elif in_channel != out_channel:
            self.shortcut_conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
            self.shortcut_bn1 = torch.nn.BatchNorm2d(out_channel)
        else:
            self.shortcut_conv1 = None
            self.shortcut_bn1 = None

        # merge
        self.relu_out = torch.nn.ReLU(inplace=True)

    def forward(self, inputs):
        # main
        main = self.conv1(inputs)
        main = self.bn1(main)
        main = self.relu1(main)
        main = self.conv2(main)
        main = self.bn2(main)

        # shortcut
        if self.shortcut_conv1 is not None:
            shortcut = self.shortcut_conv1(inputs)
            shortcut = self.shortcut_bn1(shortcut)
        else:
            shortcut = inputs

        # merge
        outs = self.relu_out(main+shortcut)
        return outs

class DeepResNetBlock(torch.nn.Module):
    '''
        input -> Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN + ShortCut -> Relu
          |                                                                       |
          -------------------------- (Conv1x1 -> BN ->) ---------------------------
    '''

    def __init__(self, in_channel, out_channel, downsample=False):
        super(DeepResNetBlock, self).__init__()

        # main branch
        mid_channel = int(out_channel / 4)
        # block1 (in_channel -> mid_channel)
        self.conv1 = torch.nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(mid_channel)
        self.relu1 = torch.nn.ReLU(inplace=True)
        # block2
        if downsample:
            self.conv2 = torch.nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1, stride=2, bias=False)
        else:
            self.conv2 = torch.nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(mid_channel)
        self.relu2 = torch.nn.ReLU(inplace=True)
        # block3 (mid_channel -> out_channel)
        self.conv3 = torch.nn.Conv2d(mid_channel, out_channel, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channel)

        # shortcut
        if downsample:
            self.shortcut_conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False)
            self.shortcut_bn1 = torch.nn.BatchNorm2d(out_channel)
        elif in_channel != out_channel:
            self.shortcut_conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
            self.shortcut_bn1 = torch.nn.BatchNorm2d(out_channel)
        else:
            self.shortcut_conv1 = None
            self.shortcut_bn1 = None

        # merge
        self.relu_out = torch.nn.ReLU(inplace=True)

    def forward(inputs):
        # main
        main = self.conv1(inputs)
        main = self.bn1(main)
        main = self.relu1(main)

        main = self.conv2(main)
        main = self.bn2(main)
        main = self.relu2(main)

        main = self.conv3(main)
        main = self.bn3(main)

        # shortcut
        if self.shortcut_conv1 is not None:
            shortcut = self.shortcut_conv1(inputs)
            shortcut = self.shortcut_bn1(shortcut)
        else:
            shortcut = inputs

        # merge
        outs = self.relu_out(main+shortcut)

# SHALLOW_BLOCK = 0
# DEEP_BLOCK = 1
class ResNet(torch.nn.Module):
    def __init__(self, input_size, res_out_channel, num_classes, resnet_type, **kargs):
        '''
            Args:
                input_size: input pic size (batch size not included, eg: (3,32,32))
                res_out_channel: out_channel of res block, 512 (resnet18/34) or 2048 (resnet > 50)
                type:
                    "7x7": first conv block is 7x7 with two downsample layers (Conv2d layer and MaxPool2d layer)
                    "3x3": first conv block is 3x3 and there is no downsample layer in begin
                    
        '''
        super(ResNet, self).__init__(**kargs)
        self.input_size = input_size
        self.type = resnet_type
        if resnet_type == "7x7":
            # stage 1
            self.conv1 = torch.nn.Conv2d(input_size[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.relu1 = torch.nn.ReLU(inplace=True)
            self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # stage 6
            self.pool2 = torch.nn.AvgPool2d(int(input_size[1]/32))
            self.flatten = torch.nn.Flatten()        
            self.linear1 = torch.nn.Linear(res_out_channel, num_classes)
        elif resnet_type == "3x3":
            # stage 1
            self.conv1 = torch.nn.Conv2d(input_size[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.relu1 = torch.nn.ReLU(inplace=True)
            # self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # stage 6
            self.pool2 = torch.nn.AvgPool2d(int(input_size[1]/8))
            self.flatten = torch.nn.Flatten()        
            self.linear1 = torch.nn.Linear(res_out_channel, num_classes)
        else:
            raise RuntimeError("Invalid resnet type") 

        # stage 2~5
        self.block1 = None
        self.block2 = None
        self.block3 = None
        self.block4 = None


    def forward(self, inputs):

        # stage 1
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu1(outputs)
        if self.type == "7x7":
            outputs = self.pool1(outputs)

        # stage 2-5
        outputs = self.block1(outputs)
        outputs = self.block2(outputs)
        outputs = self.block3(outputs)
        outputs = self.block4(outputs)

        # stage 6
        outputs = self.pool2(outputs)
        outputs = self.flatten(outputs)
        outputs = self.linear1(outputs)

        return outputs

class ShallowResNet(ResNet):
    def __init__(self, input_size, num_classes, block_num_list, resnet_type="3x3", **kargs):
        super(ShallowResNet, self).__init__(input_size, 512, num_classes, resnet_type, **kargs)

        # assert
        for i in range(4):
            assert block_num_list[i] > 0, "block num needs greater than 0!"

        # stage 2
        block1_list = []
        for i in range(block_num_list[0]):
            block1_list.append(ShallowResNetBlock(64,64))
        self.block1 = torch.nn.Sequential(*block1_list)

        # stage 3
        block2_list = []
        for i in range(block_num_list[1]):
            if i == 0:
                block2_list.append(ShallowResNetBlock(64, 128, downsample=True))
            else:
                block2_list.append(ShallowResNetBlock(128, 128))
        self.block2 = torch.nn.Sequential(*block2_list)

        # stage 4
        block3_list = []
        for i in range(block_num_list[2]):
            if i == 0:
                block3_list.append(ShallowResNetBlock(128, 256, downsample=True))
            else:
                block3_list.append(ShallowResNetBlock(256, 256))
        self.block3 = torch.nn.Sequential(*block3_list)

        # stage 5
        block4_list = []
        for i in range(block_num_list[3]):
            if i == 0:
                block4_list.append(ShallowResNetBlock(256, 512, downsample=True))
            else:
                block4_list.append(ShallowResNetBlock(512, 512))
        self.block4 = torch.nn.Sequential(*block4_list)

class DeepResNet(ResNet):
    def __init__(self, input_size, num_classes, block_num_list, resnet_type="3x3", **kargs):
        super(DeepResNet, self).__init__(input_size, 2048, num_classes, resnet_type, **kargs)

        # assert
        for i in range(4):
            assert block_num_list[i] > 0, "block num needs greater than 0!"

        # stage 2
        block1_list = []
        for i in range(block_num_list[0]):
            if i == 0:
                block1_list.append(DeepResNetBlock(64,256))
            else:
                block1_list.append(DeepResNetBlock(256,256))
        self.block1 = torch.nn.Sequential(*block1_list)

        # stage 3
        block2_list = []
        for i in range(block_num_list[1]):
            if i == 0:
                block2_list.append(DeepResNetBlock(256, 512, downsample=True))
            else:
                block2_list.append(DeepResNetBlock(512, 512))
        self.block2 = torch.nn.Sequential(*block2_list)

        # stage 4
        block3_list = []
        for i in range(block_num_list[2]):
            if i == 0:
                block3_list.append(DeepResNetBlock(512, 1024, downsample=True))
            else:
                block3_list.append(DeepResNetBlock(1024, 1024))
        self.block3 = torch.nn.Sequential(*block3_list)

        # stage 5
        block4_list = []
        for i in range(block_num_list[3]):
            if i == 0:
                block4_list.append(DeepResNetBlock(1024, 2048, downsample=True))
            else:
                block4_list.append(DeepResNetBlock(2048, 2048))
        self.block4 = torch.nn.Sequential(*block4_list)

def ResNet18(input_size, num_classes, resnet_type="3x3", **kargs):
    return ShallowResNet(input_size, num_classes, block_num_list=[2,2,2,2], resnet_type=resnet_type, **kargs)

def ResNet34(input_size, num_classes, resnet_type="3x3", **kargs):
    return ShallowResNet(input_size, num_classes, block_num_list=[3,4,6,3], resnet_type=resnet_type, **kargs)

def ResNet50(input_size, num_classes, resnet_type="3x3", **kargs):
    return DeepResNet(input_size, num_classes, block_num_list=[3,4,6,3], resnet_type=resnet_type, **kargs)

def ResNet101(input_size, num_classes, resnet_type="3x3", **kargs):
    return DeepResNet(input_size, num_classes, block_num_list=[3,4,23,3], resnet_type=resnet_type, **kargs)
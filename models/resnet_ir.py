'''
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch

class IRBlock(torch.nn.Module):
    '''
        input -> BN -> Conv3x3 -> BN -> PReLU -> Conv3x3 (downsample) -> BN + ShortCut
          |                                                                 |
          ----------------------- (Conv1x1 -> BN) ---------------------------
        other difference:
            normal resnet BasicBlock Conv3x3(inplane, plane, stride) -> Conv3x3(plane, plane)
            IRBlock: Conv3x3(inplane, inplane) -> Conv3x3(inplane, plane, stride)
    '''
    def __init__(self, in_channel, out_channel, downsample=False):
        super(IRBlock, self).__init__()
        
        # main branch
        # block1
        self.bn0 = torch.nn.BatchNorm2d(in_channel)
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channel)
        self.relu1 = torch.nn.PReLU(out_channel)
        # block2
        if downsample:
            self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        # self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
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

        # # merge
        # self.relu_out = torch.nn.PReLU()

    def forward(self, inputs):
        # main
        main = self.bn0(inputs)
        main = self.conv1(main)
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
        outs = main+shortcut
        return outs

class ResNetIR(torch.nn.Module):
    def __init__(self, input_size, num_classes, block_num_list, **kargs):
        '''
                    
        '''
        super(ResNetIR, self).__init__(**kargs)
        self.input_size = input_size

        # stage 1
        self.conv1 = torch.nn.Conv2d(input_size[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.PReLU(64)
        # self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # stage 2
        block1_list = []
        for i in range(block_num_list[0]):
            if i == 0:
                block1_list.append(IRBlock(64, 64, downsample=True))
            else:
                block1_list.append(IRBlock(64, 64))
        self.block1 = torch.nn.Sequential(*block1_list)

        # stage 3
        block2_list = []
        for i in range(block_num_list[1]):
            if i == 0:
                block2_list.append(IRBlock(64, 128, downsample=True))
            else:
                block2_list.append(IRBlock(128, 128))
        self.block2 = torch.nn.Sequential(*block2_list)

        # stage 4
        block3_list = []
        for i in range(block_num_list[2]):
            if i == 0:
                block3_list.append(IRBlock(128, 256, downsample=True))
            else:
                block3_list.append(IRBlock(256, 256))
        self.block3 = torch.nn.Sequential(*block3_list)

        # stage 5
        block4_list = []
        for i in range(block_num_list[3]):
            if i == 0:
                block4_list.append(IRBlock(256, 512, downsample=True))
            else:
                block4_list.append(IRBlock(512, 512))
        self.block4 = torch.nn.Sequential(*block4_list)

        # stage 6
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.dropout = torch.nn.Dropout(0.4)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(512*int(input_size[1]/16)*int(input_size[1]/16), num_classes)
        self.bn3 = torch.nn.BatchNorm1d(num_classes)


    def forward(self, inputs):

        # stage 1
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu1(outputs)
        # outputs = self.pool1(outputs)

        # stage 2-5
        outputs = self.block1(outputs)
        outputs = self.block2(outputs)
        outputs = self.block3(outputs)
        outputs = self.block4(outputs)

        # stage 6
        outputs = self.bn2(outputs)
        outputs = self.dropout(outputs)
        outputs = self.flatten(outputs)
        # print(outputs.size())
        outputs = self.linear1(outputs)
        outputs = self.bn3(outputs)

        return outputs

def ResNet18_IR(input_size, num_classes, **kargs):
    return ResNetIR(input_size, num_classes, block_num_list=[2,2,2,2], **kargs)

def ResNet34_IR(input_size, num_classes, **kargs):
    return ResNetIR(input_size, num_classes, block_num_list=[3,4,6,3], **kargs)

def ResNet50_IR(input_size, num_classes, **kargs):
    return ResNetIR(input_size, num_classes, block_num_list=[3,4,14,3], **kargs)

def ResNet100_IR(input_size, num_classes, **kargs):
    return ResNetIR(input_size, num_classes, block_num_list=[3,13,30,3], **kargs)


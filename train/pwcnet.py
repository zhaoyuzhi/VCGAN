import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import correlation

# ----------------------------------------
#          PWCNet Forward Method
# ----------------------------------------
def PWCEstimate(pwcnet, tensorFirst, tensorSecond, drange = False, reshape = False):
    # Ensure the frames are continuous
    assert tensorFirst.size() == tensorSecond.size()
    # The input tensor should be 4D; cuda / cpu both are OK
    B, C, H, W = tensorFirst.size()
    # if drange == False: the input is in range of [-1, 1]; however, the PWC-Net needs range of [0, 1]
    if drange == False:
        tensorFirst = (tensorFirst + 1) / 2
        tensorSecond = (tensorSecond + 1) / 2
    # Process the frames and ensure the size is the multiple of basesize
    if reshape:
        tensorPreprocessedFirst, intPreprocessedHeight, intPreprocessedWidth = Reshape_Tensor(tensorFirst)
        tensorPreprocessedSecond, intPreprocessedHeight, intPreprocessedWidth = Reshape_Tensor(tensorSecond)
        # forward
        tensorFlow = pwcnet(tensorPreprocessedFirst, tensorPreprocessedSecond)
        # alignment
        tensorFlow = 20.0 * F.interpolate(input = tensorFlow, size = (H, W), mode = 'bilinear', align_corners = False)
        tensorFlow[:, 0, :, :] *= float(W) / float(intPreprocessedWidth)
        tensorFlow[:, 1, :, :] *= float(H) / float(intPreprocessedHeight)
    else:
        # forward
        tensorFlow = pwcnet(tensorFirst, tensorSecond)
        tensorFlow = 20.0 * F.interpolate(input = tensorFlow, size = (H, W), mode = 'bilinear', align_corners = False)
    return tensorFlow

def Reshape_Tensor(tensor, basesize = 64.0):
    # The input tensor should be 4D; cuda / cpu both are OK
    B, C, H, W = tensor.size()
    # Compute the target H and W according to the given size (multiple of basesize)
    intPreprocessedHeight = int(math.floor(math.ceil(H / basesize) * basesize))
    intPreprocessedWidth = int(math.floor(math.ceil(W / basesize) * basesize))
    tensorPreprocessed = F.interpolate(input = tensor, size = (intPreprocessedHeight, intPreprocessedWidth), mode = 'bilinear', align_corners = False)
    return tensorPreprocessed, intPreprocessedHeight, intPreprocessedWidth

# ----------------------------------------
#      PWCNet Backward Method (Warp)
# ----------------------------------------

def PWCNetBackward(tensorInput, tensorFlow):

    Backward_tensorGrid = {}
    Backward_tensorPartial = {}

    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
    
    if str(tensorFlow.size()) not in Backward_tensorPartial:
        Backward_tensorPartial[str(tensorFlow.size())] = tensorFlow.new_ones([ tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3) ])

    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
    tensorInput = torch.cat([ tensorInput, Backward_tensorPartial[str(tensorFlow.size())] ], 1)

    tensorOutput = F.grid_sample(input = tensorInput, grid = (Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode = 'bilinear', padding_mode = 'border')

    tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

    return tensorOutput[:, :-1, :, :] * tensorMask

# ----------------------------------------
#              PWCNet Modules
# ----------------------------------------
class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.moduleOne = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1)
        )
        self.moduleTwo = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1)
        )
        self.moduleThr = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1)
        )
        self.moduleFou = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 2, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1)
        )
        self.moduleFiv = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1)
        )
        self.moduleSix = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 196, kernel_size = 3, stride = 2, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1)
        )
        
    def forward(self, tensorInput):
        tensorOne = self.moduleOne(tensorInput)
        tensorTwo = self.moduleTwo(tensorOne)
        tensorThr = self.moduleThr(tensorTwo)
        tensorFou = self.moduleFou(tensorThr)
        tensorFiv = self.moduleFiv(tensorFou)
        tensorSix = self.moduleSix(tensorFiv)
        return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]

class Decoder(nn.Module):
    def __init__(self, intLevel):
        super(Decoder, self).__init__()

        intPrevious = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 1]
        intCurrent = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 0]

        if intLevel < 6:
            self.moduleUpflow = nn.ConvTranspose2d(in_channels = 2, out_channels = 2, kernel_size = 4, stride = 2, padding = 1)
        if intLevel < 6:
            self.moduleUpfeat = nn.ConvTranspose2d(in_channels = intPrevious + 128 + 128 + 96 + 64 + 32, out_channels = 2, kernel_size = 4, stride = 2, padding = 1)
        if intLevel < 6:
            self.dblBackward = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

        self.moduleOne = nn.Sequential(
            nn.Conv2d(in_channels = intCurrent, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1)
        )
        self.moduleTwo = nn.Sequential(
            nn.Conv2d(in_channels = intCurrent + 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1)
        )
        self.moduleThr = nn.Sequential(
            nn.Conv2d(in_channels = intCurrent + 128 + 128, out_channels = 96, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1)
        )
        self.moduleFou = nn.Sequential(
            nn.Conv2d(in_channels = intCurrent + 128 + 128 + 96, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1)
        )
        self.moduleFiv = nn.Sequential(
            nn.Conv2d(in_channels = intCurrent + 128 + 128 + 96 + 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1)
        )
        self.moduleSix = nn.Sequential(
            nn.Conv2d(in_channels = intCurrent + 128 + 128 + 96 + 64 + 32, out_channels = 2, kernel_size = 3, stride = 1, padding = 1)
        )

    def forward(self, tensorFirst, tensorSecond, objectPrevious):

        tensorFlow = None
        tensorFeat = None

        if objectPrevious is None:
            tensorFlow = None
            tensorFeat = None
            tensorCorr = correlation.FunctionCorrelation(tensorFirst = tensorFirst, tensorSecond = tensorSecond)
            tensorVolume = F.leaky_relu(input = tensorCorr, negative_slope = 0.1, inplace = False)
            tensorFeat = torch.cat([tensorVolume], 1)

        elif objectPrevious is not None:
            tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
            tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])
            tensorWarp = PWCNetBackward(tensorInput = tensorSecond, tensorFlow = tensorFlow * self.dblBackward)
            tensorCorr = correlation.FunctionCorrelation(tensorFirst = tensorFirst, tensorSecond = tensorWarp)
            tensorVolume = F.leaky_relu(input = tensorCorr, negative_slope = 0.1, inplace = False)
            tensorFeat = torch.cat([tensorVolume, tensorFirst, tensorFlow, tensorFeat], 1)
        
        tensorFeat = torch.cat([self.moduleOne(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleTwo(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleThr(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleFou(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleFiv(tensorFeat), tensorFeat], 1)

        tensorFlow = self.moduleSix(tensorFeat)

        return { 'tensorFlow': tensorFlow, 'tensorFeat': tensorFeat }

class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()
        self.moduleMain = nn.Sequential(
            nn.Conv2d(in_channels = 81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 4, dilation = 4),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 128, out_channels = 96, kernel_size = 3, stride = 1, padding = 8, dilation = 8),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 96, out_channels = 64, kernel_size = 3, stride = 1, padding = 16, dilation = 16),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.LeakyReLU(inplace = False, negative_slope = 0.1),
            nn.Conv2d(in_channels = 32, out_channels = 2, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
        )
    
    def forward(self, tensorInput):
        return self.moduleMain(tensorInput)

# ----------------------------------------
#                 PWCNet
# ----------------------------------------
class PWCNet(nn.Module):
    def __init__(self):
        super(PWCNet, self).__init__()
        # Extractor
        self.moduleExtractor = Extractor()
        # Decoders
        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)
        # Refiner
        self.moduleRefiner = Refiner()
        
    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = self.moduleExtractor(tensorFirst)
        tensorSecond = self.moduleExtractor(tensorSecond)

        objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
        objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate)
        objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate)
        objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate)
        objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate)

        return objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])

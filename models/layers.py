"""Contains novel layer definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self, threshold=0.0):
        super(Binarizer, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        outputs = torch.sigmoid(inputs)
        #outputs = inputs.clone()
        #outputs[inputs.le(self.threshold)] = 0.0
        #outputs[inputs.gt(self.threshold)] = 1.0
        return outputs

    def backward(self, gradOutput):
        return gradOutput

'''
# changed by for audo threshold        
class Binarizer_auto(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self):
        super(Binarizer_auto, self).__init__()
        #self.threshold = nn.Parameter(torch.Tensor([threshold]), requires_grad = True)
#         self.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(0)] = 0.0
        outputs[inputs.gt(0)] = 1.0
        return outputs

    def backward(self, gradOutput):
        return gradOutput
'''
'''
class Ternarizer(torch.autograd.Function):
    """Ternarizes {-1, 0, 1} a real valued tensor."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        super(Ternarizer, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > self.threshold] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput
'''

class ElementWiseConv2d(nn.Module):
    """Modified conv with masks for weights."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 mask_init='uniform', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=0.0):
        super(ElementWiseConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups

        # imagenet pretrained weight
        self.imagenet_weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size), requires_grad=True)

        # place365 weight  no bias now
        self.place365_weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size), requires_grad=True)

        # Initialize real-valued mask weights.
        self.mask_real = self.imagenet_weight.data.new(self.imagenet_weight.size())
        
        if mask_init == '1s':
            self.mask_real.fill_(mask_scale)

        elif mask_init == 'uniform':
            self.mask_real.uniform_(-1 * mask_scale, mask_scale)

        # mask_real is now a trainable parameter.
        self.mask_real = Parameter(self.mask_real)
        '''
        # changed for audo threshold
        self.threshold = nn.Parameter(torch.Tensor([threshold]), requires_grad = False)
        '''

        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
             print('Calling binarizer with threshold:',threshold)
             self.threshold_fn = Binarizer(threshold= threshold)
        elif threshold_fn == 'ternarizer':
             print('Calling ternarizer with threshold:', threshold)
             self.threshold_fn = Ternarizer(threshold= threshold)

    def forward(self, input):
        # Get binarized/ternarized mask from real-valued mask.
        #mask_thresholded = self.threshold_fn(self.mask_real)

        #mask_thresholded = torch.sigmoid(self.mask_real)
        prob_data = self.mask_real.clone()
        prob_data[self.mask_real.le(0.5)] = 0
        prob_data[ self.mask_real.gt(0.5)] = 1
        mask_thresholded = (prob_data -  self.mask_real).detach() + self.mask_real

        # changed  for audo threshold
        #mask_thresholded = Binarizer_auto()(self.mask_real+self.threshold)

        # Mask weights with above mask.
        weight_combined = mask_thresholded * self.place365_weight + (1 - mask_thresholded) * self.imagenet_weight
        #weight_combined = self.place365_weight 

        # Perform conv using modified weight.
        return F.conv2d(input, weight_combined, None, self.stride,
                        self.padding, self.dilation, self.groups)

    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.imagenet_weight.data = fn(self.imagenet_weight.data)

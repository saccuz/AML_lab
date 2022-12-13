import torch
import torch.nn as nn
from torch.autograd import Function
import copy

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}



class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        #load weights
        self.load_state_dict(load_state_dict_from_url(model_urls['alexnet']), strict=False)
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        class_logits = self.classifier(features)
        return class_logits


class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class AlexNetDA(nn.Module):
  def __init__(self, alexnet, num_domains):
    super(AlexNetDA, self).__init__()
    # Common feature extractor
    self.features = alexnet.features

    # Category Classifier
    self.classifier = alexnet.classifier

    # Domain Classifier
    # start from 'self.classifier' weights, override last FC layer with a new one (output dim is 'num_domains')
    self.domain_classifier = copy.deepcopy(alexnet.classifier)
    last_l_idx = len(self.domain_classifier) - 1
    self.domain_classifier[last_l_idx] = nn.Linear(4096, num_domains)

  def forward(self, images, alpha=None):
    features = self.features(images)
    features = features.view(features.size(0), -1)
    
    if alpha is not None:
        # perform adaptation round
        # logits output dim is num_domains
        features = ReverseLayerF.apply(features, alpha)
        return self.domain_classifier(features)
    else:
        return self.classifier(features)


  


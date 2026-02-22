import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models
from comparemodel.Nets import  AlexNet, ResNet, BasicBlock
from comparemodel.hmtnet_fbp import GenderBranch,HMTNet  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create ResNeXt-50 model instance
resnext50 = models.resnext50_32x4d(pretrained=False)

# function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# create model instance
def load_model(pretrained_dict, new):
    model_dict = new.state_dict()
    # filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    new.load_state_dict(model_dict)
    
def load_model_state(hmt_net, state_dict_path):
    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        hmt_net = nn.DataParallel(hmt_net)
        hmt_net.load_state_dict(torch.load(state_dict_path))
    else:
        state_dict = torch.load(state_dict_path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        hmt_net.load_state_dict(new_state_dict)


# count and print parameters
total_params = count_parameters(resnext50)

# Create model instances; set YOUR_MODEL_ROOT to directory containing alexnet.pth, resnet18.pth, HMTNet.pth
YOUR_MODEL_ROOT = "your_location"
alexnet_model = AlexNet(num_classes=1)
model_path = os.path.join(YOUR_MODEL_ROOT, "alexnet.pth")
checkpoint = torch.load(model_path, map_location=torch.device("cpu"), encoding="latin1")
load_model(checkpoint, alexnet_model)

resnet_model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1)
model_path2 = os.path.join(YOUR_MODEL_ROOT, "resnet18.pth")
load_model(torch.load(model_path2, encoding="latin1"), resnet_model)

# Create HMTNET model instance
hmtnet_model = HMTNet()
hmtnet_model_path2 = os.path.join(YOUR_MODEL_ROOT, "HMTNet.pth")
load_model_state(hmtnet_model, hmtnet_model_path2)

# count and print parameters
alexnet_params = count_parameters(alexnet_model)
resnet_params = count_parameters(resnet_model)

hmtnet_params = count_parameters(hmtnet_model)
print(f"HMTNET #params: {hmtnet_params}")
print(f"AlexNet #params: {alexnet_params}")
print(f"ResNet #params: {resnet_params}")
print(f"ResNeXt-50 #params: {total_params}")
print(f"HMTNET #params (M): {hmtnet_params / 1e6:.2f}M")
print(f"AlexNet #params (M): {alexnet_params / 1e6:.2f}M")
print(f"ResNet #params (M): {resnet_params / 1e6:.2f}M")
print(f"ResNeXt-50 #params (M): {total_params / 1e6:.2f}M")


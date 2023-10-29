from models import ResNet

def get_resnet_model(resnet_type=152, pretrained=True):
    """
    A function that returns the required pre-trained resnet model
    :param resnet_number: the resnet type
    :return: the pre-trained model
    """
    if resnet_type == 18:
        return ResNet.resnet18(pretrained=pretrained, progress=True)
    elif resnet_type == 50:
        return ResNet.wide_resnet50_2(pretrained=pretrained, progress=True)
    elif resnet_type == 101:
        return ResNet.resnet101(pretrained=pretrained, progress=True)
    else:
        return ResNet.resnet152(pretrained=pretrained, progress=True)

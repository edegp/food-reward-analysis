from torchvision.models import convnext_base, convnext_tiny, resnet152, resnet50, vgg16
from torch import nn


def init_convnext_tiny():
    # Re-initialize the model and optimizer for each fold
    model = convnext_tiny(pretrained=True, weights="DEFAULT")
    # # フリーズ: すべてのパラメータを固定 (requires_grad=False)
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the classifier to output a single continuous value for regression
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)

    # # # 最終層だけを学習するように requires_grad=True を再設定
    # for param in model.classifier[2].parameters():
    #     param.requires_grad = True
    return model


def init_convnext_base():
    # Re-initialize the model and optimizer for each fold
    model = convnext_base(pretrained=True, weights="DEFAULT")
    # # フリーズ: すべてのパラメータを固定 (requires_grad=False)
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the classifier to output a single continuous value for regression
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)

    # # 最終層だけを学習するように requires_grad=True を再設定
    # for param in model.classifier[2].parameters():
    #     param.requires_grad = True
    return model


def init_resnet50():
    # Re-initialize the model and optimizer for each fold
    model = resnet50(weights="DEFAULT")
    # # フリーズ: すべてのパラメータを固定 (requires_grad=False)
    # for param in model.parameters():
    #     param.requires_grad = False
    # Modify the classifier to output a single continuous value for regression
    model.fc = nn.Linear(model.fc.in_features, 1)
    # # 最終層だけを学習するように requires_grad=True を再設定
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    return model


def init_resnet152():
    # Re-initialize the model and optimizer for each fold
    model = resnet152(weights="DEFAULT")
    # # フリーズ: すべてのパラメータを固定 (requires_grad=False)
    # for param in model.parameters():
    #     param.requires_grad = False
    # Modify the classifier to output a single continuous value for regression
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, model.fc.in_features // 4),
        nn.Linear(model.fc.in_features // 4, 1),
    )
    # # 最終層だけを学習するように requires_grad=True を再設定
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    return model


def init_vgg16():
    # Re-initialize the model and optimizer for each fold
    model = vgg16(weights="DEFAULT")
    # classifier[6]: 4096 → 1 (v9と同じ構造)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
    return model


def init_vgg16_freeze():
    # Re-initialize the model and optimizer for each fold
    model = vgg16(weights="DEFAULT")
    # フリーズ: すべてのパラメータを固定 (requires_grad=False)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, model.classifier[6].in_features
    )
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)

    # 最終層だけを学習するように requires_grad=True を再設定
    for param in model.classifier[3].parameters():
        param.requires_grad = True
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    return model

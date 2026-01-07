import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader


def get_mean_std(
    dataset,
):
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, pin_memory=0)
    cnt = 0
    fst_moment = torch.zeros(3)
    snd_moment = torch.zeros(3)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    for images, labels in loader:
        b, c, h, w = images.shape

        nb_pixels = b * h * w

        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images**2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels
    variance = torch.clamp(snd_moment - fst_moment**2, min=1e-2)
    return fst_moment, torch.sqrt(variance)


print("train transforms:")
print(
    "resize 240x240",
    end="\n↓\n",
)
print(
    "center crop 224x224",
    end="\n↓\n",
)
print(
    "random horizontal flip",
    end="\n↓\n",
)
print(
    "random affine degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.2), fill=(255, 255, 255)",
    end="\n↓\n",
)
print(
    "gaussian blur kernel_size=(5, 5), sigma=(0.1, 3)",
    end="\n↓\n",
)
print(
    "color jitter brightness=0.075, contrast=0, saturation=0.03, hue=0.03",
    end="\n↓\n",
)
print(
    "to tensor",
    end="\n↓\n",
)


def get_train_transforms(mean, std):
    print("normalize", mean, std)

    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((240, 240)),
            v2.CenterCrop(224),
            v2.RandomHorizontalFlip(),
            v2.RandomAffine(
                degrees=20,
                translate=(0.2, 0.2),
                scale=(0.7, 1.2),
                fill=(255, 255, 255),
            ),
            v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.01, 4)),
            v2.ColorJitter(brightness=0.075, contrast=0, saturation=0.03, hue=0.03),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms(mean, std):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((240, 240)),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

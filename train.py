from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


# Лабораторная работа 1, вариант 3:
# Архитектура: RegNet
# Аугментация: Rotation
# Оптимизатор варианта: RAdam
#
# ВАЖНО: torchvision.models здесь НЕ используется.
# Архитектура RegNet-Y-400MF описана вручную через nn.Module.
# Для экспериментов с pretrained загружаются только веса ImageNet в эту самописную архитектуру.

EXPERIMENTS = {
    "pretrained_adam": {"pretrained": True, "optimizer": "adam"},
    "pretrained_radam": {"pretrained": True, "optimizer": "radam"},
    "scratch_adam": {"pretrained": False, "optimizer": "adam"},
}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Официальные веса RegNet-Y-400MF ImageNet-1K.
# Используются только как state_dict, без импорта torchvision.models.
REGNET_Y_400MF_WEIGHTS_URL = "https://download.pytorch.org/models/regnet_y_400mf-e6988f5f.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual RegNet-Y-400MF + Rotation + RAdam for Stanford Dogs."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to Stanford Dogs root or Images folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment", choices=(*EXPERIMENTS.keys(), "all"), default="all")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--rotation", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None, help="For fast debug runs only.")
    parser.add_argument(
        "--pretrained-path",
        type=Path,
        default=None,
        help="Optional local .pth file with RegNet-Y-400MF ImageNet weights.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze feature extractor and train only classifier for pretrained experiments.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def images_dir(path: Path) -> Path:
    """Accept either StanfordDogs root or the inner Images directory."""
    if (path / "Images").exists():
        return path / "Images"
    return path


# -----------------------------------------------------------------------------
# Самостоятельная реализация RegNet-Y-400MF
# -----------------------------------------------------------------------------


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Аналог функции округления каналов до числа, кратного group_width."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


class Conv2dNormActivation(nn.Sequential):
    """Свёртка + BatchNorm + активация.

    Класс написан вручную, но имена внутренних слоёв совпадают с форматом
    официального state_dict: 0 - Conv2d, 1 - BatchNorm2d, 2 - ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        bias: Optional[bool] = None,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2
        if bias is None:
            bias = norm_layer is None

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        super().__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(nn.Module):
    """SE-блок: адаптивное среднее, два 1x1 Conv и масштабирование каналов."""

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, kernel_size=1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, x: Tensor) -> Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, x: Tensor) -> Tensor:
        return self._scale(x) * x


class SimpleStemIN(Conv2dNormActivation):
    """Начальный stem-блок RegNet для ImageNet: 3x3 Conv, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__(
            width_in,
            width_out,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )


class BottleneckTransform(nn.Sequential):
    """Bottleneck-преобразование RegNet: 1x1, 3x3 grouped conv + SE, 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        groups = w_b // group_width

        layers["a"] = Conv2dNormActivation(
            width_in,
            w_b,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        layers["b"] = Conv2dNormActivation(
            w_b,
            w_b,
            kernel_size=3,
            stride=stride,
            groups=groups,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        if se_ratio:
            # В RegNet-Y squeeze_channels считаются от входной ширины блока.
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )
        layers["c"] = Conv2dNormActivation(
            w_b,
            width_out,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=norm_layer,
            activation_layer=None,
        )
        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: y = ReLU(shortcut(x) + F(x))."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.proj: Optional[nn.Module] = None
        if width_in != width_out or stride != 1:
            self.proj = Conv2dNormActivation(
                width_in,
                width_out,
                kernel_size=1,
                stride=stride,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation = activation_layer(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.proj(x) if self.proj is not None else x
        return self.activation(shortcut + self.f(x))


class AnyStage(nn.Sequential):
    """Stage RegNet: несколько residual-блоков с одинаковой выходной шириной."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
        stage_index: int,
    ) -> None:
        super().__init__()
        for i in range(depth):
            block = ResBottleneckBlock(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
            )
            self.add_module(f"block{stage_index}-{i}", block)


class BlockParams:
    """Вычисление stage-параметров RegNet из исходных параметров RegNet-Y-400MF."""

    def __init__(
        self,
        depths: list[int],
        widths: list[int],
        group_widths: list[int],
        bottleneck_multipliers: list[float],
        strides: list[int],
        se_ratio: Optional[float],
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def regnet_y_400mf(cls) -> "BlockParams":
        # Эти значения соответствуют RegNet-Y-400MF:
        # depth=16, w0=48, wa=27.89, wm=2.09, group_width=8, se_ratio=0.25
        depth = 16
        w_0 = 48
        w_a = 27.89
        w_m = 2.09
        group_width = 8
        bottleneck_multiplier = 1.0
        se_ratio = 0.25

        quant = 8
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (
            torch.round(torch.divide(w_0 * torch.pow(torch.tensor(w_m), block_capacity), quant)) * quant
        ).int().tolist()

        splits = [
            w != wp or r != rp
            for w, wp, r, rp in zip(
                block_widths + [0],
                [0] + block_widths,
                block_widths + [0],
                [0] + block_widths,
            )
        ]
        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([i for i, t in enumerate(splits) if t])).int().tolist()

        strides = [2] * len(stage_widths)
        bottleneck_multipliers = [bottleneck_multiplier] * len(stage_widths)
        group_widths = [group_width] * len(stage_widths)

        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_multipliers)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]
        widths_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(widths_bot, bottleneck_multipliers)]

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths_min,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def expanded(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)


class ManualRegNetY400MF(nn.Module):
    """Самостоятельная реализация RegNet-Y-400MF.

    Имена модулей сохранены совместимыми с официальным state_dict, чтобы можно было
    загрузить ImageNet-веса без использования готовой модели torchvision.models.
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        norm_layer = lambda channels: nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
        activation = nn.ReLU
        block_params = BlockParams.regnet_y_400mf()

        self.stem = SimpleStemIN(3, 32, norm_layer, activation)

        current_width = 32
        blocks: list[tuple[str, nn.Module]] = []
        for i, (width_out, stride, depth, group_width, bottleneck_multiplier) in enumerate(block_params.expanded()):
            blocks.append(
                (
                    f"block{i + 1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )
            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(current_width, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.trunk_output(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def load_pretrained_backbone(model: ManualRegNetY400MF, args: argparse.Namespace) -> None:
    """Загрузка ImageNet-весов в самописную RegNet.

    Классификатор fc не загружается, потому что в ImageNet 1000 классов,
    а в Stanford Dogs другое число классов.
    """
    if args.pretrained_path is not None:
        state_dict = torch.load(args.pretrained_path, map_location="cpu")
    else:
        state_dict = torch.hub.load_state_dict_from_url(
            REGNET_Y_400MF_WEIGHTS_URL,
            map_location="cpu",
            progress=True,
            check_hash=True,
        )

    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("fc.")]
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading pretrained weights: {unexpected}")
    print(f"Loaded pretrained backbone. Missing keys: {missing}")


def freeze_backbone(model: ManualRegNetY400MF) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc.")


# -----------------------------------------------------------------------------
# Данные, обучение, метрики
# -----------------------------------------------------------------------------


def stratified_split(targets: list[int], seed: int) -> tuple[list[int], list[int], list[int]]:
    random.seed(seed)
    by_class: dict[int, list[int]] = {}
    for i, target in enumerate(targets):
        by_class.setdefault(target, []).append(i)

    train_ids, val_ids, test_ids = [], [], []
    for ids in by_class.values():
        random.shuffle(ids)
        train_end = int(len(ids) * 0.70)
        val_end = int(len(ids) * 0.85)
        train_ids += ids[:train_end]
        val_ids += ids[train_end:val_end]
        test_ids += ids[val_end:]

    random.shuffle(train_ids)
    random.shuffle(val_ids)
    random.shuffle(test_ids)
    return train_ids, val_ids, test_ids


def make_loaders(args: argparse.Namespace, device: torch.device):
    root = images_dir(args.data_dir)
    resize_size = round(args.image_size * 256 / 224)

    train_tf = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomRotation(args.rotation, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    base = datasets.ImageFolder(root)
    train_data = datasets.ImageFolder(root, transform=train_tf)
    val_data = datasets.ImageFolder(root, transform=test_tf)
    test_data = datasets.ImageFolder(root, transform=test_tf)

    train_ids, val_ids, test_ids = stratified_split(base.targets, args.seed)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(Subset(train_data, train_ids), shuffle=True, **loader_args)
    val_loader = DataLoader(Subset(val_data, val_ids), shuffle=False, **loader_args)
    test_loader = DataLoader(Subset(test_data, test_ids), shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader, base.classes


def make_model(num_classes: int, pretrained: bool, args: argparse.Namespace) -> nn.Module:
    model = ManualRegNetY400MF(num_classes=num_classes)
    if pretrained:
        load_pretrained_backbone(model, args)
        if args.freeze_backbone:
            freeze_backbone(model)
    return model


def make_optimizer(name: str, model: nn.Module, args: argparse.Namespace):
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "radam":
        return RAdam(params, lr=args.lr, weight_decay=args.weight_decay)
    return Adam(params, lr=args.lr, weight_decay=args.weight_decay)


def metrics(confusion: torch.Tensor, loss: float) -> dict[str, float]:
    confusion = confusion.float()
    tp = confusion.diag()
    precision_by_class = tp / confusion.sum(0).clamp_min(1)
    recall_by_class = tp / confusion.sum(1).clamp_min(1)
    f1_by_class = 2 * precision_by_class * recall_by_class / (precision_by_class + recall_by_class).clamp_min(1e-12)
    precision = precision_by_class.mean().item()
    recall = recall_by_class.mean().item()
    f1 = f1_by_class.mean().item()
    accuracy = tp.sum().item() / max(confusion.sum().item(), 1)
    return {"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer=None,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    confusion = torch.zeros(num_classes, num_classes)
    total_loss, total_count = 0.0, 0

    for batch, (x, y) in enumerate(tqdm(loader, leave=False)):
        if max_batches is not None and batch >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = loss_fn(pred, y)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(y)
        total_count += len(y)
        for true, guessed in zip(y.cpu(), pred.argmax(1).cpu()):
            confusion[true, guessed] += 1

    return metrics(confusion, total_loss / max(total_count, 1))


def save_history(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_history(rows: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(9, 4))
    plt.plot(epochs, [row["train_f1"] for row in rows], label="train F1")
    plt.plot(epochs, [row["val_f1"] for row in rows], label="val F1")
    plt.plot(epochs, [row["val_accuracy"] for row in rows], label="val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train_experiment(
    name: str,
    settings: dict,
    args: argparse.Namespace,
    loaders,
    class_names: list[str],
) -> dict:
    train_loader, val_loader, test_loader = loaders
    device = torch.device(args.device)
    out = args.output_dir / name
    out.mkdir(parents=True, exist_ok=True)

    model = make_model(len(class_names), settings["pretrained"], args).to(device)
    optimizer = make_optimizer(settings["optimizer"], model, args)
    loss_fn = nn.CrossEntropyLoss()
    history, best_f1 = [], -1.0
    best_model = out / "best_model.pt"

    print(f"\n{name}")
    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, loss_fn, device, len(class_names), optimizer, args.max_batches)
        val_m = run_epoch(model, val_loader, loss_fn, device, len(class_names), None, args.max_batches)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch}: "
            f"train f1={train_m['f1']:.4f}, val f1={val_m['f1']:.4f}, val acc={val_m['accuracy']:.4f}"
        )
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(model.state_dict(), best_model)

    model.load_state_dict(torch.load(best_model, map_location=device))
    test_m = run_epoch(model, test_loader, loss_fn, device, len(class_names), None, args.max_batches)
    result = {
        "experiment": name,
        "architecture": "ManualRegNetY400MF",
        "pretrained": settings["pretrained"],
        "optimizer": settings["optimizer"],
        **test_m,
    }

    save_history(history, out / "history.csv")
    plot_history(history, out / "learning_curves.png")
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def save_summary(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "architecture", "pretrained", "optimizer", "accuracy", "precision", "recall", "f1", "loss"]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in results])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = make_loaders(args, device)
    loaders = (train_loader, val_loader, test_loader)
    names = EXPERIMENTS.keys() if args.experiment == "all" else [args.experiment]

    results = [
        train_experiment(name, EXPERIMENTS[name], args, loaders, class_names)
        for name in names
    ]
    save_summary(results, args.output_dir)

    print("\nSummary:")
    for row in results:
        print(
            f"{row['experiment']}: "
            f"accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, "
            f"f1={row['f1']:.4f}, "
            f"loss={row['loss']:.4f}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from torchvision import datasets, models, transforms
from torchvision.transforms import InterpolationMode


EXPERIMENTS = {
    "pretrained_adam": {"pretrained": True, "optimizer": "adam"},
    "pretrained_radam": {"pretrained": True, "optimizer": "radam"},
    "scratch_adam": {"pretrained": False, "optimizer": "adam"},
}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RegNet + Rotation + RAdam for Stanford Dogs.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to Stanford Dogs Images folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment", choices=(*EXPERIMENTS.keys(), "all"), default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--rotation", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def images_dir(path: Path) -> Path:
    if (path / "Images").exists():
        return path / "Images"
    return path


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
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(Subset(train_data, train_ids), shuffle=True, **loader_args)
    val_loader = DataLoader(Subset(val_data, val_ids), shuffle=False, **loader_args)
    test_loader = DataLoader(Subset(test_data, test_ids), shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader, base.classes


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


def make_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None
    model = models.regnet_y_400mf(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_optimizer(name: str, model: nn.Module, args: argparse.Namespace):
    if name == "radam":
        return RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


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


def run_epoch(model, loader, loss_fn, device, num_classes, optimizer=None, max_batches=None) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    confusion = torch.zeros(num_classes, num_classes)
    total_loss, total_count = 0.0, 0

    for batch, (x, y) in enumerate(tqdm(loader, leave=False)):
        if max_batches is not None and batch >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()

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


def train_experiment(name: str, settings: dict, args: argparse.Namespace, loaders, class_names: list[str]) -> dict:
    train_loader, val_loader, test_loader = loaders
    device = torch.device(args.device)
    out = args.output_dir / name
    out.mkdir(parents=True, exist_ok=True)

    model = make_model(len(class_names), settings["pretrained"]).to(device)
    optimizer = make_optimizer(settings["optimizer"], model, args)
    loss_fn = nn.CrossEntropyLoss()
    history, best_f1 = [], -1.0
    best_model = out / "best_model.pt"

    print(f"\n{name}")
    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, loss_fn, device, len(class_names), optimizer, args.max_batches)
        val_m = run_epoch(model, val_loader, loss_fn, device, len(class_names), None, args.max_batches)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(f"epoch {epoch}: train f1={train_m['f1']:.4f}, val f1={val_m['f1']:.4f}")
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(model.state_dict(), best_model)

    model.load_state_dict(torch.load(best_model, map_location=device))
    test_m = run_epoch(model, test_loader, loss_fn, device, len(class_names), None, args.max_batches)
    result = {"experiment": name, "pretrained": settings["pretrained"], "optimizer": settings["optimizer"], **test_m}

    save_history(history, out / "history.csv")
    plot_history(history, out / "learning_curves.png")
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def save_summary(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "pretrained", "optimizer", "accuracy", "precision", "recall", "f1"]
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
    names = EXPERIMENTS.keys() if args.experiment == "all" else [args.experiment]
    results = [
        train_experiment(name, EXPERIMENTS[name], args, (train_loader, val_loader, test_loader), class_names)
        for name in names
    ]
    save_summary(results, args.output_dir)
    print("\nSummary:")
    for row in results:
        print(f"{row['experiment']}: accuracy={row['accuracy']:.4f}, precision={row['precision']:.4f}, recall={row['recall']:.4f}, f1={row['f1']:.4f}")


if __name__ == "__main__":
    main()

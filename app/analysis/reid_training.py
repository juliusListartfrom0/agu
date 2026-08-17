"""Train and evaluate a benchmark-disjoint basketball player ReID backbone."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REID_CHECKPOINT_SCHEMA = "agu.mobilenet-v3-small-reid.v1"


@dataclass(frozen=True)
class ReIDTrainingResult:
    checkpoint_path: Path
    checkpoint_sha256: str
    baseline_top1: float
    trained_top1: float
    train_samples: int
    validation_samples: int


class ManifestCropDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        crop_root: Path,
        class_indices: dict[str, int],
        *,
        augment: bool,
    ) -> None:
        self.samples = samples
        self.crop_root = crop_root
        self.class_indices = class_indices
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = cv2.imread(str(self.crop_root / sample["path"]))
        if image is None:
            raise RuntimeError(f"unable to read ReID crop: {sample['path']}")
        if self.augment and random.random() < 0.5:
            image = cv2.flip(image, 1)
        return preprocess_mobilenet_crop(image), self.class_indices[sample["identity_id"]]


def preprocess_mobilenet_crop(crop_bgr: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def load_training_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    supplied_hash = payload.pop("manifest_sha256", None)
    actual_hash = _json_sha256(payload)
    payload["manifest_sha256"] = supplied_hash
    if supplied_hash != actual_hash:
        raise ValueError("ReID training manifest hash mismatch")
    if payload.get("training_only") is not True or payload.get("acceptance_media_allowed") is not False:
        raise ValueError("ReID training requires acceptance-forbidden training data")
    if payload.get("benchmark_disjoint") is not True:
        raise ValueError("ReID training manifest must be benchmark-disjoint")
    return payload


def split_samples_by_sequence(
    manifest: dict[str, Any], validation_sequences: Iterable[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation = set(validation_sequences)
    if not validation:
        raise ValueError("at least one validation sequence is required")
    train: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = []
    for sample in manifest.get("samples") or []:
        sequence = str(sample["path"]).split("/", 1)[0]
        (held_out if sequence in validation else train).append(sample)
    train_ids = {sample["identity_id"] for sample in train}
    validation_ids = {sample["identity_id"] for sample in held_out}
    if not train or not held_out:
        raise ValueError("training and validation splits must both be non-empty")
    if validation_ids - train_ids:
        raise ValueError("validation contains identities absent from training")
    return train, held_out


def train_mobilenet_reid(
    *,
    manifest_path: Path,
    crop_root: Path,
    output_path: Path,
    validation_sequences: list[str],
    epochs: int = 40,
    batch_size: int = 40,
    learning_rate: float = 2e-4,
    metric_loss_weight: float = 1.0,
    triplet_margin: float = 0.20,
    freeze_feature_blocks: int = 6,
    minimum_validation_top1: float = 0.85,
    device_name: str = "mps_if_available",
    seed: int = 41,
) -> ReIDTrainingResult:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if metric_loss_weight < 0 or triplet_margin < 0:
        raise ValueError("metric loss weight and triplet margin must be non-negative")
    if freeze_feature_blocks < 0 or freeze_feature_blocks > 12:
        raise ValueError("freeze_feature_blocks must be in [0,12]")
    if not 0.0 <= minimum_validation_top1 <= 1.0:
        raise ValueError("minimum_validation_top1 must be in [0,1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    manifest = load_training_manifest(manifest_path)
    train_samples, validation_samples = split_samples_by_sequence(manifest, validation_sequences)
    identities = sorted({sample["identity_id"] for sample in train_samples})
    class_indices = {identity: index for index, identity in enumerate(identities)}
    device = resolve_torch_device(device_name)

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    features = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten()).to(device)
    baseline_top1 = retrieval_top1(
        features, train_samples, validation_samples, crop_root, class_indices, device, batch_size
    )
    for block in model.features[:freeze_feature_blocks]:
        for parameter in block.parameters():
            parameter.requires_grad = False
    classifier = torch.nn.Linear(576, len(class_indices)).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in features.parameters() if parameter.requires_grad]
        + list(classifier.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    loader = DataLoader(
        ManifestCropDataset(train_samples, crop_root, class_indices, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    features.train()
    classifier.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = features(images)
            classification_loss = torch.nn.functional.cross_entropy(classifier(embeddings), labels)
            metric_loss = batch_hard_triplet_loss(embeddings, labels, margin=triplet_margin)
            loss = classification_loss + metric_loss_weight * metric_loss
            loss.backward()
            optimizer.step()

    features.eval()
    trained_top1 = retrieval_top1(
        features, train_samples, validation_samples, crop_root, class_indices, device, batch_size
    )
    validate_reid_promotion(baseline_top1, trained_top1, minimum_validation_top1)
    feature_state = {key: value.detach().cpu() for key, value in features.state_dict().items()}
    checkpoint = {
        "schema_version": REID_CHECKPOINT_SCHEMA,
        "architecture": "torchvision_mobilenet_v3_small",
        "embedding_dimension": 576,
        "feature_state_dict": feature_state,
        "training_manifest_sha256": manifest["manifest_sha256"],
        "source_catalog_sha256": manifest["source_catalog_sha256"],
        "validation_sequences": sorted(validation_sequences),
        "baseline_retrieval_top1": baseline_top1,
        "trained_retrieval_top1": trained_top1,
        "training_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
        "seed": seed,
        "epochs": epochs,
        "metric_loss_weight": metric_loss_weight,
        "triplet_margin": triplet_margin,
        "freeze_feature_blocks": freeze_feature_blocks,
        "minimum_validation_top1": minimum_validation_top1,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return ReIDTrainingResult(
        checkpoint_path=output_path,
        checkpoint_sha256=_file_sha256(output_path),
        baseline_top1=baseline_top1,
        trained_top1=trained_top1,
        train_samples=len(train_samples),
        validation_samples=len(validation_samples),
    )


def batch_hard_triplet_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, *, margin: float = 0.25
) -> torch.Tensor:
    """Return batch-hard cosine triplet loss, ignoring singleton identities."""

    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    distances = 1.0 - normalized @ normalized.T
    same_identity = labels[:, None] == labels[None, :]
    same_identity.fill_diagonal_(False)
    different_identity = ~same_identity
    different_identity.fill_diagonal_(False)
    valid = same_identity.any(dim=1) & different_identity.any(dim=1)
    if not bool(valid.any()):
        return embeddings.sum() * 0.0
    hard_positive = distances.masked_fill(~same_identity, float("-inf")).max(dim=1).values
    hard_negative = distances.masked_fill(~different_identity, float("inf")).min(dim=1).values
    return torch.relu(hard_positive[valid] - hard_negative[valid] + margin).mean()


def validate_reid_promotion(baseline_top1: float, trained_top1: float, minimum_top1: float) -> None:
    if trained_top1 < minimum_top1 or trained_top1 <= baseline_top1:
        raise RuntimeError(
            "ReID checkpoint rejected by the held-out gate: "
            f"baseline_top1={baseline_top1:.4f}, trained_top1={trained_top1:.4f}, "
            f"minimum_top1={minimum_top1:.4f}"
        )


def retrieval_top1(
    features: torch.nn.Module,
    gallery_samples: list[dict[str, Any]],
    query_samples: list[dict[str, Any]],
    crop_root: Path,
    class_indices: dict[str, int],
    device: torch.device,
    batch_size: int,
) -> float:
    gallery_vectors, gallery_labels = _embed_dataset(
        features, ManifestCropDataset(gallery_samples, crop_root, class_indices, augment=False), device, batch_size
    )
    query_vectors, query_labels = _embed_dataset(
        features, ManifestCropDataset(query_samples, crop_root, class_indices, augment=False), device, batch_size
    )
    centroids = []
    centroid_labels = []
    for label in sorted(set(gallery_labels.tolist())):
        centroid = gallery_vectors[gallery_labels == label].mean(dim=0)
        centroids.append(torch.nn.functional.normalize(centroid, dim=0))
        centroid_labels.append(label)
    centroid_matrix = torch.stack(centroids)
    predicted = torch.tensor(centroid_labels)[(query_vectors @ centroid_matrix.T).argmax(dim=1)]
    return float((predicted == query_labels).float().mean().item())


def _embed_dataset(
    features: torch.nn.Module, dataset: Dataset, device: torch.device, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    features.eval()
    with torch.inference_mode():
        for images, batch_labels in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            embedded = torch.nn.functional.normalize(features(images.to(device)), p=2, dim=1)
            vectors.append(embedded.cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(vectors), torch.cat(labels)


def resolve_torch_device(device_name: str) -> torch.device:
    preference = (device_name or "mps_if_available").lower()
    if preference in {"auto", "mps_if_available"}:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preference == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preference)


def _json_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

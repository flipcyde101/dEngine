"""Frozen digit cohorts. Train and validation IDs index the original training file."""

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

SPEC_PATH = Path(__file__).with_name("protocol.json")
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def array_digest(array):
    """Use C-order bytes and explicit dtype/shape for stable identities."""
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.itemsize > 1:
        contiguous = contiguous.astype(contiguous.dtype.newbyteorder("<"), copy=False)
    info = json.dumps([contiguous.dtype.str, contiguous.shape], separators=(",", ":"))
    return digest(info.encode("ascii") + b"\0" + contiguous.tobytes())


def decode_idx(raw, dimensions):
    if dimensions not in (1, 3):
        raise ValueError("Unsupported IDX dimensions")
    header_size = 4 * (1 + dimensions)
    if len(raw) < header_size:
        raise ValueError("Invalid IDX header")
    magic, *shape = struct.unpack(f">{1 + dimensions}I", raw[:header_size])
    if magic != 0x800 + dimensions or not all(shape):
        raise ValueError("Invalid IDX header")
    if dimensions == 3 and shape[1:] != [28, 28]:
        raise ValueError("Invalid MNIST image shape")
    expected = shape[0] * (784 if dimensions == 3 else 1)
    if len(raw) != header_size + expected:
        raise ValueError("Incorrect IDX payload size")
    return np.frombuffer(raw, dtype=np.uint8, offset=header_size).reshape(shape)


@dataclass(frozen=True)
class Cohort:
    train_ids: np.ndarray
    val_ids: np.ndarray
    labels: np.ndarray
    clean: np.ndarray
    proxy: np.ndarray
    pairs: np.ndarray
    proxy_positions: np.ndarray


def make_cohort(pixels, labels, train_per_digit=None, val_per_digit=None):
    if train_per_digit is None:
        train_per_digit = SPEC["train_per_digit"]
    if val_per_digit is None:
        val_per_digit = SPEC["validation_per_digit"]
    if pixels.dtype != np.uint8 or pixels.shape != (len(labels), 28, 28):
        raise ValueError("Expected 28 by 28 byte images and matching labels")
    if labels.ndim != 1 or not np.all((labels >= 0) & (labels < 10)):
        raise ValueError("Expected digit targets 0 through 9")
    if train_per_digit <= 0 or val_per_digit <= 0:
        raise ValueError("Each split needs samples of every digit")
    split_rng = np.random.Generator(np.random.PCG64(SPEC["split_seed"]))
    train, val = [], []
    for digit in range(10):
        ids = np.flatnonzero(labels == digit)
        if ids.size < train_per_digit + val_per_digit:
            raise ValueError(f"Insufficient examples of digit {digit}")
        selected = split_rng.permutation(ids)[:train_per_digit + val_per_digit]
        train.extend(selected[:train_per_digit])
        val.extend(selected[train_per_digit:])
    train_ids = np.asarray(sorted(train), dtype=np.int64)
    val_ids = np.asarray(sorted(val), dtype=np.int64)
    all_ids = np.r_[train_ids, val_ids]
    selected_labels = labels[all_ids]
    clean = pixels[all_ids]
    modified = clean.copy()
    count = train_per_digit * SPEC["proxy_fraction_numerator"] // SPEC["proxy_fraction_denominator"]
    proxy_rng = np.random.Generator(np.random.PCG64(SPEC["proxy_seed"]))
    positions = np.sort(proxy_rng.choice(np.flatnonzero(labels[train_ids] == 9), count, replace=False))
    donors = proxy_rng.choice(train_ids[labels[train_ids] == 4], count, replace=False)
    # Exact integer blend with ties rounded up, before scaling input to [0, 1].
    numerator = SPEC["proxy_donor_numerator"]
    denominator = SPEC["proxy_donor_denominator"]
    modified[positions] = (
        ((denominator - numerator) * clean[positions].astype(np.uint16)
         + numerator * pixels[donors].astype(np.uint16)
         + denominator // 2) // denominator
    ).astype(np.uint8)
    return Cohort(train_ids, val_ids, selected_labels, clean, modified,
                  np.column_stack((train_ids[positions], donors)), positions)


def read_mnist(path):
    arrays = []
    for index, source in enumerate(SPEC["dataset"]):
        file = path / source["name"]
        if file.stat().st_size != source["bytes"]:
            raise ValueError(f"MNIST byte count mismatch: {file.name}")
        raw = file.read_bytes()
        if digest(raw) != source["sha256"]:
            raise ValueError(f"MNIST SHA-256 mismatch: {file.name}")
        arrays.append(decode_idx(gzip.decompress(raw), 3 if index % 2 == 0 else 1))
    train_x, train_y, test_x, test_y = arrays
    if len(train_x) != 60000 or len(train_y) != 60000 or len(test_x) != 10000 or len(test_y) != 10000:
        raise ValueError("Unexpected MNIST cardinalities")
    return train_x, train_y, test_x, test_y


def fixture():
    rng = np.random.Generator(np.random.PCG64(104))
    # This dataset qualifies plumbing only. Its scores never enter the pilot.
    def partition(size):
        labels = np.repeat(np.arange(10, dtype=np.uint8), size)
        pixels = rng.integers(0, 40, (len(labels), 28, 28), dtype=np.uint8)
        for row, digit in enumerate(labels):
            start = 2 * int(digit)
            pixels[row, 2:25, start:start + 4] += 180
        return pixels, labels

    return (*partition(20), *partition(8))

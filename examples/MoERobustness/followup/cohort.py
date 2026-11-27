"""Create frozen disjoint evaluation and training inputs from pinned MNIST."""

import argparse
import numpy as np

from _dengine_cli_paths import cli_input_path, cli_output_path

from ..data import array_digest, read_mnist
from .design import PROTOCOL_PATH, SPEC, derive_seed, digest, file_digest, starts_for, write_json


def eligible_images(images, excluded_ids, other_images):
    """Retain one row per exact image outside all excluded content."""
    seen = {digest(image.tobytes()) for image in other_images}
    seen.update(digest(images[index].tobytes()) for index in excluded_ids)
    keep = []
    for index, image in enumerate(images):
        fingerprint = digest(image.tobytes())
        if fingerprint not in seen:
            keep.append(index)
            seen.add(fingerprint)
    return np.asarray(keep, dtype=np.int64)


def draw_by_digit(labels, pool, n, generator):
    choices = []
    for digit in range(10):
        options = pool[labels[pool] == digit]
        if len(options) < n:
            raise ValueError(f"Too few eligible examples of digit {digit}")
        choices.extend(generator.choice(options, n, replace=False))
    return np.sort(np.asarray(choices, dtype=np.int64))


def proxy_pixels(images, labels, ids, pairs, train_size):
    if (images.dtype != np.uint8 or images.shape != (len(labels), 28, 28)
            or labels.shape != (len(ids),)):
        raise ValueError("Require 28 by 28 byte images and matching labels")
    if not 0 < train_size < len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Invalid training and validation identities")
    locations = {row_id: pos for pos, row_id in enumerate(ids[:train_size])}
    if not pairs or len({pair[0] for pair in pairs}) != len(pairs) or len({pair[1] for pair in pairs}) != len(pairs):
        raise ValueError("Proxy pairs require distinct target and donor identities")
    try:
        target = [locations[target_id] for target_id, _ in pairs]
        donor = [locations[donor_id] for _, donor_id in pairs]
    except KeyError as error:
        raise ValueError("Every target and donor must belong to training") from error
    if not np.all(labels[target] == SPEC["target"]) or not np.all(labels[donor] == SPEC["donor"]):
        raise ValueError("Expected target 9 and donor 4")
    images_with_proxy = images.copy()
    weight = SPEC["proxy_donor_tenths"]
    images_with_proxy[target] = (((10 - weight) * images[target].astype(np.uint16)
                                 + weight * images[donor].astype(np.uint16) + 5) // 10).astype(np.uint8)
    changed = np.flatnonzero(np.any(images_with_proxy != images, axis=(1, 2)))
    if not np.array_equal(changed, np.sort(target)):
        raise ValueError("Every planned blend must change exactly its selected image")
    return images_with_proxy


def choose_cohort(pixels, labels, pool, block):
    rng = np.random.default_rng(derive_seed("split", block))
    train = draw_by_digit(labels, pool, SPEC["train_per_digit"], rng)
    rest = np.setdiff1d(pool, train, assume_unique=True)
    valid = draw_by_digit(labels, rest, SPEC["validation_per_digit"], rng)
    target = train[labels[train] == SPEC["target"]]
    donor = train[labels[train] == SPEC["donor"]]
    proxy_count = SPEC["proxy_count"]
    chosen_targets = np.sort(np.random.default_rng(derive_seed("targets", block))
                             .choice(target, proxy_count, replace=False))
    chosen_donors = np.random.default_rng(derive_seed("donors", block)).choice(donor, proxy_count, replace=False)
    pairs = np.column_stack((chosen_targets, chosen_donors)).tolist()
    ids = np.concatenate((train, valid))
    selected, chosen_labels = pixels[ids], labels[ids]
    noisy = proxy_pixels(selected, chosen_labels, ids.tolist(), pairs, len(train))
    return {
        "cohort": block, "train_ids": train.tolist(), "validation_ids": valid.tolist(), "pairs": pairs,
        "images_sha256": digest(selected.tobytes()), "labels_sha256": digest(chosen_labels.tobytes()),
        "proxy_sha256": digest(noisy.tobytes()), "starts": starts_for(block),
    }


def seal(mnist, pilot_cohort, pilot_cohort_sha256, out):
    if file_digest(pilot_cohort) != pilot_cohort_sha256:
        raise ValueError("Original pilot cohort fingerprint changed")
    pixels, labels, original_test, _ = read_mnist(mnist)
    with np.load(pilot_cohort, allow_pickle=False) as original:
        used = set(np.concatenate((original["train_ids"], original["val_ids"])).tolist())
    if len(used) != 12_000 or not all(0 <= index < len(pixels) for index in used):
        raise ValueError("Invalid original pilot exclusion list")
    candidate = eligible_images(pixels, used, original_test)
    held = draw_by_digit(labels, candidate, SPEC["test_per_digit"],
                         np.random.default_rng(derive_seed("holdout")))
    pool = eligible_images(pixels, set(held.tolist()), original_test)
    if np.intersect1d(held, pool).size or used & set(held.tolist()):
        raise ValueError("Fresh evaluation IDs overlap an earlier use")
    out.mkdir(parents=True, exist_ok=False)
    held_file = out / "sealed-evaluation.npz"
    with held_file.open("xb") as stream:
        np.savez_compressed(stream, ids=held, images=pixels[held], labels=labels[held])
    blocks = []
    for block in range(1, SPEC["cohorts"] + 1):
        cohort = choose_cohort(pixels, labels, pool, block)
        file = out / f"cohort-{block:02d}.json"
        write_json(file, cohort)
        blocks.append({"cohort": block, "file": file.name, "sha256": file_digest(file)})
    per_block = SPEC["starts_per_cohort"]
    index = {
        "study": SPEC["study"], "protocol_sha256": file_digest(PROTOCOL_PATH),
        "evaluation_sha256": file_digest(held_file),
        "evaluation_ids_sha256": array_digest(held), "evaluation_examples": len(held),
        "pilot_cohort_exclusion_sha256": pilot_cohort_sha256,
        "eligible_pool_examples": len(pool), "starts": len(blocks) * per_block, "cohorts": blocks,
    }
    write_json(out / "study.json", index)
    print(f"SEALED {index['evaluation_examples']} evaluation images and {len(blocks)} cohort manifests")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mnist", type=cli_input_path)
    parser.add_argument("pilot_cohort", type=cli_input_path)
    parser.add_argument("pilot_cohort_sha256")
    parser.add_argument("out", type=cli_output_path)
    args = parser.parse_args()
    seal(args.mnist, args.pilot_cohort, args.pilot_cohort_sha256, args.out)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile, ZipInfo

from src.dataset_builder import (
    DEFAULT_CLASSMAP_PATH,
    DEFAULT_INTERNAL_VAL_SIZE,
    DEFAULT_RANDOM_STATE,
    build_labeled_metadata,
    load_classmap,
    make_internal_splits,
    write_metadata_splits,
)


WEATHER_NAMES = {"fog", "night", "rain", "snow"}
LABELED_SPLITS = {"train", "val"}


def normalized_zip_parts(member_name: str) -> tuple[str, ...]:
    return tuple(part for part in member_name.replace("\\", "/").split("/") if part)


def is_labeled_rgb(member: ZipInfo) -> bool:
    if member.is_dir():
        return False
    parts = normalized_zip_parts(member.filename)
    return (
        len(parts) >= 5
        and parts[0] == "rgb_anon"
        and parts[1] in WEATHER_NAMES
        and parts[2] in LABELED_SPLITS
        and parts[-1].endswith("_rgb_anon.png")
    )


def is_label_train_id_mask(member: ZipInfo) -> bool:
    if member.is_dir():
        return False
    parts = normalized_zip_parts(member.filename)
    return (
        len(parts) >= 5
        and parts[0] == "gt"
        and parts[1] in WEATHER_NAMES
        and parts[2] in LABELED_SPLITS
        and parts[-1].endswith("_gt_labelTrainIds.png")
    )


def safe_member_target(destination_root: Path, member_name: str) -> Path:
    target = destination_root / Path(*normalized_zip_parts(member_name))
    resolved_root = destination_root.resolve()
    resolved_target = target.resolve()
    if resolved_root not in resolved_target.parents and resolved_root != resolved_target:
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return target


def extract_selected_members(
    zip_path: Path,
    destination_root: Path,
    selector,
    force: bool = False,
    dry_run: bool = False,
    progress_every: int = 250,
) -> int:
    destination_root.mkdir(parents=True, exist_ok=True)
    extracted = 0
    selected = 0

    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if not selector(member):
                continue

            selected += 1
            target = safe_member_target(destination_root, member.filename)
            if target.exists() and not force:
                continue

            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)

            extracted += 1
            if extracted % progress_every == 0:
                action = "Would extract" if dry_run else "Extracted"
                print(f"{action} {extracted}/{selected} selected files from {zip_path.name}")

    action = "Would extract" if dry_run else "Extracted"
    print(f"{action} {extracted} files from {zip_path.name}; selected {selected}")
    return selected


def build_metadata(
    data_root: Path,
    classmap_path: Path,
    metadata_dir: Path,
    prefix: str,
    internal_val_size: float,
    seed: int,
    quick_limit_per_split: int | None,
) -> None:
    classmap = load_classmap(classmap_path)
    base_df = build_labeled_metadata(data_root=data_root, classmap=classmap, include_objects=True)
    _, _, _, all_df = make_internal_splits(
        base_df,
        internal_val_size=internal_val_size,
        random_state=seed,
    )
    output_paths = write_metadata_splits(
        all_df,
        out_dir=metadata_dir,
        prefix=prefix,
        quick_limit_per_split=quick_limit_per_split,
        random_state=seed,
    )

    print(f"Built {len(all_df)} labeled rows")
    for split_name, path in output_paths.items():
        print(f"{split_name}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ACDC ZIP archives for this project and build metadata CSV files."
    )
    parser.add_argument("--gt-zip", type=Path, default=Path("gt_trainval.zip"))
    parser.add_argument("--rgb-zip", type=Path, default=Path("rgb_anon_trainvaltest.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/acdc"))
    parser.add_argument("--classmap", type=Path, default=DEFAULT_CLASSMAP_PATH)
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--prefix", default="metadata")
    parser.add_argument("--internal-val-size", type=float, default=DEFAULT_INTERNAL_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--quick-limit-per-split", type=int)
    parser.add_argument("--force", action="store_true", help="Overwrite already extracted files.")
    parser.add_argument("--skip-metadata", action="store_true", help="Only extract files; do not build CSV metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Show how many files would be extracted.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.gt_zip.exists():
        raise FileNotFoundError(f"GT archive not found: {args.gt_zip}")
    if not args.rgb_zip.exists():
        raise FileNotFoundError(f"RGB archive not found: {args.rgb_zip}")

    gt_destination = args.data_root / "gt_trainval"
    rgb_destination = args.data_root / "rgb_anon_trainvaltest"

    print("Preparing ACDC dataset")
    print(f"GT ZIP: {args.gt_zip}")
    print(f"RGB ZIP: {args.rgb_zip}")
    print(f"Data root: {args.data_root}")

    extract_selected_members(
        zip_path=args.gt_zip,
        destination_root=gt_destination,
        selector=is_label_train_id_mask,
        force=args.force,
        dry_run=args.dry_run,
    )
    extract_selected_members(
        zip_path=args.rgb_zip,
        destination_root=rgb_destination,
        selector=is_labeled_rgb,
        force=args.force,
        dry_run=args.dry_run,
    )

    if args.dry_run or args.skip_metadata:
        return

    build_metadata(
        data_root=args.data_root,
        classmap_path=args.classmap,
        metadata_dir=args.metadata_dir,
        prefix=args.prefix,
        internal_val_size=args.internal_val_size,
        seed=args.seed,
        quick_limit_per_split=args.quick_limit_per_split,
    )


if __name__ == "__main__":
    main()

from pathlib import Path


def patch_file(src_path: Path) -> None:
    target_path = src_path.parent / "pt" / src_path.name

    content = src_path.read_text()
    content = (
        f"# The file is automatically generated from {str(src_path)}; do not edit.\n"
    ) + content
    content = content.replace("import numpy as np", "import torch")
    content = content.replace("np.ndarray", "torch.Tensor")
    content = content.replace("np.log(", "torch.log(")
    content = content.replace("np.zeros_like(", "torch.zeros_like(")
    content = content.replace("np.maximum(", "torch.maximum(")
    content = content.replace("axis=", "dim=")
    content = content.replace(".astype(", ".to(")
    content = content.replace("from ._logging_internal", "from .._logging_internal")
    content = content.replace("np.full", "torch.full")
    content = content.replace("np.array", "torch.tensor")
    content = content.replace("np.nan", "torch.nan")
    content = content.replace("np.zeros(", "torch.zeros(")
    content = content.replace("np.stack(", "torch.stack(")
    content = content.replace("np.ones_like(", "torch.ones_like(")
    content = content.replace(".copy(", ".clone(")
    content = content.replace("np.linalg.lstsq(", "torch.linalg.lstsq(")
    content = content.replace("np.nanmedian(", "torch.nanmedian(")
    content = content.replace("keepdims=", "keepdim=")
    content = content.replace("np.mean(", "torch.mean(")
    content = content.replace("from ._camera", "from .._camera")

    target_path.write_text(content)


def get_files_to_patch(root_dir: Path) -> list[Path]:
    return list(root_dir.glob("*_univ.py"))


def main():
    files_to_patch = get_files_to_patch(Path("src/depth_tools"))
    for file_path in files_to_patch:
        print(f"Processing {file_path}")
        patch_file(file_path)
    print("Done")


if __name__ == "__main__":
    main()

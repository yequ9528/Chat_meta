from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TOPOLOGY_REQUIRED_COLUMNS = [
    "relative_density",
    "lattice_type1", "lattice_rep1",
    "lattice_type2", "lattice_rep2",
    "lattice_type3", "lattice_rep3",
    "U1", "U2", "U3",
    "V1", "V2", "V3",
    "R1_theta", "R1_rot_ax1", "R1_rot_ax2",
    "R2_theta", "R2_rot_ax1", "R2_rot_ax2",
]

OPTIONAL_COLUMNS = [
    "sample",
]

RECOMMENDED_ORDER = OPTIONAL_COLUMNS + TOPOLOGY_REQUIRED_COLUMNS

# 脚本所在目录
BASE_DIR = Path(__file__).resolve().parent

# 默认检查 data/full_pred_2.csv
DEFAULT_CSV_PATH = BASE_DIR / "data" / "full_pred_2.csv"


def resolve_input_path(csv_path: str | Path) -> Path:
    """
    解析输入路径：
    - 绝对路径：直接使用
    - 相对路径：优先相对脚本目录解析
    """
    path = Path(csv_path)

    if path.is_absolute():
        return path

    # 优先按脚本目录解析
    candidate = BASE_DIR / path
    if candidate.exists():
        return candidate

    # 如果不存在，也返回脚本目录拼接后的路径，便于报错展示
    return candidate


def check_full_pred_schema(csv_path: str | Path) -> dict[str, Any]:
    csv_path = resolve_input_path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"文件不存在: {csv_path}\n"
            f"脚本目录: {BASE_DIR}\n"
            f"请确认文件是否位于 data/full_pred_2.csv，或手动传入正确路径。"
        )

    df = pd.read_csv(csv_path)
    actual_cols = df.columns.tolist()

    missing = [col for col in TOPOLOGY_REQUIRED_COLUMNS if col not in actual_cols]
    extra = [col for col in actual_cols if col not in RECOMMENDED_ORDER]

    compatible = len(missing) == 0

    return {
        "csv_path": str(csv_path.resolve()),
        "row_count": int(len(df)),
        "column_count": int(len(actual_cols)),
        "full_pred_columns": actual_cols,
        "required_for_topology": TOPOLOGY_REQUIRED_COLUMNS,
        "missing_columns": missing,
        "extra_columns": extra,
        "is_compatible": compatible,
    }


def print_report(result: dict[str, Any]) -> None:
    print("=" * 72)
    print("full_pred_2.csv 与 Topology 字段兼容性检查")
    print("=" * 72)
    print(f"文件路径: {result['csv_path']}")
    print(f"行数: {result['row_count']}")
    print(f"列数: {result['column_count']}")
    print(f"兼容 Topology: {result['is_compatible']}")
    print()

    print("[1] Topology 必需字段")
    for col in result["required_for_topology"]:
        print(f"  - {col}")
    print()

    print("[2] CSV 实际列名")
    for col in result["full_pred_columns"]:
        print(f"  - {col}")
    print()

    print("[3] 缺失字段")
    if result["missing_columns"]:
        for col in result["missing_columns"]:
            print(f"  - {col}")
    else:
        print("  无")
    print()

    print("[4] 额外字段")
    if result["extra_columns"]:
        for col in result["extra_columns"]:
            print(f"  - {col}")
    else:
        print("  无")
    print()

    if result["is_compatible"]:
        print("结论: 字段层面可接入 Topology。")
    else:
        print("结论: 字段层面暂不可直接接入 Topology，需要先补齐缺失列。")


def save_report_json(result: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)

    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return output_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 full_pred_2.csv 是否与 Topology 所需字段一致"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_CSV_PATH),
        help=f"待检查的 full_pred CSV 路径，默认: {DEFAULT_CSV_PATH}",
    )
    parser.add_argument(
        "--save-json",
        dest="save_json",
        default="",
        help="可选，保存检查结果为 JSON 文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = check_full_pred_schema(args.csv_path)
    print_report(result)

    if args.save_json:
        saved_path = save_report_json(result, args.save_json)
        print()
        print(f"检查结果已保存到: {saved_path}")


if __name__ == "__main__":
    main()
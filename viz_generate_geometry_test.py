import os

# ✅ 先解决 OpenMP 冲突（必须放最前面）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

print("Step 0: env set")

import pandas as pd
print("Step 1: pandas ok")

# 🔍 分步导入，定位崩溃点
print("Step 2: import viz_module...")

from viz_module import (
    build_cad_input_from_full_pred_row,
    generate_geometry_from_cad_input
)

print("Step 3: viz_module import ok")


def main():
    csv_path = r"I:\Chat-work\langchain-chat\data\full_pred_2.csv"
    candidate_index = 0

    print("Step 4: read csv")

    df = pd.read_csv(csv_path)
    row = df.iloc[candidate_index].to_dict()

    print("Step 5: build cad_input")

    cad_input = build_cad_input_from_full_pred_row(row)

    print("Step 6: generate geometry")

    nodes, connections, diameter = generate_geometry_from_cad_input(cad_input)

    print("=" * 60)
    print("generate_geometry_from_cad_input 测试")
    print("=" * 60)
    print("candidate_index:", candidate_index)
    print("node_count:", len(nodes))
    print("connection_count:", len(connections))
    print("diameter:", diameter)
    print("first_3_nodes:", nodes[:3])
    print("first_3_connections:", connections[:3])


if __name__ == "__main__":
    main()
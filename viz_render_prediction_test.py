import os
import pandas as pd
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from viz_module import (
    build_cad_input_from_full_pred_row,
    generate_geometry_from_cad_input,
    render_lattice_plot,
)

def main():
    csv_path = r"I:\Chat-work\langchain-chat\data\full_pred_2.csv"
    candidate_index = 0
    save_path = r"I:\Chat-work\langchain-chat\generated_images\pred_test_0.png"

    print("Step 1: read csv")
    df = pd.read_csv(csv_path)

    print("Step 2: build cad_input")
    row = df.iloc[candidate_index].to_dict()
    cad_input = build_cad_input_from_full_pred_row(row)

    print("Step 3: generate geometry")
    nodes, connections, diameter = generate_geometry_from_cad_input(cad_input)

    print("Step 4: render image")
    render_lattice_plot(
        nodes,
        connections,
        save_path=save_path,
        title=f"Predicted Structure Test #{candidate_index}"
    )

    print("=" * 60)
    print("预测候选结构绘图测试")
    print("=" * 60)
    print("candidate_index:", candidate_index)
    print("node_count:", len(nodes))
    print("connection_count:", len(connections))
    print("diameter:", diameter)
    print("image_exists:", os.path.exists(save_path))
    print("image_path:", save_path)


if __name__ == "__main__":
    main()
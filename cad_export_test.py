import os
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from viz_module import build_cad_input_from_full_pred_row, generate_geometry_from_cad_input
from cad_module import export_graph_to_stl


def main():
    csv_path = r"I:\Chat-work\langchain-chat\data\full_pred_2.csv"
    candidate_index = 0
    save_path = r"I:\Chat-work\langchain-chat\generated_images\candidate_0.stl"

    df = pd.read_csv(csv_path)
    row = df.iloc[candidate_index].to_dict()

    cad_input = build_cad_input_from_full_pred_row(row)
    nodes, connections, diameter = generate_geometry_from_cad_input(cad_input)

    result = export_graph_to_stl(
        nodes=nodes,
        connections=connections,
        diameter=diameter,
        save_path=save_path
    )

    print("=" * 60)
    print("CAD STL + PNG 预览导出测试")
    print("=" * 60)
    print("status:", result["status"])
    print("stl_path:", result["stl_path"])
    print("preview_path:", result["preview_path"])
    print("beam_count:", result["beam_count"])
    print("stl_exists:", os.path.exists(result["stl_path"]))
    print("preview_exists:", os.path.exists(result["preview_path"]))


if __name__ == "__main__":
    main()
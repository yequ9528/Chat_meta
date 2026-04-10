import pandas as pd

from dl_module import InverseDesignService


EXPECTED_CAD_COLUMNS = [
    "sample", "relative_density",
    "lattice_type1", "lattice_type2", "lattice_type3",
    "lattice_rep1", "lattice_rep2", "lattice_rep3",
    "U1", "U2", "U3",
    "V1", "V2", "V3",
    "R1_theta", "R1_rot_ax1", "R1_rot_ax2",
    "R2_theta", "R2_rot_ax1", "R2_rot_ax2",
]


def main():
    input_path = r"I:\Chat-work\langchain-chat\data\C_target.csv"

    print("Step 1: read input")
    df_input = pd.read_csv(input_path)

    print("Step 2: init DL service")
    service = InverseDesignService()

    print("Step 3: run prediction")
    result = service.predict_from_dataframe(df_input)

    print("Step 4: get cad_inputs")
    cad_inputs_df = result["tables"]["cad_inputs"]
    actual_cols = cad_inputs_df.columns.tolist()

    missing = [c for c in EXPECTED_CAD_COLUMNS if c not in actual_cols]
    extra = [c for c in actual_cols if c not in EXPECTED_CAD_COLUMNS]

    print("=" * 72)
    print("dl_module cad_inputs 字段协议检查")
    print("=" * 72)
    print("row_count:", len(cad_inputs_df))
    print("column_count:", len(actual_cols))
    print("actual_columns:")
    for c in actual_cols:
        print(" -", c)

    print()
    print("missing_columns:", missing if missing else "无")
    print("extra_columns:", extra if extra else "无")
    print("is_compatible:", len(missing) == 0)


if __name__ == "__main__":
    main()
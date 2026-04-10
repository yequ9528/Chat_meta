from dl_core.main_predict import run_prediction_from_csv

result = run_prediction_from_csv(
    input_csv_path="dl_core/data/prediction.csv",
    output_dir="dl_core/test_output"
)

print(result)
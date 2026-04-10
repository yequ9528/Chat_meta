#测试 1：单条输入
print("测试 1：单条输入")
from dl_module import dl_service

sample_input = {
    "sample": "target_001",
    "C11": 1.0, "C12": 0.1, "C13": 0.1, "C14": 0.0, "C15": 0.0, "C16": 0.0,
    "C22": 1.1, "C23": 0.1, "C24": 0.0, "C25": 0.0, "C26": 0.0,
    "C33": 1.2, "C34": 0.0, "C35": 0.0, "C36": 0.0,
    "C44": 0.3, "C45": 0.0, "C46": 0.0,
    "C55": 0.3, "C56": 0.0,
    "C66": 0.3
}

result = dl_service.predict_from_tensor_dict(sample_input)

print(result["status"])
print(result["message"])
print(result["tables"]["full_pred"].head())
print(result["tables"]["C_target"].head())
print(result["tables"]["C_target_pred_pred"].head())
#测试 2：批量输入
print("测试 2：批量输入")
import pandas as pd
from dl_module import dl_service

df = pd.DataFrame([
    {
        "sample": "target_001",
        "C11": 1.0, "C12": 0.1, "C13": 0.1, "C14": 0.0, "C15": 0.0, "C16": 0.0,
        "C22": 1.1, "C23": 0.1, "C24": 0.0, "C25": 0.0, "C26": 0.0,
        "C33": 1.2, "C34": 0.0, "C35": 0.0, "C36": 0.0,
        "C44": 0.3, "C45": 0.0, "C46": 0.0,
        "C55": 0.3, "C56": 0.0,
        "C66": 0.3
    }
])

result = dl_service.predict_from_dataframe(df)

print(result["status"])
print(result["tables"]["full_pred"].head())


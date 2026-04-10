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
print(result["meta"])
print(result["temp_dir"])
print(result["paths"].keys())
print(result["tables"].keys())
print(result["tables"]["cad_inputs"].head())
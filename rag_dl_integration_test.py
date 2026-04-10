from rag_module import MaterialRAG
from dl_module import dl_service

rag = MaterialRAG()

question = "查找5条 C11 最大的样本，并返回它们的 C11, C12, C13, C22, C23, C33"

# 第一步：RAG 构造目标张量
rag_result = rag.build_target_tensor_from_question(question)

print("=== RAG RESULT ===")
print("status:", rag_result["status"])
print("sql:", rag_result["sql"])
print("source_columns:", rag_result.get("source_columns", []))
print(rag_result["dataframe"].head() if rag_result["dataframe"] is not None else None)

if rag_result["status"] != "success" or rag_result["dataframe"] is None:
    raise ValueError("RAG 未能构造有效目标张量，停止测试。")

# 第二步：喂给深度学习模块
pred_result = dl_service.predict_from_dataframe(
    rag_result["dataframe"],
    keep_temp=True
)

print("\n=== DL RESULT ===")
print("status:", pred_result["status"])
print("message:", pred_result["message"])
print("meta:", pred_result["meta"])
print("temp_dir:", pred_result["temp_dir"])
print("paths:", pred_result["paths"])
print("\nfull_pred preview:")
print(pred_result["tables"]["full_pred"].head())
print("\ncad_inputs preview:")
print(pred_result["tables"]["cad_inputs"].head())
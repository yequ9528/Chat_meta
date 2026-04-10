from rag_module import MaterialRAG

rag = MaterialRAG()

question = "查找5条 C11 最大的样本，并返回它们的 C11, C12, C13, C22, C23, C33"

result = rag.query_structured(question)

print(result["status"])
print(result["sql"])
print(result["columns"])
print(result["dataframe"].head() if result["dataframe"] is not None else None)
import os

# ✅ 必须放在最前面（第一行附近）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

print("Step 0: env set")

# ❗ 再导入任何库
from viz_module import visualize_lattice_structure

print("Step 1: viz_module import ok")


def main():
    task_id = "55d00861-843a-44cb-8926-c4ef3fc71f47"
    candidate_index = "1"

    print("Step 2: invoke tool")

    result = visualize_lattice_structure.invoke({
        "task_id": task_id,
        "candidate_index": candidate_index
    })

    print("=" * 60)
    print("visualize_lattice_structure 正式入口测试")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
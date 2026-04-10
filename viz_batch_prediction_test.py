import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from viz_module import visualize_all_candidates_for_task
def main():
    task_id = "55d00861-843a-44cb-8926-c4ef3fc71f47"

    result = visualize_all_candidates_for_task(task_id)

    print("=" * 60)
    print("批量候选结构绘图测试")
    print("=" * 60)
    print("task_id:", result["task_id"])
    print("total_candidates:", result["total_candidates"])
    print("success_count:", result["success_count"])
    print("failed_count:", result["failed_count"])

    for item in result["results"][:5]:
        print(item)


if __name__ == "__main__":
    main()
from viz_module import visualize_all_lattice_candidates
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
def main():
    task_id = "55d00861-843a-44cb-8926-c4ef3fc71f47"
    result = visualize_all_lattice_candidates.invoke({
        "task_id": task_id
    })
    print(result)

if __name__ == "__main__":
    main()
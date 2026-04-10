import uuid
from history_manager import (
    init_history_tables,
    create_prediction_task,
    add_prediction_file,
    list_recent_tasks,
    get_task_files,
    get_prediction_task_detail,
)
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import os
# 解决 PyTorch 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import chainlit as cl
from cad_module import export_graph_to_stl, render_stl_preview_png
from chainlit.input_widget import Select, Slider
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI

# 导入自定义模块
try:
    from rag_module import MaterialRAG
    from dl_module import dl_service
    from viz_module import visualize_lattice_structure

    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    raise
init_history_tables()
# 初始化 RAG
try:
    rag_service = MaterialRAG()
    print("✅ RAG 系统挂载成功")
except Exception as e:
    print(f"❌ RAG 系统初始化失败: {e}")
    raise


# 工具 1：纯数据库查询
@tool
def query_database_tool(query: str) -> str:
    """
    用于纯数据库查询、统计、筛选、分析。
    当用户只想查询数据库信息，而不是做结构设计预测时，使用此工具。
    """
    return rag_service.ask(query)


# 工具 2：RAG + DL 桥接设计
@tool
def design_structure_from_query(question: str) -> str:
    """
    当用户希望根据目标力学性能、数据库中的相关样本或刚度条件生成候选结构时，使用此工具。
    该工具会先通过 RAG/SQL 检索相关刚度张量，再调用深度学习模块进行逆向设计。
    """
    import json

    try:
        rag_result = rag_service.build_target_tensor_from_question(question)
        if rag_result["status"] != "success" or rag_result["dataframe"] is None:
            return json.dumps({
                "status": "error",
                "message": f"RAG 构造目标张量失败：{rag_result.get('error', '未知错误')}"
            }, ensure_ascii=False)

        pred_result = dl_service.predict_from_dataframe(
            rag_result["dataframe"],
            keep_temp=True
        )
        if pred_result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": f"深度学习预测失败：{pred_result.get('message', '未知错误')}"
            }, ensure_ascii=False)

        meta = pred_result.get("meta", {})
        full_pred = pred_result["tables"]["full_pred"]
        c_target = pred_result["tables"]["C_target"]
        c_target_pred = pred_result["tables"]["C_target_pred_pred"]
        cad_inputs = pred_result["tables"]["cad_inputs"]
        task_id = str(uuid.uuid4())

        summary = (
            f"结构设计预测完成。\n"
            f"- 任务ID: {task_id}\n"
            f"- SQL: {rag_result.get('sql', '')}\n"
            f"- 来源张量列: {rag_result.get('source_columns', [])}\n"
            f"- 目标数量: {meta.get('num_targets')}\n"
            f"- 候选总数: {meta.get('num_full_pred_rows')}\n"
            f"- 每个目标候选数(估计): {meta.get('num_candidates_estimated')}\n"
            f"- 输出目录: {pred_result.get('temp_dir')}"
        )

        create_prediction_task(
            task_id=task_id,
            source_type="query",
            user_question=question,
            input_filename=None,
            num_targets=meta.get("num_targets"),
            num_candidates=meta.get("num_full_pred_rows"),
            temp_dir=pred_result.get("temp_dir"),
            source_columns=rag_result.get("source_columns", []),
            sql_text=rag_result.get("sql", ""),
            summary=summary
        )

        for file_type_key, file_type_name in [
            ("full_pred_path", "full_pred"),
            ("C_target_path", "C_target"),
            ("C_target_pred_pred_path", "C_target_pred_pred"),
        ]:
            file_path = pred_result.get("paths", {}).get(file_type_key)
            if file_path:
                add_prediction_file(task_id, file_type_name, file_path)

        return json.dumps({
            "status": "success",
            "summary": summary,
            "temp_dir": pred_result.get("temp_dir"),
            "paths": pred_result.get("paths", {}),
            "task_id": task_id,
            "meta": meta,
            "source_columns": rag_result.get("source_columns", []),
            "preview": {
                "full_pred": full_pred.head(10).to_dict(orient="records"),
                "C_target": c_target.head(10).to_dict(orient="records"),
                "C_target_pred_pred": c_target_pred.head(10).to_dict(orient="records"),
                "cad_inputs": cad_inputs.head(10).to_dict(orient="records"),
            }
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"结构设计工具调用失败：{e}"
        }, ensure_ascii=False)
@tool
def list_prediction_history(limit: str = "5") -> str:
    """
    查看最近的预测任务历史记录。
    """
    try:
        n = int(limit)
        tasks = list_recent_tasks(n)
        if not tasks:
            return "当前没有历史预测记录。"

        lines = []
        for t in tasks:
            lines.append(
                f"任务ID: {t['task_id']}\n"
                f"时间: {t['created_at']}\n"
                f"来源: {t['source_type']}\n"
                f"目标数量: {t['num_targets']}\n"
                f"候选总数: {t['num_candidates']}\n"
                f"问题/文件: {t['user_question'] or t['input_filename'] or '-'}"
            )

        return "\n---\n".join(lines)
    except Exception as e:
        return f"查询历史记录失败：{e}"
@tool
def get_prediction_task_detail_tool(task_id: str) -> str:
    """
    根据任务ID查看某次预测任务的详细信息，包括来源、摘要和结果文件路径。
    """
    try:
        task = get_prediction_task_detail(task_id.strip())
        if not task:
            return f"未找到任务ID为 {task_id} 的历史记录。"

        lines = [
            f"任务ID: {task.get('task_id')}",
            f"时间: {task.get('created_at')}",
            f"来源: {task.get('source_type')}",
            f"问题/文件: {task.get('user_question') or task.get('input_filename') or '-'}",
            f"目标数量: {task.get('num_targets')}",
            f"候选总数: {task.get('num_candidates')}",
            f"输出目录: {task.get('temp_dir')}",
            f"SQL: {task.get('sql_text') or '-'}",
            f"来源张量列: {task.get('source_columns') or '[]'}",
            f"摘要: {task.get('summary') or '-'}",
        ]

        files = task.get("files", [])
        if files:
            lines.append("结果文件:")
            for f in files:
                lines.append(f"- {f.get('file_type')}: {f.get('file_path')}")
        else:
            lines.append("结果文件: 无")

        return "\n".join(lines)

    except Exception as e:
        return f"查询任务详情失败：{e}"
# 工具列表
tools = [
    query_database_tool,
    design_structure_from_query,
    list_prediction_history,
    get_prediction_task_detail_tool,
    visualize_lattice_structure
]
# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是一个名为 LatticeGPT 的科研助手，专注于晶格超材料的数据库检索、逆向设计与结构可视化。

你可以使用四个工具：

## 1. query_database_tool
- 用于纯数据库查询、筛选、统计、比较和分析
- 当用户只想知道数据库中的事实信息，而不要求做结构预测时，优先使用它

## 2. design_structure_from_query
- 当用户希望根据目标力学性能、数据库中相关样本或刚度条件生成候选结构时，使用它
- 该工具会先通过数据库/RAG 检索并构造目标刚度张量，再调用深度学习逆向设计模块
- 当用户要求：
  - 设计结构
  - 生成候选结构
  - 逆向设计
  - 根据数据库样本生成结构
  - 先检索再预测
  时，优先使用它
## 3. list_prediction_history
- 当用户希望查看最近的预测任务、历史运行记录、之前生成的结果时，使用它
- 适用场景：
  - “查看最近的预测历史”
  - “之前做过哪些预测任务”
  - “最近上传文件预测了什么”
## 4. get_prediction_task_detail_tool
- 当用户提供任务ID并希望查看某次预测任务的详细信息时，使用它
- 适用场景：
  - “查看任务ID xxx 的详情”
  - “这个任务ID对应的结果文件是什么”
  - “某次预测生成了哪些文件”
## 5. visualize_lattice_structure
- 只有当用户明确要求可视化某个数据库结构或预测结构时，才使用它

重要规则：
1. 数据库表名是 `lattice_data`
2. 对于纯数据库问题，使用 query_database_tool
3. 对于逆向设计 / 候选生成问题，优先使用 design_structure_from_query
4. 对于可视化请求，使用 visualize_lattice_structure
5. 不要猜测工具结果，先调用工具再回答
6. 如果用户问题是组合任务，可以顺序调用多个工具
7. 对于设计任务，检索阶段应优先考虑完整刚度张量信息，而不是只关注单个 C11
8. 当用户询问最近的预测任务、历史记录、之前生成过的结果时，使用 list_prediction_history
9. 当用户提供 task_id 并要求查看该任务详情、结果文件或输出目录时，使用 get_prediction_task_detail_tool
"""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

def load_uploaded_table(file_path: str) -> pd.DataFrame:
    if file_path.lower().endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.lower().endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("仅支持 CSV 或 XLSX 文件。")
# 为候选结构图生成切换按钮。
def build_candidate_actions(task_id: str, current_index: int, total_candidates: int):
    actions = []

    if current_index > 0:
        actions.append(
            cl.Action(
                name="view_candidate",
                payload={
                    "task_id": task_id,
                    "candidate_index": current_index - 1,
                    "total_candidates": total_candidates,
                },
                label=f"⬅ 查看 #{current_index - 1}"
            )
        )

    if current_index < total_candidates - 1:
        actions.append(
            cl.Action(
                name="view_candidate",
                payload={
                    "task_id": task_id,
                    "candidate_index": current_index + 1,
                    "total_candidates": total_candidates,
                },
                label=f"查看 #{current_index + 1} ➡"
            )
        )

    return actions


async def send_candidate_preview_message(task_id: str, candidate_index: int, total_candidates: int):
    viz_result = visualize_lattice_structure.invoke({
        "structure_id": "",
        "task_id": task_id,
        "candidate_index": str(candidate_index)
    })

    if viz_result.get("status") != "success":
        await cl.Message(
            content=f"⚠️ 候选结构图生成失败：{viz_result.get('message', '未知错误')}"
        ).send()
        return

    image_path = viz_result.get("image_path")
    if not image_path or not os.path.exists(image_path):
        await cl.Message(
            content=f"⚠️ 候选结构图生成成功，但图片文件不存在：{image_path}"
        ).send()
        return

    # -----------------------------
    # 1. 导出当前候选 STL
    # -----------------------------
    stl_path = os.path.join(
        "generated_images",
        f"candidate_{task_id}_{candidate_index}.stl"
    )

    stl_file_element = None
    cad_preview_element = None

    try:
        export_result = export_graph_to_stl(
            nodes=viz_result.get("nodes") or [],
            connections=viz_result.get("connections") or [],
            diameter=float(viz_result.get("diameter") or 0.0),
            save_path=stl_path
        )

        if export_result.get("status") == "success" and os.path.exists(stl_path):
            stl_file_element = cl.File(
                name=os.path.basename(stl_path),
                path=stl_path,
                display="inline"
            )

            # -----------------------------
            # 2. 渲染 STL 杆件预览图
            # -----------------------------
            cad_preview_path = os.path.join(
                "generated_images",
                f"candidate_{task_id}_{candidate_index}_cad.png"
            )

            render_stl_preview_png(
                stl_path=stl_path,
                png_path=cad_preview_path
            )

            if os.path.exists(cad_preview_path):
                cad_preview_element = cl.Image(
                    path=cad_preview_path,
                    name=os.path.basename(cad_preview_path),
                    display="inline"
                )

    except Exception:
        stl_file_element = None
        cad_preview_element = None

    # -----------------------------
    # 3. 删除上一次候选预览消息
    # -----------------------------
    old_preview_msg = cl.user_session.get("candidate_preview_msg")
    if old_preview_msg is not None:
        try:
            await old_preview_msg.remove()
        except Exception:
            pass

    # -----------------------------
    # 4. 组装显示元素
    # -----------------------------
    elements = [
        cl.Image(
            path=image_path,
            name=os.path.basename(image_path),
            display="inline"
        )
    ]

    if cad_preview_element is not None:
        elements.append(cad_preview_element)

    if stl_file_element is not None:
        elements.append(stl_file_element)

    # -----------------------------
    # 5. 发送新预览消息
    # -----------------------------
    preview_msg = cl.Message(
        content=(
            "### 候选结构图预览\n"
            f"- 当前展示：第 {candidate_index} 个候选结构\n"
            f"- 总候选数：{total_candidates}\n"
            f"- 节点数：{len(viz_result.get('nodes') or [])}\n"
            f"- 连接数：{len(viz_result.get('connections') or [])}\n"
            f"- 梁直径：{viz_result.get('diameter'):.6f}\n"
            f"- STL导出：{'成功' if stl_file_element is not None else '失败'}\n"
            f"- 杆件预览图：{'成功' if cad_preview_element is not None else '失败'}"
        ),
        elements=elements,
        actions=build_candidate_actions(task_id, candidate_index, total_candidates)
    )

    await preview_msg.send()
    cl.user_session.set("candidate_preview_msg", preview_msg)

@cl.action_callback("view_candidate")
async def on_view_candidate(action: cl.Action):
    """
    点击候选切换按钮后，重新展示对应 candidate 的结构图。
    """
    payload = action.payload or {}
    task_id = str(payload.get("task_id", "")).strip()
    candidate_index = int(payload.get("candidate_index", 0))
    total_candidates = int(payload.get("total_candidates", 0))

    if not task_id:
        await cl.Message(content="⚠️ 缺少 task_id，无法切换候选结构。").send()
        return

    await send_candidate_preview_message(
        task_id=task_id,
        candidate_index=candidate_index,
        total_candidates=total_candidates
    )
async def handle_uploaded_tensor_file(file_element) -> bool:
    """
    处理用户上传的刚度张量文件。
    返回 True 表示本次消息已经被附件流程处理，不再走普通 Agent。
    """
    try:
        df = load_uploaded_table(file_element.path)

        pred_result = dl_service.predict_from_dataframe(df, keep_temp=True)
        if pred_result["status"] != "success":
            await cl.Message(
                content=f"⚠️ 文件预测失败：{pred_result.get('message', '未知错误')}"
            ).send()
            return True

        meta = pred_result.get("meta", {})
        paths = pred_result.get("paths", {})
        tables = pred_result.get("tables", {})

        task_id = str(uuid.uuid4())

        summary = (
            f"已根据上传文件完成预测。\n"
            f"- 任务ID: {task_id}\n"
            f"- 目标数量: {meta.get('num_targets')}\n"
            f"- 候选总数: {meta.get('num_full_pred_rows')}\n"
            f"- 每个目标候选数(估计): {meta.get('num_candidates_estimated')}\n"
            f"- 输出目录: {pred_result.get('temp_dir')}"
        )

        create_prediction_task(
            task_id=task_id,
            source_type="upload",
            user_question=None,
            input_filename=getattr(file_element, "name", None),
            num_targets=meta.get("num_targets"),
            num_candidates=meta.get("num_full_pred_rows"),
            temp_dir=pred_result.get("temp_dir"),
            source_columns=[],
            sql_text=None,
            summary=summary
        )

        for file_type_key, file_type_name in [
            ("full_pred_path", "full_pred"),
            ("C_target_path", "C_target"),
            ("C_target_pred_pred_path", "C_target_pred_pred"),
        ]:
            file_path = paths.get(file_type_key)
            if file_path:
                add_prediction_file(task_id, file_type_name, file_path)

        await cl.Message(content=summary).send()

        # -----------------------------
        # 调试信息：列名与文件路径
        # -----------------------------
        if "full_pred" in tables and tables["full_pred"] is not None:
            print("full_pred columns:", tables["full_pred"].columns.tolist())
        if "C_target" in tables and tables["C_target"] is not None:
            print("C_target columns:", tables["C_target"].columns.tolist())
        if "C_target_pred_pred" in tables and tables["C_target_pred_pred"] is not None:
            print("C_target_pred_pred columns:", tables["C_target_pred_pred"].columns.tolist())

        for key, value in paths.items():
            print(f"{key}: {value}, exists={os.path.exists(value) if value else False}")

        # -----------------------------
        # 1. full_pred 预览
        # -----------------------------
        if "full_pred" in tables and tables["full_pred"] is not None:
            df_full_pred = tables["full_pred"].copy()

            preview_cols = [
                col for col in [
                    "sample",
                    "relative_density",
                    "U1", "U2", "U3",
                    "lattice_type1",
                    "V1", "V2", "V3"
                ] if col in df_full_pred.columns
            ]

            if preview_cols:
                df_full_pred = df_full_pred[preview_cols].head(10).reset_index(drop=True)
            else:
                df_full_pred = df_full_pred.head(10).reset_index(drop=True)

            df_full_pred = df_full_pred.round(4)

            await cl.Message(
                content=(
                    "### full_pred 列说明\n"
                    "- `sample`: 目标样本编号\n"
                    "- `relative_density`: 相对密度\n"
                    "- `U1, U2, U3`: 结构参数 U\n"
                    "- `lattice_type1`: 主晶格类型\n"
                    "- `V1, V2, V3`: 结构参数 V"
                )
            ).send()

            await cl.Message(
                content="### full_pred 预览",
                elements=[
                    cl.Dataframe(
                        name="uploaded_full_pred_preview",
                        display="inline",
                        data=df_full_pred
                    )
                ]
            ).send()

        # -----------------------------
        # 2. C_target 预览
        # -----------------------------
        if "C_target" in tables and tables["C_target"] is not None:
            df_c_target = tables["C_target"].copy()

            target_preview_cols = [
                col for col in [
                    "sample",
                    "C11", "C12", "C13",
                    "C22", "C23", "C33",
                    "C44", "C55", "C66"
                ] if col in df_c_target.columns
            ]

            if target_preview_cols:
                df_c_target = df_c_target[target_preview_cols].head(10).reset_index(drop=True)
            else:
                df_c_target = df_c_target.head(10).reset_index(drop=True)

            df_c_target = df_c_target.round(4)

            await cl.Message(
                content=(
                    "### C_target 列说明\n"
                    "- `sample`: 目标样本编号\n"
                    "- `C11, C12, C13, C22, C23, C33`: 主要法向刚度分量\n"
                    "- `C44, C55, C66`: 主要剪切刚度分量\n"
                    "- 该表表示用户上传文件中的目标刚度张量（预览）"
                )
            ).send()

            await cl.Message(
                content="### C_target 预览",
                elements=[
                    cl.Dataframe(
                        name="uploaded_c_target_preview",
                        display="inline",
                        data=df_c_target
                    )
                ]
            ).send()

        # -----------------------------
        # 3. C_target_pred_pred 预览
        # -----------------------------
        if "C_target_pred_pred" in tables and tables["C_target_pred_pred"] is not None:
            df_c_target_pred = tables["C_target_pred_pred"].copy()

            pred_preview_cols = [
                col for col in [
                    "sample",
                    "C11", "C12", "C13",
                    "C22", "C23", "C33",
                    "C44", "C55", "C66"
                ] if col in df_c_target_pred.columns
            ]

            if pred_preview_cols:
                df_c_target_pred = df_c_target_pred[pred_preview_cols].head(10).reset_index(drop=True)
            else:
                df_c_target_pred = df_c_target_pred.head(10).reset_index(drop=True)

            df_c_target_pred = df_c_target_pred.round(4)

            await cl.Message(
                content=(
                    "### C_target_pred_pred 列说明\n"
                    "- `sample`: 目标样本编号\n"
                    "- `C11, C12, C13, C22, C23, C33`: 回代预测得到的主要法向刚度分量\n"
                    "- `C44, C55, C66`: 回代预测得到的主要剪切刚度分量\n"
                    "- 该表用于验证预测结构对应的刚度是否接近目标刚度"
                )
            ).send()

            await cl.Message(
                content="### C_target_pred_pred 预览",
                elements=[
                    cl.Dataframe(
                        name="uploaded_c_target_pred_preview",
                        display="inline",
                        data=df_c_target_pred
                    )
                ]
            ).send()

        # -----------------------------
        # 4. 下载文件
        # -----------------------------
        file_elements = []
        file_map = {
            "full_pred_path": "full_pred.csv",
            "C_target_path": "C_target.csv",
            "C_target_pred_pred_path": "C_target_pred_pred.csv"
        }

        for key, label in file_map.items():
            file_path = paths.get(key)
            if file_path and os.path.exists(file_path):
                file_elements.append(
                    cl.File(
                        name=label,
                        path=file_path,
                        display="inline"
                    )
                )

        if file_elements:
            await cl.Message(
                content=(
                    "### 下载结果文件\n"
                    "- `full_pred.csv`: 候选结构主结果表\n"
                    "- `C_target.csv`: 目标刚度张量表\n"
                    "- `C_target_pred_pred.csv`: 回代验证刚度表"
                ),
                elements=file_elements
            ).send()
        else:
            await cl.Message(content="⚠️ 未找到可下载的结果文件。").send()

        # 5. 自动生成并展示第 0 个候选结构图（支持切换）
        await send_candidate_preview_message(
            task_id=task_id,
            candidate_index=0,
            total_candidates=meta.get("num_full_pred_rows", 0)
        )

        return True

    except Exception as e:
        await cl.Message(content=f"⚠️ 读取或处理上传文件失败：{e}").send()
        return True
# Chainlit 启动
@cl.on_chat_start
async def start():
    print("应用正在启动...")

    settings = await cl.ChatSettings([
        Select(id="Model", label="Model", values=["gpt-4o-mini", "gpt-4"], initial_index=0),
        Slider(id="Temperature", label="Temperature", initial=0, min=0, max=1, step=0.1),
    ]).send()

    await cl.Message(
        content="""
# Welcome to LatticeGPT

**LatticeGPT** is an AI system for **lattice metamaterial retrieval, inverse design, visualization, and candidate structure management**.  
It integrates database querying, deep learning inverse design, and 3D structure visualization in one interface.

---

### Core Features

- 🔍 **Database Query**
  - Search material and lattice structure properties from `materials.db`  
  - Supports filtering by lattice type, C11 stiffness, and relative density  

- 🧠 **Inverse Design**
  - Generate candidate lattice structures from:
    - Uploaded target stiffness CSV/XLSX files  
    - Natural language queries
  - Uses RAG + DL modules to produce full_pred, C_target, and C_target_pred_pred tables  

- 🎨 **Visualization**
  - Render database or predicted structures as 3D rod-frame plots  
  - Includes automatic STL export and PNG rod-preview  

- 📄 **File Upload & Prediction**
  - Upload CSV/XLSX files with target tensors  
  - Receive structured preview of:
    - **full_pred**: candidate structures  
    - **C_target**: target stiffness tensors  
    - **C_target_pred_pred**: predicted stiffness verification  

- 🔄 **Candidate Switching**
  - Navigate among multiple predicted structures using forward/backward buttons  
  - Displays node count, connection count, and beam diameter  

- 💾 **Download Results**
  - Export CSV files: `full_pred.csv`, `C_target.csv`, `C_target_pred_pred.csv`  
  - Export STL files for CAD applications  

- 🕒 **History Management**
  - View recent prediction tasks  
  - Check detailed task info including result files, summary, and source columns  

---

### Quick Start

1. **Start a Query**
   - Enter a question or upload a target stiffness file
2. **View Candidates**
   - Check the first predicted structure and switch among other candidates
3. **Preview & Export**
   - Preview rod-frame plots and STL files, download CSV tables
4. **Check History**
   - Review previously completed tasks and exported results
"""
    ).send()

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True
        )

        cl.user_session.set("agent", agent_executor)
        cl.user_session.set("chat_history", [])

        print("✅ Agent 初始化完成")
    except Exception as e:
        await cl.Message(content=f"❌ 系统初始化失败: {e} (请检查 API Key)").send()


# -----------------------------
# Chainlit 消息处理
# -----------------------------
@cl.on_message
async def main(message: cl.Message):
    import json

    agent = cl.user_session.get("agent")
    if not agent:
        await cl.Message(content="⚠️ Agent 未就绪，请刷新页面重试。").send()
        return

    chat_history = cl.user_session.get("chat_history") or []
    # -----------------------------
    # 0. 优先处理用户上传的附件
    # -----------------------------
    if getattr(message, "elements", None):
        uploaded_files = [
            el for el in message.elements
            if hasattr(el, "path") and el.path and (
                el.path.lower().endswith(".csv") or el.path.lower().endswith(".xlsx")
            )
        ]

        if uploaded_files:
            # 当前先处理第一个文件
            handled = await handle_uploaded_tensor_file(uploaded_files[0])

            # 更新多轮历史
            chat_history.append(HumanMessage(content=message.content or f"[上传文件] {uploaded_files[0].name}"))
            chat_history.append(AIMessage(content=f"已完成上传文件预测：{uploaded_files[0].name}"))
            cl.user_session.set("chat_history", chat_history)

            if handled:
                return

    try:
        res = await agent.ainvoke({
            "input": message.content,
            "chat_history": chat_history
        })

        response_text = res["output"]

        # -----------------------------
        # 1. 解析工具中间结果
        # -----------------------------
        intermediate_steps = res.get("intermediate_steps", [])
        design_payload = None

        for step in intermediate_steps:
            try:
                action, tool_output = step
                tool_name = getattr(action, "tool", "")
                if tool_name == "design_structure_from_query":
                    parsed = json.loads(tool_output)
                    if parsed.get("status") == "success":
                        design_payload = parsed
                    else:
                        response_text = parsed.get("message", response_text)
            except Exception:
                pass

        # 用工具里的 summary 替换默认长文本
        if design_payload:
            response_text = design_payload.get("summary", response_text)

        # 清理 response_text 中的图片 markdown / 图片路径
        # 避免前端先显示一个坏图占位，再由下面统一发送真实图片
        response_text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', response_text)  # 去掉所有 markdown 图片
        response_text = re.sub(r'[A-Za-z]:[\\/][^\s\)]*generated_images[\\/][^\s\)]*\.png', '', response_text)  # 去掉 Windows 绝对图片路径
        response_text = re.sub(r'generated_images[\\/][^\s\)]*\.png', '', response_text)  # 去掉相对图片路径
        response_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', response_text)  # 压缩多余空行
        response_text = response_text.strip()
        # -----------------------------
        # 2. 先发主摘要消息
        # -----------------------------
        await cl.Message(content=response_text).send()

        # -----------------------------
        # 3. 如果是设计任务，分开发表格
        # -----------------------------
        if design_payload:
            preview = design_payload.get("preview", {})
            paths = design_payload.get("paths", {})

            if preview.get("full_pred"):
                df_full_pred = pd.DataFrame(preview["full_pred"])
                await cl.Message(
                    content="### full_pred 预览",
                    elements=[
                        cl.Dataframe(
                            name="full_pred_preview",
                            display="inline",
                            data=df_full_pred
                        )
                    ]
                ).send()

            if preview.get("C_target"):
                df_c_target = pd.DataFrame(preview["C_target"])
                await cl.Message(
                    content="### C_target 预览",
                    elements=[
                        cl.Dataframe(
                            name="C_target_preview",
                            display="inline",
                            data=df_c_target
                        )
                    ]
                ).send()

            if preview.get("C_target_pred_pred"):
                df_c_target_pred = pd.DataFrame(preview["C_target_pred_pred"])
                await cl.Message(
                    content="### C_target_pred_pred 预览",
                    elements=[
                        cl.Dataframe(
                            name="C_target_pred_pred_preview",
                            display="inline",
                            data=df_c_target_pred
                        )
                    ]
                ).send()

            # -----------------------------
            # 4. 单独发下载文件
            # -----------------------------
            file_elements = []
            file_map = {
                "full_pred_path": "full_pred.csv",
                "C_target_path": "C_target.csv",
                "C_target_pred_pred_path": "C_target_pred_pred.csv"
            }

            for key, label in file_map.items():
                file_path = paths.get(key)
                if file_path and os.path.exists(file_path):
                    file_elements.append(
                        cl.File(
                            name=label,
                            path=file_path,
                            display="inline"
                        )
                    )

            if file_elements:
                await cl.Message(
                    content="### 下载结果文件",
                    elements=file_elements
                ).send()
            # -----------------------------
            # 5. 自动生成并展示第 0 个候选结构图（支持切换）
            # -----------------------------
            await send_candidate_preview_message(
                task_id=design_payload.get("task_id", ""),
                candidate_index=0,
                total_candidates=design_payload.get("meta", {}).get("num_full_pred_rows", 0)
            )

        # -----------------------------
        # 5. 自动抓取图片，单独发
        # -----------------------------
        image_pattern_relative = r'generated_images[\\/][^\s\)]*\.png'
        image_pattern_absolute = r'[A-Za-z]:[\\/][^\s\)]*generated_images[\\/][^\s\)]*\.png'

        raw_images = (
            re.findall(image_pattern_relative, str(res)) +
            re.findall(image_pattern_absolute, str(res)) +
            re.findall(image_pattern_relative, response_text) +
            re.findall(image_pattern_absolute, response_text)
        )

        # 统一路径格式后再去重，避免同一张图因为相对/绝对路径不同而重复显示
        normalized_images = []
        seen = set()

        for img_path in raw_images:
            abs_path = os.path.abspath(img_path)
            norm_path = os.path.normpath(abs_path)

            if norm_path not in seen and os.path.exists(norm_path):
                seen.add(norm_path)
                normalized_images.append(norm_path)

        image_elements = []
        for img_path in normalized_images:
            image_elements.append(
                cl.Image(path=img_path, name=os.path.basename(img_path), display="inline")
            )

        if image_elements:
            await cl.Message(
                content="### 可视化结果",
                elements=image_elements
            ).send()
        # -----------------------------
        # 6. 更新多轮历史
        # -----------------------------
        chat_history.append(HumanMessage(content=message.content))
        chat_history.append(AIMessage(content=response_text))
        cl.user_session.set("chat_history", chat_history)

    except Exception as e:
        await cl.Message(content=f"⚠️ 运行出错: {e}").send()
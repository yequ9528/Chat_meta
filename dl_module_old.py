import os
import sys
import torch
import numpy as np
import time
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# --- 1. 环境与路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 尝试导入 src 中的模块
try:
    sys.path.append(os.path.join(current_dir, 'src'))
    from src.model_utils import invModel_output
    from src.normalization import decodeOneHot
    from src.loadDataset import getSavedNormalization
except ImportError as e:
    print(f"⚠️ [DL Module] Import Error: {e}")
    print("请确保 'src' 文件夹与 dl_module_old.py 在同一目录。")


class TrussInverseDesigner:
    def __init__(self, models_dir='models'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔮 [DL Module] Initializing Designer on {self.device}...")

        # 设定基材的杨氏模量 (参考 main_predict.py)
        # 钛合金 Ti-6Al-4V 约为 114 GPa
        self.BASE_E = 114.0

        self.models_dir = os.path.join(current_dir, models_dir)

        try:
            # 加载归一化参数
            self.F1_scaling, self.C_ort_scaling, self.C_scaling, self.V_scaling, self.C_hat_scaling = getSavedNormalization()
            print("✅ Normalization scalers loaded.")

            # 加载模型
            g1_path = os.path.join(self.models_dir, "G1.pt")
            g2_path = os.path.join(self.models_dir, "G2.pt")

            if not os.path.exists(g1_path) or not os.path.exists(g2_path):
                raise FileNotFoundError(f"Model files not found in {self.models_dir}")

            # weights_only=False 解决 PyTorch 2.6+ 安全加载问题
            self.G1 = torch.load(g1_path, map_location=self.device, weights_only=False)
            self.G2 = torch.load(g2_path, map_location=self.device, weights_only=False)

            self.G1.eval()
            self.G2.eval()
            print("✅ Inverse Models (G1 & G2) loaded successfully.")

        except Exception as e:
            print(f"❌ [DL Module] Initialization Failed: {str(e)}")
            self.G1 = None

    def _create_isotropic_tensor(self, E_relative, nu=0.3):
        """
        将相对杨氏模量 (E_target / E_base) 转换为 21维刚度张量
        """
        # 使用相对 E 计算 C，这样得到的 C 也是相对的
        mu = E_relative / (2 * (1 + nu))
        lam = (E_relative * nu) / ((1 + nu) * (1 - 2 * nu))

        C11 = 2 * mu + lam
        C12 = lam
        C44 = mu

        C_vec = np.zeros(21)
        # 填充对角线 C11, C22, C33
        C_vec[[0, 6, 11]] = C11
        # 填充剪切 C44, C55, C66
        C_vec[[15, 18, 20]] = C44
        # 填充非对角 C12, C13, C23
        C_vec[[1, 2, 7]] = C12

        return torch.FloatTensor(C_vec).unsqueeze(0)

    def predict(self, target_youngs_modulus_mpa, temperature=200., num_samples=1):
        """
        核心预测函数
        :param target_youngs_modulus_mpa: 用户输入的目标刚度，单位 MPa
        """
        if self.G1 is None:
            return [{"error": "Model not initialized properly."}]

        # ==========================================
        # 🔧 关键修正 1: 单位统一 (MPa -> GPa)
        # ==========================================
        target_val_gpa = target_youngs_modulus_mpa / 1000.0

        # ==========================================
        # 🔧 关键修正 2: 相对化 (GPa -> Relative to Base E)
        # 模型训练时使用的是相对刚度 (C / 114.0)
        # ==========================================
        target_val_relative = target_val_gpa / self.BASE_E

        # 1. 准备数据 (使用相对值构造 Tensor)
        C_target_raw = self._create_isotropic_tensor(target_val_relative).to(self.device)

        # 2. 归一化 (现在的 C_target_raw 应该在 0.0~0.2 之间，归一化后在正常范围)
        C_target_normalized = self.C_scaling.normalize(C_target_raw)

        if num_samples > 1:
            C_target_normalized = C_target_normalized.repeat(num_samples, 1)

        # 3. 推理
        with torch.no_grad():
            rho_U_pred, V_pred, _, _, topology_pred = invModel_output(
                self.G1, self.G2, C_target_normalized, temperature, 'gumbel'
            )

        # 4. 后处理 & 反归一化
        results = []

        # 解码拓扑
        topology_labels = decodeOneHot(topology_pred)

        # 拼接并反归一化 F1 特征 (包含密度)
        F1_features_raw = torch.cat((rho_U_pred, topology_labels), dim=1)
        F1_features_real = self.F1_scaling.unnormalize(F1_features_raw)

        # 反归一化几何参数 V
        V_real = self.V_scaling.unnormalize(V_pred)

        # 转为 Numpy
        F1_np = F1_features_real.cpu().numpy()
        V_np = V_real.cpu().numpy()
        topo_np = topology_labels.cpu().numpy()

        for i in range(num_samples):
            # 获取真实密度
            real_density = float(F1_np[i][0])
            # 限制范围 [0, 1]
            real_density = max(0.0, min(1.0, real_density))

            res = {
                "input_MPa": target_youngs_modulus_mpa,
                "model_input_relative": target_val_relative,  # 调试用
                "predicted_density": round(real_density, 4),
                "structure_type_id": topo_np[i].tolist(),
                "geometry_params": V_np[i].tolist(),
                "note": "AI Design (Base E=114 GPa)"
            }
            results.append(res)

        return results


# 单例导出供 Agent 使用
dl_designer = TrussInverseDesigner()


# --- Tool 定义 ---
class DesignInput(BaseModel):
    stiffness: float = Field(..., description="Target stiffness (Young's Modulus) in MPa.")


@tool(args_schema=DesignInput)
def predict_structure_from_stiffness(stiffness: float):
    """
    Design a new material structure for a target stiffness (MPa).
    Use this when the user wants to generate or predict a structure.
    """
    try:
        results = dl_designer.predict(target_youngs_modulus_mpa=stiffness)
        best = results[0]
        return (f"Design Prediction (Target: {stiffness} MPa):\n"
                f"- Rec. Density: {best['predicted_density'] * 100:.2f}%\n"
                f"- Structure ID: {best['structure_type_id']}\n"
                f"- Geom Params: {[round(x, 3) for x in best['geometry_params'][:3]]}...")
    except Exception as e:
        return f"Prediction Error: {e}"


# --- 独立测试代码 ---
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 开始 dl_module 独立测试 (Fixed Version)")
    print("=" * 60)

    # 测试案例: 500 MPa 和 4000 MPa
    # 上线密度是10%
    test_cases = [500.0, 4000.0]

    for val in test_cases:
        print(f"\n🎯 测试输入: {val} MPa")
        try:
            res = dl_designer.predict(target_youngs_modulus_mpa=val, num_samples=1)
            rel_in = res[0]['model_input_relative']
            dens = res[0]['predicted_density']
            print(f"✅ 相对刚度输入: {rel_in:.6f} (应 < 1.0)")
            print(f"✅ 预测密度: {dens * 100:.2f}%")
        except Exception as e:
            print(f"❌ 失败: {e}")
            import traceback

            traceback.print_exc()

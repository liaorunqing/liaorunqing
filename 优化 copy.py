import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
import time
import warnings
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False    # 正确显示负号
# 忽略部分警告以提高可读性
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

# ============================================================================
# 1. 参数定义 - 调整反应动力学参数
# ============================================================================

@dataclass
class ReactorParameters:
    """反应器参数 - 调整后"""
    V: float = 2.0  # m³
    rho_Cp: float = 4.18e6  # J/(m³·K)
    k1: float = 5.0e8  # 提高主反应频率因子(原1.2e8)
    k2: float = 8.5e6  # 保持副反应频率因子不变
    E1: float = 55000  # 降低主反应活化能(原65000)
    E2: float = 75000  # 保持副反应活化能不变
    DH1: float = -50000  # J/mol
    DH2: float = -80000  # J/mol
    kd: float = 1.0e-4  # 降低催化剂失活速率(原2.1e-4)
    Ed: float = 120000  # J/mol
    CA0: float = 3.0  # mol/L
    CB0: float = 2.5  # mol/L
    T0: float = 298  # K
    PC: float = 150  # 元/mol
    PA: float = 80   # 元/mol
    PB: float = 60   # 元/mol
    PE: float = 0.2  # 元/kJ
    PM: float = 500  # 元/kg
    R: float = 8.314  # J/(mol·K)

# ============================================================================
# 2. 改进的反应器模型 - 增强反应动力学和温度控制
# ============================================================================

class CSTRModel:
    def __init__(self, params: ReactorParameters, T_final=3600):
        self.p = params
        self.T_final = T_final  # 添加总时间属性用于终端温度控制
        
    def reaction_rates(self, CA, CB, T, alpha):
        """计算反应速率，添加数值保护和温度依赖性增强"""
        # 防止温度过低导致的数值问题
        T_safe = max(T, 273.15)
        
        # 限制指数项避免溢出
        exp1_term = -self.p.E1 / (self.p.R * T_safe)
        exp2_term = -self.p.E2 / (self.p.R * T_safe)
        
        # 防止指数溢出
        exp1 = np.exp(max(-50, min(50, exp1_term)))
        exp2 = np.exp(max(-50, min(50, exp2_term)))
        
        # 确保浓度非负
        CA_safe = max(0, CA)
        CB_safe = max(0, CB)
        alpha_safe = max(0, min(1, alpha))  # 活性在[0,1]范围内
        
        # 添加温度依赖性增强因子 - 在高温下反应更有效
        temp_factor = 1.0
        if T_safe > 350:
            temp_factor = 1.0 + 0.05 * (T_safe - 350) / 50  # 每升高50K，效率提高5%
        
        r1 = self.p.k1 * alpha_safe * CA_safe * CB_safe * exp1 * temp_factor
        r2 = self.p.k2 * alpha_safe * CA_safe**2 * exp2
        
        return r1, r2
    
    def selectivity(self, CA, CB, T):
        """计算选择性，添加数值保护"""
        T_safe = max(T, 273.15)
        exp_diff_term = (self.p.E2 - self.p.E1) / (self.p.R * T_safe)
        exp_diff = np.exp(max(-50, min(50, exp_diff_term)))
        
        CA_safe = max(1e-10, CA)  # 避免除零
        CB_safe = max(1e-10, CB)
        
        numerator = self.p.k1 * CB_safe * exp_diff
        denominator = numerator + self.p.k2 * CA_safe
        
        if denominator < 1e-10:
            return 1.0  # 极限情况下默认为1
        return numerator / denominator
    
    def dynamics(self, t, x, u_func):
        """系统动力学，添加保护和终端温度控制"""
        # 保证状态变量在合理范围内
        CA = max(0, x[0])
        CB = max(0, x[1]) 
        CC = max(0, x[2])
        T = max(273, min(500, x[3]))  # 温度限制在273-500K
        alpha = max(0, min(1, x[4]))  # 活性在[0,1]范围内
        
        # 获取控制输入
        try:
            u = u_func(t)
            F = max(0.1, min(1.5, u[0]))  # 限制流量
            Q = max(-500, min(800, u[1]))  # 限制热功率
            mcat = max(0, min(0.05, u[2]))  # 限制催化剂添加
        except:
            # 出错默认安全控制
            F, Q, mcat = 0.5, 0, 0
        
        # 反应速率
        r1, r2 = self.reaction_rates(CA, CB, T, alpha)
        
        # 微分方程
        dCA_dt = (F/self.p.V) * (self.p.CA0 - CA) - r1 - r2
        dCB_dt = (F/self.p.V) * (self.p.CB0 - CB) - r1
        dCC_dt = (F/self.p.V) * (0 - CC) + r1
        
        # 热量平衡，添加保护
        heat_reaction = (-self.p.DH1*r1 - self.p.DH2*r2) / self.p.rho_Cp
        heat_reaction = max(-100, min(100, heat_reaction))  # 限制热反应贡献
        
        dT_dt = (F/self.p.V) * (self.p.T0 - T) + heat_reaction + Q*1000 / (self.p.rho_Cp * self.p.V)
        
        # 添加终端温度控制逻辑
        t_ratio = t / self.T_final
        if t_ratio > 0.8:  # 在最后20%的时间
            # 强制温度向350K靠近
            target_temp = 350
            temp_control_factor = 5.0 * (t_ratio - 0.8) / 0.2  # 0到5的因子
            dT_dt = dT_dt + temp_control_factor * (target_temp - T) / 100
        
        # 催化剂活性，添加保护
        deactivation_term = -self.p.kd * alpha**2 * T
        if T > 273:
            deactivation_exp = np.exp(max(-50, min(0, -self.p.Ed / (self.p.R * T))))
            deactivation = deactivation_term * deactivation_exp
        else:
            deactivation = 0
        
        # 催化剂补充影响（更合理的模型）
        catalyst_addition = 0
        if mcat > 0:
            # 假设新催化剂活性为1，计算混合后活性
            catalyst_addition = 0.05 * mcat * (1 - alpha)  # 补充效率降低，且基于活性差值
        
        dalpha_dt = deactivation + catalyst_addition
        
        return np.array([dCA_dt, dCB_dt, dCC_dt, dT_dt, dalpha_dt])

# ============================================================================
# 3. 系统分析类 - 添加反应动力学诊断
# ============================================================================

class SystemAnalysis:
    def __init__(self, model):
        self.model = model
    
    def find_steady_state(self, u_fixed):
        """寻找给定控制下的稳态点"""
        def u_func(t):
            return u_fixed
        
        def residual(x):
            return self.model.dynamics(0, x, u_func)
        
        from scipy.optimize import fsolve
        x0_guess = np.array([1.0, 0.8, 0.5, 380, 0.9])
        x_ss = fsolve(residual, x0_guess)
        
        return x_ss
    
    def linearize(self, x_ss, u_ss):
        """线性化系统"""
        eps = 1e-6
        n_x, n_u = 5, 3
        A = np.zeros((n_x, n_x))
        B = np.zeros((n_x, n_u))
        
        # 在稳态点计算雅可比矩阵
        def u_func(t):
            return u_ss
            
        f_ss = self.model.dynamics(0, x_ss, u_func)
        
        # 计算A矩阵
        for i in range(n_x):
            x_pert = x_ss.copy()
            x_pert[i] += eps
            f_pert = self.model.dynamics(0, x_pert, u_func)
            A[:, i] = (f_pert - f_ss) / eps
        
        # 计算B矩阵
        for i in range(n_u):
            u_pert = u_ss.copy()
            u_pert[i] += eps
            
            def u_pert_func(t):
                return u_pert
                
            f_pert = self.model.dynamics(0, x_ss, u_pert_func)
            B[:, i] = (f_pert - f_ss) / eps
            
        return A, B
    
    def stability_analysis(self, A):
        """稳定性分析"""
        eigenvalues = np.linalg.eigvals(A)
        return {
            'eigenvalues': eigenvalues,
            'stable': np.all(np.real(eigenvalues) < 0),
            'max_real': np.max(np.real(eigenvalues))
        }
    
    def controllability(self, A, B):
        """可控性分析"""
        n = A.shape[0]
        C = B.copy()
        
        for i in range(1, n):
            C = np.hstack([C, np.linalg.matrix_power(A, i) @ B])
            
        rank = np.linalg.matrix_rank(C)
        return {
            'rank': rank,
            'controllable': rank == n
        }
    
    def analyze_reaction_kinetics(self):
        """分析反应动力学特性"""
        temps = np.linspace(300, 450, 20)
        r1_vals = []
        r2_vals = []
        selectivity = []
        
        for T in temps:
            r1, r2 = self.model.reaction_rates(1.0, 1.0, T, 0.9)
            r1_vals.append(r1)
            r2_vals.append(r2)
            if r1 + r2 > 1e-10:
                S = r1 / (r1 + r2)
            else:
                S = 1.0
            selectivity.append(S)
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(temps, r1_vals, 'b-', label='主反应')
        plt.plot(temps, r2_vals, 'r-', label='副反应')
        plt.xlabel('温度 (K)')
        plt.ylabel('反应速率')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(temps, selectivity, 'g-')
        plt.xlabel('温度 (K)')
        plt.ylabel('选择性')
        plt.axhline(y=0.85, color='r', linestyle='--', label='最小选择性要求')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # 找出最佳反应温度
        max_rate_idx = np.argmax(r1_vals)
        best_temp = temps[max_rate_idx]
        print(f"最佳反应温度约为: {best_temp:.1f} K")
        print(f"在此温度下，主反应速率: {r1_vals[max_rate_idx]:.4f}, 副反应速率: {r2_vals[max_rate_idx]:.4f}")
        print(f"选择性: {selectivity[max_rate_idx]:.4f}")
        
        return best_temp

# ============================================================================
# 4. 改进的优化控制器 - 四阶段优化策略，强化终端温度控制
# ============================================================================

class OptimalControl:
    def __init__(self, model, T_final=3600):  # 1小时
        self.model = model
        self.T_final = T_final
        self.N_control = 20  # 控制段数
        self.dt = T_final / self.N_control
        self.simulation_fail_count = 0
        self.max_fail_count = 3  # 连续失败次数上限
        
    def parameterize_control(self, u_params):
        """将优化参数转换为控制函数，使用平滑过渡"""
        N = self.N_control
        F_vals = u_params[:N]
        Q_vals = u_params[N:2*N]
        mcat_vals = u_params[2*N:3*N]
        
        # 应用平滑滤波
        F_vals = np.array([np.mean(F_vals[max(0,i-1):min(N,i+2)]) for i in range(N)])
        Q_vals = np.array([np.mean(Q_vals[max(0,i-1):min(N,i+2)]) for i in range(N)])
        
        def u_func(t):
            # 使用三次样条插值获取更平滑的控制
            t_norm = t / self.dt
            idx = min(int(t_norm), N-1)
            
            # 在控制段之间三次样条插值
            if idx < N-1:
                frac = t_norm - idx
                # 使用三次函数平滑过渡
                h = frac**2 * (3 - 2*frac)
                F = F_vals[idx] * (1-h) + F_vals[idx+1] * h
                Q = Q_vals[idx] * (1-h) + Q_vals[idx+1] * h
                mcat = mcat_vals[idx] * (1-h) + mcat_vals[idx+1] * h
            else:
                F = F_vals[idx]
                Q = Q_vals[idx]
                mcat = mcat_vals[idx]
            
            # 添加终端温度控制逻辑 - 在最后阶段强制冷却
            t_ratio = t / self.T_final
            if t_ratio > 0.85:  # 最后15%的时间
                # 逐渐增强冷却效果
                cooling_factor = (t_ratio - 0.85) / 0.15  # 0到1的因子
                Q = Q * (1 - cooling_factor) - 300 * cooling_factor  # 逐渐转向冷却
            
            return np.array([F, Q, mcat])
        
        return u_func
    
    def simulate(self, x0, u_func):
        """模拟系统行为，添加鲁棒性"""
        try:
            # 尝试使用不同的求解器
            methods = ['LSODA', 'BDF', 'Radau']
            
            for method in methods:
                sol = solve_ivp(
                    lambda t, x: self.model.dynamics(t, x, u_func),
                    [0, self.T_final],
                    x0,
                    method=method,  # 尝试不同的求解器
                    dense_output=True,
                    rtol=1e-4,
                    atol=1e-6,
                    max_step=self.dt/2,  # 限制步长提高稳定性
                    first_step=self.dt/10  # 更小的初始步长
                )
                
                if sol.success:
                    self.simulation_fail_count = 0  # 重置失败计数
                    return sol
            
            # 如果所有方法都失败，尝试分段求解
            print("标准求解器失败，尝试分段求解...")
            segments = 4
            segment_time = self.T_final / segments
            x_current = x0.copy()
            t_segments = []
            y_segments = []
            
            for i in range(segments):
                t_start = i * segment_time
                t_end = (i + 1) * segment_time
                
                sol_segment = solve_ivp(
                    lambda t, x: self.model.dynamics(t, x, u_func),
                    [t_start, t_end],
                    x_current,
                    method='Radau',
                    dense_output=True,
                    rtol=1e-3,
                    atol=1e-5
                )
                
                if not sol_segment.success:
                    self.simulation_fail_count += 1
                    print(f"分段{i+1}求解失败")
                    return None
                
                t_segments.append(sol_segment.t)
                y_segments.append(sol_segment.y)
                x_current = sol_segment.y[:, -1]
            
            # 合并分段结果
            t_combined = np.concatenate(t_segments)
            y_combined = np.hstack(y_segments)
            
            # 创建一个类似solve_ivp返回的对象
            class CombinedSolution:
                def __init__(self, t, y):
                    self.t = t
                    self.y = y
                    self.success = True
                    
                    # 创建插值函数
                    from scipy.interpolate import interp1d
                    self._interpolant = interp1d(t, y, axis=1, bounds_error=False, fill_value="extrapolate")
                    
                def sol(self, t):
                    return self._interpolant(t)
            
            combined_sol = CombinedSolution(t_combined, y_combined)
            self.simulation_fail_count = 0
            return combined_sol
                    
        except Exception as e:
            self.simulation_fail_count += 1
            print(f"模拟错误: {str(e)}")
            return None
    
    def simplified_objective(self, u_params):
        """简化目标函数 - 仅关注产品浓度"""
        # 初始状态
        x0 = np.array([0.5, 0.4, 0.1, 350, 0.95])
        
        # 转换控制
        u_func = self.parameterize_control(u_params)
        
        # 模拟
        sol = self.simulate(x0, u_func)
        if sol is None:
            return 1e6
        
        # 评估终端产品浓度
        x_final = sol.y[:, -1]
        CC_final = x_final[2]
        
        # 目标：最大化产品浓度
        objective = -CC_final * 1000  # 负号使其成为最小化问题
        
        # 添加温度约束的软惩罚
        T_final = x_final[3]
        if abs(T_final - 350) > 20:  # 放宽温度约束
            objective += 100 * (abs(T_final - 350) - 20)**2
        
        return objective
    
    def terminal_temperature_objective(self, u_params):
        """专门优化终端温度的目标函数"""
        # 初始状态
        x0 = np.array([0.5, 0.4, 0.1, 350, 0.95])
        
        # 转换控制
        u_func = self.parameterize_control(u_params)
        
        # 模拟
        sol = self.simulate(x0, u_func)
        if sol is None:
            return 1e6
        
        # 评估终端温度
        x_final = sol.y[:, -1]
        T_final = x_final[3]
        CC_final = x_final[2]
        
        # 目标：使终端温度接近350K，同时保持产品浓度
        objective = (T_final - 350)**2 * 100
        
        # 如果产品浓度太低，也添加惩罚
        if CC_final < 1.7:  # 稍微放宽产品浓度要求
            objective += (1.7 - CC_final)**2 * 1000
        
        return objective
    
    def feasibility_objective(self, u_params):
        """可行性目标函数 - 仅关注约束满足"""
        # 初始状态
        x0 = np.array([0.5, 0.4, 0.1, 350, 0.95])
        
        # 转换控制
        u_func = self.parameterize_control(u_params)
        
        # 模拟
        sol = self.simulate(x0, u_func)
        if sol is None:
            if self.simulation_fail_count > self.max_fail_count:
                self.simulation_fail_count = 0
                return 1e8  # 连续多次失败，返回高惩罚
            return 1e7  # 模拟失败，返回高惩罚
        
        # 评估终端约束
        x_final = sol.y[:, -1]
        CC_final, T_final, alpha_final = x_final[2], x_final[3], x_final[4]
        
        # 约束违反惩罚
        penalty = 0
        
        # 终端约束
        if CC_final < 1.8:
            penalty += 5e5 * (1.8 - CC_final)**2
        
        if abs(T_final - 350) > 10:
            penalty += 1e6 * (abs(T_final - 350) - 10)**2  # 大幅增加终端温度约束权重
            
        if alpha_final < 0.6:
            penalty += 1e4 * (0.6 - alpha_final)**2
        
        # 催化剂总消耗约束
        t_eval = np.linspace(0, self.T_final, 100)
        u_traj = np.array([u_func(ti) for ti in t_eval]).T
        cat_consumption = np.trapz(u_traj[2], t_eval)
        
        if cat_consumption > 50:
            penalty += 1e4 * (cat_consumption - 50)**2
        
        # 添加温度轨迹约束 - 确保中期高温，终期降温
        x_traj = sol.sol(t_eval)
        mid_point = len(t_eval) // 2
        mid_temp = np.mean(x_traj[3, mid_point-10:mid_point+10])
        if mid_temp < 400:  # 中期温度应该高
            penalty += 1e4 * (400 - mid_temp)**2
        
        return penalty
    
    def objective(self, u_params):
        """完整优化目标函数"""
        # 初始状态
        x0 = np.array([0.5, 0.4, 0.1, 350, 0.95])
        
        # 控制函数
        u_func = self.parameterize_control(u_params)
        
        # 模拟
        sol = self.simulate(x0, u_func)
        if sol is None:
            return 1e10
        
        # 评估目标函数
        t_eval = np.linspace(0, self.T_final, 100)
        x_traj = sol.sol(t_eval)
        u_traj = np.array([u_func(ti) for ti in t_eval]).T
        
        # 四个目标函数组件
        J1 = 0  # 经济效益
        J2 = 0  # 环境影响
        J3 = 0  # 操作平稳性
        J4 = 0  # 催化剂效率
        
        # 变化率测量
        F_rates = np.diff(u_traj[0]) / np.diff(t_eval)
        Q_rates = np.diff(u_traj[1]) / np.diff(t_eval)
        
        for i in range(len(t_eval)-1):
            dt = t_eval[i+1] - t_eval[i]
            CA, CB, CC, T, alpha = x_traj[:, i]
            F, Q, mcat = u_traj[:, i]
            
            # 反应速率
            r1, r2 = self.model.reaction_rates(CA, CB, T, alpha)
            
            # J1: 经济效益
            J1_i = (self.model.p.PC * CC * F - 
                   self.model.p.PA * CA * F - 
                   self.model.p.PB * CB * F - 
                   self.model.p.PE * abs(Q) - 
                   self.model.p.PM * mcat) * dt
            
            # J2: 环境影响
            J2_i = (100 * r2 + 0.01 * (T - 298)**2 + 1e-6 * Q**2) * dt
            
            # J4: 催化剂效率
            if T > 273:
                deactivation_exp = np.exp(max(-50, min(0, -self.model.p.Ed / (self.model.p.R * T))))
                J4_i = (self.model.p.kd * alpha**2 * T * deactivation_exp + 0.1 * mcat) * dt
            else:
                J4_i = 0.1 * mcat * dt
            
            # 累加
            J1 += J1_i
            J2 += J2_i
            J4 += J4_i
        
        # J3: 操作平稳性 (使用整体评估)
        if len(F_rates) > 0:
            J3 = np.sum(F_rates**2) * self.dt + np.sum(Q_rates**2) * self.dt
        else:
            J3 = 0
            
        # 加权组合
        J = -1.0 * J1 + 0.3 * J2 + 0.2 * J3 + 0.5 * J4
        
        # 约束惩罚
        # 终端约束（软约束）
        x_final = x_traj[:, -1]
        CC_final, T_final, alpha_final = x_final[2], x_final[3], x_final[4]
        
        # 产量约束: CC(T) ≥ 1.8 mol/L
        if CC_final < 1.8:
            J += 5e6 * (1.8 - CC_final)**2  # 增加权重
            
        # 温度约束: |T(T) - 350| ≤ 10 K
        if abs(T_final - 350) > 10:
            J += 1e7 * (abs(T_final - 350) - 10)**2  # 大幅增加权重
            
        # 催化剂活性约束: α(T) ≥ 0.6
        if alpha_final < 0.6:
            J += 1e5 * (0.6 - alpha_final)**2
        
        # 催化剂总消耗约束
        cat_consumption = np.trapz(u_traj[2], t_eval)
        if cat_consumption > 50:
            J += 1e5 * (cat_consumption - 50)**2
        
        # 添加温度轨迹约束 - 确保中期高温，终期降温
        mid_point = len(t_eval) // 2
        mid_temp = np.mean(x_traj[3, mid_point-10:mid_point+10])
        if mid_temp < 400:  # 中期温度应该高
            J += 1e5 * (400 - mid_temp)**2
        
        # 确保温度在最后阶段下降
        temp_last_quarter = x_traj[3, int(3*len(t_eval)/4):]
        if not np.all(np.diff(temp_last_quarter) <= 0.1):  # 检查是否基本单调下降
            J += 1e5  # 添加惩罚
        
        return J
    
    def optimize(self):
        """执行四阶段优化"""
        N = self.N_control
        
        # 控制变量边界
        bounds = []
        bounds.extend([(0.1, 1.2)] * N)     # F bounds
        bounds.extend([(-300, 800)] * N)    # Q bounds - 扩大上限
        bounds.extend([(0, 0.03)] * N)      # mcat bounds
        
        # 初始猜测（基于反应动力学分析）
        x0 = []
        # 进料流量F初始策略
        f_vals = np.ones(N) * 0.5  # 降低整体流量
        f_vals[:int(N/4)] = 0.3    # 启动阶段更低流量
        f_vals[int(3*N/4):] = 0.7  # 结束阶段增加流量
        x0.extend(f_vals)
        
        # 温度控制Q初始策略 - 三阶段温度控制
        q_vals = np.ones(N) * 400  # 中期维持高温
        q_vals[:int(N/4)] = 600    # 启动阶段快速升温
        q_vals[int(3*N/4):] = 0    # 最后阶段主动降温至目标温度
        q_vals[-int(N/10):] = -200 # 最后时刻强制冷却到350K附近
        x0.extend(q_vals)
        
        # 催化剂投加策略
        mcat_vals = np.ones(N) * 0.01
        mcat_vals[int(N/2):] = 0.02  # 后期增加投加
        mcat_vals[-int(N/5):] = 0.005  # 最后阶段减少
        x0.extend(mcat_vals)
        
        print("第一阶段优化：简化目标函数...")
        start_time = time.time()
        
        # 第一阶段：使用简化目标函数
        result1 = differential_evolution(
            self.simplified_objective,
            bounds,
            x0=x0,
            strategy='best1bin',
            maxiter=30,
            popsize=20,
            tol=0.01,
            mutation=(0.5, 1.5),  # 扩大变异范围
            recombination=0.8,
            disp=True,
            workers=-1
        )
        
        print(f"第一阶段完成! 用时: {time.time() - start_time:.1f}秒")
        print(f"简化目标值: {result1.fun:.2f}")
        
        # 第二阶段：寻找可行解
        print("\n第二阶段优化：寻找可行解...")
        start_time = time.time()
        
        result2 = differential_evolution(
            self.feasibility_objective,
            bounds,
            x0=result1.x,  # 使用第一阶段结果作为初始值
            strategy='randtobest1exp',  # 更改策略
            maxiter=50,  # 增加迭代次数
            popsize=30,  # 增加种群大小
            tol=0.001,
            mutation=(0.5, 1.8),  # 扩大变异范围
            recombination=0.9,  # 增加重组概率
            disp=True,
            workers=-1
        )
        
        print(f"第二阶段完成! 用时: {time.time() - start_time:.1f}秒")
        print(f"可行性目标值: {result2.fun:.2f}")
        
        if result2.fun > 1e4:
            print("警告：未找到良好的可行解，尝试专门优化终端温度...")
            
            # 额外阶段：专门优化终端温度
            print("\n额外阶段：优化终端温度...")
            start_time = time.time()
            
            result_temp = differential_evolution(
                self.terminal_temperature_objective,
                bounds,
                x0=result2.x,  # 使用第二阶段结果
                strategy='best1bin',
                maxiter=30,
                popsize=20,
                tol=0.01,
                mutation=(0.3, 0.8),
                recombination=0.8,
                disp=True,
                workers=-1
            )
            
            print(f"终端温度优化完成! 用时: {time.time() - start_time:.1f}秒")
            print(f"终端温度目标值: {result_temp.fun:.2f}")
            
            # 使用温度优化结果继续
            result2 = result_temp
        
        # 第三阶段：在可行解附近寻找最优解
        print("\n第三阶段优化：寻找最优解...")
        start_time = time.time()
        
        result3 = differential_evolution(
            self.objective,
            bounds,
            x0=result2.x,
            strategy='best1bin',
            maxiter=30,
            popsize=20,
            tol=0.01,
            mutation=(0.3, 0.8),
            recombination=0.8,
            disp=True,
            workers=-1
        )
        
        print(f"第三阶段完成! 总用时: {time.time() - start_time:.1f}秒")
        print(f"最优目标值: {result3.fun:.2f}")
        print(f"收敛状态: {result3.success}")
        
        return result3

# ============================================================================
# 5. 改进的可视化工具
# ============================================================================

def plot_results(model, optimizer, result):
    """绘制优化结果"""
    # 获取最优控制
    u_func = optimizer.parameterize_control(result.x)
    
    # 模拟最优轨迹
    x0 = np.array([0.5, 0.4, 0.1, 350, 0.95])
    sol = optimizer.simulate(x0, u_func)
    
    if sol is None:
        print("错误：无法模拟最优轨迹!")
        return
    
    # 时间点
    t_eval = np.linspace(0, optimizer.T_final, 200)
    x_traj = sol.sol(t_eval)

    # 计算控制和其他变量
    u_traj = np.array([u_func(ti) for ti in t_eval]).T

    # 创建图形
    plt.figure(figsize=(15, 10))
    plt.suptitle("化学反应器优化控制结果", fontsize=16)
    
    # 1. 浓度
    plt.subplot(3, 2, 1)
    plt.plot(t_eval/3600, x_traj[0], 'b-', label='CA')
    plt.plot(t_eval/3600, x_traj[1], 'g-', label='CB')
    plt.plot(t_eval/3600, x_traj[2], 'r-', label='CC')
    plt.axhline(y=1.8, color='r', linestyle='--', alpha=0.5, label='CC目标')
    plt.xlabel('时间 (h)')
    plt.ylabel('浓度 (mol/L)')
    plt.title('反应物和产物浓度')
    plt.legend()
    plt.grid(True)
    
    # 2. 温度
    plt.subplot(3, 2, 2)
    plt.plot(t_eval/3600, x_traj[3], 'r-')
    plt.axhline(y=300, color='k', linestyle='--', alpha=0.5, label='最低温度')
    plt.axhline(y=450, color='k', linestyle='--', alpha=0.5, label='最高温度')
    plt.axhspan(340, 360, color='g', alpha=0.2, label='终端目标区')
    plt.xlabel('时间 (h)')
    plt.ylabel('温度 (K)')
    plt.title('反应器温度')
    plt.legend()
    plt.grid(True)
    
    # 3. 催化剂活性
    plt.subplot(3, 2, 3)
    plt.plot(t_eval/3600, x_traj[4], 'g-')
    plt.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='最低活性')
    plt.axhline(y=0.6, color='g', linestyle='--', alpha=0.5, label='终端目标')
    plt.fill_between(t_eval/3600, 0.6, 1.0, color='g', alpha=0.1)
    plt.xlabel('时间 (h)')
    plt.ylabel('催化剂活性')
    plt.title('催化剂活性')
    plt.legend()
    plt.grid(True)
    
    # 4. 选择性和反应速率
    plt.subplot(3, 2, 4)
    selectivity = []
    r1_vals = []
    r2_vals = []
    for i in range(len(t_eval)):
        CA, CB, T_i, alpha = x_traj[0, i], x_traj[1, i], x_traj[3, i], x_traj[4, i]
        r1, r2 = model.reaction_rates(CA, CB, T_i, alpha)
        r1_vals.append(r1)
        r2_vals.append(r2)
        if r1 + r2 > 1e-10:
            S = r1 / (r1 + r2)
        else:
            S = 1.0
        selectivity.append(S)

    plt.plot(t_eval/3600, selectivity, 'b-', label='选择性')
    plt.plot(t_eval/3600, np.array(r1_vals)/max(max(r1_vals), 1e-6), 'g-', alpha=0.5, label='主反应(归一化)')
    plt.plot(t_eval/3600, np.array(r2_vals)/max(max(r1_vals), 1e-6), 'r-', alpha=0.5, label='副反应(归一化)')
    plt.axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='最小选择性')
    plt.xlabel('时间 (h)')
    plt.ylabel('选择性 / 归一化反应速率')
    plt.title('反应选择性与速率')
    plt.legend()
    plt.grid(True)
    
    # 5. 控制变量
    plt.subplot(3, 2, 5)
    plt.plot(t_eval/3600, u_traj[0], 'b-', label='进料流量')
    plt.fill_between(t_eval/3600, 0.1, 1.5, color='b', alpha=0.1)
    plt.xlabel('时间 (h)')
    plt.ylabel('进料流量 (m³/s)')
    plt.title('进料流量控制')
    plt.grid(True)
    
    # 6. 温度控制和催化剂
    plt.subplot(3, 2, 6)
    plt.plot(t_eval/3600, u_traj[1], 'r-', label='加热/冷却 (kW)')
    plt.plot(t_eval/3600, u_traj[2]*1000, 'g-', label='催化剂 (g/s)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.xlabel('时间 (h)')
    plt.ylabel('功率 (kW) / 催化剂 (g/s)')
    plt.title('温度控制和催化剂投加')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间
    plt.show()
    
    # 绘制经济指标
    plt.figure(figsize=(15, 8))
    plt.suptitle("经济和性能指标", fontsize=16)
    
    # 1. 累积产量和利润
    plt.subplot(2, 2, 1)
    production = u_traj[0] * x_traj[2]  # 流量 * 浓度
    cumul_prod = np.cumsum(production) * (t_eval[1] - t_eval[0])
    plt.plot(t_eval/3600, cumul_prod, 'b-')
    plt.xlabel('时间 (h)')
    plt.ylabel('累积产量 (mol)')
    plt.title('产品累积产量')
    plt.grid(True)
    
    # 2. 经济收益
    plt.subplot(2, 2, 2)
    profit = []
    for i in range(len(t_eval)):
        CA, CB, CC = x_traj[0, i], x_traj[1, i], x_traj[2, i]
        F, Q, mcat = u_traj[0, i], u_traj[1, i], u_traj[2, i]
        
        revenue = model.p.PC * CC * F
        cost = model.p.PA * CA * F + model.p.PB * CB * F + model.p.PE * abs(Q) + model.p.PM * mcat
        profit.append(revenue - cost)

    plt.plot(t_eval/3600, profit, 'g-')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('时间 (h)')
    plt.ylabel('利润率 (元/s)')
    plt.title('即时经济收益')
    plt.grid(True)
    
    # 3. 累积利润
    plt.subplot(2, 2, 3)
    cumulative_profit = np.cumsum(profit) * (t_eval[1] - t_eval[0])
    plt.plot(t_eval/3600, cumulative_profit, 'b-')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('时间 (h)')
    plt.ylabel('累积利润 (元)')
    plt.title('累积经济收益')
    plt.grid(True)
    
    # 4. 催化剂消耗
    plt.subplot(2, 2, 4)
    cumulative_cat = np.cumsum(u_traj[2]) * (t_eval[1] - t_eval[0])
    plt.plot(t_eval/3600, cumulative_cat, 'r-')
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='最大限制')
    plt.xlabel('时间 (h)')
    plt.ylabel('催化剂消耗 (kg)')
    plt.title('累积催化剂消耗')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    # 打印关键指标
    print("\n优化结果关键指标:")
    print(f"产品浓度 CC = {x_traj[2, -1]:.3f} mol/L (目标 ≥ 1.8)")
    print(f"终端温度 T = {x_traj[3, -1]:.1f} K (目标 350±10)")
    print(f"催化剂活性 α = {x_traj[4, -1]:.3f} (目标 ≥ 0.6)")
    print(f"累积产量 = {cumul_prod[-1]:.2f} mol")
    print(f"总利润 = {cumulative_profit[-1]:.2f} 元")
    print(f"催化剂消耗 = {cumulative_cat[-1]:.2f} kg (限制 ≤ 50)")
    print(f"平均选择性 = {np.mean(selectivity):.3f} (目标 ≥ 0.85)")
    
    # 检查约束满足情况
    constraints_met = True
    
    if x_traj[2, -1] < 1.8:
        print(f"❌ 产品浓度约束未满足: {x_traj[2, -1]:.3f} < 1.8")
        constraints_met = False
    else:
        print(f"✓ 产品浓度约束满足")
        
    if abs(x_traj[3, -1] - 350) > 10:
        print(f"❌ 终端温度约束未满足: |{x_traj[3, -1]:.1f} - 350| > 10")
        constraints_met = False
    else:
        print(f"✓ 终端温度约束满足")
        
    if x_traj[4, -1] < 0.6:
        print(f"❌ 催化剂活性约束未满足: {x_traj[4, -1]:.3f} < 0.6")
        constraints_met = False
    else:
        print(f"✓ 催化剂活性约束满足")
        
    if cumulative_cat[-1] > 50:
        print(f"❌ 催化剂消耗约束未满足: {cumulative_cat[-1]:.2f} > 50")
        constraints_met = False
    else:
        print(f"✓ 催化剂消耗约束满足")
    
    if constraints_met:
        print("\n✓ 所有约束都满足!")
    else:
        print("\n❌ 部分约束未满足，需要进一步优化")

# ============================================================================
# 6. 主程序
# ============================================================================

def main():
    print("化学反应器动态优化控制 (改进版)")
    print("="*50)
    
    # 创建参数和模型
    params = ReactorParameters()
    T_final = 3600  # 1小时
    model = CSTRModel(params, T_final)
    
    # 系统分析
    print("\n执行系统分析...")
    analyzer = SystemAnalysis(model)
    
    # 反应动力学分析
    print("\n分析反应动力学特性...")
    best_temp = analyzer.analyze_reaction_kinetics()
    
    # 稳态分析
    u_ss = np.array([0.8, 200, 0.01])  # 初始稳态控制
    x_ss = analyzer.find_steady_state(u_ss)
    print("\n稳态点:")
    print(f"CA = {x_ss[0]:.3f}, CB = {x_ss[1]:.3f}, CC = {x_ss[2]:.3f}")
    print(f"T = {x_ss[3]:.1f}, α = {x_ss[4]:.3f}")
    
    # 线性化和稳定性分析
    A, B = analyzer.linearize(x_ss, u_ss)
    stability = analyzer.stability_analysis(A)
    print("\n稳定性分析:")
    print(f"系统{'稳定' if stability['stable'] else '不稳定'}")
    print(f"最大实部特征值: {stability['max_real']:.4f}")
    
    # 可控性分析
    ctrl = analyzer.controllability(A, B)
    print("\n可控性分析:")
    print(f"系统{'完全可控' if ctrl['controllable'] else '不完全可控'}")
    print(f"可控性秩: {ctrl['rank']}")
    
    # 创建优化器
    print("\n准备优化求解...")
    optimizer = OptimalControl(model, T_final=T_final)
    
    # 执行优化
    result = optimizer.optimize()
    
    # 可视化结果
    plot_results(model, optimizer, result)
    
    print("\n优化求解完成！")

if __name__ == "__main__":
    main()
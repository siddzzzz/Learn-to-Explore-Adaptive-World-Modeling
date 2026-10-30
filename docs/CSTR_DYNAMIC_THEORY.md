# Continuous Stirred Tank Reactor (CSTR) Dynamic Control & Safety Theory

## 1. Physical Differential Equations of Continuous CSTR
The continuous stirred tank reactor (CSTR) with an external cooling jacket is a benchmark benchmark in non-linear process control (Seborg et al., 2016; Rawlings et al., 2017).

An irreversible exothermic first-order reaction $A \to B$ occurs inside the reactor volume $V$:
$$\dot{C}_A = \frac{q}{V}(C_{Af} - C_A) - r_A(C_A, T)$$
$$\dot{T} = \frac{q}{V}(T_f - T) + \frac{-\Delta H}{\rho C_p} r_A(C_A, T) + \frac{UA}{V \rho C_p}(T_c - T)$$
$$\dot{T}_c = \frac{q_c}{V_c}(T_{cf} - T_c) + \frac{UA}{V_c \rho_c C_{pc}}(T - T_c)$$

where the reaction rate $r_A$ follows the Arrhenius temperature dependence:
$$r_A(C_A, T) = k_0 \exp\left(-\frac{E_a}{R T}\right) C_A$$

### Nominal Reactor Constants:
- Reactor volume: $V = 100.0\text{ L}$
- Jacket volume: $V_c = 20.0\text{ L}$
- Feed flow rate: $q = 100.0\text{ L/min}$
- Feed concentration: $C_{Af} = 1.0\text{ mol/L}$
- Feed temperature: $T_f = 350.0\text{ K}$
- Coolant feed temperature: $T_{cf} = 300.0\text{ K}$
- Heat capacity ratio: $\rho C_p = 1000.0\text{ J/(L}\cdot\text{K)}$
- Coolant heat capacity: $\rho_c C_{pc} = 1000.0\text{ J/(L}\cdot\text{K)}$
- Heat transfer coefficient: $UA = 20,000.0\text{ J/(min}\cdot\text{K)}$

## 2. Exothermic Non-Linearity & Thermal Runaway
Because the Arrhenius term $\exp(-E_a / (RT))$ is strongly convex in $T$:
$$\frac{\partial^2 r_A}{\partial T^2} > 0 \quad \text{for } T < \frac{E_a}{2R}$$
Any small increase in core temperature $T$ exponentially accelerates heat generation:
$$\dot{Q}_{\text{gen}} = (-\Delta H) V k_0 \exp\left(-\frac{E_a}{RT}\right) C_A$$
while heat removal via the cooling jacket is strictly linear in temperature:
$$\dot{Q}_{\text{rem}} = UA (T - T_c) + q \rho C_p (T - T_f)$$
When $\dot{Q}_{\text{gen}} > \dot{Q}_{\text{rem}}$, the system enters a self-accelerating thermal runaway catastrophe, causing irreversible destruction of the reactor.

## 3. Dynamic Control Barrier Functions (CBFs) for Continuous Trajectories
To guarantee forward invariance of the safe set $\mathcal{C} = \{s \in \mathcal{S} \mid T(s) \le T_{\text{max}}\}$, we define:
$$h(s) = T_{\text{max}} - T$$
Along the continuous vector field:
$$\dot{h}(s) = -\dot{T} = -\left[ \frac{q}{V}(T_f - T) + \frac{(-\Delta H)}{\rho C_p} r_A(C_A, T) + \frac{UA}{V \rho C_p}(T_c - T) \right]$$
The continuous-time barrier condition requires:
$$\dot{h}(s) \ge -\gamma h(s)$$
In discrete-time sampling with step size $\Delta t$:
$$h(s_{t+1}) \ge (1 - \alpha) h(s_t), \quad \alpha \in (0, 1]$$

Under epistemic model ambiguity where $(E_a, \Delta H, k_0)$ are governed by candidate hypotheses $\theta \sim b_t(\theta)$:
$$\text{LCB}_h(s_t, u_t) = \mathbb{E}_{\theta \sim b_t}[h(\hat{s}_{t+1, \theta})] - \sqrt{\frac{1-\delta}{\delta}} \text{Std}_{\theta \sim b_t}[h(\hat{s}_{t+1, \theta})] \ge (1 - \alpha) h(s_t)$$
Ensures that the cooling flow rate $u_t = q_c$ strictly prevents thermal explosion with probability at least $1 - \delta$ while the agent actively experiments to identify the true kinetic constants.

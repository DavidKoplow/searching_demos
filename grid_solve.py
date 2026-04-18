import numpy as np
np.set_printoptions(precision=4, suppress=True)

# States: 0=1A, 1=1B, 2=2A, 3=2B
state_names = ['1A', '1B', '2A', '2B']
R = np.array([1.0, 1.0, -1.0, 1.0])
gamma = 0.9

# Transition matrices (row = from, col = to)
T_N = np.array([
    [0.95, 0.05, 0,    0   ],
    [0.1,  0.2,  0,    0.7 ],
    [0,    0,    0.9,  0.1 ],
    [0,    0.1,  0.1,  0.8 ],
])
T_S = np.array([
    [0.95, 0.05, 0,    0   ],
    [0.1,  0.8,  0,    0.1 ],
    [0,    0,    0.9,  0.1 ],
    [0,    0.7,  0.1,  0.2 ],
])
T_E = np.array([
    [0.65, 0.35, 0,    0   ],
    [0.1,  0.8,  0,    0.1 ],
    [0,    0,    0.3,  0.7 ],
    [0,    0.1,  0.1,  0.8 ],
])
T_W = np.array([
    [0.95, 0.05, 0,    0   ],
    [0.7,  0.2,  0,    0.1 ],
    [0,    0,    0.9,  0.1 ],
    [0,    0.1,  0.7,  0.2 ],
])
T_all = {'N': T_N, 'S': T_S, 'E': T_E, 'W': T_W}

# Verify all rows sum to 1
for name, T in T_all.items():
    sums = T.sum(axis=1)
    assert np.allclose(sums, 1), f"T_{name} row sums: {sums}"

# ============================================================
# Part 1: Passive (uniform random policy)
# ============================================================
T_passive = (T_N + T_S + T_E + T_W) / 4
print("=" * 60)
print("PART 1: Passive Markov Reward Process")
print("=" * 60)
print(f"\nT_passive = (T_N + T_S + T_E + T_W) / 4 =")
print(T_passive)

# Solve linear system: (I - γ T) V = R
A = np.eye(4) - gamma * T_passive
V_passive = np.linalg.solve(A, R)
print(f"\nSolving (I - γT)V = R:")
print(f"  I - γT =")
print(A)
print(f"\n  V_passive:")
for i in range(4):
    print(f"    V({state_names[i]}) = {V_passive[i]:.4f}")

# Verify with value iteration
V = np.zeros(4)
for k in range(2000):
    V_new = R + gamma * (T_passive @ V)
    if np.max(np.abs(V_new - V)) < 1e-12:
        print(f"\n  Value iteration converged at k={k}")
        break
    V = V_new
print(f"  VI check: {V}")

# Show first few iterations
print("\n  First few VI iterations:")
V = np.zeros(4)
for k in range(6):
    print(f"    V_{k} = [{V[0]:7.3f}, {V[1]:7.3f}, {V[2]:7.3f}, {V[3]:7.3f}]")
    V = R + gamma * (T_passive @ V)
print(f"    V_{6} = [{V[0]:7.3f}, {V[1]:7.3f}, {V[2]:7.3f}, {V[3]:7.3f}]")

# ============================================================
# Part 2: MDP (optimal policy)
# ============================================================
print("\n" + "=" * 60)
print("PART 2: MDP Value Iteration")
print("=" * 60)

V = np.zeros(4)
for k in range(2000):
    Q = np.zeros((4, 4))
    for i, a in enumerate(['N', 'S', 'E', 'W']):
        Q[:, i] = R + gamma * (T_all[a] @ V)
    V_new = np.max(Q, axis=1)
    if np.max(np.abs(V_new - V)) < 1e-12:
        print(f"Converged at iteration {k}")
        break
    V = V_new

V_star = V
policy = np.argmax(Q, axis=1)
action_names = ['N', 'S', 'E', 'W']

print(f"\n  V*:")
for i in range(4):
    print(f"    V*({state_names[i]}) = {V_star[i]:.4f}")

print(f"\n  π*:")
for i in range(4):
    print(f"    π*({state_names[i]}) = {action_names[policy[i]]}")

print(f"\n  Q-values at convergence:")
for i, a in enumerate(['N', 'S', 'E', 'W']):
    Q[:, i] = R + gamma * (T_all[a] @ V_star)
print(f"  {'':>6s}  {'N':>8s}  {'S':>8s}  {'E':>8s}  {'W':>8s}")
for s in range(4):
    print(f"  {state_names[s]:>6s}  {Q[s,0]:8.4f}  {Q[s,1]:8.4f}  {Q[s,2]:8.4f}  {Q[s,3]:8.4f}")

# Show first few iterations
print("\n  First few VI iterations:")
V = np.zeros(4)
for k in range(6):
    print(f"    V_{k} = [{V[0]:7.3f}, {V[1]:7.3f}, {V[2]:7.3f}, {V[3]:7.3f}]")
    Q = np.zeros((4, 4))
    for i, a in enumerate(['N', 'S', 'E', 'W']):
        Q[:, i] = R + gamma * (T_all[a] @ V)
    V = np.max(Q, axis=1)
print(f"    V_{6} = [{V[0]:7.3f}, {V[1]:7.3f}, {V[2]:7.3f}, {V[3]:7.3f}]")

# Verify with linear system for the optimal policy
T_pi = np.zeros((4, 4))
for s in range(4):
    T_pi[s] = T_all[action_names[policy[s]]][s]
A_pi = np.eye(4) - gamma * T_pi
V_check = np.linalg.solve(A_pi, R)
print(f"\n  Linear system check: {V_check}")

# ============================================================
# Part 3: POMDP
# ============================================================
print("\n" + "=" * 60)
print("PART 3: POMDP Belief Updates")
print("=" * 60)

# Binary observation: Normal (not trap) / Trap
P_obs = {
    'Normal': np.array([1.0, 1.0, 0.0, 1.0]),
    'Trap':   np.array([0.0, 0.0, 1.0, 0.0]),
}

# Start uniform
b = np.array([0.25, 0.25, 0.25, 0.25])
print(f"\n  b_0 = {b}")

# Observation 1: observe Normal (no transition yet)
b_corr = P_obs['Normal'] * b
eta = b_corr.sum()
b = b_corr / eta
print(f"\n  Observe Normal (no transition):")
print(f"    b_corr = P(Normal|s) * b_0 = {b_corr}")
print(f"    η = {eta:.4f}")
print(f"    b_1 = {b}")

# Passive transition
b_pred = T_passive.T @ b
print(f"\n  Passive transition (T_passive^T @ b_1):")
print(f"    b_pred = {b_pred}")

# Observation 2: observe Normal
b_corr = P_obs['Normal'] * b_pred
eta = b_corr.sum()
b = b_corr / eta
print(f"\n  Observe Normal:")
print(f"    b_corr = P(Normal|s) * b_pred = {b_corr}")
print(f"    η = {eta:.4f}")
print(f"    b_2 = {b}")

# Compute QMDP for each action
print(f"\n  QMDP approximation: Q(b_2, a) = Σ_s b_2(s) [R(s) + γ Σ_s' T(s,a,s') V*(s')]")
print(f"  Using V* from Part 2: {V_star}")
best_a = None
best_q = -np.inf
for a in ['N', 'S', 'E', 'W']:
    q_per_state = R + gamma * (T_all[a] @ V_star)
    q = b @ q_per_state
    print(f"    Q(b_2, {a}) = {q:.4f}   (per-state: {q_per_state})")
    if q > best_q:
        best_q = q
        best_a = a

print(f"\n  Best action: {best_a} with Q = {best_q:.4f}")

# Also show what happens if we compute expected immediate R + γ expected next V*
print(f"\n  Decomposition:")
exp_R = b @ R
print(f"    E[R | b_2] = {exp_R:.4f}")
for a in ['N', 'S', 'E', 'W']:
    exp_V_next = b @ (T_all[a] @ V_star)
    q = exp_R + gamma * exp_V_next
    print(f"    E[V*(s') | b_2, {a}] = {exp_V_next:.4f}  →  Q = {exp_R:.4f} + {gamma}×{exp_V_next:.4f} = {q:.4f}")

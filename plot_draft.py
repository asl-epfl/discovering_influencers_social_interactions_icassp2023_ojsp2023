import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import networkx as nx
import seaborn as sns
from matplotlib import cm
import matplotlib as mpl


colors = sns.color_palette("tab10")
colors_dark = sns.color_palette("dark")
font = {'size': 17}
matplotlib.rc('font', **font)

def running_mean(x, window_size):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

# plot 1

plt.figure(figsize=(10, 6))
error_true = np.load('draft/error_21_graph_known_L.npy')
error_w1 = np.load('draft/error_21_graph_w1.npy')[:40000]
error_w10 = np.load('draft/error_21_graph_w10.npy')
error_w50 = np.load('draft/error_21_graph_w50.npy')

plt.plot(list(range(error_true.shape[0])), error_true,
         color=colors[2], label='known expectation', alpha=0.7, linewidth=6.)
plt.plot(list(range(error_w1.shape[0])), error_w1,
         color=colors[1], label='$M = 1$', alpha=1, linewidth=2)
plt.plot(list(range(error_w10.shape[0])), error_w10,
         color=colors[0], label='$M = 10$', alpha=1, linewidth=2)
plt.plot(list(range(error_w50.shape[0])), error_w50,
         color=colors_dark[4], label='$M = 50$', alpha=1, linewidth=2)
plt.xlim(-200, 40000)
plt.yscale('log')
plt.legend()
plt.xlabel('Time')
plt.ylabel('$\|\|\widetilde{A}_i\|\|_{F}^2$')
plt.grid()
# plt.show()
plt.savefig('draft/graph_21_error.png', dpi=300, bbox_inches='tight')


# plot 1 adapt

plt.figure(figsize=(10, 6))
error_w50_p = np.load('draft/error_21_graph_w50_perturbe.npy')
time_p = 10000
eps = 1
# error_w100 = np.load('draft/error_graph_w100.npy')

plt.vlines(time_p, ymin=0 - eps/2, ymax=np.array(error_w50_p).max() + eps,
             color=colors[1], label='time when graph changes', linewidth=4, alpha=0.5)
plt.plot(list(range(error_w50_p.shape[0])), error_w50_p,
         color=colors_dark[4], label='$M = 50$', alpha=1, linewidth=2)
plt.legend()
plt.xlabel('Time')
plt.ylabel('$\|\|\widetilde{A}_i\|\|_{F}$')
plt.grid()
# plt.show()
plt.savefig('draft/graph_21_error_perturbe.png', dpi=300, bbox_inches='tight')

# plot 2

plt.figure(figsize=(10, 6))
t1 = 10
t2 = 10000

error_L_w2 = np.load('draft/error_21_L_w1.npy')#[t1:t2]
error_L_w10 = np.load('draft/error_21_L_w10.npy')#[t1:t2]
error_L_w50 = np.load('draft/error_21_L_w50.npy')#[t1:t2]

error_L_w2 = running_mean(error_L_w2, 50)[t1:t2]
error_L_w10 = running_mean(error_L_w10, 50)[t1:t2]
error_L_w50 = running_mean(error_L_w50, 50)[t1:t2]

plt.plot(list(range(t1, t2)), error_L_w2,
         color=colors[1], label='$M = 1$', alpha=1, linewidth=2)
plt.plot(list(range(t1, t2)), error_L_w10,
         color=colors[0], label='$M = 10$', alpha=1, linewidth=2)
plt.plot(list(range(t1, t2)), error_L_w50,
         color=colors_dark[4], label='$M = 50$', alpha=1, linewidth=2)
plt.legend()
plt.xlabel('Time')
plt.ylabel('$\|\|\widetilde{L}_i\|\|_{F}^2$')
plt.grid()
plt.xlim(0, t2)
plt.yscale('log')
# plt.show()
plt.savefig('draft/L_21_error_.png', dpi=300, bbox_inches='tight')

# plot 3

matrix_true = np.load('draft/matrix_21_true.npy')
matrix_learned = np.load('draft/matrix_21_w50.npy')
matrix_learned[matrix_learned < 0.] = 0.

cmap = sns.color_palette("mako", as_cmap=True)
vmin = min(matrix_learned.min(), matrix_true.min())
vmax = max(matrix_learned.max(), matrix_true.max())

plt.figure()
sns.heatmap(matrix_true, yticklabels=False, xticklabels=False, cbar=True,
            vmin=vmin, cmap=cmap)
# plt.show()
plt.savefig('draft/adj_21_true.png', dpi=300, bbox_inches='tight')

plt.figure()
sns.heatmap(matrix_learned, yticklabels=False, xticklabels=False, cbar=True,
            vmin=vmin, cmap=cmap)
# plt.show()
plt.savefig('draft/adj_21_w50.png', dpi=300, bbox_inches='tight')

# plot 4

plt.figure(figsize=(10, 4))
influences_true = np.load('draft/influences_21_true_1inf.npy')
influences_w50 = np.load('draft/influences_21_w50_1inf.npy')
influences_true = influences_true / influences_true.sum()
influences_w50 = influences_w50 / influences_w50.sum()
window = 50

agents = influences_w50.shape[0]
plt.scatter(np.arange(agents), influences_true, s=150, color=colors[2], alpha=0.8,
            label="true")
plt.vlines(np.arange(agents), 0, influences_true, color=colors[2], alpha=0.8, )
plt.scatter(np.arange(agents), influences_w50, s=150, color=colors_dark[4], alpha=0.8,
            label="$M = " + str(window) + '$')
plt.vlines(np.arange(agents), 0, influences_w50, color=colors_dark[4], alpha=0.8, )
plt.xticks([0,5,10,15,20])
plt.grid()
plt.xlabel('agent')
plt.ylabel('influence')
plt.legend()
# plt.show()
plt.savefig('draft/influences_21_w50_big_1inf.png', dpi=300, bbox_inches='tight')

font = {'size': 9}
matplotlib.rc('font', **font)

plt.figure(figsize=(10, 4))
influences_true = np.load('draft/influences_21_true_1inf.npy')
influences_w50 = np.load('draft/influences_21_w50_1inf.npy')
influences_true = influences_true / influences_true.sum()
influences_w50 = influences_w50 / influences_w50.sum()
window = 50

agents = influences_w50.shape[0]
plt.scatter(np.arange(agents), influences_true, s=150, color=colors[2], alpha=0.8,
            label="true")
plt.vlines(np.arange(agents), 0, influences_true, color=colors[2], alpha=0.8, )
plt.scatter(np.arange(agents), influences_w50, s=150, color=colors_dark[4], alpha=0.8,
            label="$M = " + str(window) + '$')
plt.vlines(np.arange(agents), 0, influences_w50, color=colors_dark[4], alpha=0.8, )
plt.xticks([0,5,10,15,20])
plt.grid()
plt.xlabel('Agent')
plt.ylabel('Influence')
plt.legend()
# plt.show()
plt.savefig('draft/influences_21_w50_1inf.png', dpi=300, bbox_inches='tight')

font = {'size': 17}
matplotlib.rc('font', **font)

# plot 5

np.random.seed(33)
matrix_true = np.load('draft/matrix_21.npy')
matrix_learned = np.load('draft/matrix_21_w50.npy')
matrix_learned[matrix_learned < 0.] = 0.

combination_matrix = matrix_true + matrix_true.T
combination_matrix_learned = matrix_learned + matrix_learned.T
adj_matrix = matrix_true + matrix_true.T
adj_matrix[adj_matrix > 0] = 1

pos = nx.spring_layout(nx.from_numpy_matrix(adj_matrix))
G = nx.from_numpy_matrix(combination_matrix)
edges = G.edges()
weights = [G[u][v]['weight']*5 for u, v in edges]

G_learned = nx.from_numpy_matrix(combination_matrix_learned)
edges_learned = G_learned.edges()
weights_learned = [G_learned[u][v]['weight']*5 for u, v in edges_learned]

plt.figure(figsize=(8, 7))
nx.draw_networkx_nodes(G, pos=pos, node_color=colors[4], node_size=350)
nx.draw_networkx_edges(G, pos=pos, edge_color='black', width=weights)
# plt.show()
plt.savefig('draft/graph_21.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(8, 7))
nx.draw_networkx_nodes(G_learned, pos=pos, node_color=colors[4], node_size=350)
nx.draw_networkx_edges(G_learned, pos=pos, edge_color='black', width=weights_learned)
# plt.show()
plt.savefig('draft/graph_21_w50.png', dpi=300, bbox_inches='tight')

# plot 6

np.random.seed(33)
state_est_0 = np.load('draft/state_true_rate_k_21_0inf.npy')
state_est_1 = np.load('draft/state_true_rate_k_21_1inf.npy')
state_est_3 = np.load('draft/state_true_rate_k_21_3inf.npy')
state_est_1_less = np.load('draft/state_true_rate_k_21_1inf_less.npy')
state_est_3_less = np.load('draft/state_true_rate_k_21_3inf_less.npy')

plt.figure(figsize=(10, 6))
t1 = 30000
t2 = 40000

plt.plot(list(range(t1, t2)), state_est_3[t1:t2],
         color=colors[1], label='$3$ influential agents', alpha=1, linewidth=2)
plt.plot(list(range(t1, t2)), state_est_1[t1:t2],
         color=colors[0], label='$1$ influential agent', alpha=1, linewidth=2)
plt.plot(list(range(t1, t2)), state_est_3_less[t1:t2],
         color=colors_dark[-1], label='$3$ (less) influential agents', alpha=1, linewidth=2)
plt.plot(list(range(t1, t2)), state_est_1_less[t1:t2],
         color=colors_dark[-4], label='$1$ (less) influential agent', alpha=1, linewidth=2)
plt.plot(list(range(t1, t2)), state_est_0[t1:t2],
         color=colors_dark[4], label='$0$ influential agents', alpha=1, linewidth=2)
plt.xlim(30000, 40000)
font = {'size': 14}
matplotlib.rc('font', **font)
plt.legend()
font = {'size': 17}
matplotlib.rc('font', **font)
plt.xlabel('Time')
plt.ylabel('$r_i$')
plt.grid()
# plt.show()
plt.savefig('draft/state_estimation2.png', dpi=300, bbox_inches='tight')

# plot 7

plt.figure(figsize=(10, 6))
error_w50_p = np.load('draft/error_21_graph_w50_perturbe_slow_001.npy')
# error_w50_p2 = np.load('draft/error_21_graph_w50_perturbe_slow_001_2.npy')
error_w50_p3 = np.load('draft/error_21_graph_w50_perturbe_slow_001_3.npy')
# error_w50_p4 = np.load('draft/error_21_graph_w50_perturbe_slow_001_4.npy')

eps = 1
# error_w100 = np.load('draft/error_graph_w100.npy')

# plt.vlines(time_p, ymin=0 - eps/2, ymax=np.array(error_w50_p).max() + eps,
#              color=colors[1], label='time when graph changes', linewidth=4, alpha=0.5)
# plt.plot(list(range(error_w50_p2.shape[0])), error_w50_p2,
#          color=colors_dark[-1], label='$\mu = 0.05$', alpha=1, linewidth=2)
plt.plot(list(range(error_w50_p.shape[0])), error_w50_p,
         color=colors_dark[4], label='$\mu = 0.1$', alpha=1, linewidth=2)
plt.plot(list(range(error_w50_p3.shape[0])), error_w50_p3,
         color=colors_dark[-4], label='$\mu = 0.2$', alpha=1, linewidth=2)
# plt.plot(list(range(error_w50_p4.shape[0])), error_w50_p4,
#          color=colors_dark[-3], label='$\mu = 0.4$', alpha=1, linewidth=2)
plt.yscale('log')
plt.legend()
plt.xlabel('Time')
plt.ylabel('$\|\|\widetilde{A}_i\|\|_{F}$')
plt.grid()
# plt.show()
plt.savefig('draft/graph_21_error_perturbe_001.png', dpi=300, bbox_inches='tight')


# plot 8

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
error_w50_p = np.load('draft/error_21_graph_w50_perturbe_state.npy')
time_p = 20000
eps = 1
# error_w100 = np.load('draft/error_graph_w100.npy')

ax.vlines(time_p, ymin=0 - eps/2, ymax=np.array(error_w50_p).max() + eps,
             color=colors[1], label='time when state changes', linewidth=4, alpha=0.5)
ax.plot(list(range(error_w50_p.shape[0])), error_w50_p,
         color=colors_dark[4], label='$M = 50$', alpha=1, linewidth=2)
ax.set_yscale('log')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('$\|\|\widetilde{A}_i\|\|_{F}$')
ax.grid()


font = {'size': 14}
matplotlib.rc('font', **font)
axins = ax.inset_axes([0.53, 0.35, 0.4, 0.4]) #[x0, y0, width, height]
# axins.set_yscale('log')
x1, x2, y1, y2 = time_p-100, time_p+100, 0.695, 0.712
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.plot(error_w50_p, color=colors_dark[4], linewidth=2)
axins.vlines(time_p, ymin=0 - eps/2, ymax=np.array(error_w50_p).max() + eps,
             color=colors[1], linewidth=4, alpha=0.5)

font = {'size': 17}
matplotlib.rc('font', **font)
# plt.show()
plt.savefig('draft/graph_21_error_perturbe_state.png', dpi=300, bbox_inches='tight')
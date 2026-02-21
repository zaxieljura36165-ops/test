"""
CA-HMA-TD3 论文风格架构图生成（参考AR-MAD4PG版式）
仅绘图，不修改算法实现代码。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False


def draw_box(ax, x, y, w, h, text, fc, ec, fs=12, bold=False, lw=1.8, alpha=0.95, radius=0.08):
        box = FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0.02,rounding_size={radius}",
                facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha
        )
        ax.add_patch(box)
        ax.text(
                x + w / 2, y + h / 2, text,
                ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal', linespacing=1.35
        )
        return box


def draw_arrow(ax, x1, y1, x2, y2, color='#2f2f2f', lw=1.9, dashed=False, cs=None):
        props = dict(
                arrowstyle='-|>,head_width=0.18,head_length=0.22',
                color=color,
                lw=lw,
                linestyle='--' if dashed else '-',
                shrinkA=2,
                shrinkB=2,
        )
        if cs is not None:
                props['connectionstyle'] = cs
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=props)


def draw_cylinder(ax, x, y, w, h, body_fc, edge, fs=12):
        body = FancyBboxPatch(
                (x, y), w, h,
                boxstyle='round,pad=0.01,rounding_size=0.12',
                facecolor=body_fc, edgecolor=edge, linewidth=1.9, alpha=0.96
        )
        ax.add_patch(body)
        top = Ellipse((x + w / 2, y + h), w * 0.98, h * 0.25, facecolor='#a8c0e8', edgecolor=edge, linewidth=1.5)
        ax.add_patch(top)
        ax.text(x + w / 2, y + h * 0.67, 'Replay Buffer', ha='center', va='center', fontsize=fs, fontweight='bold')
        ax.text(x + w / 2, y + h * 0.32, r'$(o^t, a^t, r^t, o^{t+1}, done)$', ha='center', va='center', fontsize=fs - 1)


fig, ax = plt.subplots(figsize=(14.5, 19.5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 22)
ax.axis('off')

# 色板（偏论文风）
C_INPUT = '#eceff4'
C_AGENT = '#f6f7fb'
C_LEARN = '#f7dede'
C_CORE = '#dcd8ee'
C_POLICY = '#efedf8'
C_NOISE = '#ece0f8'
C_ENV = '#b8d8d8'
C_OUT = '#eceff4'

# 1) Input
ax.text(0.9, 20.95, 'Input:', fontsize=20, fontweight='bold', va='center')
draw_box(
        ax, 3.2, 20.35, 12.0, 1.0,
        r'Local System Observation  $o_i^t$' + '\n' +
        'CA-HMA-TD3 Model Hyperparameters',
        C_INPUT, '#6d7690', fs=18, bold=True, lw=1.8, radius=0.03
)

# 2) Learner + Replay Buffer（上层）
draw_box(ax, 1.8, 16.95, 8.0, 2.25, 'Learner', C_LEARN, '#9a677a', fs=20, bold=True, lw=1.8, radius=0.10)
draw_box(ax, 2.25, 17.55, 2.6, 0.85, 'Target Policy', '#fbeeee', '#c08ea0', fs=15, bold=True, lw=1.2)
draw_box(ax, 5.10, 17.55, 2.6, 0.85, 'Target Critic', '#fbeeee', '#c08ea0', fs=15, bold=True, lw=1.2)
ax.text(2.1, 16.3, r'$\tau=0.005$ soft update  |  Twin-Q  |  policy delay', fontsize=12)

draw_cylinder(ax, 10.2, 17.0, 4.8, 2.0, '#9fb7dc', '#5a6f96', fs=16)

# 3) Agent 主体（叠层风格，贴近论文图）
for dx, dy in [(0.58, -0.56), (0.30, -0.28)]:
        ax.add_patch(FancyBboxPatch(
                (1.25 + dx, 5.8 + dy), 13.6, 10.1,
                boxstyle='round,pad=0.02,rounding_size=0.10',
                facecolor=C_AGENT, edgecolor='#7b82a1', linewidth=1.2, alpha=0.9
        ))

draw_box(ax, 1.25, 5.8, 13.6, 10.1, 'Hierarchical TD3 Agent', C_AGENT, '#6e7699', fs=16, bold=True, lw=1.8, radius=0.10)

# 内核区域
draw_box(ax, 2.1, 7.75, 11.9, 7.15, '', C_CORE, '#7077a1', fs=12, lw=1.4, radius=0.08)
ax.text(8.05, 14.25, 'Update parameters', fontsize=16, fontweight='bold', ha='center')

# 高层模块
draw_box(
        ax, 2.55, 12.25, 7.9, 1.6,
        'High-Level TD3 Actor/Critic\n'
        r'$\alpha,\;mode\_{logits}(V2I/V2V),\;p_{base}$  (200→256→256→128→4)',
        C_POLICY, '#6d6a9a', fs=13, bold=True, lw=1.2
)

# 路由逻辑
draw_box(
        ax, 2.55, 11.2, 7.9, 0.85,
        r'Routing: $\alpha<0.01\to$Local, else mode=0$\to$V2I, mode=1$\to$V2V',
        '#f5f2da', '#9d8b2f', fs=12, bold=True, lw=1.1
)

# 低层并列
draw_box(ax, 2.55, 9.3, 2.45, 1.55, 'V2I TD3\nActor+Critic\nout: 6', '#e8f3ff', '#4177b1', fs=12, bold=True, lw=1.1)
draw_box(ax, 5.25, 9.3, 2.45, 1.55, 'V2V TD3\nActor+Critic\nout: 6', '#e8f3ff', '#4177b1', fs=12, bold=True, lw=1.1)
draw_box(ax, 7.95, 9.3, 2.45, 1.55, 'Local TD3\nActor+Critic\nout: 1', '#e8f3ff', '#4177b1', fs=12, bold=True, lw=1.1)

# 噪声控制器
draw_box(
        ax, 10.95, 9.25, 2.65, 4.6,
        'Context-Aware\nAdaptive Noise\nController\n\n'
        '5 factors\n8 contexts\n3 phases\n'
        r'$\sigma=\sigma_{global}\cdot f_{context}$',
        C_NOISE, '#7b54a5', fs=12, bold=True, lw=1.2
)

# 中间动作集合
draw_box(ax, 4.2, 8.2, 4.5, 0.78, r'Combined Action  $a_i^t$', '#f9f5ff', '#6d6a9a', fs=14, bold=True, lw=1.1)

# 右侧 store 立柱
draw_box(ax, 11.05, 8.05, 2.25, 1.0, 'Store', '#f5f5f7', '#666b84', fs=15, bold=True, lw=1.0)
draw_box(ax, 11.3, 7.0, 1.75, 0.72, r'$h_i^{t+1}$', '#ffffff', '#7c7f9f', fs=14, bold=True, lw=1.0, radius=0.2)
draw_box(ax, 11.3, 6.15, 1.75, 0.72, r'$o_i^{t+1}$', '#ffffff', '#7c7f9f', fs=14, bold=True, lw=1.0, radius=0.2)
draw_box(ax, 11.3, 5.3, 1.75, 0.72, r'$r_i^t$', '#ffffff', '#7c7f9f', fs=14, bold=True, lw=1.0, radius=0.2)

# 环境条
draw_box(ax, 2.1, 4.95, 11.9, 1.0, 'V2X / VEC Environment', C_ENV, '#568585', fs=18, bold=True, lw=1.3, radius=0.08)

# 左侧输入小圆角节点
draw_box(ax, 0.55, 11.35, 1.35, 0.78, r'$o_i^t$', '#ffffff', '#7c7f9f', fs=16, bold=True, lw=1.1, radius=0.25)
draw_box(ax, 0.55, 10.35, 1.35, 0.78, r'$h_i^t$', '#ffffff', '#7c7f9f', fs=16, bold=True, lw=1.1, radius=0.25)

# 关键连线
draw_arrow(ax, 8.0, 20.35, 8.0, 18.95, color='#3b3f52', lw=2.0)   # input to learner
draw_arrow(ax, 7.25, 16.95, 7.25, 14.9, color='#3b3f52', lw=1.9)    # learner to update

draw_arrow(ax, 9.7, 18.1, 10.2, 18.1, color='#3b3f52', lw=1.9)       # learner to buffer
draw_arrow(ax, 11.7, 17.0, 11.7, 9.05, color='#3b3f52', lw=1.9)      # buffer to store

draw_arrow(ax, 1.9, 11.75, 2.55, 13.05, color='#3b3f52', lw=1.8)     # o_t to high-level
draw_arrow(ax, 1.9, 10.75, 2.55, 10.1, color='#3b3f52', lw=1.8)      # h_t to low-level

draw_arrow(ax, 6.5, 12.25, 6.5, 11.2, color='#3b3f52', lw=1.6)       # high to routing
draw_arrow(ax, 6.5, 11.2, 3.8, 10.85, color='#3b3f52', lw=1.6)       # route to v2i
draw_arrow(ax, 6.5, 11.2, 6.5, 10.85, color='#3b3f52', lw=1.6)       # route to v2v
draw_arrow(ax, 6.5, 11.2, 9.2, 10.85, color='#3b3f52', lw=1.6)       # route to local

draw_arrow(ax, 11.0, 11.8, 10.4, 10.2, color='#7b54a5', lw=1.5, dashed=True)  # noise to low-level
draw_arrow(ax, 11.0, 12.7, 10.4, 13.0, color='#7b54a5', lw=1.5, dashed=True)   # noise to high-level

draw_arrow(ax, 3.8, 9.3, 5.2, 8.95, color='#3b3f52', lw=1.5)
draw_arrow(ax, 6.5, 9.3, 6.5, 8.98, color='#3b3f52', lw=1.5)
draw_arrow(ax, 9.2, 9.3, 7.7, 8.95, color='#3b3f52', lw=1.5)

draw_arrow(ax, 6.45, 8.2, 6.45, 5.95, color='#3b3f52', lw=2.0)        # action to env
draw_arrow(ax, 11.0, 5.95, 11.0, 8.05, color='#3b3f52', lw=1.8)        # env to store
draw_arrow(ax, 12.2, 9.05, 12.2, 17.0, color='#3b3f52', lw=1.8)        # store to buffer

draw_arrow(ax, 10.2, 17.8, 7.9, 18.1, color='#3b3f52', lw=1.6, cs='arc3,rad=0.35')  # sampled batch to learner
ax.text(8.25, 18.55, 'sample (B_E/B_D)', fontsize=11, style='italic')

# 输出
ax.text(0.9, 1.95, 'Output:', fontsize=20, fontweight='bold', va='center')
draw_box(
        ax, 3.2, 1.35, 12.0, 1.0,
        r'Task Offloading Decision  $a_i^t$' + '\n' +
        r'$\{\alpha, mode, rsu\_action/neighbor\_action, power, freq\}$',
        C_OUT, '#6d7690', fs=18, bold=True, lw=1.8, radius=0.03
)

# 图题
ax.text(8.0, 0.55, 'Fig. 3.  Architecture of the CA-HMA-TD3 model.',
                ha='center', va='center', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('d:/test/test/results/plots/CA_HMA_TD3_Architecture.png',
                        dpi=260, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('d:/test/test/results/plots/CA_HMA_TD3_Architecture.jpg',
                        dpi=260, bbox_inches='tight', facecolor='white', edgecolor='none')

print('图片已保存:')
print('  PNG: d:/test/test/results/plots/CA_HMA_TD3_Architecture.png')
print('  JPG: d:/test/test/results/plots/CA_HMA_TD3_Architecture.jpg')

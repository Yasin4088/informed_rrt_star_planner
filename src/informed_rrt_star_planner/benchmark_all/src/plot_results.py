#!/usr/bin/env python3
"""
Full Pipeline Benchmark Visualization
PointCloud → Informed RRT* → B-Spline → L-BFGS → Final Trajectory
Generates publication-quality figures comparing Classical vs Enhanced.
"""

import csv
import sys
import os
import subprocess
import numpy as np
from collections import defaultdict
import glob

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed to register 3D projection
    from matplotlib import font_manager as fm
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Run: pip3 install matplotlib numpy")
    print("Generating ASCII-only summary...\n")


SCENARIO_CN_LABELS = {
    '1_Sparse_Pillars': '场景1：稀疏立柱',
    '2_Dense_Clutter': '场景2：密集障碍',
    '3_Maze': '场景3：迷宫通道',
    '4_Long_Range': '场景4：长距离规划',
    '5_Multi_Level': '场景5：多层空间',
    '6_Forest': '场景6：森林场景',
    '7_Urban_Canyon': '场景7：城市峡谷',
    '8_Random_Field': '场景8：随机场景',
}

SCENARIO_CN_SHORT_LABELS = {
    '1_Sparse_Pillars': '场景1\n稀疏立柱',
    '2_Dense_Clutter': '场景2\n密集障碍',
    '3_Maze': '场景3\n迷宫通道',
    '4_Long_Range': '场景4\n长距离',
    '5_Multi_Level': '场景5\n多层空间',
    '6_Forest': '场景6\n森林',
    '7_Urban_Canyon': '场景7\n城市峡谷',
    '8_Random_Field': '场景8\n随机场景',
}


def scenario_name_cn(scenario, short=False):
    if short:
        return SCENARIO_CN_SHORT_LABELS.get(scenario, scenario)
    return SCENARIO_CN_LABELS.get(scenario, scenario)


def configure_cjk_font():
    """
    Configure matplotlib to use a Chinese-capable font when available.
    Returns True when Chinese font was configured successfully.
    """
    if not HAS_MPL:
        return False

    has_addfont = hasattr(fm.fontManager, 'addfont')

    def can_render_zh(font_file):
        """Return True if font file contains common Chinese glyphs."""
        try:
            from matplotlib.ft2font import FT2Font
            cmap = FT2Font(font_file).get_charmap()
            for ch in ('场', '景', '障', '碍'):
                if ord(ch) not in cmap:
                    return False
            return True
        except Exception:
            # On very old matplotlib, probing can fail; fall back to trusting file.
            return True

    # 1) Prefer known CJK font files on Linux workstations.
    preferred_files = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        '/usr/share/fonts/truetype/arphic/ukai.ttc',
    ]
    for font_file in preferred_files:
        if not os.path.exists(font_file):
            continue
        if not can_render_zh(font_file):
            continue
        try:
            if has_addfont:
                fm.fontManager.addfont(font_file)
            real_name = fm.FontProperties(fname=font_file).get_name()
            if not real_name:
                continue
            plt.rcParams['font.sans-serif'] = [real_name, 'DejaVu Sans']
            plt.rcParams['font.family'] = 'sans-serif'
            return True
        except Exception:
            continue

    # 2) Scan fontconfig result for Chinese-supporting font files.
    try:
        out = subprocess.check_output(
            ['fc-list', ':lang=zh', '-f', '%{file}\\n'],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        seen = set()
        for line in out.splitlines():
            font_file = line.strip()
            if not font_file or font_file in seen:
                continue
            seen.add(font_file)
            if not font_file or not os.path.exists(font_file):
                continue
            if not can_render_zh(font_file):
                continue
            try:
                if has_addfont:
                    fm.fontManager.addfont(font_file)
                real_name = fm.FontProperties(fname=font_file).get_name()
                if not real_name:
                    continue
                plt.rcParams['font.sans-serif'] = [real_name, 'DejaVu Sans']
                plt.rcParams['font.family'] = 'sans-serif'
                return True
            except Exception:
                continue
    except Exception:
        pass

    # 3) Fallback: try known family names via fc-match.
    cjk_candidates = [
        'Noto Sans CJK SC',
        'Noto Serif CJK SC',
        'WenQuanYi Zen Hei',
        'AR PL UMing CN',
        'SimHei',
        'Microsoft YaHei',
    ]
    for font_name in cjk_candidates:
        try:
            out = subprocess.check_output(
                ['fc-match', '-f', '%{file}\\n', font_name],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if out and os.path.exists(out) and can_render_zh(out):
                if has_addfont:
                    fm.fontManager.addfont(out)
                real_name = fm.FontProperties(fname=out).get_name()
                if not real_name:
                    continue
                plt.rcParams['font.sans-serif'] = [real_name, 'DejaVu Sans']
                plt.rcParams['font.family'] = 'sans-serif'
                return True
        except Exception:
            pass

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    return False


def load_csv(path):
    """Load pipeline benchmark CSV and return structured data."""
    data = defaultdict(lambda: {'Classical': {}, 'Enhanced': {}})
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rrt_time_key = 'RRT_FirstSolutionTime(ms)' if 'RRT_FirstSolutionTime(ms)' in row else 'RRT_Time(ms)'
            scenario = row['Scenario']
            algo = row['Algorithm']
            data[scenario][algo] = {
                'success_rate': float(row['SuccessRate(%)']),
                'rrt_time': float(row[rrt_time_key]),
                'rrt_std_time': float(row['RRT_StdTime(ms)']),
                'rrt_path_len': float(row['RRT_PathLen(m)']),
                'rrt_std_path_len': float(row['RRT_StdPathLen(m)']),
                'rrt_nodes': float(row['RRT_Nodes']),
                'rrt_clearance': float(row['RRT_Clearance(m)']),
                'bspline_ctrl_pts': float(row['BSpline_CtrlPts']),
                'bspline_conv_time': float(row['BSpline_ConvTime(ms)']),
                'lbfgs_time': float(row['LBFGS_Time(ms)']),
                'lbfgs_std_time': float(row['LBFGS_StdTime(ms)']),
                'lbfgs_iters': float(row['LBFGS_Iters']),
                'lbfgs_cost_reduction': float(row['LBFGS_CostReduction(%)']),
                'lbfgs_rebounds': float(row['LBFGS_Rebounds']),
                'traj_length': float(row['Traj_Length(m)']),
                'traj_std_length': float(row['Traj_StdLength(m)']),
                'traj_clearance': float(row['Traj_Clearance(m)']),
                'traj_smoothness': float(row['Traj_Smoothness(rad)']),
                'traj_max_vel': float(row['Traj_MaxVel(m/s)']),
                'traj_max_acc': float(row['Traj_MaxAcc(m/s2)']),
                'total_time': float(row['Total_Time(ms)']),
            }
    return data


def load_trial_data(search_dir):
    """Load per-trial raw metrics from trial_metrics_*.csv files."""
    trials = defaultdict(lambda: {'Classical': defaultdict(list), 'Enhanced': defaultdict(list)})
    pattern = os.path.join(search_dir, 'trial_metrics_*.csv')
    files = sorted(glob.glob(pattern))
    for fp in files:
        with open(fp, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario = row['Scenario']
                algo = row['Algorithm']
                for k, v in row.items():
                    if k in ('Scenario', 'Algorithm'):
                        continue
                    try:
                        trials[scenario][algo][k].append(float(v))
                    except Exception:
                        pass
    return trials, files


def mean_ci(values, z=1.96):
    """Return mean and normal-approx 95% CI half width."""
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return mean, z * se


def paired_improvement_ci(classical_vals, enhanced_vals, eps=1e-9):
    """
    Compute paired percentage improvement and CI.
    Positive means enhanced is better for lower-is-better metrics.
    """
    c = np.array(classical_vals, dtype=float)
    e = np.array(enhanced_vals, dtype=float)
    n = min(c.size, e.size)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    c = c[:n]
    e = e[:n]
    valid = np.abs(c) > eps
    c = c[valid]
    e = e[valid]
    if c.size == 0:
        return 0.0, 0.0, 0.0, 0
    imp = (c - e) / c * 100.0
    m = float(np.mean(imp))
    ci = 1.96 * float(np.std(imp, ddof=1) / np.sqrt(imp.size)) if imp.size > 1 else 0.0
    return m, m - ci, m + ci, int(imp.size)


def sign_test_pvalue(classical_vals, enhanced_vals, lower_better=True):
    """Approximate paired sign-test p-value without scipy."""
    c = np.array(classical_vals, dtype=float)
    e = np.array(enhanced_vals, dtype=float)
    n = min(c.size, e.size)
    if n == 0:
        return 1.0
    c = c[:n]
    e = e[:n]
    if lower_better:
        wins = int(np.sum(e < c))
        losses = int(np.sum(e > c))
    else:
        wins = int(np.sum(e > c))
        losses = int(np.sum(e < c))
    total = wins + losses
    if total == 0:
        return 1.0
    k = min(wins, losses)
    # Two-sided exact binomial with p=0.5
    p = 0.0
    for i in range(0, k + 1):
        p += np.math.comb(total, i) * (0.5 ** total)
    return min(1.0, 2.0 * p)


def ascii_summary(data):
    """Print ASCII summary table."""
    print("\n" + "=" * 100)
    print("  FULL PIPELINE BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    metrics = [
        ('Success Rate (%)', 'success_rate'),
        ('RRT Time (ms)', 'rrt_time'),
        ('RRT Path Len (m)', 'rrt_path_len'),
        ('RRT Nodes', 'rrt_nodes'),
        ('RRT Clearance (m)', 'rrt_clearance'),
        ('L-BFGS Time (ms)', 'lbfgs_time'),
        ('L-BFGS Iters', 'lbfgs_iters'),
        ('L-BFGS Cost Reduc (%)', 'lbfgs_cost_reduction'),
        ('Traj Length (m)', 'traj_length'),
        ('Traj Clearance (m)', 'traj_clearance'),
        ('Traj Smoothness', 'traj_smoothness'),
        ('Total Time (ms)', 'total_time'),
    ]

    for scenario in sorted(data.keys()):
        print(f"\n  [{scenario}]")
        c = data[scenario].get('Classical', {})
        e = data[scenario].get('Enhanced', {})
        if not c or not e:
            continue
        print(f"  {'Metric':<30} {'Classical':>12} {'Enhanced':>12} {'Improve':>10}")
        print(f"  {'-'*64}")
        for name, key in metrics:
            cv = c.get(key, 0)
            ev = e.get(key, 0)
            if cv > 0 and key != 'success_rate':
                imp = (cv - ev) / cv * 100
                imp_str = f"{imp:+.0f}%"
            else:
                imp_str = "N/A"
            if isinstance(cv, float):
                print(f"  {name:<30} {cv:>12.2f} {ev:>12.2f} {imp_str:>10}")
            else:
                print(f"  {name:<30} {cv:>12} {ev:>12} {imp_str:>10}")

    print("\n" + "=" * 100)


def plot_comparison(data, output_dir='.'):
    """Generate all comparison plots."""
    if not HAS_MPL:
        ascii_summary(data)
        return

    os.makedirs(output_dir, exist_ok=True)
    legacy_files = [
        'fig2_safety_quality.png',
        'fig3_stage_breakdown.png',
        'fig4_radar.png',
        'fig5_improvement_summary.png',
        'fig6_path_evolution.png',
        'fig7_success_rate.png',
    ]
    for lf in legacy_files:
        p = os.path.join(output_dir, lf)
        if os.path.exists(p):
            os.remove(p)

    scenarios = sorted(data.keys())
    n = len(scenarios)
    x = np.arange(n)
    width = 0.35

    C_COLOR = '#4472C4'
    E_COLOR = '#ED7D31'

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    plt.rcParams['axes.unicode_minus'] = False
    has_cjk = configure_cjk_font()
    if has_cjk:
        print("Using Chinese labels for scenario names.")
    else:
        print("No CJK font detected; Chinese labels may not render correctly.")

    short_names = []
    for s in scenarios:
        name = scenario_name_cn(s, short=True)
        short_names.append(name)

    def get_vals(key):
        cv = [data[s]['Classical'].get(key, 0) for s in scenarios]
        ev = [data[s]['Enhanced'].get(key, 0) for s in scenarios]
        return cv, ev

    def save_grouped_bar(metric_key, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=(10, 5))
        ct, et = get_vals(metric_key)
        ax.bar(x - width / 2, ct, width, label='Classical Informed RRT*', color=C_COLOR)
        ax.bar(x + width / 2, et, width, label='Enhanced Informed RRT*', color=E_COLOR)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=0, ha='center', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        print(f"  Saved: {filename}")

    # One chart per image for easier manual editing
    save_grouped_bar('rrt_time', 'Time (ms)', 'Informed RRT* Search Time', 'fig1_pipeline_timing.png')
    save_grouped_bar('lbfgs_time', 'Time (ms)', 'L-BFGS Optimization Time', 'fig2_lbfgs_time.png')
    save_grouped_bar('total_time', 'Time (ms)', 'End-to-End Pipeline Time', 'fig3_total_time.png')
    save_grouped_bar('rrt_path_len', 'Path Length (m)', 'Informed RRT* Raw Path Length', 'fig4_rrt_path_length.png')
    save_grouped_bar('traj_length', 'Trajectory Length (m)', 'Final Optimized Trajectory Length', 'fig5_traj_length.png')
    save_grouped_bar('lbfgs_iters', 'Iterations', 'L-BFGS Iteration Count', 'fig6_lbfgs_iters.png')
    save_grouped_bar('rrt_clearance', 'Clearance (m)', 'Informed RRT* Min Clearance', 'fig7_rrt_clearance.png')
    save_grouped_bar('traj_clearance', 'Clearance (m)', 'Final Trajectory Min Clearance', 'fig8_traj_clearance.png')
    save_grouped_bar('traj_smoothness', 'Total Turning Angle (rad)', 'Trajectory Smoothness (lower = smoother)', 'fig9_traj_smoothness.png')
    save_grouped_bar('traj_max_vel', 'Max Velocity (m/s)', 'Max Velocity', 'fig10_max_velocity.png')
    save_grouped_bar('traj_max_acc', 'Max Acceleration (m/s²)', 'Max Acceleration', 'fig11_max_acceleration.png')
    save_grouped_bar('lbfgs_cost_reduction', 'Cost Reduction (%)', 'Informed RRT* L-BFGS Cost Reduction', 'fig12_lbfgs_cost_reduction.png')

    # ── Stage breakdown: split into two separate single-chart images ──
    stages = ['Informed RRT* Search', 'B-Spline Conv', 'L-BFGS Opt']
    c_stage_avg = [0, 0, 0]
    e_stage_avg = [0, 0, 0]
    for s in scenarios:
        c_stage_avg[0] += data[s]['Classical']['rrt_time']
        c_stage_avg[1] += data[s]['Classical']['bspline_conv_time']
        c_stage_avg[2] += data[s]['Classical']['lbfgs_time']
        e_stage_avg[0] += data[s]['Enhanced']['rrt_time']
        e_stage_avg[1] += data[s]['Enhanced']['bspline_conv_time']
        e_stage_avg[2] += data[s]['Enhanced']['lbfgs_time']
    c_stage_avg = [v/n for v in c_stage_avg]
    e_stage_avg = [v/n for v in e_stage_avg]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_stage = ['#5B9BD5', '#A5A5A5', '#FFC000']
    bottom = np.zeros(2)
    for i, (stage, color) in enumerate(zip(stages, colors_stage)):
        vals = [c_stage_avg[i], e_stage_avg[i]]
        bars = ax.bar(['Classical Informed RRT*', 'Enhanced Informed RRT*'], vals, width=0.5,
                      bottom=bottom, label=stage, color=color, edgecolor='white')
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2, bottom[j] + val/2,
                        f'{val:.0f}ms', ha='center', va='center', fontsize=9, fontweight='bold')
        bottom += np.array(vals)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Average Pipeline Time Breakdown')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig13_stage_breakdown_avg.png'))
    plt.close(fig)
    print(f"  Saved: fig13_stage_breakdown_avg.png")

    # Per-scenario stacked for Enhanced (single chart)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, scenario in enumerate(scenarios):
        stages_vals = [
            data[scenario]['Enhanced']['rrt_time'],
            data[scenario]['Enhanced']['bspline_conv_time'],
            data[scenario]['Enhanced']['lbfgs_time'],
        ]
        bottom_val = 0
        for j, (val, color) in enumerate(zip(stages_vals, colors_stage)):
            ax.bar(i, val, 0.7, bottom=bottom_val, color=color, edgecolor='white', linewidth=0.3)
            bottom_val += val
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=0, ha='center', fontsize=8)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Enhanced Informed RRT*: Per-Scenario Pipeline Breakdown')
    ax.legend(stages, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig14_stage_breakdown_enhanced.png'))
    plt.close(fig)
    print(f"  Saved: fig14_stage_breakdown_enhanced.png")

    # Remove obsolete radar image if it exists from earlier runs.
    radar_path = os.path.join(output_dir, 'fig4_radar.png')
    if os.path.exists(radar_path):
        os.remove(radar_path)
        print("  Removed: fig4_radar.png")

    # ── Figure 5: Improvement Summary (horizontal bars) ───────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    imp_pairs = [
        ('Informed RRT* Search Time', 'rrt_time', True),
        ('Informed RRT* Path Length', 'rrt_path_len', True),
        ('Informed RRT* Tree Nodes', 'rrt_nodes', True),
        ('Informed RRT* Clearance', 'rrt_clearance', False),
        ('Informed RRT* L-BFGS Opt Time', 'lbfgs_time', True),
        ('Informed RRT* L-BFGS Iterations', 'lbfgs_iters', True),
        ('Informed RRT* L-BFGS Cost Reduc.', 'lbfgs_cost_reduction', False),
        ('Informed RRT* Trajectory Length', 'traj_length', True),
        ('Informed RRT* Trajectory Clearance', 'traj_clearance', False),
        ('Trajectory Smoothness', 'traj_smoothness', True),
        ('Informed RRT* Max Velocity', 'traj_max_vel', True),
        ('Informed RRT* Max Acceleration', 'traj_max_acc', True),
        ('Informed RRT* End-to-End Time', 'total_time', True),
        ('Success Rate', 'success_rate', False),
    ]

    imp_labels = []; imp_vals = []
    for label, key, lb in imp_pairs:
        vals = []
        for s in scenarios:
            cv = data[s]['Classical'][key]
            ev = data[s]['Enhanced'][key]
            if cv > 0 and key != 'success_rate':
                imp = (cv - ev) / cv * 100.0
                if not lb:
                    imp = -imp
                vals.append(imp)
        imp_labels.append(label)
        imp_vals.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)

    colors_bar = [E_COLOR if v > 0 else C_COLOR for v in imp_vals]
    bars = ax.barh(imp_labels, imp_vals, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Improvement (%) — Positive = Enhanced Better')
    ax.set_title('Average Improvement of Enhanced Informed RRT* over Classical Informed RRT*\n(across all pipeline stages and scenarios)')
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, imp_vals):
        x_pos = val + (1 if val >= 0 else -5)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', fontsize=9, fontweight='bold',
                color=E_COLOR if val > 0 else C_COLOR)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig15_improvement_summary.png'))
    plt.close(fig)
    print(f"  Saved: fig15_improvement_summary.png")

    # ── Figure 6: Path Length Evolution (Informed RRT* → Optimized) ────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    x2 = np.arange(n) * 3
    for i, scenario in enumerate(scenarios):
        c_rrt = data[scenario]['Classical']['rrt_path_len']
        c_traj = data[scenario]['Classical']['traj_length']
        e_rrt = data[scenario]['Enhanced']['rrt_path_len']
        e_traj = data[scenario]['Enhanced']['traj_length']

        # Classical: Informed RRT* → final
        ax.plot([x2[i], x2[i] + 1], [c_rrt, c_traj], 'o-', color=C_COLOR,
                markersize=6, linewidth=2, alpha=0.8)
        # Enhanced: Informed RRT* → final
        ax.plot([x2[i], x2[i] + 1], [e_rrt, e_traj], 's--', color=E_COLOR,
                markersize=6, linewidth=2, alpha=0.8)

    # Legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=C_COLOR, label='Classical Informed RRT* → Optimized', markersize=8, linewidth=2),
        Line2D([0], [0], marker='s', color=E_COLOR, label='Enhanced Informed RRT* → Optimized', markersize=8, linewidth=2, linestyle='--'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')
    ax.set_xticks([xi + 0.5 for xi in x2])
    ax.set_xticklabels(short_names, rotation=0, ha='center', fontsize=8)
    ax.set_ylabel('Path Length (m)')
    ax.set_title('Path Length Evolution: Raw Informed RRT* → B-Spline + L-BFGS Optimized')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig16_path_evolution.png'))
    plt.close(fig)
    print(f"  Saved: fig16_path_evolution.png")

    # ── Figure 7: Success Rate Comparison ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    c_sr = [data[s]['Classical']['success_rate'] for s in scenarios]
    e_sr = [data[s]['Enhanced']['success_rate'] for s in scenarios]
    ax.bar(x - width/2, c_sr, width, label='Classical Informed RRT*', color=C_COLOR)
    ax.bar(x + width/2, e_sr, width, label='Enhanced Informed RRT*', color=E_COLOR)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate Comparison Across Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=0, ha='center', fontsize=8)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    for i in range(n):
        if c_sr[i] > 0:
            ax.text(i - width/2, c_sr[i] + 1, f'{c_sr[i]:.0f}%', ha='center', fontsize=8, fontweight='bold')
        if e_sr[i] > 0:
            ax.text(i + width/2, e_sr[i] + 1, f'{e_sr[i]:.0f}%', ha='center', fontsize=8, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig17_success_rate.png'))
    plt.close(fig)
    print(f"  Saved: fig17_success_rate.png")

    print(f"\nAll figures saved to: {output_dir}/")


def plot_route_showcase(search_dir='.', output_dir='.', defense_style=False):
    """Generate per-scenario route showcase figures from trajectory CSV files."""
    if not HAS_MPL:
        return

    configure_cjk_font()

    def sample_box_surface(center, half_extents, step):
        cx, cy, cz = center
        hx, hy, hz = half_extents
        xs = np.arange(cx - hx, cx + hx + 1e-9, step)
        ys = np.arange(cy - hy, cy + hy + 1e-9, step)
        zs = np.arange(cz - hz, cz + hz + 1e-9, step)
        pts = []
        for x in xs:
            for y in ys:
                pts.append((x, y, cz - hz))
                pts.append((x, y, cz + hz))
        for x in xs:
            for z in zs:
                pts.append((x, cy - hy, z))
                pts.append((x, cy + hy, z))
        for y in ys:
            for z in zs:
                pts.append((cx - hx, y, z))
                pts.append((cx + hx, y, z))
        return pts

    def sample_cylinder_surface(center, radius, height, radial_steps=20, z_steps=10):
        cx, cy, cz = center
        z0 = cz - height * 0.5
        z1 = cz + height * 0.5
        pts = []
        angles = np.linspace(0.0, 2.0 * np.pi, radial_steps, endpoint=False)
        zs = np.linspace(z0, z1, z_steps)
        for a in angles:
            x = cx + radius * np.cos(a)
            y = cy + radius * np.sin(a)
            for z in zs:
                pts.append((x, y, z))
        # Top/bottom rims for better 2D projection visibility.
        for a in angles:
            x = cx + radius * np.cos(a)
            y = cy + radius * np.sin(a)
            pts.append((x, y, z0))
            pts.append((x, y, z1))
        return pts

    def sample_block_along_line(start, goal, num_blocks, block_radius, step):
        s = np.array(start, dtype=float)
        g = np.array(goal, dtype=float)
        d = g - s
        dist = np.linalg.norm(d)
        if dist < 1e-9:
            return []
        d /= dist
        pts = []
        offsets = np.arange(-block_radius, block_radius + 1e-9, step)
        for i in range(num_blocks):
            t = (i + 1.0) / (num_blocks + 1.0)
            c = s + d * dist * t
            for dx in offsets:
                for dy in offsets:
                    for dz in offsets:
                        p = c + np.array([dx, dy, dz], dtype=float)
                        pts.append((p[0], p[1], p[2]))
        return pts

    def clip_points(pts, bounds):
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        out = []
        for x, y, z in pts:
            if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                out.append((x, y, z))
        return out

    def scenario_obstacle_points(scenario):
        """
        Reconstruct scenario obstacles in Python using same scene logic as benchmark_core.h.
        This gives route figures with obstacle context for defense slides.
        """
        if scenario == '1_Sparse_Pillars':
            bounds = (0.0, 30.0, -10.0, 10.0, 0.0, 3.0)
            start = (1.0, 0.0, 1.5)
            goal = (28.0, 0.0, 1.5)
            seed = 100
            rng = np.random.default_rng(seed)
            pts = sample_block_along_line(start, goal, num_blocks=4, block_radius=0.8, step=0.24)
            for _ in range(10):
                cx = rng.uniform(2.0, 28.0)
                cy = rng.uniform(-8.0, 8.0)
                r = rng.uniform(0.3, 0.8)
                pts.extend(sample_cylinder_surface((cx, cy, 1.5), r, 3.0, radial_steps=20, z_steps=8))
            return bounds, clip_points(pts, bounds)

        if scenario == '2_Dense_Clutter':
            bounds = (0.0, 30.0, -10.0, 10.0, 0.0, 3.0)
            start = (1.0, 0.0, 1.5)
            goal = (28.0, 0.0, 1.5)
            seed = 200
            rng = np.random.default_rng(seed)
            pts = sample_block_along_line(start, goal, num_blocks=5, block_radius=0.9, step=0.24)
            for _ in range(40):
                cx = rng.uniform(2.0, 28.0)
                cy = rng.uniform(-8.0, 8.0)
                r = rng.uniform(0.3, 0.7)
                pts.extend(sample_cylinder_surface((cx, cy, 1.5), r, 3.0, radial_steps=16, z_steps=7))
            return bounds, clip_points(pts, bounds)

        if scenario == '3_Maze':
            bounds = (0.0, 30.0, -10.0, 10.0, 0.0, 3.0)
            pts = []
            gap_y = [2.0, -2.8, 3.2, -2.0, 2.6]
            x_positions = [4.0 + 5.0 * i for i in range(5)]
            ys = np.arange(-10.0, 10.0 + 1e-9, 0.25)
            zs = np.arange(0.0, 3.0 + 1e-9, 0.25)
            for wall_x, gy in zip(x_positions, gap_y):
                for y in ys:
                    if abs(y - gy) < 1.2:
                        continue
                    for z in zs:
                        pts.append((wall_x, y, z))
            return bounds, pts

        if scenario == '4_Long_Range':
            bounds = (0.0, 50.0, -10.0, 10.0, 0.0, 3.0)
            start = (1.0, 0.0, 1.5)
            goal = (48.0, 0.0, 1.5)
            seed = 500
            rng = np.random.default_rng(seed)
            pts = sample_block_along_line(start, goal, num_blocks=7, block_radius=0.9, step=0.24)
            for _ in range(25):
                cx = rng.uniform(2.0, 48.0)
                cy = rng.uniform(-8.0, 8.0)
                r = rng.uniform(0.3, 0.8)
                pts.extend(sample_cylinder_surface((cx, cy, 1.5), r, 3.0, radial_steps=18, z_steps=8))
            return bounds, clip_points(pts, bounds)

        if scenario == '5_Multi_Level':
            bounds = (0.0, 30.0, -10.0, 10.0, 0.0, 6.0)
            seed = 600
            rng = np.random.default_rng(seed)
            pts = []
            pts.extend(sample_box_surface((8.0, -4.0, 1.0), (3.0, 3.0, 0.15), step=0.3))
            pts.extend(sample_box_surface((15.0, 4.0, 2.5), (3.0, 3.0, 0.15), step=0.3))
            pts.extend(sample_box_surface((22.0, -3.0, 4.0), (2.0, 4.0, 0.15), step=0.3))
            pts.extend(sample_cylinder_surface((10.0, 0.0, 3.0), 0.4, 6.0, radial_steps=18, z_steps=16))
            pts.extend(sample_cylinder_surface((18.0, 0.0, 3.0), 0.4, 6.0, radial_steps=18, z_steps=16))
            for _ in range(8):
                cx = rng.uniform(2.0, 28.0)
                cy = rng.uniform(-8.0, 8.0)
                r = rng.uniform(0.2, 0.5)
                pts.extend(sample_cylinder_surface((cx, cy, 3.0), r, 6.0, radial_steps=14, z_steps=16))
            return bounds, clip_points(pts, bounds)

        if scenario == '6_Forest':
            bounds = (0.0, 40.0, -10.0, 10.0, 0.0, 5.0)
            start = (1.0, 0.0, 2.0)
            goal = (38.0, 0.0, 2.0)
            seed = 700
            rng = np.random.default_rng(seed)
            pts = sample_block_along_line(start, goal, num_blocks=6, block_radius=0.7, step=0.24)
            for _ in range(50):
                cx = rng.uniform(2.0, 38.0)
                cy = rng.uniform(-9.0, 9.0)
                h = rng.uniform(1.0, 5.0)
                r = rng.uniform(0.08, 0.2)
                pts.extend(sample_cylinder_surface((cx, cy, h * 0.5), r, h, radial_steps=12, z_steps=8))
            return bounds, clip_points(pts, bounds)

        if scenario == '7_Urban_Canyon':
            bounds = (0.0, 40.0, -10.0, 10.0, 0.0, 8.0)
            pts = []
            seed = 800
            rng = np.random.default_rng(seed)
            for i in range(6):
                x = 3.0 + i * 6.0
                h = 3.0 + 4.0 * abs(np.sin(i * 1.2))
                pts.extend(sample_box_surface((x, 5.5, h / 2.0), (2.0, 3.5, h / 2.0), step=0.35))
                pts.extend(sample_box_surface((x + 1.5, -5.5, h / 2.0 + 0.5), (2.0, 3.5, h / 2.0 + 0.5), step=0.35))
                if i in (2, 4):
                    pts.extend(sample_box_surface((x, 0.0, 5.0), (0.4, 2.0, 1.5), step=0.3))
            for _ in range(10):
                cx = rng.uniform(2.0, 38.0)
                cy = rng.uniform(-8.0, 8.0)
                r = rng.uniform(0.2, 0.4)
                pts.extend(sample_cylinder_surface((cx, cy, 4.0), r, 8.0, radial_steps=14, z_steps=14))
            return bounds, clip_points(pts, bounds)

        if scenario == '8_Random_Field':
            bounds = (0.0, 35.0, -10.0, 10.0, 0.0, 5.0)
            start = (1.0, 0.0, 2.5)
            goal = (33.0, 0.0, 2.5)
            seed = 1000
            rng = np.random.default_rng(seed)
            pts = sample_block_along_line(start, goal, num_blocks=5, block_radius=0.9, step=0.24)
            for _ in range(22):
                cx = rng.uniform(2.0, 33.0)
                cy = rng.uniform(-9.0, 9.0)
                cz = rng.uniform(0.0, 5.0)
                side = rng.uniform(0.2, 1.0)
                pts.extend(sample_box_surface((cx, cy, cz), (side, side, side * 2.0), step=0.28))
            for _ in range(22):
                cx = rng.uniform(2.0, 33.0)
                cy = rng.uniform(-9.0, 9.0)
                cz = rng.uniform(0.0, 5.0)
                side = rng.uniform(0.2, 1.0)
                pts.extend(sample_cylinder_surface((cx, cy, cz), side * 0.6, side * 3.0, radial_steps=12, z_steps=10))
            return bounds, clip_points(pts, bounds)

        # Fallback bounds from trajectory if scenario is unknown.
        return None, []

    def maybe_downsample_points(pts, max_points=7000, seed=0):
        if len(pts) <= max_points:
            return np.array(pts, dtype=float) if pts else np.empty((0, 3), dtype=float)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        arr = np.array(pts, dtype=float)
        return arr[idx]

    traj_files = sorted(glob.glob(os.path.join(search_dir, 'trajectory_*.csv')))
    if not traj_files:
        print(f"No trajectory CSV found under: {search_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    color_map = {'Classical': '#4472C4', 'Enhanced': '#ED7D31'}
    style_map = {'Classical': '-', 'Enhanced': '--'}
    stage_name = {'rrt': 'Informed RRT* raw', 'optimized': 'Optimized'}

    def parse_one_file(fp):
        parsed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        with open(fp, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                algo = row.get('algorithm', '')
                stage = row.get('stage', '')
                trial = row.get('trial', '0')
                try:
                    pt = (float(row['x']), float(row['y']), float(row['z']))
                except Exception:
                    continue
                parsed[algo][stage][trial].append(pt)
        return parsed

    def first_nonempty_stage(stage_dict):
        if not stage_dict:
            return []
        for trial_id in sorted(stage_dict.keys(), key=lambda v: int(v) if str(v).isdigit() else v):
            pts = stage_dict[trial_id]
            if pts:
                return pts
        return []

    if defense_style:
        route_linewidth = 2.8
        rrt_linewidth = 1.9
        marker_start = 88
        marker_goal = 150
        obstacle_alpha_2d = 0.24
        obstacle_alpha_3d = 0.16
        route_figsize = (19, 5.9)
        route_dpi = 340
        legend_font = 9
    else:
        route_linewidth = 2.3
        rrt_linewidth = 1.6
        marker_start = 70
        marker_goal = 130
        obstacle_alpha_2d = 0.18
        obstacle_alpha_3d = 0.12
        route_figsize = (18, 5.5)
        route_dpi = None
        legend_font = 8

    generated = []
    for fp in traj_files:
        scenario = os.path.splitext(os.path.basename(fp))[0].replace('trajectory_', '', 1)
        scenario_cn = scenario_name_cn(scenario, short=False)
        parsed = parse_one_file(fp)
        bounds, obstacle_pts = scenario_obstacle_points(scenario)
        obstacle_arr = maybe_downsample_points(obstacle_pts, max_points=8000, seed=42)

        fig = plt.figure(figsize=route_figsize)
        ax_xy = fig.add_subplot(1, 3, 1)
        ax_xz = fig.add_subplot(1, 3, 2)
        ax_3d = fig.add_subplot(1, 3, 3, projection='3d')

        if obstacle_arr.size > 0:
            ax_xy.scatter(obstacle_arr[:, 0], obstacle_arr[:, 1], s=2.2, c='#7f7f7f', alpha=obstacle_alpha_2d, label='Obstacles')
            ax_xz.scatter(obstacle_arr[:, 0], obstacle_arr[:, 2], s=2.2, c='#7f7f7f', alpha=obstacle_alpha_2d, label='Obstacles')
            ax_3d.scatter(obstacle_arr[:, 0], obstacle_arr[:, 1], obstacle_arr[:, 2],
                          s=1.3, c='#8c8c8c', alpha=obstacle_alpha_3d, depthshade=False)

        starts = []
        goals = []
        for algo in ('Classical', 'Enhanced'):
            if algo not in parsed:
                continue
            rrt_pts = first_nonempty_stage(parsed[algo].get('rrt', {}))
            opt_pts = first_nonempty_stage(parsed[algo].get('optimized', {}))

            if rrt_pts:
                arr = np.array(rrt_pts)
                ax_xy.plot(arr[:, 0], arr[:, 1], color=color_map[algo], linestyle=':', linewidth=rrt_linewidth,
                           alpha=0.8, label=f'{algo} {stage_name["rrt"]}')
                ax_xz.plot(arr[:, 0], arr[:, 2], color=color_map[algo], linestyle=':', linewidth=rrt_linewidth,
                           alpha=0.8, label=f'{algo} {stage_name["rrt"]}')
                ax_3d.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=color_map[algo], linestyle=':', linewidth=1.5, alpha=0.8)

            if opt_pts:
                arr = np.array(opt_pts)
                ax_xy.plot(arr[:, 0], arr[:, 1], color=color_map[algo], linestyle=style_map[algo], linewidth=route_linewidth,
                           label=f'{algo} {stage_name["optimized"]}')
                ax_xz.plot(arr[:, 0], arr[:, 2], color=color_map[algo], linestyle=style_map[algo], linewidth=route_linewidth,
                           label=f'{algo} {stage_name["optimized"]}')
                ax_3d.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=color_map[algo], linestyle=style_map[algo],
                           linewidth=route_linewidth, label=f'{algo} optimized')
            # Start/goal markers should come from the planned RRT endpoints,
            # not from potentially shifted post-optimization samples.
            if rrt_pts:
                rrt_arr = np.array(rrt_pts)
                starts.append(rrt_arr[0, :])
                goals.append(rrt_arr[-1, :])

        if starts:
            start = starts[0]
            ax_xy.scatter(start[0], start[1], marker='o', s=marker_start, c='green', edgecolors='black', zorder=10, label='Start')
            ax_xz.scatter(start[0], start[2], marker='o', s=marker_start, c='green', edgecolors='black', zorder=10, label='Start')
            ax_3d.scatter(start[0], start[1], start[2], marker='o', s=marker_start - 5, c='green', edgecolors='black', zorder=10)
        if goals:
            goal = goals[0]
            ax_xy.scatter(goal[0], goal[1], marker='*', s=marker_goal, c='red', edgecolors='black', zorder=10, label='Goal')
            ax_xz.scatter(goal[0], goal[2], marker='*', s=marker_goal, c='red', edgecolors='black', zorder=10, label='Goal')
            ax_3d.scatter(goal[0], goal[1], goal[2], marker='*', s=marker_goal - 10, c='red', edgecolors='black', zorder=10)

        ax_xy.set_title(f'{scenario_cn}（俯视图 X-Y）')
        ax_xy.set_xlabel('X (m)')
        ax_xy.set_ylabel('Y (m)')
        ax_xy.grid(alpha=0.3)
        ax_xy.axis('equal')

        ax_xz.set_title(f'{scenario_cn}（侧视图 X-Z）')
        ax_xz.set_xlabel('X (m)')
        ax_xz.set_ylabel('Z (m)')
        ax_xz.grid(alpha=0.3)

        ax_3d.set_title(f'{scenario_cn}（三维视图）')
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        if defense_style:
            ax_3d.view_init(elev=24, azim=-56)
        else:
            ax_3d.view_init(elev=23, azim=-58)
        ax_3d.grid(True, alpha=0.2)

        if bounds is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            ax_xy.set_xlim(xmin, xmax)
            ax_xy.set_ylim(ymin, ymax)
            ax_xz.set_xlim(xmin, xmax)
            ax_xz.set_ylim(zmin, zmax)
            ax_3d.set_xlim(xmin, xmax)
            ax_3d.set_ylim(ymin, ymax)
            ax_3d.set_zlim(zmin, zmax)

            span_x = xmax - xmin
            span_y = ymax - ymin
            span_z = zmax - zmin
            if hasattr(ax_3d, 'set_box_aspect'):
                ax_3d.set_box_aspect((max(span_x, 1e-6), max(span_y, 1e-6), max(span_z, 1e-6)))

        handles, labels = ax_xy.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            uniq[l] = h
        ax_xy.legend(list(uniq.values()), list(uniq.keys()), fontsize=legend_font, loc='best')
        h3d, l3d = ax_3d.get_legend_handles_labels()
        if h3d:
            ax_3d.legend(h3d, l3d, fontsize=max(7, legend_font - 1), loc='upper right')

        plt.tight_layout()
        out_name = f'route_{scenario}.png'
        save_kwargs = {'dpi': route_dpi} if route_dpi is not None else {}
        fig.savefig(os.path.join(output_dir, out_name), **save_kwargs)
        plt.close(fig)
        generated.append((scenario, out_name, parsed))
        print(f"  Saved: {out_name}")

    # One-page overview for defense slides
    if generated:
        cols = 2
        rows = int(np.ceil(len(generated) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, max(4.5, rows * 3.5)))
        axes = np.atleast_1d(axes).reshape(rows, cols)

        for idx, (scenario, _, parsed) in enumerate(generated):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            scenario_short_cn = scenario_name_cn(scenario, short=True).replace('\n', ' ')
            for algo in ('Classical', 'Enhanced'):
                if algo not in parsed:
                    continue
                opt_pts = first_nonempty_stage(parsed[algo].get('optimized', {}))
                if not opt_pts:
                    continue
                arr = np.array(opt_pts)
                ax.plot(arr[:, 0], arr[:, 1], color=color_map[algo], linestyle=style_map[algo], linewidth=2, label=algo)
                ax.scatter(arr[0, 0], arr[0, 1], marker='o', s=15, c='green', zorder=5)
                ax.scatter(arr[-1, 0], arr[-1, 1], marker='*', s=25, c='red', zorder=5)

            ax.set_title(scenario_short_cn, fontsize=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(alpha=0.25)
            ax.axis('equal')
            if idx == 0:
                ax.legend(fontsize=8, loc='best')

        total_slots = rows * cols
        for idx in range(len(generated), total_slots):
            r = idx // cols
            c = idx % cols
            axes[r, c].axis('off')

        plt.tight_layout()
        overview_kwargs = {'dpi': 320} if defense_style else {}
        fig.savefig(os.path.join(output_dir, 'route_overview_all_scenarios.png'), **overview_kwargs)
        plt.close(fig)
        print("  Saved: route_overview_all_scenarios.png")


def plot_defense_figures(data, trial_data, output_dir='.'):
    """Generate defense-focused figures using per-trial data."""
    if not HAS_MPL or not trial_data:
        return

    scenarios = sorted(data.keys())

    # Figure A: Overall KPI with 95% CI error bars
    metric_cfg = [
        ('RRT_Time(ms)', True, 'RRT Time'),
        ('LBFGS_Time(ms)', True, 'L-BFGS Time'),
        ('Total_Time(ms)', True, 'Total Time'),
        ('Traj_Length(m)', True, 'Traj Length'),
        ('Traj_Clearance(m)', False, 'Traj Clearance'),
        ('Success', False, 'Success Rate'),
    ]

    kpi_names = []
    kpi_vals = []
    kpi_err = []
    kpi_colors = []
    for key, lower_better, label in metric_cfg:
        scenario_improvements = []
        for s in scenarios:
            if s not in trial_data:
                continue
            c = trial_data[s]['Classical'].get(key, [])
            e = trial_data[s]['Enhanced'].get(key, [])
            if not c or not e:
                continue
            c = np.array(c, dtype=float)
            e = np.array(e, dtype=float)
            n = min(len(c), len(e))
            c = c[:n]
            e = e[:n]
            if key == 'Success':
                # Success is 0/1; convert to percentage point gain
                imp = (np.mean(e) - np.mean(c)) * 100.0
            elif lower_better:
                valid = np.abs(c) > 1e-9
                if np.any(valid):
                    imp = np.mean((c[valid] - e[valid]) / c[valid] * 100.0)
                else:
                    imp = 0.0
            else:
                valid = np.abs(c) > 1e-9
                if np.any(valid):
                    imp = np.mean((e[valid] - c[valid]) / c[valid] * 100.0)
                else:
                    imp = 0.0
            scenario_improvements.append(float(imp))

        m, ci = mean_ci(scenario_improvements)
        kpi_names.append(label)
        kpi_vals.append(m)
        kpi_err.append(ci)
        kpi_colors.append('#2CA02C' if m >= 0 else '#D62728')

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(kpi_names))
    ax.bar(x, kpi_vals, yerr=kpi_err, capsize=5, color=kpi_colors, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(kpi_names, rotation=20, ha='right')
    ax.set_ylabel('Improvement (%) or pp for Success')
    ax.set_title('Defense KPI Summary: Enhanced vs Classical (95% CI across scenarios)')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(kpi_vals):
        ax.text(i, v + (1 if v >= 0 else -2), f'{v:+.1f}', ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'figA_defense_kpi_ci.png'))
    plt.close(fig)
    print("  Saved: figA_defense_kpi_ci.png")

    # Figure B: Scenario-level total time improvements with CI (forest style)
    labels = []
    means = []
    lows = []
    highs = []
    for s in scenarios:
        if s not in trial_data:
            continue
        c = trial_data[s]['Classical'].get('Total_Time(ms)', [])
        e = trial_data[s]['Enhanced'].get('Total_Time(ms)', [])
        m, lo, hi, n = paired_improvement_ci(c, e)
        if n <= 0:
            continue
        labels.append(scenario_name_cn(s, short=False))
        means.append(m)
        lows.append(lo)
        highs.append(hi)

    if labels:
        y = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.5)))
        ax.hlines(y, lows, highs, color='#1F77B4', linewidth=2)
        ax.plot(means, y, 'o', color='#FF7F0E', markersize=6)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Paired Improvement in Total Time (%)')
        ax.set_title('Scenario-wise End-to-End Speedup (95% CI, paired trials)')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'figB_total_time_forest.png'))
        plt.close(fig)
        print("  Saved: figB_total_time_forest.png")

    # Figure C: Distribution boxplot for critical metrics (all scenarios pooled)
    critical = [
        ('Total_Time(ms)', 'Total Time (ms)'),
        ('RRT_Time(ms)', 'RRT Time (ms)'),
        ('Traj_Clearance(m)', 'Trajectory Clearance (m)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, title) in zip(axes, critical):
        cvals, evals = [], []
        for s in scenarios:
            if s not in trial_data:
                continue
            cvals.extend(trial_data[s]['Classical'].get(key, []))
            evals.extend(trial_data[s]['Enhanced'].get(key, []))
        if not cvals or not evals:
            ax.set_title(title + '\n(no trial data)')
            continue
        b = ax.boxplot([cvals, evals], labels=['Classical', 'Enhanced'],
                       patch_artist=True, showfliers=False)
        b['boxes'][0].set(facecolor='#4472C4', alpha=0.7)
        b['boxes'][1].set(facecolor='#ED7D31', alpha=0.7)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Pooled Trial Distributions (robustness evidence)', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'figC_distribution_boxplots.png'))
    plt.close(fig)
    print("  Saved: figC_distribution_boxplots.png")

    # Export significance summary text for defense speaking points
    report_path = os.path.join(output_dir, 'defense_stats_summary.txt')
    with open(report_path, 'w') as f:
        f.write("Defense Statistical Summary\n")
        f.write("==========================\n\n")
        f.write("Metric-level paired sign-test p-values (lower p -> stronger evidence):\n")
        metric_tests = [
            ('Total_Time(ms)', True),
            ('RRT_Time(ms)', True),
            ('LBFGS_Time(ms)', True),
            ('Traj_Length(m)', True),
            ('Traj_Clearance(m)', False),
            ('Success', False),
        ]
        for key, lower_better in metric_tests:
            pv = []
            for s in scenarios:
                if s not in trial_data:
                    continue
                c = trial_data[s]['Classical'].get(key, [])
                e = trial_data[s]['Enhanced'].get(key, [])
                if not c or not e:
                    continue
                pv.append(sign_test_pvalue(c, e, lower_better=lower_better))
            if pv:
                f.write(f"- {key}: median p = {np.median(pv):.4f}, scenarios = {len(pv)}\n")
            else:
                f.write(f"- {key}: no data\n")
    print("  Saved: defense_stats_summary.txt")


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'pipeline_benchmark_results.csv'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'plots'
    extra_args = [a.strip().lower() for a in sys.argv[3:]]
    defense_style = ('--defense-style' in extra_args) or ('--defense' in extra_args)

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run benchmark first: ./full_pipeline_benchmark")
        print(f"Expected CSV at: {csv_path}")
        sys.exit(1)

    print(f"Loading: {csv_path}")
    data = load_csv(csv_path)
    print(f"Found {len(data)} scenarios.\n")
    if defense_style:
        print("Defense style mode: ON (thicker lines, unified view, high-resolution export).")
    plot_comparison(data, output_dir)
    plot_route_showcase(
        search_dir=os.path.dirname(os.path.abspath(csv_path)),
        output_dir=output_dir,
        defense_style=defense_style,
    )

    # Keep presentation figures clean: do not generate CI / significance figures by default.
    print("Skip CI/significance figures for cleaner presentation output.")

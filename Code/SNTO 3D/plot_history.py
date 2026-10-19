import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors


def save_pdf(file_path, **kwargs):
    """Guarded save of the current figure as pdf. With FORCE=False, skip the write
    when the output pdf already exists (leaving it untouched); with FORCE=True,
    (re-)write it. Extra kwargs (e.g. bbox_inches) pass through to plt.savefig."""
    if os.path.isfile(file_path) and not FORCE:
        print(f'[exists] {file_path}')
        return
    plt.savefig(file_path, format='pdf', dpi=600, **kwargs)

colors_density = ['#FFFFFF', '#addc30', '#5ec962', '#28ae80', '#21918c', '#2c72e8', '#3b526b', '#472d7b', '#440154']  # White to the purple end
cmap_density = LinearSegmentedColormap.from_list('custom_cmap', colors_density, N=128)

colors_phase = ['#FFFFFF', '#33a645', '#3b4cc0', '#b40426'] 
Density = [0, 0.4, 0.6, 1.0]
D_norm = (np.array(Density) - min(Density)) / (max(Density) - min(Density))  # [0.0, 0.4, 0.6, 1.0] -> [0.0, 0.4, 0.6, 1.0] (already normalized in this case)
color_tuples = list(zip(D_norm, colors_phase))
cmap_phases = LinearSegmentedColormap.from_list('custom_density_cmap', color_tuples, N=256)

colors_contours = ["#3b4fbf", "#79a2f2", "#c2d7f2", "#f2c9bb", "#f2856d", "#bf0637"]
cmap_contours = LinearSegmentedColormap.from_list("custom_cmap", colors_contours, N=128)

cmap_curves = {'blue_dark': "#3b4cc0",'blue_light': "#8db0fe",
                 'red_dark':"#b40426",'red_light':"#f4987a",
                 'green_dark': "#33a645",'green_light': "#afd991",
                 'gray_dark': "#555555",'gray_light': "#bababa",
                 'purple_dark': "#6659a7",'purple_light': "#c2bdde",
                 'orange_dark': "#ef7e21",'orange_light': "#f6c092",
                 'yellow_dark': "#f6d31b",'yellow_light': "#fae98f",
                 'cyan_dark': "#7bddf3",'cyan_light': "#bbedf2",
                 'magenta_dark': "#fe3da7",'magenta_light': "#fe9fd3",
                 }

def hex_to_rgba(hex_color, alpha=1.0):
    """
    Convert hex color to RGBA format and adjust alpha (transparency).
    Default alpha is 1.0 (fully opaque).
    
    Parameters:
    - hex_color: str, color in hex format (e.g., '#3b4cc0')
    - alpha: float, transparency level (0.0 to 1.0)
    
    Returns:
    - rgba: tuple, (r, g, b, a)
    """
    rgba = mcolors.to_rgba(hex_color)
    return (rgba[0], rgba[1], rgba[2], alpha)

def moving_average(arr, window_size):
    """
    Apply a simple moving average filter to smooth the input array.
    
    Parameters:
    - arr: numpy array, the input data to be smoothed.
    - window_size: int, the size of the moving window for averaging.
    
    Returns:
    - smoothed: numpy array, the smoothed data.
    """
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

EXAMPLES = [
    'Results/L_beam_3D/C2_L_beam_TO10_Sophia',
    'Results/L_beam_3D/C2_L_beam_TO11_Sophia',
    'Results/Portal_frame_half_3D/C2_Portal_frame_half_TO10_Sophia',
    'Results/Portal_frame_half_3D/C2_Portal_frame_half_TO11_Sophia',
]
FORCE = False       # True -> re-plot everything; False -> skip a pdf that already exists

iter = [29999]

for save_folder in EXAMPLES:
    if not os.path.isdir(save_folder):
        print(f'[skip] no results folder: {save_folder}'); continue
    print(f'[plot] {save_folder}')
    for run_num in range(1,6):
        run_folder = os.path.join(save_folder, f'Run_{run_num}/')
        if not os.path.isdir(run_folder):
            continue
        with open(run_folder + "timeHistory.json", "r") as file:
            timeHistory = json.load(file)

        if 'f' in timeHistory.keys():
            f = timeHistory['f']
        if 'ps' in timeHistory.keys():
            ps = timeHistory['ps']
        if 'ALPHA_B' in timeHistory.keys():
                ALPHA_B = timeHistory['ALPHA_B']
        scale = timeHistory['alpha']
        loss_compliance = timeHistory['loss_obj']
        loss_dem = timeHistory['loss_dem']
        loss_adj = timeHistory['loss_adj']
        loss_mConstraint = timeHistory['loss_mConstraint']
        loss_cConstraint = timeHistory['loss_cConstraint']
        SE = timeHistory['strain_energy']
        EW = timeHistory['external_work']
        SE_adj = timeHistory['strain_energy_adj']
        EW_adj = timeHistory['body_energy_adj']
        SE_smoothed = moving_average(np.array(SE), window_size=10)
        SE_adj_smoothed = moving_average(np.array(SE_adj), window_size=10)
        EW = moving_average(np.array(EW), window_size=10)
        EW_adj = moving_average(np.array(EW_adj), window_size=10)
        loss_dem_smoothed = moving_average(np.array(loss_dem), window_size=10)
        loss_adj_smoothed = moving_average(np.array(loss_adj), window_size=10)
        loss_mConstraint_smoothed = moving_average(np.array(loss_mConstraint), window_size=10)
        massfrac = timeHistory['massfrac']
        grey = timeHistory['grey']
        # three stress-violation ratios (see driver header for definitions). All reads
        # guarded so this never KeyErrors on runs that predate these keys.
        stress_violation_1 = timeHistory.get('stress_violation_1', [])
        stress_violation_2 = timeHistory.get('stress_violation_2', [])
        stress_violation_3 = timeHistory.get('stress_violation_3', [])
        if len(stress_violation_1) == 0:
            stress_violation_1 = timeHistory.get('stress_violation', [])   # legacy single series
        # raw violating-element counts for the separate TO_evolution_number.pdf plot
        n_viol_solid  = timeHistory.get('stress_viol_count_solid', [])
        n_h_act       = timeHistory.get('n_h_activate', [])
        n_h_act_solid = timeHistory.get('n_h_activate_solid', [])
        # sy schedule + per-epoch density-change diagnostic (guarded: absent in older runs)
        sy = timeHistory.get('sy', [])
        rho_diff_max = timeHistory.get('rho_diff_max', [])
        rho_diff_avg = timeHistory.get('rho_diff_avg', [])
        rho_diff_q95 = timeHistory.get('rho_diff_q95', [])
        b = timeHistory['b']
        p = timeHistory['p']
        wc = timeHistory['wc']
        print(massfrac[-1])

        blue_dark = hex_to_rgba(cmap_curves['blue_dark'])
        gray_dark = hex_to_rgba(cmap_curves['gray_dark'])
        blue_light = hex_to_rgba(cmap_curves['blue_light'])
        red_dark = hex_to_rgba(cmap_curves['red_dark'])
        red_light = hex_to_rgba(cmap_curves['red_light'])
        green_dark = hex_to_rgba(cmap_curves['green_dark'])
        green_light = hex_to_rgba(cmap_curves['green_light'])
        cyan_dark = hex_to_rgba(cmap_curves['cyan_dark'])
        cyan_light = hex_to_rgba(cmap_curves['cyan_light'])
        magenta_dark = hex_to_rgba(cmap_curves['magenta_dark'])
        magenta_light = hex_to_rgba(cmap_curves['magenta_light'])
        yellow_dark = hex_to_rgba(cmap_curves['yellow_dark'])
        yellow_light = hex_to_rgba(cmap_curves['yellow_light'])
        orange_dark = hex_to_rgba(cmap_curves['orange_dark'])

        # plot parameters
        xticks = [0,300,600,900,1200,1500]
        yticks = [1e-4,1e-3,1e-2,1e-1,1,10]
        xlim = [0,iter[0]]
        ylim = [1e-11,5]
        font = "Times New Roman"
        fontsize_title = 20
        fontsize_anno = 20
        fontsize_label = 18
        fontsize_ticks = 18
        fontsize_legend = 18
        anno_loc = [-0.15,0.99]

        box_linewidth = 1.5
        legend_linewidth = 1
        colorbar_shrink = 0.6
        curve_linewidth = 2

        # Create the plot with size (8, 6)
        plt.rcParams.update({
            'text.usetex': False,
            'font.family': font,
            'axes.labelsize': fontsize_label,
            'axes.titlesize': fontsize_title,
            'xtick.labelsize': fontsize_ticks,
            'ytick.labelsize': fontsize_ticks,
            'legend.fontsize': fontsize_legend, 
        })
        plt.rcParams["mathtext.fontset"] = "stix"

        # plot loss histories
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogy(loss_compliance, label=r"$L_C(\boldsymbol{z}^{\prime}(\boldsymbol{x};\boldsymbol{\theta},\boldsymbol{\phi}))$", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
        ax.semilogy(loss_dem, color=green_light, linestyle='-', linewidth=curve_linewidth)
        ax.semilogy(loss_dem_smoothed, label=r"$L_P(\boldsymbol{z}^{\prime}(\boldsymbol{x};\boldsymbol{\theta},\boldsymbol{\phi}))$", color=green_dark, linestyle='-', linewidth=curve_linewidth)
        ax.semilogy(loss_adj, color=cyan_light, linestyle='-', linewidth=curve_linewidth)
        ax.semilogy(loss_adj_smoothed, label=r"$L_{ADJ}(\boldsymbol{z}^{\prime}(\boldsymbol{x};\boldsymbol{\theta},\boldsymbol{\phi}))$", color=cyan_dark, linestyle='-', linewidth=curve_linewidth)
        ax.semilogy(loss_mConstraint, color=red_light, linestyle='-', linewidth=curve_linewidth)
        ax.semilogy(loss_mConstraint_smoothed, label=r"$C_1^2(\boldsymbol{z}_{\rho}^{\prime}(\boldsymbol{x};\boldsymbol{\theta},\boldsymbol{\phi}_{\rho}))$", color=red_dark, linestyle='-', linewidth=curve_linewidth)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
        file_name = "TO_evolution_loss.pdf"
        file_path = f"{run_folder}/{file_name}"
        save_pdf(file_path)

        # ---- body vs comp density-gradient magnitude (PolyStress-style split) ----
        if 'grad_comp_norm' in timeHistory:
            gcn = np.array(timeHistory['grad_comp_norm'], dtype=float)
            gbn = np.array(timeHistory['grad_body_norm'], dtype=float)
            gcm = np.array(timeHistory.get('grad_comp_max', []), dtype=float)
            gbm = np.array(timeHistory.get('grad_body_max', []), dtype=float)
            fig2, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))
            axA.semilogy(gcn, label=r"comp $\|\partial/\partial\rho\|_2$", color=blue_dark, linewidth=curve_linewidth)
            axA.semilogy(gbn, label=r"body $\|\partial/\partial\rho\|_2$", color=red_dark, linewidth=curve_linewidth)
            axA.set_xlabel("Epochs"); axA.set_ylabel(r"$\|\partial(\cdot)/\partial\rho\|_2$")
            axA.set_title("L2 norm"); axA.set_xlim([0, len(gcn)])
            legA = axA.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
            legA.get_frame().set_linewidth(legend_linewidth)
            if gcm.size and gbm.size:
                axB.semilogy(gcm, label=r"comp $\max|\partial/\partial\rho|$", color=blue_dark, linewidth=curve_linewidth)
                axB.semilogy(gbm, label=r"body $\max|\partial/\partial\rho|$", color=red_dark, linewidth=curve_linewidth)
            axB.set_xlabel("Epochs"); axB.set_ylabel(r"$\max|\partial(\cdot)/\partial\rho|$")
            axB.set_title("max (peak element)"); axB.set_xlim([0, len(gcm) if gcm.size else len(gcn)])
            legB = axB.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
            legB.get_frame().set_linewidth(legend_linewidth)
            fig2.tight_layout()
            save_pdf(f"{run_folder}/TO_evolution_body_comp.pdf")
            plt.close(fig2)


        # plot energy histories
        fig, ax = plt.subplots(figsize=(8, 6))
        anno_loc = [-0.1,0.99]
        xlim = [0,iter[0]]
        ylim = [0,0.2]
        ax.semilogy(2 * np.array(SE) , color=blue_light, linestyle='-', linewidth=curve_linewidth)
        ax.semilogy(2 * (SE_smoothed), label=r"$2 \times$Strain Energy", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
        ax.semilogy(EW, label="External Work", color=red_dark, linestyle='-', linewidth=curve_linewidth)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Energy (mJ)")
        ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
        file_name = "TO_evolution_energy.pdf"
        file_path = f"{run_folder}/{file_name}"
        save_pdf(file_path)
        # plt.show()

        # plot energy history inset
        fig, ax = plt.subplots(figsize=(8, 6))
        anno_loc = [-0.1,0.99]
        xlim = [0,2000]
        ylim = [0,0.5]
        ax.semilogy(2 * np.array(SE) , color=blue_light, linestyle='-', linewidth=curve_linewidth)
        ax.semilogy(2 * (SE_smoothed ), label=r"$2 \times$Strain Energy", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
        ax.semilogy(EW, label="External Work", color=red_dark, linestyle='-', linewidth=curve_linewidth)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Energy (mJ)")
        ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        file_name = "TO_evolution_energy_inset.pdf"
        file_path = f"{run_folder}/{file_name}"
        save_pdf(file_path)
        # plt.show()

        # plot energy histories
        fig, ax = plt.subplots(figsize=(8, 6))
        anno_loc = [-0.1,0.99]
        xlim = [0,iter[0]]
        # xlim = [14999,19999]
        ylim = [0,5]
        ax.semilogy(2 * np.array(SE_adj) , color=blue_light, linestyle='-', linewidth=curve_linewidth)
        ax.semilogy(2 * (SE_adj_smoothed), label=r"$2 \times$Strain Energy (adj)", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
        ax.semilogy(-EW_adj, label="body Work (adj)", color=red_dark, linestyle='-', linewidth=curve_linewidth)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Energy (mJ)")
        ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
        file_name = "TO_evolution_energy_adj.pdf"
        file_path = f"{run_folder}/{file_name}"
        save_pdf(file_path)
        # plt.show()

        # plot energy histories
        fig, ax = plt.subplots(figsize=(8, 6))
        anno_loc = [-0.1,0.99]
        xlim = [0,iter[0]]
        xlim = [16999,18500]
        ylim = [0,5]
        ax.scatter(
            range(len(SE_adj)),
            2 * np.array(SE_adj),
            label=r"$2 \times$Strain Energy (adj)",
            color=blue_dark,
            marker='.',
        )
        ax.scatter(
            range(len(EW_adj)),
            -np.array(EW_adj),
            label=r"body work (adj)",
            color=red_dark,
            marker='.',
        )
        ax.set_yscale('log')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Energy (mJ)")
        ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
        file_name = "TO_evolution_energy_adj2.pdf"
        file_path = f"{run_folder}/{file_name}"
        save_pdf(file_path)
        # plt.show()

        # plot gray and massfrac
        color_list = [red_dark, magenta_dark, yellow_dark, cyan_dark]
        grey = np.array(grey)
        m,n = grey.shape
        fig, ax = plt.subplots(figsize=(8, 6))
        anno_loc = [-0.15,0.99]
        xlim = [0,iter[0]]
        ylim = [0,1]
        ax.plot(massfrac, label=r"Mass Fraction", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
        if len(stress_violation_1) > 0:
            ax.plot(stress_violation_1, label=r"Stress Violation 1", color=green_dark,
                                linestyle='-', linewidth=curve_linewidth)
        if len(stress_violation_2) > 0:
            ax.plot(stress_violation_2, label=r"Stress Violation 2", color=orange_dark,
                                linestyle='-', linewidth=curve_linewidth)
        if len(stress_violation_3) > 0:
            ax.plot(stress_violation_3, label=r"Stress Violation 3", color=magenta_dark,
                                linestyle='-', linewidth=curve_linewidth)
        for i in range(n):
            ax.plot(grey[:,i], label=f"Gray Fraction {i}", color=color_list[i], linestyle='-',linewidth=curve_linewidth)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Fraction")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
        file_name = "TO_evolution_gray.pdf"
        file_path = f"{run_folder}/{file_name}"
        save_pdf(file_path)
        # plt.show()

        # plot the raw violating-element counts (separate figure)
        fig, ax = plt.subplots(figsize=(8, 6))
        if len(n_viol_solid) > 0:
            ax.plot(n_viol_solid, label=r"Viol count (solid)", color=green_dark,
                                linestyle='-', linewidth=curve_linewidth)
        if len(n_h_act) > 0:
            ax.plot(n_h_act, label=r"h active (all)", color=orange_dark,
                                linestyle='-', linewidth=curve_linewidth)
        if len(n_h_act_solid) > 0:
            ax.plot(n_h_act_solid, label=r"h active (solid)", color=magenta_dark,
                                linestyle='-', linewidth=curve_linewidth)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Number of elements")
        ax.set_xlim([0, iter[0]])
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        legend.get_frame().set_linewidth(legend_linewidth)
        file_name = "TO_evolution_number.pdf"
        file_path = f"{run_folder}/{file_name}"
        save_pdf(file_path)
        # plt.show()

        # plot gray and massfrac
        color_list = [red_dark, magenta_dark, yellow_dark, cyan_dark]
        grey = np.array(grey)
        m,n = grey.shape
        fig, ax = plt.subplots(figsize=(8, 6))
        anno_loc = [-0.15,0.99]
        xlim = [0,iter[0]]
        ylim = [0,1]
        ax.semilogy(scale, label=r"Scale", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Scale")
        ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
        file_name = "TO_evolution_scale.pdf"
        file_path = f"{run_folder}/{file_name}"
        save_pdf(file_path)

        # plot gray and massfrac
        color_list = [red_dark, magenta_dark, yellow_dark, cyan_dark]
        grey = np.array(grey)
        m,n = grey.shape
        fig, ax = plt.subplots(figsize=(8, 6))
        anno_loc = [-0.15,0.99]
        xlim = [0,iter[0]]
        # ylim = [0,1]
        ax.plot(b, label=r"b", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("b")
        ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
        file_name = "TO_evolution_b.pdf"
        file_path = f"{run_folder}/{file_name}"
        save_pdf(file_path)
        # plt.show()

        # plot gray and massfrac
        color_list = [red_dark, magenta_dark, yellow_dark, cyan_dark]
        m,n = grey.shape
        fig, ax = plt.subplots(figsize=(8, 6))
        anno_loc = [-0.15,0.99]
        xlim = [0,iter[0]]
        # ylim = [0,1]
        ax.plot(ALPHA_B, label=r"alpha_b", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Alpha_b")
        ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
        file_name = "TO_evolution_ALPHA_B.pdf"
        file_path = f"{run_folder}/{file_name}"
        save_pdf(file_path)
        # plt.show()

        # plot gray and massfrac
        color_list = [red_dark, magenta_dark, yellow_dark, cyan_dark]
        grey = np.array(grey)
        m,n = grey.shape
        fig, ax = plt.subplots(figsize=(8, 6))
        anno_loc = [-0.15,0.99]
        xlim = [0,iter[0]]
        # ylim = [0,1]
        ax.plot(p, label=r"p", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
        if 'ps' in timeHistory.keys():
            ax.plot(ps, label=r"ps", color=red_dark, linestyle='-', linewidth=curve_linewidth)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("p")
        ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
        file_name = "TO_evolution_p.pdf"
        file_path = f"{run_folder}/{file_name}"
        save_pdf(file_path)
        # plt.show()

        # plot sy (yield-stress) schedule
        if len(sy) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            xlim = [0, iter[0]]
            ax.plot(sy, label=r"$\sigma_y$", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
            ax.set_xlabel("Epochs")
            ax.set_ylabel(r"$\sigma_y$")
            ax.set_xlim(xlim)
            legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
            legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
            file_name = "TO_evolution_sy.pdf"
            file_path = f"{run_folder}/{file_name}"
            save_pdf(file_path)
            # plt.show()

        # plot per-epoch density-change diagnostic (max / avg / 95%-quantile of
        # |rho_cur - rho_prev| over all elements). First epoch is NaN (no previous design).
        if len(rho_diff_max) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            xlim = [0, iter[0]]
            ax.semilogy(rho_diff_max, label=r"max$|\Delta\rho|$", color=red_dark, linestyle='-', linewidth=curve_linewidth)
            ax.semilogy(rho_diff_q95, label=r"$Q_{95}|\Delta\rho|$", color=green_dark, linestyle='-', linewidth=curve_linewidth)
            ax.semilogy(rho_diff_avg, label=r"avg$|\Delta\rho|$", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
            ax.set_xlabel("Epochs")
            ax.set_ylabel(r"$|\Delta\rho|$ per epoch")
            ax.set_xlim(xlim)
            legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
            legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
            file_name = "TO_evolution_diff.pdf"
            file_path = f"{run_folder}/{file_name}"
            save_pdf(file_path)
            # plt.show()

        # plot gray and massfrac
        color_list = [red_dark, magenta_dark, yellow_dark, cyan_dark]
        grey = np.array(grey)
        m,n = grey.shape
        fig, ax = plt.subplots(figsize=(8, 6))
        anno_loc = [-0.15,0.99]
        xlim = [0,iter[0]]
        # ylim = [0,1]
        ax.plot(wc, label=r"wc", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("wc")
        ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
        file_name = "TO_evolution_wc.pdf"
        file_path = f"{run_folder}/{file_name}"
        save_pdf(file_path)
        # plt.show()

        # plot number of active (non-void) elements (left axis) + active fraction (right axis)
        if 'active_total' in timeHistory.keys():
            active = np.array(timeHistory['active_total'], dtype=float)
            fig, ax = plt.subplots(figsize=(8, 6))
            xlim = [0, iter[0]]
            l1, = ax.plot(active, label=r"Active elements", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Number of active elements")
            ax.set_xlim(xlim)
            lines = [l1]
            if 'count_total' in timeHistory.keys():
                total = np.array(timeHistory['count_total'], dtype=float)
                frac = np.divide(active, total, out=np.zeros_like(active), where=total > 0)
                ax2 = ax.twinx()   # active fraction on the right axis
                l2, = ax2.plot(frac, label=r"Active fraction", color=red_dark, linestyle='--', linewidth=curve_linewidth)
                ax2.set_ylabel("Active fraction")
                ax2.set_ylim([0, 1])
                lines.append(l2)
            # third axis: per-epoch compute time (ms). It should DECREASE as void
            # exclusion shrinks the active-element set. Raw (light) shows the mask-
            # refresh spikes; the moving average (dark) shows the underlying trend.
            if 'epoch_time' in timeHistory.keys():
                et = np.array(timeHistory['epoch_time'], dtype=float) * 1e3   # seconds -> ms
                ax3 = ax.twinx()
                ax3.spines['right'].set_position(('outward', 60))   # offset the 3rd axis
                ax3.plot(et, color=green_light, linestyle='-', linewidth=curve_linewidth * 0.5, alpha=0.35)  # raw
                if len(et) > 51:
                    ets = moving_average(et, 51)
                    l3, = ax3.plot(np.arange(len(ets)) + 25, ets, label=r"Epoch time (ms)",
                                   color=green_dark, linestyle='-', linewidth=curve_linewidth)
                else:
                    l3, = ax3.plot(et, label=r"Epoch time (ms)", color=green_dark,
                                   linestyle='-', linewidth=curve_linewidth)
                ax3.set_ylabel("Epoch time (ms)")
                lines.append(l3)
            legend = ax.legend(lines, [ln.get_label() for ln in lines], frameon=True, facecolor='white', edgecolor='black', framealpha=1)
            legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
            file_name = "TO_evolution_active.pdf"
            file_path = f"{run_folder}/{file_name}"
            save_pdf(file_path, bbox_inches='tight')
            plt.close(fig)
            # plt.show()

        if 'f' in timeHistory.keys():
            # plot gray and massfrac
            color_list = [red_dark, magenta_dark, yellow_dark, cyan_dark]
            grey = np.array(grey)
            m,n = grey.shape
            fig, ax = plt.subplots(figsize=(8, 6))
            anno_loc = [-0.15,0.99]
            xlim = [0,iter[0]]
            # ylim = [0,1]
            ax.plot(f, label=r"f", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("f")
            ax.set_xlim(xlim)
            # ax.set_ylim(ylim)
            legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
            legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box
            file_name = "TO_evolution_f.pdf"
            file_path = f"{run_folder}/{file_name}"
            save_pdf(file_path)
            # plt.show()



#!/usr/bin/env python3
"""Regenerate print-sized manuscript plots from measured and CPU CSV evidence."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2A56A0", "#D65B49", "#317E53", "#9A5EA6", "#D28F2C", "#4896AE"]


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def clean(ax):
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.savefig(path, dpi=300)
    plt.close(fig)


def residual_map(records, path, controlled=False):
    fig, ax = plt.subplots(figsize=(7.3, 4.4), constrained_layout=True)
    grouping = "family" if controlled else "matrix"
    px, py = ("rho_l", "rho_k") if controlled else ("exact_projection_residual", "exact_rank_residual")
    for i, name in enumerate(dict.fromkeys(r[grouping] for r in records)):
        rr = [r for r in records if r[grouping] == name]
        label = name.replace("projection-limited spectrum", "Unresolved capture spectrum").replace("rank-limited spectrum", "Rank-tail spectrum")
        ax.scatter([float(r[px]) for r in rr], [float(r[py]) for r in rr], s=16,
                   alpha=.7, color=COLORS[i], label=label)
    ax.plot([0, 1], [0, 1], color="#666666", linewidth=1)
    ax.axvline(.2, color="#666666", linestyle="--", linewidth=1)
    ax.axhline(.2, color="#666666", linestyle="--", linewidth=1)
    ax.set(xlim=(-.02,1.02), ylim=(-.02,1.02),
           xlabel=r"Projection residual $\rho_\ell$", ylabel=r"Target-rank residual $\rho_k$")
    if controlled:
        ax.legend(loc="upper left", frameon=False, fontsize=11, markerscale=1.5)
    else:
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3,
                  frameon=False, fontsize=11, markerscale=1.5)
    clean(ax)
    save(fig, path)


def rank_curves(summary, path):
    fig, ax = plt.subplots(figsize=(7.3,4.3), constrained_layout=True)
    for i, name in enumerate(dict.fromkeys(r["matrix"] for r in summary)):
        rr = sorted([r for r in summary if r["matrix"] == name], key=lambda r:int(r["rank"]))
        ax.errorbar([int(r["rank"]) for r in rr], [float(r["best_rho_k_mean"]) for r in rr],
                    yerr=[float(r["best_rho_k_ci95_halfwidth"]) for r in rr],
                    marker="o", linewidth=1.5, capsize=3, color=COLORS[i], label=name)
    ax.axhline(.2, color="#666666", linestyle="--", linewidth=1)
    ax.set(ylim=(-.025,1.025), xlabel="Requested rank k", ylabel=r"Best tested $\rho_k$")
    ax.set_xticks([8,16,32,64])
    ax.legend(frameon=False, ncol=2, fontsize=11)
    clean(ax)
    save(fig,path)


def cost_map(records,path):
    fig, ax = plt.subplots(figsize=(7.3,4.7), constrained_layout=True)
    names = list(dict.fromkeys(r["matrix"] for r in records))
    cmap = plt.get_cmap("tab10")
    for i,name in enumerate(names):
        rr=[r for r in records if r["matrix"]==name and r.get("state_status","executed")=="executed"]
        ax.scatter([float(r["gpu_pipeline_ms"]) for r in rr], [float(r["exact_rank_residual"]) for r in rr],
                   s=13, alpha=.75, color=cmap(i), label=name)
        best=min(rr,key=lambda r:float(r["exact_rank_residual"]))
        ax.scatter(float(best["gpu_pipeline_ms"]),float(best["exact_rank_residual"]),
                   s=70,marker="*",edgecolors="black",linewidth=.6,color=cmap(i),zorder=5)
    ax.set_yscale("log")
    ax.set(xlabel="Probe-inclusive GPU interval (ms)",ylabel=r"Full-energy $\rho_k$")
    ax.legend(frameon=False,ncol=3,fontsize=10,loc="lower center",bbox_to_anchor=(.5,1.02))
    clean(ax)
    save(fig,path)


def matrix_name(row):
    return row["input"].replace("\\","/").rsplit("/",1)[-1].removesuffix(".mtx")


def stage_comparison(r4090,ra800,path):
    left={matrix_name(r):r for r in r4090}
    right={matrix_name(r):r for r in ra800}
    candidates=["west2021","lp_osa_30","lp_pds_02","ca-HepPh","ca-AstroPh","email-EuAll"]
    names=[n for n in candidates if n in left and n in right]
    if len(names)!=6:
        raise ValueError("missing paired representative inputs")
    for name in names:
        for key in ["rank","width","power_steps","seed"]:
            if left[name][key]!=right[name][key]:
                raise ValueError("unmatched platform state")
    stages=[("spmm_forward_ms","Forward products"),("spmm_transpose_ms","Transpose products"),
            ("qr_ms","QR"),("svd_ms","Projected spectrum")]
    fig,axes=plt.subplots(1,2,figsize=(8.0,4.6),sharex=True,sharey=True,constrained_layout=True)
    y=np.arange(len(names))
    for ax,data,title in [(axes[0],left,"RTX 4090D"),(axes[1],right,"A800")]:
        accum=np.zeros(len(names))
        for i,(key,label) in enumerate(stages):
            values=np.array([float(data[name][key]) for name in names])
            ax.barh(y,values,left=accum,color=COLORS[i],label=label,height=.55)
            accum+=values
        ax.set_title(title)
        ax.set_xlabel("GPU stage time (ms)")
        ax.set_yticks(y,names)
        ax.invert_yaxis() if ax is axes[0] else None
        clean(ax)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,ncol=2,loc="outside lower center",fontsize=11)
    save(fig,path)


def scale_curves(summary,path):
    fig,axes=plt.subplots(1,2,figsize=(8.0,3.5),constrained_layout=True)
    for i,family in enumerate(dict.fromkeys(r["family"] for r in summary)):
        rr=sorted([r for r in summary if r["family"]==family and r["power_steps"]=="2"
                   and int(r["n_valid"])>0],key=lambda r:int(r["target_rank"]))
        x=[int(r["target_rank"]) for r in rr]
        for ax,key in zip(axes,["mean_rank_k_residual","mean_gpu_pipeline_ms"]):
            ax.plot(x,[float(r[key]) for r in rr],color=COLORS[i],marker="o",label=family)
    axes[0].set_ylabel(r"Target-rank residual $\rho_k$")
    axes[1].set_ylabel("GPU interval (ms)")
    for ax in axes:
        ax.set_xlabel("Requested rank k")
        clean(ax)
    axes[0].legend(frameon=False,fontsize=10)
    save(fig,path)


def graph_rank_power(summary,path):
    fig,axes=plt.subplots(1,2,figsize=(8.0,3.6),constrained_layout=True)
    for i,name in enumerate(dict.fromkeys(r["matrix"] for r in summary)):
        by_rank=sorted([r for r in summary if r["matrix"]==name and r["power_steps"]=="2"],key=lambda r:int(r["rank"]))
        by_power=sorted([r for r in summary if r["matrix"]==name and r["rank"]=="16"],key=lambda r:int(r["power_steps"]))
        axes[0].plot([int(r["rank"]) for r in by_rank],[float(r["mean_rho_k"]) for r in by_rank],marker="o",color=COLORS[i],label=name)
        axes[1].plot([int(r["power_steps"]) for r in by_power],[float(r["mean_rho_k"]) for r in by_power],marker="o",color=COLORS[i],label=name)
    axes[0].set_title("(a) Rank and width at q=2")
    axes[0].set_xlabel("Requested rank k")
    axes[0].set_xticks([8,16,32])
    axes[1].set_title("(b) Power response at k=16")
    axes[1].set_xlabel("Power steps q")
    axes[1].set_xticks([0,1,2])
    for ax in axes:
        ax.set_ylabel(r"Target-rank residual $\rho_k$")
        clean(ax)
    axes[0].legend(frameon=False,fontsize=10)
    save(fig,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root",type=Path,required=True)
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--rtx4090-csv",type=Path,required=True)
    parser.add_argument("--a800-csv",type=Path,required=True)
    args=parser.parse_args()
    root,assets=args.work_root,args.assets
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,
                         "axes.labelsize":13,"axes.titlesize":13,"legend.fontsize":11})
    residual_map(rows(root/"review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv"),assets/"fig1_residual_diagnostic.png")
    rank_curves(rows(root/"review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_summary.csv"),assets/"fig2_rank_curves.png")
    residual_map(rows(assets/"controlled_spectrum_diagnosis.csv"),assets/"fig6_controlled_spectra.png",controlled=True)
    cost_map(rows(root/"review_experiments/round16_exact_fixed_grid/fixed_grid.csv"),assets/"fig3_fixed_grid_frontier.png")
    stage_comparison(rows(args.rtx4090_csv),rows(args.a800_csv),assets/"fig4_cross_platform_stages.png")
    scale_curves(rows(assets/"extended_scale_summary.csv"),assets/"fig10_extended_scale.png")
    graph_rank_power(rows(assets/"nonlp_rank_sweep/nonlp_rank_sweep_summary.csv"),assets/"nonlp_rank_sweep/fig22_nonlp_rank_sweep.png")
    print("Regenerated seven measured-data manuscript figures at print size.")


if __name__=="__main__":
    main()

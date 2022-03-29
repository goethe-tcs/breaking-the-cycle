#!/usr/bin/env python3
import os
import platform
import argparse
import re
import subprocess
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns

ARG_PARSER = argparse.ArgumentParser(description="Evaluation script for 'bench-par-vs-seq' experiment logs")
ARG_PARSER.add_argument('--cutoff', '-c', type=int, default=5, help='All datapoints with an elapsed_sec_algo smaller thant this threshold are ignored.')
ARG_PARSER.add_argument('--plot-width', type=float, default=5.5, help='Width of a single plot of the figure')
ARG_PARSER.add_argument('--plot-height', type=float, default=4.3, help='Height of a single plot of the figure')
ARG_PARSER.add_argument('logs', nargs='+', help='One or multiple paths to log directories of the experiment (relative to the project directory)')


def load_csv(path: str):
    df = pd.read_csv(path)
    df = df.rename(columns={'elapsed_sec_algo': 'elapsed_sec'})
    df = df[['graph', 'elapsed_sec']]  # drop all unneeded columns

    thread_match = re.search('p_([0-9]*)_evals_[0-9]*_arr_[0-9]*\\.csv', path)
    if thread_match is None:
        raise ValueError(f"Failed to extract thread number from path '{path}'!")
    df['thread_count'] = int(thread_match.group(1))
    return df


def get_data(job_path: str, cutoff: int):
    # load all csv files
    par_data_frames = []
    for csv_file in glob(f'{job_path}/*.csv'):
        par_data_frames.append(load_csv(csv_file))
    data = pd.concat(par_data_frames)

    # retrieve sequential (1 thread) rows & calculate mean of sequential elapsed sec
    seq_data = data[data['thread_count'] == 1]
    seq_data = seq_data.groupby('graph').aggregate({'elapsed_sec': 'mean'})
    data['elapsed_sec_seq_mean'] = data.apply(lambda x: seq_data.loc[x['graph']].elapsed_sec, axis=1)

    #
    data['factor'] = data['elapsed_sec'] / data['elapsed_sec_seq_mean']
    data = data[data.elapsed_sec_seq_mean >= cutoff]
    return data


def sns_violinplot(x: str, y: str, ax: plt.Axes, data: pd.DataFrame, evenspacing=False):
    df = pd.DataFrame()
    df['x'] = data[x]
    df['y'] = data[y]

    color = (0.73333333, 0.835294118, 0.909803922)
    sns.violinplot(ax=ax, data=df, width=0.9, scale='width', linewidth=1, color=color)

    labels = list(data[x])
    if evenspacing:
        ticks = np.arange(1, len(labels) + 1)
    else:
        ticks = labels
    ax.set_xticks(ticks, labels=labels)


def plt_violinplot(y: str, x: str, ax: plt.Axes, data: pd.DataFrame, evenspacing=False):
    data = data.groupby(x)[y].aggregate(list)
    labels = list(data.index)

    if evenspacing:
        ticks = np.arange(1, len(labels) + 1)
        positions = None
    else:
        ticks = labels
        positions = labels

    parts = ax.violinplot(data, positions=positions, showmedians=True, showextrema=True, widths=2)
    ax.set_xticks(ticks, labels=labels)
    for line_collection in [parts['cbars'], parts['cmins'], parts['cmaxes']]:
        line_collection.set_linewidth(1)
    parts['cmedians'].set_linewidth(2)


def all_designpoints_plot(ax: plt.Axes, df: pd.DataFrame):
    color_count = df['thread_count'].nunique()
    palette = sns.color_palette('flare', color_count)
    sns.scatterplot(data=df, x='elapsed_sec_seq_mean', y='factor', hue='thread_count', palette=palette, ax=ax)
    ax.legend(title='Thread Anzahl', title_fontsize='small', fontsize='small')

    ax.set_title('relative Laufzeitsteigerung einzelner Designpunkte\n(1 Punkt pro Designpunkt)')
    ax.set_xscale('log')
    ax.set_ylabel('t_parallel / avg(t_sequentiell)')
    ax.set_xlabel('t_sequentiell (Sekunden)')
    ax.yaxis.set_major_locator(plticker.MaxNLocator(min_n_ticks=10))


def individual_slowdown_plot(ax: plt.Axes, df: pd.DataFrame):
    plt_violinplot(y='factor', x='thread_count', ax=ax, data=df)

    ax.set_title('relative Laufzeitsteigerung einzelner Designpunkte\n(gruppiert nach Thread Anzahl)')
    ax.set_ylabel('t_parallel / avg(t_sequentiell)')
    ax.set_xlabel('Thread Anzahl')
    ax.yaxis.set_major_locator(plticker.MaxNLocator(min_n_ticks=10))
    ax.xaxis.set_major_locator(plticker.FixedLocator([1, 5, 10, 15, 20, 25, 30, 35, 40]))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(1))


def overall_speedup_plot(ax: plt.Axes, job_path: str):
    # load job log file
    log_file_paths = glob(f'{job_path}/*.log')
    if len(log_file_paths) == 0:
        print("No log files found, skipping improvement plot!")
        return

    bench_thread_counts = []
    bench_total_elapsed_sec = []
    for file_path in log_file_paths:
        with open(file_path, "r") as log_file:
            log = log_file.read().replace('\n', '')

            thread_count = re.search('Running benchmark with [0-9]* design points using ([0-9]*) threads', log)
            if thread_count is None:
                raise ValueError(f"Failed to extract thread count from file '{file_path}'!")
            bench_thread_counts.append(int(thread_count.group(1)))

            elapsed_sec = re.search('Benchmark finished in ([0-9]*\\.[0-9]*) seconds', log)
            if elapsed_sec is None:
                raise ValueError(f"Failed to extract elapsed sec from file '{file_path}'!")
            bench_total_elapsed_sec.append(float(elapsed_sec.group(1)))

    rows = list(zip(bench_thread_counts, bench_total_elapsed_sec))
    df = pd.DataFrame(data=rows, columns=['thread_count', 'total_elapsed_sec'])

    df_seq = df[df['thread_count'] == 1]
    total_elapsed_sec_seq_mean = df_seq['total_elapsed_sec'].mean()

    df['factor'] = df['total_elapsed_sec'] / total_elapsed_sec_seq_mean
    df.sort_index(inplace=True)

    # draw
    plt_violinplot(y='factor', x='thread_count', ax=ax, data=df)
    ax.set_title('relative Laufzeitverringerung eines ganzen Experiments\n(verglichen mit sequentieller Gesamtlaufzeit)')
    ax.set_ylabel('t_parallel_gesamt / t_sequentiell_gesamt')
    ax.set_xlabel('Thread Anzahl')
    ax.set_yscale('log')

    ax.yaxis.set_major_locator(plticker.LogLocator(10, subs=np.arange(0.0, 1.1, 0.1)))
    ax.yaxis.set_minor_locator(plticker.LogLocator(10, subs=np.arange(0.0, 1.1, 0.05)))
    ax.yaxis.set_major_formatter(plticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_minor_formatter(plticker.NullFormatter())

    ax.xaxis.set_major_locator(plticker.FixedLocator([1, 5, 10, 15, 20, 25, 30, 35, 40]))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(1))


def main():
    args = ARG_PARSER.parse_args()
    script_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(f'{script_path}/../..')

    data_sets = [(i, job_path, get_data(job_path, args.cutoff)) for (i, job_path) in enumerate(args.logs)]

    fig_size = (len(data_sets) * args.plot_width, 3 * args.plot_height)
    fig, axs = plt.subplots(3, len(data_sets), figsize=fig_size, dpi=300, sharey='row', sharex='none', squeeze=False)
    grid = plt.GridSpec(3, len(data_sets))

    for (i, job_path, df) in data_sets:
        # remove trailing slashes from log directory paths
        job_path = job_path.rstrip('/')
        # retrieve last directory name as job name
        job_name = os.path.basename(os.path.normpath(job_path))

        # add subplot for column title
        row = fig.add_subplot(grid[0, i:i+1])
        row.set_title(f'{job_name}\n\n\n', fontweight='semibold')
        # hide subplot itself
        row.set_frame_on(False)
        row.axis('off')

        all_designpoints_plot(axs[0, i], df)
        individual_slowdown_plot(axs[1, i], df)
        overall_speedup_plot(axs[2, i], job_path)

        for row in axs:
            for ax in row:
                ax.grid(axis='y', alpha=0.4)

    title = f'Laufzeit Veränderung durch Parallelisierung\nDatenpunkte mit Laufzeit < {args.cutoff} Sekunden entfernt'
    fig.suptitle(title, fontsize=15)
    fig.set_facecolor('w')
    fig.tight_layout()
    fig_file_path = f'{args.logs[0]}/fig_cutoff_{args.cutoff}.png'
    fig.savefig(fig_file_path)
    print(f"Saved figure to '{fig_file_path}'")

    # open figure
    if platform.system() == 'Darwin':       # macOS
        subprocess.call(('open', fig_file_path))
    elif platform.system() == 'Windows':    # Windows
        os.startfile(fig_file_path)
    else:                                   # linux variants
        subprocess.call(('xdg-open', fig_file_path))


main()

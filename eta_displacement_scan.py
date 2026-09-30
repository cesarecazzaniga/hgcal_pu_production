#!/usr/bin/env python3
"""Cross-check the eta dependence of the displacement distribution directly
from gensim-particle truth (npz), with NO recipe/sampler involved.

Purpose: sample_local_pu_around_probe.py's synthetic test showed the
displacement shape bulk shifting to larger d as the probe moves forward
(higher |eta|), attributed to p = pt*cosh(eta) growing with |eta| under a
momentum-dependent displacement model. This script tests that directly on
the npz truth: apply a sequence of eta windows marching toward the forward
region and overlay the resulting displacement shapes, plus a summary trend
of mean log10(d3d) and displaced fraction vs eta.

By default the shape/trend plots use only DISPLACED particles (d3d above
--displaced_threshold), matching the recipe's own displacement tree, which
is built only from that displaced population. Pass --all_particles to
additionally make an "_all" version of both plots built from EVERY
particle in the window, displaced or not -- useful to see the full
displacement distribution (the large non-displaced/prompt spike plus the
displaced tail) rather than only the tail's shape. Since log10(0) is
undefined, particles at or below 10**d_range_lo are clipped into the
lowest bin (an "immeasurably small displacement" underflow) rather than
dropped; the console output reports how many were clipped per window.

Usage:
    python eta_displacement_scan.py --npz gensim_particles.npz \\
        --eta_edges 0.0,0.5,1.0,1.5,2.0,2.5,3.0 --outdir eta_scan --all_particles

--eta_edges defines N windows [e_i, e_{i+1}) in |eta|; particles are folded
to |eta| (both detector endcaps combined) unless --no_fold is given, in
which case eta windows are taken literally (signed).
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

from sample_pu_event import load_npz, SPECIES_GROUPS, _group_for_pid

hep.style.use("CMS")


def _add_cms_label(ax, label='Simulation', **kwargs):
    hep.cms.label("Preliminary", ax=ax, data=False, **kwargs, fontsize=16, com=14)


def _save(fig, outfile):
    fig.savefig(outfile, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Wrote {outfile}')


def eta_window_mask(eta, lo, hi, fold):
    e = np.abs(eta) if fold else eta
    return (e >= lo) & (e < hi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', required=True)
    p.add_argument('--displaced_threshold', type=float, default=1e-2,
                    help='d3d [cm] above which a particle counts as displaced (default 1e-2, i.e. 100 um)')
    p.add_argument('--eta_edges', type=str, default='0.0,0.5,1.0,1.5,2.0,2.5,3.0',
                    help='comma-separated increasing eta edges defining sequential windows '
                         '[e_i, e_{i+1}) marching toward the forward region')
    p.add_argument('--no_fold', action='store_true',
                    help='use signed eta windows instead of folding to |eta|')
    p.add_argument('--pdgid_group', choices=[g[0] for g in SPECIES_GROUPS] + ['all', 'other'], default='all',
                    help='restrict to one SPECIES_GROUPS category (photon/electron/muon/hadron), '
                         'or "all" particles (default)')
    p.add_argument('--pt_min', type=float, default=None, help='optional pt [GeV] lower cut')
    p.add_argument('--n_bins', type=int, default=40, help='number of log10(d3d) bins for shape plots')
    p.add_argument('--d_range', type=str, default='-4,2',
                    help='log10(d3d/cm) range for the shape histograms, "lo,hi"')
    p.add_argument('--outdir', default='.')
    p.add_argument('--cms_label', default='Simulation')
    p.add_argument('--all_particles', action='store_true',
                    help='also make "_all" shape/trend plots built from every particle in each '
                         'window (displaced or not), not only the displaced tail')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    edges = np.array([float(x) for x in args.eta_edges.split(',')])
    if len(edges) < 2:
        raise SystemExit('--eta_edges needs at least 2 values to define one window')
    fold = not args.no_fold

    npz = load_npz(args.npz, args.displaced_threshold)

    sel = np.ones(len(npz['eta']), dtype=bool)
    if args.pdgid_group != 'all':
        group_of = np.array([_group_for_pid(pid) for pid in npz['pdgId']])
        sel &= (group_of == args.pdgid_group)
    if args.pt_min is not None:
        sel &= npz['pt'] >= args.pt_min

    d_lo, d_hi = (float(x) for x in args.d_range.split(','))
    disp_edges = np.linspace(d_lo, d_hi, args.n_bins + 1)
    centers = .5 * (disp_edges[:-1] + disp_edges[1:])

    cmap = plt.get_cmap('viridis')
    n_windows = len(edges) - 1
    colors = [cmap(i / max(1, n_windows - 1)) for i in range(n_windows)]
    floor = disp_edges[0]  # log10(d3d) floor; particles at/below 10**floor clip into bin 0

    def run_scan(only_displaced, suffix, ylabel_extra):
        """Build the shape-overlay and trend plots for either the
        displaced-only population (only_displaced=True) or every particle
        in each window (False, with d3d clipped to the floor so d3d<=0 or
        very small values land in the lowest bin instead of being dropped
        by log10)."""
        fig, ax = plt.subplots(figsize=(9, 7))
        summary = []  # (eta_lo, eta_hi, eta_center, n_in_window, n_displaced,
                       #  displaced_frac, mean_log10d, median_log10d, n_clipped)
        for i in range(n_windows):
            lo, hi = edges[i], edges[i + 1]
            m = sel & eta_window_mask(npz['eta'], lo, hi, fold)
            n_in_window = int(m.sum())
            n_displaced = int((m & npz['displaced']).sum())
            if n_in_window == 0:
                print(f'window [{lo:.2f}, {hi:.2f}): 0 particles, skipping')
                continue
            displaced_frac = n_displaced / n_in_window
            if only_displaced:
                d3d_sel = npz['d3d'][m & npz['displaced']]
                n_clipped = 0
            else:
                d3d_sel = npz['d3d'][m]
                n_clipped = int((d3d_sel <= 10.**floor).sum())
            n_used = len(d3d_sel)
            log_d = np.log10(np.maximum(d3d_sel, 10.**floor)) if n_used > 0 else np.array([])
            counts, _ = np.histogram(log_d, bins=disp_edges)
            widths = np.diff(disp_edges)
            area = (counts * widths).sum()
            shape = counts / area if area > 0 else counts.astype(float)
            label = (f'{lo:.2g}' + r'$\leq|\eta|<$' + f'{hi:.2g}' if fold
                      else f'{lo:.2g}' + r'$\leq\eta<$' + f'{hi:.2g}')
            ax.step(centers, shape, where='mid', color=colors[i], linewidth=1.8, label=label)
            summary.append((lo, hi, .5 * (lo + hi), n_in_window, n_displaced, displaced_frac,
                              float(np.mean(log_d)) if n_used > 0 else np.nan,
                              float(np.median(log_d)) if n_used > 0 else np.nan, n_clipped))

        if not only_displaced:
            ax.axvline(floor, color='gray', linewidth=1, linestyle=':')
            ax.annotate('clipped\nunderflow', xy=(floor, ax.get_ylim()[1]), xytext=(4, -4),
                        textcoords='offset points', fontsize=9, color='gray', va='top')
        ax.set_xlabel('log10(3D displacement / cm)')
        ax.set_ylabel('Displacement shape [a.u.]')
        #sert log scale for the y-axis if the shapes span several orders of magnitude
        ax.set_yscale('log')
        ax.legend(fontsize=11, ncol=2)
        _add_cms_label(ax, args.cms_label)
        _save(fig, os.path.join(args.outdir, f'eta_scan_displacement_shapes{suffix}.png'))

        eta_centers = [s[2] for s in summary]
        mean_logd = [s[6] for s in summary]
        median_logd = [s[7] for s in summary]
        disp_frac = [s[5] for s in summary]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True,
                                         gridspec_kw={'hspace': 0.15})
        ax1.plot(eta_centers, mean_logd, marker='o', color='#3f90da', label='mean log10(d3d)')
        ax1.plot(eta_centers, median_logd, marker='s', color='#bd1f01', label='median log10(d3d)')
        ax1.set_ylabel(f'log10(3D disp. / cm)\n({ylabel_extra})', fontsize=15)
        ax1.legend(fontsize=11)
        ax2.plot(eta_centers, disp_frac, marker='o', color='#3f90da')
        ax2.set_ylabel('displaced fraction', fontsize=15)
        ax2.set_xlabel(r'window center $|\eta|$' if fold else r'window center $\eta$')
        _add_cms_label(ax1, args.cms_label)
        _save(fig, os.path.join(args.outdir, f'eta_scan_displacement_trend{suffix}.png'))

        print(f'\neta window scan summary{suffix or " (displaced only)"}:')
        header = f'{"eta_lo":>8} {"eta_hi":>8} {"n_in_win":>10} {"n_disp":>8} {"disp_frac":>10} ' \
                 f'{"mean_log10d":>12} {"median_log10d":>14}'
        if not only_displaced:
            header += f' {"n_clipped":>10}'
        print(header)
        for lo, hi, c, n_in, n_disp, frac, mlogd, medlogd, n_clip in summary:
            row = f'{lo:8.2f} {hi:8.2f} {n_in:10d} {n_disp:8d} {frac:10.4f} {mlogd:12.4f} {medlogd:14.4f}'
            if not only_displaced:
                row += f' {n_clip:10d}'
            print(row)

    run_scan(only_displaced=True, suffix='', ylabel_extra='displaced only')
    if args.all_particles:
        run_scan(only_displaced=False, suffix='_all', ylabel_extra='all particles, clipped floor')


if __name__ == '__main__':
    main()
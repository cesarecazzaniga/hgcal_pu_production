#!/usr/bin/env python3
"""
characterize_gensim_particles.py

Reads the .npz produced by extract_gensim_particles.py and characterizes
what a single minbias interaction typically looks like -- density (dN/deta,
and per-event multiplicity, both overall and specifically within HGCAL's
eta acceptance), momentum spectrum, species composition, and production
vertices. This is the input the eventual local-PU gun's sampling recipe
needs (see this project's earlier pipeline discussion) -- these are
properties of ONE isolated minbias interaction (pileup_gensim.root has no
probe/signal particle in it at all), not "density around a probe" -- that
comparison only becomes meaningful once combined with a real probe
direction later.

VERTEX PLOTS (vx, vy, vz = GenParticle PRODUCTION vertex, in cm):
A particle's raw vertex position mixes two unrelated things: (1) where the
interaction itself happened (beamspot smearing: ~cm in z, ~10s of um in
x/y, possibly with an offset from the origin), and (2) how far the particle
was produced from that interaction point (decays in flight of its parent).
The raw R = sqrt(vx^2+vy^2) distribution is plotted since it is the direct
quantity, but to separate the two effects the script also estimates a
per-event primary vertex (PV) as the per-coordinate MEDIAN over the event's
particles -- robust as long as most particles in an event are prompt, which
holds for minbias -- and then plots
  * the PV z and x-y distributions (= the vertex smearing the gun needs to
    reproduce; z matters most for HGCAL, since it shifts where a given eta
    lands on the detector face), and
  * each particle's DISPLACEMENT from its event's PV (= the decay-in-flight
    part), overall and per species.
NOTE on what to expect: genParticles only contains decays done BY THE
GENERATOR. With the standard CMS Pythia settings only particles with
ctau < 10 mm are decayed there -- K0S, Lambda, charged pi/K etc. are left
stable and decayed later by GEANT (visible only in SimTrack/SimVertex). So
the displacement tail here comes from heavy flavour, taus and similar, and
is expected to be small; this is not the full picture of displaced
production in the detector.

Usage:
    python characterize_gensim_particles.py \\
        --npz gensim_particles.npz \\
        --outdir gensim_characterization/
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

# HGCAL EE acceptance -- the physically relevant window for this whole
# project, shown as a shaded band on the density plot and used to compute
# the "multiplicity within HGCAL acceptance" summary number.
HGCAL_ETA_MIN = 1.5
HGCAL_ETA_MAX = 3.0

PDGID_NAMES = {
    11: 'e-', -11: 'e+', 13: 'mu-', -13: 'mu+', 22: 'gamma',
    111: 'pi0', 211: 'pi+', -211: 'pi-',
    321: 'K+', -321: 'K-', 130: 'K0L', 310: 'K0S',
    2212: 'p', -2212: 'pbar', 2112: 'n', -2112: 'nbar',
    12: 'nu_e', -12: 'nu_e_bar', 14: 'nu_mu', -14: 'nu_mu_bar',
    16: 'nu_tau', -16: 'nu_tau_bar',
    }


def pdgid_label(pdgid):
    return PDGID_NAMES.get(int(pdgid), str(int(pdgid)))


def _add_cms_label(ax, label='Simulation', **kwargs):
    hep.cms.label("Preliminary", ax=ax, data=False, **kwargs, fontsize=16, com=14)


def _save(fig, outfile):
    fig.savefig(outfile, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Wrote {outfile}')


def estimate_event_pv(event_idx, v, n_events):
    """
    Per-event median of one vertex coordinate, vectorized (no python loop
    over events): sort by (event, value), then pick each event's middle
    entry. Events with no particles get NaN.
    """
    order = np.lexsort((v, event_idx))
    counts = np.bincount(event_idx, minlength=n_events)
    starts = np.cumsum(counts) - counts
    pv = np.full(n_events, np.nan)
    has = counts > 0
    pv[has] = v[order][starts[has] + counts[has] // 2]
    return pv


def plot_log_spectrum_by_species(values, pdgId, species_ids, xlabel, outfile, cms_label,
                                  vmin, legend_title=None):
    """log-log step histogram of `values` (> vmin only): all particles + given species."""
    fig, ax = plt.subplots(figsize=(10, 7))
    keep = values > vmin
    if keep.sum() < 2:
        print(f'Skipping {outfile}: fewer than 2 particles with value > {vmin}.')
        plt.close(fig)
        return
    bins = np.logspace(np.log10(vmin), np.log10(values[keep].max() * 1.001), 50)
    ax.hist(values[keep], bins=bins, histtype='step', linewidth=2, color='black', label='all')
    for pid in species_ids:
        sel = keep & (pdgId == pid)
        if sel.sum() < 2:
            continue
        ax.hist(values[sel], bins=bins, histtype='step', linewidth=1.5, label=pdgid_label(pid))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Particles')
    ax.legend(fontsize=12, title=legend_title, title_fontsize=12)
    _add_cms_label(ax, cms_label)
    _save(fig, outfile)


def characterize_vertices(d, event_idx, pdgId, n_events, top_ids, labels_for_print, args):
    vx = d['vx'].astype(np.float64)
    vy = d['vy'].astype(np.float64)
    vz = d['vz'].astype(np.float64)

    # --- Per-event primary vertex (median over the event's particles) ---
    pv_x = estimate_event_pv(event_idx, vx, n_events)
    pv_y = estimate_event_pv(event_idx, vy, n_events)
    pv_z = estimate_event_pv(event_idx, vz, n_events)
    ok = np.isfinite(pv_z)

    print(f'\nEstimated primary vertex per event (median of particle vertices), '
          f'{ok.sum()} events with >=1 particle:')
    for name, pv in (('x', pv_x), ('y', pv_y), ('z', pv_z)):
        print(f'  PV {name}: mean={pv[ok].mean():+.5f} cm  std={pv[ok].std():.5f} cm')
    if pv_z[ok].std() == 0:
        print('  NOTE: all PVs identical -- vertex smearing was apparently not applied in this '
              'sample; the PV plots below are then trivial.')

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(pv_z[ok], bins=60, color='#3f90da', edgecolor='none')
    ax.set_xlabel('Primary vertex z [cm]')
    ax.set_ylabel('Events')
    ax.legend([f'mean = {pv_z[ok].mean():.2f} cm, std = {pv_z[ok].std():.2f} cm'], fontsize=13)
    _add_cms_label(ax, args.cms_label)
    _save(fig, os.path.join(args.outdir, 'vertex_pv_z.png'))

    fig, ax = plt.subplots(figsize=(9, 8))
    h = ax.hist2d(pv_x[ok], pv_y[ok], bins=50, cmap='viridis', cmin=1)
    ax.set_xlabel('Primary vertex x [cm]')
    ax.set_ylabel('Primary vertex y [cm]')
    ax.ticklabel_format(style='sci', scilimits=(-2, 2))
    fig.colorbar(h[3], ax=ax, label='Events')
    _add_cms_label(ax, args.cms_label)
    _save(fig, os.path.join(args.outdir, 'vertex_pv_xy.png'))

    # --- Raw production radius R = sqrt(vx^2 + vy^2), w.r.t. the detector origin ---
    r_raw = np.hypot(vx, vy)
    print(f'\nRaw production radius R=sqrt(vx^2+vy^2): median={np.median(r_raw):.5f} cm  '
          f'99%={np.percentile(r_raw, 99):.5f} cm  max={r_raw.max():.4f} cm')
    plot_log_spectrum_by_species(
        r_raw, pdgId, top_ids[:4],
        xlabel='Production radius R = sqrt(vx^2+vy^2) [cm]',
        outfile=os.path.join(args.outdir, 'vertex_R.png'),
        cms_label=args.cms_label, vmin=1e-4,
        legend_title='w.r.t. detector origin (includes beamspot)',
        )

    # --- Displacement from the event's own PV (removes the beamspot part) ---
    dx = vx - pv_x[event_idx]
    dy = vy - pv_y[event_idx]
    dz = vz - pv_z[event_idx]
    d_xy = np.hypot(dx, dy)
    d_3d = np.sqrt(dx**2 + dy**2 + dz**2)

    thr = args.displaced_threshold
    displaced = d_3d > thr
    print(f'\nDisplacement from event PV (3D distance > {thr:g} cm counts as displaced):')
    print(f'  {"all":>8}: {100. * displaced.mean():6.2f}% displaced')
    for pid, lab in zip(top_ids, labels_for_print):
        sel = pdgId == pid
        med = np.median(d_3d[sel & displaced]) if (sel & displaced).any() else float('nan')
        print(f'  {lab:>8}: {100. * displaced[sel].mean():6.2f}% displaced  '
              f'(median 3D displacement of those: {med:.4f} cm)')

    plot_log_spectrum_by_species(
        d_xy, pdgId, top_ids[:4],
        xlabel='Transverse displacement from event PV [cm]',
        outfile=os.path.join(args.outdir, 'vertex_displacement_xy.png'),
        cms_label=args.cms_label, vmin=thr,
        legend_title=f'{100. * (d_xy > thr).mean():.1f}% of particles above {thr:g} cm shown',
        )
    plot_log_spectrum_by_species(
        np.abs(dz), pdgId, top_ids[:4],
        xlabel='|z displacement| from event PV [cm]',
        outfile=os.path.join(args.outdir, 'vertex_displacement_z.png'),
        cms_label=args.cms_label, vmin=thr,
        legend_title=f'{100. * (np.abs(dz) > thr).mean():.1f}% of particles above {thr:g} cm shown',
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--npz', required=True, help="Output of extract_gensim_particles.py.")
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--n_top_species', type=int, default=8,
                         help="Number of individual species to show in the composition/spectrum "
                              "plots; everything else is grouped into a single 'other' bucket.")
    parser.add_argument('--displaced_threshold', type=float, default=1e-4,
                         help="Distance from the event PV [cm] above which a particle counts as "
                              "displaced, and lower edge of the displacement plots (default "
                              "1e-4 cm = 1 um; photons from pi0 decays sit at ~nm-um scale, "
                              "i.e. below this).")
    parser.add_argument('--cms_label', default='Simulation')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.npz)
    event_idx = d['event_idx']
    pdgId = d['pdgId']
    pt = d['pt']
    eta = d['eta']
    energy = d['energy']
    n_events = int(d['n_events'])

    print(f'{args.npz}: {len(pdgId)} particles across {n_events} events '
          f'({len(pdgId)/n_events:.1f} per event on average).')

    # --- Composition: fraction of each species, top N + "other" ---
    unique_ids, counts = np.unique(pdgId, return_counts=True)
    order = np.argsort(-counts)
    unique_ids, counts = unique_ids[order], counts[order]

    top_ids = unique_ids[:args.n_top_species]
    top_counts = counts[:args.n_top_species]
    other_count = counts[args.n_top_species:].sum() if len(counts) > args.n_top_species else 0

    labels = [pdgid_label(pid) for pid in top_ids]
    top_labels = list(labels)
    values = list(top_counts)
    if other_count > 0:
        labels.append('other')
        values.append(other_count)
    pct = 100. * np.array(values) / len(pdgId)

    print(f'\nComposition (top {args.n_top_species} species + other):')
    for lab, v, p in zip(labels, values, pct):
        print(f'  {lab:>8}: {v:>10} ({p:5.1f}%)')

    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(labels))
    ax.bar(x, pct, color='#3f90da', edgecolor='none')
    for xi, p in zip(x, pct):
        ax.text(xi, p, f'{p:.1f}%', ha='center', va='bottom', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel('Fraction of particles [%]')
    ax.set_ylim(0, pct.max() * 1.15)
    ax.legend([f'{len(pdgId)} particles, {n_events} events'], fontsize=13)
    _add_cms_label(ax, args.cms_label)
    _save(fig, os.path.join(args.outdir, 'composition.png'))

    # --- Density: dN/deta ---
    fig, ax = plt.subplots(figsize=(10, 7))
    n_bins = 60
    counts_eta, bin_edges = np.histogram(eta, bins=n_bins, range=(-5, 5))
    bin_width = bin_edges[1] - bin_edges[0]
    dn_deta = counts_eta / (n_events * bin_width)
    bin_centers = .5 * (bin_edges[:-1] + bin_edges[1:])
    ax.step(bin_centers, dn_deta, where='mid', color='#3f90da', linewidth=2)
    ax.axvspan(HGCAL_ETA_MIN, HGCAL_ETA_MAX, alpha=0.15, color='#bd1f01', label='HGCAL EE acceptance')
    ax.axvspan(-HGCAL_ETA_MAX, -HGCAL_ETA_MIN, alpha=0.15, color='#bd1f01')
    ax.set_xlabel('eta')
    ax.set_ylabel('dN/deta per event')
    ax.legend(fontsize=13)
    _add_cms_label(ax, args.cms_label)
    _save(fig, os.path.join(args.outdir, 'density_dNdeta.png'))

    # --- Multiplicity: per-event particle count, overall and within HGCAL acceptance ---
    mult_overall = np.bincount(event_idx, minlength=n_events)
    in_hgcal = (np.abs(eta) >= HGCAL_ETA_MIN) & (np.abs(eta) <= HGCAL_ETA_MAX)
    mult_hgcal = np.bincount(event_idx[in_hgcal], minlength=n_events)

    print(f'\nMultiplicity per event:')
    print(f'  overall:          mean={mult_overall.mean():.1f}  median={np.median(mult_overall):.1f}  '
          f'[{mult_overall.min()}, {mult_overall.max()}]')
    print(f'  within HGCAL EE:  mean={mult_hgcal.mean():.2f}  median={np.median(mult_hgcal):.1f}  '
          f'[{mult_hgcal.min()}, {mult_hgcal.max()}]  '
          f'(both endcaps combined, {HGCAL_ETA_MIN}<|eta|<{HGCAL_ETA_MAX})')

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(mult_hgcal, bins=np.arange(mult_hgcal.max() + 2) - 0.5, color='#3f90da', edgecolor='none')
    ax.set_xlabel(f'Particles per event within HGCAL EE acceptance '
                   f'({HGCAL_ETA_MIN}<|eta|<{HGCAL_ETA_MAX}, both endcaps)')
    ax.set_ylabel('Events')
    ax.legend([f'mean = {mult_hgcal.mean():.2f}'], fontsize=13)
    _add_cms_label(ax, args.cms_label)
    _save(fig, os.path.join(args.outdir, 'multiplicity_hgcal_acceptance.png'))

    # --- Spectrum: pt distribution, overall and for the top species ---
    fig, ax = plt.subplots(figsize=(10, 7))
    pt_positive = pt[pt > 0]
    bins = np.logspace(np.log10(max(pt_positive.min(), 1e-3)), np.log10(pt_positive.max()), 50)
    ax.hist(pt_positive, bins=bins, histtype='step', linewidth=2, color='black', label='all')
    for pid in top_ids[:4]:  # keep this one less busy -- top 4 species only
        sel = pdgId == pid
        if sel.sum() < 2:
            continue
        ax.hist(pt[sel][pt[sel] > 0], bins=bins, histtype='step', linewidth=1.5,
                label=pdgid_label(pid))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('pt [GeV]')
    ax.set_ylabel('Particles')
    ax.legend(fontsize=12)
    _add_cms_label(ax, args.cms_label)
    _save(fig, os.path.join(args.outdir, 'pt_spectrum.png'))

    # --- pt vs eta, to check whether the spectrum genuinely varies with
    # eta (informs whether the eventual gun's recipe needs to be
    # eta-binned or can use one global spectrum) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    hb = ax.hexbin(eta, np.clip(pt, 1e-3, None), gridsize=50, mincnt=1, cmap='viridis',
                    yscale='log', bins='log')
    ax.axvline(HGCAL_ETA_MIN, color='red', linestyle='--', linewidth=1)
    ax.axvline(HGCAL_ETA_MAX, color='red', linestyle='--', linewidth=1)
    ax.axvline(-HGCAL_ETA_MIN, color='red', linestyle='--', linewidth=1)
    ax.axvline(-HGCAL_ETA_MAX, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('eta')
    ax.set_ylabel('pt [GeV]')
    fig.colorbar(hb, ax=ax, label='particles per bin (log)')
    _add_cms_label(ax, args.cms_label)
    _save(fig, os.path.join(args.outdir, 'pt_vs_eta.png'))

    # --- Production vertices: PV (beamspot) distributions, raw R, displacement from PV ---
    if all(k in d.files for k in ('vx', 'vy', 'vz')):
        characterize_vertices(d, event_idx, pdgId, n_events, top_ids, top_labels, args)
    else:
        print('\nNo vx/vy/vz in this .npz -- skipping vertex plots.')


if __name__ == '__main__':
    main()
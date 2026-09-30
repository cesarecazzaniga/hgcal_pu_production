#!/usr/bin/env python3
"""
sample_pu_event.py

Prototype/reference implementation of "draw a full local-PU particle list
as if N pileup interactions occurred", reading the recipe file produced by
build_pu_sampling_recipe.py -- density, species, pt AND, new in this
version, DISPLACED PRODUCTION VERTEX. This is a direct blueprint for the
eventual C++ gun -- same steps (Poisson count per eta bin scaled by N,
species draw from the composition cumulative distribution, pt draw via
TH1::GetRandom()-equivalent, then a Bernoulli "is this particle displaced"
draw and, if so, a distance draw from the matching species+momentum
histogram), just prototyped/validated in Python first.

KEY STATISTICAL POINT (unchanged, see the conversation this was built from
for the full reasoning): the sum of N independent Poisson(mu) draws is
itself Poisson(N*mu) -- an exact property, not an approximation. So
scaling to "as if N_pu interactions" only means scaling each eta bin's
Poisson mean by N_pu; the composition/pt/displacement parts of the recipe
describe what ONE interaction's particles look like and don't change.

Two modes for how N_pu itself is chosen (--fluctuate):
  fixed:      exactly N_pu particles' worth of density every event.
  fluctuate:  N_pu itself is drawn once per event from Poisson(N_pu),
              shared across all eta bins for that event -- matches how
              real HL-LHC pileup fluctuates event to event.

DISPLACEMENT SAMPLING (new). After species (pdgId) and pt are drawn for a
particle, this script also draws:
  - eta: a REPRESENTATIVE value within the particle's eta bin, needed only
    to get a momentum p = pt*cosh(eta) for the displacement lookup below --
    NOT a finer-grained density model than the recipe's own eta bins (see
    _representative_eta). Uniform within the bin for any FINITE bin
    (including the central one straddling eta=0 -- an earlier version of
    this script forced eta=0 there instead, which biased p low for that
    bin's displacement lookup specifically; see _representative_eta's
    docstring for the fix and why it only affected displacement, not pt/
    eta/composition). Only the two open-ended outer bins, with no finite
    width to draw within, fall back to their one finite edge.
  - phi: uniform in [-pi, pi) (azimuthal symmetry of the collision).
  - is this particle displaced? Bernoulli draw using the "displacement"
    tree's row for this (pdgId, momentum) -- same lookup the C++ gun needs.
  - if displaced, a 3D distance from that row's own histogram (or the
    all-species fallback if hist_idx == -1), exactly as described in
    build_pu_sampling_recipe.py's docstring: pow(10, GetRandom()) on the
    log10(cm) histogram.
The displacement direction itself is always taken along the particle's own
momentum (cos(angle) = 1), matching the conclusion of the characterization
step (most real displaced particles satisfy this well); sampling the
measured angle spread instead (displacement_cos_angle) is not implemented
here to keep the prototype simple -- flag it if you want it added.

CONTROL PLOTS (--outdir): samples --n_events full events and compares the
AGGREGATE sampled output against TWO independent references:
  - "recipe": what the recipe file itself predicts, with no simulation of
    real data involved -- i.e. a closure test of THIS SCRIPT'S sampling
    logic against the numbers stored in the ROOT file.
  - "npz truth" (--npz): the same quantity measured straight from
    extract_gensim_particles.py's own output, with NO recipe involved --
    i.e. a check of the whole pipeline (extract -> characterize -> build
    recipe -> sample) against the ground truth, including whatever
    information the recipe's binning/aggregation steps threw away.
  ETA, PT AND DISPLACEMENT ARE SHAPE COMPARISONS, NOT ABSOLUTE RATES: each
  of the three curves (sampled, recipe, npz truth) in density_dNdeta.png,
  pt_spectrum*.png and displacement_log10d*.png is drawn on ONE shared set
  of bin edges (the recipe's own binning for that quantity -- the recipe
  file itself has nothing finer to offer for pt/eta, and its displacement
  histograms all share one log10(cm) binning by construction) and
  normalized to unit area before plotting, so what's compared is purely
  SHAPE: does the sampler reproduce the right eta/pt/displacement profile,
  independent of whether the overall rate is also right. Absolute-rate
  agreement is checked separately and is NOT visible in these three plots
  any more -- total_multiplicity.png covers the overall per-event count,
  composition.png covers per-species fractions (already a shape quantity),
  and the console momentum-dependence table covers the displaced FRACTION
  (a ratio, so also normalization-independent) per species. If you need to
  re-add an absolute check specifically for eta/pt/displacement shape,
  that's a real regression from an earlier version of this script that
  plotted per-event densities instead -- ask if you want a *_rate.png
  variant back alongside the shape one.
  Each shape plot below --outdir gets a ratio-of-shapes panel underneath:
  sampled/npz-truth and recipe/npz-truth, both against 1.0. Plots:
  - total_multiplicity.png: per-event total particle count vs the
    theoretical Poisson mean. No npz-truth line: total multiplicity is
    intrinsically an N_pu-scaled quantity, and a single-interaction npz
    file has no N_pu built in to compare against -- the one exception
    without a ratio panel, and the one plot left as an absolute check.
  - density_dNdeta.png: sampled vs recipe vs npz truth eta SHAPE, all
    three on the recipe's own (finite) eta bins.
  - composition.png: sampled vs npz vs recipe species fractions (already
    unit-normalized by construction, unchanged by the above).
  - pt_spectrum.png: sampled vs recipe vs npz truth pt SHAPE, all three on
    the recipe's own (linear + log-tailed) pt bins.
  - displacement_log10d.png: sampled vs npz truth vs a purely analytic
    "recipe" SHAPE curve, still built from aggregate_recipe_displacement's
    n_pu/n_particles/displaced_fraction weighting (see its docstring) --
    that weighting still sets the RELATIVE contribution of each (species,
    momentum-bin) row to the summed shape, it's only the OVERALL scale
    that _shape_density then normalizes away -- no sampling involved in
    this curve either, so this plot alone checks THREE independent things
    against each other.
  - pt_spectrum_<group>.png and displacement_log10d_<group>.png, one pair
    per species group in SPECIES_GROUPS (photon/electron/muon/hadron;
    anything else, mostly neutrinos, is skipped): the same two shape plots
    above, restricted to that group's own particles, on the SAME shared
    pt/displacement bin edges as the aggregate plots (not a per-group
    binning) so shapes are directly comparable group to group too. The
    displacement version always has a genuine ANALYTIC recipe shape curve
    (the recipe's displacement tree is already keyed by pdgId, see
    aggregate_recipe_displacement's pdgid_filter). The pt version has one
    too IF the recipe was built with --pt_by_species (see load_recipe /
    aggregate_recipe_pt_for_group); otherwise (older recipe files, or
    --pt_by_species not passed) it falls back automatically to sampled vs
    npz truth only, exactly as before. A group with zero particles in both
    sampled and npz is skipped with a console note rather than an empty
    plot.
  A per-species, per-momentum-bin table (npz-measured displaced fraction in
  several finer momentum bins vs the recipe's own coarser step vs this
  script's sampled fraction in the same fine bins) is PRINTED to the
  console instead of plotted, for the --n_console_species most abundant
  species -- this is where the recipe's momentum binning choice is most
  directly tested, but N species x several momentum bins x a ratio panel
  each was judged not worth the extra plot files for a prototype; ask if
  you want it as a figure instead.
Given N_pu=200 already gives ~thousands of particles per single sampled
event, --n_events in the tens is normally already enough for smooth
control plots -- no separate large-statistics mode needed.

PER-SPECIES PT (NEW): if the --recipe file was built with
build_pu_sampling_recipe.py's --pt_by_species flag, draw_pt (and hence
sample_one_event) now draws each particle's pt from ITS OWN species
group's histogram in that eta bin (photon/electron/muon/hadron/other, see
SPECIES_GROUPS/_group_for_pid), falling back to the bin's old aggregated
histogram wherever a species-group histogram is missing or empty. This is
purely a consumer-side change: an older recipe file (no --pt_by_species)
still works exactly as before, since bin_info['pt_by_group'] is then just
empty and every draw falls through to the aggregate histogram. The same
distinction also finally gives pt_spectrum_<group>.png a genuine "recipe
(own bins)" curve (aggregate_recipe_pt_for_group) whenever it's available.

GEOMETRIC ETA CUT AT BUILD TIME vs HGCAL-ONLY RESTRICTION HERE: if the
recipe was built with build_pu_sampling_recipe.py's --geometric_cut, the
recipe file itself already contains ONLY the HGCAL-region bins (no
central/outer bins to begin with) and its displacement tree already only
reflects HGCAL-region particles -- so this script's own
--hgcal_eta_min/--hgcal_eta_max below becomes a no-op (nothing left to
filter out) as long as the two ranges agree. The two flags solve the same
problem at different stages: --geometric_cut bakes the restriction into
the recipe file (and, unlike the flag below, actually restricts the
displacement tree's content too, not just which eta bins get sampled from
-- see build_pu_sampling_recipe.py's module docstring), while
--hgcal_eta_min/--hgcal_eta_max filters an already-built, unrestricted
recipe after the fact.

HGCAL-ONLY RESTRICTION (--hgcal_eta_min/--hgcal_eta_max): both the central
(barrel) eta bin and the open-ended very-forward outer bins get a cruder
eta treatment than the finite forward bins HGCAL actually cares about (see
_representative_eta's docstring) -- neither is HGCAL acceptance to begin
with. Rather than trying to model those regions better, passing both flags
restricts EVERYTHING (sampling, every control plot, the npz-truth
comparison, and the console table) to just the recipe's finite eta bins
lying entirely within hgcal_eta_min <= |eta| <= hgcal_eta_max, via
filter_bins_to_eta_range -- this sidesteps the barrel/outer-bin modeling
question entirely for anyone who only cares about validating the HGCAL
region the gun is actually being built for.

Usage:
    python sample_pu_event.py \
        --recipe pu_sampling_recipe.root \
        --npz gensim_particles.npz \
        --n_pu 200 \
        --n_events 20 \
        --outdir pu_sample_control_plots/
"""

import argparse
import os

import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")


def _add_cms_label(ax, label='Simulation', **kwargs):
    hep.cms.label("Preliminary", ax=ax, data=False, **kwargs, fontsize=16, com=14)


def _save(fig, outfile):
    fig.savefig(outfile, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Wrote {outfile}')


def _ratio_fig(figsize=(10, 8.5)):
    """Two-panel figure: main axes on top, a ratio axes below, sharing x."""
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.06})
    rax.axhline(1., color='black', linewidth=1, linestyle=':')
    rax.set_ylabel('ratio to\nnpz truth')
    return fig, ax, rax


def _safe_ratio(num, den):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(den > 0, num / den, np.nan)
    return r


# ---------------------------------------------------------------------------
# Vertex estimator, duplicated from build_pu_sampling_recipe.py so this
# script can measure "npz truth" displacement on its own, standalone.
# ---------------------------------------------------------------------------

def _mode_representative(event_idx, v, n_events, subset):
    """For each event, index (into the full arrays) of one particle
    belonging to the largest group of `subset` particles sharing a
    bit-identical `v`, and that group's size. -1 / 0 if none."""
    rep = np.full(n_events, -1, dtype=np.int64)
    n_mode = np.zeros(n_events, dtype=np.int64)
    sel = np.flatnonzero(subset)
    if len(sel) == 0:
        return rep, n_mode
    order = np.lexsort((v[sel], event_idx[sel]))
    idx_s = sel[order]
    ev_s, v_s = event_idx[idx_s], v[idx_s]
    new_run = np.ones(len(v_s), dtype=bool)
    new_run[1:] = (ev_s[1:] != ev_s[:-1]) | (v_s[1:] != v_s[:-1])
    run_start = np.flatnonzero(new_run)
    run_len = np.diff(np.append(run_start, len(v_s)))
    run_ev = ev_s[run_start]
    o = np.lexsort((run_len, run_ev))
    is_last = np.ones(len(o), dtype=bool)
    is_last[:-1] = run_ev[o][1:] != run_ev[o][:-1]
    rep[run_ev[o][is_last]] = idx_s[run_start[o][is_last]]
    n_mode[run_ev[o][is_last]] = run_len[o][is_last]
    return rep, n_mode


def estimate_event_pv(event_idx, pdgId, vx, vy, vz, n_events):
    """Per-event primary vertex = the vertex shared by the largest group of
    the event's HADRONS (falls back to all particles if none); same
    estimator and same reasoning as build_pu_sampling_recipe.py and
    characterize_gensim_particles.py, duplicated here to keep this
    validation script standalone."""
    is_hadron = np.abs(pdgId) > 100
    rep, _ = _mode_representative(event_idx, vz, n_events, is_hadron)
    rep_all, _ = _mode_representative(event_idx, vz, n_events, np.ones(len(vz), dtype=bool))
    no_hadron = rep < 0
    rep[no_hadron] = rep_all[no_hadron]
    ok = rep >= 0
    out = []
    for v in (vx, vy, vz):
        a = np.full(n_events, np.nan)
        a[ok] = v[rep[ok]]
        out.append(a)
    return out[0], out[1], out[2]


def load_npz(npz_path, displaced_threshold):
    """Ground truth, computed directly from extract_gensim_particles.py's
    output with no recipe involved. Returns per-particle arrays plus the
    per-event PV subtraction needed to measure displacement. 'phi' is
    included (when present in the npz) so consumers that need real
    angular position -- e.g. sample_local_pu_around_probe.py's radial-
    profile closure test -- can compute an actual ΔR against some probe
    direction; nothing in THIS script's own plots uses it."""
    d = np.load(npz_path)
    event_idx = d['event_idx']
    pdgId = d['pdgId']
    pt = d['pt'].astype(np.float64)
    eta = d['eta'].astype(np.float64)
    n_events = int(d['n_events'])
    p = pt * np.cosh(eta)

    pv_x, pv_y, pv_z = estimate_event_pv(event_idx, pdgId, d['vx'].astype(np.float64),
                                          d['vy'].astype(np.float64), d['vz'].astype(np.float64),
                                          n_events)
    dx = d['vx'].astype(np.float64) - pv_x[event_idx]
    dy = d['vy'].astype(np.float64) - pv_y[event_idx]
    dz = d['vz'].astype(np.float64) - pv_z[event_idx]
    d3d = np.sqrt(dx**2 + dy**2 + dz**2)

    out = dict(pdgId=pdgId, pt=pt, eta=eta, p=p, d3d=d3d,
               displaced=d3d > displaced_threshold, n_events=n_events)
    if 'phi' in d.files:
        out['phi'] = d['phi'].astype(np.float64)
    return out


# ---------------------------------------------------------------------------
# Species groupings -- used both when loading per-species pt histograms
# (build_pu_sampling_recipe.py's --pt_by_species) and for the per-species
# control plots below. Hadron uses the same |pdgId| > 100 convention
# already used elsewhere in this file (_mode_representative's PV
# estimator); anything matching none of these (mostly neutrinos) is
# "other" -- left out of the per-species PLOTS but still counted in the
# aggregate ones and, for sampling, given its own "other" pt histogram
# when the recipe has per-species pt at all.
# ---------------------------------------------------------------------------
SPECIES_GROUPS = [
    ('photon',   lambda pid: pid == 22),
    ('electron', lambda pid: np.abs(pid) == 11),
    ('muon',     lambda pid: np.abs(pid) == 13),
    ('hadron',   lambda pid: np.abs(pid) > 100),
]


def _group_for_pid(pid):
    """Which SPECIES_GROUPS bucket a single pdgId falls in, 'other' if none
    -- mirrors build_pu_sampling_recipe.py's _group_for_pid exactly, since
    the two must agree on which pt histogram name corresponds to which
    particle."""
    for name, sel in SPECIES_GROUPS:
        if sel(pid):
            return name
    return 'other'


# ---------------------------------------------------------------------------
# Recipe loading and sampling
# ---------------------------------------------------------------------------

def draw_from_hist(edges, cumulative, rng):
    """GetRandom()-equivalent: pick a bin by content via the precomputed
    cumulative distribution, then a uniform position inside it. Works
    unchanged for variable-width bins (e.g. the pt spectrum's log-spaced
    tail, or the log10(cm) displacement histograms)."""
    u = rng.random()
    idx = min(int(np.searchsorted(cumulative, u)), len(edges) - 2)
    lo, hi = edges[idx], edges[idx + 1]
    return rng.uniform(lo, hi)


def load_recipe(path):
    f = uproot.open(path)
    keys = set(k.split(';')[0] for k in f.keys())
    n_events_npz = int(f['meta']['n_events_npz'].array(library='np')[0])
    density = f['density'].arrays(library='np')
    comp = f['composition'].arrays(library='np')

    n_bins = len(density['eta_lo'])
    bins = []
    any_pt_by_group = False
    for b in range(n_bins):
        sel = comp['eta_bin_idx'] == b
        pdgids = comp['pdgId'][sel]
        fractions = comp['fraction'][sel]
        cumulative = np.cumsum(fractions)
        cumulative = cumulative / cumulative[-1] if len(cumulative) else cumulative

        counts, edges = f[f'pt_spectrum_eta{b}'].to_numpy()
        pt_cumulative = np.cumsum(counts)
        pt_cumulative = pt_cumulative / pt_cumulative[-1] if pt_cumulative[-1] > 0 else None

        # Per-species pt histograms (build_pu_sampling_recipe.py's
        # --pt_by_species): present only if that flag was used when the
        # recipe was built. When present, both the sampler (draw_pt) and
        # aggregate_recipe_pt_for_group use them; when absent, everything
        # falls back to the aggregated-over-species pt_cumulative above,
        # exactly the old behaviour.
        pt_by_group = {}
        for gname in [g for g, _ in SPECIES_GROUPS] + ['other']:
            hname = f'pt_spectrum_eta{b}_{gname}'
            if hname in keys:
                g_counts, g_edges = f[hname].to_numpy()
                g_cum = np.cumsum(g_counts)
                pt_by_group[gname] = {
                    'edges': g_edges,
                    'counts': g_counts,
                    'cumulative': g_cum / g_cum[-1] if g_cum[-1] > 0 else None,
                    }
        if pt_by_group:
            any_pt_by_group = True

        bins.append({
            'eta_lo': density['eta_lo'][b],
            'eta_hi': density['eta_hi'][b],
            'mean_multiplicity': density['mean_multiplicity'][b],
            'pdgids': pdgids,
            'fractions': fractions,
            'cumulative_species': cumulative,
            'pt_edges': edges,
            'pt_counts': counts,
            'pt_cumulative': pt_cumulative,
            'pt_by_group': pt_by_group,
            })
    if any_pt_by_group:
        print('load_recipe: found per-species pt histograms (--pt_by_species) -- pt is now '
              'sampled and validated per species group, not just per eta bin.')

    # --- displacement: one row per (pdgId, momentum bin) ---
    disp = f['displacement'].arrays(library='np')
    displacement_by_pdgid = {}
    for pid, p_lo, p_hi, n_part, frac, hidx in zip(
            disp['pdgId'], disp['p_lo'], disp['p_hi'], disp['n_particles'],
            disp['displaced_fraction'], disp['hist_idx']):
        displacement_by_pdgid.setdefault(int(pid), []).append(dict(
            p_lo=float(p_lo), p_hi=float(p_hi), n_particles=int(n_part),
            displaced_fraction=float(frac), hist_idx=int(hidx)))
    for rows in displacement_by_pdgid.values():
        rows.sort(key=lambda r: r['p_lo'])

    disp_hists = {}
    for idx in sorted(set(int(h) for h in disp['hist_idx'] if h >= 0)):
        counts, edges = f[f'displacement_log10cm_{idx}'].to_numpy()
        cumulative = np.cumsum(counts)
        disp_hists[idx] = dict(edges=edges, counts=counts,
                                cumulative=cumulative / cumulative[-1] if cumulative[-1] > 0 else None)
    counts, edges = f['displacement_log10cm_all'].to_numpy()
    cumulative = np.cumsum(counts)
    disp_hists['all'] = dict(edges=edges, counts=counts,
                              cumulative=cumulative / cumulative[-1] if cumulative[-1] > 0 else None)

    return bins, displacement_by_pdgid, disp_hists, n_events_npz


def filter_bins_to_eta_range(bins, eta_min, eta_max):
    """Keep only recipe eta bins lying ENTIRELY within eta_min <= |eta| <=
    eta_max, on either the positive or negative side -- what --hgcal_eta_min
    /--hgcal_eta_max uses to restrict sampling (and, via the matching npz
    mask in make_control_plots, the npz-truth comparison too) to just the
    HGCAL-instrumented region. This naturally drops BOTH the central/barrel
    bin (|eta| < eta_min, not HGCAL) and any very-forward open-ended outer
    bins (|eta| > eta_max, typically HF/CASTOR, also not HGCAL) -- an
    open-ended bin can never satisfy "entirely within" a finite range, so
    it's excluded automatically, no special-casing needed. displacement_by_
    pdgid/disp_hists need no equivalent filtering: the displacement model
    is keyed by (pdgId, momentum), not eta, so it's already region-
    agnostic -- restricting `bins` alone is enough to restrict which
    particles ever get sampled (and hence ever reach a displacement draw)
    to the kept region."""
    out = []
    for b in bins:
        lo, hi = b['eta_lo'], b['eta_hi']
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue  # open-ended bin: can't be "entirely within" a finite range
        pos_side = lo >= eta_min and hi <= eta_max
        neg_side = hi <= -eta_min and lo >= -eta_max
        if pos_side or neg_side:
            out.append(b)
    return out


def draw_species(bin_info, rng):
    if len(bin_info['pdgids']) == 0:
        return None
    u = rng.random()
    idx = np.searchsorted(bin_info['cumulative_species'], u)
    idx = min(idx, len(bin_info['pdgids']) - 1)
    return int(bin_info['pdgids'][idx])


def draw_pt(bin_info, pid, rng):
    """Draw a pt for a particle of species `pid` in this eta bin. If the
    recipe has per-species pt histograms for this bin (--pt_by_species,
    see load_recipe), draws from `pid`'s own group's histogram; otherwise
    (or if that group's histogram in this bin is empty) falls back to the
    bin's aggregated-over-species histogram, exactly the old behaviour."""
    group = bin_info['pt_by_group'].get(_group_for_pid(pid)) if bin_info['pt_by_group'] else None
    if group is not None and group['cumulative'] is not None:
        return draw_from_hist(group['edges'], group['cumulative'], rng)
    if bin_info['pt_cumulative'] is None:
        return None
    return draw_from_hist(bin_info['pt_edges'], bin_info['pt_cumulative'], rng)


def _representative_eta(eta_lo, eta_hi, rng):
    """A concrete eta for one sampled particle, used ONLY to get a momentum
    p = pt*cosh(eta) for the displacement lookup -- the density/pt parts of
    the recipe never needed more than the bin index, and this is not meant
    as a finer density model than the recipe's own eta bins. Any FINITE bin
    (including the central one straddling eta=0) draws uniformly within it;
    only the two open-ended outer bins, which have no finite width to draw
    within, fall back to their one finite edge.

    FIXED BUG: the central bin used to always return exactly 0. instead of
    drawing uniformly like every other finite bin. Since p = pt*cosh(eta)
    and cosh grows with |eta| (e.g. cosh(1.5) = 2.35 vs cosh(0) = 1), that
    forced every particle landing in the central bin -- usually the
    highest-population bin -- to get its displacement drawn using a
    systematically UNDERESTIMATED momentum, which then picks the wrong
    (too-low-p) row of the recipe's (pdgId, momentum) displacement table:
    wrong displaced_fraction AND wrong histogram shape for that particle.
    pt/eta/composition never call this function (they only need the bin
    index), so they were unaffected -- this is why only the per-species
    displacement shapes looked wrong while everything else validated fine.
    The bias pushes particles toward lower-p (typically smaller-
    displacement) rows, so it shows up as a SAMPLED DEFICIT at large
    log10(displacement) specifically -- exactly what a real recipe file
    with a populous central bin would produce."""
    if not np.isfinite(eta_lo) and not np.isfinite(eta_hi):
        return 0.
    if not np.isfinite(eta_lo):
        return eta_hi
    if not np.isfinite(eta_hi):
        return eta_lo
    return rng.uniform(eta_lo, eta_hi)


def draw_displacement(pid, p, displacement_by_pdgid, disp_hists, rng):
    """3D displacement distance [cm] for one particle (0. if not displaced),
    following exactly the gun-side recipe from build_pu_sampling_recipe.py's
    docstring: look up the (species, momentum) row, roll the displaced-
    fraction Bernoulli draw, and if displaced draw pow(10, GetRandom()) on
    that row's histogram (or the all-species fallback if hist_idx==-1)."""
    rows = displacement_by_pdgid.get(int(pid))
    if not rows:
        return 0.
    row = rows[-1]
    for r in rows:
        if r['p_lo'] <= p < r['p_hi']:
            row = r
            break
    if rng.random() >= row['displaced_fraction']:
        return 0.
    h = disp_hists[row['hist_idx']] if row['hist_idx'] >= 0 else disp_hists['all']
    if h['cumulative'] is None:
        return 0.
    return 10. ** draw_from_hist(h['edges'], h['cumulative'], rng)


def sample_one_event(bins, displacement_by_pdgid, disp_hists, n_pu, fluctuate, rng):
    """Returns (n_pu_used, particles), particles a list of
    (pdgId, eta, phi, pt, d3d, bin_index) tuples for one full as-if-N_pu
    event. eta/phi are only a representative direction for the displacement
    lookup (see _representative_eta); d3d is the drawn 3D displacement
    distance in cm (0. if the particle was drawn as prompt)."""
    n_pu_this_event = rng.poisson(n_pu) if fluctuate else n_pu

    particles = []
    for b, bin_info in enumerate(bins):
        mean_this_bin = n_pu_this_event * bin_info['mean_multiplicity']
        n_particles_this_bin = rng.poisson(mean_this_bin)
        for _ in range(n_particles_this_bin):
            pid = draw_species(bin_info, rng)
            pt = draw_pt(bin_info, pid, rng) if pid is not None else None
            if pid is None or pt is None:
                continue
            eta = _representative_eta(bin_info['eta_lo'], bin_info['eta_hi'], rng)
            phi = rng.uniform(-np.pi, np.pi)
            p = pt * np.cosh(eta)
            d3d = draw_displacement(pid, p, displacement_by_pdgid, disp_hists, rng)
            particles.append((pid, eta, phi, pt, d3d, b))
    return n_pu_this_event, particles


# ---------------------------------------------------------------------------
# Recipe-only (no sampling) aggregate predictions
# ---------------------------------------------------------------------------

def aggregate_recipe_composition(bins, n_pu):
    """Recipe's own expected species composition, aggregated across all
    bins and weighted by each bin's (N_pu-scaled) mean multiplicity."""
    totals, grand_total = {}, 0.
    for bin_info in bins:
        w = n_pu * bin_info['mean_multiplicity']
        grand_total += w
        for pid, frac in zip(bin_info['pdgids'], bin_info['fractions']):
            totals[int(pid)] = totals.get(int(pid), 0.) + w * frac
    if grand_total > 0:
        totals = {k: v / grand_total for k, v in totals.items()}
    return totals


def aggregate_recipe_pt(bins, n_pu):
    """Recipe's own expected pt distribution (counts PER EVENT, i.e. one
    N_pu-scaled event), aggregated across bins AND species, on the first
    bin's pt edges (all bins share one --n_pt_bins/--pt_max/--n_pt_tail_bins
    binning by construction). Always available -- build_pu_sampling_recipe.py
    always writes the aggregated-over-species pt_spectrum_eta{b}, regardless
    of --pt_by_species. See aggregate_recipe_pt_for_group for the
    per-species-group version."""
    edges = bins[0]['pt_edges']
    total_counts = np.zeros(len(edges) - 1)
    for bin_info in bins:
        w = n_pu * bin_info['mean_multiplicity']
        n_in_bin = bin_info['pt_counts'].sum()
        if n_in_bin > 0:
            total_counts += w * (bin_info['pt_counts'] / n_in_bin)
    return total_counts, edges


def aggregate_recipe_pt_for_group(bins, n_pu, group_name):
    """Per-species-group analogue of aggregate_recipe_pt, ONLY available
    when the recipe was built with --pt_by_species (see load_recipe):
    returns (None, None) if bins[0] has no 'pt_by_group' data at all, so
    callers can fall back to the old "sampled vs npz truth only" plot.

    Weighting mirrors aggregate_recipe_pt exactly: each bin contributes
    w = n_pu * bin_info['mean_multiplicity'] (the bin's own N_pu-scaled
    expected particle count, ALL species), and this group's own histogram
    counts are turned into that group's SHARE of the bin's spectrum by
    dividing by n_in_bin -- the bin's TOTAL count (not the group's own
    count) -- exactly like aggregate_recipe_composition turns per-species
    counts into fractions of the bin's total multiplicity. This keeps the
    normalization consistent with the aggregate curve: summing this
    function's result over every group in SPECIES_GROUPS plus 'other'
    reproduces aggregate_recipe_pt's own curve, up to pt_by_group's
    (eta_bin, species) binning being a further split of the same
    pt_spectrum_eta{b} data aggregate_recipe_pt reads."""
    if not bins or not bins[0].get('pt_by_group'):
        return None, None
    edges = bins[0]['pt_edges']
    total_counts = np.zeros(len(edges) - 1)
    for bin_info in bins:
        g = bin_info['pt_by_group'].get(group_name)
        if g is None:
            continue
        w = n_pu * bin_info['mean_multiplicity']
        n_in_bin = bin_info['pt_counts'].sum()  # whole-bin (all species) denominator, by design
        if n_in_bin > 0:
            total_counts += w * (g['counts'] / n_in_bin)
    return total_counts, edges


def aggregate_recipe_displacement(displacement_by_pdgid, disp_hists, n_pu, n_events_npz,
                                   pdgid_filter=None):
    """Recipe's own PURELY ANALYTIC prediction (no sampling at all) for the
    aggregate log10(3D displacement) distribution of displaced particles in
    one N_pu-scaled event: each (species, momentum-bin) row's histogram
    SHAPE is weighted by n_pu * (row.n_particles / n_events_npz) *
    row.displaced_fraction -- the expected number of displaced particles
    that row contributes to ONE interaction, scaled up by n_pu -- and
    summed; valid because every displacement histogram in the file shares
    the same log10(cm) bin edges by construction. row.n_particles is a RAW
    count across the whole npz file the recipe was built from, hence the
    /n_events_npz (read from the recipe's own "meta" tree) to turn it back
    into a per-interaction rate -- forgetting this factor overshoots by
    exactly n_events_npz, easy to miss since nothing else in this ratio
    depends on it.

    pdgid_filter: optional callable(pdgId) -> bool. When given, only rows
    for pdgIds passing the filter are summed -- this is what makes a
    genuine per-species-group ANALYTIC recipe curve possible (unlike pt,
    see aggregate_recipe_pt's docstring): the displacement tree already
    has one row per (pdgId, momentum bin), so restricting the sum by pdgId
    needs no extra information the recipe file doesn't already store."""
    edges = disp_hists['all']['edges']
    total = np.zeros(len(edges) - 1)
    for pid, rows in displacement_by_pdgid.items():
        if pdgid_filter is not None and not pdgid_filter(pid):
            continue
        for row in rows:
            h = disp_hists[row['hist_idx']] if row['hist_idx'] >= 0 else disp_hists['all']
            if h['counts'].sum() == 0:
                continue
            w = n_pu * (row['n_particles'] / n_events_npz) * row['displaced_fraction']
            total += w * (h['counts'] / h['counts'].sum())
    return total, edges


def aggregate_recipe_prompt_weight(displacement_by_pdgid, n_pu, n_events_npz, pdgid_filter=None):
    """Recipe's own analytic expectation for the number of PROMPT (non-
    displaced) particles in one N_pu-scaled event -- the natural companion
    to aggregate_recipe_displacement's displaced weight, using the exact
    same per-row rate (n_pu * row.n_particles/n_events_npz) but multiplied
    by (1 - row.displaced_fraction) instead of row.displaced_fraction.
    There is no shape to distribute this over: a prompt particle sits at
    literally d3d=0 by construction (see build_pu_sampling_recipe.py's
    gun-side recipe), so this returns a single number, not a histogram --
    callers that want to show it alongside a displacement shape plot
    should put it in one dedicated 'prompt' bin rather than trying to
    spread it across the log10(d) axis. Same whole-recipe-acceptance-
    average caveat as aggregate_recipe_displacement applies (row.n_particles
    is not cone- or eta-window-restricted)."""
    total = 0.
    for pid, rows in displacement_by_pdgid.items():
        if pdgid_filter is not None and not pdgid_filter(pid):
            continue
        for row in rows:
            total += n_pu * (row['n_particles'] / n_events_npz) * (1. - row['displaced_fraction'])
    return total


def _prepend_prompt_bin(edges, sampled_counts, recipe_counts, npz_counts,
                         n_prompt_sampled, n_prompt_recipe, n_prompt_npz):
    """Bolt one extra 'prompt' bin (same width as the existing, uniform
    `edges`) onto the low-d3d end of a displacement shape comparison, so
    the plot shows the full displaced-vs-prompt picture instead of only
    the displaced tail's own internal shape -- see aggregate_recipe_
    prompt_weight's docstring for why a prompt particle (d3d=0 exactly)
    has no shape of its own to histogram. Each n_prompt_* should already
    be on the SAME counting convention as that curve's own counts/
    recipe_counts/npz_counts (raw sampled counts, the recipe's own
    n_pu-scaled analytic rate, raw or weighted npz counts respectively) --
    this function only concatenates, it does no rescaling. recipe_counts/
    npz_counts/n_prompt_recipe/n_prompt_npz may be None together (mirrors
    _plot_shape_comparison's own None handling) to omit that curve.
    Returns (full_edges, full_sampled_counts, full_recipe_counts,
    full_npz_counts)."""
    bin_width = edges[1] - edges[0]
    full_edges = np.concatenate(([edges[0] - bin_width], edges))
    full_sampled = np.concatenate(([n_prompt_sampled], sampled_counts))
    full_recipe = (np.concatenate(([n_prompt_recipe], recipe_counts))
                    if recipe_counts is not None else None)
    full_npz = (np.concatenate(([n_prompt_npz], npz_counts))
                if npz_counts is not None else None)
    return full_edges, full_sampled, full_recipe, full_npz


# ---------------------------------------------------------------------------
# Console-only diagnostic: recipe's momentum binning vs a finer npz truth
# ---------------------------------------------------------------------------

def print_species_momentum_table(npz, sampled_pid, sampled_p, sampled_disp,
                                  displacement_by_pdgid, n_species, n_fine_bins):
    """For the n_species most abundant species: displaced fraction in
    n_fine_bins quantile-spaced momentum bins (finer than the recipe's own,
    which used as many as --min_displaced_stat allowed), measured directly
    from npz, next to the recipe's own coarser step function evaluated at
    the same bin centers, and this script's sampled fraction in the same
    fine bins. This is where the recipe's momentum binning choice is most
    directly tested; kept as a printout rather than a figure, see module
    docstring."""
    ids, counts = np.unique(npz['pdgId'], return_counts=True)
    ids = ids[np.argsort(-counts)][:n_species]

    print(f'\nMomentum dependence of the displaced fraction, npz truth (fine bins) vs '
          f'recipe (its own coarser step) vs sampled (same fine bins), '
          f'top {n_species} species:')
    for pid in ids:
        sel = npz['pdgId'] == pid
        p = npz['p'][sel]
        edges = np.quantile(p, np.linspace(0., 1., n_fine_bins + 1))
        edges[0], edges[-1] = 0., np.inf
        edges = np.unique(edges)
        rows = displacement_by_pdgid.get(int(pid), [])

        print(f'  pdgId {int(pid)}:')
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = sel & (npz['p'] >= lo) & (npz['p'] < hi)
            truth_frac = npz['displaced'][in_bin].mean() if in_bin.any() else float('nan')

            mid = .5 * (lo + hi) if np.isfinite(hi) else lo + 1.
            recipe_frac = float('nan')
            for r in rows:
                if r['p_lo'] <= mid < r['p_hi']:
                    recipe_frac = r['displaced_fraction']
                    break

            samp_sel = (sampled_pid == pid) & (sampled_p >= lo) & (sampled_p < hi)
            sampled_frac = sampled_disp[samp_sel].mean() if samp_sel.any() else float('nan')

            print(f'    p in [{lo:>9.2f}, {hi:>9.2f}):  npz={100*truth_frac:6.2f}%  '
                  f'recipe={100*recipe_frac:6.2f}%  sampled={100*sampled_frac:6.2f}%  '
                  f'(n_npz={int(in_bin.sum())}, n_sampled={int(samp_sel.sum())})')


# ---------------------------------------------------------------------------
# Control plots
# ---------------------------------------------------------------------------

def _shape_density(counts, widths):
    """Unit-area shape density: counts/(total_counts * bin_width), so that
    sum(shape * widths) == 1 for any binning -- this is what makes shapes
    on DIFFERENT bin widths (e.g. pt's linear-then-log-tail bins) directly
    comparable as curves, and is the only thing these control plots check
    any more (see module docstring: absolute rate moved to
    total_multiplicity.png). NaN (not 0) when there's no data at all, so a
    genuinely-empty curve doesn't silently plot as a flat zero line."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.full_like(counts, np.nan)
    return counts / (total * widths)


def _plot_shape_comparison(edges, sampled_counts, recipe_counts, npz_counts, n_events,
                            cms_label, outfile, xlabel, ylabel, title_extra='',
                            xscale='linear', yscale='log', ratio_ylim=(0.5, 1.5),
                            vline_x=None, vline_label=None):
    """Unit-area SHAPE comparison of up to three curves (sampled always,
    recipe and npz truth optional) that all share ONE set of bin `edges` --
    every caller below passes counts already histogrammed onto the same
    edges for all three, so this function itself does no rebinning; it only
    normalizes each to a shape density (_shape_density) and plots them with
    a ratio-of-shapes panel. recipe_counts=None omits the recipe curve/
    ratio (the per-species pt-spectrum plots, see aggregate_recipe_pt's
    docstring); npz_counts=None omits the npz-truth curve/ratio (no --npz
    given). vline_x/vline_label: optional dotted vertical marker (e.g. to
    flag a 'prompt' bin bolted onto an otherwise-continuous axis, see
    sample_local_pu_around_probe.py's displacement_log10d_cone.png) -- a
    plain visual cue, does not affect any of the plotted values.

    The ratio panel always compares "sampled" against whichever OTHER
    curve is the better ground truth: npz truth when given (npz/recipe
    both independent of sampled, so both get a ratio line), otherwise
    recipe (the only other curve left). FIXED BUG: this used to nest BOTH
    ratio lines (recipe/npz AND sampled/npz) inside "if npz_counts is not
    None", so with no --npz the ratio panel was silently left completely
    EMPTY -- no sampled/recipe ratio was ever drawn even though recipe_
    shape was available and sampled_shape is always available. The main
    panel's sampled curve (dashed blue) was still drawn in that case, just
    with no ratio to accompany it, and it can visually disappear under an
    exactly-overlapping solid recipe curve -- easy to misread as "the PU
    gun curve isn't there at all" when really only the ratio was missing."""
    widths = np.diff(edges)
    centers = .5 * (edges[:-1] + edges[1:])
    sampled_shape = _shape_density(sampled_counts, widths)
    recipe_shape = _shape_density(recipe_counts, widths) if recipe_counts is not None else None

    fig, ax, rax = _ratio_fig()
    if npz_counts is not None:
        npz_shape = _shape_density(npz_counts, widths)
        ax.step(centers, npz_shape, where='mid', color='black', linewidth=1.5, label='npz truth')
        if recipe_shape is not None:
            rax.step(centers, _safe_ratio(recipe_shape, npz_shape), where='mid',
                     color='#bd1f01', linewidth=1.6, label='recipe / npz')
        rax.step(centers, _safe_ratio(sampled_shape, npz_shape), where='mid',
                 color='#3f90da', linewidth=1.6, label='sampled / npz')
    elif recipe_shape is not None:
        # No npz truth to compare against -- fall back to the only other
        # curve available, so the ratio panel isn't left empty (see FIXED
        # BUG note above).
        rax.step(centers, _safe_ratio(sampled_shape, recipe_shape), where='mid',
                 color='#3f90da', linewidth=1.6, label='sampled / recipe')
    if recipe_shape is not None:
        ax.step(centers, recipe_shape, where='mid', color='#bd1f01', linewidth=2,
                label='recipe (own bins)')
    ax.step(centers, sampled_shape, where='mid', color='#3f90da', linewidth=2, linestyle='--',
            label=f'sampled ({n_events} events)')
    if xscale == 'log':
        ax.set_xscale('log')
    if yscale == 'log':
        ax.set_yscale('log')
    ax.set_ylabel(ylabel); ax.legend(fontsize=12)
    if vline_x is not None:
        ax.axvline(vline_x, color='gray', linewidth=1, linestyle=':')
        rax.axvline(vline_x, color='gray', linewidth=1, linestyle=':')
        if vline_label:
            ax.annotate(vline_label, xy=(vline_x, ax.get_ylim()[1]), xytext=(4, -4),
                        textcoords='offset points', fontsize=9, color='gray', va='top')
    if title_extra:
        # A species tag (e.g. "(hadron)") as a plain corner annotation
        # rather than appended to the y-axis label: two adjacent stacked
        # two-line y-labels (this one + the ratio panel's own) otherwise
        # visually run into each other.
        ax.text(0.02, 0.97, title_extra.strip(), transform=ax.transAxes, ha='left', va='top',
                fontsize=14, fontweight='bold')
    rax.set_xlabel(xlabel); rax.set_ylim(*ratio_ylim)
    if npz_counts is not None or recipe_shape is not None:
        rax.legend(fontsize=10)
    rax.set_ylabel('ratio of\nshapes')
    _add_cms_label(ax, cms_label)
    _save(fig, outfile)


def make_control_plots(bins, displacement_by_pdgid, disp_hists, n_events_npz, n_pu, fluctuate,
                        n_events, outdir, rng, cms_label, npz_path, displaced_threshold,
                        n_console_species, n_console_p_bins, eta_min=None, eta_max=None):
    os.makedirs(outdir, exist_ok=True)

    all_totals, all_pdgids, all_pts, all_ps, all_d3d, all_bin_idxs = [], [], [], [], [], []
    for _ in range(n_events):
        _, particles = sample_one_event(bins, displacement_by_pdgid, disp_hists, n_pu, fluctuate, rng)
        all_totals.append(len(particles))
        for pid, eta, phi, pt, d3d, b in particles:
            all_pdgids.append(pid); all_pts.append(pt)
            all_ps.append(pt * np.cosh(eta)); all_d3d.append(d3d); all_bin_idxs.append(b)

    all_totals = np.array(all_totals)
    all_pdgids = np.array(all_pdgids)
    all_pts = np.array(all_pts)
    all_ps = np.array(all_ps)
    all_d3d = np.array(all_d3d)
    all_bin_idxs = np.array(all_bin_idxs)
    all_displaced = all_d3d > 0.

    expected_total = n_pu * sum(b['mean_multiplicity'] for b in bins)
    print(f'\nSampled {n_events} events, {len(all_pdgids)} particles total '
          f'({100. * all_displaced.mean():.2f}% displaced).')
    print(f'Total multiplicity: observed mean={all_totals.mean():.1f}, '
          f'expected={expected_total:.1f}, observed std={all_totals.std():.1f} '
          f'(Poisson prediction sqrt(expected)={np.sqrt(expected_total):.1f})')

    npz = load_npz(npz_path, displaced_threshold) if npz_path else None
    if npz is not None and (eta_min is not None or eta_max is not None):
        # Keep the npz-truth comparison apples-to-apples with `bins` having
        # already been restricted by --hgcal_eta_min/--hgcal_eta_max in
        # main(): mask npz's own particles to the same |eta| window (on
        # either side) before any plot or the console table uses them.
        lo = eta_min if eta_min is not None else 0.
        hi = eta_max if eta_max is not None else np.inf
        keep = (np.abs(npz['eta']) >= lo) & (np.abs(npz['eta']) <= hi)
        n_before = len(npz['pdgId'])
        for key in ('pdgId', 'pt', 'eta', 'p', 'd3d', 'displaced'):
            npz[key] = npz[key][keep]
        print(f'--hgcal_eta_min/--hgcal_eta_max: kept {keep.sum()}/{n_before} npz particles '
              f'({lo} <= |eta| <= {hi}) for the npz-truth comparison.')

    # --- Plot 1: total multiplicity (no npz-truth line, see docstring) ---
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.hist(all_totals, bins=30, color='#3f90da', edgecolor='none')
    ax.axvline(expected_total, color='#bd1f01', linestyle='--', linewidth=2,
               label=f'recipe expectation = {expected_total:.0f}')
    ax.set_xlabel('Total sampled particles per event')
    ax.set_ylabel('Events')
    ax.legend(fontsize=13)
    _add_cms_label(ax, cms_label)
    _save(fig, os.path.join(outdir, 'total_multiplicity.png'))

    # --- Plot 2: eta SHAPE, on the recipe's own (finite) eta bins -- the
    # only binning available for "sampled" anyway, since a particle's
    # representative eta (_representative_eta) is not a finer density
    # model, just a within-bin draw for the displacement momentum lookup.
    # Built from the UNION of each finite bin's own (eta_lo, eta_hi)
    # breakpoints rather than assuming the kept bins are contiguous: with
    # no --hgcal_eta_min/--hgcal_eta_max filtering the finite bins ARE
    # contiguous (central bin sits between the two forward ones) so this
    # reduces to the same edges as before, but once that flag drops the
    # central bin, the two forward bins are separated by a gap -- naively
    # concatenating eta_lo[:1] with eta_hi would then silently misassign
    # the second (forward+) bin's width and center to the wrong range.
    # Any gap between kept bins becomes its own zero-count "bin" here
    # instead, which is harmless (a zero-count bin doesn't change
    # _shape_density's normalization) and keeps every real bin's own
    # width/center correct.
    finite_idx = [i for i, b in enumerate(bins)
                  if np.isfinite(b['eta_lo']) and np.isfinite(b['eta_hi'])]
    breakpoints = sorted(set(v for i in finite_idx for v in (bins[i]['eta_lo'], bins[i]['eta_hi'])))
    eta_edges = np.array(breakpoints)
    sampled_eta_counts = np.zeros(len(eta_edges) - 1)
    recipe_eta_counts = np.zeros(len(eta_edges) - 1)
    for i in finite_idx:
        slot = breakpoints.index(bins[i]['eta_lo'])
        sampled_eta_counts[slot] = np.sum(all_bin_idxs == i)
        recipe_eta_counts[slot] = n_pu * bins[i]['mean_multiplicity']
    npz_eta_counts = np.histogram(npz['eta'], bins=eta_edges)[0] if npz is not None else None
    _plot_shape_comparison(eta_edges, sampled_eta_counts, recipe_eta_counts, npz_eta_counts,
                            n_events, cms_label, os.path.join(outdir, 'density_dNdeta.png'),
                            xlabel='eta', ylabel='dN/deta shape [a.u.]')

    # --- Plot 3: species composition ---
    recipe_comp = aggregate_recipe_composition(bins, n_pu)
    unique_ids, counts = np.unique(all_pdgids, return_counts=True)
    order = np.argsort(-counts)
    unique_ids, counts = unique_ids[order], counts[order]
    sampled_frac = counts / len(all_pdgids)
    recipe_frac = np.array([recipe_comp.get(int(pid), 0.) for pid in unique_ids])

    fig, ax, rax = _ratio_fig()
    x = np.arange(len(unique_ids))
    width = 0.27
    ax.bar(x - width, 100 * sampled_frac, width, color='#3f90da', label='sampled')
    ax.bar(x, 100 * recipe_frac, width, color='#bd1f01', label='recipe')
    if npz is not None:
        npz_frac = np.array([np.mean(npz['pdgId'] == pid) for pid in unique_ids])
        ax.bar(x + width, 100 * npz_frac, width, color='black', label='npz truth')
        rax.scatter(x - width / 2, _safe_ratio(sampled_frac, npz_frac), color='#3f90da', zorder=3)
        rax.scatter(x + width / 2, _safe_ratio(recipe_frac, npz_frac), color='#bd1f01', zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([str(int(pid)) for pid in unique_ids])
    ax.set_ylabel('Fraction of particles [%]')
    ax.legend(fontsize=12)
    rax.set_xticks(x); rax.set_xticklabels([str(int(pid)) for pid in unique_ids], rotation=90)
    rax.set_xlabel('pdgId'); rax.set_ylim(0.5, 1.5)
    _add_cms_label(ax, cms_label)
    _save(fig, os.path.join(outdir, 'composition.png'))

    # --- Plot 4: pt SHAPE, on the recipe's own bins (linear, then a
    # log-spaced tail) -- the same shared edges npz truth and sampled are
    # both histogrammed onto below, so all three curves are directly
    # comparable bin for bin. ---
    recipe_pt_counts, pt_edges = aggregate_recipe_pt(bins, n_pu)
    sampled_pt_counts = np.histogram(all_pts, bins=pt_edges)[0]
    npz_pt_counts = np.histogram(npz['pt'], bins=pt_edges)[0] if npz is not None else None
    _plot_shape_comparison(pt_edges, sampled_pt_counts, recipe_pt_counts, npz_pt_counts,
                            n_events, cms_label, os.path.join(outdir, 'pt_spectrum.png'),
                            xlabel='pt [GeV]', ylabel='pt shape [a.u.]', xscale='log')

    # --- Plot 5: aggregate displacement-magnitude SHAPE, on disp_edges
    # (disp_hists['all']['edges'] -- shared by every displacement histogram
    # in the recipe file by construction, so nothing to choose here). ---
    recipe_disp_counts, disp_edges = aggregate_recipe_displacement(
        displacement_by_pdgid, disp_hists, n_pu, n_events_npz)
    sampled_log_d = np.log10(all_d3d[all_displaced]) if all_displaced.any() else np.array([])
    sampled_disp_counts = np.histogram(sampled_log_d, bins=disp_edges)[0]
    if npz is not None:
        npz_log_d = np.log10(npz['d3d'][npz['displaced']]) if npz['displaced'].any() else np.array([])
        npz_disp_counts = np.histogram(npz_log_d, bins=disp_edges)[0]
    else:
        npz_disp_counts = None

    # Prepend a PROMPT (d3d=0) bin, same convention as the cone gun's
    # displacement_log10d_cone.png -- see aggregate_recipe_prompt_weight
    # and _prepend_prompt_bin's docstrings. npz's prompt count here is a
    # plain raw count (no chord/cone weighting -- this script has no cone,
    # unlike sample_local_pu_around_probe.py), matching npz_disp_counts's
    # own (also unweighted) convention above.
    n_prompt_sampled = int((~all_displaced).sum())
    n_prompt_recipe = aggregate_recipe_prompt_weight(displacement_by_pdgid, n_pu, n_events_npz)
    n_prompt_npz = int((~npz['displaced']).sum()) if npz is not None else None
    full_disp_edges, full_sampled_disp_counts, full_recipe_disp_counts, full_npz_disp_counts = \
        _prepend_prompt_bin(disp_edges, sampled_disp_counts, recipe_disp_counts, npz_disp_counts,
                             n_prompt_sampled, n_prompt_recipe, n_prompt_npz)
    n_total_sampled = n_prompt_sampled + int(sampled_disp_counts.sum())
    n_total_recipe = n_prompt_recipe + recipe_disp_counts.sum()
    print(f'\nPrompt fraction -- sampled: {n_prompt_sampled}/{n_total_sampled} = '
          f'{100. * n_prompt_sampled / n_total_sampled:.2f}%; recipe: '
          f'{100. * n_prompt_recipe / n_total_recipe:.2f}%'
          + (f'; npz truth: {100. * n_prompt_npz / (n_prompt_npz + npz_disp_counts.sum()):.2f}%'
             if npz is not None else ''))

    _plot_shape_comparison(full_disp_edges, full_sampled_disp_counts, full_recipe_disp_counts,
                            full_npz_disp_counts, n_events, cms_label,
                            os.path.join(outdir, 'displacement_log10d.png'),
                            xlabel='log10(3D displacement / cm)  (leftmost bin: prompt, d3d=0)',
                            ylabel='Displacement shape [a.u.]',
                            vline_x=disp_edges[0], vline_label='prompt | displaced')

    # --- Plots 6-13: same two quantities (pt spectrum, displacement), each
    # split into the four SPECIES_GROUPS (photon/electron/muon/hadron), so
    # e.g. a spectral difference between photons and hadrons that washes
    # out in the aggregate curve above is visible directly. Every group
    # reuses the SAME pt_edges/disp_edges as the aggregate plots above (not
    # a per-group binning), so shapes are comparable group to group too.
    # See aggregate_recipe_pt's docstring for why the pt-spectrum group
    # plots have no "recipe (own bins)" curve -- the recipe file itself
    # never stored pt split by species, only aggregated per eta bin. ---
    for group_name, group_sel in SPECIES_GROUPS:
        sampled_mask = group_sel(all_pdgids)
        n_sampled_this_group = int(sampled_mask.sum())
        title_extra = f'({group_name})'

        # -- pt spectrum for this species group: sampled vs npz, PLUS a
        # genuine analytic recipe curve (aggregate_recipe_pt_for_group) when
        # the recipe was built with --pt_by_species -- None/None otherwise,
        # in which case _plot_shape_comparison falls back to the old
        # sampled-vs-npz-only plot automatically. Same shared pt_edges as
        # the aggregate plot either way. --
        group_pt_sampled = all_pts[sampled_mask]
        if npz is not None:
            group_pt_npz = npz['pt'][group_sel(npz['pdgId'])]
        else:
            group_pt_npz = None
        group_recipe_pt_counts, _ = aggregate_recipe_pt_for_group(bins, n_pu, group_name)
        if n_sampled_this_group > 0 or (group_pt_npz is not None and len(group_pt_npz) > 0):
            group_sampled_pt_counts = np.histogram(group_pt_sampled, bins=pt_edges)[0]
            group_npz_pt_counts = np.histogram(group_pt_npz, bins=pt_edges)[0] \
                if group_pt_npz is not None else None
            _plot_shape_comparison(pt_edges, group_sampled_pt_counts, group_recipe_pt_counts,
                                    group_npz_pt_counts, n_events, cms_label,
                                    os.path.join(outdir, f'pt_spectrum_{group_name}.png'),
                                    xlabel='pt [GeV]', ylabel='pt shape [a.u.]',
                                    xscale='log', title_extra=title_extra)
        else:
            print(f'\n(no {group_name} particles sampled or in npz: skipping pt_spectrum_{group_name}.png)')

        # -- displacement for this species group: sampled + a genuine
        # per-species-group analytic recipe curve (see aggregate_recipe_
        # displacement's pdgid_filter docstring) + npz truth, same shared
        # disp_edges as the aggregate plot --
        group_recipe_disp_counts, _ = aggregate_recipe_displacement(
            displacement_by_pdgid, disp_hists, n_pu, n_events_npz, pdgid_filter=group_sel)
        group_disp_mask = sampled_mask & all_displaced
        group_sampled_log_d = np.log10(all_d3d[group_disp_mask]) if group_disp_mask.any() else np.array([])
        group_sampled_disp_counts = np.histogram(group_sampled_log_d, bins=disp_edges)[0]
        if npz is not None:
            group_npz_disp_mask = group_sel(npz['pdgId']) & npz['displaced']
            group_npz_log_d = np.log10(npz['d3d'][group_npz_disp_mask]) if group_npz_disp_mask.any() \
                else np.array([])
            group_npz_disp_counts = np.histogram(group_npz_log_d, bins=disp_edges)[0]
        else:
            group_npz_disp_counts = None

        # Same prepended PROMPT bin as the aggregate displacement_log10d.png
        # above, restricted to this species group throughout.
        group_n_prompt_sampled = int((sampled_mask & ~all_displaced).sum())
        group_n_prompt_recipe = aggregate_recipe_prompt_weight(
            displacement_by_pdgid, n_pu, n_events_npz, pdgid_filter=group_sel)
        group_n_prompt_npz = (int((group_sel(npz['pdgId']) & ~npz['displaced']).sum())
                                if npz is not None else None)
        group_full_edges, group_full_sampled, group_full_recipe, group_full_npz = _prepend_prompt_bin(
            disp_edges, group_sampled_disp_counts, group_recipe_disp_counts, group_npz_disp_counts,
            group_n_prompt_sampled, group_n_prompt_recipe, group_n_prompt_npz)
        _plot_shape_comparison(group_full_edges, group_full_sampled, group_full_recipe,
                                group_full_npz, n_events, cms_label,
                                os.path.join(outdir, f'displacement_log10d_{group_name}.png'),
                                xlabel='log10(3D displacement / cm)  (leftmost bin: prompt, d3d=0)',
                                ylabel='Displacement shape [a.u.]', title_extra=title_extra,
                                vline_x=disp_edges[0], vline_label='prompt | displaced')

    if npz is not None:
        print_species_momentum_table(npz, all_pdgids, all_ps, all_displaced,
                                      displacement_by_pdgid, n_console_species, n_console_p_bins)
    else:
        print('\n(no --npz given: skipping the npz-truth curves/ratios and the momentum-'
              'dependence console table)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recipe', required=True, help="Output of build_pu_sampling_recipe.py.")
    parser.add_argument('--npz', type=str, default=None,
                         help="Output of extract_gensim_particles.py -- if given, control plots "
                              "add a direct-from-data 'npz truth' curve and ratio panel; if "
                              "omitted, only the recipe-vs-sampled comparison is shown, as "
                              "before this option existed.")
    parser.add_argument('--displaced_threshold', type=float, default=1e-2,
                         help="3D distance from the event PV [cm] above which an npz particle "
                              "counts as displaced, for the npz-truth curves. MUST match the "
                              "value passed to build_pu_sampling_recipe.py when the --recipe "
                              "file was built, or the comparison is apples-to-oranges. Default "
                              "matches that script's own default.")
    parser.add_argument('--n_pu', type=int, default=200, help="Target number of PU interactions.")
    parser.add_argument('--fluctuate', action='store_true', default=False,
                         help="Draw the actual N_pu for each event from Poisson(--n_pu) instead "
                              "of using a fixed count every event -- see module docstring.")
    parser.add_argument('--n_events', type=int, default=3,
                         help="Number of events to sample. Only the first 3 get their particle "
                              "details printed regardless of this value; if --outdir is given, "
                              "ALL sampled events feed the control plots -- set this higher "
                              "(e.g. 20+) when you actually want the plots to be meaningful.")
    parser.add_argument('--outdir', type=str, default=None,
                         help="If given, sample --n_events events and produce control plots -- "
                              "see module docstring for the six plots produced.")
    parser.add_argument('--n_console_species', type=int, default=6,
                         help="Number of most-abundant species to print the momentum-dependence "
                              "displaced-fraction table for (only when --npz and --outdir are "
                              "both given).")
    parser.add_argument('--n_console_p_bins', type=int, default=5,
                         help="Number of quantile-spaced momentum bins per species in that table.")
    parser.add_argument('--hgcal_eta_min', type=float, default=None,
                         help="If given (together with --hgcal_eta_max), restrict EVERYTHING -- "
                              "sampling itself, all control plots, the npz-truth comparison, and "
                              "the console table -- to recipe eta bins lying entirely within "
                              "hgcal_eta_min <= |eta| <= hgcal_eta_max. Drops both the central/"
                              "barrel bin (|eta| < hgcal_eta_min) and any very-forward open-ended "
                              "outer bins (|eta| > hgcal_eta_max, typically HF/CASTOR) -- neither "
                              "is HGCAL, so this sidesteps their cruder eta modeling entirely "
                              "rather than trying to fix it (see _representative_eta's docstring). "
                              "Typical HGCAL coverage is roughly 1.5 to 3.0; check your recipe's "
                              "own bin edges (printed below) to set these to match exactly.")
    parser.add_argument('--hgcal_eta_max', type=float, default=None, help="See --hgcal_eta_min.")
    parser.add_argument('--cms_label', default='Simulation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    bins, displacement_by_pdgid, disp_hists, n_events_npz = load_recipe(args.recipe)

    if args.hgcal_eta_min is not None or args.hgcal_eta_max is not None:
        eta_min = args.hgcal_eta_min if args.hgcal_eta_min is not None else 0.
        eta_max = args.hgcal_eta_max if args.hgcal_eta_max is not None else np.inf
        n_before = len(bins)
        mult_before = sum(b['mean_multiplicity'] for b in bins)
        bins = filter_bins_to_eta_range(bins, eta_min, eta_max)
        mult_after = sum(b['mean_multiplicity'] for b in bins)
        print(f'--hgcal_eta_min/--hgcal_eta_max: kept {len(bins)}/{n_before} recipe eta bins '
              f'({eta_min} <= |eta| <= {eta_max}), {mult_after:.4f}/{mult_before:.4f} '
              f'mean_multiplicity per interaction retained. Kept bins:')
        for b in bins:
            print(f'    [{b["eta_lo"]:.3f}, {b["eta_hi"]:.3f}]  mean_multiplicity={b["mean_multiplicity"]:.5f}')
        if not bins:
            raise SystemExit('No recipe eta bins survived --hgcal_eta_min/--hgcal_eta_max -- check '
                              'these against the bin edges your --recipe file actually uses.')

    rng = np.random.default_rng(args.seed)

    n_preview = min(3, args.n_events)
    for i in range(n_preview):
        n_pu_used, particles = sample_one_event(bins, displacement_by_pdgid, disp_hists,
                                                 args.n_pu, args.fluctuate, rng)
        n_disp = sum(1 for p in particles if p[4] > 0)
        print(f'Event {i}: N_pu={n_pu_used}, {len(particles)} particles sampled '
              f'({n_disp} displaced)')
        for pid, eta, phi, pt, d3d, b in particles[:5]:
            disp_str = f'd3d={d3d:.4f} cm' if d3d > 0 else 'prompt'
            print(f'    pdgId={pid:>6}  pt={pt:.3f} GeV  eta~{eta:+.2f}  phi={phi:+.2f}  '
                  f'{disp_str}  eta bin={b} [{bins[b]["eta_lo"]:.2f}, {bins[b]["eta_hi"]:.2f}]')
        if len(particles) > 5:
            print(f'    ... ({len(particles) - 5} more)')

    if args.outdir:
        # Re-seed so the preview events above don't consume the RNG stream
        # the control-plot statistics are drawn from.
        rng = np.random.default_rng(args.seed)
        make_control_plots(bins, displacement_by_pdgid, disp_hists, n_events_npz, args.n_pu,
                            args.fluctuate, args.n_events, args.outdir, rng, args.cms_label,
                            args.npz, args.displaced_threshold, args.n_console_species,
                            args.n_console_p_bins, eta_min=args.hgcal_eta_min,
                            eta_max=args.hgcal_eta_max)


if __name__ == '__main__':
    main()
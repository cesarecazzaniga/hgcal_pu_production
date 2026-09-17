#!/usr/bin/env python3
"""
build_pu_sampling_recipe.py

Reads the .npz produced by extract_gensim_particles.py and builds a ROOT
file the eventual local-PU gun can fetch at runtime to sample realistic
particle multiplicity, species, pt, and DISPLACED PRODUCTION VERTICES --
eta-binned for the kinematics, per the discussion that a single global (non-eta-
binned) density/spectrum isn't safely assumable here.

WHY A ROOT FILE, NOT JSON/TEXT: the eventual consumer is C++ CMSSW code.
A ROOT file with plain TTrees + TH1Ds needs no custom parser on the C++
side at all -- just TFile::Open(), TTree::GetEntry() in a loop, and
TH1::GetRandom() for the draws, all standard ROOT idioms.

CONTENTS -- KINEMATICS (unchanged):
  - TTree "density": one row per eta bin (eta_lo, eta_hi, mean_multiplicity
    -- mean number of particles per event in that bin). The gun should
    draw a Poisson-distributed count around this mean per bin, matching
    how real PU multiplicity itself fluctuates event to event (see e.g.
    arXiv:1801.09721's own description: "the actual number of interactions
    ... fluctuates, following a Poisson distribution").
  - TTree "composition": one row per (eta_bin_idx, pdgId, fraction).
    UNLIKE characterize_gensim_particles.py's own composition plot (which
    truncates to top-N + "other" purely for plot readability), this
    keeps EVERY species that appears, un-truncated -- a sampling recipe
    needs the real distribution, not a display-simplified one. The gun
    builds a per-bin cumulative distribution from these rows to draw a
    species.
  - Named histograms "pt_spectrum_eta{i}", one per eta bin (aggregated
    across all species within that bin, NOT further split by species --
    see module docstring in the conversation this was built from for why:
    splitting both ways would fragment current statistics too far to be
    reliable. Straightforward to extend once more statistics exist.)
    TH1::GetRandom() draws directly from these. Binning: linear up to
    --pt_max (default: 99th percentile), then log-spaced tail bins up to
    the highest pt in the sample -- variable-width bins, which GetRandom()
    handles correctly (it picks a bin by CONTENT, then a uniform position
    inside it).

CONTENTS -- DISPLACED PRODUCTION VERTICES.
DIVISION OF LABOUR: the PRIMARY-VERTEX smearing (beamspot position/width,
incl. time) is NOT part of this recipe -- it is left to CMSSW's standard
VtxSmeared step, which shifts every vertex of the gun's HepMC event by one
common, randomly drawn offset. This recipe only describes where a particle
is produced RELATIVE to its interaction point, a quantity that is invariant
under that common shift. So the gun works in the un-smeared frame: prompt
particles at (0,0,0), displaced ones at their displacement vector, each in
its own HepMC::GenVertex; VtxSmeared then moves all of them together.
(To measure the displacement in the input sample, which IS smeared, this
script still estimates each event's PV internally -- vertex shared by the largest group of the event's hadrons
of the event's particle vertices -- and subtracts it. The PV spread is only
printed, as a cross-check against the VtxSmeared scenario you will run the
gun with; it is not written to the file.)

  - TTree "displacement": one row per (pdgId, momentum bin) --
        pdgId, p_lo, p_hi [GeV], n_particles, displaced_fraction, hist_idx
    displaced_fraction = fraction of particles of that species and
    momentum range produced further than --displaced_threshold (3D) from
    their PV. Conditioned on SPECIES because the displaced fraction varies
    by orders of magnitude between species, and on MOMENTUM p = pt*cosh(eta)
    because both the chance of coming from a displaced decay and the flight
    distance of the parent (~ beta*gamma*ctau) grow with it; sampling the
    displacement independently of momentum would get exactly the energetic
    particles wrong. Momentum rather than eta is the conditioning variable
    since it is what drives the boost; the eta dependence enters through it
    (p = pt*cosh(eta), with pt and eta drawn from the eta-binned part).
    The momentum binning is ADAPTIVE, per species: quantiles of the
    displaced particles' momenta, with as many bins (up to --n_p_bins) as
    --min_displaced_stat allows, so every histogram has comparable
    statistics. The first bin starts at 0, the last ends at +inf, so every
    momentum matches exactly one row. Species with fewer displaced
    particles than --min_displaced_stat get a single row (p in [0, inf))
    with their own measured displaced_fraction but hist_idx = -1, meaning:
    use the all-species fallback histogram.
  - TH1Ds "displacement_log10cm_{hist_idx}" and "displacement_log10cm_all":
    distribution of log10(3D displacement / cm). Stored in log10 because
    the distances span orders of magnitude; TH1::GetRandom() is uniform
    within a bin, which is a sensible interpolation in log space and a poor
    one in linear space. The gun must do pow(10, GetRandom()).
  - TH1D "displacement_cos_angle": cos of the angle between the
    displacement vector and the particle's own momentum, displaced
    particles only. For decay products of boosted parents this peaks
    sharply at +1, which justifies the simple direction model below; the
    script prints how well that holds for this sample.

  GUN-SIDE RECIPE for one particle (after species, eta, phi, pt are drawn):
        p   = pt * cosh(eta);
        row = the "displacement" row with this pdgId and p_lo <= p < p_hi;
        pos = (0, 0, 0);  ct = 0;                  // un-smeared frame
        if (rng.Uniform() < row.displaced_fraction) {
            h   = row.hist_idx >= 0 ? displacement_log10cm_{hist_idx}
                                    : displacement_log10cm_all;
            d   = pow(10, h->GetRandom());         // cm
            pos = d * p_hat;                       // p_hat = unit momentum
            ct  = d;                               // parent flew ~ at c
        }
        // HepMC in CMSSW is in MM: multiply pos and ct by 10.
        vtx = new HepMC::GenVertex(HepMC::FourVector(10*pos, 10*ct));
    All prompt particles can share one GenVertex at the origin; each
    displaced particle needs its own. Displacing ALONG the momentum keeps
    the particle pointing back at the interaction point. If
    displacement_cos_angle turns out broad, draw cos(angle) from it and a
    uniform azimuth around p_hat instead.

  THINGS TO KEEP IN MIND on the CMSSW side:
    * VtxSmeared must stay in the sequence (that is the whole point) and
      needs nothing special: BaseEvtVtxGenerator shifts ALL vertices of the
      HepMC event by the same 4-vector, so displacements survive untouched.
    * Consequence of one common shift: every particle of a gun event,
      probe included, comes from the SAME interaction point in z and t. In
      real pileup the interactions are spread over sigma_z ~ 4-5 cm and
      sigma_t ~ 180 ps. For the local energy density in HGCAL this hardly
      matters (a few cm in z moves the impact point by ~1 cm, and PU is
      smooth on that scale), but the gun's PU is perfectly in time with the
      probe -- relevant if timing is used downstream.
    * genParticles only contains decays done by the GENERATOR (ctau <
      10 mm with standard CMS Pythia settings). K0S, Lambda, charged pi/K
      are stable here and get shot by the gun as such -- GEANT then decays
      them in the gun's own SIM step, so their displaced products come for
      free and must NOT be added to this recipe (that would double count).
    * CLOSURE TEST: run extract_gensim_particles.py +
      characterize_gensim_particles.py on the gun's own GEN-SIM output and
      compare vertex_displacement_*.png and the printed displaced fractions
      with those of the minbias sample.

Eta bins default to a modest number (--n_eta_bins, default 6) spanning
+/-(HGCAL_ETA_MIN, HGCAL_ETA_MAX) plus everything outside that range
grouped into two outer bins -- deliberately coarse given current
statistics; narrow this once you have more events (more statistics
directly buys you finer eta binning here, nothing else needs to change).

Usage:
    python build_pu_sampling_recipe.py \\
        --npz gensim_particles.npz \\
        --outfile pu_sampling_recipe.root \\
        --n_eta_bins 6
"""

import argparse

import numpy as np
import uproot


HGCAL_ETA_MIN = 1.5
HGCAL_ETA_MAX = 3.0


def _mode_representative(event_idx, v, n_events, subset):
    """
    For each event, index (into the full arrays) of one particle belonging to
    the largest group of particles of `subset` sharing a bit-identical `v`,
    and the size of that group. -1 / 0 for events without any subset particle.
    """
    rep = np.full(n_events, -1, dtype=np.int64)
    n_mode = np.zeros(n_events, dtype=np.int64)
    sel = np.flatnonzero(subset)
    if len(sel) == 0:
        return rep, n_mode
    order = np.lexsort((v[sel], event_idx[sel]))
    idx_s = sel[order]
    ev_s, v_s = event_idx[idx_s], v[idx_s]
    # run-length encode identical (event, value) pairs ...
    new_run = np.ones(len(v_s), dtype=bool)
    new_run[1:] = (ev_s[1:] != ev_s[:-1]) | (v_s[1:] != v_s[:-1])
    run_start = np.flatnonzero(new_run)
    run_len = np.diff(np.append(run_start, len(v_s)))
    run_ev = ev_s[run_start]
    # ... and keep the longest run of each event
    o = np.lexsort((run_len, run_ev))
    is_last = np.ones(len(o), dtype=bool)
    is_last[:-1] = run_ev[o][1:] != run_ev[o][:-1]
    rep[run_ev[o][is_last]] = idx_s[run_start[o][is_last]]
    n_mode[run_ev[o][is_last]] = run_len[o][is_last]
    return rep, n_mode


def estimate_event_pv(event_idx, pdgId, vx, vy, vz, n_events):
    """
    Per-event primary vertex = the production vertex shared by the largest
    group of HADRONS in the event. Returns (pv_x, pv_y, pv_z, n_mode); NaN / 0
    for events with no particles. n_mode = size of that group (1 = no two
    particles agree, PV of that event not trustworthy).

    Why this works: all prompt particles of an event share a BIT-IDENTICAL
    vertex (same generator point, same VtxSmeared shift), while displaced
    particles sit elsewhere -- so the most frequent exact value is the PV even
    where prompt particles are a minority, which a median does not survive
    (low-multiplicity forward events are dominated by photons from pi0s that
    flew um to 100s of um).
    Why hadrons only: decay products ALSO share a vertex with their siblings
    -- the two photons of a pi0 above all. In a diffractive event with two
    prompt hadrons and one TeV pi0, "2 particles at the PV" ties with "2
    photons at the pi0 decay point". Photons and leptons are overwhelmingly
    decay products, hadrons overwhelmingly prompt, so restricting the vote to
    hadrons removes the tie. Events without any hadron fall back to all
    particles. Identity is decided on vz (the coordinate with by far the
    largest spread); x and y are read off the same group.
    """
    is_hadron = np.abs(pdgId) > 100
    rep, n_mode = _mode_representative(event_idx, vz, n_events, is_hadron)
    rep_all, n_mode_all = _mode_representative(event_idx, vz, n_events, np.ones(len(vz), dtype=bool))
    no_hadron = rep < 0
    rep[no_hadron], n_mode[no_hadron] = rep_all[no_hadron], n_mode_all[no_hadron]

    ok = rep >= 0
    pv = []
    for v in (vx, vy, vz):
        out = np.full(n_events, np.nan)
        out[ok] = v[rep[ok]]
        pv.append(out)
    return pv[0], pv[1], pv[2], n_mode


def build_vertex_recipe(d, n_events, args):
    """
    Returns (hists, displacement_tree): dict name -> (counts, edges), and the
    dict of arrays for the "displacement" TTree. PV smearing itself is left to
    CMSSW's VtxSmeared -- the PV is only estimated here to subtract it.
    """
    event_idx = d['event_idx']
    pdgId = d['pdgId']
    eta = d['eta'].astype(np.float64)
    phi = d['phi'].astype(np.float64)
    mom = d['pt'].astype(np.float64) * np.cosh(eta)
    vx = d['vx'].astype(np.float64)
    vy = d['vy'].astype(np.float64)
    vz = d['vz'].astype(np.float64)

    hists = {}

    # --- per-event PV: internal only, to measure displacements relative to it ---
    pv = {}
    pv['x'], pv['y'], pv['z'], n_mode = estimate_event_pv(event_idx, pdgId, vx, vy, vz, n_events)
    ok = np.isfinite(pv['z'])
    n_shaky = int((n_mode[ok] < 2).sum())
    if n_shaky > 0:
        print(f'\nNOTE: in {n_shaky} of {ok.sum()} events no two particles share a vertex, so the PV '
              f'of those events is a guess (irrelevant if this is a tiny fraction).')
    print(f'\nPrimary vertex of the INPUT sample ({ok.sum()} events with >=1 particle) -- not '
          f'written to the recipe, PV smearing is left to VtxSmeared. For comparison with the '
          f'scenario you run the gun with:')
    for c in 'xyz':
        vals = pv[c][ok]
        print(f'  pv_{c}: mean={vals.mean():+.5f} cm  std={vals.std():.5f} cm')

    # --- displacement from the event PV ---
    dx = vx - pv['x'][event_idx]
    dy = vy - pv['y'][event_idx]
    dz = vz - pv['z'][event_idx]
    d3d = np.sqrt(dx**2 + dy**2 + dz**2)
    displaced = d3d > args.displaced_threshold
    n_displaced = int(displaced.sum())
    print(f'\nDisplacement: {n_displaced} of {len(d3d)} particles '
          f'({100. * displaced.mean():.2f}%) further than {args.displaced_threshold:g} cm '
          f'from their event PV.')

    log_lo = np.log10(args.displaced_threshold)
    log_hi = np.log10(d3d[displaced].max()) + 1e-6 if n_displaced > 0 else log_lo + 1.
    log_edges = np.linspace(log_lo, log_hi, args.n_disp_bins + 1)

    def log_hist(mask):
        counts, _ = np.histogram(np.log10(d3d[mask]), bins=log_edges)
        return counts.astype(np.float64), log_edges

    hists['displacement_log10cm_all'] = log_hist(displaced)

    # Per-(species, momentum bin) rows. Species sorted by abundance, for a readable printout.
    ids, n_per_species = np.unique(pdgId, return_counts=True)
    ids = ids[np.argsort(-n_per_species)]

    cols = {k: [] for k in ('pdgId', 'p_lo', 'p_hi', 'n_particles', 'displaced_fraction', 'hist_idx')}
    next_idx = 0
    print(f'  {"pdgId":>8} {"p range [GeV]":>20} {"n":>10} {"displaced":>10} {"median d":>11}  histogram')
    for pid in ids:
        is_species = pdgId == pid
        species_displaced = is_species & displaced
        n_disp = int(species_displaced.sum())

        # Adaptive momentum binning: quantiles of the DISPLACED particles' momenta, as many
        # bins as the statistics allow, so that each histogram gets >= min_displaced_stat.
        own_hist = n_disp >= args.min_displaced_stat
        p_edges = np.array([0., np.inf])
        if own_hist:
            n_p = max(1, min(args.n_p_bins, n_disp // args.min_displaced_stat))
            inner = np.quantile(mom[species_displaced], np.linspace(0., 1., n_p + 1))[1:-1]
            p_edges = np.concatenate(([0.], np.unique(inner[inner > 0]), [np.inf]))

        for p_lo, p_hi in zip(p_edges[:-1], p_edges[1:]):
            in_bin = is_species & (mom >= p_lo) & (mom < p_hi)
            in_bin_displaced = in_bin & displaced
            n_all, n_d = int(in_bin.sum()), int(in_bin_displaced.sum())

            if own_hist:
                hist_idx = next_idx
                next_idx += 1
                hist_name = f'displacement_log10cm_{hist_idx}'
                hists[hist_name] = log_hist(in_bin_displaced)
            else:
                hist_idx = -1
                hist_name = 'displacement_log10cm_all (fallback)' if n_d > 0 else '-'

            cols['pdgId'].append(int(pid))
            cols['p_lo'].append(float(p_lo))
            cols['p_hi'].append(float(p_hi))
            cols['n_particles'].append(n_all)
            cols['displaced_fraction'].append(n_d / n_all if n_all > 0 else 0.)
            cols['hist_idx'].append(hist_idx)

            med = f'{np.median(d3d[in_bin_displaced]):.4f} cm' if n_d > 0 else '-'
            p_range = f'[{p_lo:.2f}, {p_hi:.2f})'
            print(f'  {int(pid):>8} {p_range:>20} {n_all:>10} '
                  f'{100. * cols["displaced_fraction"][-1]:>9.2f}% {med:>11}  {hist_name}')

    print('  (if "displaced" and "median d" rise with p within a species, the momentum '
          'conditioning is doing real work)')

    # Angle between displacement and momentum direction: validates (or not) the
    # "displace along the momentum" gun model.
    if n_displaced > 0:
        e, p = eta[displaced], phi[displaced]
        p_hat = np.stack((np.cos(p) / np.cosh(e), np.sin(p) / np.cosh(e), np.tanh(e)))
        disp = np.stack((dx[displaced], dy[displaced], dz[displaced]))
        cos_angle = np.clip((p_hat * disp).sum(axis=0) / d3d[displaced], -1., 1.)
        print(f'\n  cos(angle between displacement and momentum): median={np.median(cos_angle):.4f}, '
              f'{100. * (cos_angle > 0.9).mean():.1f}% above 0.9, '
              f'{100. * (cos_angle > 0.99).mean():.1f}% above 0.99')
        print('  -> if these are high, displacing along the momentum direction in the gun is a '
              'good model; if not, sample displacement_cos_angle too (see module docstring).')
    else:
        cos_angle = np.array([])
    counts, edges = np.histogram(cos_angle, bins=args.n_disp_bins, range=(-1., 1.))
    hists['displacement_cos_angle'] = (counts.astype(np.float64), edges)

    tree = {
        'pdgId': np.array(cols['pdgId'], dtype=np.int32),
        'p_lo': np.array(cols['p_lo'], dtype=np.float64),
        'p_hi': np.array(cols['p_hi'], dtype=np.float64),
        'n_particles': np.array(cols['n_particles'], dtype=np.int64),
        'displaced_fraction': np.array(cols['displaced_fraction'], dtype=np.float64),
        'hist_idx': np.array(cols['hist_idx'], dtype=np.int32),
        }
    return hists, tree


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--npz', required=True, help="Output of extract_gensim_particles.py.")
    parser.add_argument('--outfile', required=True, help="Output ROOT file for the gun to fetch.")
    parser.add_argument('--n_eta_bins', type=int, default=6,
                         help="Number of eta bins spanning the HGCAL EE acceptance "
                              "(both endcaps). Deliberately coarse by default given current "
                              "statistics -- see module docstring.")
    parser.add_argument('--n_pt_bins', type=int, default=40,
                         help="Number of bins for each per-eta-bin pt spectrum histogram.")
    parser.add_argument('--pt_max', type=float, default=None,
                         help="Upper edge of the LINEARLY binned part of the pt spectrum "
                              "histograms (GeV). Default: 99th percentile of the data, to avoid "
                              "a few outliers stretching every bin's resolution.")
    parser.add_argument('--n_pt_tail_bins', type=int, default=15,
                         help="Number of additional LOG-spaced bins from --pt_max up to the "
                              "highest pt in the data, so the high-pt tail (few particles, but a "
                              "large share of the energy) is not silently dropped. "
                              "TH1::GetRandom() handles variable-width bins correctly. 0 = old "
                              "behaviour: everything above --pt_max is discarded.")
    parser.add_argument('--n_p_bins', type=int, default=3,
                         help="Maximum number of momentum bins per species for the displacement "
                              "sampling. The actual number is adaptive per species: as many as "
                              "--min_displaced_stat allows, up to this.")
    parser.add_argument('--n_disp_bins', type=int, default=50,
                         help="Number of bins for the log10(displacement) and cos(angle) "
                              "histograms.")
    parser.add_argument('--displaced_threshold', type=float, default=1e-2,
                         help="3D distance from the event PV [cm] above which a particle counts "
                              "as displaced; also the lower edge of the displacement histograms. "
                              "Below it the gun places the particle exactly at the PV. Default "
                              "1e-2 cm = 100 um: irrelevant for a calorimeter ~3 m away, and "
                              "it keeps the ~um flight of boosted pi0s (which would otherwise "
                              "flag most energetic photons and Dalitz e+- as 'displaced') out "
                              "of the recipe. NB characterize_gensim_particles.py defaults to "
                              "1e-4 for its plots; pass the same value to both to compare.")
    parser.add_argument('--min_displaced_stat', type=int, default=300,
                         help="Minimum number of displaced particles per displacement histogram. "
                              "A species with fewer than this in total points to the "
                              "all-species fallback (hist_idx = -1); one with k times this gets "
                              "up to k momentum bins (capped by --n_p_bins).")
    args = parser.parse_args()

    d = np.load(args.npz)
    pdgId = d['pdgId']
    pt = d['pt']
    eta = d['eta']
    n_events = int(d['n_events'])

    # Eta bins: HGCAL_ETA_MIN..HGCAL_ETA_MAX split into n_eta_bins on each
    # side (both endcaps, mirrored), plus two open-ended outer bins
    # covering everything beyond +/-HGCAL_ETA_MAX and one central bin for
    # everything inside +/-HGCAL_ETA_MIN (central region, not directly relevant to HGCAL but
    # kept so every particle in the file lands in SOME bin -- makes the
    # density TTree's total row-sum a useful cross-check against the
    # file's own overall multiplicity).
    hgcal_edges_pos = np.linspace(HGCAL_ETA_MIN, HGCAL_ETA_MAX, args.n_eta_bins + 1)
    bin_edges = np.concatenate((
        [-np.inf], -hgcal_edges_pos[::-1], hgcal_edges_pos, [np.inf]
        ))
    n_bins_total = len(bin_edges) - 1
    print(f'Using {n_bins_total} total eta bins (incl. the central bin and 2 open-ended outer bins): '
          f'{np.round(bin_edges, 2)}')

    bin_idx_per_particle = np.clip(np.digitize(eta, bin_edges) - 1, 0, n_bins_total - 1)

    pt_max = args.pt_max if args.pt_max is not None else float(np.percentile(pt[pt > 0], 99))
    pt_bin_edges = np.linspace(0., pt_max, args.n_pt_bins + 1)
    print(f'pt spectrum histograms: {args.n_pt_bins} linear bins, [0, {pt_max:.2f}] GeV')
    above = pt > pt_max
    if args.n_pt_tail_bins > 0 and above.any():
        tail = np.logspace(np.log10(pt_max), np.log10(float(pt.max()) * 1.0001), args.n_pt_tail_bins + 1)
        pt_bin_edges = np.concatenate((pt_bin_edges, tail[1:]))
        print(f'  + {args.n_pt_tail_bins} log-spaced tail bins up to {pt_bin_edges[-1]:.1f} GeV, holding '
              f'{100. * above.mean():.2f}% of the particles but '
              f'{100. * pt[above].sum() / pt.sum():.1f}% of the summed pt')
    elif above.any():
        print(f'  WARNING: {100. * above.mean():.2f}% of the particles ({100. * pt[above].sum() / pt.sum():.1f}% '
              f'of the summed pt) lie above {pt_max:.2f} GeV and are DROPPED from the spectra.')

    # --- density TTree ---
    density_eta_lo, density_eta_hi, density_mean_mult = [], [], []
    # --- composition TTree ---
    comp_bin_idx, comp_pdgid, comp_fraction = [], [], []
    # --- per-bin pt histograms ---
    pt_hists = {}

    for b in range(n_bins_total):
        sel = bin_idx_per_particle == b
        n_in_bin = int(sel.sum())
        mean_mult = n_in_bin / n_events

        density_eta_lo.append(float(bin_edges[b]))
        density_eta_hi.append(float(bin_edges[b + 1]))
        density_mean_mult.append(mean_mult)

        if n_in_bin > 0:
            ids, counts = np.unique(pdgId[sel], return_counts=True)
            fractions = counts / n_in_bin
            for pid, frac in zip(ids, fractions):
                comp_bin_idx.append(b)
                comp_pdgid.append(int(pid))
                comp_fraction.append(float(frac))

            counts_pt, _ = np.histogram(pt[sel], bins=pt_bin_edges)
        else:
            counts_pt = np.zeros(len(pt_bin_edges) - 1)

        pt_hists[f'pt_spectrum_eta{b}'] = (counts_pt.astype(np.float64), pt_bin_edges)

    print(f'\nDensity summary ({n_bins_total} bins):')
    for b in range(n_bins_total):
        print(f'  eta [{density_eta_lo[b]:>7.2f}, {density_eta_hi[b]:>7.2f}]: '
              f'mean_mult={density_mean_mult[b]:.3f}')

    total_from_bins = sum(density_mean_mult) * n_events
    print(f'\nCross-check: sum of per-bin counts = {int(round(total_from_bins))}, '
          f'total particles in file = {len(pdgId)} (should match exactly)')

    # --- vertex part ---
    needed = ('vx', 'vy', 'vz', 'phi', 'event_idx')
    if all(k in d.files for k in needed):
        vertex_hists, displacement_tree = build_vertex_recipe(d, n_events, args)
    else:
        missing = [k for k in needed if k not in d.files]
        print(f'\nWARNING: {missing} missing from {args.npz} -- writing the recipe WITHOUT any '
              f'displacement information.')
        vertex_hists, displacement_tree = {}, None

    with uproot.recreate(args.outfile) as f:
        f['density'] = {
            'eta_lo': np.array(density_eta_lo),
            'eta_hi': np.array(density_eta_hi),
            'mean_multiplicity': np.array(density_mean_mult),
            }
        f['composition'] = {
            'eta_bin_idx': np.array(comp_bin_idx, dtype=np.int32),
            'pdgId': np.array(comp_pdgid, dtype=np.int32),
            'fraction': np.array(comp_fraction),
            }
        for name, (counts, edges) in pt_hists.items():
            f[name] = (counts, edges)
        if displacement_tree is not None:
            f['displacement'] = displacement_tree
        for name, (counts, edges) in vertex_hists.items():
            f[name] = (counts, edges)

    print(f'\nWrote {args.outfile}')
    print(f'  density: {n_bins_total} rows')
    print(f'  composition: {len(comp_bin_idx)} rows ({len(np.unique(comp_pdgid))} distinct species total)')
    print(f'  {len(pt_hists)} pt spectrum histograms (pt_spectrum_eta0 .. pt_spectrum_eta{n_bins_total-1})')
    if displacement_tree is not None:
        n_own = int((displacement_tree['hist_idx'] >= 0).sum())
        print(f'  displacement: {len(displacement_tree["pdgId"])} rows (species x momentum bin), '
              f'{n_own} with their own displacement_log10cm_{{idx}} histogram, the rest use '
              f'displacement_log10cm_all')
        print(f'  (no PV histograms: primary-vertex smearing is left to CMSSW VtxSmeared)')
        print(f'  displacement_cos_angle histogram')


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
sample_local_pu_around_probe.py

Prototype/reference implementation of a LOCAL pileup gun: given a probe
particle's direction (eta_probe, phi_probe) -- e.g. as produced by
DisplacedParticleGunProducerFlatEta.cc -- samples the extra pileup
particles the recipe (build_pu_sampling_recipe.py's output) predicts
WITHIN A CONE of radius --cone_radius around that direction, instead of
across the whole eta-binned, phi-uniform acceptance the recipe describes.
This is the direct extension of sample_pu_event.py's "as-if-N_pu-
interactions" idea to a spatially-localized gun: same species/pt/
displacement draws per particle (reused unchanged from sample_pu_event.py),
but restricted geometrically to a disk around one probe, which is what a
"shoot the local pileup profile around a probe" HGCAL gun actually needs
(you don't want to simulate an entire full-acceptance PU event to look at
the ~cm-scale energy density around one probe particle).

THE ONE REAL SUBTLETY (why this needs its own script, not just a smaller
--n_pu in sample_pu_event.py): every mean_multiplicity in the recipe's
"density" TTree is a rate integrated over the FULL 2*pi in phi (build_pu_
sampling_recipe.py never records any phi dependence, and sample_pu_event.py
draws phi uniformly in [-pi, pi) for exactly this reason). A disk of
radius R around one probe direction does NOT cover the full phi range at
any eta, so naively reusing mean_multiplicity would drastically
overcount. The fix: treat (eta, phi) as a flat Euclidean plane -- the
standard convention already implicit in the collider ΔR = sqrt(Δeta^2 +
Δphi^2) cone metric -- and treat each recipe eta bin's local particle
density as UNIFORM in that plane (areal density = mean_multiplicity /
(bin_width * 2*pi), particles per unit eta per unit phi, per interaction;
this uniform-within-bin assumption is exactly the same one
_representative_eta already makes in sample_pu_event.py, just applied to
BOTH eta and phi now instead of only eta). The expected number of
particles a bin contributes inside the cone is then areal_density * (area
of that bin's eta-phi strip inside the disk) -- a closed-form circular-
segment integral (_circle_segment_area below), summed over every recipe
eta bin the disk overlaps. Sampling within an overlapping bin is
correspondingly uniform-in-area rejection sampling over (that bin's strip)
INTERSECT (the disk) -- mirroring the DisplacedParticleGunProducerFlatEta.cc
Origin plane's own kUniformArea convention, just done in (eta, phi) instead
of (x, y).

WHAT DOESN'T CHANGE: species (draw_species), pt (draw_pt) and displacement
(draw_displacement) are drawn EXACTLY as in sample_pu_event.py, from
whichever recipe eta bin a particle's sampled position falls in -- the
cone only changes HOW MANY particles come from each bin and WHERE within
the bin they land, not the per-bin species/pt/displacement model itself.
displacement in particular is keyed only by (pdgId, momentum), with no eta
dependence at all in the recipe file (see build_pu_sampling_recipe.py's
docstring) -- so its predicted SHAPE is identical whether you're sampling
the whole recipe acceptance or one small cone within it; only the eta-
binned parts (density -> Poisson counts, and pt if --pt_by_species) are
reweighted by the cone geometry. See aggregate_recipe_pt_in_cone below.

RECOMMENDED RECIPE: build the --recipe file with build_pu_sampling_
recipe.py's --geometric_cut, so it contains ONLY finite HGCAL-region eta
bins (see that script's docstring) -- this cone gun needs FINITE bin
widths to convert mean_multiplicity into an areal density at all (a
central/outer OPEN-ENDED bin has infinite width, so its areal density is
formally zero and its contribution is skipped with a warning if the cone
reaches it; see sample_local_pu_around_probe's docstring). If your probe
sits close to the recipe's own --eta_min/--eta_max edge, part of the cone
may fall outside every bin the recipe describes at all -- this
UNDERESTIMATES the true local density there (the recipe simply has no
information about particles beyond its own cut), and this script prints
the fraction of the cone's area actually covered by the recipe so you can
judge whether this matters for your probe position and cone size.

CENTERING CONVENTION: the cone is centered on the probe's DIRECTION (eta,
phi) as seen from the origin -- i.e. where its trajectory points, NOT its
production point (which may itself be displaced). This matches the
standard "ΔR from a particle" convention (jets, isolation cones, etc.) and
is what "local energy density around a probe in the calorimeter" means
physically: the calorimeter cares about angular proximity to the probe's
line of flight, not proximity to its (possibly displaced, generally
sub-cm) production vertex.

INTEGRATION SKETCH (CMSSW side, not implemented here): the natural home
for this is a new EDProducer that both generates the probe (reusing
DisplacedParticleGunProducerFlatEta's resolveParticle/appendParticleToGenEvent
machinery almost verbatim -- nothing about this needs to change) AND, in
the same produce() call, loops over the recipe's eta bins overlapping the
probe's cone, draws a Poisson count per bin from the areal weight computed
below, and calls appendParticleToGenEvent again for each with a
(pdgId, momentum, vertex) tuple built exactly like sample_one_event's
displacement recipe already documents (position = 0 for prompt, or
d*p_hat with ct=d for displaced, in the SAME un-smeared frame as the probe
-- VtxSmeared then shifts everything, probe and local PU alike, together,
same as sample_pu_event.py's docstring already describes for the
whole-acceptance gun). One produce() call, one shared GenEvent, is
simplest -- no HepMC merging across producers needed.

Usage:
    python sample_local_pu_around_probe.py \\
        --recipe pu_sampling_recipe_hgcal.root \\
        --eta_probe 2.2 --phi_probe 0.5 --cone_radius 0.4 \\
        --n_pu 200 --n_events 50 --outdir local_pu_control_plots/
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sample_pu_event as spe


def _wrap_phi(phi):
    return (phi + np.pi) % (2. * np.pi) - np.pi


def _circle_segment_area(lo, hi, R):
    """Area of the intersection between a disk of radius R centered at the
    origin and the horizontal strip {(x, y): lo <= x <= hi}, in a flat
    plane -- exactly the ΔR = sqrt(Δeta^2 + Δphi^2) convention applied to
    one recipe eta bin's (shifted so the probe sits at x=0) [eta_lo,
    eta_hi] range. lo/hi may be +-inf (open-ended recipe bins): the
    closed-form integral of the circle's width 2*sqrt(R^2-x^2) naturally
    clips to 0 outside [-R, R], no special-casing needed."""
    def F(x):
        x = np.clip(x, -R, R)
        return x * np.sqrt(np.maximum(0., R * R - x * x)) + R * R * np.arcsin(np.clip(x / R, -1., 1.))
    hi_c, lo_c = min(hi, R), max(lo, -R)
    if hi_c <= lo_c:
        return 0.
    return F(hi_c) - F(lo_c)


def cone_bin_weights(bins, n_pu, eta_probe, cone_radius):
    """For each FINITE recipe eta bin, its expected contribution (Poisson
    mean, for one N_pu-scaled event) to the cone: areal_density * area of
    that bin's eta-phi strip inside the disk. Returns (weights, coverage),
    weights = {bin_index: mean_count}; coverage = fraction of the disk's
    OWN area (pi*R^2) actually covered by some recipe bin -- less than 1
    means the recipe has no information for part of the cone (probe too
    close to the recipe's own eta acceptance edge), which UNDERESTIMATES
    the true local density there; see module docstring."""
    weights = {}
    covered_area = 0.
    for b, bin_info in enumerate(bins):
        lo, hi = bin_info['eta_lo'], bin_info['eta_hi']
        if not (np.isfinite(lo) and np.isfinite(hi)):
            area_b = _circle_segment_area(lo - eta_probe, hi - eta_probe, cone_radius)
            if area_b > 0:
                print(f'WARNING: cone overlaps a non-finite recipe eta bin [{lo}, {hi}] -- its local '
                      f'areal density is not defined (infinite bin width); this bin\'s contribution to '
                      f'the cone is SKIPPED, which underestimates the local density there. Use a '
                      f'--geometric_cut recipe and/or a smaller --cone_radius / probe further from the '
                      f'acceptance edge to avoid this.')
            continue
        area_b = _circle_segment_area(lo - eta_probe, hi - eta_probe, cone_radius)
        if area_b <= 0:
            continue
        covered_area += area_b
        rho_b = bin_info['mean_multiplicity'] / ((hi - lo) * 2. * np.pi)
        weights[b] = n_pu * rho_b * area_b
    disk_area = np.pi * cone_radius**2
    coverage = covered_area / disk_area if disk_area > 0 else 0.
    return weights, coverage


def sample_local_pu_around_probe(bins, displacement_by_pdgid, disp_hists, n_pu, fluctuate,
                                  eta_probe, phi_probe, cone_radius, rng, max_attempts=10000):
    """Returns (n_pu_used, particles, bin_weights): particles a list of
    (pdgId, eta, phi, pt, d3d, bin_index) tuples, eta/phi the particle's
    REAL sampled direction (absolute, already offset by the probe's own
    eta_probe/phi_probe) -- unlike sample_pu_event.py's _representative_eta,
    there is no separate "representative" placeholder here: position IS
    the thing being sampled, at real resolution within the cone, not just
    a within-bin draw for the momentum lookup. Species/pt/displacement are
    drawn exactly as in sample_one_event, using whichever recipe bin the
    particle's own eta falls into."""
    n_pu_this_event = rng.poisson(n_pu) if fluctuate else n_pu
    bin_weights, _ = cone_bin_weights(bins, n_pu_this_event, eta_probe, cone_radius)

    particles = []
    for b, mean_n in bin_weights.items():
        bin_info = bins[b]
        lo, hi = bin_info['eta_lo'], bin_info['eta_hi']
        box_eta_lo = max(lo - eta_probe, -cone_radius)
        box_eta_hi = min(hi - eta_probe, cone_radius)
        n_b = rng.poisson(mean_n)
        for _ in range(n_b):
            pid = spe.draw_species(bin_info, rng)
            if pid is None:
                continue
            pt = spe.draw_pt(bin_info, pid, rng)
            if pt is None:
                continue
            # Uniform-in-area rejection sampling within (bin strip) ∩ (disk)
            # -- same convention as DisplacedParticleGunProducerFlatEta.cc's
            # own kUniformArea radial distribution, applied in (eta, phi).
            d_eta, d_phi = 0.5 * (box_eta_lo + box_eta_hi), 0.
            for _attempt in range(max_attempts):
                d_eta = rng.uniform(box_eta_lo, box_eta_hi)
                d_phi = rng.uniform(-cone_radius, cone_radius)
                if d_eta**2 + d_phi**2 <= cone_radius**2:
                    break
            eta = eta_probe + d_eta
            phi = _wrap_phi(phi_probe + d_phi)
            p = pt * np.cosh(eta)
            d3d = spe.draw_displacement(pid, p, displacement_by_pdgid, disp_hists, rng)
            particles.append((pid, eta, phi, pt, d3d, b))
    return n_pu_this_event, particles


# ---------------------------------------------------------------------------
# Cone-aware analytic predictions (pt is eta-bin-dependent, so its recipe
# curve must be reweighted by the cone's OWN per-bin weights, not the bin's
# full mean_multiplicity -- see module docstring. Displacement is keyed
# only by (pdgId, momentum), never by eta, so its shape is IDENTICAL inside
# a cone or across the whole recipe -- aggregate_recipe_displacement is
# reused UNCHANGED, no cone-specific version needed.)
# ---------------------------------------------------------------------------

def aggregate_recipe_pt_in_cone(bins, bin_weights, group_name=None):
    """Cone-weighted analogue of sample_pu_event.py's aggregate_recipe_pt /
    aggregate_recipe_pt_for_group: same per-bin normalization convention
    (each bin's own histogram divided by that bin's OWN total pt_counts,
    to turn it into a per-bin shape), but weighted by bin_weights[b] (the
    cone's own expected count from that bin, from cone_bin_weights) instead
    of n_pu * bin's full mean_multiplicity -- a cone straddling two bins
    with different pt spectra samples them in proportion to how much of
    the CONE falls in each, not their overall rate."""
    edges = bins[0]['pt_edges']
    total_counts = np.zeros(len(edges) - 1)
    for b, w in bin_weights.items():
        bin_info = bins[b]
        if group_name is None:
            counts, n_in_bin = bin_info['pt_counts'], bin_info['pt_counts'].sum()
        else:
            g = bin_info['pt_by_group'].get(group_name)
            if g is None:
                continue
            counts, n_in_bin = g['counts'], bin_info['pt_counts'].sum()
        if n_in_bin > 0:
            total_counts += w * (counts / n_in_bin)
    return total_counts, edges


# ---------------------------------------------------------------------------
# npz-truth references for the cone -- NOT a naive ΔR cut on npz. npz is a
# flat, phi-uniform sample with no probe direction of its own: a literal
# "keep npz particles within cone_radius of (eta_probe, phi_probe)" cut
# would, for a small cone, throw away the vast majority of npz's
# statistics purely by chance (only the ~cone_radius/pi fraction of phi
# space near phi_probe survives), for no gain in accuracy -- the model
# assumes phi-uniformity in the first place, so a cut in phi does not
# select a physically different population, just a smaller sample of the
# same one. Each truth reference below instead uses whichever restriction
# actually matches what the corresponding analytic/sampled curve depends
# on, keeping full npz statistics wherever position doesn't matter:
#   - pt: depends on eta (which recipe bin) but not phi -- weight every
#     npz particle within the disk's eta EXTENT by the disk's own local
#     chord width at that particle's eta (2*sqrt(R^2-(eta-eta_probe)^2)),
#     the exact area-weighting aggregate_recipe_pt_in_cone itself uses, and
#     histogram with those weights. No phi cut at all.
#   - displacement: the recipe's per-row CONDITIONAL model only depends on
#     (pdgId, momentum) not eta directly -- but the AGGREGATE/marginal
#     displacement shape absolutely can still shift with eta, because
#     p = pt*cosh(eta) grows with |eta| at fixed pt (and the pt spectrum
#     itself can vary by eta bin -- see build_pu_sampling_recipe.py's
#     --pt_by_species). An earlier version of this function used the
#     FULL, unrestricted npz sample as "truth" reasoning that displacement
#     was "eta-blind" -- that conflated the conditional model (correct)
#     with the marginal shape (not eta-blind at all): the full npz usually
#     spans a much wider eta range than the cone (or even --recipe's own
#     --geometric_cut region), so its momentum mix, and hence its
#     aggregate displacement shape, is a genuinely DIFFERENT population,
#     not just a noisier estimate of the same one. Fixed to use the same
#     chord-weighted eta-window restriction as pt, for a fair local
#     comparison -- see npz_displacement_truth_in_cone.
#
#     One consequence worth knowing: the recipe's OWN analytic curve
#     (aggregate_recipe_displacement) has no way to be cone-local at all
#     -- its rows are built by pooling every particle across the recipe's
#     WHOLE eta acceptance that lands in a given momentum bin, so it is
#     intrinsically a whole-acceptance AVERAGE, not localized to any one
#     cone. The SAMPLED curve, by contrast, computes each particle's
#     momentum from ITS OWN real (local, near eta_probe) sampled eta
#     before the row lookup -- so once npz truth is properly localized,
#     expect the SAMPLED curve to track it more closely than the RECIPE
#     curve does, especially for a probe sitting off-center in the
#     recipe's own eta acceptance. A visible recipe/npz gap that the
#     sampled/npz ratio does NOT share is this known limitation showing
#     up correctly, not a sampler bug.
#   - radial profile (dN/dR): NOT a literal ΔR cut using npz's real phi --
#     an earlier version of this script did that, but it throws away all
#     but the ~cone_radius/pi fraction of npz statistics that happen to
#     land near phi_probe by chance, and for a modest --npz or small cone
#     that can leave too little to plot at all. Instead, exploit the SAME
#     phi-uniformity assumption the whole recipe already relies on (this
#     introduces nothing not already assumed elsewhere): build a FINE
#     real-eta density directly from ALL npz particles near eta_probe (no
#     phi cut, full statistics), then integrate that density through
#     thin annuli using the exact same circular-segment geometry
#     cone_bin_weights uses for the recipe's own analytic curve -- this is
#     just a higher-RESOLUTION version of that same calculation, using
#     npz's finer real eta information instead of the recipe's coarse eta
#     bins. See npz_radial_profile_in_cone. The literal ΔR cut is still
#     printed as a low-statistics, position-based cross-check (it is the
#     only one of these that actually tests phi-uniformity against real
#     data rather than assuming it), just not plotted as the main curve.
# ---------------------------------------------------------------------------

def _chord_weight(d_eta, cone_radius):
    """Probability that a phi-UNIFORM particle at this eta (i.e. one whose
    real phi we are NOT cutting on -- see module note) happens to fall
    within the disk's phi range at that eta: (chord width at this eta) /
    (full 2*pi phi range) = sqrt(R^2-d_eta^2)/pi. NOT just the chord width
    itself -- every npz particle in the eta window already represents "one
    particle somewhere across the full 2*pi of phi" since phi is
    unrestricted, so weighting by the raw chord width alone would double
    count by a factor of 2*pi (verified: an earlier version of this
    function was off by exactly that factor)."""
    return np.sqrt(np.maximum(0., cone_radius**2 - d_eta**2)) / np.pi


def npz_pt_truth_in_cone(npz, eta_probe, cone_radius, pt_edges):
    """Chord-weighted pt histogram from ALL npz particles within the
    disk's eta extent (see module note above) -- the full-statistics,
    unbiased estimator of what aggregate_recipe_pt_in_cone predicts. Only
    used as a SHAPE (unit-normalized downstream), so the overall
    normalization convention doesn't matter here, but _chord_weight is
    used consistently with the (normalization-sensitive) total-count
    cross-check in main() anyway."""
    d_eta = npz['eta'] - eta_probe
    in_window = np.abs(d_eta) <= cone_radius
    weights = _chord_weight(d_eta[in_window], cone_radius)
    counts, _ = np.histogram(npz['pt'][in_window], bins=pt_edges, weights=weights)
    return counts, int(in_window.sum())


def npz_displacement_truth_in_cone(npz, eta_probe, cone_radius, disp_edges):
    """Chord-weighted displacement-shape truth from npz particles within
    the disk's eta extent -- same weighting as npz_pt_truth_in_cone, for
    the same reason: the aggregate displacement shape depends on eta
    indirectly (through the eta-dependent momentum mix), even though the
    recipe's per-row model does not depend on eta directly. See the
    module note above for why this replaced an earlier full-npz-sample
    version."""
    d_eta = npz['eta'] - eta_probe
    in_window = np.abs(d_eta) <= cone_radius
    weights = _chord_weight(d_eta[in_window], cone_radius)
    displaced_in_window = npz['displaced'][in_window]
    log_d = (np.log10(npz['d3d'][in_window][displaced_in_window])
             if displaced_in_window.any() else np.array([]))
    w_displaced = weights[displaced_in_window]
    counts, _ = np.histogram(log_d, bins=disp_edges, weights=w_displaced)
    return counts, int(in_window.sum()), int(displaced_in_window.sum())


def npz_radial_profile_in_cone(npz, eta_probe, cone_radius, r_edges, n_events_npz, n_pu, n_fine=200):
    """High-statistics npz-truth dN/dR shape: build a FINE per-unit-eta
    density from ALL npz particles within the disk's eta extent (no phi
    cut -- see module note above), then integrate that density through
    each [r_edges[i], r_edges[i+1]] annulus via the same circular-segment
    geometry cone_bin_weights uses, one fine eta cell at a time. This is
    exactly cone_bin_weights's own calculation, just with npz's finer real
    eta histogram standing in for the recipe's coarser per-bin
    mean_multiplicity. Returns counts (expected-count-per-radial-bin,
    already n_pu-scaled so it's on the same footing as the recipe/sampled
    curves -- _shape_density then unit-normalizes it like everything
    else)."""
    d_eta = npz['eta'] - eta_probe
    sel = np.abs(d_eta) <= cone_radius
    fine_edges = np.linspace(-cone_radius, cone_radius, n_fine + 1)
    fine_counts, _ = np.histogram(d_eta[sel], bins=fine_edges)
    fine_width = fine_edges[1] - fine_edges[0]
    f_npz = fine_counts / (n_events_npz * fine_width)  # dN/deta, mean per event, fine-binned

    def expected_count_within(Rp):
        total = 0.
        for lo, hi, dens in zip(fine_edges[:-1], fine_edges[1:], f_npz):
            if dens <= 0:
                continue
            total += dens * _circle_segment_area(lo, hi, Rp)
        return n_pu * total

    cum = np.array([expected_count_within(r) for r in r_edges])
    return np.diff(cum), int(sel.sum())


def npz_radial_profile_literal_cut(npz, eta_probe, phi_probe, cone_radius):
    """The literal ΔR cut using npz's own real (eta, phi) -- a low-
    statistics but position-based (not phi-uniformity-assuming) cross-
    check, printed to the console rather than plotted (see module note).
    Returns n_in_disk, or None if npz has no 'phi'."""
    if 'phi' not in npz:
        return None
    d_eta = npz['eta'] - eta_probe
    d_phi = _wrap_phi(npz['phi'] - phi_probe)
    dR = np.sqrt(d_eta**2 + d_phi**2)
    return int((dR <= cone_radius).sum())


def _rebin_histogram(old_edges, old_counts, new_edges):
    """Redistribute a histogram from old_edges onto new_edges, assuming
    counts are uniformly distributed WITHIN each old bin (a good enough
    approximation as long as new_edges aren't much coarser than the
    physical structure of the underlying curve). Used to put a curve that
    is only available pre-histogrammed (the recipe's own analytic
    prediction, built by summing per-row histograms rather than from raw
    per-particle values) onto the SAME bins as sampled/npz curves, which
    can just be re-histogrammed directly from their raw values instead."""
    new_counts = np.zeros(len(new_edges) - 1)
    old_widths = np.diff(old_edges)
    density = np.divide(old_counts, old_widths, out=np.zeros_like(old_counts, dtype=float),
                         where=old_widths > 0)
    for lo, hi, dens in zip(old_edges[:-1], old_edges[1:], density):
        if dens == 0:
            continue
        for j, (nlo, nhi) in enumerate(zip(new_edges[:-1], new_edges[1:])):
            overlap = min(hi, nhi) - max(lo, nlo)
            if overlap > 0:
                new_counts[j] += dens * overlap
    return new_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recipe', required=True, help="Output of build_pu_sampling_recipe.py -- "
                         "STRONGLY recommended to have been built with --geometric_cut (see module "
                         "docstring): this script needs FINITE eta bins to define a local areal "
                         "density at all.")
    parser.add_argument('--npz', type=str, default=None,
                         help="Output of extract_gensim_particles.py -- if given, adds an npz-truth "
                              "curve/ratio to every shape plot, using whichever restriction actually "
                              "matches what that plot depends on (see the module note above "
                              "npz_pt_truth_in_cone) rather than a naive ΔR cut that would waste "
                              "most of npz's statistics on a small cone.")
    parser.add_argument('--displaced_threshold', type=float, default=1e-2,
                         help="Must match the value used to build --recipe (build_pu_sampling_"
                              "recipe.py's own default) for the npz-truth displacement curve to be "
                              "apples-to-apples.")
    parser.add_argument('--n_plot_bins', type=int, default=None,
                         help="If given, REBIN the pt, displacement, and radial-profile shape plots "
                              "onto this many bins each (same count for all three, over each "
                              "quantity's own natural range), instead of pt/displacement inheriting "
                              "however many bins --recipe happened to be built with. This is also "
                              "the direct lever against a jagged/noisy sampled or recipe curve: a "
                              "cone with modest statistics (small --cone_radius, few --n_events, or "
                              "few displaced particles behind the recipe's own per-row histograms) "
                              "looks noisy on fine bins for the same reason any histogram does "
                              "-- coarsening the DISPLAY bins (this flag) trades resolution for "
                              "robustness without needing to rebuild --recipe with different "
                              "--n_pt_bins/--n_disp_bins, which would fragment ITS OWN statistics "
                              "instead. Default (unset): keep each plot's existing, possibly "
                              "differing, bin counts, exactly as before this flag existed.")
    parser.add_argument('--eta_probe', type=float, required=True, help="Probe direction eta.")
    parser.add_argument('--phi_probe', type=float, required=True, help="Probe direction phi [-pi, pi].")
    parser.add_argument('--cone_radius', type=float, required=True,
                         help="Disk radius R in the flat (eta, phi) plane (ΔR = "
                              "sqrt(Δeta^2+Δphi^2) convention) around the probe direction to sample "
                              "local pileup within.")
    parser.add_argument('--n_pu', type=int, default=200, help="Target number of PU interactions "
                         "(same meaning as sample_pu_event.py's --n_pu -- scales the recipe's own "
                         "per-interaction rate, now restricted to the cone).")
    parser.add_argument('--fluctuate', action='store_true', default=False)
    parser.add_argument('--n_events', type=int, default=50, help="Number of toy cone realizations "
                         "to sample for the control plots.")
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--cms_label', default='Simulation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    bins, displacement_by_pdgid, disp_hists, n_events_npz = spe.load_recipe(args.recipe)

    # phi_probe doesn't enter cone_bin_weights -- the disk is azimuthally
    # symmetric around the probe direction and the recipe has no phi
    # dependence to begin with, so only eta_probe and cone_radius matter
    # for the per-bin weights.
    weights_nominal, coverage = cone_bin_weights(bins, args.n_pu, args.eta_probe, args.cone_radius)
    expected_total = sum(weights_nominal.values())
    print(f'Cone around (eta={args.eta_probe:+.3f}, phi={args.phi_probe:+.3f}), R={args.cone_radius}: '
          f'expected {expected_total:.3f} particles per N_pu={args.n_pu}-scaled event, from '
          f'{len(weights_nominal)} recipe eta bin(s).')
    print(f'Recipe coverage of the cone\'s own area: {100. * coverage:.1f}%'
          + ('' if coverage > 0.999 else ' -- part of the cone falls outside every recipe eta bin '
                                          '(probe too close to the recipe\'s own eta acceptance edge); '
                                          'the local density there is UNDERESTIMATED, see module docstring.'))
    for b, w in sorted(weights_nominal.items()):
        print(f'    bin {b} [{bins[b]["eta_lo"]:.3f}, {bins[b]["eta_hi"]:.3f}]: expected {w:.3f}')

    rng = np.random.default_rng(args.seed)
    n_preview = min(3, args.n_events)
    for i in range(n_preview):
        n_pu_used, particles = sample_local_pu_around_probe(
            bins, displacement_by_pdgid, disp_hists, args.n_pu, args.fluctuate,
            args.eta_probe, args.phi_probe, args.cone_radius, rng)
        n_disp = sum(1 for p in particles if p[4] > 0)
        print(f'Event {i}: N_pu={n_pu_used}, {len(particles)} local PU particles sampled '
              f'({n_disp} displaced)')

    npz = spe.load_npz(args.npz, args.displaced_threshold) if args.npz else None
    if npz is not None and 'phi' not in npz:
        print('\n(--npz given but the file has no \'phi\' array -- the radial-profile npz-truth '
              'curve will be skipped; pt and displacement npz-truth curves do not need it.)')

    if not args.outdir:
        return

    rng = np.random.default_rng(args.seed)  # re-seed, same convention as sample_pu_event.py
    os.makedirs(args.outdir, exist_ok=True)

    all_totals, all_pid, all_pt, all_deta, all_dphi, all_d3d = [], [], [], [], [], []
    for _ in range(args.n_events):
        _, particles = sample_local_pu_around_probe(
            bins, displacement_by_pdgid, disp_hists, args.n_pu, args.fluctuate,
            args.eta_probe, args.phi_probe, args.cone_radius, rng)
        all_totals.append(len(particles))
        for pid, eta, phi, pt, d3d, b in particles:
            all_pid.append(pid); all_pt.append(pt); all_d3d.append(d3d)
            all_deta.append(eta - args.eta_probe)
            all_dphi.append(_wrap_phi(phi - args.phi_probe))
    all_totals = np.array(all_totals)
    all_pid = np.array(all_pid); all_pt = np.array(all_pt); all_d3d = np.array(all_d3d)
    all_deta = np.array(all_deta); all_dphi = np.array(all_dphi)
    all_displaced = all_d3d > 0.

    print(f'\nSampled {args.n_events} cone realizations, {len(all_pid)} particles total, '
          f'mean={all_totals.mean():.2f} (expected {expected_total:.2f}), std={all_totals.std():.2f} '
          f'(Poisson prediction sqrt(expected)={np.sqrt(expected_total):.2f})')

    # --- Plot 1: total count in the cone, closure vs the analytic expectation,
    # plus (if --npz given) an independent npz-based cross-check of that
    # same expectation: sum of every npz particle's own chord-width weight
    # within the disk's eta extent (same weighting as npz_pt_truth_in_cone),
    # turned into a per-N_pu-scaled-event rate via n_pu/n_events_npz -- this
    # tests the recipe's OWN eta binning against finer-grained npz truth,
    # independent of anything this script's sampler does. ---
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.hist(all_totals, bins=30, color='#3f90da', edgecolor='none')
    ax.axvline(expected_total, color='#bd1f01', linestyle='--', linewidth=2,
               label=f'recipe (own bins) expectation = {expected_total:.2f}')
    if npz is not None:
        d_eta_npz = npz['eta'] - args.eta_probe
        in_window = np.abs(d_eta_npz) <= args.cone_radius
        chord_weights = _chord_weight(d_eta_npz[in_window], args.cone_radius)
        npz_expected_total = args.n_pu * chord_weights.sum() / n_events_npz
        ax.axvline(npz_expected_total, color='black', linestyle=':', linewidth=2,
                   label=f'npz truth expectation = {npz_expected_total:.2f}')
        print(f'npz-truth cross-check of the cone total: {npz_expected_total:.2f} '
              f'(recipe/npz ratio = {expected_total / npz_expected_total:.3f}), from '
              f'{int(in_window.sum())} npz particles in the disk\'s eta window.')
    ax.set_xlabel('Particles sampled inside the cone, per event')
    ax.set_ylabel('Events')
    ax.legend(fontsize=13)
    spe._add_cms_label(ax, args.cms_label)
    spe._save(fig, os.path.join(args.outdir, 'total_multiplicity_in_cone.png'))

    # --- Plot 2: (deta, dphi) coverage of the cone -- a 2D histogram, with
    # the disk boundary and any recipe eta-bin edges crossing it overlaid,
    # so a visible density STEP at a bin boundary is recognizable as an
    # expected feature of the recipe's own per-bin-uniform model, not a bug. ---
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hist2d(all_deta, all_dphi, bins=40,
                    range=[[-args.cone_radius, args.cone_radius], [-args.cone_radius, args.cone_radius]],
                    cmap='viridis')
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(args.cone_radius * np.cos(theta), args.cone_radius * np.sin(theta),
            color='white', linewidth=1.5, linestyle='--')
    for bin_info in bins:
        for edge in (bin_info['eta_lo'], bin_info['eta_hi']):
            d = edge - args.eta_probe
            if np.isfinite(d) and abs(d) < args.cone_radius:
                ax.axvline(d, color='white', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('eta - eta_probe'); ax.set_ylabel('phi - phi_probe (wrapped)')
    ax.set_aspect('equal')
    fig.colorbar(hb[3], ax=ax, label='particles')
    spe._add_cms_label(ax, args.cms_label)
    spe._save(fig, os.path.join(args.outdir, 'cone_coverage_2d.png'))

    # --- Plot 3: radial profile dN/dR shape -- closure test of the
    # uniform-in-area sampling + per-bin geometric weighting together.
    # Analytic expectation built the same way as cone_bin_weights, but
    # accumulated on a fine R grid: expected count within radius R' is
    # itself sum_b rho_b * area_b(R'), so its derivative (finite
    # difference) is the expected dN/dR shape. npz truth: see
    # npz_radial_profile_in_cone's docstring -- a high-statistics analytic
    # curve using npz's own fine-grained eta density, not a literal (low-
    # stats) ΔR cut; that literal cut is still computed and printed as an
    # independent, position-based cross-check. ---
    dR = np.sqrt(all_deta**2 + all_dphi**2)
    n_r_bins = args.n_plot_bins if args.n_plot_bins else 40
    r_edges = np.linspace(0., args.cone_radius, n_r_bins + 1)

    def expected_count_within(Rp):
        w, _ = cone_bin_weights(bins, args.n_pu, args.eta_probe, Rp)
        return sum(w.values())

    cum_expected = np.array([expected_count_within(r) for r in r_edges])
    expected_counts_per_bin = np.diff(cum_expected)
    sampled_counts_per_bin = np.histogram(dR, bins=r_edges)[0]
    npz_dR_counts = None
    if npz is not None:
        npz_dR_counts, n_npz_in_window = npz_radial_profile_in_cone(
            npz, args.eta_probe, args.cone_radius, r_edges, n_events_npz, args.n_pu)
        print(f'npz-truth radial profile: analytic curve from {n_npz_in_window} npz particles in '
              f'the disk\'s eta window (fine-binned, phi-uniformity-based -- see module docstring).')
        n_literal = npz_radial_profile_literal_cut(npz, args.eta_probe, args.phi_probe, args.cone_radius)
        if n_literal is not None:
            print(f'  cross-check: {n_literal} npz particles land in the disk under a LITERAL ΔR cut '
                  f'using npz\'s own real eta AND phi (not plotted -- low statistics by construction, '
                  f'since only ~cone_radius/pi of npz\'s full phi range can land in any one disk).')
    spe._plot_shape_comparison(r_edges, sampled_counts_per_bin, expected_counts_per_bin, npz_dR_counts,
                                args.n_events, args.cms_label,
                                os.path.join(args.outdir, 'radial_profile.png'),
                                xlabel='Delta R from probe', ylabel='dN/dR shape [a.u.]')

    # --- Plot 4: aggregate pt shape, sampled vs the cone-weighted recipe
    # curve (see aggregate_recipe_pt_in_cone's docstring for why this needs
    # its own per-bin weighting, unlike sample_pu_event.py's version), plus
    # the chord-weighted npz truth (npz_pt_truth_in_cone). If --n_plot_bins
    # is given, all three curves are rebinned onto that many bins spanning
    # the recipe's own pt range (log-spaced, matching the log-x display):
    # sampled/npz are re-histogrammed directly from their raw pt values
    # (no approximation), while the recipe curve -- only available pre-
    # histogrammed -- goes through _rebin_histogram. ---
    recipe_pt_counts, pt_edges = aggregate_recipe_pt_in_cone(bins, weights_nominal)
    if args.n_plot_bins:
        # The recipe's own pt histogram starts at exactly 0. (build_pu_
        # sampling_recipe.py's linear low-pt part) -- log10(0) is
        # undefined, so the log-spaced rebin starts from the smallest
        # POSITIVE edge instead; the negligible sliver of the original
        # [0, that edge) bin is folded into the new first bin by
        # _rebin_histogram's own overlap logic (its content still gets
        # assigned to whichever new bin it falls under, just clipped
        # rather than extending the log axis down to zero).
        pt_lo = pt_edges[pt_edges > 0].min()
        new_pt_edges = np.logspace(np.log10(pt_lo), np.log10(pt_edges[-1]), args.n_plot_bins + 1)
        new_pt_edges[0] = pt_edges[0]  # extend the first bin down to the recipe's true lower
                                        # edge (0.) instead of pt_lo, so _rebin_histogram doesn't
                                        # silently drop the [0, pt_lo) sliver's content
        recipe_pt_counts = _rebin_histogram(pt_edges, recipe_pt_counts, new_pt_edges)
        pt_edges = new_pt_edges
    sampled_pt_counts = np.histogram(all_pt, bins=pt_edges)[0]
    npz_pt_counts = None
    if npz is not None:
        npz_pt_counts, n_npz_pt = npz_pt_truth_in_cone(npz, args.eta_probe, args.cone_radius, pt_edges)
        print(f'npz-truth pt shape: {n_npz_pt} npz particles in the disk\'s eta window '
              f'(chord-width weighted, no phi cut -- see module docstring).')
    spe._plot_shape_comparison(pt_edges, sampled_pt_counts, recipe_pt_counts, npz_pt_counts,
                                args.n_events, args.cms_label,
                                os.path.join(args.outdir, 'pt_spectrum_cone.png'),
                                xlabel='pt [GeV]', ylabel='pt shape [a.u.]', xscale='log')

    # --- Plot 5: aggregate displacement shape -- reuses aggregate_recipe_
    # displacement UNCHANGED (eta-blind by construction, see module
    # docstring), so it's the same curve inside a cone as for the whole
    # recipe; only the SAMPLED curve is cone-specific -- see the module
    # note above on why this is a known, expected whole-acceptance-average
    # limitation of the recipe's own curve, not a bug. npz truth is
    # chord-weighted within the disk's eta window (npz_displacement_truth_
    # in_cone), NOT the full unrestricted sample -- see that function's
    # docstring. Same --n_plot_bins rebinning as pt above, linear this
    # time (the quantity is already log10(displacement)). ---
    recipe_disp_counts, disp_edges = spe.aggregate_recipe_displacement(
        displacement_by_pdgid, disp_hists, args.n_pu, n_events_npz)
    if args.n_plot_bins:
        new_disp_edges = np.linspace(disp_edges[0], disp_edges[-1], args.n_plot_bins + 1)
        recipe_disp_counts = _rebin_histogram(disp_edges, recipe_disp_counts, new_disp_edges)
        disp_edges = new_disp_edges
    sampled_log_d = np.log10(all_d3d[all_displaced]) if all_displaced.any() else np.array([])
    sampled_disp_counts = np.histogram(sampled_log_d, bins=disp_edges)[0]
    npz_disp_counts = None
    if npz is not None:
        npz_disp_counts, n_npz_in_window, n_npz_displaced = npz_displacement_truth_in_cone(
            npz, args.eta_probe, args.cone_radius, disp_edges)
        print(f'npz-truth displacement shape: {n_npz_in_window} npz particles in the disk\'s eta '
              f'window ({n_npz_displaced} displaced), chord-width weighted -- NOT the full '
              f'unrestricted npz sample, see module docstring.')
    spe._plot_shape_comparison(disp_edges, sampled_disp_counts, recipe_disp_counts, npz_disp_counts,
                                args.n_events, args.cms_label,
                                os.path.join(args.outdir, 'displacement_log10d_cone.png'),
                                xlabel='log10(3D displacement / cm)', ylabel='Displacement shape [a.u.]')
    if args.n_plot_bins:
        print(f'\n--n_plot_bins={args.n_plot_bins}: pt, displacement, and radial-profile plots all '
              f'rebinned to {args.n_plot_bins} bins each.')


if __name__ == '__main__':
    main()
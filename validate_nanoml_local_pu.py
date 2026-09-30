#!/usr/bin/env python3
"""
validate_nanoml_local_pu.py

Event-by-event closure test of the CMSSW local-PU gun's ACTUAL GEN-level
output (DisplacedParticleGunProducerFlatEtaWithLocalPU, read back from a
nanoML production over the GEN-SIM it produced) against the two things
that predicted it: the sampling recipe's own analytic cone expectation,
and (optionally) the original PU npz truth sample the recipe was built
from -- both evaluated using THIS SAME cone (same R, and EACH EVENT'S OWN
probe eta/phi, not one fixed probe direction repeated many times like
sample_local_pu_around_probe.py's own toy closure test). This is the
"did the C++ producer actually do what the Python prototype said it
would" check -- everything here is downstream of, and independent of,
that prototype's own validation.

WHAT "PROBE" MEANS HERE: nanoML's GenPart collection now carries an
EXACT, generator-level truth label for this -- GenPart_isLocalPU (0 =
probe, 1 = locally-sampled local PU), sourced from
DisplacedParticleGunProducerFlatEtaWithLocalPU's own "particleOrigin"
product via genParticleOriginTable's edm::ValueMap<int> (see
GenParticleOriginValueMapProducer.cc and nanoHGCML_cff.py). This
REPLACES the earlier vertex-distance heuristic ("largest 3D distance to
GenVtx") this script used before that branch existed: the heuristic was
a model-free but indirect proxy (it could mis-tag an event whose probe
happened to be produced close to the origin, and required a hand-tuned
--min_probe_distance floor); GenPart_isLocalPU is the producer's own
ground truth and needs no such floor, no distance threshold, and no
pdgId assumption at all. --probe_pdgid is kept purely as an optional
sanity cross-check on top of the label, and a mismatch is reported as a
warning, not silently ignored or fatal. (The vertex-distance reasoning
above -- PU anchored at the origin/primary-vertex region, the probe's
own vertex sitting near Geometry.Production.Z -- is still exactly why
GenPart_vx/vy/vz vs GenVtx_x/y/z gives the right per-particle 3D
displacement used below for the prompt/displaced split; it's just no
longer what SELECTS the probe.)

WHAT'S COMPARED, AND WHY EACH GETS ITS OWN TREATMENT:
  - total particles in the cone (count/density): the recipe's cone_bin_
    weights depends on eta_probe (a cone at different eta_probe overlaps
    different, differently-dense recipe bins), so this is recomputed
    PER EVENT at that event's own eta_probe and averaged -- exactly what
    DisplacedParticleGunProducerFlatEtaWithLocalPU.cc's own addLocalPileup
    does internally, just recomputed here in Python from first principles
    as an independent check.
  - pt shape: also eta-bin-dependent (see aggregate_recipe_pt_in_cone's
    own docstring in sample_local_pu_around_probe.py) -- summed PER EVENT
    the same way.
  - displacement shape (and prompt fraction): the recipe's displacement
    table has NO eta dependence at all (keyed only by pdgId and momentum,
    pooled across the recipe's WHOLE eta acceptance when it was built) --
    so, exactly as sample_local_pu_around_probe.py's own Plot 5 does, this
    is compared against ONE whole-acceptance-average analytic curve, not
    recomputed per event. See aggregate_recipe_displacement's own
    docstring (in sample_pu_event.py) for why that is a known limitation
    of the recipe's OWN curve, not something this validation can fix.
  - npz truth: costs O(n_events x len(npz)) if done for every event, so
    it's restricted to a random --npz_check_events subsample (default
    200) of the processed events, clearly reported as such. The pt/
    displacement/prompt npz-truth helpers are reused UNCHANGED from
    sample_local_pu_around_probe.py (chord-weighted, eta-window
    restricted -- see that module's own docstring for why this is NOT a
    naive phi cut).
  - EVERYTHING above is ALSO split by species GROUP (photon/electron/
    muon/hadron, sample_pu_event.py's SPECIES_GROUPS), mirroring that
    module's own make_control_plots Plots 6-13: pt_spectrum_cone_
    <group>.png and displacement_log10d_cone_<group>.png, plus a
    composition_groups_cone.png bar chart and a printed "Species-GROUP
    summary" table (mean count/cone, local areal density, and fraction of
    all in-cone particles -- sim/recipe/npz side by side, an "other"
    bucket, mostly neutrinos, included so the fractions visibly sum to
    ~100%). The per-group RECIPE pt curve is only genuine (not all-zero)
    if --recipe was built with build_pu_sampling_recipe.py's
    --pt_by_species -- see aggregate_recipe_pt_for_group's own docstring;
    this script prints a NOTE and falls back to sim-vs-npz-only for that
    one curve if it's missing. The per-group RECIPE displacement curve has
    no such caveat (the recipe's displacement table is already per-pdgId).

On EVERY plot below where a curve is available, sim + recipe + npz truth
are drawn TOGETHER on the same axes (never split across separate plots):
"sim"/"sampled" here means the ACTUAL simulated GEN-level output of
DisplacedParticleGunProducerFlatEtaWithLocalPU, read back from --nanoml
via GenPart_isLocalPU -- NOT a Python toy sample (that's what
sample_local_pu_around_probe.py's own closure test uses "sampled" for);
"recipe" is build_pu_sampling_recipe.py's analytic prediction; "npz
truth" is the recipe's own underlying extract_gensim_particles.py sample,
independent of both. Every shape comparison (pt, displacement) is
UNIT-AREA normalized before plotting (spe._plot_shape_comparison), so
curves built from different absolute event counts (all processed events
for sim/recipe, a subsample for npz) remain directly comparable -- only
their SHAPES are being checked here, exactly as in the scripts this one
validates against.

Usage:
    python validate_nanoml_local_pu.py \\
        --nanoml nanoml_output.root \\
        --recipe pu_sampling_recipe_hgcal.root \\
        --npz synthetic_gensim.npz \\
        --n_pu 200 --cone_radius 0.4 --probe_pdgid 22 \\
        --outdir nanoml_validation_plots/
"""

import argparse
import os
import sys

import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sample_pu_event as spe
import sample_local_pu_around_probe as slp


def _wrap_phi(phi):
    return (phi + np.pi) % (2. * np.pi) - np.pi


def _event_pv_from_particles(vx_i, vy_i, vz_i, pdg_i):
    """Robust per-event primary-vertex estimate computed DIRECTLY from
    this event's own GenPart_vx/vy/vz, with no separate 'event vertex'
    branch involved at all: the vertex of the LARGEST group of the
    event's hadrons that share a bit-identical vz (falls back to all
    particles if there are no hadrons) -- the overwhelming majority of
    local-PU particles really do sit exactly at the true smeared primary
    vertex, so its vz value is, by construction, the single most common
    one in the event. Same estimator and reasoning as sample_pu_event.py's
    own estimate_event_pv/`_mode_representative` (needed there because
    the npz truth has no vertex branch to trust either), just written for
    one event's arrays directly instead of a big batched (event_idx, v)
    pair across many events -- see find_probe_index's docstring and the
    module docstring for why this is used INSTEAD OF GenVtx_x/y/z: for at
    least one real sample, GenVtx_x/y/z turned out not to be the same
    smeared frame VtxSmeared actually applied to GenPart_vx/vy/vz, which
    made EVERY particle (including true local PU sitting at the real PV)
    look displaced by a large, roughly constant, spurious offset."""
    for mask in (np.abs(pdg_i) > 100, np.ones(len(pdg_i), dtype=bool)):
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        vals, counts = np.unique(vz_i[idx], return_counts=True)
        best_vz = vals[np.argmax(counts)]
        rep = idx[np.flatnonzero(vz_i[idx] == best_vz)[0]]
        return float(vx_i[rep]), float(vy_i[rep]), float(vz_i[rep])
    return 0., 0., 0.  # unreachable for a non-empty event (the all-True mask always matches)


def load_nanoml_events(path, tree, max_events=None, pv_source='mode'):
    """Loads the jagged GenPart_* branches (one numpy object-array entry
    per event) needed to recover each particle's own (un-smeared-frame)
    displacement -- GenPart_vx/vy/vz minus the event's own smeared
    primary vertex, exactly undoing VtxSmeared's one common per-event
    shift (see DisplacedParticleGunProducerFlatEtaWithLocalPU.cc's own
    comments on this). GenPart_isLocalPU is the exact probe/PU truth
    label this script now selects on (see module docstring);
    GenPart_genBarcode is loaded too, purely as an optional diagnostic
    (e.g. to eyeball that PU barcodes really do start right after the
    probe's), not used in any selection logic below. GenVtx_x/y/z is
    only REQUIRED if --pv_source=genvtx; it's always loaded when present
    anyway (for the printed PV cross-check against --pv_source=mode's own
    estimate -- see main()), but its absence is only fatal if genvtx was
    explicitly asked for. Deliberately avoids a hard dependency on
    awkward-array: uproot's library='np' already returns one numpy array
    of numpy arrays (dtype object) per jagged branch, which is all this
    script needs."""
    f = uproot.open(path)
    t = f[tree]
    required = ['GenPart_eta', 'GenPart_phi', 'GenPart_pt', 'GenPart_pdgId',
                'GenPart_vx', 'GenPart_vy', 'GenPart_vz', 'GenPart_isLocalPU']
    optional = ['GenPart_genBarcode', 'GenVtx_x', 'GenVtx_y', 'GenVtx_z']
    available = set(t.keys())
    missing = [b for b in required if b not in available]
    if missing:
        raise KeyError(f"--nanoml tree '{tree}' is missing branch(es) {missing} -- check --tree, or "
                        f"that this sample's nanoHGCML_cff.py was built with genParticleOriginTable wired "
                        f"in (see nanoHGCML_cff.py's isLocalPU/genBarcode externalVariables and "
                        f"customizeNoLocalPUTagging) and its GEN-SIM step kept particleOrigin/"
                        f"particleBarcode in outputCommands (see run_gun_nanoml.sh).")
    if pv_source == 'genvtx' and any(b not in available for b in ('GenVtx_x', 'GenVtx_y', 'GenVtx_z')):
        raise KeyError("--pv_source=genvtx requires GenVtx_x/y/z, which are missing from this tree -- "
                        "use --pv_source=mode (the default) instead, which needs no vertex branch beyond "
                        "GenPart_vx/vy/vz itself.")
    branches = required + [b for b in optional if b in available]
    return t.arrays(branches, library='np', entry_stop=max_events)


def find_probe_index(is_local_pu_i):
    """Exact probe selector using the generator-level GenPart_isLocalPU
    truth label (see module docstring) -- replaces the earlier vertex-
    distance heuristic entirely. The cone is always centered on GenPart
    index 0, matching DisplacedParticleGunProducerFlatEtaWithLocalPU::
    produce() exactly: probe(s) are appended FIRST (barcodes 1..
    NParticles) and etaProbe/phiProbe are taken from particleIndex==0
    only, however many probe particles NParticles the event actually
    has. Returns (probe_idx, ok); ok is False for an empty event or if
    index 0 is unexpectedly NOT flagged isLocalPU==0 (wrong tree/file,
    or a sample not made with this gun -- see module docstring)."""
    if len(is_local_pu_i) == 0:
        return None, False
    if int(is_local_pu_i[0]) != 0:
        return None, False
    return 0, True


def aggregate_recipe_composition_in_cone(bins, bin_weights):
    """Cone-weighted analogue of sample_pu_event.py's aggregate_recipe_
    composition: species fractions weighted by each overlapping bin's OWN
    cone contribution (bin_weights[b], from cone_bin_weights) instead of
    its full n_pu * mean_multiplicity -- same reasoning as aggregate_
    recipe_pt_in_cone. Returns {pdgId: expected_count} (NOT yet turned
    into fractions, so callers can sum multiple events' worth first)."""
    totals = {}
    for b, w in bin_weights.items():
        bin_info = bins[b]
        for pid, frac in zip(bin_info['pdgids'], bin_info['fractions']):
            totals[int(pid)] = totals.get(int(pid), 0.) + w * frac
    return totals


def _npz_subset(npz, mask):
    """A view of an npz-truth dict restricted to `mask` -- used to build
    one per-SPECIES_GROUP npz dict (photon/electron/muon/hadron) so the
    existing npz_pt_truth_in_cone/npz_displacement_truth_in_cone/
    npz_prompt_weight_in_cone helpers (sample_local_pu_around_probe.py)
    can be reused UNCHANGED per group, exactly as they're used for the
    whole (ungrouped) npz sample. Every array field is subset elementwise;
    'n_events' (a scalar -- the npz file's own total event count) is
    passed through unchanged since none of those helpers ever read it."""
    return {k: (v[mask] if k != 'n_events' else v) for k, v in npz.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nanoml', required=True, help="nanoML ROOT file produced from the "
                         "local-PU gun's GEN-SIM output.")
    parser.add_argument('--tree', default='Events', help="TTree name inside --nanoml.")
    parser.add_argument('--recipe', required=True, help="Same build_pu_sampling_recipe.py output "
                         "used to configure the gun (LocalPileup.RecipeFile) -- STRONGLY recommended "
                         "to have been built with --geometric_cut, same requirement as "
                         "sample_local_pu_around_probe.py.")
    parser.add_argument('--npz', default=None, help="Optional: the recipe's own underlying "
                         "extract_gensim_particles.py npz truth, for an independent third-way check "
                         "against the actual GEANT-level (well, GEN-level) simulated output.")
    parser.add_argument('--n_pu', type=float, required=True, help="Must match LocalPileup.NPu from "
                         "the config that produced --nanoml, or the recipe-expectation curve is "
                         "meaningless.")
    parser.add_argument('--cone_radius', type=float, required=True, help="Must match "
                         "LocalPileup.ConeRadius from the config that produced --nanoml.")
    parser.add_argument('--probe_pdgid', type=int, default=22, help="Sanity cross-check only -- the "
                         "probe is identified exactly via GenPart_isLocalPU==0 (see module docstring), "
                         "not by this; a mismatch is reported as a warning.")
    parser.add_argument('--displaced_threshold', type=float, default=1e-2, help="cm. Must match the "
                         "value used to build --recipe for the displacement/prompt comparisons to be "
                         "apples-to-apples.")
    parser.add_argument('--pv_source', choices=['mode', 'genvtx'], default='mode', help="How to get "
                         "each event's smeared primary vertex, subtracted from GenPart_vx/vy/vz to "
                         "recover each particle's own displacement. 'mode' (default, robust): estimate "
                         "it directly from GenPart_vx/vy/vz itself, as the vertex shared by the LARGEST "
                         "group of the event's hadrons (same estimator sample_pu_event.py's "
                         "estimate_event_pv uses for npz truth, which has no vertex branch to trust "
                         "either) -- correct regardless of whether GenVtx_x/y/z matches VtxSmeared's "
                         "actual frame for this sample. 'genvtx': trust the GenVtx_x/y/z branch "
                         "directly (the original approach) -- use this only if you've confirmed it "
                         "actually matches (see the printed PV cross-check below, for the first few "
                         "used events, when the two disagree).")
    parser.add_argument('--max_events', type=int, default=None, help="Process only the first N "
                         "events (default: all events in --nanoml).")
    parser.add_argument('--npz_check_events', type=int, default=200, help="The npz-truth chord-"
                         "weighted comparison costs O(n_events x len(npz)) -- restrict it to a "
                         "random subsample of this many processed events rather than every event. "
                         "Ignored if --npz is not given.")
    parser.add_argument('--n_plot_bins', type=int, default=None, help="Rebin the pt and displacement "
                         "shape plots onto this many bins each -- same meaning as "
                         "sample_local_pu_around_probe.py's own --n_plot_bins.")
    parser.add_argument('--outdir', default=None, help="If given, write control plots here; "
                         "otherwise only print the numeric summary.")
    parser.add_argument('--cms_label', default='Simulation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    bins, displacement_by_pdgid, disp_hists, n_events_npz_recipe = spe.load_recipe(args.recipe)
    npz = spe.load_npz(args.npz, args.displaced_threshold) if args.npz else None

    arrs = load_nanoml_events(args.nanoml, args.tree, args.max_events, args.pv_source)
    n_events = len(arrs['GenPart_eta'])
    print(f'Loaded {n_events} event(s) from {args.nanoml}:{args.tree}.')

    rng = np.random.default_rng(args.seed)
    npz_check_idx = (set(rng.choice(n_events, size=min(args.npz_check_events, n_events),
                                     replace=False).tolist())
                      if npz is not None else set())

    n_in_cone_sim, coverage_list, expected_total_recipe = [], [], []
    pt_cone_all, d3d_displaced_all = [], []
    pdg_cone_all = []
    n_prompt_sim = 0
    n_skipped = n_pdgid_mismatch = 0

    pt_edges_ref = bins[0]['pt_edges']
    recipe_pt_counts_total = np.zeros(len(pt_edges_ref) - 1)
    recipe_species_totals = {}

    npz_expected_total = []
    npz_pt_counts_total = None
    npz_prompt_total = 0.
    npz_disp_counts_total = None
    n_npz_events_used = 0

    # --- species-GROUP (photon/electron/muon/hadron) accumulators -- same
    # three quantities (in-cone count, pt shape, displacement shape) as the
    # aggregate-over-species ones above, just split by SPECIES_GROUPS
    # (sample_pu_event.py), mirroring that module's own make_control_plots
    # Plots 6-13 but cone-restricted/per-event-weighted like everything
    # else in this script. has_pt_by_group mirrors aggregate_recipe_pt_
    # for_group's own availability check: a per-group RECIPE pt curve only
    # exists if --recipe was built with build_pu_sampling_recipe.py's
    # --pt_by_species; sim and npz-truth per-group pt curves are always
    # available regardless (they come from real per-particle pdgId, not
    # the recipe file). ---
    species_groups = spe.SPECIES_GROUPS
    has_pt_by_group = bool(bins[0].get('pt_by_group')) if bins else False
    if not has_pt_by_group:
        print('NOTE: --recipe has no per-species pt histograms (build_pu_sampling_recipe.py\'s '
              '--pt_by_species) -- the per-group pt_spectrum_cone_<group>.png plots below will '
              'show simulated and npz-truth curves only, no genuine per-group recipe curve (same '
              'limitation as aggregate_recipe_pt_for_group -- see its docstring).')

    n_in_cone_sim_group = {g: [] for g, _ in species_groups}
    pt_cone_group = {g: [] for g, _ in species_groups}
    d3d_displaced_group = {g: [] for g, _ in species_groups}
    n_prompt_sim_group = {g: 0 for g, _ in species_groups}
    recipe_pt_counts_group = {g: np.zeros(len(pt_edges_ref) - 1) for g, _ in species_groups}

    npz_by_group = ({gname: _npz_subset(npz, gsel(npz['pdgId'])) for gname, gsel in species_groups}
                     if npz is not None else {})
    npz_expected_group = {g: [] for g, _ in species_groups}
    npz_pt_counts_group = {g: None for g, _ in species_groups}
    npz_prompt_group = {g: 0. for g, _ in species_groups}
    npz_disp_counts_group = {g: None for g, _ in species_groups}

    n_pv_cross_check_printed = 0
    for i in range(n_events):
        eta_i, phi_i = arrs['GenPart_eta'][i], arrs['GenPart_phi'][i]
        pt_i, pdg_i = arrs['GenPart_pt'][i], arrs['GenPart_pdgId'][i]
        vx_i, vy_i, vz_i = arrs['GenPart_vx'][i], arrs['GenPart_vy'][i], arrs['GenPart_vz'][i]
        is_local_pu_i = arrs['GenPart_isLocalPU'][i]

        probe_idx, ok = find_probe_index(is_local_pu_i)
        if not ok:
            n_skipped += 1
            continue
        if int(pdg_i[probe_idx]) != args.probe_pdgid:
            n_pdgid_mismatch += 1

        # Per-event smeared primary vertex -- see --pv_source's own help
        # for why 'mode' (default) is preferred over trusting GenVtx_x/y/z
        # directly. Cross-check printed for the first 3 USED events either
        # way, so a GenVtx/mode disagreement (a sign GenVtx isn't actually
        # VtxSmeared's own frame for this sample) is visible immediately
        # rather than only showing up much later as an all-zero displaced
        # histogram with a suspicious 0% prompt fraction.
        gvx_mode, gvy_mode, gvz_mode = _event_pv_from_particles(vx_i, vy_i, vz_i, pdg_i)
        gvx_branch, gvy_branch, gvz_branch = (float(arrs['GenVtx_x'][i]), float(arrs['GenVtx_y'][i]),
                                               float(arrs['GenVtx_z'][i])) if 'GenVtx_x' in arrs else \
                                              (None, None, None)
        if n_pv_cross_check_printed < 3 and gvx_branch is not None:
            d_pv = np.sqrt((gvx_mode - gvx_branch)**2 + (gvy_mode - gvy_branch)**2
                            + (gvz_mode - gvz_branch)**2)
            print(f'  PV cross-check (event {i}): mode-estimate=({gvx_mode:.4f}, {gvy_mode:.4f}, '
                  f'{gvz_mode:.4f})  GenVtx branch=({gvx_branch:.4f}, {gvy_branch:.4f}, '
                  f'{gvz_branch:.4f})  |difference|={d_pv:.4f} cm'
                  + ('  <-- LARGE: GenVtx_x/y/z does not look like the same smeared frame as '
                     'GenPart_vx/vy/vz for this sample -- --pv_source=mode (the default) is the '
                     'one to trust here.' if d_pv > 10. * args.displaced_threshold else ''))
            n_pv_cross_check_printed += 1
        gvx, gvy, gvz = ((gvx_mode, gvy_mode, gvz_mode) if args.pv_source == 'mode'
                         else (gvx_branch, gvy_branch, gvz_branch))

        eta_probe, phi_probe = float(eta_i[probe_idx]), float(phi_i[probe_idx])

        # Exact PU mask from the truth label (isLocalPU==1) -- correctly
        # excludes every probe particle even if NParticles>1, not just
        # index 0, unlike the old "mask everything but probe_idx" heuristic.
        dist = np.sqrt((vx_i - gvx)**2 + (vy_i - gvy)**2 + (vz_i - gvz)**2)
        pu_mask = (is_local_pu_i == 1)
        deta = eta_i[pu_mask] - eta_probe
        dphi = _wrap_phi(phi_i[pu_mask] - phi_probe)
        incone = np.hypot(deta, dphi) <= args.cone_radius

        pt_c = pt_i[pu_mask][incone]
        pdg_c = pdg_i[pu_mask][incone]
        d3d_c = dist[pu_mask][incone]
        displaced_c = d3d_c > args.displaced_threshold

        n_in_cone_sim.append(int(incone.sum()))
        pt_cone_all.append(pt_c)
        pdg_cone_all.append(pdg_c)
        if displaced_c.any():
            d3d_displaced_all.append(d3d_c[displaced_c])
        n_prompt_sim += int((~displaced_c).sum())

        weights_i, coverage_i = slp.cone_bin_weights(bins, args.n_pu, eta_probe, args.cone_radius)
        coverage_list.append(coverage_i)
        expected_total_recipe.append(sum(weights_i.values()))
        counts_i, _ = slp.aggregate_recipe_pt_in_cone(bins, weights_i)
        recipe_pt_counts_total += counts_i
        for pid, w in aggregate_recipe_composition_in_cone(bins, weights_i).items():
            recipe_species_totals[pid] = recipe_species_totals.get(pid, 0.) + w

        # --- per-species-GROUP sim + recipe, same event, same cone -- see
        # accumulator setup comment above. ---
        for gname, gsel in species_groups:
            gmask = gsel(pdg_c)
            n_in_cone_sim_group[gname].append(int(gmask.sum()))
            if gmask.any():
                pt_cone_group[gname].append(pt_c[gmask])
                disp_g = displaced_c[gmask]
                n_prompt_sim_group[gname] += int((~disp_g).sum())
                if disp_g.any():
                    d3d_displaced_group[gname].append(d3d_c[gmask][disp_g])
            if has_pt_by_group:
                counts_i_g, _ = slp.aggregate_recipe_pt_in_cone(bins, weights_i, group_name=gname)
                recipe_pt_counts_group[gname] += counts_i_g

        if npz is not None and i in npz_check_idx:
            n_npz_events_used += 1
            d_eta_npz = npz['eta'] - eta_probe
            in_window = np.abs(d_eta_npz) <= args.cone_radius
            chord_weights = slp._chord_weight(d_eta_npz[in_window], args.cone_radius)
            npz_expected_total.append(args.n_pu * chord_weights.sum() / n_events_npz_recipe)

            pt_counts_i, _ = slp.npz_pt_truth_in_cone(npz, eta_probe, args.cone_radius, pt_edges_ref)
            npz_pt_counts_total = (pt_counts_i if npz_pt_counts_total is None
                                    else npz_pt_counts_total + pt_counts_i)

            prompt_w, _ = slp.npz_prompt_weight_in_cone(npz, eta_probe, args.cone_radius)
            npz_prompt_total += prompt_w

            disp_counts_i, _, _ = slp.npz_displacement_truth_in_cone(
                npz, eta_probe, args.cone_radius, disp_hists['all']['edges'])
            npz_disp_counts_total = (disp_counts_i if npz_disp_counts_total is None
                                      else npz_disp_counts_total + disp_counts_i)

            # --- same three npz-truth quantities, per species group, using
            # the group-filtered npz_by_group views built once above. ---
            for gname, _ in species_groups:
                npz_g = npz_by_group[gname]
                if len(npz_g['eta']) == 0:
                    continue
                d_eta_g = npz_g['eta'] - eta_probe
                in_window_g = np.abs(d_eta_g) <= args.cone_radius
                chord_w_g = slp._chord_weight(d_eta_g[in_window_g], args.cone_radius)
                npz_expected_group[gname].append(args.n_pu * chord_w_g.sum() / n_events_npz_recipe)

                pt_counts_i_g, _ = slp.npz_pt_truth_in_cone(npz_g, eta_probe, args.cone_radius,
                                                              pt_edges_ref)
                npz_pt_counts_group[gname] = (pt_counts_i_g if npz_pt_counts_group[gname] is None
                                               else npz_pt_counts_group[gname] + pt_counts_i_g)

                prompt_w_g, _ = slp.npz_prompt_weight_in_cone(npz_g, eta_probe, args.cone_radius)
                npz_prompt_group[gname] += prompt_w_g

                disp_counts_i_g, _, _ = slp.npz_displacement_truth_in_cone(
                    npz_g, eta_probe, args.cone_radius, disp_hists['all']['edges'])
                npz_disp_counts_group[gname] = (disp_counts_i_g if npz_disp_counts_group[gname] is None
                                                 else npz_disp_counts_group[gname] + disp_counts_i_g)

    n_used = len(n_in_cone_sim)
    if n_used == 0:
        raise RuntimeError('No event had GenPart index 0 flagged GenPart_isLocalPU==0 -- either an '
                            'empty sample, the wrong --tree, or this file was not produced with '
                            'DisplacedParticleGunProducerFlatEtaWithLocalPU\'s genParticleOriginTable '
                            'wired in (see module docstring). Nothing to validate.')
    if n_skipped:
        print(f'WARNING: skipped {n_skipped}/{n_events} event(s) -- GenPart index 0 was not flagged '
              f'GenPart_isLocalPU==0 (empty event, or an unexpected sample layout).')
    if n_pdgid_mismatch:
        print(f'WARNING: the GenPart_isLocalPU==0-selected probe had pdgId != --probe_pdgid='
              f'{args.probe_pdgid} in {n_pdgid_mismatch}/{n_used} event(s) -- double-check '
              f'--probe_pdgid.')

    n_in_cone_sim = np.array(n_in_cone_sim)
    expected_total_recipe = np.array(expected_total_recipe)
    coverage_arr = np.array(coverage_list)
    pt_cone_all = np.concatenate(pt_cone_all) if pt_cone_all else np.array([])
    pdg_cone_all = np.concatenate(pdg_cone_all) if pdg_cone_all else np.array([])
    d3d_displaced_all = np.concatenate(d3d_displaced_all) if d3d_displaced_all else np.array([])
    n_displaced_sim = len(d3d_displaced_all)

    for gname, _ in species_groups:
        n_in_cone_sim_group[gname] = np.array(n_in_cone_sim_group[gname])
        pt_cone_group[gname] = (np.concatenate(pt_cone_group[gname])
                                 if pt_cone_group[gname] else np.array([]))
        d3d_displaced_group[gname] = (np.concatenate(d3d_displaced_group[gname])
                                       if d3d_displaced_group[gname] else np.array([]))
        npz_expected_group[gname] = np.array(npz_expected_group[gname])

    disk_area = np.pi * args.cone_radius**2
    print(f'\n{n_used} event(s) with an identified probe (R={args.cone_radius}, N_pu={args.n_pu}):')
    print(f'  simulated  mean particles/cone = {n_in_cone_sim.mean():.3f} +- '
          f'{n_in_cone_sim.std() / np.sqrt(n_used):.3f}  (std={n_in_cone_sim.std():.3f}, '
          f'Poisson-expected std=sqrt(mean)={np.sqrt(n_in_cone_sim.mean()):.3f})')
    print(f'  recipe     mean expectation (own bins, evaluated at each event\'s own eta_probe) = '
          f'{expected_total_recipe.mean():.3f}   ->  sim/recipe = '
          f'{n_in_cone_sim.mean() / expected_total_recipe.mean():.3f}')
    print(f'  mean recipe coverage of the cone\'s own area = {100. * coverage_arr.mean():.1f}%'
          + ('' if coverage_arr.min() > 0.999 else
             f' (min over events = {100. * coverage_arr.min():.1f}% -- some events have the cone '
             f'spilling past the recipe\'s own finite eta acceptance; see build_pu_sampling_recipe.py'
             f'\'s --geometric_cut and this module\'s docstring)'))
    print(f'  simulated local areal density = {n_in_cone_sim.mean() / disk_area:.3f} particles / '
          f'(eta x phi unit area)   vs   recipe = {expected_total_recipe.mean() / disk_area:.3f}')

    if npz is not None:
        npz_expected_total = np.array(npz_expected_total)
        print(f'  npz-truth  mean expectation ({n_npz_events_used}-event random subsample of the '
              f'same probe directions) = {npz_expected_total.mean():.3f}   ->  sim/npz = '
              f'{n_in_cone_sim.mean() / npz_expected_total.mean():.3f}, recipe/npz = '
              f'{expected_total_recipe.mean() / npz_expected_total.mean():.3f}')

    n_total_sim = n_prompt_sim + n_displaced_sim
    n_prompt_recipe = spe.aggregate_recipe_prompt_weight(displacement_by_pdgid, args.n_pu,
                                                           n_events_npz_recipe)
    recipe_disp_counts, disp_edges = spe.aggregate_recipe_displacement(
        displacement_by_pdgid, disp_hists, args.n_pu, n_events_npz_recipe)
    n_total_recipe = n_prompt_recipe + recipe_disp_counts.sum()
    print(f'\nPrompt fraction -- simulated: {n_prompt_sim}/{n_total_sim} = '
          f'{100. * n_prompt_sim / n_total_sim:.2f}%;  recipe (whole-acceptance average, eta-blind '
          f'by construction -- see module docstring): {100. * n_prompt_recipe / n_total_recipe:.2f}%'
          + (f';  npz truth (same {n_npz_events_used}-event subsample): '
             f'{100. * npz_prompt_total / (npz_prompt_total + npz_disp_counts_total.sum()):.2f}%'
             if npz is not None else ''))

    # --- species-GROUP (photon/electron/muon/hadron) summary: mean count/
    # cone, local areal density, and fraction of ALL in-cone particles
    # (including the ungrouped "other" remainder, mostly neutrinos), for
    # sim/recipe/npz side by side. recipe_group_totals collapses the
    # pdgId-level recipe_species_totals (already cone-and-per-event-
    # weighted) via spe._group_for_pid, so "other" falls out for free
    # (no separate accumulator needed) as whatever recipe_species_totals
    # sums to that no SPECIES_GROUPS selector claims. ---
    recipe_group_totals = {}
    for pid, w in recipe_species_totals.items():
        recipe_group_totals[spe._group_for_pid(pid)] = recipe_group_totals.get(
            spe._group_for_pid(pid), 0.) + w
    recipe_total_all = sum(recipe_species_totals.values())
    npz_total_mean = npz_expected_total.mean() if npz is not None else None

    print(f'\nSpecies-GROUP summary in the cone (density in particles / (eta x phi unit area), '
          f'disk area = pi*R^2 = {disk_area:.4f}):')
    for gname, _ in species_groups:
        n_sim_g = n_in_cone_sim_group[gname]
        mean_sim_g = n_sim_g.mean() if len(n_sim_g) else 0.
        frac_sim_g = mean_sim_g / n_in_cone_sim.mean() if n_in_cone_sim.mean() > 0 else float('nan')
        mean_recipe_g = recipe_group_totals.get(gname, 0.) / n_used
        frac_recipe_g = (recipe_group_totals.get(gname, 0.) / recipe_total_all
                          if recipe_total_all > 0 else float('nan'))
        line = (f'  {gname:>9s}:  sim mean/cone={mean_sim_g:7.3f} (dens={mean_sim_g / disk_area:6.2f}, '
                f'{100. * frac_sim_g:5.2f}%)   recipe mean/cone={mean_recipe_g:7.3f} '
                f'(dens={mean_recipe_g / disk_area:6.2f}, {100. * frac_recipe_g:5.2f}%)')
        if npz is not None and len(npz_expected_group[gname]):
            mean_npz_g = npz_expected_group[gname].mean()
            frac_npz_g = mean_npz_g / npz_total_mean if npz_total_mean > 0 else float('nan')
            line += (f'   npz mean/cone={mean_npz_g:7.3f} (dens={mean_npz_g / disk_area:6.2f}, '
                     f'{100. * frac_npz_g:5.2f}%)')
        print(line)
    # "other" (mostly neutrinos) -- whatever's left over, for the fractions
    # above to visibly sum to ~100% rather than silently omitting it.
    mean_sim_other = n_in_cone_sim.mean() - sum(
        (n_in_cone_sim_group[g].mean() if len(n_in_cone_sim_group[g]) else 0.) for g, _ in species_groups)
    frac_sim_other = mean_sim_other / n_in_cone_sim.mean() if n_in_cone_sim.mean() > 0 else float('nan')
    mean_recipe_other = recipe_group_totals.get('other', 0.) / n_used
    frac_recipe_other = (recipe_group_totals.get('other', 0.) / recipe_total_all
                          if recipe_total_all > 0 else float('nan'))
    print(f'  {"other":>9s}:  sim mean/cone={mean_sim_other:7.3f} (dens={mean_sim_other / disk_area:6.2f}, '
          f'{100. * frac_sim_other:5.2f}%)   recipe mean/cone={mean_recipe_other:7.3f} '
          f'(dens={mean_recipe_other / disk_area:6.2f}, {100. * frac_recipe_other:5.2f}%)  '
          f'(mostly neutrinos -- not otherwise plotted)')

    # --- species composition (a lightweight text table, not a plot): sim
    # counts in-cone vs the cone-and-per-event-weighted recipe expectation. ---
    print('\nSpecies composition in the cone (top species by simulated count):')
    sim_ids, sim_counts = np.unique(pdg_cone_all, return_counts=True)
    order = np.argsort(-sim_counts)
    recipe_total = sum(recipe_species_totals.values())
    for pid in sim_ids[order][:10]:
        sim_frac = sim_counts[order][list(sim_ids[order]).index(pid)] / len(pdg_cone_all)
        recipe_frac = recipe_species_totals.get(int(pid), 0.) / recipe_total if recipe_total > 0 else float('nan')
        n_sim_pid = int(sim_counts[list(sim_ids).index(pid)])
        print(f'    pdgId {int(pid):>6d}:  sim={100. * sim_frac:6.2f}%  recipe={100. * recipe_frac:6.2f}%  '
              f'(n_sim={n_sim_pid})')

    if not args.outdir:
        return
    os.makedirs(args.outdir, exist_ok=True)

    # --- Plot 1: per-event count in the cone ---
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.hist(n_in_cone_sim, bins=30, color='#3f90da', edgecolor='none', label='simulated (nanoML)')
    ax.axvline(expected_total_recipe.mean(), color='#bd1f01', linestyle='--', linewidth=2,
               label=f'recipe mean expectation = {expected_total_recipe.mean():.2f}')
    if npz is not None:
        ax.axvline(npz_expected_total.mean(), color='black', linestyle=':', linewidth=2,
                   label=f'npz-truth mean expectation = {npz_expected_total.mean():.2f}')
    ax.set_xlabel('Particles in the cone, per event')
    ax.set_ylabel('Events')
    ax.legend(fontsize=13)
    spe._add_cms_label(ax, args.cms_label)
    spe._save(fig, os.path.join(args.outdir, 'total_multiplicity_in_cone.png'))

    # --- Plot 2: pt shape ---
    pt_edges = pt_edges_ref
    if args.n_plot_bins:
        pt_lo = pt_edges[pt_edges > 0].min()
        new_pt_edges = np.logspace(np.log10(pt_lo), np.log10(pt_edges[-1]), args.n_plot_bins + 1)
        new_pt_edges[0] = pt_edges[0]
        recipe_pt_counts_total = slp._rebin_histogram(pt_edges, recipe_pt_counts_total, new_pt_edges)
        if npz_pt_counts_total is not None:
            npz_pt_counts_total = slp._rebin_histogram(pt_edges, npz_pt_counts_total, new_pt_edges)
        pt_edges = new_pt_edges
        sampled_pt_counts = np.histogram(pt_cone_all, bins=pt_edges)[0]
    else:
        sampled_pt_counts = np.histogram(pt_cone_all, bins=pt_edges)[0]
    spe._plot_shape_comparison(pt_edges, sampled_pt_counts, recipe_pt_counts_total, npz_pt_counts_total,
                                n_used, args.cms_label,
                                os.path.join(args.outdir, 'pt_spectrum_cone.png'),
                                xlabel='pt [GeV]', ylabel='pt shape [a.u.]', xscale='log')

    # --- Plot 3: displacement shape, with prepended prompt bin ---
    if args.n_plot_bins:
        new_disp_edges = np.linspace(disp_edges[0], disp_edges[-1], args.n_plot_bins + 1)
        recipe_disp_counts = slp._rebin_histogram(disp_edges, recipe_disp_counts, new_disp_edges)
        if npz_disp_counts_total is not None:
            npz_disp_counts_total = slp._rebin_histogram(disp_edges, npz_disp_counts_total, new_disp_edges)
        disp_edges = new_disp_edges
    sampled_log_d = np.log10(d3d_displaced_all) if n_displaced_sim else np.array([])
    sampled_disp_counts = np.histogram(sampled_log_d, bins=disp_edges)[0]

    full_disp_edges, full_sampled, full_recipe, full_npz = spe._prepend_prompt_bin(
        disp_edges, sampled_disp_counts, recipe_disp_counts, npz_disp_counts_total,
        n_prompt_sim, n_prompt_recipe, npz_prompt_total if npz is not None else None)
    spe._plot_shape_comparison(full_disp_edges, full_sampled, full_recipe, full_npz,
                                n_used, args.cms_label,
                                os.path.join(args.outdir, 'displacement_log10d_cone.png'),
                                xlabel='log10(3D displacement / cm)  (leftmost bin: prompt, d3d=0)',
                                ylabel='Displacement shape [a.u.]',
                                vline_x=disp_edges[0], vline_label='prompt | displaced')

    # --- Plot: species-GROUP composition/fractions in the cone (bar chart,
    # analogous to sample_pu_event.py's own composition.png but restricted
    # to SPECIES_GROUPS + 'other', and cone-and-per-event-weighted like
    # everything else here instead of the whole-recipe-acceptance-average
    # aggregate_recipe_composition uses). Reuses the exact same means as
    # the printed Species-GROUP summary table above, so the two agree by
    # construction. ---
    group_labels = [g for g, _ in species_groups] + ['other']
    sim_frac_bars, recipe_frac_bars = [], []
    npz_frac_bars = [] if npz is not None else None
    for gname in group_labels:
        if gname == 'other':
            mean_sim_x, mean_recipe_x = mean_sim_other, mean_recipe_other
        else:
            n_sim_x = n_in_cone_sim_group[gname]
            mean_sim_x = n_sim_x.mean() if len(n_sim_x) else 0.
            mean_recipe_x = recipe_group_totals.get(gname, 0.) / n_used
        sim_frac_bars.append(mean_sim_x / n_in_cone_sim.mean() if n_in_cone_sim.mean() > 0 else 0.)
        recipe_frac_bars.append(mean_recipe_x / expected_total_recipe.mean()
                                 if expected_total_recipe.mean() > 0 else 0.)
        if npz is not None:
            if gname == 'other':
                mean_npz_x = npz_total_mean - sum(
                    (npz_expected_group[g].mean() if len(npz_expected_group[g]) else 0.)
                    for g, _ in species_groups)
            else:
                mean_npz_x = npz_expected_group[gname].mean() if len(npz_expected_group[gname]) else 0.
            npz_frac_bars.append(mean_npz_x / npz_total_mean if npz_total_mean and npz_total_mean > 0 else 0.)

    sim_frac_bars = np.array(sim_frac_bars)
    recipe_frac_bars = np.array(recipe_frac_bars)
    fig, ax, rax = spe._ratio_fig()
    x = np.arange(len(group_labels))
    width = 0.27
    ax.bar(x - width, 100 * sim_frac_bars, width, color='#3f90da', label='simulated (nanoML PU gun)')
    ax.bar(x, 100 * recipe_frac_bars, width, color='#bd1f01', label='recipe')
    if npz is not None:
        npz_frac_bars = np.array(npz_frac_bars)
        ax.bar(x + width, 100 * npz_frac_bars, width, color='black', label='npz truth')
        rax.scatter(x - width / 2, spe._safe_ratio(sim_frac_bars, npz_frac_bars), color='#3f90da', zorder=3)
        rax.scatter(x + width / 2, spe._safe_ratio(recipe_frac_bars, npz_frac_bars), color='#bd1f01', zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(group_labels)
    ax.set_ylabel('Fraction of particles in the cone [%]')
    ax.legend(fontsize=12)
    rax.set_xticks(x); rax.set_xticklabels(group_labels)
    rax.set_xlabel('species group'); rax.set_ylim(0.5, 1.5)
    spe._add_cms_label(ax, args.cms_label)
    spe._save(fig, os.path.join(args.outdir, 'composition_groups_cone.png'))

    # --- Plots 4+: same two quantities (pt spectrum, displacement), split
    # by species GROUP (photon/electron/muon/hadron) -- mirrors sample_pu_
    # event.py's own make_control_plots Plots 6-13, but cone-restricted and
    # per-event-weighted like every other recipe curve in this script (see
    # the accumulator setup comment near the top of main()). Every group
    # reuses the SAME pt_edges/disp_edges as the aggregate Plots 2/3 above
    # (post --n_plot_bins rebin, if given), so shapes stay comparable group
    # to group and to the aggregate curve, exactly like sample_pu_event.py's
    # own per-group plots. ---
    for gname, gsel in species_groups:
        n_sim_g = len(pt_cone_group[gname])
        npz_has_g = npz is not None and npz_pt_counts_group[gname] is not None
        if n_sim_g == 0 and not npz_has_g:
            print(f'(no {gname} particles simulated or in npz: skipping {gname} plots)')
            continue

        # -- pt spectrum: sim + npz truth always; a genuine per-group
        # RECIPE curve only if has_pt_by_group (see accumulator setup). --
        recipe_pt_g = recipe_pt_counts_group[gname] if has_pt_by_group else None
        if recipe_pt_g is not None and args.n_plot_bins:
            recipe_pt_g = slp._rebin_histogram(pt_edges_ref, recipe_pt_g, pt_edges)
        npz_pt_g = npz_pt_counts_group[gname]
        if npz_pt_g is not None and args.n_plot_bins:
            npz_pt_g = slp._rebin_histogram(pt_edges_ref, npz_pt_g, pt_edges)
        sampled_pt_g = np.histogram(pt_cone_group[gname], bins=pt_edges)[0]
        spe._plot_shape_comparison(pt_edges, sampled_pt_g, recipe_pt_g, npz_pt_g,
                                    n_used, args.cms_label,
                                    os.path.join(args.outdir, f'pt_spectrum_cone_{gname}.png'),
                                    xlabel='pt [GeV]', ylabel='pt shape [a.u.]', xscale='log',
                                    title_extra=f'({gname})')

        # -- displacement, with the same prepended prompt bin as the
        # aggregate plot -- the recipe curve is the SAME KIND of whole-
        # acceptance-average analytic prediction as the aggregate one
        # (just restricted to this group via pdgid_filter), NOT cone-local
        # -- see module docstring for why that's a known limitation of the
        # recipe's own curve, not something fixable here. --
        group_recipe_disp_orig, disp_edges_orig = spe.aggregate_recipe_displacement(
            displacement_by_pdgid, disp_hists, args.n_pu, n_events_npz_recipe, pdgid_filter=gsel)
        group_n_prompt_recipe = spe.aggregate_recipe_prompt_weight(
            displacement_by_pdgid, args.n_pu, n_events_npz_recipe, pdgid_filter=gsel)
        group_recipe_disp = (slp._rebin_histogram(disp_edges_orig, group_recipe_disp_orig, disp_edges)
                              if args.n_plot_bins else group_recipe_disp_orig)

        npz_disp_g = npz_disp_counts_group[gname]
        if npz_disp_g is not None and args.n_plot_bins:
            npz_disp_g = slp._rebin_histogram(disp_edges_orig, npz_disp_g, disp_edges)
        group_n_prompt_npz = npz_prompt_group[gname] if npz is not None else None

        sampled_log_d_g = (np.log10(d3d_displaced_group[gname])
                            if len(d3d_displaced_group[gname]) else np.array([]))
        sampled_disp_g = np.histogram(sampled_log_d_g, bins=disp_edges)[0]

        g_full_edges, g_full_sampled, g_full_recipe, g_full_npz = spe._prepend_prompt_bin(
            disp_edges, sampled_disp_g, group_recipe_disp, npz_disp_g,
            n_prompt_sim_group[gname], group_n_prompt_recipe, group_n_prompt_npz)
        spe._plot_shape_comparison(g_full_edges, g_full_sampled, g_full_recipe, g_full_npz,
                                    n_used, args.cms_label,
                                    os.path.join(args.outdir, f'displacement_log10d_cone_{gname}.png'),
                                    xlabel='log10(3D displacement / cm)  (leftmost bin: prompt, d3d=0)',
                                    ylabel='Displacement shape [a.u.]',
                                    vline_x=disp_edges[0], vline_label='prompt | displaced',
                                    title_extra=f'({gname})')

    print(f'\nWrote control plots to {args.outdir}/')


if __name__ == '__main__':
    main()
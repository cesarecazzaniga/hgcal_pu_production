#!/usr/bin/env python3
"""
extract_gensim_particles.py

Reads gen-level particle info directly from one or more raw GEN-SIM files
(e.g. a minbias/pileup production, pileup_gensim_*.root) via FWLite, and
dumps everything MERGED into a single flat, easily-analyzable .npz -- a
first step toward characterizing what a full PU simulation looks like, for
later use building a sampling recipe for a local-PU gun (see this project's
own earlier discussion of that pipeline).

WHY FWLITE, NOT UPROOT: unlike every nanoML-reading script in this
project, a GEN-SIM file stores raw EDM C++ collections (std::vector<
reco::GenParticle>, SimTrack, SimVertex, ...), not flat NanoAOD-style
branches. uproot cannot deserialize these without CMSSW's own ROOT
dictionaries loaded -- FWLite (DataFormats.FWLite.Events/Handle) is the
standard CMSSW-native way to read a raw EDM file's collections directly,
without needing a full cmsRun job/config. REQUIRES an actual
cmsenv-sourced CMSSW environment to run -- this will NOT work in a plain
Python/uproot environment, no matter how the other scripts in this
project were run.

Reads the `genParticles` collection (the generator-level record) --
NOT SimTracks/SimHits (post-GEANT-propagation info) -- since the goal
here is characterizing what particles/kinematics a minbias interaction
PRODUCES, which is the natural level to build a gun's own sampling
distributions from. If you also want post-GEANT info (e.g. what actually
reaches HGCAL, accounting for decays/interactions in flight), the same
file also has g4SimHits' SimTrack/SimVertex collections available via
the same FWLite mechanism -- ask if you want that version too, the
read pattern is analogous but the collection/handle type differs.

By default, keeps only STABLE, FINAL-STATE particles (status()==1,
standard PYTHIA convention) -- a minbias event's full genParticles
collection also includes intermediate partons/resonances from the parton
shower and hadronization history, which are not directly relevant to
"what does the gun need to reproduce" (the gun shoots stable final
particles). Pass --all_status to keep everything instead.

MULTIPLE INPUT FILES / MERGING:
  * Pass any number of .root files (shell globs work as usual). An argument
    ending in .txt is treated as a file list: one path per line, blank
    lines and lines starting with '#' ignored. Both can be mixed.
  * Files are processed in the order given. `event_idx` in the output is a
    GLOBAL running index, unique across all files (0 .. n_events-1), so
    downstream per-event groupbys work unchanged on the merged output.
  * `file_idx` (per particle) indexes into the saved `files` array, so any
    particle can be traced back to its source file. `n_events_per_file`
    records how many events were actually processed from each file.
  * --max_events is a GLOBAL cap on the merged total, not per file; files
    after the cap is reached are not opened.
  * A file that fails to open, or lacks the requested collection, aborts
    the job by default. Pass --skip_bad_files to warn and continue instead
    (such files get n_events_per_file == 0).

Usage (run inside a cmsenv-sourced CMSSW environment):
    python extract_gensim_particles.py \\
        pileup_gensim_1.root pileup_gensim_2.root \\
        --outfile gensim_particles.npz \\
        --max_events 5000

    python extract_gensim_particles.py pileup_gensim_*.root filelist.txt
"""

import argparse
import sys

import numpy as np


# (output key, dtype, getter) for every per-particle quantity read from GenParticle.
PARTICLE_FIELDS = [
    ('pdgId',  np.int32,   lambda p: p.pdgId()),
    ('status', np.int32,   lambda p: p.status()),
    ('pt',     np.float32, lambda p: p.pt()),
    ('eta',    np.float32, lambda p: p.eta()),
    ('phi',    np.float32, lambda p: p.phi()),
    ('energy', np.float32, lambda p: p.energy()),
    ('vx',     np.float32, lambda p: p.vx()),
    ('vy',     np.float32, lambda p: p.vy()),
    ('vz',     np.float32, lambda p: p.vz()),
    ]


def expand_inputs(inputs):
    """Expand .txt file lists into paths; drop duplicates, preserving order."""
    paths = []
    for item in inputs:
        if item.endswith('.txt'):
            with open(item) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        paths.append(line)
        else:
            paths.append(item)

    seen, unique = set(), []
    for path in paths:
        if path in seen:
            print(f'WARNING: {path} given more than once, using it only once.')
            continue
        seen.add(path)
        unique.append(path)
    return unique


def process_file(path, file_idx, event_offset, n_remaining, handle, label, keep_all_status):
    """
    Read up to n_remaining events (None = all) from one file.

    Returns (n_events_processed, dict of per-particle numpy arrays for this file).
    event_idx values start at event_offset, so they are globally unique.
    """
    from DataFormats.FWLite import Events

    events = Events(path)
    n_in_file = events.size()
    n_process = n_in_file if n_remaining is None else min(n_in_file, n_remaining)
    print(f'{path}: {n_in_file} events found, processing {n_process}.')

    event_idx = []
    columns = {name: [] for name, _, _ in PARTICLE_FIELDS}

    n_done = 0
    for i, event in enumerate(events):
        if i >= n_process:
            break
        if i % 500 == 0:
            print(f'  event {i}/{n_process} (global event {event_offset + i})...')

        event.getByLabel(label, handle)
        if not handle.isValid():
            raise RuntimeError(
                f"collection '{label[0]}' not found in {path} -- check with "
                f"'edmDumpEventContent {path}' and pass --collection_label."
                )

        for p in handle.product():
            if not keep_all_status and p.status() != 1:
                continue
            event_idx.append(event_offset + i)
            for name, _, getter in PARTICLE_FIELDS:
                columns[name].append(getter(p))
        n_done += 1

    # Convert to compact numpy arrays per file, so memory stays reasonable when merging many files.
    arrays = {name: np.array(columns[name], dtype=dtype) for name, dtype, _ in PARTICLE_FIELDS}
    arrays['event_idx'] = np.array(event_idx, dtype=np.int32)
    arrays['file_idx'] = np.full(len(event_idx), file_idx, dtype=np.int32)
    return n_done, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infiles', type=str, nargs='+',
                         help="GEN-SIM .root file(s) to read and merge. An argument ending in "
                              ".txt is read as a file list (one path per line).")
    parser.add_argument('--outfile', type=str, default='gensim_particles.npz',
                         help="Output .npz path (single merged file).")
    parser.add_argument('--collection_label', type=str, default='genParticles',
                         help="EDM module label of the GenParticle collection. 'genParticles' "
                              "is the standard label for a plain GEN-SIM step; check with "
                              "'edmDumpEventContent your_file.root' if this doesn't match "
                              "(e.g. if the file went through additional processing).")
    parser.add_argument('--max_events', type=int, default=None,
                         help="Stop after this many events IN TOTAL across all input files "
                              "(default: all).")
    parser.add_argument('--all_status', action='store_true', default=False,
                         help="Keep all particles regardless of status() -- default keeps only "
                              "status()==1 (stable, final-state), see module docstring.")
    parser.add_argument('--skip_bad_files', action='store_true', default=False,
                         help="Warn and continue if a file cannot be opened/read, instead of "
                              "aborting.")
    args = parser.parse_args()

    try:
        import ROOT  # noqa: F401
        from DataFormats.FWLite import Handle
    except ImportError:
        sys.exit(
            "ERROR: could not import ROOT/DataFormats.FWLite. This script requires an actual "
            "cmsenv-sourced CMSSW environment -- it cannot run in a plain Python/uproot "
            "environment (see module docstring for why: GEN-SIM stores raw EDM C++ "
            "collections, not flat NanoAOD-style branches). Run 'cmsenv' in a CMSSW_X_Y_Z/src "
            "area first."
            )

    files = expand_inputs(args.infiles)
    if not files:
        sys.exit('ERROR: no input files given.')
    print(f'{len(files)} input file(s).')

    handle = Handle('std::vector<reco::GenParticle>')
    label = (args.collection_label,)

    per_file_arrays = []
    n_events_per_file = np.zeros(len(files), dtype=np.int64)
    n_events_total = 0

    for file_idx, path in enumerate(files):
        n_remaining = None
        if args.max_events is not None:
            n_remaining = args.max_events - n_events_total
            if n_remaining <= 0:
                print(f'--max_events {args.max_events} reached, not opening remaining '
                      f'{len(files) - file_idx} file(s).')
                break

        try:
            n_done, arrays = process_file(
                path, file_idx, n_events_total, n_remaining, handle, label, args.all_status)
        except Exception as e:  # FWLite/ROOT raise assorted exception types on bad files
            if not args.skip_bad_files:
                sys.exit(f'ERROR while reading {path}: {e}\n'
                         f'(pass --skip_bad_files to skip such files instead of aborting)')
            print(f'WARNING: skipping {path}: {e}')
            continue

        per_file_arrays.append(arrays)
        n_events_per_file[file_idx] = n_done
        n_events_total += n_done

    if n_events_total == 0:
        sys.exit('ERROR: no events processed, nothing written.')

    keys = ['event_idx', 'file_idx'] + [name for name, _, _ in PARTICLE_FIELDS]
    merged = {k: np.concatenate([a[k] for a in per_file_arrays]) for k in keys}
    n_particles_total = len(merged['event_idx'])

    print(f'\nDone: {n_particles_total} particles kept across {n_events_total} events from '
          f'{int((n_events_per_file > 0).sum())} file(s) '
          f'({n_particles_total / n_events_total:.1f} per event on average).')

    np.savez(
        args.outfile,
        n_events=n_events_total,
        n_events_per_file=n_events_per_file,
        files=np.array(files),
        **merged,
        )
    print(f'Wrote {args.outfile}')


if __name__ == '__main__':
    main()
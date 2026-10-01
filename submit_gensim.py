#!/usr/bin/env python3
"""
submit_gensim_pu.py

Computes how many condor jobs are needed to produce a given total number
of GEN-SIM minbias events at a chosen events-per-job, generates a unique
(seed, output file, n_events) triple per job, writes an HTCondor job-list
file + .sub file, and (optionally) submits.

Each job's output goes DIRECTLY to a remote xrootd URL -- CMSSW's own
PoolOutputModule writes natively to remote storage (confirmed from your
existing run_gensim.sh: outputfile=${2} was already an xrootd URL passed
straight to cmsRun) -- no separate stage-out step needed, unlike e.g. the
hgcal_npz.py pipeline's manual local_copy()/seutils.cp() staging (that
was needed there because plain np.savez can't write to xrootd directly;
CMSSW's own I/O can).

X.509 PROXY (--x509_proxy_path): the generated .sub file includes
use_x509userproxy/x509userproxy, needed for xrdcp/CMSSW to authenticate
against storage01.lcg.cscs.ch (a non-CERN site -- MY.SendCredential
alone forwards CERN-SSO tokens, which aren't what this endpoint accepts;
see this project's own debugging history for the "Auth failed: No
protocols left to try" failure this fixes). Defaults to $X509_USER_PROXY
if set, else the standard /tmp/x509up_u<uid> location -- override with
--x509_proxy_path for a non-standard location (e.g. an AFS-based proxy).
Checked for existence (and validity, if voms-proxy-info is available)
BEFORE writing the .sub file, so a missing/expired proxy is caught here
rather than discovered only after a failed batch submission.

Uses HTCondor's "queue ... from <file>" mechanism -- one .sub file, one
job-list text file with one line per job, rather than N separate .sub
files or N separate condor_submit calls. This is the standard, simple
way to submit a batch of jobs that each need different arguments.

Usage:
    # Dry run first -- check the job count/seeds/filenames before submitting
    python submit_gensim_pu.py \\
        --n_events_total 500000 \\
        --n_events_per_job 5000 \\
        --outdir root://storage01.lcg.cscs.ch:1096//pnfs/lcg.cscs.ch/cms/trivcat/store/user/cazzanig/pu_samples_hgcal/ \\
        --label pu_minbias \\
        --dry_run

    # Non-standard (e.g. AFS-based) proxy location
    python submit_gensim_pu.py \\
        ... \\
        --x509_proxy_path /afs/cern.ch/user/c/cazzanig/x509up_u143977

    # Submit for real (drop --dry_run)
    python submit_gensim_pu.py \\
        --n_events_total 500000 \\
        --n_events_per_job 5000 \\
        --outdir root://storage01.lcg.cscs.ch:1096//pnfs/lcg.cscs.ch/cms/trivcat/store/user/cazzanig/pu_samples_hgcal/ \\
        --label pu_minbias
"""

import argparse
import math
import os
import shutil
import subprocess
import sys


def default_proxy_path():
    """Best-effort default: $X509_USER_PROXY if set, else the standard
    /tmp/x509up_u<uid> convention. Always overridable via
    --x509_proxy_path for a non-standard location (e.g. an AFS-based
    proxy, as seen in this project's own debugging history)."""
    if 'X509_USER_PROXY' in os.environ and os.environ['X509_USER_PROXY']:
        return os.environ['X509_USER_PROXY']
    return f'/tmp/x509up_u{os.getuid()}'


def check_proxy(proxy_path):
    """Checks the proxy file exists and, if voms-proxy-info is
    available, that it isn't expired -- printed BEFORE writing the .sub
    file, so a missing/expired proxy is caught here rather than only
    discovered after a failed batch submission. Never raises -- a check
    that can't complete (tool unavailable, unexpected output) prints a
    warning and lets the caller decide whether to proceed, rather than
    blocking submission on a diagnostic step that itself might be
    unreliable in some environments."""
    if not os.path.isfile(proxy_path):
        print(f'WARNING: no proxy file found at {proxy_path}.')
        print(f'  Run: voms-proxy-init --voms cms')
        print(f'  (or pass --x509_proxy_path if your proxy lives somewhere non-standard)')
        return False

    print(f'Proxy file found at {proxy_path}.')
    if not shutil.which('voms-proxy-info'):
        print(f'  (voms-proxy-info not available here to check validity -- proceeding, but '
              f'verify manually if unsure)')
        return True

    result = subprocess.run(['voms-proxy-info', '-file', proxy_path, '-timeleft'],
                             capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  WARNING: voms-proxy-info could not read this proxy (may be invalid/corrupt):')
        print(f'  {result.stderr.strip()}')
        return False

    try:
        timeleft_sec = int(result.stdout.strip())
    except ValueError:
        print(f'  (could not parse voms-proxy-info output: {result.stdout.strip()!r} -- proceeding)')
        return True

    if timeleft_sec <= 0:
        print(f'  WARNING: proxy has EXPIRED. Run: voms-proxy-init --voms cms')
        return False

    print(f'  Valid, {timeleft_sec/3600:.1f} hours remaining.')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n_events_total', type=int, required=True,
                         help="Total number of GEN-SIM minbias events wanted, across all jobs.")
    parser.add_argument('--n_events_per_job', type=int, required=True,
                         help="Events per job. The LAST job is shrunk if --n_events_total isn't "
                              "evenly divisible, so the total produced matches exactly (never "
                              "overshoots).")
    parser.add_argument('--outdir', required=True,
                         help="Remote xrootd base directory for output files (must already "
                              "exist -- see the gfal-mkdir step this script prints below; "
                              "PoolOutputModule does not create remote directories on write).")
    parser.add_argument('--label', default='pu_minbias',
                         help="Short label used in output filenames and job-list/sub filenames.")
    parser.add_argument('--base_seed', type=int, default=1,
                         help="First job's seed; subsequent jobs use base_seed+1, +2, ... "
                              "(guarantees distinct random seeds across jobs -- reusing a seed "
                              "would silently produce IDENTICAL events in two different jobs).")
    parser.add_argument('--executable', default='run_gensim.sh',
                         help="Path to the condor executable script (expects 3 positional args: "
                              "seed, outputfile, n_events -- see run_gensim.sh).")
    parser.add_argument('--job_flavour', default='testmatch',
                         help="HTCondor +JobFlavour.")
    parser.add_argument('--logdir', default='logs',
                         help="Local directory for job stdout/stderr/log files -- created if "
                              "missing (condor requires this to exist BEFORE submission, a "
                              "common gotcha if skipped).")
    parser.add_argument('--outfile_prefix', default='submit_gensim_pu',
                         help="Prefix for the generated job-list and .sub filenames.")
    parser.add_argument('--x509_proxy_path', default=None,
                         help="Path to your X.509 grid proxy, forwarded to jobs via "
                              "use_x509userproxy/x509userproxy in the .sub file -- needed for "
                              "xrdcp/CMSSW to authenticate against storage01.lcg.cscs.ch (see "
                              "module docstring). Default: $X509_USER_PROXY if set, else "
                              "/tmp/x509up_u<uid>. Checked for existence/validity before writing "
                              "the .sub file.")
    parser.add_argument('--skip_proxy_check', action='store_true', default=False,
                         help="Skip the proxy existence/validity check entirely and write the "
                              ".sub file regardless. Not recommended -- mainly useful for "
                              "generating/inspecting the .sub file on a machine without a proxy "
                              "at all (e.g. while testing this script itself).")
    parser.add_argument('--dry_run', action='store_true', default=False,
                         help="Write the job-list/.sub files and print the plan, but do NOT "
                              "call condor_submit.")
    args = parser.parse_args()

    if args.n_events_total <= 0 or args.n_events_per_job <= 0:
        sys.exit("ERROR: --n_events_total and --n_events_per_job must both be > 0.")

    n_jobs = math.ceil(args.n_events_total / args.n_events_per_job)
    print(f'{args.n_events_total} events total, {args.n_events_per_job} per job '
          f'-> {n_jobs} job(s).')

    outdir = args.outdir if args.outdir.endswith('/') else args.outdir + '/'

    jobs = []  # (seed, outputfile, n_events)
    n_remaining = args.n_events_total
    for i in range(n_jobs):
        seed = args.base_seed + i
        n_this_job = min(args.n_events_per_job, n_remaining)
        n_remaining -= n_this_job
        outfile = f'{outdir}{args.label}_job{i}_seed{seed}.root'
        jobs.append((seed, outfile, n_this_job))

    assert n_remaining == 0, f"Internal error: {n_remaining} events unaccounted for"
    assert sum(j[2] for j in jobs) == args.n_events_total, "Job split doesn't sum to the requested total"

    print(f'\nFirst 3 jobs (of {n_jobs}):')
    for seed, outfile, n in jobs[:3]:
        print(f'  seed={seed}  n_events={n}  outfile={outfile}')
    if n_jobs > 3:
        print(f'  ... ({n_jobs - 3} more)')
    print(f'Last job: seed={jobs[-1][0]}  n_events={jobs[-1][2]}  outfile={jobs[-1][1]}'
          + ('  <- shrunk to make the total exact' if jobs[-1][2] != args.n_events_per_job else ''))

    # --- X.509 proxy check, BEFORE writing the .sub file ---
    proxy_path = args.x509_proxy_path or default_proxy_path()
    print(f'\nX.509 proxy check:')
    proxy_ok = check_proxy(proxy_path)
    if not proxy_ok and not args.skip_proxy_check:
        sys.exit(f'\nERROR: proxy check failed for {proxy_path} -- fix this before submitting '
                  f'(see warnings above), or pass --skip_proxy_check to write the .sub file '
                  f'anyway (not recommended -- jobs will fail the same way as before).')
    elif not proxy_ok:
        print(f'  --skip_proxy_check set: writing .sub file anyway despite the warning above.')

    # --- Write the job-list file (HTCondor "queue ... from <file>" input) ---
    joblist_path = f'{args.outfile_prefix}_joblist.txt'
    with open(joblist_path, 'w') as f:
        for seed, outfile, n in jobs:
            f.write(f'{seed}, {outfile}, {n}\n')
    print(f'\nWrote {joblist_path} ({n_jobs} lines)')

    # --- Write the .sub file ---
    os.makedirs(args.logdir, exist_ok=True)
    print(f'Ensured {args.logdir}/ exists (condor requires this before submission)')

    sub_path = f'{args.outfile_prefix}.sub'
    sub_content = f"""executable            = {args.executable}
arguments             = $(seed) $(outfile) $(nevents)

output                = {args.logdir}/{args.label}.$(ClusterId).$(ProcId).out
error                 = {args.logdir}/{args.label}.$(ClusterId).$(ProcId).err
log                    = {args.logdir}/{args.label}.$(ClusterId).log

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

+JobFlavour            = "{args.job_flavour}"

# request access to EOS/AFS tokens
MY.SendCredential      = true

# X.509 proxy -- needed for xrdcp/CMSSW to authenticate against
# storage01.lcg.cscs.ch (a non-CERN site; MY.SendCredential above
# forwards CERN-SSO tokens, which aren't what this endpoint accepts).
# Explicit path, since HTCondor's auto-detection defaults to
# /tmp/x509up_u<uid>, which may not be where your proxy actually lives.
use_x509userproxy      = true
x509userproxy           = {proxy_path}

queue seed, outfile, nevents from {joblist_path}
"""
    with open(sub_path, 'w') as f:
        f.write(sub_content)
    print(f'Wrote {sub_path}')

    print(f'\nBefore submitting for the first time, make sure the remote output directory '
          f'exists (PoolOutputModule does not create it on write):\n'
          f'  gfal-mkdir -p {outdir}')

    if args.dry_run:
        print(f'\n--dry_run set: NOT submitting. Review {sub_path}/{joblist_path}, then rerun '
              f'without --dry_run, or submit directly with:\n  condor_submit {sub_path}')
        return

    print(f'\nSubmitting {sub_path} ({n_jobs} jobs)...')
    result = subprocess.run(['condor_submit', sub_path], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(f'condor_submit failed (exit code {result.returncode})')


if __name__ == '__main__':
    main()
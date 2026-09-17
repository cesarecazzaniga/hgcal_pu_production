#!/bin/bash
set -euo pipefail

# $1 = seed, $2 = FINAL remote output destination (xrootd URL), $3 = number
# of events for this job.
SEED=${1}
REMOTE_OUTFILE=${2}
NEVENTS=${3}
CMSSW_DIR=/afs/cern.ch/user/c/cazzanig/private/llps_hgcal/CMSSW_15_1_0/src   # <-- adjust to your actual release path
CFG_DIR=${CMSSW_DIR}/hgcal_pu_production     # <-- adjust to where cfg_gensim_D121.py lives

# ---------------------------------------------------------------------------
# Isolated per-job scratch directory (unchanged from before).
# ---------------------------------------------------------------------------
WORKDIR=/tmp/${USER}_gensimpu_${SEED}_$$
mkdir -p ${WORKDIR}
echo "==> Working directory: ${WORKDIR}"
trap 'rm -rf ${WORKDIR}' EXIT
cd ${WORKDIR}

# ---------------------------------------------------------------------------
# X.509 proxy diagnostics -- run EARLY, before anything else, so the job
# log shows directly whether a valid credential is actually present at
# runtime. This is the fastest way to confirm/deny whether
# use_x509userproxy in the .sub file is actually working, without
# needing to log into a worker node or guess.
#
# Even with use_x509userproxy=true correctly staging a proxy file onto
# the worker node, xrdcp (and CMSSW) still need X509_USER_PROXY
# explicitly pointing at it. HTCondor OFTEN sets this automatically, but
# relying on that silently is a common source of exactly the
# "Auth failed" failure seen here -- made explicit below rather than
# assumed.
# ---------------------------------------------------------------------------
echo "==> X509 proxy diagnostics:"
echo "    X509_USER_PROXY = ${X509_USER_PROXY:-<UNSET>}"
if [ -n "${X509_USER_PROXY:-}" ] && [ -f "${X509_USER_PROXY}" ]; then
    echo "    Proxy file exists at that path."
    if command -v voms-proxy-info >/dev/null 2>&1; then
        voms-proxy-info -all -file "${X509_USER_PROXY}" || echo "    WARNING: voms-proxy-info could not read the proxy (may be expired/invalid)"
    else
        echo "    (voms-proxy-info not available in this environment to inspect it further)"
    fi
else
    echo "    WARNING: X509_USER_PROXY is unset or does not point to an existing file."
    echo "    If the xrdcp stage-out below fails with 'Auth failed', THIS is almost"
    echo "    certainly why -- check use_x509userproxy/x509userproxy in the .sub file,"
    echo "    and that voms-proxy-init was run (with enough remaining validity) before submission."
fi
echo ""

# ---------------------------------------------------------------------------
# CMSSW environment (unchanged from before).
# ---------------------------------------------------------------------------
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12                   # <-- match your release's SCRAM_ARCH

cd ${CMSSW_DIR}
eval $(scram runtime -sh)
cd ${WORKDIR}
echo "==> CMSSW environment set up from ${CMSSW_DIR}"

# ---------------------------------------------------------------------------
# Run cmsRun from WORKDIR (unchanged from before).
# ---------------------------------------------------------------------------
LOCAL_OUTFILE="${WORKDIR}/$(basename ${REMOTE_OUTFILE})"

echo "==> Running cmsRun (seed=${SEED}, n=${NEVENTS}), writing locally to ${LOCAL_OUTFILE}"
cmsRun ${CFG_DIR}/cfg_gensim_D121.py thing=minbias n=${NEVENTS} seed=${SEED} outputfile=file:${LOCAL_OUTFILE}
echo "==> cmsRun finished OK"

# ---------------------------------------------------------------------------
# Stage out to remote storage. If this fails with "Auth failed" again,
# the diagnostics printed above should already show whether a valid
# proxy was ever present -- check that output FIRST, before re-checking
# the .sub file.
# ---------------------------------------------------------------------------
echo "==> Staging out ${LOCAL_OUTFILE} -> ${REMOTE_OUTFILE}"
xrdcp -f "${LOCAL_OUTFILE}" "${REMOTE_OUTFILE}"

echo "==> Done (seed=${SEED})"
from __future__ import print_function
from pprint import pprint, pformat
from time import strftime

import common

import FWCore.ParameterSet.Config as cms


def gensim(thing, n_events, seed=1, output_file=None):
    gensim_driver = common.CMSDriver('TTbar_14TeV_TuneCP5_cfi', '--no_exec')
    gensim_driver.kwargs.update({
        '-s'             : 'GEN,SIM',
        '--conditions'   : 'auto:phase2_realistic_T33_13TeV',
        '--beamspot'     : 'HLLHC14TeV',
        '--datatier'     : 'GEN-SIM',
        '--eventcontent' : 'FEVTDEBUG',
        '--geometry'     : 'ExtendedRun4D121',
        '--era'          : 'Phase2C22I13M9',
        #'--procModifier' : 'fineCalo',
        })
    process = common.load_process_from_driver(gensim_driver, 'gensim_driver.py')
    common.rng(process, seed)
    common.activate_finecalo(process)
    process.maxEvents.input = cms.untracked.int32(n_events)
    process.source.firstLuminosityBlock = cms.untracked.uint32(1)

    if output_file is None: output_file = 'file:{}_gensim_D121_fine_n{}_{}.root'.format(thing, n_events, strftime('%b%d'))
    common.logger.info('Output: %s', output_file)
    process.FEVTDEBUGoutput.fileName = cms.untracked.string(output_file)

    process.FEVTDEBUGoutput.outputCommands.append("keep *_*G4*_*_*")
    process.FEVTDEBUGoutput.outputCommands.append("keep SimClustersedmAssociation_mix_*_*")
    process.FEVTDEBUGoutput.outputCommands.append("keep CaloParticlesedmAssociation_mix_*_*")

    common.add_generator(process, thing)
    return process


from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('analysis')
options.register('thing', 'minbias', VarParsing.multiplicity.singleton, VarParsing.varType.string, 'Choices: "tau", "muon", "minbias"')
options.register('n', 1, VarParsing.multiplicity.singleton, VarParsing.varType.int, 'Number of events')
options.register('seed', 1, VarParsing.multiplicity.singleton, VarParsing.varType.int, 'RNG seed')
options.register('outputfile', None, VarParsing.multiplicity.singleton, VarParsing.varType.string, 'Path to output file')
options.parseArguments()
process = gensim(options.thing, n_events=options.n, seed=options.seed, output_file=options.outputfile)
common.logger.info('Created process %s', process)


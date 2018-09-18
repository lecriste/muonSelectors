#!/bin/env python
#
# PyROOT study of standard selector performance using sim-hit matching 
# to identify fake and signal muons
#
import os, re, ROOT, sys, pickle, time
from pprint import pprint
from math import *
from array import array
from DataFormats.FWLite import Events, Handle
import numpy as np


##
## User Input
##

# dataset: /RelValTTbar_13/CMSSW_9_4_0_pre3-PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/MINIAODSIM
# lfns = [
#     '/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/D2849502-12C5-E711-8C77-24BE05C6E7C1.root',
#     '/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/181D6340-1EC5-E711-98D5-24BE05C68681.root',
#     '/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/008E7DB2-33C5-E711-BF25-9CDC714A4690.root',
#     '/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/D2E5D119-3FC5-E711-B101-E0071B73C630.root',
#     '/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/585F37EB-4EC5-E711-98C6-E0071B6CAD10.root',
#     '/store/user/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+D2849502-12C5-E711-8C77-24BE05C6E7C1.root',
#     '/store/user/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+181D6340-1EC5-E711-98D5-24BE05C68681.root',
#     '/store/user/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+008E7DB2-33C5-E711-BF25-9CDC714A4690.root',
#     '/store/user/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+D2E5D119-3FC5-E711-B101-E0071B73C630.root',
#     '/store/user/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+585F37EB-4EC5-E711-98C6-E0071B6CAD10.root'
#     ]

# /RelValQCD_FlatPt_15_3000HS_13/CMSSW_9_4_0_pre3-PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/MINIAODSIM
# /store/relval/CMSSW_9_4_0_pre3/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/561435D4-FBC4-E711-82FA-24BE05C626B1.root
# /store/relval/CMSSW_9_4_0_pre3/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/82E0EA96-12C5-E711-8045-E0071B693B41.root
# /store/relval/CMSSW_9_4_0_pre3/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/80FC1CBE-26C5-E711-8203-4C79BA1810F3.root
# /store/relval/CMSSW_9_4_0_pre3/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/46BE889F-3DC5-E711-B8CB-E0071B695B81.root
#
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValQCD_FlatPt_15_3000HS_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+561435D4-FBC4-E711-82FA-24BE05C626B1.root
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValQCD_FlatPt_15_3000HS_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+82E0EA96-12C5-E711-8045-E0071B693B41.root
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValQCD_FlatPt_15_3000HS_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+80FC1CBE-26C5-E711-8203-4C79BA1810F3.root
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValQCD_FlatPt_15_3000HS_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+46BE889F-3DC5-E711-B8CB-E0071B695B81.root

# /RelValZMM_13/CMSSW_9_4_0_pre3-PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/MINIAODSIM
# /store/relval/CMSSW_9_4_0_pre3/RelValZMM_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/5228FC24-10C5-E711-9B90-E0071B73B6C0.root
# /store/relval/CMSSW_9_4_0_pre3/RelValZMM_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/FE37E710-28C5-E711-98C6-4C79BA320467.root
# /store/relval/CMSSW_9_4_0_pre3/RelValZMM_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/2E5505C2-3DC5-E711-BA98-4C79BA180C81.root
# /store/relval/CMSSW_9_4_0_pre3/RelValZMM_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/D817C8EA-4EC5-E711-882C-24BE05C3EC61.root
# 
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+5228FC24-10C5-E711-9B90-E0071B73B6C0.root
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+FE37E710-28C5-E711-98C6-4C79BA320467.root
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+2E5505C2-3DC5-E711-BA98-4C79BA180C81.root
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+D817C8EA-4EC5-E711-882C-24BE05C3EC61.root

def getPFNs(lfns):
    files = []
    for file in lfns:
        fullpath = "/eos/cms/" + file
        if os.path.exists(fullpath):
            files.append(fullpath)
        else:
            raise Exception("File not found: %s" % fullpath)
    return files

# files = getPFNs(lfns)
studies = {
    'TTbar':{
        'files':[
            #'/data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+D2849502-12C5-E711-8C77-24BE05C6E7C1.root',
            #'/data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+181D6340-1EC5-E711-98D5-24BE05C68681.root',
            #'/data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+008E7DB2-33C5-E711-BF25-9CDC714A4690.root',
            #'/data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+D2E5D119-3FC5-E711-B101-E0071B73C630.root',
            #'/data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+585F37EB-4EC5-E711-98C6-E0071B6CAD10.root'
            # through AAA
            #'root://cmsxrootd.fnal.gov//store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/D2E5D119-3FC5-E711-B101-E0071B73C630.root',
            #'root://cmsxrootd.fnal.gov//store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/D2849502-12C5-E711-8C77-24BE05C6E7C1.root',
            #'root://cmsxrootd.fnal.gov//store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/585F37EB-4EC5-E711-98C6-E0071B6CAD10.root',
            #'root://cmsxrootd.fnal.gov//store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/181D6340-1EC5-E711-98D5-24BE05C68681.root',
            #'root://cmsxrootd.fnal.gov//store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/008E7DB2-33C5-E711-BF25-9CDC714A4690.root'
            # on disk
            'root://cmsxrootd.fnal.gov//store/relval/CMSSW_10_2_2/RelValTTbar_13/MINIAODSIM/PU25ns_102X_mcRun2_asymptotic_v3_FastSim-v2/10000/5858CF63-1B9E-E811-B34F-0025905A610C.root',
            'root://cmsxrootd.fnal.gov//store/relval/CMSSW_10_2_2/RelValTTbar_13/MINIAODSIM/PU25ns_102X_mcRun2_asymptotic_v3_FastSim-v2/10000/E666A4CB-A19D-E811-94ED-0025905A60DE.root'
            ],
        'maxBkgEff':0.15,
        'name':'ttbar'
        },
    'QCD + Zmm':{
        'files':[
            #'/data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+5228FC24-10C5-E711-9B90-E0071B73B6C0.root',
            #'/data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+FE37E710-28C5-E711-98C6-4C79BA320467.root',
            #'/data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+2E5505C2-3DC5-E711-BA98-4C79BA180C81.root',
            #'/data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+D817C8EA-4EC5-E711-882C-24BE05C3EC61.root',
            #'/data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValQCD_FlatPt_15_3000HS_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+561435D4-FBC4-E711-82FA-24BE05C626B1.root',
            #'/data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValQCD_FlatPt_15_3000HS_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+82E0EA96-12C5-E711-8045-E0071B693B41.root',
            #'/data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValQCD_FlatPt_15_3000HS_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+80FC1CBE-26C5-E711-8203-4C79BA1810F3.root',
            #'/data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValQCD_FlatPt_15_3000HS_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+46BE889F-3DC5-E711-B8CB-E0071B695B81.root'
            # through AAA
            #'root://cmsxrootd.fnal.gov//data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+5228FC24-10C5-E711-9B90-E0071B73B6C0.root',
            #'root://cmsxrootd.fnal.gov//data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+FE37E710-28C5-E711-98C6-4C79BA320467.root',
            #'root://cmsxrootd.fnal.gov//data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+2E5505C2-3DC5-E711-BA98-4C79BA180C81.root',
            #'root://cmsxrootd.fnal.gov//data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+D817C8EA-4EC5-E711-882C-24BE05C3EC61.root',
            #'root://cmsxrootd.fnal.gov//data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValQCD_FlatPt_15_3000HS_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+561435D4-FBC4-E711-82FA-24BE05C626B1.root',
            #'root://cmsxrootd.fnal.gov//data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValQCD_FlatPt_15_3000HS_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+82E0EA96-12C5-E711-8045-E0071B693B41.root',
            #'root://cmsxrootd.fnal.gov//data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValQCD_FlatPt_15_3000HS_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+80FC1CBE-26C5-E711-8203-4C79BA1810F3.root',
            #'root://cmsxrootd.fnal.gov//data/dmytro/tmp/store+relval+CMSSW_9_4_0_pre3+RelValQCD_FlatPt_15_3000HS_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+46BE889F-3DC5-E711-B8CB-E0071B695B81.root'
            # on disk
            'root://cmsxrootd.fnal.gov//store/relval/CMSSW_10_2_2/RelValZMM_13/MINIAODSIM/PU25ns_102X_mcRun2_asymptotic_v3_FastSim-v2/10000/0E846746-009E-E811-A436-0025905B8580.root'
            # more than one file
            # ?
            ],
        'maxBkgEff':0.015,
        'name':'qcd_zmm'
        }
    }

muonHandle, muonLabel = Handle("std::vector<pat::Muon>"), "slimmedMuons"

minPt = 20
maxPt = 1e9

n_events_limit = None
# n_events_limit = 10000

ROOT.gROOT.SetBatch(True)

##
## Main part
##

def LeptonMVA(muon,mva_threshold):
    dB2D  = abs(muon.dB(ROOT.pat.Muon.BS2D))
    dB3D  = abs(muon.dB(ROOT.pat.Muon.PV3D))
    edB3D = abs(muon.edB(ROOT.pat.Muon.PV3D))
    sip3D = 0.0
    if edB3D>0: sip3D = dB3D/edB3D
    # dz = abs(muon.muonBestTrack().dz(primaryVertex.position()));
    if muon.pt()>5 and muon.isLooseMuon() and muon.passed(ROOT.reco.Muon.MiniIsoLoose) and sip3D<8.0 and dB2D<0.05:
        if muon.mvaValue()>mva_threshold:
            return True
    return False

def pfIsolation(muon):
    chIso  = muon.pfIsolationR04().sumChargedHadronPt
    nIso   = muon.pfIsolationR04().sumNeutralHadronEt
    phoIso = muon.pfIsolationR04().sumPhotonEt
    puIso  = muon.pfIsolationR04().sumPUPt
    dbCorrectedIsolation = chIso + max( nIso + phoIso - .5*puIso, 0. )
    dbCorectedRelIso = dbCorrectedIsolation/muon.pt()
    return dbCorectedRelIso

def tkIsolation(muon):
    return muon.isolationR03().sumPt/muon.pt()

def print_canvas(canvas, output_name_without_extention, path):
    if not os.path.exists(path):
        os.makedirs(path)
    canvas.Print("%s/%s.png"%(path,output_name_without_extention))
    canvas.Print("%s/%s.pdf"%(path,output_name_without_extention))
    canvas.Print("%s/%s.root"%(path,output_name_without_extention))

for study,info in studies.items():
    print "Processing %s" % study
    title = study
    maxBkgEff = info['maxBkgEff']
    files = info['files']

    print "Number of files: %d" % len(files)

    events = Events(files)
    print "Total number of events: %d" % events.size()

    nevents = 0
    nSigTotal = 0
    nSigSelected = {}
    nBkgTotal = 0
    nBkgSelected = {}
    selectors = {
        'CutBasedIdLoose':{
            'mask':ROOT.reco.Muon.CutBasedIdLoose,
            'display':False
            },
        'CutBasedIdMediumPrompt':{
            'mask':ROOT.reco.Muon.CutBasedIdMediumPrompt,
            'display':False
            },
        'CutBasedIdTight':{
            'mask':ROOT.reco.Muon.CutBasedIdTight,
            'display':False
            },
        'SoftCutBasedId':{
            'mask':ROOT.reco.Muon.SoftCutBasedId,
            'display':False
            },
        'MvaLoose':{
            'mask':ROOT.reco.Muon.MvaLoose,
            'display':True,
            'marker':20,
            'color':ROOT.kBlack
            },
        'MvaTight':{
            'mask':ROOT.reco.Muon.MvaTight,
            'display':True,
            'marker':20,
            'color':ROOT.kBlack
            },
        'Tight ID + Loose PFIso':{
            'mask':ROOT.reco.Muon.CutBasedIdTight|ROOT.reco.Muon.PFIsoLoose,
            'display':True,
            'marker':22,
            'color':ROOT.kBlue
            },
        'Tight ID + Tight PFIso':{
            'mask':ROOT.reco.Muon.CutBasedIdTight|ROOT.reco.Muon.PFIsoTight,
            'display':True,
            'marker':22,
            'color':ROOT.kBlue
            },
        'MediumPrompt ID + Loose TkIso':{
            'mask':ROOT.reco.Muon.CutBasedIdMediumPrompt|ROOT.reco.Muon.TkIsoLoose,
            'display':True,
            'marker':23,
            'color':ROOT.kRed
            },
        'MediumPromt ID + Tight TkIso':{
            'mask':ROOT.reco.Muon.CutBasedIdMediumPrompt|ROOT.reco.Muon.TkIsoTight,
            'display':True,
            'marker':23,
            'color':ROOT.kRed
            },
        }

    for selector in selectors:
        nSigSelected[selector] = 0
        nBkgSelected[selector] = 0

    # array of mva values for the ROC curve
    mvaValues = np.arange(-1.0, 1.0, 0.05)
    pfIsoValues = np.arange(0.00, 0.40, 0.01)
    tkIsoValues = np.arange(0.00, 0.40, 0.01)
    # print mvaValues

    mvaEffSig = array( "f" )
    mvaEffBkg = array( "f" )
    for mva in mvaValues:
        mvaEffSig.append(0.0)
        mvaEffBkg.append(0.0)

    pfIsoEffSig = array( "f" )
    pfIsoEffBkg = array( "f" )
    for pfIso in pfIsoValues:
        pfIsoEffSig.append(0.0)
        pfIsoEffBkg.append(0.0)

    tkIsoEffSig = array( "f" )
    tkIsoEffBkg = array( "f" )
    for tkIso in tkIsoValues:
        tkIsoEffSig.append(0.0)
        tkIsoEffBkg.append(0.0)

    # loop over events
    for event in events:
        if n_events_limit and nevents>=n_events_limit: break
        if (nevents+1)%10000==0: print "Processing event ",nevents+1

        event.getByLabel(muonLabel, muonHandle)
        muons = muonHandle.product()
        for muon in muons:
            if muon.pt()<minPt or muon.pt()>maxPt: continue
            # signal muons
            trueMuon = (muon.simType()==ROOT.reco.MatchedPrimaryMuon)
            if trueMuon or nevents%2:
                nSigTotal += 1
            else:
                nBkgTotal += 1
            for name,selector in selectors.items():
                passed = muon.passed(selector['mask'])
                # print "\tpt: %0.1f \t%s" % (muon.pt(),tight)
                if passed:
                    if trueMuon:
                        nSigSelected[name]+=1
                    else:
                        nBkgSelected[name]+=1
            for i in range(len(mvaValues)):
                passed = LeptonMVA(muon,mvaValues[i])
                if passed:
                    if trueMuon:
                        mvaEffSig[i]+=1
                    else:
                        mvaEffBkg[i]+=1
            for i in range(len(pfIsoValues)):
                passed = muon.passed(ROOT.reco.Muon.CutBasedIdTight) and (pfIsolation(muon)<pfIsoValues[i])
                if passed:
                    if trueMuon:
                        pfIsoEffSig[i]+=1
                    else:
                        pfIsoEffBkg[i]+=1
            for i in range(len(tkIsoValues)):
                passed = muon.passed(ROOT.reco.Muon.CutBasedIdMediumPrompt) and (tkIsolation(muon)<tkIsoValues[i])
                if passed:
                    if trueMuon:
                        tkIsoEffSig[i]+=1
                    else:
                        tkIsoEffBkg[i]+=1
        nevents += 1

    for i in range(len(mvaValues)):
        mvaEffSig[i]/=nSigTotal
        mvaEffBkg[i]/=nBkgTotal

    for i in range(len(pfIsoValues)):
        pfIsoEffSig[i]/=nSigTotal
        pfIsoEffBkg[i]/=nBkgTotal

    for i in range(len(tkIsoValues)):
        tkIsoEffSig[i]/=nSigTotal
        tkIsoEffBkg[i]/=nBkgTotal

    # print mvaEffSig
    # print mvaEffBkg

    c1 = ROOT.TCanvas("c1", "ROC curve example",700,700)
    c1.SetLeftMargin(0.15)
    graphs = []

    graphMva = ROOT.TGraph(len(mvaEffSig), mvaEffSig, mvaEffBkg)
    graphMva.SetTitle(title)
    graphMva.SetMinimum(0.00)
    graphMva.SetMaximum(maxBkgEff)
    graphMva.SetLineColor(ROOT.kMagenta)
    graphMva.SetLineWidth(3)
    graphMva.GetXaxis().SetTitle("Signal Efficiency")
    graphMva.GetYaxis().SetTitle("Background Efficiency")
    graphMva.Draw("AC")
    graphs.append(graphMva)

    graphPfIso = ROOT.TGraph(len(pfIsoEffSig), pfIsoEffSig, pfIsoEffBkg)
    graphPfIso.SetLineColor(ROOT.kBlue)
    graphPfIso.SetLineWidth(3)
    graphPfIso.Draw("C same")
    graphs.append(graphPfIso)

    graphTkIso = ROOT.TGraph(len(tkIsoEffSig), tkIsoEffSig, tkIsoEffBkg)
    graphTkIso.SetLineColor(ROOT.kRed)
    graphTkIso.SetLineWidth(3)
    graphTkIso.Draw("C same")
    graphs.append(graphTkIso)

    print "Processed %d events" % nevents
    print "N signal muons: %d" % (nSigTotal)
    print "N background muons: %d" % (nBkgTotal)
    for selector in sorted(selectors):
        print "\t%s" % selector
        effS = float(nSigSelected[selector])/nSigTotal
        effB = float(nBkgSelected[selector])/nBkgTotal
        print "\t\tSig: N:%d (%0.2f%%)" % (nSigSelected[selector],100.*effS)
        print "\t\tBkg: N:%d (%0.2f%%)" % (nBkgSelected[selector],100.*effB)
        if selectors[selector]['display']:
            effSigArray = array("f",[effS])
            effBkgArray = array("f",[effB])
            graph = ROOT.TGraph(len(effSigArray), effSigArray, effBkgArray)
            graph.SetMarkerStyle(selectors[selector]['marker'])
            graph.SetMarkerColor(selectors[selector]['color'])
            graph.SetMarkerSize(2)
            graph.Draw("P same")
            graphs.append(graph)

    c1.Update()
    # path = "/eos/user/d/dmytro/www/plots/94X_ROCs/"
    path = "10_2_2_ROCs"
    print_canvas(c1, info['name'], path)
    # raw_input( ' ... ' )

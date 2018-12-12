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

#CMSSW = '9_4_0_pre3'
#CMSSW = '10_2_2'
#CMSSW = '10_3_0_pre4'
CMSSW = '10_2_5'

RelValQCD = 'RelValQCD_FlatPt_15_3000HS_13'
# dataset: /RelValTTbar_13/CMSSW_'+CMSSW+'-PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/MINIAODSIM
# lfns = [
#     '/store/relval/CMSSW_'+CMSSW+'/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/D2849502-12C5-E711-8C77-24BE05C6E7C1.root',
#     '/store/relval/CMSSW_'+CMSSW+'/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/181D6340-1EC5-E711-98D5-24BE05C68681.root',
#     '/store/relval/CMSSW_'+CMSSW+'/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/008E7DB2-33C5-E711-BF25-9CDC714A4690.root',
#     '/store/relval/CMSSW_'+CMSSW+'/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/D2E5D119-3FC5-E711-B101-E0071B73C630.root',
#     '/store/relval/CMSSW_'+CMSSW+'/RelValTTbar_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/585F37EB-4EC5-E711-98C6-E0071B6CAD10.root',
#     '/store/user/dmytro/tmp/store+relval+CMSSW_'+CMSSW+'+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+D2849502-12C5-E711-8C77-24BE05C6E7C1.root',
#     '/store/user/dmytro/tmp/store+relval+CMSSW_'+CMSSW+'+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+181D6340-1EC5-E711-98D5-24BE05C68681.root',
#     '/store/user/dmytro/tmp/store+relval+CMSSW_'+CMSSW+'+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+008E7DB2-33C5-E711-BF25-9CDC714A4690.root',
#     '/store/user/dmytro/tmp/store+relval+CMSSW_'+CMSSW+'+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+D2E5D119-3FC5-E711-B101-E0071B73C630.root',
#     '/store/user/dmytro/tmp/store+relval+CMSSW_'+CMSSW+'+RelValTTbar_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+585F37EB-4EC5-E711-98C6-E0071B6CAD10.root'
#     ]

# /'+RelValQCD+'+/CMSSW_'+CMSSW+'-PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/MINIAODSIM
# /store/relval/CMSSW_'+CMSSW+'/'+RelValQCD+'/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/561435D4-FBC4-E711-82FA-24BE05C626B1.root
# /store/relval/CMSSW_'+CMSSW+'/'+RelValQCD+'/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/82E0EA96-12C5-E711-8045-E0071B693B41.root
# /store/relval/CMSSW_'+CMSSW+'/'+RelValQCD+'/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/80FC1CBE-26C5-E711-8203-4C79BA1810F3.root
# /store/relval/CMSSW_'+CMSSW+'/'+RelValQCD+'/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/46BE889F-3DC5-E711-B8CB-E0071B695B81.root
#
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_'+CMSSW+'+'+RelValQCD+'+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+561435D4-FBC4-E711-82FA-24BE05C626B1.root
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_'+CMSSW+'+'+RelValQCD+'+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+82E0EA96-12C5-E711-8045-E0071B693B41.root
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_'+CMSSW+'+'+RelValQCD+'+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+80FC1CBE-26C5-E711-8203-4C79BA1810F3.root
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_'+CMSSW+'+'+RelValQCD+'+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+46BE889F-3DC5-E711-B8CB-E0071B695B81.root

# /RelValZMM_13/CMSSW_'+CMSSW+'-PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/MINIAODSIM
# /store/relval/CMSSW_'+CMSSW+'/RelValZMM_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/5228FC24-10C5-E711-9B90-E0071B73B6C0.root
# /store/relval/CMSSW_'+CMSSW+'/RelValZMM_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/FE37E710-28C5-E711-98C6-4C79BA320467.root
# /store/relval/CMSSW_'+CMSSW+'/RelValZMM_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/2E5505C2-3DC5-E711-BA98-4C79BA180C81.root
# /store/relval/CMSSW_'+CMSSW+'/RelValZMM_13/MINIAODSIM/PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1/10000/D817C8EA-4EC5-E711-882C-24BE05C3EC61.root
# 
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_'+CMSSW+'+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+5228FC24-10C5-E711-9B90-E0071B73B6C0.root
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_'+CMSSW+'+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+FE37E710-28C5-E711-98C6-4C79BA320467.root
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_'+CMSSW+'+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+2E5505C2-3DC5-E711-BA98-4C79BA180C81.root
# /eos/cms/store/user/dmytro/tmp/store+relval+CMSSW_'+CMSSW+'+RelValZMM_13+MINIAODSIM+PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1+10000+D817C8EA-4EC5-E711-882C-24BE05C3EC61.root

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
redirectors = ['xrootd-cms.infn.it','cmsxrootd.fnal.gov','cms-xrd-global.cern.ch']
redirector = redirectors[0] # 0 does not work for QCD + Zmm
print "Accessing files through redirector '%s'" % redirector
inputDatasets = './datasets/'
studies = {
    'TTbar':{
        'files':{
            '9_4_0_pre3':
                #open(inputDatasets+"RelValTTbar_13-PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50_v1-MINIAODSIM-dmytro.txt").readlines(),
                # through AAA
                open(inputDatasets+"RelValTTbar_13-PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50_v1-MINIAODSIM.txt").readlines(),
            '10_2_2':[# on disk 
                # TTbar
                # cannot use FastSim because muon.simType()==ROOT.reco.MatchedPrimaryMuon would not work
                #'root://'+redirector+'//store/relval/CMSSW_'+CMSSW+'/RelValTTbar_13/MINIAODSIM/PU25ns_102X_mcRun2_asymptotic_v3_FastSim-v2/10000/5858CF63-1B9E-E811-B34F-0025905A610C.root',
                #'root://'+redirector+'//store/relval/CMSSW_'+CMSSW+'/RelValTTbar_13/MINIAODSIM/PU25ns_102X_mcRun2_asymptotic_v3_FastSim-v2/10000/E666A4CB-A19D-E811-94ED-0025905A60DE.root'
                ],
            '10_3_0_pre4':[
                'root://'+redirector+'//store/relval/CMSSW_'+CMSSW+'/RelValTTbar_13/MINIAODSIM/PU25ns_103X_mcRun2_asymptotic_v1-v1/10000/6B898672-0B4B-184C-95DA-153402FCBAE9.root'
                ]
            },
        'maxBkgEff':0.15,
        'name':'ttbar'
        },
    'QCD + Zmm':{
        'files':{
            '9_4_0_pre3':
                # ZMM
                #open(inputDatasets+"RelValZMM_13-PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1-MINIAODSIM-dmytro.txt").readlines(),
                # through AAA
                open(inputDatasets+"RelValZMM_13-PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1-MINIAODSIM.txt").readlines(),
                # QCD
                #open(inputDatasets+"RelValQCD_13-PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1-MINIAODSIM-dmytro.txt").readlines(),
                # through AAA
                open(inputDatasets+"RelValQCD_13-PU25ns_94X_mc2017_realistic_PixFailScenario_IDEAL_HS_AVE50-v1-MINIAODSIM.txt").readlines(),
                # cannot use FastSim because muon.simType()==ROOT.reco.MatchedPrimaryMuon would not work 
                #['root://'+redirector+'//store/relval/CMSSW_9_4_0/'+RelValQCD+'/MINIAODSIM/PU25ns_94X_mcRun2_asymptotic_v0_FastSim-v1/10000/8886E4B0-C7C8-E711-9A30-0CC47A4C8EEA.root',
                #'root://'+redirector+'//store/relval/CMSSW_9_4_0/'+RelValQCD+'/MINIAODSIM/PU25ns_94X_mcRun2_asymptotic_v0_FastSim-v1/10000/90C0EB07-F1C7-E711-9DF2-0CC47A4C8F1C.root',
                #'root://'+redirector+'//store/relval/CMSSW_9_4_0/'+RelValQCD+'/MINIAODSIM/PU25ns_94X_mcRun2_asymptotic_v0_FastSim-v1/10000/AA79BD20-4CC8-E711-91F7-0CC47A4C8E5E.root',],
            '10_2_2':[# on disk 
                # ZMM
                # cannot use FastSim because muon.simType()==ROOT.reco.MatchedPrimaryMuon would not work 
                #'root://'+redirector+'//store/relval/CMSSW_'+CMSSW+'/RelValZMM_13/MINIAODSIM/PU25ns_102X_mcRun2_asymptotic_v3_FastSim-v2/10000/0E846746-009E-E811-A436-0025905B8580.root',
                # QCD
                # cannot use FastSim because muon.simType()==ROOT.reco.MatchedPrimaryMuon would not work
                #'root://'+redirector+'//store/relval/CMSSW_'+CMSSW+'/'+RelValQCD+'/MINIAODSIM/PU25ns_102X_mcRun2_asymptotic_v3_FastSim-v2/10000/0EAC5300-9C9D-E811-BE5C-0025905B8580.root',
                #'root://'+redirector+'//store/relval/CMSSW_'+CMSSW+'/'+RelValQCD+'/MINIAODSIM/PU25ns_102X_mcRun2_asymptotic_v3_FastSim-v2/10000/605937C8-869E-E811-A110-0025905A612C.root',
                #'root://'+redirector+'//store/relval/CMSSW_'+CMSSW+'/'+RelValQCD+'/MINIAODSIM/PU25ns_102X_mcRun2_asymptotic_v3_FastSim-v2/10000/FA0C4EAA-689E-E811-A3EA-0025905B85B2.root'
                ],
            '10_2_0':[# on disk 
                # QCD
                # cannot use FastSim because muon.simType()==ROOT.reco.MatchedPrimaryMuon would not work
                #'root://'+redirector+'//store/relval/CMSSW_'+CMSSW+'/'+RelValQCD+'/MINIAODSIM/PU25ns_102X_mcRun2_asymptotic_v3_gcc7_FastSim-v1/10000/A055C028-C08A-E811-A564-0025905B8576.root',
                #'root://'+redirector+'//store/relval/CMSSW_'+CMSSW+'/'+RelValQCD+'/MINIAODSIM/PU25ns_102X_mcRun2_asymptotic_v3_gcc7_FastSim-v1/10000/D0721909-4D8B-E811-8811-0CC47A78A3E8.root',
                #'root://'+redirector+'//store/relval/CMSSW_'+CMSSW+'/'+RelValQCD+'/MINIAODSIM/PU25ns_102X_mcRun2_asymptotic_v3_gcc7_FastSim-v1/10000/8A7EF76C-F689-E811-8B3B-0CC47A7C3424.root'
                ],
            '10_3_0_pre4':[
                # QCD
                'root://'+redirector+'//store/relval/CMSSW_'+CMSSW+'/'+RelValQCD+'/MINIAODSIM/PU25ns_103X_mcRun2_asymptotic_v1-v1/10000/528B2B26-F3B2-8249-A2DE-541D3FEB03F3.root',
                'root://'+redirector+'//store/relval/CMSSW_'+CMSSW+'/'+RelValQCD+'/MINIAODSIM/PU25ns_103X_mcRun2_asymptotic_v1-v1/10000/9860357D-F043-B24E-BD2E-A5B0BEF477AE.root',
                ],
            },
        'maxBkgEff':0.015,
        'name':'qcd_zmm'
        },
    'J/#psi -> mu mu':{
        'files':{
            '10_2_5':open(inputDatasets+"JpsiToMuMu_JpsiPt8_13TeV-RunIIAutumn18DR_PUAvg50ForMUOVal_102X_upgrade2018-AODSIM.txt").readlines()
            },
        'maxBkgEff':0.15,
        'name':'JPsiToMuMu'
        }
    }


muonHandle, muonLabel = Handle("std::vector<pat::Muon>"), "slimmedMuons"

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
        if muon.mvaValue() > mva_threshold:
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
    canvas.Print("%s/%s.png" % (path,output_name_without_extention))
    canvas.Print("%s/%s.pdf" % (path,output_name_without_extention))
    canvas.Print("%s/%s.root"% (path,output_name_without_extention))

# https://github.com/cms-sw/cmssw/blob/387393ddf3bc9ff50c532bb1dec288f180e64796/DataFormats/MuonReco/interface/MuonSimInfo.h#L31
muonSimTypes = {'MatchedPrimaryMuon': 4, 'MatchedMuonFromHeavyFlavour': 3}
extendedMuonSimType = {'MatchedMuonFromB': 8}
muonSimTypes.update(extendedMuonSimType)

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
        'display':True,
        'marker':24,
        'color':ROOT.kBlack
        },
    'MVALoose ID':{
        'mask':ROOT.reco.Muon.MvaLoose,
        'display':True,
        'marker':20,
        'color':ROOT.kBlack
        },
    'SoftMVA ID':{
        'mask':ROOT.reco.Muon.SoftMvaId,
        'display':True,
        'marker':20,
        'color':ROOT.kBlack
        },
    'MVATight ID':{
        'mask':ROOT.reco.Muon.MvaTight,
        'display':True,
        'marker':20,
        'color':ROOT.kBlack
        },
    'Tight ID + Loose PFIso':{
        'mask':ROOT.reco.Muon.CutBasedIdTight|ROOT.reco.Muon.PFIsoLoose,
        'display':True,
        'marker':22,
        'color':ROOT.kBlack
        },
    'Tight ID + Tight PFIso':{
        'mask':ROOT.reco.Muon.CutBasedIdTight|ROOT.reco.Muon.PFIsoTight,
        'display':True,
        'marker':22,
        'color':ROOT.kBlack
        },
    'MediumPrompt ID + Loose TkIso':{
        'mask':ROOT.reco.Muon.CutBasedIdMediumPrompt|ROOT.reco.Muon.TkIsoLoose,
        'display':True,
        'marker':23,
        'color':ROOT.kBlack
        },
    'MediumPrompt ID + Tight TkIso':{
        'mask':ROOT.reco.Muon.CutBasedIdMediumPrompt|ROOT.reco.Muon.TkIsoTight,
        'display':True,
        'marker':23,
        'color':ROOT.kBlack
        },
    }

bigNumber = 1e9
preSelection = {'BPH-16-004':{'minPt':4.0, 'maxPt':bigNumber, 'minEta':-1.4, 'maxEta':1.4},
                'BPH-18-002':{'minPt':2.5, 'maxPt':bigNumber, 'minEta':-2.4, 'maxEta':2.4},
                'BPH-15-005':{'minPt':3.5, 'maxPt':bigNumber, 'minEta':-2.4, 'maxEta':2.4},
               }

n_events_limit = None
#n_events_limit = 10000*2
n_events_limit = 9000

for study,info in studies.items():
    #if not 'bar' in study: continue
    dataset = CMSSW + ' ' + study
    print "\nProcessing %s" % dataset

    maxBkgEff = info['maxBkgEff']
    if CMSSW in info['files']:
        files = info['files'][CMSSW]
    else:
        print "No %s in %s dictionary, skipping it" % (CMSSW, study)
        continue
    print "\nNumber of input files: %d" % len(files)
    if not len(files):
        print "No input files provided for %s" % dataset
        continue

    events = Events(files)
    totEvents = events.size()
    print "Total number of events: %d" % totEvents
    maxEvents = totEvents
    if n_events_limit and n_events_limit<totEvents:
        maxEvents = n_events_limit
    print "\nWill process %d events" % maxEvents


    for analysis,cuts in preSelection.items():
        label = analysis + ' from ' + CMSSW + ' ' + study

        print "\nProcessing %s with %s pre-selection:" % (dataset, analysis)
        print "%.2f\t< pT(muon) [GeV]\t< %.2f" % (cuts['minPt'],  cuts['maxPt'])
        print "%.2f\t< eta(muon)\t\t< %.2f"    % (cuts['minEta'], cuts['maxEta'])

        nevents = 0
        nSigTotal = {}
        nSigSelected = {}
        nBkgTotal = {}
        nBkgSelected = {}
        ROC = {}
    
        for muonSimType in muonSimTypes:
            # Define ROC curves
            ROC[muonSimType] = {}
            # arrays of mva values for the ROC curves
            # np.arange retursn evenly spaced values within a given interval ([start, ]stop, [step, ]dtype=None)
            ROC[muonSimType]['LeptonMVA'] = {'values':np.arange(-1.0, 1.0, 0.05)}
            ROC[muonSimType]['SoftMVA'] = {'values':np.arange(-1.0, 1.0, 0.05)}
            # TODO: how do we choose such values?
            ROC[muonSimType]['TightID + PFIso'] = {'values':np.arange(0.00, 0.40, 0.01)}
            ROC[muonSimType]['MediumPromptID + TkIso'] = {'values':np.arange(0.00, 0.40, 0.01)}
            for iROC in ROC[muonSimType].values():
                iROC['effSig'] = np.zeros(len(iROC['values']), dtype=float)
                iROC['effBkg'] = np.zeros(len(iROC['values']), dtype=float)
    
            # Define graphs
            nSigTotal[muonSimType] = 0
            nBkgTotal[muonSimType] = 0
            nSigSelected[muonSimType] = {}
            nBkgSelected[muonSimType] = {}
            for selector in selectors:
                nSigSelected[muonSimType][selector] = 0
                nBkgSelected[muonSimType][selector] = 0
    
    
        # Loop over events
        events.toBegin()
        for event in events:
            if nevents >= maxEvents: break
            if (nevents+1) % max(1,maxEvents/10) == 0: print "Processing event", nevents+1
    
            event.getByLabel(muonLabel, muonHandle)
            muons = muonHandle.product()
            # Loop over muons
            for muon in muons:
    
                # Pre-selection
                if muon.pt()  < cuts['minPt']  or muon.pt()  > cuts['maxPt'] : continue
                if muon.eta() < cuts['minEta'] or muon.eta() > cuts['maxEta']: continue
    
                for muonSimType in muonSimTypes:
                    simType = muon.simType() if muonSimTypes[muonSimType] < 5 else muon.simExtType()
                    # signal or background muons
                    trueMuon = (simType == muonSimTypes[muonSimType])
                    if trueMuon:
                        nSigTotal[muonSimType] += 1
                    else:
                        nBkgTotal[muonSimType] += 1
    
                    # Evaluate selectors
                    for name, selector in selectors.items():
                        passed = muon.passed(selector['mask'])
                        if passed:
                            if trueMuon:
                                nSigSelected[muonSimType][name] += 1
                            else:
                                nBkgSelected[muonSimType][name] += 1
    
                    # Evaluate ROCs
                    for name,iROC in ROC[muonSimType].items():
                        for i in range(len(iROC['values'])):
                            passed = False
                            if name == 'LeptonMVA':
                                passed = LeptonMVA(muon,iROC['values'][i])
                            elif name == 'SoftMVA':
                                passed = muon.softMvaValue() > iROC['values'][i]
                            elif name == 'TightID + PFIso':
                                passed = muon.passed(ROOT.reco.Muon.CutBasedIdTight) and (pfIsolation(muon) < iROC['values'][i])
                            elif name == 'MediumPromptID + TkIso':
                                passed = muon.passed(ROOT.reco.Muon.CutBasedIdMediumPrompt) and (tkIsolation(muon) < iROC['values'][i])
    
                            if passed:
                                if trueMuon:
                                    iROC['effSig'][i] += 1
                                else:
                                    iROC['effBkg'][i] += 1
    
            nevents += 1
    
        print "Processed %d events" % nevents
    
        allEffBkg = np.empty(0)
        for muonSimType in muonSimTypes:
            print "N signal (%s) muons: %d" % (muonSimType, nSigTotal[muonSimType])
            print "N background muons: %d" % (nBkgTotal[muonSimType])
    
            if not nSigTotal[muonSimType]:
                print "WARNING: No signal (%s) muons (nSigTotal == %d) for %s" % (muonSimType, nSigTotal[muonSimType], label)
                continue
            if not nBkgTotal[muonSimType]:
                print "WARNING: No background muons (nBkgTotal == %d) for %s" % (nBkgTotal[muonSimType], label)
                continue
    
            # Normalization
            for name, iROC in ROC[muonSimType].items():
                iROC['effSig'] /= nSigTotal[muonSimType]
                iROC['effBkg'] /= nBkgTotal[muonSimType]
    
                allEffBkg = np.append(allEffBkg, np.amax(iROC['effBkg']) )
    
        effBkgMax = np.amax(allEffBkg)
        print "\nDrawing graphs for %s" % label
    
        c1 = ROOT.TCanvas("c1", "ROC curve", 700,700)
        c1.SetLeftMargin(0.15)
    
        colorOffset = ROOT.kBlack
    
        for muonSimType in muonSimTypes:
            c1.Clear()
            graphs = []
    
            # ROC graphs
            nGraph = 0
            for name,iROC in ROC[muonSimType].items():
                nGraph += 1
                color = colorOffset + nGraph
                # apparently yellow is not kYellow but kYellow/100 +1
                if color >= (ROOT.kYellow/100 +1): color += 1
                graph = ROOT.TGraph(len(iROC['values']), iROC['effBkg'], iROC['effSig'])
                graph.SetTitle(name) # see definition of 'passed'
                graph.SetLineColor(color)
                graph.SetLineWidth(3)
                if nGraph == 1:
                    graph.SetMinimum(0)
                    graph.SetMaximum(1)
                    graph.Draw("AC")
                    graph.GetXaxis().SetTitle("Background efficiency")
                    graph.GetYaxis().SetTitle("Signal ("+muonSimType+") efficiency")
                    graph.GetXaxis().SetLimits(0, max(maxBkgEff, 0.02 + effBkgMax));
                else:
                    graph.Draw("C same")
                graphs.append(graph)
    
            # Single point graphs
            for selector in sorted(selectors):
                effS = float(nSigSelected[muonSimType][selector]) / nSigTotal[muonSimType]
                effB = float(nBkgSelected[muonSimType][selector]) / nBkgTotal[muonSimType]
                if selectors[selector]['display']:
                    effSigArray = array("f",[effS])
                    effBkgArray = array("f",[effB])
                    graph = ROOT.TGraph(len(effSigArray), effBkgArray, effSigArray)
                    graph.SetTitle(selector)
                    graph.SetMarkerStyle(selectors[selector]['marker'])
                    graph.SetMarkerColor(selectors[selector]['color'])
                    graph.SetMarkerSize(1.5)
                    graph.Draw("P same")
                    graphs.append(graph)
    
                print "\t\n%s for %s" % (selector, muonSimType)
                print "\t\tSig: N:%d (%0.2f%%)" % (nSigSelected[muonSimType][selector],100.*effS)
                print "\t\tBkg: N:%d (%0.2f%%)" % (nBkgSelected[muonSimType][selector],100.*effB)
    
            c1.Update()
    
            legY1 = 0.25
            legLine = 0.03
            c1.BuildLegend(0.5, legY1, 0.95, legY1 + legLine*len(graphs))
    
            #c1.SetTitle(study) # gets overridden by TGraph title
            # still does not work
            ##gStyle->SetOptTitle(0);
            #style = ROOT.TStyle()
            #style.SetOptTitle(0);
            #style.UseCurrentStyle()
            #c1.Update()
    
            canvasTitle = ROOT.TPaveLabel(0.1, 0.92, 0.9, 0.99, label, "NDC")
            canvasTitle.SetFillColor(ROOT.kWhite)
            canvasTitle.SetBorderSize(1)
            canvasTitle.SetLineColor(0)
            canvasTitle.Draw();
    
            # path = "/eos/user/d/dmytro/www/plots/"+CMSSW+"_ROCs/"
            path = "plots/ROCs_"+CMSSW
            print_canvas(c1, info['name']+'_'+analysis+'_'+muonSimType, path)
            # raw_input( ' ... ' )
    
        print "Finish processing %s\n" % label

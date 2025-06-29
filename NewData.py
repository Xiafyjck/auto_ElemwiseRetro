import json
import re
import math
import numpy as np
import operator
import requests
import time
from tqdm import tqdm
from pymatgen.core import Composition

import torch
import matplotlib.pyplot as plt

from collections import defaultdict

import os
import pandas as pd

elem_library            = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al',
                           'Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe',
                           'Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr',
                           'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
                           'Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm',
                           'Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W',
                           'Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
                           'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf',
                           'Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds',
                           'Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

alkali_metal            = ['Li','Na','K','Rb','Cs']
alkaline_earth_metal    = ['Be','Mg','Ca','Sr','Ba']
transition_metal        = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
                           'Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd','Hf',
                           'Ta','W','Re','Os','Ir','Pt','Au','Hg']
lanthanide_elem         = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
actinide_elem           = ['Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']
post_transition_metal   = ['Al','Ga','In','Sn','Tl','Pb','Bi']
metalloid               = ['B','Si','Ge','As','Sb','Te']
non_metal               = ['H','C','N','O','F','P','S','Cl','Se','Br','I']
noble_gas               = ['He','Ne','Ar','Kr','Xe']
artificial_elem         = ['Tc','Pm','Po','At','Rn','Fr','Ra','Rf','Db','Sg','Bh',
                           'Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

essen_elem = alkali_metal + alkaline_earth_metal + transition_metal \
             + lanthanide_elem + actinide_elem + post_transition_metal + metalloid + ['P','Se','S']

inorg_elem = alkali_metal + alkaline_earth_metal + transition_metal \
             + lanthanide_elem + actinide_elem + post_transition_metal + metalloid

def simple_convert(precursor_ids_path: str, reactions_path: str)-> dict:
    reaction_df = pd.read_csv(reactions_path)
    global_precursor_ids = json.load(open(precursor_ids_path, "r"))

    for id, precursors in global_precursor_ids.items():
        assert len(precursors) == 1
        precursor = precursors[0]
        pattern = '[^()1-9a-zA-Z]?[1-9]?H2O$'
        if re.search(pattern, precursor) != None:
            print(f"precursor: {precursor} 将去除水合部分, ", end="")
            precursor = re.sub(pattern, '', precursor)
            print(precursor)
            global_precursor_ids[id] = [precursor]

        precursor_source_elem, _ = get_SourceElem(precursor)
        if len(precursor_source_elem) == 0:
            print(f"ignore non_source precursor: {precursor}")
            global_precursor_ids[id] = []
        if len(precursor_source_elem) > 1:
            print(f"reactions involving {precursor} with multi source elements will be deleted")
            global_precursor_ids[id] = ["multiSourceElem"]


    reactions = []
    for _, row in reaction_df.iterrows():
        is1to1 = True
        reaction = {}
        reaction["Target"] = row["target"]

        precursor_ids = row["precursor_ids"].split(",")
        reaction["Precursors"] = []

        target_source_elem, _ = get_SourceElem(reaction["Target"])
        precursors_source_elem = []
        for precursor_id in precursor_ids:
            if len(global_precursor_ids[precursor_id]) == 0:
                continue
            precursor = global_precursor_ids[precursor_id][0]

            if precursor != "multiSourceElem":
                p_source_elem, _ = get_SourceElem(precursor)
                # 检查是否有重复的源元素
                for elem in p_source_elem:
                    if elem in precursors_source_elem:
                        is1to1 = False
                        break
                precursors_source_elem += p_source_elem
            reaction["Precursors"].append(global_precursor_ids[precursor_id][0])

        if "multiSourceElem" in reaction["Precursors"]:
            continue
        if is1to1 and set(target_source_elem) == set(precursors_source_elem):
            pass
        else:
            if not is1to1:
                print(f"repeted elements")
            print(f"{reaction['Target']} <- {reaction['Precursors']} reaction source elements are not 1 to 1, and will be deleted")
            continue

        reaction["doi"] = row["doi"]
        reaction["year"] = row["year"]
        reaction["split"] = row["type"]

        reactions.append(reaction)

    print(f"original length: {len(reaction_df)}, filtered length: {len(reactions)}")

    return reactions

def get_SourceElem(str_composition: str):
    source_elem = []
    env_elem = []

    comp = Composition(str_composition).get_el_amt_dict()
    elements = list(comp.keys())
    for e in elements:
        if e in essen_elem:
            source_elem.append(e)
        else:
            env_elem.append(e)

    source_elem = list(set(source_elem))
    env_elem = list(set(env_elem))

    return source_elem, env_elem

def filter_by_CommerciallyViable(reactions:list, CAS_collection:set)->list:
    for i, reaction in enumerate(reactions):
        for j, precursor in enumerate(reaction["Precursors"]):
            if precursor in ['LiCO3','Li3CO3']:
                reactions[i]["Precursors"][j] = 'Li2CO3'
            elif precursor in ['Li(OH)','Li7OH']:
                reactions[i]["Precursors"][j] = 'LiOH' 
            elif precursor in ['LiCH3COO']:
                reactions[i]["Precursors"][j] = 'CH3COOLi'
            elif precursor in ['B2O5']:
                reactions[i]["Precursors"][j] = 'B2O3'
            elif precursor in ['B(OH)3','BO3H3']:
                reactions[i]["Precursors"][j] = 'H3BO3'
            elif precursor in ['NaCO3']:
                reactions[i]["Precursors"][j] = 'Na2CO3'
            elif precursor in ['MgO2']:
                reactions[i]["Precursors"][j] = 'MgO'
            elif precursor in ['(MgCO3)4.Mg(OH)2','(MgCO3)4Mg(OH)2']:
                reactions[i]["Precursors"][j] = 'Mg5(CO3)4(OH)2'
            elif precursor in ['((Al2O3))','Al2O2']:
                reactions[i]["Precursors"][j] = 'Al2O3'
            elif precursor in ['Al(OH)2']:
                reactions[i]["Precursors"][j] = 'Al(OH)3'
            elif precursor in ['AlOOH']:
                reactions[i]["Precursors"][j] = 'AlO(OH)'
            elif precursor in ['C8H20O4Si']:
                reactions[i]["Precursors"][j] = 'Si(OC2H5)4'
            elif precursor in ['(NH4)H2PO4','H2NH4PO4','NH4(H2PO4)','(NH4)(H2PO4)','NH4(H2)PO4']:
                reactions[i]["Precursors"][j] = 'NH4H2PO4'
            elif precursor in ['(NH4)2.HPO4']:
                reactions[i]["Precursors"][j] = '(NH4)2HPO4'
            elif precursor in ['P4P4']:
                reactions[i]["Precursors"][j] = 'P'
            elif precursor in ['(NH4)2(HSO4)2']:
                reactions[i]["Precursors"][j] = '(NH4)(HSO4)'
            elif precursor in ['K2O','K2O3']:
                reactions[i]["Precursors"][j] = 'KO2'
            elif precursor in ['Ca(CO3)2','Ca2CO3']:
                reactions[i]["Precursors"][j] = 'CaCO3'
            elif precursor in ['Ca(NO3)']:
                reactions[i]["Precursors"][j] = 'Ca(NO3)2'
            elif precursor in ['Ti(C4H9O)4','C16H36O4Ti','Ti(OCH2CH2CH2CH3)4']:
                reactions[i]["Precursors"][j] = 'Ti(OC4H9)4'
            elif precursor in ['C12H28O4Ti']:
                reactions[i]["Precursors"][j] = 'Ti(OCH(CH3)2)4'
            elif precursor in ['Cr(NO3)6','Cr(NO3)2']:
                reactions[i]["Precursors"][j] = 'Cr(NO3)3'
            elif precursor in ['Mn2(CO3)7','Mn2CO3']:
                reactions[i]["Precursors"][j] = 'MnCO3'
            elif precursor in ['Mn(NO3)7']:
                reactions[i]["Precursors"][j] = 'Mn(NO3)2'
            elif precursor in ['(CH3COO)2Mn','Mn(COOCH3)2']:
                reactions[i]["Precursors"][j] = 'Mn(CH3COO)2'
            elif precursor in ['MnO(OH)']:
                reactions[i]["Precursors"][j] = 'MnOOH'
            elif precursor in ['Co(CO3)','Co2(CO3)3']:
                reactions[i]["Precursors"][j] = 'CoCO3'
            elif precursor in ['Co(OH)3']:
                reactions[i]["Precursors"][j] = 'Co(OH)2'
            elif precursor in ['Co(NO3)3']:
                reactions[i]["Precursors"][j] = 'Co(NO3)2'
            elif precursor in ['Co(CH3COO)3','(CH3COO)2Co']:
                reactions[i]["Precursors"][j] = 'Co(CH3COO)2'
            elif precursor in ['Co2']:
                reactions[i]["Precursors"][j] = 'Co'
            elif precursor in ['Co2(C2O4)3']:
                reactions[i]["Precursors"][j] = 'CoC2O4'
            elif precursor in ['Ni2(CO3)3']:
                reactions[i]["Precursors"][j] = 'NiCO3'
            elif precursor in ['Ni(OH)3']:
                reactions[i]["Precursors"][j] = 'Ni(OH)2'
            elif precursor in ['Ni(NO3)3']:
                reactions[i]["Precursors"][j] = 'Ni(NO3)2'
            elif precursor in ['Ni(CH3COO)3']:
                reactions[i]["Precursors"][j] = 'Ni(CH3COO)2'
            elif precursor in ['Cu(NO3)3']:
                reactions[i]["Precursors"][j] = 'Cu(NO3)2'
            elif precursor in ['GaO2']:
                reactions[i]["Precursors"][j] = 'Ga2O3'
            elif precursor in ['Ge2O3']:
                reactions[i]["Precursors"][j] = 'GeO2'
            elif precursor in ['Sr2CO3','Sr(CO3)2','Sr(CO3)','Sr3CO3']:
                reactions[i]["Precursors"][j] = 'SrCO3'
            elif precursor in ['Sr(NO3)3']:
                reactions[i]["Precursors"][j] = 'Sr(NO3)2'
            elif precursor in ['Sr(C2H3O2)2']:
                reactions[i]["Precursors"][j] = 'Sr(CH3COO)2'
            elif precursor in ['YO1.5','Y3O3']:
                reactions[i]["Precursors"][j] = 'Y2O3'
            elif precursor in ['Y(NO3)']:
                reactions[i]["Precursors"][j] = 'Y(NO3)3'
            elif precursor in ['ZrO3','ZrO','Zr2O']:
                reactions[i]["Precursors"][j] = 'ZrO2'
            elif precursor in ['Nb2O3','Nb2O9','Nb2O6']:
                reactions[i]["Precursors"][j] = 'Nb2O5'
            elif precursor in ['RhO2']:
                reactions[i]["Precursors"][j] = 'Rh2O3'
            elif precursor in ['Ag2NO3','Ag(NO3)2']:
                reactions[i]["Precursors"][j] = 'AgNO3'
            elif precursor in ['TeO3','Te2O']:
                reactions[i]["Precursors"][j] = 'TeO2'
            elif precursor in ['H6TeO6']:
                reactions[i]["Precursors"][j] = 'Te(OH)6'
            elif precursor in ['Ba2CO3','Ba3CO3']:
                reactions[i]["Precursors"][j] = 'BaCO3'
            elif precursor in ['Ba(NO3)']:
                reactions[i]["Precursors"][j] = 'Ba(NO3)2'
            elif precursor in ['Ba(C2H3O2)2']:
                reactions[i]["Precursors"][j] = 'Ba(CH3COO)2'
            elif precursor in ['La2CO3','LaCO3']:
                reactions[i]["Precursors"][j] = 'La2(CO3)3'
            elif precursor in ['LaO3','LaO1.5','La2O5']:
                reactions[i]["Precursors"][j] = 'La2O3'
            elif precursor in ['La(NO3)2']:
                reactions[i]["Precursors"][j] = 'La(NO3)3'
            elif precursor in ['CeO','Ce2O3']:
                reactions[i]["Precursors"][j] = 'CeO2'
            elif precursor in ['Ce(NO3)4','Ce(NO3)']:
                reactions[i]["Precursors"][j] = 'Ce(NO3)3'
            elif precursor in ['Nd2O5']:
                reactions[i]["Precursors"][j] = 'Nd2O3'
            elif precursor in ['Sm2O']:
                reactions[i]["Precursors"][j] = 'Sm2O3'
            elif precursor in ['Sm(NO3)2']:
                reactions[i]["Precursors"][j] = 'Sm(NO3)3'
            elif precursor in ['EuO']:
                reactions[i]["Precursors"][j] = 'Eu2O3'
            elif precursor in ['DyO1.5']:
                reactions[i]["Precursors"][j] = 'Dy2O3'
            elif precursor in ['HoO']:
                reactions[i]["Precursors"][j] = 'Ho2O3'
            elif precursor in ['Ta2O9','Ta2O3']:
                reactions[i]["Precursors"][j] = 'Ta2O5'
            elif precursor in ['(NH4)10H2W12O42']:
                reactions[i]["Precursors"][j] = '(NH4)10H2(W2O7)6'
            elif precursor in ['IrO6']:
                reactions[i]["Precursors"][j] = 'IrO2'
            elif precursor in ['BiO1.5','Bi3O','BiO3','Bi2O5']:
                reactions[i]["Precursors"][j] = 'Bi2O3'
            elif precursor in ['(Pu0.93)O2']:
                reactions[i]["Precursors"][j] = 'PuO2' 

    filtered_reactions = []
    for reaction in reactions:
        flag = True
        for precursor in reaction["Precursors"]:
            if precursor not in CAS_collection:
                flag = False
                break
        if flag:
            filtered_reactions.append(reaction)
    print(f"CAS_collection length: {len(CAS_collection)}")
    print(f"original length: {len(reactions)}, filtered length by CAS_collection: {len(filtered_reactions)}")
    return filtered_reactions

def data_preprocess(reactions:list)->list:
    with open("data_processing/raw/CAS_collection.json", "r") as f:
        CAS_collection = json.load(f)
    filtered_reactions = filter_by_CommerciallyViable(reactions, CAS_collection)
    return filtered_reactions

def get_anion_part(formula, source_elem):
    comp = Composition(formula).get_el_amt_dict()
    ca_count = 0
    an_count = 0
    anion = ""


    for elem, stoi in comp.items():
        if str(elem) in source_elem:
            ca_count += 1
        else:
            an_count += 1
            anion += str(elem) + str(stoi)

    if anion != "":
        # print(formula, source_elem, comp, anion, ca_count, an_count)
        anion = str(Composition(anion).get_integer_formula_and_factor()[0])
    else:
        # 可能是单质
        pass

    return anion

def find_precursors_total_anion_framework(reactions):
    pre_total_anions = defaultdict(int)
    for reaction in reactions:
        for precursor in reaction["Precursors"]:
            precursor_source_elem, _ = get_SourceElem(precursor)
            precursor_anion = get_anion_part(precursor, precursor_source_elem)
            pre_total_anions[precursor_anion] += 1

    return pre_total_anions

def stoi_count_dict_from_source_template(reactions, pre_total_anions):
    stoi_ll_dict = {}
    for s_elem in elem_library:
        for template in pre_total_anions:
            stoi_ll_dict[s_elem+template] = {}
    for i, reaction in enumerate(reactions):
        for j, precursor in enumerate(reaction["Precursors"]):
            p_source_elem, _ = get_SourceElem(precursor)
            template = get_anion_part(precursor, p_source_elem)
            if len(p_source_elem) == 1:
                if precursor not in stoi_ll_dict[p_source_elem[0]+template]:
                    stoi_ll_dict[p_source_elem[0]+template][precursor] = 1
                else:
                    stoi_ll_dict[p_source_elem[0]+template][precursor] += 1
            else:
                raise NotImplementedError("No single source element precursors")
    return stoi_ll_dict

def most_frequent_stoi_dict_from_source_template(reactions, pre_total_anions):
    stoi_ll_dict = {}
    for s_elem in elem_library:
        for template in pre_total_anions:
            stoi_ll_dict[s_elem+template] = {}
    for i, reaction in enumerate(reactions):
        for j, precursor in enumerate(reaction['Precursors']):
            p_source_elem, _ = get_SourceElem(precursor)
            template = get_anion_part(precursor, p_source_elem)
            if len(p_source_elem) == 1:
                if precursor not in stoi_ll_dict[p_source_elem[0]+template]:
                    stoi_ll_dict[p_source_elem[0]+template][precursor] = 1
                else:
                    stoi_ll_dict[p_source_elem[0]+template][precursor] += 1
            else:
                raise NotImplementedError("No single source element precursors")

    most_count = 0
    minor_count = 0
    stoi_dict = {}
    for s_elem in elem_library:
        for template in pre_total_anions:
            stoi_dict[s_elem+template] = []
            if len(stoi_ll_dict[s_elem+template]) != 0:
                di = stoi_ll_dict[s_elem+template]
                most_frequent = [k for k,v in di.items() if max(di.values()) == v]
                if len(most_frequent) == 1:
                    stoi_dict[s_elem+template] = [most_frequent[0]]
                else:
                    #print(stoi_ll_dict[s_elem+template])
                    stoi_dict[s_elem+template] = [most_frequent[0]]
                for k,v in di.items():
                    minor_count += v
                most_count += max(di.values())
                minor_count -= max(di.values())

    return stoi_dict, most_count, minor_count

def main(dataset_name=None):
    if dataset_name is None:
        import argparse
        parser = argparse.ArgumentParser(description='Process dataset for ElemwiseRetro')
        parser.add_argument('--dataset_name', type=str, default='retro', help='Dataset name')
        args = parser.parse_args()
        dataset_name = args.dataset_name
    
    precursor_ids_path = f"data_processing/raw/{dataset_name}_precursor_id.json"
    reactions_path = f"data_processing/raw/{dataset_name}_split.csv"

    reactions = simple_convert(precursor_ids_path, reactions_path)
    # with open(f"data_processing/raw/CAS_collection.json", "r") as f:
    #     CAS_collection = json.load(f)
    # reactions = filter_by_CommerciallyViable(reactions, CAS_collection)

    for i, reaction in enumerate(reactions):
        reactions[i]["Target"] = [reactions[i]["Target"]]

    pre_total_anions = find_precursors_total_anion_framework(reactions)
    stoi_ll_dict = stoi_count_dict_from_source_template(reactions, pre_total_anions)
    stoi_dict, _, _ = most_frequent_stoi_dict_from_source_template(reactions, pre_total_anions)
    
    os.makedirs(f"./ElemwiseRetro/dataset/{dataset_name}", exist_ok=True)

    file_path = f"./ElemwiseRetro/dataset/{dataset_name}/pre_anion_part.json"
    with open(file_path, 'w') as outfile:
        json.dump(pre_total_anions, outfile, indent=4)

    file_path = f"./ElemwiseRetro/dataset/{dataset_name}/stoi_ll_dict.json"
    with open(file_path, 'w') as outfile:
        json.dump(stoi_ll_dict, outfile, indent=4)

    file_path = f"./ElemwiseRetro/dataset/{dataset_name}/stoi_dict.json"
    with open(file_path, 'w') as outfile:
        json.dump(stoi_dict, outfile, indent=4)

    file_path = f"./ElemwiseRetro/dataset/{dataset_name}/InorgSyn_dataset_TP.json"
    with open(file_path, 'w') as outfile:
        json.dump(reactions, outfile, indent=4)

if __name__ == "__main__":
    main()
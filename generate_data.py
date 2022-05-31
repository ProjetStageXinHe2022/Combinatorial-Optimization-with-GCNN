import ecole
import shutil
import os
import sys
import time
import numpy as np
import pyscipopt as pyscip
import pickle as pkl
from tqdm import trange

from dump_data import *

def generate_dataset(scip_parameters,path = "DataSet/",nb_cons = [100,200,300,400,500],nb_var = [1,1.5,2],density = [0.1,0.15,0.2]):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    env = ecole.environment.Branching(\
                observation_function = ecole.observation.NodeBipartite(),scip_params = scip_parameters)
    gapList = []
    for row in nb_cons:
        for coef_col in nb_var: 
            for d in density:
                col = int(coef_col * row)
                setCover = ecole.instance.SetCoverGenerator(n_rows = row, n_cols = col,density = d) 
                print("Generate with Row:%d,Col:%d,Density:%f" % (row,col,d))
                for n in trange(1,101):
                    done = False
                    while(not done):
                        try:
                            problme_name = \
                                "set_cover_{"+row.__str__()+"*"+col.__str__()+"_"+d.__str__()+"_"+n.__str__()+"}"
                            instance = next(setCover)
                            #save problm lp
                            problme_name = \
                                "set_cover_{"+row.__str__()+"*"+col.__str__()+"_"+d.__str__()+"_"+n.__str__()+"}"
                            os.mkdir(path+problme_name)
                            instance.write_problem(path+problme_name+"/problem.lp")
                            #save features
                            obs, _, _, _, _ = env.reset(instance)
                            if obs.row_features.shape[0] != row:
                                raise Exception
                            #save constraintes features
                            dumpRowFeatures(path+problme_name+"/constraints_features.json",obs.row_features)
                            #save variables features
                            dumpVariableFeatures(path+problme_name+"/variables_features.json",obs.variable_features)
                            #save edges features
                            original_indice = obs.variable_features[:,0]
                            dumpEdgeFeatures(path+problme_name+"/edges_features.json",obs.edge_features,original_indice)
                            #get et save label
                            solver = ecole.scip.Model.from_file(path+problme_name+"/problem.lp")
                            aspyscip = solver.as_pyscipopt()
                            aspyscip.setPresolve(pyscip.SCIP_PARAMSETTING.OFF)
                            aspyscip.optimize()
                            gapList.append(aspyscip.getGap())
                            dumpSolution_Ecole(path+problme_name+"/label.json",aspyscip)
                            
                            done = True             
                        except Exception as ex:
    #                         print("Erreur:%s"%ex)
                            done = False
                            shutil.rmtree(path+problme_name)
    gap = np.array(gapList)
    return np.mean(gap)

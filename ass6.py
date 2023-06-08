import numpy as nmpy
from sklearn.cluster import KMeans
import itertools
import random
import math
import time
import copy
import sys

Discard_Set = dict()
Comp_set = dict()

def mahalanobis_dist_clstr(cl1, cl2):
    n1 = len(cl1["N"])
    n2 = len(cl2["N"])
    cent1 = cl1["SUM"] / n1
    cent2 = cl2["SUM"] / n2
    var1 = cl1["SUMSQ"] / n1 - (cent1 * cent1)
    var2 = cl2["SUMSQ"] / n2 - (cent2 * cent2)
    x1 = (cent1 - cent2) / var1
    x2 = (cent1 - cent2) / var2
    p1 = math.sqrt(nmpy.dot(x1, x1))
    p2 = math.sqrt(nmpy.dot(x2, x2))
    return min(p1, p2)
    
def adjust_resSets(RetSet):
    DiscardSet_sum = 0
    CompSet_sum = 0
    for cl in Discard_Set:
        DiscardSet_sum += len(Discard_Set[cl]["N"])
    for cl in Comp_set:
        CompSet_sum += len(Comp_set[cl]["N"])
    return DiscardSet_sum, len(Comp_set), CompSet_sum, len(RetSet)

def mahalanobis_dist_pts(pt, cluster):
    sm = cluster["SUM"]
    n = len(cluster["N"])
    smsq = cluster["SUMSQ"]
    midpt = sm / n
    variance = smsq / n - (sm / n) * (sm / n)
    z = (pt - midpt)/variance
    mahala_dist = math.sqrt(nmpy.dot(z, z))
    return mahala_dist
    
def populateDS(clid_chdata, points_index):
    for ccd in clid_chdata:
        if ccd[0] in Discard_Set:
            Discard_Set[ccd[0]]["N"].append(points_index[tuple(ccd[1])])
            Discard_Set[ccd[0]]["SUMSQ"] += ccd[1] * ccd[1]
            Discard_Set[ccd[0]]["SUM"] += ccd[1]
        else:
            Discard_Set[ccd[0]] = dict()
            Discard_Set[ccd[0]]["SUMSQ"] = ccd[1] * ccd[1]
            Discard_Set[ccd[0]]["N"] = [points_index[tuple(ccd[1])]]
            Discard_Set[ccd[0]]["SUM"] = ccd[1]

def populateCS(clid_RS, RetSet_key, points_index):
    for clno in clid_RS:
        if clno[0] not in RetSet_key:
            if clno[0] in Comp_set:
                Comp_set[clno[0]]["SUMSQ"] += clno[1] * clno[1]
                Comp_set[clno[0]]["SUM"] += clno[1]
                Comp_set[clno[0]]["N"].append(points_index[tuple(clno[1])])
            else:
                Comp_set[clno[0]] = dict()
                Comp_set[clno[0]]["SUMSQ"] = clno[1] * clno[1]
                Comp_set[clno[0]]["SUM"] = clno[1]
                Comp_set[clno[0]]["N"] = [points_index[tuple(clno[1])]]

if __name__ == "__main__":
    #inputs
    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    st = time.time()
    with open(input_file) as inpf:
        data_in_file = inpf.readlines()
    # preprocessing
    cleaned_data = list(map(lambda line: line.strip("\n").split(','), data_in_file))
    index_points = dict([(int(data[0]), tuple(map(lambda val:float(val), data[2:]))) for data in cleaned_data])
    points_index = dict(zip(list(index_points.values()), list(index_points.keys())))
    
    data = list(map(lambda val: nmpy.array(val), list(index_points.values())))
    chnk_size = int(len(data) / 5)
    random.shuffle(data)
    # step 1
    first_chunk = data[0:chnk_size]
    
    #step 2
    k_means = KMeans(n_clusters = n_cluster * 25)
    k_means = k_means.fit(first_chunk)
    
    #step 3
    pts_per_cluster = dict()
    cluster_nums = k_means.labels_
    for lbl in cluster_nums:
        pts_per_cluster[lbl] = pts_per_cluster.get(lbl, 0) + 1
    
    ind_of_pts_RS = []
    for cl_num in pts_per_cluster:
        if pts_per_cluster[cl_num] < 20:
            ind_of_pts_RS += [ind for ind, cl_no in enumerate(cluster_nums) if cl_no == cl_num]
    RetSet = []
    for RetSetInd in ind_of_pts_RS:
        RetSet.append(first_chunk[RetSetInd])
    
    sorted_ind_RS = sorted(ind_of_pts_RS)
    for RetSetInd in reversed(sorted_ind_RS):
        first_chunk.pop(RetSetInd)

    # step 4
    k_means = KMeans(n_clusters = n_cluster)
    k_means = k_means.fit(first_chunk)
    cluster_nums  = k_means.labels_
    
    # step 5
    zipped_clid_chdata = zip(cluster_nums, first_chunk)
    clid_chdata = tuple(zipped_clid_chdata)
    populateDS(clid_chdata, points_index)

    # step 6
    lengthRS = len(RetSet)
    if RetSet:
        if lengthRS > 1:
            k_means = KMeans(n_clusters = lengthRS - 1)
        else:
            k_means = KMeans(n_clusters = lengthRS)

        k_means = k_means.fit(RetSet)
        cluster_nums = k_means.labels_
        pts_per_cluster = dict()
        for lbl in cluster_nums:
            pts_per_cluster[lbl] = pts_per_cluster.get(lbl, 0) + 1
        RetSet_key = []
        for ind in pts_per_cluster:
            if pts_per_cluster[ind] == 1:
                RetSet_key.append(ind)
        ind_of_pts_RS = []
        if RetSet_key:
            for k in RetSet_key:
                ind_of_pts_RS.append(list(cluster_nums).index(k))
        clid_RS_zipped = zip(cluster_nums, RetSet)
        clid_RS = tuple(clid_RS_zipped)
        populateCS(clid_RS, RetSet_key, points_index)
        new_RetSet = []
        srtedindxes = sorted(ind_of_pts_RS)
        for x in reversed(srtedindxes):
            new_RetSet.append(RetSet[x])
        RetSet = copy.deepcopy(new_RetSet)
    DiscardSet_sum, CompSet_clstr, CompSet_sum, RetSet_sum = adjust_resSets(RetSet)
    
    output = "The intermediate results:\n"
    output += "Round 1: " + str(DiscardSet_sum) + "," + str(CompSet_clstr) + "," + str(CompSet_sum) + "," + str(RetSet_sum) + "\n"
    for _ in range(0,4):
        # step 7
        if _ != 3:
            shuffled_data = data[chnk_size * (_ + 1) : chnk_size * (_ + 2)]
        else:
            shuffled_data = data[chnk_size * 4:]
            
        # step 8
        pos_discSet = set()
        shd_len = len(shuffled_data)
        for ind in range(shd_len):
            pt_set = shuffled_data[ind]
            pos_discDict = dict()
            for cluster in Discard_Set:
                pos_discDict[cluster] = mahalanobis_dist_pts(pt_set, Discard_Set[cluster])
            lst_mahala_dist = list(pos_discDict.values())
            mahalanobis_dist = min(lst_mahala_dist)
            for pd in pos_discDict:
                if pos_discDict[pd] == mahalanobis_dist:
                    cluster = pd
            if mahalanobis_dist < 2 * (math.sqrt(len(pt_set))):
                Discard_Set[cluster]["N"].append(points_index[tuple(pt_set)])
                Discard_Set[cluster]["SUMSQ"] += pt_set * pt_set
                Discard_Set[cluster]["SUM"] += pt_set
                pos_discSet.add(ind)

        # step 9
        if Comp_set:
            pos_compSet = set()
            for x in range(len(shuffled_data)):
                if x not in pos_discSet:
                    pt_set = shuffled_data[x]
                    pos_discDict = dict()
                    for cluster in Comp_set:
                        pos_discDict[cluster] = mahalanobis_dist_pts(pt_set, Comp_set[cluster])
                    tval = list(pos_discDict.values())
                    mahalanobis_dist = min(tval)
                    for pd in pos_discDict:
                        if pos_discDict[pd] == mahalanobis_dist:
                            cluster = pd
                    if mahalanobis_dist < 2 * (math.sqrt(len(pt_set))):
                        Comp_set[cluster]["N"].append(points_index[tuple(pt_set)])
                        Comp_set[cluster]["SUMSQ"] += pt_set * pt_set
                        Comp_set[cluster]["SUM"] += pt_set
                        pos_compSet.add(x)
        
        # step 10
        try:
            ind_mrgd = pos_compSet.union(pos_discSet)
        except NameError:
            ind_mrgd = pos_discSet
        
        for x in range(len(shuffled_data)):
            if x not in ind_mrgd:
                RetSet.append(shuffled_data[x])
                
        len_RS = len(RetSet)
        if RetSet:
            if len_RS > 1:
                k_means = KMeans(n_clusters = len_RS - 1)
            else:
                k_means = KMeans(n_clusters = len_RS)
            
            k_means = k_means.fit(RetSet)
            RetSet_clst = set(k_means.labels_)
            CompSet_clst = set(Comp_set.keys())
            common = CompSet_clst.intersection(RetSet_clst)
            mix = CompSet_clst.union(RetSet_clst)
            dup_lbl = dict()
            for com in common:
                while True:
                    ranval = random.randint(100, len(shuffled_data))
                    if ranval not in mix:
                        break
                dup_lbl[com] = ranval
                mix.add(ranval)
            lbl = list(k_means.labels_)
            for x in range(len(lbl)):
                if lbl[x] in dup_lbl:
                    lbl[x] = dup_lbl[lbl[x]]
            
            pts_per_cluster = dict()
            for l in lbl:
                pts_per_cluster[l] = pts_per_cluster.get(l, 0) + 1
            
            RetSet_key = []
            ind_of_pts_RS = []
            new_RetSet = []
            flag = True
            for k in pts_per_cluster:
                if pts_per_cluster[k] == 1:
                    RetSet_key.append(k)
            if RetSet_key:
                for k in RetSet_key:
                    ind_of_pts_RS.append(lbl.index(k))
            
            zipped_lblRS = zip(lbl, RetSet)
            lbl_rs = tuple(zipped_lblRS)
            populateCS(lbl_rs, RetSet_key, points_index)
            srted_iop = sorted(ind_of_pts_RS)
            for i in reversed(srted_iop):
                new_RetSet.append(RetSet[i])
            RetSet = copy.deepcopy(new_RetSet)
        
        while 0 == 0:
            merge_list = []
            lcsk = list(Comp_set.keys())
            act_clstr = set(Comp_set.keys())
            CSKeys = list(itertools.combinations(lcsk, 2))
            for csk in CSKeys:
                mahalanobis_dist = mahalanobis_dist_clstr(Comp_set[csk[0]], Comp_set[csk[1]])
                if mahalanobis_dist < 2 * (math.sqrt(len(Comp_set[csk[0]]["SUM"]))):
                    flag = False
                    Comp_set[csk[0]]["N"] = Comp_set[csk[0]]["N"] + Comp_set[csk[1]]["N"]
                    Comp_set[csk[0]]["SUMSQ"] += Comp_set[csk[1]]["SUMSQ"]
                    Comp_set[csk[0]]["SUM"] += Comp_set[csk[1]]["SUM"]
                    Comp_set.pop(csk[1])
                    break            
            tmp = set(Comp_set.keys())
            if tmp == act_clstr:
                break

        CompSet_clstr = list(Comp_set.keys())
        if _ == 3 and Comp_set:
            for csgrp in CompSet_clstr:
                pos_discDict = dict()
                for dscl in Discard_Set:
                    pos_discDict[dscl] = mahalanobis_dist_clstr(Discard_Set[dscl], Comp_set[csgrp])
                l_ind_DS = list(pos_discDict.values())
                mahalanobis_dist = min(l_ind_DS)
                for pd in pos_discDict:
                    if pos_discDict[pd] == mahalanobis_dist:
                        cluster = pd
                len_CS = len(Comp_set[csgrp]["SUM"])
                if mahalanobis_dist < 2 * math.sqrt(len_CS):
                    Discard_Set[cluster]["N"] = Discard_Set[cluster]["N"] + Comp_set[csgrp]["N"]
                    Discard_Set[cluster]["SUMSQ"] += Comp_set[csgrp]["SUMSQ"]
                    Discard_Set[cluster]["SUM"] += Comp_set[csgrp]["SUM"]
                    Comp_set.pop(csgrp)
        DiscardSet_sum, CompSet_clstr, CompSet_sum, RetSet_sum = adjust_resSets(RetSet)
        output += "Round " + str(_ + 2) + ": " + str(DiscardSet_sum) + "," + str(CompSet_clstr) + "," + str(CompSet_sum) + "," + str(RetSet_sum) + "\n"

    output += "\nThe clustering results:\n"
    for cluster in Discard_Set:
        Discard_Set[cluster]["N"] = set(Discard_Set[cluster]["N"])
    if Comp_set:
        for cluster in Comp_set:
            Comp_set[cluster]["N"] = set(Comp_set[cluster]["N"])

    RetSet_set = set()
    for pt in RetSet:
        RetSet_set.add(points_index[tuple(pt)])

    for pt in range(len(index_points)):
        if pt in RetSet_set:
            output += str(pt) + ",-1\n"
        else:
            for cluster in Discard_Set:
                if pt in Discard_Set[cluster]["N"]:
                    output += str(pt) + "," + str(cluster) + "\n"
                    break
            for cluster in Comp_set:
                if pt in Comp_set[cluster]["N"]:
                    output += str(pt) + ",-1\n"
                    break

    with open(sys.argv[3], "w") as writer:
        writer.writelines(output)

    et = time.time()
    print(f"Duration: {et - st}")
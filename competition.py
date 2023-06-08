"""

My Homework 3 was having a few errors due to which it wasn't generating required output. In this code
I have initially tired to get the correct output. To have a more accurate recommendation system, the 
hybrid here uses model based collaborative filtering as the base. To improve the performance of model
based CF I performed data preprocessing so as to have more information from the same data. I have 
performed Principal Component Analysis to increase data interpretability and dimension reduction. I
used only the 10 most important features thus generated. Over model based CF, I then utitlised item 
based collaborative filtering. I used prediction generared by item based as one of the feature to 
model based CF.

Error Distribution:
>=0 and <1: 105773
>=1 and <2: 34123
>=2 and <3: 6310
>=3 and <4: 790
>=4: 0

RSME: 0.9782285911303371
Duration: 861.13
"""

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from pyspark import SparkContext, SparkConf
from datetime import datetime
from itertools import chain
import numpy as nmpy
import pandas as pnds
import json
import time
import math
import sys

NAN = [nmpy.nan]
result_test_data = dict()
result_train_data = dict()

def item_based(bid_uid_dblt, similarity):
    if bid_uid_dblt[1] not in training_data_uid:
        if bid_uid_dblt[0] in bid_actRates:
            temp = bid_actRates[bid_uid_dblt[0]]
            users_oth = list(temp.values())
            avrage_busId = sum(users_oth) / len(users_oth)
            return bid_uid_dblt, avrage_busId
        else:
            return bid_uid_dblt, 3.75

    rate_uid = training_data_uid[bid_uid_dblt[1]]

    ratings = []
    if bid_uid_dblt[0] not in bid_meanRates:
        for ind in rate_uid:
            ratings.append(rate_uid[ind])
        avrage_usrId = sum(ratings)/len(ratings)
        return bid_uid_dblt, avrage_usrId
        
    to_rate = bid_meanRates[bid_uid_dblt[0]]
    to_rate_set = set(to_rate.keys())
    
    sim_list = []
    rating_list = []
    
    for ind in rate_uid:
        ind_rate = (ind, rate_uid[ind]) 
        neighbor = bid_meanRates[ind_rate[0]]
        common_vals = set(neighbor.keys()) & to_rate_set
        if len(common_vals) < 3:
            continue
        dblt = tuple(sorted([ind_rate[0], bid_uid_dblt[0]]))
        if dblt in similarity:
            sim = similarity[dblt]
            sim_list.append(sim)
            rating_list.append(ind_rate[1])
        else:
            parta = 0
            partb = 0
            for intr in common_vals:
                parta += to_rate[intr] * to_rate[intr]
                partb += neighbor[intr] * neighbor[intr]
            parta = math.sqrt(parta)
            partb = math.sqrt(partb)
            sim_den = parta * partb
            if sim_den == 0:
                similarity[dblt] = 0
                continue
            sim = 0
            for intr in common_vals:
                sim += to_rate[intr] * neighbor[intr]
            sim = sim / sim_den
            if not sim < 0:
                sim = sim * (1 + 0.1)
            else:
                eph = (1 - 0.1) * abs(sim)
                sim += eph

            similarity[dblt] = sim
            sim_list.append(sim)
            rating_list.append(ind_rate[1])
    
    if sim_list == []:
        users_oth = list(bid_actRates[bid_uid_dblt[0]].values())
        avrage_busId = sum(users_oth)/len(users_oth)
        return bid_uid_dblt, avrage_busId
    elif len(sim_list) >= 5:
        sim_rating = sorted(tuple(zip(sim_list, rating_list)), key=lambda x: x[0])[0:5]
        den = 0
        num = 0
        for sr in sim_rating:
            den += abs(sr[0])
            num += sr[0] * sr[1]
        if num <= 25:
            temp = bid_actRates[bid_uid_dblt[0]]
            users_oth = list(temp.values())
            avrage_busId = sum(users_oth) / len(users_oth)
            return bid_uid_dblt, avrage_busId
        else:
            pred_val = num / den
            return bid_uid_dblt, pred_val
    else:
        temp = bid_actRates[bid_uid_dblt[0]]
        users_oth = list(temp.values())
        avrage_busId = sum(users_oth) / len(users_oth)
        return bid_uid_dblt, avrage_busId

"""
 Finds mean rating per business
 Finds normalised rating (actual rate - mean) and appends to res
 Appends actual rating to result
"""
def normalise_ratings(bid_listuid):
    total = 0
    for uids_strs in bid_listuid[1]:
        total += uids_strs[1]
    rBar = total / len(bid_listuid[1])

    res=[]
    for uids_strs in bid_listuid[1]:
        res.append((uids_strs[0], uids_strs[1]-rBar))

    k=[]
    for uids_strs in bid_listuid[1]:
        k.append((uids_strs[0], uids_strs[1]))
    return bid_listuid[0],dict(k),dict(res)

def get_matrix_train(val, uids_dfs, bids_dfs):
    list2 = [result_train_data[(val[0], val[1][0])]]
    if val[1][0] in uids_dfs.index:
        if val[0] in bids_dfs.index:
            list1 = list(uids_dfs.loc[val[1][0]])
            return  list1 + list(bids_dfs.loc[val[0]]) +  list2
        else:
            return list1 + NAN * bids_dfs.shape[1] + list2
    elif val[1][0] not in uids_dfs.index:
        shp = uids_dfs.shape[1]
        if val[0] in bids_dfs.index:
            return NAN * shp + list(bids_dfs.loc[val[0]]) + list2
    else:
        return NAN * shp + NAN * bids_dfs.shape[1] + list2

def get_matrix_test(val, uids_dfs, bids_dfs):
    uds = uids_dfs.shape[1]
    rtd = [result_test_data[val]]
    if val[1] in uids_dfs.index:
        if val[0] in bids_dfs.index:
            return list(uids_dfs.loc[val[1]]) + list(bids_dfs.loc[val[0]]) + rtd
        else:
            return list(uids_dfs.loc[val[1]]) + NAN * bids_dfs.shape[1] + rtd
    elif val[1] not in uids_dfs.index:
        if val[0] in bids_dfs.index:
            return NAN * uds + list(bids_dfs.loc[val[0]]) + rtd
    else:
        return NAN * uds + NAN * bids_dfs.shape[1] + rtd

def read_user(input_line):
    uid = input_line["user_id"]
    avg_str = input_line["average_stars"]
    rev_count = input_line["review_count"]
    fans = input_line["fans"]
    frnds = input_line["friends"]
    yelp_since = input_line["yelping_since"]
    useful = input_line["useful"]
    funny = input_line["funny"]
    cool = input_line["cool"]
    comp_hot = input_line["compliment_hot"]
    comp_more = input_line["compliment_more"]
    comp_prof = input_line["compliment_profile"]
    comp_cute = input_line["compliment_cute"]
    comp_lst = input_line["compliment_list"]
    comp_note = input_line["compliment_note"]
    comp_pl = input_line["compliment_plain"]
    comp_cool = input_line["compliment_cool"]
    comp_fun = input_line["compliment_funny"]
    comp_write = input_line["compliment_writer"]
    comp_ph = input_line["compliment_photos"]
    return uid, avg_str, rev_count, fans, frnds, yelp_since, useful, funny, cool, comp_hot, comp_more, comp_prof, comp_cute, comp_lst, comp_note, comp_pl, comp_cool, comp_fun, comp_write, comp_ph

def map_user(inpt):
    uid = user_ids_srno[inpt[0]]
    avg_str = float(inpt[1])
    rev_count = int(inpt[2])
    fans = int(inpt[3]) * int(inpt[3])
    frnds = 0
    if inpt[4] != "None":
        val = len(inpt[4].split(","))
        frnds = val * val
    yelp_since = (datetime.now() - datetime.strptime(inpt[5], "%Y-%m-%d")).days
    useful = int(inpt[6])
    funny = int(inpt[7])
    cool = int(inpt[8])
    comp_hot = int(inpt[9])
    comp_more = int(inpt[10])
    comp_prof = int(inpt[11])
    comp_cute = int(inpt[12])
    comp_lst = int(inpt[13])
    comp_note = int(inpt[14])
    comp_pl = int(inpt[15])
    comp_cool = int(inpt[16])
    comp_fun = int(inpt[17])
    comp_write = int(inpt[18])
    comp_ph = int(inpt[19])
    
    return uid, avg_str, rev_count, fans, frnds, yelp_since, useful, funny, cool, comp_hot, comp_more, comp_prof, comp_cute, comp_lst, comp_note, comp_pl, comp_cool, comp_fun, comp_write, comp_ph

def read_business(input_line):
    bid = input_line["business_id"]
    star = input_line["stars"]
    rev_count = input_line["review_count"]
    state = input_line["state"]
    cat = input_line["categories"]
    city = input_line["city"]
    lat = input_line["latitude"]
    longi = input_line["longitude"]
    hrs = input_line["hours"]
    return bid, star, rev_count, state, cat, city, lat, longi, hrs

def map_business(inpt):
    bid = business_ids_srno[inpt[0]]
    star = float(inpt[1])
    rev_count = int(inpt[2])
    state = inpt[3]
    cat = set()
    if inpt[4]:
        cat = set(inpt[4].split(", "))
    city = inpt[5]
    lat = float(inpt[6])
    longi = float(inpt[7])
    hrs = set()
    if inpt[8]:
        hrs = set(inpt[8].keys())
    return bid, star, rev_count, state, cat, city, lat, longi, hrs
        
def itemBasedCF():
    tdc = testing_data.collect()
    similarity = {}
    for val in tdc:
        bid_uid_dblt, pred_val = item_based(val, similarity)
        result_test_data[bid_uid_dblt] = pred_val
    
    inp_dta = training_data.map(lambda val: (val[0], val[1][0])).collect()
    for dblt in inp_dta:
        pair_y = dblt[1]
        pair_x = dblt[0]
        stars_training_data_uid = training_data_uid[pair_y].pop(pair_x)
        if training_data_uid[pair_y] == dict():
            training_data_uid.pop(pair_y)

        stars_mean_bid_meanRates = bid_meanRates[pair_x].pop(pair_y)
        if bid_meanRates[pair_x] == dict():
            bid_meanRates.pop(pair_x)

        stars_bid_actRates = bid_actRates[pair_x].pop(pair_y)
        if bid_actRates[pair_x] == dict():
            bid_actRates.pop(pair_x)

        bid_uid_dblt, pred_val = item_based(dblt, similarity)
        result_train_data[bid_uid_dblt] = pred_val
   
        if pair_y not in training_data_uid:
            training_data_uid[pair_y] = dict()
            training_data_uid[pair_y][pair_x] = stars_training_data_uid
        else:
            training_data_uid[pair_y][pair_x] = stars_training_data_uid

        if pair_x not in bid_meanRates:
            bid_meanRates[pair_x] = dict()
            bid_meanRates[pair_x][pair_y] = stars_mean_bid_meanRates
        else:
            bid_meanRates[pair_x][pair_y] = stars_mean_bid_meanRates

        if pair_x not in bid_actRates:
            bid_actRates[pair_x] = dict()
            bid_actRates[pair_x][pair_y] = stars_bid_actRates
        else:
            bid_actRates[pair_x][pair_y] = stars_bid_actRates
    
    print("Finished Item-based pred_val")

if __name__ == """__main__""":
    folder_path = sys.argv[1]
    test_path = sys.argv[2]
    st = time.time()
    conf = SparkConf().setMaster("local").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    
    # reading and preprocessing training data
    train_file = sc.textFile(folder_path + "yelp_train.csv")
    train_data_raw = train_file.filter(lambda val: not val.endswith("stars"))
    training_data = train_data_raw.map(lambda line: (line.split(',')[1], line.split(',')[0], line.split(',')[2]))
    busIds_train = training_data.map(lambda val: val[0]).distinct()
    distinct_bids_train = set(busIds_train.collect())
    uids_train = training_data.map(lambda val: val[1]).distinct()
    distinct_uids_train = set(uids_train.collect())
    
    # reading and preprocessing testing data
    test_file = sc.textFile(test_path)
    test_data_raw = test_file.filter(lambda val: not val.endswith("stars"))
    testing_data = test_data_raw.map(lambda line: (line.split(',')[1], line.split(',')[0]))
    bids_test = testing_data.map(lambda val: val[0]).distinct()
    distinct_bids_test = set(bids_test.collect())
    uid_test = testing_data.map(lambda val: val[1]).distinct()
    distinct_uids_test = set(uid_test.collect())
    
    business_ids = distinct_bids_train | distinct_bids_test
    business_ids_copy = business_ids
    srno_business_ids = dict(enumerate(business_ids))
    enum_business_ids_copy = enumerate(business_ids_copy)
    business_ids_srno = dict(map(lambda val:val[::-1], enum_business_ids_copy))
    
    user_ids = distinct_uids_train | distinct_uids_test
    user_ids_copy = user_ids
    srno_user_ids = dict(enumerate(user_ids))
    enum_user_ids_copy = enumerate(user_ids_copy)
    user_ids_srno = dict(map(lambda val:val[::-1], enum_user_ids_copy))
    
    training_data = training_data.map(lambda val: (business_ids_srno[val[0]], (user_ids_srno[val[1]], int(val[2][0])))).cache()
    testing_data =  testing_data.map(lambda val: (business_ids_srno[val[0]], user_ids_srno[val[1]])).cache()
 
    bid_uid_str_grpd = training_data.groupByKey()
    bid_uid_str_lst = bid_uid_str_grpd.map(lambda val: (val[0], list(val[1])))
    bid_uid_str_normalisd = bid_uid_str_lst.map(normalise_ratings).collect()
    a=[]
    b=[]
    c=[]
    for val in bid_uid_str_normalisd:
        a.append(val[0])
        b.append(val[1])
        c.append(val[2])
    bid_actRates = dict(zip(a,b))
    bid_meanRates = dict(zip(a,c))

    training_data_uid = training_data.map(lambda val: (val[1][0], (val[0], val[1][1]))).groupByKey()
    training_data_uid = training_data_uid.map(lambda val: (val[0], list(val[1])))
    training_data_uid = training_data_uid.map(lambda val: (val[0], dict(zip([v[0] for v in val[1]], [v[1] for v in val[1]])))).collect()
    a = []
    b = []
    for td in training_data_uid:
        a.append(td[0])
        b.append(td[1])
    training_data_uid = dict(zip(a, b))
    
    print("Performing Item based pred_val.")
    itemBasedCF()
    
    print("Reading files for Model based Collaborative Filtering.")
    usrIds = []
    with open(folder_path + "user.json") as file_reader:
        while True:
            data = file_reader.readlines(5000000)
            if not data:
                break
            uid_file_data = sc.parallelize(data).map(lambda val: json.loads(val))
            uid_file_data = uid_file_data.map(read_user)
            uid_file_data = uid_file_data.filter(lambda val: val[0] in user_ids_srno).map(map_user)
            usrIds += uid_file_data.collect()
    
    bid_file_data = sc.textFile(folder_path + "business.json")
    bid_file_data = bid_file_data.map(lambda val: json.loads(val))
    bid_file_data = bid_file_data.map(read_business)

    bid_file_data = bid_file_data.filter(lambda val: val[0] in business_ids_srno)

    bid_file_data = bid_file_data.map(map_business)
    busIds = bid_file_data.collect()

    
    cols_uid = ["user_id", "average_stars", "review_count", "fans_sqr", "friends_sqr", "yelping_since", "useful", "funny", "cool", "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute", "compliment_list", "compliment_note", "compliment_plain", "compliment_cool", "compliment_funny", "compliment_writer", "compliment_photos"]
    cols_bid = ["business_id", "stars", "review_count", "state", "categories", "city", "latitude", "longitude", "hours"]
    uids_dfs = pnds.DataFrame(usrIds, columns = cols_uid)
    bids_dfs = pnds.DataFrame(busIds, columns = cols_bid)

    
    #-------------------------------------------
    print("Performing PCA.")
    princCompAnalysis = PCA(n_components = 5, svd_solver = 'full')
    xarr = uids_dfs.iloc[:, 9:]
    x = nmpy.array(xarr)
    print(x.shape)
    compl = princCompAnalysis.fit_transform(x)
    print(compl.shape)
    compl = nmpy.transpose(compl)
    len_compl = len(compl)
    for lc in range(len_compl):
        indx = "compliment_" + str(lc)
        uids_dfs[indx] = compl[lc]
    uids_dfs = uids_dfs.drop(columns = ["compliment_hot", "compliment_more", "compliment_profile", "compliment_cute", "compliment_list", "compliment_note", "compliment_plain", "compliment_cool", "compliment_funny", "compliment_writer", "compliment_photos"])
    
    print(bids_dfs.city.unique())
    city = pnds.get_dummies(bids_dfs.city, prefix = "city")
    city = nmpy.array(city)
    print(city.shape)
    princCompAnalysis = PCA(n_components = 10, svd_solver = 'full')
    city = princCompAnalysis.fit_transform(city)
    print(city.shape)
    city = nmpy.transpose(city)
    ctlen = len(city)
    for ct in range(ctlen):
        indx = "city_" + str(ct)
        bids_dfs[indx] = city[ct]
    bids_dfs = bids_dfs.drop(columns = ["city"])
    
    state = pnds.get_dummies(bids_dfs.state, prefix = "state")
    cols = [state.columns[-1]]
    state = state.drop(columns = cols)
    concat_lst = [bids_dfs, state]
    bids_dfs = pnds.concat(concat_lst, axis = 1)
    bids_dfs = bids_dfs.drop(columns = ["state"])

    set_cat = set()
    mat_cats = []
    for bidcat in bids_dfs.categories:
        set_cat = set(chain(set_cat, bidcat))
    for categs in set_cat:
        val1 = bids_dfs.categories.apply(lambda val: (val & set([categs])))
        val1 = val1.apply(lambda val: 0 if val == set() else 1)
        mat_cats.append(val1)
    mat_cats = nmpy.transpose(nmpy.array(mat_cats))
    princCompAnalysis = PCA(n_components=10, svd_solver='full')
    mat_cats = princCompAnalysis.fit_transform(mat_cats)
    mat_cats = nmpy.transpose(mat_cats)
    len_mc = len(mat_cats)
    for mc in range(len_mc):
        bids_dfs["category_" + str(mc)] = mat_cats[mc]
    bids_dfs = bids_dfs.drop(columns = ["categories"])

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    days_mat = []
    for wd in weekdays:
        temp = bids_dfs.hours.apply(lambda val: (val & set([wd])))
        temp = temp.apply(lambda val: 0 if val == set() else 1)
        days_mat.append(temp)
    len_dm = len(days_mat)
    for ds in range(len_dm):
        ind = "Day_"+str(ds)
        bids_dfs[ind] = days_mat[ds]
    bids_dfs = bids_dfs.drop(columns=["hours"])
    
    print("Finished reading files for model-based CF")
    
    uids_dfs_pred = uids_dfs.set_index("user_id")
    bids_dfs_pred = bids_dfs.set_index("business_id")
    X = []
    for train_data in training_data.collect():
        temp = get_matrix_train(train_data, uids_dfs_pred, bids_dfs_pred)
        X.append(temp)
        
    X = nmpy.array(X)
    test_x_val = []
    for test_data in testing_data.collect():
        temp = get_matrix_test(test_data, uids_dfs_pred, bids_dfs_pred)
        test_x_val.append(temp)
    test_x_val = nmpy.array(test_x_val)
    temp = training_data.map(lambda val: val[1][1]).collect()
    Y = nmpy.array(temp)

    print(X.shape)
    print("Finished preparing X and Y")

    et = time.time()
    print(f"Duration: {et-st}")

    import xgboost as xgb
    from joblib import dump, load

    xgbr = load("model.md")
    
    estim_y_val = xgbr.predict(test_x_val)
    
    model_fin = ""
    rows = testing_data.collect()
    row_len = len(rows)
    for v in range(row_len):
        model_fin += srno_user_ids[rows[v][1]] + "," + srno_business_ids[rows[v][0]] + ","
        if estim_y_val[v] < 1:
            model_fin += str(1)
        elif estim_y_val[v] > 5:
            model_fin += str(5)
        else:
            model_fin += str(estim_y_val[v])
        model_fin += "\n"

    output = "user_id, business_id, stars\n"
    output += model_fin
    
    with open(sys.argv[3], "w") as out_file:
        out_file.writelines(output)

    et = time.time()
    print(f"Duration: {et-st}")
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json
import argparse
import ast
import os
current = os.getcwd()

def str_to_np_array(arr_str):

    '''
    sample format [[501,9311,2.3],[535,8391,2.12]]

    Each record shoul be [country_code ,mcc, us_tran_amt]

    country_code: the data type should be integer and its value should be less than 901
    mcc: the data type should be integer and its value should be less than 10000
    us_tran_amt: the data type should be float

    '''
    list_ = ast.literal_eval(arr_str)
    
    # Convert list to numpy array
    numpy_array = np.array(list_)
    
    data_country = numpy_array[:,0].astype(np.int32)
    data_mcc = numpy_array[:,1].astype(np.int32)
    data_us_amt = numpy_array[:,2].astype(np.float32)
    
    return data_country,data_mcc,data_us_amt

def dataset_set():
    MODEL_PATH_TEST =  os.path.join(os.getcwd(),"vsa_model","data","vsi_sample_inference_result_2.csv")
    df = pd.read_csv(MODEL_PATH_TEST)
    x,Y = df.iloc[:,:-1].values.tolist(), df.iloc[:,-1].values.tolist()
    return x, Y

def make_prediction(INPUT):
    MODEL_PATH_JSON = os.path.join(os.getcwd(),"vsa_model","models","vsi_model_global_2024.json")
    MODEL_PATH_WEIGHTS = os.path.join(os.getcwd(),"vsa_model","models","vsi_model_global_2024.h5")
    loaded_model = tf.keras.models.model_from_json(json.load(open(MODEL_PATH_JSON,'r')))
    loaded_model.load_weights(MODEL_PATH_WEIGHTS)
    data_country,data_mcc,data_us_amt = str_to_np_array(INPUT)
    out = loaded_model.predict([data_country, data_mcc,data_us_amt])
    OUTPUT = out.tolist()
    OUTPUT = [sublist[0][0]for sublist in OUTPUT]
    return OUTPUT

def main():
    MODEL_PATH_JSON = os.path.join(current,"vsa_model","models","vsi_model_global_2024.json")
    MODEL_PATH_WEIGHTS = os.path.join(current,"vsa_model","models","vsi_model_global_2024.h5")
    X,y = dataset_set()
    INPUT = str(X)
    loaded_model = tf.keras.models.model_from_json( json.load(open(MODEL_PATH_JSON,'r')))
    loaded_model.load_weights(MODEL_PATH_WEIGHTS)
    print("Loaded model from disk")
    data_country,data_mcc,data_us_amt = str_to_np_array(INPUT)
    out = loaded_model.predict([data_country, data_mcc,data_us_amt])
    OUTPUT = out.tolist()
    OUTPUT = [sublist[0][0]for sublist in OUTPUT]
    df_out = pd.DataFrame(X, columns = ['country_code' ,'mcc', 'us_tran_amt'])
    df_out['predict'] = OUTPUT
    df_out['predict'] = df_out['predict'].round(5)
    df_out['actual'] = y
    df_out['actual'] = df_out['actual'].round(5)
    return df_out

if __name__ == '__main__':

    main()
##default : Dynamic Batch
import argparse
import tensorflow as tf
import os
import re
from functools import reduce
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2



def convert_h5to_pb(h5Path,pbPath):
    model = tf.keras.models.load_model(h5Path,compile=False)
    # model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)   
    #frozen_func = tf.compat.v1.graph_util.convert_variables_to_constants(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    # print("-" * 50)
    # print("Frozen model layers: ")
    # for layer in layers:
    #     print(layer)

    # print("-" * 50)
    # print("Frozen model inputs: ")
    # print(frozen_func.inputs)
    # print("Frozen model outputs: ")
    # print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=pbPath[:pbPath.rfind(r"/")],
                      name=pbPath[pbPath.rfind(r"/")+1:],
                      as_text=False)
    ipsN,opsN=str(frozen_func.inputs[0]),str(frozen_func.outputs[0])
    # print(ipsN)
    # print(opsN)
    inputNodeName=ipsN[ipsN.find("\"")+1:ipsN.find(":")]
    outputNodeName=opsN[opsN.find("\"")+1:opsN.find(":")]
    # print(inputNodeName)
    # print(outputNodeName)
    inputShapeK=ipsN[ipsN.find("=(")+2:ipsN.find("),")] 
    inputShapeF=re.findall(r"\d+\.?\d*",inputShapeK)
    inputShape=reduce(lambda x, y: x + 'x' + y, inputShapeF)
    # print(inputShape)

    return inputNodeName,outputNodeName,inputShape


def convert_hdf5_to_trt(model_type, work_dir, model_name, abfcmode_Idx, workspace='3072', optBatch='20', maxBatch='100'):
    if model_type=='HRRP':
        hdfPath = work_dir+"/model/"+model_name+".hdf5"
    elif model_type=='ABFC':
        hdfPath = work_dir+"/model/"+model_name+"_feature_"+abfcmode_Idx+".hdf5"
        trtPath = work_dir+"/"+model_name+"_feature_"+abfcmode_Idx+".trt"
    elif model_type=='FewShot':
        hdfPath = work_dir+"/model/"+model_name+".hdf5"
    elif model_type=='ATEC':
        hdfPath = work_dir+"/model/fea_ada_trans.hdf5"
        trtPath = work_dir+"/"+model_name+".trt"
    pbPath  = work_dir+"/model/temp.pb"
    oxPath  = work_dir+"/model/temp.onnx"
    
    
    try:
        inputNodeName,outputNodeName,inputShape=convert_h5to_pb(hdfPath,pbPath)
        #pb converto onnx
        '''python -m tf2onnx.convert  --input temp.pb --inputs Input:0 --outputs Identity:0 --output temp.onnx --opset 11'''
        os.system("python -m tf2onnx.convert  --input "+pbPath+" --inputs "+inputNodeName+":0 --outputs "+outputNodeName+":0 --output "+oxPath+" --opset 11")
        #onnx converto trt
        '''trtexec --explicitBatch --workspace=3072  --minShapes=Input:0:1x128x64x1 --optShapes=Input:0:20x128x64x1 --maxShapes=Input:0:100x128x64x1 --onnx=temp.onnx --saveEngine=temp.trt --fp16'''
        os.system("trtexec --onnx="+oxPath+" --saveEngine="+trtPath+" --workspace="+workspace+" --minShapes=Input:0:1x"+inputShape+\
        " --optShapes=Input:0:"+optBatch+"x"+inputShape+" --maxShapes=Input:0:"+maxBatch+"x"+str(inputShape)+" --fp16")
    except Exception as e:
        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--model_type', help='the directory of the training data',default="HRRP")
    parser.add_argument('--work_dir', help='the directory of the training data',default="../../db/trainLogs")
    parser.add_argument('--model_name', help='the directory of the training data',default="model")
    parser.add_argument('--abfcmode_Idx', help='the model index that be choosed one ',default="39")
    args = parser.parse_args()

    convert_hdf5_to_trt(args.model_type, args.work_dir, args.model_name, args.abfcmode_Idx)
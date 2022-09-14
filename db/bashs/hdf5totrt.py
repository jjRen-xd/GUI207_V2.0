##default : Dynamic Batch

import argparse
import tensorflow as tf
import keras as K
import os
import re
from functools import reduce
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def convert_h5to_pb(h5Path,pbPath):
    model = tf.keras.models.load_model(h5Path,compile=False)
    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)   
    #frozen_func = tf.compat.v1.graph_util.convert_variables_to_constants(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=pbPath[:pbPath.rfind(r"/")],
                      name=pbPath[pbPath.rfind(r"/")+1:],
                      as_text=False)
    ipsN,opsN=str(frozen_func.inputs[0]),str(frozen_func.outputs[0])
    print(ipsN)
    print(opsN)
    inputNodeName=ipsN[ipsN.find("\"")+1:ipsN.find(":")]
    outputNodeName=opsN[opsN.find("\"")+1:opsN.find(":")]
    print(inputNodeName)
    print(outputNodeName)
    inputShapeK=ipsN[ipsN.find("=(")+2:ipsN.find("),")] 
    inputShapeF=re.findall(r"\d+\.?\d*",inputShapeK)
    inputShape=reduce(lambda x, y: x + 'x' + y, inputShapeF)
    print(inputShape)

    return inputNodeName,outputNodeName,inputShape

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--dfPath', '-df', help='Path to hdf5 model, necessary.',required=True)
parser.add_argument('--pbPath', '-pb', help='Path to pb model, not necessary.',default="./temp.pb")
parser.add_argument('--oxPath', '-ox', help='Path to onnx model, not necessary.',default="./temp.onnx")
parser.add_argument('--trtPath', '-trt', help='Path to trt model, not necessary.',default="./temp.trt")
parser.add_argument('--workspace', '-workspace', help='trtexec workspace.',default="3072")

parser.add_argument('--optBatch', '-optBatch', help='the optBatch of engine.',default="20")
parser.add_argument('--maxBatch', '-maxBatch', help='the maxBatch of engine.',default="100")

args = parser.parse_args()

if __name__ == '__main__':
    try:
        inputNodeName,outputNodeName,inputShape=convert_h5to_pb(args.dfPath,args.pbPath)
        #pb converto onnx
        '''python -m tf2onnx.convert  --input temp.pb --inputs Input:0 --outputs Identity:0 --output temp.onnx --opset 11'''
        os.system("python -m tf2onnx.convert  --input "+args.pbPath+" --inputs "+inputNodeName+":0 --outputs "+outputNodeName+":0 --output "+args.oxPath+" --opset 11")
        #onnx converto trt
        '''trtexec --explicitBatch --workspace=3072  --minShapes=Input:0:1x128x64x1 --optShapes=Input:0:20x128x64x1 --maxShapes=Input:0:100x128x64x1 --onnx=temp.onnx --saveEngine=temp.trt --fp16'''
        os.system("trtexec --onnx="+args.oxPath+" --saveEngine="+args.trtPath+" --workspace="+args.workspace+" --minShapes=Input:0:1x"+inputShape+\
        " --optShapes=Input:0:"+args.optBatch+"x"+inputShape+" --maxShapes=Input:0:"+args.maxBatch+"x"+inputShape+" --fp16")
    
    except Exception as e:
        print(e)
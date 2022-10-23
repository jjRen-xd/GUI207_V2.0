import re
from functools import reduce
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def convert_h5to_pbb(h5Path,pbPath):
    model = tf.keras.models.load_model(h5Path,compile=False)
    # model.summary()
    full_model = tf.function(lambda Input: model(Input))
    print("aaaaaaaaaaaaaaaaaaa")
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    print("bbbbbbbbbbbbbbbbbbb")
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)   
    #frozen_func = tf.compat.v1.graph_util.convert_variables_to_constants(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]

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

def convert_h5to_pb():
    model = tf.keras.models.load_model("D:/lyh/GUI207_V2.0/db/trainLogs/2022-09-28-17-30-03-AFS-fea39/model/afstest_feature_39.hdf5",compile=False)
    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
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
                      logdir="D:/lyh/GUI207_V2.0/db/trainLogs/2022-09-28-17-30-03-AFS-fea39/model/",
                      name="afstest_feature_39.pb",
                      as_text=False)
convert_h5to_pbb("D:/lyh/GUI207_V2.0/db/trainLogs/2022-09-28-17-30-03-AFS-fea39/model/afstest_feature_39.hdf5","D:/lyh/GUI207_V2.0/db/trainLogs/2022-09-28-17-30-03-AFS-fea39/model/afstest_feature_39.pb")
#python -m tf2onnx.convert  --input 1000_feature_39.pb --inputs Input:0 --outputs Identity:0 --output ../onnx_model/1000_feature_39.onnx --opset 11
#trtexec --onnx=500_feature_39.onnx --saveEngine=500_feature_39.trt --workspace=4096 --minShapes=Input:0:1x39 --optShapes=Input:0:1x39 --maxShapes=Input:0:100x39 --fp16
#python -m tf2onnx.convert  --input feature_22.pb --inputs Input:0 --outputs Identity:0 --output ../onnx_model/feature_22.onnx --opset 11
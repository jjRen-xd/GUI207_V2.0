import torch,os
import torch.onnx

def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cuda'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model = torch.load(checkpoint)
    model.to(device)
    model.eval()
    
    
    torch.onnx.export(  model, 
                        input, 
                        onnx_path, 
                        verbose=False, 
                        training=False,
                        input_names=input_names, 
                        output_names=output_names,
                        do_constant_folding=False
    ) #指定模型的输入，以及onnx的输出路径

    print("Exporting .pth model to onnx model has been successful!")


if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES']='1'
    checkpoint = './checkpoints/MsmcNet_hrrp_-15db_after2.pth'
    onnx_path = './checkpoints/MsmcNet_hrrp_-15db_after2.onnx'
    trt_path = './checkpoints/MsmcNet_hrrp_-15db_after2.trt'
    #device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    input = torch.randn(1, 1, 128, 2).to('cuda')
    pth_to_onnx(input, checkpoint, onnx_path)

    cmd_onnx2trt="trtexec.exe --explicitBatch --workspace=3072 --minShapes=input:1x1x"+\
        "128x2 --optShapes=input:20x1x"+\
        "128x2 --maxShapes=input:512x1x"+\
        "128x2 --onnx="                 + \
        onnx_path+ " --saveEngine="+\
        trt_path+ " --fp16"
    os.system(cmd_onnx2trt)
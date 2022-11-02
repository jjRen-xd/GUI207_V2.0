INCLUDEPATH += \
  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/include" \
  D:/lyh/evn/TensorRT-7.2.3.4.Windows10.x86_64.cuda-11.1.cudnn8.1/TensorRT-7.2.3.4/include \
  D:/lyh/softs/MATLAB/R2022a/extern/include \
  D:/lyh/GUI207_V2.0/lib/TRANSFER \
  D:/win_anaconda/include \
  D:/win_anaconda/Lib/site-packages/numpy/core/include/numpy \
  D:/lyh/evn/OpenCV4.5.4/opencv/build/include

LIBS += \
  -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/lib/x64" \
  -lcudart \
  -lcuda \
  -lcudadevrt \
  -LD:/lyh/evn/TensorRT-7.2.3.4.Windows10.x86_64.cuda-11.1.cudnn8.1/TensorRT-7.2.3.4/lib \
  -lnvinfer \
  -LD:/lyh/softs/MATLAB/R2022a/extern/lib/win64/microsoft \
  -llibmat \
  -llibmx \
  -llibmex \
  -llibeng \
  -lmclmcr \
  -lmclmcrrt \
  -LD:/lyh/GUI207_V2.0/build/lib/TRANSFER \
  -lToHrrp \
  -LD:/win_anaconda/libs \
  -lpython39 \
  -LD:/win_anaconda/Lib/site-packages/numpy/core/lib \
  -lnpymath \
  -LD:/lyh/evn/OpenCV4.5.4/opencv/build/x64/vc15/lib \
  -lopencv_world454


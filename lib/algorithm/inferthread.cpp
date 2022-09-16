#include "inferthread.h"

InferThread::InferThread(QSemaphore *s,std::queue<std::vector<float>>* sharedQ,QMutex *l):sem(s),sharedQue(sharedQ),lock(l)
{
    // 网络输出标签对应类别名称初始化
    label2class[0] ="bigball"; //"XQ";'bigball','DT','Moxiu','sallball','taper', 'WD'
    label2class[1] ="DT"; //"DQ";
    label2class[2] ="Moxiu"; //"Z";
    label2class[3] ="sallball"; //"QDZ";
    label2class[4] ="taper"; //"DT";
    label2class[5] ="WD"; //"FG";
    for(auto &item: label2class){
        class2label[item.second] = item.first;
    }

    trtInfer = new TrtInfer(class2label);
    qRegisterMetaType<QVariant>("QVariant");
}

void InferThread::run(){
    if(inferMode=="real_time_infer"){
        while(true){
            sem->acquire(1);
            //QMutexLocker x(lock);
            std::vector<float> temp(sharedQue->front());
            qDebug()<<"(InferThread::run) acquire得到的temp.size()= "<<temp.size();
            int preIdx;
            std::vector<float> degrees;
            //degrees={0.1,0.2,0.3,0.1,0.2,0.1};preIdx=2;
            trtInfer->realTimeInfer(temp, modelPath, dataProcess, &preIdx, degrees);
            QVariant qv; qv.setValue(degrees);
            emit sigInferResult(preIdx,qv);
            //qDebug()<<"InferThread::run  emit's  is "<<asdf;
            sharedQue->pop();
        }
    }
}

void InferThread::setInferMode(std::string infermode){
    inferMode=infermode;
}

void InferThread::setParmOfRTI(std::string modelP,bool dataP){
    modelPath=modelP;
    dataProcess=dataP;
}

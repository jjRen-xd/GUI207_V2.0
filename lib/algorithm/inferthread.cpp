#include "inferthread.h"

InferThread::InferThread(QSemaphore *s,std::queue<std::vector<float>>* sharedQ,QMutex *l):sem(s),sharedQue(sharedQ),lock(l)
{
    qRegisterMetaType<QVariant>("QVariant");
}

void InferThread::run(){
    if(inferMode=="real_time_infer"){
        trtInfer_rti->createEngine(modelPath);
        emit modelAlready();
        while(!isInterruptionRequested()){
            sem->acquire(1);
            qDebug()<<"(InferThread::run)  acquired one";
            //QMutexLocker x(lock);
            std::vector<float> temp(sharedQue->front());
            //qDebug()<<"(InferThread::run) acquire得到的temp.size()= "<<temp.size();
            int preIdx;
            std::vector<float> degrees;
            //degrees={0.1,0.2,0.3,0.1,0.2,0.1};preIdx=2;
            trtInfer_rti->realTimeInfer(temp, modelPath, dataProcess, &preIdx, degrees);
            QVariant qv; qv.setValue(degrees);
            emit sigInferResult(preIdx,qv);//monitor主线程槽函数接收
            //qDebug()<<"InferThread::run  emit's  is "<<asdf;
            sharedQue->pop();
        }
    }
}
void InferThread::setClass2LabelMap(std::map<std::string, int> class2label){
    trtInfer_rti = new TrtInfer(class2label);
    qDebug()<<"(InferThread::setClass2LabelMap) class2label.size()=="<<class2label.size();
}
void InferThread::setInferMode(std::string infermode){
    inferMode=infermode;
}

void InferThread::setParmOfRTI(std::string modelP,bool ifDataPreProcess){
    dataProcess=ifDataPreProcess;
    modelPath=modelP;
}

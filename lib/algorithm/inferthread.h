#ifndef INFERTHREAD_H
#define INFERTHREAD_H
#include <QThread>
#include <QSemaphore>
#include <QMutex>
#include <QMutexLocker>
#include <QVariant>
#include <queue>
#include "./lib/algorithm/trtinfer.h"
Q_DECLARE_METATYPE(std::vector<float>);
class InferThread:public QThread
{
    Q_OBJECT   //申明需要信号与槽机制支持
public:
    InferThread(QSemaphore *sem,std::queue<std::vector<float>>* sharedQue,QMutex *l);
    ~InferThread(){
        requestInterruption();
        quit();
        wait();
    };
    void setInferMode(std::string infermode);

    void run();
    QSemaphore *sem;

    
    void setParmOfRTI(std::string modelPath,bool ifDataPreProcess);
    void setClass2LabelMap(std::map<std::string, int> class2label);
    std::queue<std::vector<float>>* sharedQue;
    QMutex *lock;
signals:
    void sigInferResult(int,QVariant);
    void modelAlready();
private:
    TrtInfer* trtInfer_rti;
    //trtInfer需要的参数
    std::string inferMode="";
    std::string modelPath="";
    bool dataProcess=true;


};

#endif // INFERTHREAD_H

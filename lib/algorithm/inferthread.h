#ifndef INFERTHREAD_H
#define INFERTHREAD_H
#include <QThread>
#include <QSemaphore>
#include <QMutex>
#include <QMutexLocker>
#include "./lib/algorithm/trtinfer.h"

class InferThread:public QThread
{
    Q_OBJECT   //申明需要信号与槽机制支持
public:
    InferThread(QSemaphore *sem,std::queue<std::vector<float>>* sharedQue,QMutex *l);
    void setInferMode(std::string infermode);

    void run();
    QSemaphore *sem;

    TrtInfer* trtInfer;
    void setParmOfRTI(std::string modelPath,bool dataProcess);

    std::queue<std::vector<float>>* sharedQue;
    QMutex *lock;
signals:
    void sigInferResult(QString);

private:
    std::map<int, std::string> label2class;
    std::map<std::string, int> class2label;

    //trtInfer需要的参数
    std::string inferMode="";
    std::string modelPath="";
    bool dataProcess=false;


};

#endif // INFERTHREAD_H

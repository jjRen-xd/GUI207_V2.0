#ifndef SOCKETCLIENT_H
#define SOCKETCLIENT_H
#include <mat.h>
#include <math.h>
#include <stdio.h>
#include <io.h>
#include <winsock.h>   // windows平台的网络库头文件
#include <QDebug>
#include <QThread>
#include <QMutex>
#include <QMutexLocker>
#include "./lib/dataprocess/matdataprocess.h"
#include "./lib/dataprocess/customdataset.h"
class SocketClient:public QThread
{
    Q_OBJECT
public:
    SocketClient();
    ~SocketClient(){
        requestInterruption();
        quit();
        wait();
    };
    void initSocketClient();
    SOCKET createClientSocket(const char* ip);
    void run();
    void setClass2LabelMap(std::map<std::string, int> class2label);
    void setParmOfRTI(std::string modelPath);
    bool startOrstop=true;
    bool m_flag;
    QMutex m_lock;
    void StopThread(){
        QMutexLocker lock(&m_lock);
        m_flag = !m_flag;
        qDebug()<<"StopThread hrerererer";
    }
public
slots:
    void startOrstop_slot(bool);
signals:
    void sigClassName(int);


private:
    std::string datasetlPath="";
    bool dataProcess=false;
    std::map<std::string, int> class2label;
    
};

#endif // SOCKETCLIENT_H

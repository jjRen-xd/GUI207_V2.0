#ifndef SOCKETCLIENT_H
#define SOCKETCLIENT_H
#include <mat.h>
#include <math.h>
#include <stdio.h>
#include <io.h>
#include <winsock.h>   // windows平台的网络库头文件
#include <QDebug>
#include <QThread>
#include "./lib/dataprocess/matdataprocess.h"
#include "./lib/dataprocess/customdataset.h"
class SocketClient:public QThread
{
    Q_OBJECT
public:
    SocketClient();
    void initSocketClient();
    SOCKET createClientSocket(const char* ip);
    void run();
    void setClass2LabelMap(std::map<std::string, int> class2label);
    void setParmOfRTI(std::string modelPath);
signals:
    void sigClassName(int);

private:
    std::string datasetlPath="";
    bool dataProcess=false;
    std::map<std::string, int> class2label;
};

#endif // SOCKETCLIENT_H

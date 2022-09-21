#ifndef SOCKETCLIENT_H
#define SOCKETCLIENT_H
#include <mat.h>
#include <math.h>
#include <stdio.h>
#include <io.h>
#include <winsock.h>   // windows平台的网络库头文件
#include <QDebug>
#include <QThread>
#include "./lib/algorithm/matdataprocess.h"
#include "./lib/algorithm/customdataset.h"
class SocketClient:public QThread
{
    Q_OBJECT
public:
    SocketClient();
    void initSocketClient();
    SOCKET createClientSocket(const char* ip);
    void run();
signals:
    void sigClassName(int);

};

#endif // SOCKETCLIENT_H

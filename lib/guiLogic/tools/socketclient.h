#ifndef SOCKETCLIENT_H
#define SOCKETCLIENT_H
#include <mat.h>
#include <math.h>
#include <stdio.h>
#include <io.h>
#include <windows.h>
#include <winsock.h>   // windows平台的网络库头文件
#include <QDebug>
#include <QThread>
class SocketClient:public QThread
{

public:
    SocketClient();
    void initSocketClient();
    SOCKET createClientSocket(const char* ip);
    void run();
};

#endif // SOCKETCLIENT_H

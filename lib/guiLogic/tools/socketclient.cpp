/*模拟信号发送的线程，是socketServer的生产者*/
#include "socketclient.h"
#pragma comment(lib,"ws2_32.lib")   // 库文件
#define PORT 2287
#define RECEIVE_BUF_SIZ 512

SocketClient::SocketClient(){
    //moveToThread(this); 
}

void SocketClient::initSocketClient() {
    WORD w_req = MAKEWORD(2, 2);//版本号
    WSADATA wsadata;
    // 成功：WSAStartup函数返回零
    if (WSAStartup(w_req, &wsadata) != 0) {
        qDebug() << "(SocketClient::initialization) 初始化套接字库失败！";
    }
    else {
        //qDebug()<< "初始化套接字库成功！";
    }
}

SOCKET SocketClient::createClientSocket(const char* ip){
    SOCKET c_client = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (c_client == INVALID_SOCKET){
        qDebug() << "(SocketClient::createClientSocket) 套接字创建失败！";
        WSACleanup();
    }
    else {
        //qDebug() << "(SocketClient::createClientSocket) 套接字创建成功！";
    }

    //2.连接服务器
    struct sockaddr_in addr;   // sockaddr_in, sockaddr  老版本和新版的区别
    addr.sin_family = AF_INET;  // 和创建socket时必须一样
    addr.sin_port = htons(PORT);       // 端口号  大端（高位）存储(本地)和小端（低位）存储(网络），两个存储顺序是反着的  htons 将本地字节序转为网络字节序
    addr.sin_addr.S_un.S_addr = inet_addr(ip); //inet_addr将点分十进制的ip地址转为二进制

    if (::connect(c_client, (struct sockaddr*)&addr, sizeof(addr)) == INVALID_SOCKET){
        qDebug() << "(SocketClient::createClientSocket)服务器连接失败！" ;
        WSACleanup();
    }
    else {
        qDebug() << "(SocketClient::createClientSocket)服务器连接成功！" ;
    }
    return c_client;
}

void SocketClient::run(){
    qDebug()<<"SocketClient::run is in thread:"<<QThread::currentThreadId();
    m_flag = true;
    char send_buf[BUFSIZ];
    SOCKET s_server;
    initSocketClient();
    s_server = createClientSocket("127.0.0.1");
    int inputLen=128;
    // for (auto const &pair: class2label) {
    //     qDebug()<< "{" << QString::fromStdString(pair.first) << ": " << QString::number(pair.second) << "}";
    // }
    //qDebug()<<"(SocketClient::run)datasetlPath=="<<QString::fromStdString(datasetlPath);
    auto mydataset = CustomDataset(datasetlPath, false, ".mat", class2label,inputLen);//发的数据不做归一化预处理。inputLen要和单一样本长度一致，而不能是可能更大的输入层数据长度
    int mydataset_size=mydataset.labels.size();
    int classIdx_rightnow=mydataset.labels[0];
    //qDebug()<<"(SocketClient::run)mydataset_size=="<<QString::number(mydataset.labels.size());
    for(int i=0;i<mydataset_size;i++){
        if(isInterruptionRequested()) break;
        while(!startOrstop){};//如果是0就卡在这里
        // QMutexLocker lock(&m_lock);
        // if(!m_flag) break;
        // while(!m_flag){};//如果是0就卡在这里

        for(int j=0;j<inputLen;j++){
            float floatVariable = mydataset.data[i][j];
            std::string str = std::to_string(floatVariable);
            strcpy(send_buf, str.c_str());
            if (send(s_server, send_buf, BUFSIZ, 0) < 0) {
                qDebug() << "发送失败！" ;
                break;
            }
            if (i > 0) _sleep(1);
        }
        if(mydataset.labels[i]!=classIdx_rightnow){//如果发送的类别变了的话，发送新的类别信号
            classIdx_rightnow=mydataset.labels[i];
            emit sigClassName(classIdx_rightnow);
        }
        if (i == 0){
            _sleep(2500);
            emit sigClassName(classIdx_rightnow);
        }
        qDebug()<< "==================Send "<<QString::number(inputLen)<<"==============="<< QString::number(i);
    }
    qDebug()<< "600个发送完毕";  
}
void SocketClient::setClass2LabelMap(std::map<std::string, int> class2label0){
    class2label=class2label0;
    qDebug()<<"(SocketClient::setClass2LabelMap) class2label.size()=="<<class2label.size();
}
void SocketClient::setParmOfRTI(std::string datasetP){
    datasetlPath=datasetP;
}

void SocketClient::startOrstop_slot(bool startorstop){
    startOrstop=startorstop;
    qDebug()<<"startOrstop="<<startOrstop;
    qDebug()<<"startOrstop_slot function is in thread:"<<QThread::currentThreadId();
}

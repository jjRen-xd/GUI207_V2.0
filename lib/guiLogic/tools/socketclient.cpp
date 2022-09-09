#include "socketclient.h"
//#pragma comment(lib,"ws2_32.lib")   // 库文件
//#define PORT 2287
//#define RECEIVE_BUF_SIZ 512

//void oneNormalization_copy(std::vector<float> &list){
//    //特征归一化
//    float dMaxValue = *max_element(list.begin(),list.end());  //求最大值
//    float dMinValue = *min_element(list.begin(),list.end());  //求最小值
//    for (int f = 0; f < list.size(); ++f) {
//        list[f] = (1-0)*(list[f]-dMinValue)/(dMaxValue-dMinValue+1e-8)+0;//极小值限制
//    }
//}

//void getDataFromMat_copy(std::string targetMatFile,int emIdx,bool dataProcess,float *data,int inputLen){
//    MATFile* pMatFile = NULL;
//    mxArray* pMxArray = NULL;
//    // 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）
//    double* matdata;
//    pMatFile = matOpen(targetMatFile.c_str(), "r");
//    if(!pMatFile){
//        qDebug()<<"(SocketClient:getDataFromMat_copy)文件指针空！！！！！！";
//        return;
//    }

//    std::string matVariable=targetMatFile.substr(
//                targetMatFile.find_last_of('/')+1,
//                targetMatFile.find_last_of('.')-targetMatFile.find_last_of('/')-1).c_str();//假设数据变量名同文件名的话
//    qDebug()<<"(SocketClient:getDataFromMat_copy)matVariable=="<<QString::fromStdString(matVariable);
//    pMxArray = matGetVariable(pMatFile,matVariable.c_str());
//    if(!pMxArray){
//        qDebug()<<"(SocketClient:getDataFromMat_copy)pMxArray变量没找到！！！！！！";
//        return;
//    }
//    matdata = (double*)mxGetData(pMxArray);
//    int M = mxGetM(pMxArray);  //M行数
//    int N = mxGetN(pMxArray);  //N 列数
//    if(emIdx>N) emIdx=N-1; //说明是随机数

//    std::vector<float> onesmp;//存当前样本
//    for(int i=0;i<M;i++){
//        onesmp.push_back(matdata[emIdx*M+i]);
//    }
//    if(dataProcess) oneNormalization_copy(onesmp);//归一化
//    for(int i=0;i<inputLen;i++){
//        data[i]=onesmp[i%M];//matlab按列存储
//    }
//}

SocketClient::SocketClient(){

}

//void SocketClient::initSocketClient() {
//    WORD w_req = MAKEWORD(2, 2);//版本号
//    WSADATA wsadata;
//    // 成功：WSAStartup函数返回零
//    if (WSAStartup(w_req, &wsadata) != 0) {
//        qDebug() << "(SocketClient::initialization) 初始化套接字库失败！";
//    }
//    else {
//        //qDebug()<< "初始化套接字库成功！";
//    }
//}

//SOCKET SocketClient::createClientSocket(const char* ip){
//    SOCKET c_client = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
//    if (c_client == INVALID_SOCKET){
//        qDebug() << "(SocketClient::createClientSocket) 套接字创建失败！";
//        WSACleanup();
//    }
//    else {
//        //qDebug() << "(SocketClient::createClientSocket) 套接字创建成功！";
//    }

//    //2.连接服务器
//    struct sockaddr_in addr;   // sockaddr_in, sockaddr  老版本和新版的区别
//    addr.sin_family = AF_INET;  // 和创建socket时必须一样
//    addr.sin_port = htons(PORT);       // 端口号  大端（高位）存储(本地)和小端（低位）存储(网络），两个存储顺序是反着的  htons 将本地字节序转为网络字节序
//    addr.sin_addr.S_un.S_addr = inet_addr(ip); //inet_addr将点分十进制的ip地址转为二进制

//    if (::connect(c_client, (struct sockaddr*)&addr, sizeof(addr)) == INVALID_SOCKET){
//        qDebug() << "(SocketClient::createClientSocket)服务器连接失败！" ;
//        WSACleanup();
//    }
//    else {
//        qDebug() << "(SocketClient::createClientSocket)服务器连接成功！" ;
//    }
//    return c_client;
//}

//void SocketClient::run(){
//    char send_buf[BUFSIZ];
//    SOCKET s_server;
//    initSocketClient();
//    s_server = createClientSocket("127.0.0.1");
//    std::string targetPath = "E:/207Project/GUI207_V2.0/db/datasets/falseHRRPmat_1x128/DT/hrrp128.mat";
//    int inputLen = 128;
//    float* indata = new float[inputLen]; std::fill_n(indata, inputLen, 0);
//    for (int i = 0; i < 99; i++) {
//        getDataFromMat_copy(targetPath,i,false, indata, inputLen);
//        for (int j = 0; j < inputLen; j++) {
//            float floatVariable = indata[j];
//            std::string str = std::to_string(floatVariable);
//            strcpy(send_buf, str.c_str());
//            if (send(s_server, send_buf, BUFSIZ, 0) < 0) {
//                qDebug() << "发送失败！" ;
//                break;
//            }if (i > 0) _sleep(30);
//            std::string tem = send_buf;  //qDebug() << "send " << tem ;
//        }if (i == 0) _sleep(1000);
//        //std::cout << std::endl << "==================Send 128===============" << std::endl;
//    }
//}



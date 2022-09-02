#include "modelTrain.h"

ModelTrain::ModelTrain(QTextBrowser* Widget, QLabel* trainImg, QLabel* valImg, QLabel* confusionMat,
                       QTextEdit* timeRestEdit, QProgressBar* trainProgressBar):
    OutShow(Widget),
    trainImg(trainImg),
    valImg(valImg),
    confusionMat(confusionMat),
    timeRestEdit(timeRestEdit),
    trainProgressBar(trainProgressBar),
    process_train(new QProcess(this)){
    QObject::connect(process_train, &QProcess::readyReadStandardOutput, this, &ModelTrain::readLogOutput);
    QObject::connect(process_train, &QProcess::readyReadStandardError, this, &ModelTrain::readLogError);
}

ModelTrain::~ModelTrain(){

}

void ModelTrain::readLogOutput(){
    /* 读取终端输出并显示 */
    QByteArray cmdOut = process_train->readAllStandardOutput();
    if(!cmdOut.isEmpty()){
        QString logs=QString::fromLocal8Bit(cmdOut);
        QStringList lines = logs.split("\n");
        int len=lines.length();
        for(int i=0;i<len;i++){
            QStringList Infos = lines[i].simplified().split(" ");
            if(Infos.length()<2){
                continue;
            }
            if(lines[i].contains("Train Ending",Qt::CaseSensitive)){
                OutShow->append("===========================Train Ending===========================");
                showLog=false;
                showTrianResult();
            }
            else if(Infos[0]=="RestTime:"){
                qDebug() << Infos;
                float m_dTotalTime = Infos[1].toFloat()+0.5;
                if(m_dTotalTime>0){
                    int H = m_dTotalTime / (60*60);
                    int M = (m_dTotalTime- (H * 60 * 60)) / 60;
                    int S = (m_dTotalTime - (H * 60 * 60)) - M * 60;
                    QString hour = QString::number(H);
                    if (hour.length() == 1) hour = "0" + hour;
                    QString min = QString::number(M);
                    if (min.length() == 1) min = "0" + min;
                    QString sec = QString::number(S);
                    if (sec.length() == 1) sec = "0" + sec;
                    QString qTime = hour + ":" + min + ":" + sec;
                    timeRestEdit->setText(qTime);
                }
            }
            else if(Infos[0]=="Schedule:"){
                qDebug() << Infos;
                trainProgressBar->setValue(Infos[1].toInt());
            }
            else if(showLog){
                OutShow->append(lines[i]);
            }
        }
    }
    OutShow->update();
}

void ModelTrain::readLogError(){
    /* 读取终端Error并显示 */
    QByteArray cmdOut = process_train->readAllStandardError();
    timeRestEdit->setText("训练出错");
    OutShow->append(QString::fromLocal8Bit(cmdOut));
    OutShow->update();
    stopTrain();
}

void ModelTrain::startTrain(QString cmd){
  // TODO add code here
    if(process_train->state()==QProcess::Running){
        showLog=false;
        process_train->close();
        process_train->kill();
    }
    showLog=true;
    OutShow->setText("===========================Train Starting===========================");
    timeRestEdit->setText("启动模块并计算中");
    trainProgressBar->setValue(0);
    trainImg->clear();
    valImg->clear();
    confusionMat->clear();
    process_train->setProcessChannelMode(QProcess::MergedChannels);
    process_train->start("cmd.exe");
    process_train->write(cmd.toLocal8Bit() + '\n');

    // QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    // //QProcessEnvironment env; //有时候要用这个空环境变量才行，否则可能会出现一些污染的问题
    // env.insert("PYTHONPATH", "D:/win_anaconda/Lib/site-packages");
    // process_train->setProcessEnvironment(env);
    // QStringList params;
    // QString pythonPath = "D:/win_anaconda/python.exe";
    // // 要执行的文件
    // QString pythonScript = "D:/lyh/GUI207_V2.0/db/bashs/hrrp/hrrpTrainTest.py";
    // //params << pythonScript <<" --data_dir D:/lyh/GUI207_V2.0/db/datasets/falseHRRPmat_1x128 --batch_size 32 --max_epochs 3";
    // // 设置工作目录
    // process_train->setWorkingDirectory("D:/lyh/GUI207_V2.0/db/bashs/hrrp");
    // process_train->start(pythonPath, params);
}

void ModelTrain::showTrianResult(){
    timeRestEdit->setText("训练完毕");
    trainProgressBar->setValue(100);
    if(modelTypeId==0){
        trainImg->setPixmap(QPixmap(dataRoot+"/training_accuracy.jpg"));
        valImg->setPixmap(QPixmap(dataRoot+"/verification_accuracy.jpg"));
        confusionMat->setPixmap(QPixmap(dataRoot+"/confusion_matrix.jpg"));
    }
    else if(modelTypeId==1){
        trainImg->setPixmap(QPixmap(dataRoot+"/training_accuracy.jpg"));
        valImg->setPixmap(QPixmap(dataRoot+"/features_Accuracy.jpg"));
        confusionMat->setPixmap(QPixmap(dataRoot+"/confusion_matrix.jpg"));
    }
}

void ModelTrain::stopTrain(){
    showLog=false;
    timeRestEdit->setText("停止训练");
    OutShow->append("===========================Train Stoping===========================");
    if(process_train->state()==QProcess::Running){
        process_train->close();
        process_train->kill();
    }
}

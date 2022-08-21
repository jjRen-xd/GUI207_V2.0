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

void ModelTrain::startTrain(int modelType, QString dataDir, QString bathSize, QString maxEpoch){
  // TODO add code here
    QString cmd;
    modelTypeId = modelType;
    switch(modelType){
        case 0:cmd = "activate TF2 && python ../db/bashs/hrrp/train.py"
                    " --data_dir "+dataDir+" --batch_size "+bathSize+" --max_epochs "+maxEpoch;break;
        case 1:cmd = "activate TF2 && python ../db/bashs/afs/train.py"
                    " --data_dir "+dataDir+" --batch_size "+bathSize+" --max_epochs "+maxEpoch;break;
    }
    dataRoot = dataDir;
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

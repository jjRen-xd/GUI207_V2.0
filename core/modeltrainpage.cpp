#include "modelTrainPage.h"

ModelTrainPage::ModelTrainPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo,
                               ModelInfo *globalModelInfo, ModelDock *modelDock):
    ui(main_ui),terminal(bash_terminal),datasetInfo(globalDatasetInfo),
    modelInfo(globalModelInfo), modelDock(modelDock){

    ui->fewShotWidget->setVisible(false);
    ui->oldClassNumEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^[1-9][0-9]{1,3}$")));
    ui->dataNumPercentEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^0\\.[0-9]{0,1}[1-9]$")));
    ui->preTrainEpochEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^[1-9][0-9]{1,3}[1-9]$")));
    ui->trainEpochEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^[1-9][0-9]{1,4}[1-9]$")));
    ui->trainBatchEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^[1-9][0-9]{1,4}[1-9]$")));
    ui->saveModelNameEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("[a-zA-Z0-9_]+$")));

    processTrain = new QProcess();
    refreshGlobalInfo();

    connect(processTrain, &QProcess::readyReadStandardOutput, this, &ModelTrainPage::monitorTrainProcess);
    connect(ui->startTrainButton, &QPushButton::clicked, this, &ModelTrainPage::startTrain);
    connect(ui->stopTrainButton,  &QPushButton::clicked, this, &ModelTrainPage::stopTrain);
    connect(ui->modelTypeBox, &QComboBox::currentIndexChanged, this, &ModelTrainPage::changeTrainType);

}


void ModelTrainPage::refreshGlobalInfo(){
    if(QString::fromStdString(datasetInfo->selectedName)!=""){
        ui->choosedDataText->setText(QString::fromStdString(datasetInfo->selectedName));
        this->choicedDatasetPATH = QString::fromStdString(datasetInfo->getAttri(datasetInfo->selectedType,datasetInfo->selectedName,"PATH"));
    }
    else{
        ui->choosedDataText->setText("未指定");
        this->choicedDatasetPATH = "";
    }
}


void ModelTrainPage::changeTrainType(){
    int modelType=ui->modelTypeBox->currentIndex();

    ui->tabWidget->removeTab(0);
    ui->tabWidget->removeTab(1);
    ui->tabWidget->removeTab(2);
    if(modelType==0){
        ui->tabWidget->addTab(ui->tab,"训练集准确率");
        ui->tabWidget->addTab(ui->tab_2,"验证集准确率");
        ui->tabWidget->addTab(ui->tab_3,"混淆矩阵");
        ui->tabWidget->setCurrentIndex(0);
    }
    else if(modelType==1){
        ui->tabWidget->addTab(ui->tab_2,"特征关联性能");
        ui->tabWidget->addTab(ui->tab_3,"混淆矩阵");
        ui->tabWidget->setCurrentIndex(0);
    }
//    else if(modelType==2){
//        ui->tabWidget->addTab(ui->tab_2,"鐗瑰緛鍑嗙‘鐜?);
//        ui->tabWidget->addTab(ui->tab_3,"娣锋穯鐭╅樀");
//        ui->stackedWidget->setCurrentIndex(1);
//    }
}


void ModelTrainPage::startTrain(){
    if(choicedDatasetPATH==""){
        QMessageBox::warning(NULL,"错误","未选择训练数据集!");
        return;
    }
    QDateTime dateTime(QDateTime::currentDateTime());
    time = dateTime.toString("yyyy-MM-dd-hh-mm-ss");
    trainModelType = ui->modelTypeBox->currentIndex();
    batchSize = ui->trainBatchEdit->text();
    epoch = ui->trainEpochEdit->text();
    saveModelName = ui->saveModelNameEdit->text();
    if(trainModelType<2){
        if(batchSize=="" || epoch=="" || saveModelName==""){
            QMessageBox::warning(NULL,"错误","请检查各项文本框中训练参数是否正确配置!");
            return;
        }
        uiInitial();
        switch(trainModelType){
            case 0:cmd = "activate TF2 && python ../../api/bashs/hrrp/train.py --data_dir "+choicedDatasetPATH+ \
                         " --time "+time+" --batch_size "+batchSize+" --max_epochs "+epoch+" --model_name "+saveModelName;break;
            case 1:cmd = "activate TF2 && python ../../api/bashs/afs/train.py --data_dir "+choicedDatasetPATH+ \
                         " --time "+time+" --batch_size "+batchSize+" --max_epochs "+epoch+" --model_name "+saveModelName;break;
        }
    }
    else if(trainModelType==2){

    }
    qDebug() << cmd;
    execuCmd(cmd);
}

void ModelTrainPage::uiInitial(){
    ui->startTrainButton->setEnabled(true);
    ui->trainProgressBar->setMaximum(100);
    ui->trainProgressBar->setValue(0);
    ui->textBrowser->clear();
    ui->train_img->clear();
    ui->val_img->clear();
    ui->confusion_mat->clear();
}

void ModelTrainPage::execuCmd(QString cmd){
  // TODO add code here
    if(processTrain->state()==QProcess::Running){
        processTrain->close();
        processTrain->kill();
    }
    showLog=false;
    ui->startTrainButton->setEnabled(false);
    processTrain->setProcessChannelMode(QProcess::MergedChannels);
    processTrain->start("cmd.exe");
    ui->textBrowser->setText("===================Train Starting===================");
    ui->trainProgressBar->setMaximum(0);
    ui->trainProgressBar->setValue(0);
    processTrain->write(cmd.toLocal8Bit() + '\n');
}

void ModelTrainPage::stopTrain(){
    QString cmd="\\x03";
    processTrain->write(cmd.toLocal8Bit() + '\n');
    showLog=false;
    ui->startTrainButton->setEnabled(true);
    ui->trainProgressBar->setMaximum(100);
    ui->trainProgressBar->setValue(0);
    ui->textBrowser->append("===================Train Stoping===================");
    if(processTrain->state()==QProcess::Running){
        processTrain->close();
        processTrain->kill();
    }
}

void ModelTrainPage::monitorTrainProcess(){
    /* 读取终端输出并显示 */
    QByteArray cmdOut = processTrain->readAllStandardOutput();
    if(!cmdOut.isEmpty()){
        QString logs=QString::fromLocal8Bit(cmdOut);
        QStringList lines = logs.split("\n");
        int len=lines.length();
        for(int i=0;i<len;i++){
            QStringList Infos = lines[i].simplified().split(" ");
            if(lines[i].contains("Train Ended",Qt::CaseSensitive)){
                ui->textBrowser->append("===================Train Ended===================");
                showLog=false;
                ui->startTrainButton->setEnabled(true);
            //    导入训练好的模型至系统
//                modelDock->importModelAfterTrain(modelTypes[trainModelType], saveModelName);
                showTrianResult();
                if(processTrain->state()==QProcess::Running){
                    processTrain->close();
                    processTrain->kill();
                }
            }
            else if(lines[i].contains(cmd,Qt::CaseSensitive)){
                showLog=true;
            }
            else if(lines[i].contains("Train Failed",Qt::CaseSensitive)){
                ui->startTrainButton->setEnabled(true);
                QDateTime dateTime(QDateTime::currentDateTime());
                ui->textBrowser->append(dateTime.toString("yyyy-MM-dd-hh-mm-ss")+" - 网络模型训练出错：");
                for(i++;i<len;i++){
                    ui->textBrowser->append(lines[i]);
                }
                stopTrain();
            }
            else if(showLog){
                ui->textBrowser->append(lines[i]);
            }
        }
    }
    ui->textBrowser->update();
}


void ModelTrainPage::showTrianResult(){
    ui->trainProgressBar->setMaximum(100);
    ui->trainProgressBar->setValue(100);
    //TODO
    QDir dir("../../db/trainLogs");
    QStringList dirList = dir.entryList(QDir::Dirs);
    foreach (auto dir , dirList){
        if(dir.contains(time)){
            QString wordir    = "../../db/trainLogs/"+dir;
            if(trainModelType==0){
                ui->train_img->setPixmap(QPixmap(wordir+"/training_accuracy.jpg"));
                ui->val_img->setPixmap(QPixmap(wordir+"/verification_accuracy.jpg"));
                ui->confusion_mat->setPixmap(QPixmap(wordir+"/confusion_matrix.jpg"));
            }
            else if(trainModelType==1){
                ui->val_img->setPixmap(QPixmap(wordir+"/features_Accuracy.jpg"));
                ui->confusion_mat->setPixmap(QPixmap(wordir+"/confusion_matrix.jpg"));
            }
            else if(trainModelType==2){
//                ui->val_img->setPixmap(QPixmap(wordir+"/features_Accuracy.jpg"));
//                ui->confusion_mat->setPixmap(QPixmap(wordir+"/confusion_matrix.jpg"));
            }
        }
    }
}

void ModelTrainPage::editModelFile(){
    int modelType=ui->modelTypeBox->currentIndex();
    QString modelFilePath;
    switch(modelType){
        case 0:modelFilePath="../../api/bashs/hrrp/train.py";break;
        case 1:modelFilePath="../../api/bashs/afs/afs_model.py";break;
        case 2:modelFilePath="../../api/bashs/incremental/model.py";break;
    }
    QString commd="gvim " + modelFilePath;
    system(commd.toStdString().c_str());
}

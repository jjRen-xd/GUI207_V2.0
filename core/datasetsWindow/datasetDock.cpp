#include "datasetDock.h"

#include <QStandardItemModel>
#include <QFileDialog>
#include <QMessageBox>
#include <time.h>

using namespace std;

DatasetDock::DatasetDock(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo):
    ui(main_ui),
    terminal(bash_terminal),
    datasetInfo(globalDatasetInfo)
{
    // 数据集导入事件
    connect(ui->action_import_HRRP, &QAction::triggered, this, [this]{importDataset("HRRP");});
    connect(ui->action_import_RCS, &QAction::triggered, this, [this]{importDataset("RCS");});
    connect(ui->action_import_RADIO, &QAction::triggered, this, [this]{importDataset("RADIO");});
    connect(ui->action_import_IMAGE, &QAction::triggered, this, [this]{importDataset("IMAGE");});

    // 数据集删除事件
    connect(ui->action_Delete_dataset, &QAction::triggered, this, &DatasetDock::deleteDataset);

    // 当前数据集预览树按类型成组
    this->datasetTreeViewGroup["HRRP"] = ui->treeView_HRRP;
    this->datasetTreeViewGroup["RCS"] = ui->treeView_RCS;
    this->datasetTreeViewGroup["RADIO"] = ui->treeView_RADIO;
    this->datasetTreeViewGroup["IMAGE"] = ui->treeView_IMAGE;

    // 数据集信息预览label按属性成组 std::map<std::string, QLabel*>
    this->attriLabelGroup["datasetName"] = ui->label_datasetDock_datasetName;
    this->attriLabelGroup["PATH"] = ui->label_datasetDock_PATH;
    this->attriLabelGroup["freq"] = ui->label__datasetDock_freq;
    this->attriLabelGroup["claNum"] = ui->label_datasetDock_claNum;
    this->attriLabelGroup["targetNumEachCla"] = ui->label_datasetDock_targetNumEachCla;
    this->attriLabelGroup["note"] = ui->label_datasetDock_note;

    // 显示图表成组
    chartGroup.push_back(ui->label_datasetDock_chartView01);
    chartGroup.push_back(ui->label_datasetDock_chartView02);
    chartInfoGroup.push_back(ui->label_datasetDock_chart1);
    chartInfoGroup.push_back(ui->label_datasetDock_chart2);

    // 初始化TreeView
    reloadTreeView();

    for(auto &currTreeView: datasetTreeViewGroup){
        //链接节点点击事件
        //重复绑定信号和槽函数导致bug弹窗已修复
        connect(currTreeView.second, SIGNAL(clicked(QModelIndex)), this, SLOT(treeItemClicked(QModelIndex)));
    }
}

DatasetDock::~DatasetDock(){


}


void DatasetDock::importDataset(string type){
    QString rootPath = QFileDialog::getExistingDirectory(NULL,"请选择数据集目录","./",QFileDialog::ShowDirsOnly);
    if(rootPath == ""){
        QMessageBox::warning(NULL,"提示","数据集打开失败!");
        return;
    }
    QString datasetName = rootPath.split('/').last();

    vector<string> allXmlNames;
    dirTools->getFiles(allXmlNames, ".xml",rootPath.toStdString());
    if (allXmlNames.empty()){
        terminal->print("添加数据集成功，但该数据集没有说明文件.xml！");
        QMessageBox::warning(NULL, "添加数据集", "添加数据集成功，但该数据集没有说明文件.xml！");
    }
    else{
        QString xmlPath = rootPath + "/" + QString::fromStdString(allXmlNames[0]);
        datasetInfo->addItemFromXML(xmlPath.toStdString());
        terminal->print("添加数据集成功:"+xmlPath);
        QMessageBox::information(NULL, "添加数据集", "添加数据集成功！");
    }
    this->datasetInfo->modifyAttri(type, datasetName.toStdString(),"PATH", rootPath.toStdString());
    this->reloadTreeView();
    qDebug()<<"import and writeToXML";
    this->datasetInfo->writeToXML(datasetInfo->defaultXmlPath);
}

void DatasetDock::deleteDataset(){
    if(previewType==""||previewName==""){
        QMessageBox::information(NULL, "错误", "未选择任何数据集!");
        return;
    }
    QMessageBox confirmMsg;
    confirmMsg.setText(QString::fromStdString("确认要删除数据集："+previewType+"->"+previewName));
    confirmMsg.setStandardButtons(QMessageBox::No | QMessageBox::Yes);
    if(confirmMsg.exec() == QMessageBox::Yes){
        this->datasetInfo->deleteItem(previewType,previewName);
        this->reloadTreeView();
        qDebug()<<"delete and writeToXML";
        this->datasetInfo->writeToXML(datasetInfo->defaultXmlPath);
        terminal->print(QString::fromStdString("数据集删除成功:"+previewName));
        QMessageBox::information(NULL, "删除数据集", "数据集删除成功！");
    }

    return;
}

void DatasetDock::reloadTreeView(){
    for(auto &currTreeView: datasetTreeViewGroup){
        // 不可编辑节点
        currTreeView.second->setEditTriggers(QAbstractItemView::NoEditTriggers);
        currTreeView.second->setHeaderHidden(true);
        // 构建节点
        vector<string> datasetNames = datasetInfo->getNamesInType(currTreeView.first);
        QStandardItemModel *treeModel = new QStandardItemModel(datasetNames.size(),1);
        int idx = 0;
        for(auto &datasetName: datasetNames){
            QStandardItem *nameItem = new QStandardItem(datasetName.c_str());
            treeModel->setItem(idx, 0, nameItem);
            idx += 1;
        }
        currTreeView.second->setModel(treeModel);
        //链接节点点击事件
//        connect(currTreeView.second, SIGNAL(clicked(QModelIndex)), this, SLOT(treeItemClicked(QModelIndex)));
    }
}

void DatasetDock::treeItemClicked(const QModelIndex &index){
    // 获取点击预览数据集的类型和名称

    string clickedType = ui->tabWidget_datasetType->currentWidget()->objectName().split("_")[1].toStdString();
    string clickedName = datasetTreeViewGroup[clickedType]->model()->itemData(index).values()[0].toString().toStdString();
    this->previewName = clickedName;
    this->previewType = clickedType;

    // 显示数据集预览属性信息
    map<string,string> attriContents = datasetInfo->getAllAttri(previewType, previewName);
    for(auto &currAttriLabel: attriLabelGroup){
        currAttriLabel.second->setText(QString::fromStdString(attriContents[currAttriLabel.first]));
    }
    // 获取所有类别子文件夹
    string rootPath = datasetInfo->getAttri(previewType, previewName, "PATH");
    ui->datadirEdit->setText(QString::fromStdString(rootPath));
    vector<string> subDirNames;
    if(dirTools->getDirs(subDirNames, rootPath)){
        //先确定数据集中数据文件的format
        vector<string> allFileWithTheClass;
        QString dataFileFormat;
        if(dirTools->getAllFiles(allFileWithTheClass,rootPath +"/"+subDirNames[0])){//当前的逻辑是根据数据集第一类数据的文件夹下的数据文件格式确定整个数据集数据文件格式。
            dataFileFormat = QString::fromStdString(allFileWithTheClass[2]).split('.').last();//因为前两个是.和..
        }else{qDebug()<<QString::fromStdString(rootPath +"/"+subDirNames[0])<<"此路径下没有带后缀的数据文件,可能出错";}
        //给Cache加上数据集的数据文件类型  这意味着不点dataseTree
//        this->datasetInfo->modifyAttri(previewType, previewName ,"dataFileFormat", dataFileFormat.toStdString());
        //qDebug()<<"数据集的dataFileFormat："<<QString::fromStdString(datasetInfo->getAttri(previewType, previewName, "dataFileFormat"));
//        qDebug() << "treeItemClicked and writeToXML :" << QString::fromStdString(previewType) << QString::fromStdString(previewName) << QString::fromStdString(dataFileFormat.toStdString());
//        this->datasetInfo->writeToXML(datasetInfo->defaultXmlPath);
        for(int i = 0; i<chartGroup.size(); i++){
            srand((unsigned)time(NULL));
            // 随机选取类别
            string choicedClass = subDirNames[(rand()+i)%subDirNames.size()];
            string classPath = rootPath +"/"+ choicedClass;
            // 随机选取数据
            if(dataFileFormat==QString::fromStdString("txt")){//可以被淘汰的
                vector<string> allTxtFile;
                if(dirTools->getFiles(allTxtFile, ".txt", classPath)){
                    string choicedFile = allTxtFile[(rand())%allTxtFile.size()];
                    QString txtFilePath = QString::fromStdString(classPath + "/" + choicedFile);
                    choicedFile = QString::fromStdString(choicedFile).split(".").first().toStdString();
                    // 绘图
                    Chart *previewChart = new Chart(chartGroup[i],"HRRP(Ephi),Polarization HP(1)[Magnitude in dB]",txtFilePath);
                    previewChart->drawImage(chartGroup[i],"HRRP",0);
                    chartInfoGroup[i]->setText(QString::fromStdString(choicedClass+":"+choicedFile));
                }
            }
            else if(dataFileFormat==QString::fromStdString("mat")){
                vector<string> allMatFile;
                if(dirTools->getFiles(allMatFile, ".mat", classPath)){
                    int randomIdx = (rand())%5000;
//                    std::cout<<"(DatasetDock::treeItemClicked)randomIdx:"<<randomIdx;
                    //绘图
                    QString matFilePath = QString::fromStdString(classPath + "/" + allMatFile[0]);
                    QString chartTitle="Temporary Title";
                    if(clickedType=="HRRP") chartTitle="HRRP(Ephi),Polarization HP(1)[Magnitude in dB]";
                    else if (clickedType=="RADIO") chartTitle="RADIO Temporary Title";
                    Chart *previewChart = new Chart(chartGroup[i],chartTitle,matFilePath);
                    previewChart->drawImage(chartGroup[i],clickedType,randomIdx);
                    chartInfoGroup[i]->setText(QString::fromStdString(choicedClass+":"+matFilePath.split(".").first().toStdString()));


                }
            }
            else{
                qDebug()<<dataFileFormat<<"没有对应此种数据文件类型的解析！";///////////////////////////////
                QMessageBox::information(NULL, "数据集错误", "没有对应此种数据文件类型的解析！");
            }
        }
    }
}

#include <Python.h>
#include "datasetDock.h"
#include <cstdlib>
#include <QStandardItemModel>
#include <QFileDialog>
#include <QMessageBox>
#include <time.h>


#pragma comment(lib,"ToHRRP.lib")

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
    this->datasetTreeViewGroup["RADIO"] = ui->treeView_RADIO;
    this->datasetTreeViewGroup["HRRP"] = ui->treeView_HRRP;
    this->datasetTreeViewGroup["FEATURE"] = ui->treeView_FEATURE;
    this->datasetTreeViewGroup["RCS"] = ui->treeView_RCS;
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
        connect(currTreeView.second, SIGNAL(clicked(QModelIndex)), this, SLOT(treeItemClicked(QModelIndex)));}
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
    this->datasetInfo->writeToXML(datasetInfo->defaultXmlPath);
}

void DatasetDock::deleteDataset(){
    QMessageBox confirmMsg;
    confirmMsg.setText(QString::fromStdString("确认要删除数据集："+previewType+"->"+previewName));
    confirmMsg.setStandardButtons(QMessageBox::No | QMessageBox::Yes);
    if(confirmMsg.exec() == QMessageBox::Yes){
        this->datasetInfo->deleteItem(previewType,previewName);
        this->reloadTreeView();
        this->datasetInfo->writeToXML(datasetInfo->defaultXmlPath);
        terminal->print(QString::fromStdString("数据集删除成功:"+previewName));
        QMessageBox::information(NULL, "删除数据集", "数据集删除成功！");
    }
    else{}

    return;
}

void DatasetDock::onActionTransRadio(){//Radio101 to Hrrp128
    QModelIndex curIndex = datasetTreeViewGroup["RADIO"]->currentIndex();
    QStandardItem *currItem = static_cast<QStandardItemModel*>(datasetTreeViewGroup["RADIO"]->model())->itemFromIndex(curIndex);
    string clickedName = currItem->data(0).toString().toStdString();
    if(curIndex.isValid()){
        qDebug()<<"onActionView_here  "<<QString::fromStdString(clickedName);
    }
    string sourceDataSetPath = datasetInfo->getAttri("RADIO", clickedName, "PATH");
    //创建平行的datasetpath文件夹
    string destDataSetPath=sourceDataSetPath+"_HRRP";
    if(0 == opendir(destDataSetPath.c_str())){//destPath目录不存在就建立一个
        if (CreateDirectoryA(destDataSetPath.c_str(), NULL))  printf("Create Dir Failed...\n");
        else    printf("Creaat Dir %s Successed...\n", destDataSetPath.c_str());
    }

    SearchFolder *dirTools = new SearchFolder();
    // 寻找子文件夹 WARN:数据集的路径一定不能包含汉字 否则遍历不到文件路径
    std::vector<std::string> sourceSubDirs;
    dirTools->getDirs(sourceSubDirs, sourceDataSetPath);
    for(auto &subDir: sourceSubDirs){
        //创建平行的subDir文件夹
        if(0 == opendir((destDataSetPath+"/"+subDir).c_str())){//destPath目录不存在就建立一个
            if (CreateDirectoryA((destDataSetPath+"/"+subDir).c_str(), NULL))  printf("Create Dir Failed...\n");
            else    printf("Creaat Dir %s Successed...\n", (destDataSetPath+"/"+subDir).c_str());
        }
        // 寻找每个子文件夹下的样本文件
        std::vector<std::string> fileNames;
        std::string subDirPath = sourceDataSetPath+"/"+subDir;
        dirTools->getFiles(fileNames, ".mat", subDirPath);
        std:: string sourcePath,destPath;

        for(auto &fileName: fileNames){
            sourcePath=subDirPath+"/"+fileName;
            destPath=destDataSetPath+"/"+subDir+"/"+"hrrp128";  //WARRRRRRRRRRING!
            if (!ToHRRPInitialize()) {qDebug()<<"Could not initialize ToHRRP!";return;}
            mwArray sPath(sourcePath.c_str());
            mwArray dPath(destPath.c_str());
            mwArray tranF("0");
            ToHRRP(1,tranF,sPath,dPath);//调用
        }
    }
    QMessageBox::information(NULL, "转换数据集", "Radio数据集转换成功！");
    this->datasetInfo->modifyAttri("HRRP", clickedName+"_HRRP","PATH", destDataSetPath);
    this->reloadTreeView();
    this->datasetInfo->writeToXML(datasetInfo->defaultXmlPath);
}

void DatasetDock::onTreeViewMenuRequestedRadio(const QPoint &pos){
    QModelIndex curIndex = datasetTreeViewGroup["RADIO"]->indexAt(pos);
    if (curIndex.isValid()){ // 右键选中了有效index
        QIcon transIcon = QApplication::style()->standardIcon(QStyle::SP_DesktopIcon);
        // 创建菜单
        QMenu menu;
        menu.addAction(transIcon, tr("转换至HRRP"), this, &DatasetDock::onActionTransRadio);
//        menu.addSeparator();
//        menu.addAction(test, tr("测试"), this, &DatasetDock::onActionTest);
        menu.exec(QCursor::pos());
    }
}

void DatasetDock::onActionExtractFea(){
    //system("python D:\\lyh\\GUI207_V2.0\\db\\datasets\\feature_extraction.py --data_path D:\\lyh\\GUI207_V2.0\\db\\datasets\\falseHRRPmat_1x128 --save_path D:\\lyh\\GUI207_V2.0\\db\\datasets\\fea_falseHRRPmat_1x128");
    QModelIndex curIndex = datasetTreeViewGroup["HRRP"]->currentIndex();
    QStandardItem *currItem = static_cast<QStandardItemModel*>(datasetTreeViewGroup["HRRP"]->model())->itemFromIndex(curIndex);
    string clickedName = currItem->data(0).toString().toStdString();
    std::string sourceDataSetPath = datasetInfo->getAttri("HRRP", clickedName, "PATH");
    std::string destDataSetPath=sourceDataSetPath+"_FEATURE";
    std::string commd="python ../../db/datasets/feature_extraction.py --data_path "+sourceDataSetPath+" --save_path "+destDataSetPath;
    qDebug()<<QString::fromStdString(commd);
    system(commd.c_str());
//    Py_Initialize();
//    PyObject* pModule = NULL;
//    PyObject* pFunc = NULL;

//    PyRun_SimpleString("import sys");
//    //PyRun_SimpleString("sys.path.append('.\')");
//    //PyRun_SimpleString("sys.path.append('')");

//    pModule = PyImport_ImportModule("feature_extraction");
//    if( pModule == NULL ){
//        qDebug()<<"(DatasetDock::onActionExtractFea)pModule not found" ;
//        return ;
//    }
//    pFunc = PyObject_GetAttrString(pModule, "doinfer");
//    PyObject* pArgs = PyTuple_New(2);

//    PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", sourceDataSetPath.c_str()));
//    PyTuple_SetItem(pArgs, 1, Py_BuildValue("s", destDataSetPath.c_str()));
//    PyObject* pReturn = PyObject_CallObject(pFunc, pArgs);//Excute
//    int nResult;
//    // i表示转换成int型变量。
//    // PyArg_Parse的最后一个参数，必须加上“&”符号
//    PyArg_Parse(pReturn, "i", &nResult);
//    qDebug()<< "(DatasetDock::onActionExtractFea)feature_extraction return result is " << nResult ;
//    Py_Finalize();

    QMessageBox::information(NULL, "特征提取", "HRRP数据集特征提取成功！");
    this->datasetInfo->modifyAttri("FEATURE", clickedName+"_FEATURE","PATH", destDataSetPath);
    this->reloadTreeView();
}

void DatasetDock::onTreeViewMenuRequestedHrrp(const QPoint &pos){
    QModelIndex curIndex = datasetTreeViewGroup["HRRP"]->indexAt(pos);
    if (curIndex.isValid()){ // 右键选中了有效index
        QIcon transIcon = QApplication::style()->standardIcon(QStyle::SP_DesktopIcon);
        // 创建菜单
        QMenu menu;
        menu.addAction(transIcon, tr("提取特征"), this, &DatasetDock::onActionExtractFea);
//        menu.addSeparator();
//        menu.addAction(test, tr("测试"), this, &DatasetDock::onActionTest);
        menu.exec(QCursor::pos());
    }
}

void DatasetDock::reloadTreeView(){
    for(auto &currTreeView: datasetTreeViewGroup){
        // 不可编辑节点
        currTreeView.second->setEditTriggers(QAbstractItemView::NoEditTriggers);
        currTreeView.second->setHeaderHidden(true);
        // 构建节点
        vector<string> datasetNames = datasetInfo->getNamesInType(currTreeView.first);//得到指定类别的数据集
        QStandardItemModel *treeModel = new QStandardItemModel(datasetNames.size(),1);
        int idx = 0;
        for(auto &datasetName: datasetNames){
            QStandardItem *nameItem = new QStandardItem(datasetName.c_str());
            treeModel->setItem(idx, 0, nameItem);
            idx += 1;
        }
        currTreeView.second->setModel(treeModel);
        //链接节点右键事件
        if(currTreeView.first=="RADIO"){
            currTreeView.second->setContextMenuPolicy(Qt::CustomContextMenu);
            connect(currTreeView.second, &QTreeView::customContextMenuRequested, this, &DatasetDock::onTreeViewMenuRequestedRadio);
        }
        if(currTreeView.first=="HRRP"){
            currTreeView.second->setContextMenuPolicy(Qt::CustomContextMenu);
            connect(currTreeView.second, &QTreeView::customContextMenuRequested, this, &DatasetDock::onTreeViewMenuRequestedHrrp);
        }
        //connect(currTreeView.second, SIGNAL(clicked(QModelIndex)), this, SLOT(treeItemClicked(QModelIndex)));
    }

}

void DatasetDock::treeItemClicked(const QModelIndex &index){
    // 获取点击预览数据集的类型和名称
    string clickedType = ui->tabWidget_datasetType->currentWidget()->objectName().split("_")[1].toStdString();
    string clickedName = datasetTreeViewGroup[clickedType]->model()->itemData(index).values()[0].toString().toStdString();
    this->previewName = clickedName;
    this->previewType = clickedType;
    //qDebug()<<"(DatasetDock::treeItemClicked)点击预览数据集的类型和名称 "<<QString::fromStdString(clickedType)<<"  "<<QString::fromStdString(clickedName);

    // 显示数据集预览属性信息
    map<string,string> attriContents = datasetInfo->getAllAttri(previewType, previewName);
    for(auto &currAttriLabel: attriLabelGroup){
        currAttriLabel.second->setText(QString::fromStdString(attriContents[currAttriLabel.first]));
    }
    // 获取所有类别子文件夹
    string rootPath = datasetInfo->getAttri(previewType, previewName, "PATH");
    vector<string> subDirNames;
    if(dirTools->getDirs(subDirNames, rootPath)){
        //先确定数据集中数据文件的format
        vector<string> allFileWithTheClass;
        QString dataFileFormat;
        if(dirTools->getAllFiles(allFileWithTheClass,rootPath +"/"+subDirNames[0])){//当前的逻辑是根据数据集第一类数据的文件夹下的数据文件格式确定整个数据集数据文件格式。
            dataFileFormat = QString::fromStdString(allFileWithTheClass[2]).split('.').last();//因为前两个是.和..
        }else{qDebug()<<QString::fromStdString(rootPath +"/"+subDirNames[0])<<"此路径下没有带后缀的数据文件,可能出错";}
        //给Cache加上数据集的数据文件类型  这意味着不点dataseTree
        this->datasetInfo->modifyAttri(previewType, previewName ,"dataFileFormat", dataFileFormat.toStdString());
        //qDebug()<<"数据集的dataFileFormat："<<QString::fromStdString(datasetInfo->getAttri(previewType, previewName, "dataFileFormat"));
        this->datasetInfo->writeToXML(datasetInfo->defaultXmlPath);
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
                    std::cout<<"(DatasetDock::treeItemClicked)randomIdx:"<<randomIdx;
                    //绘图
                    QString matFilePath = QString::fromStdString(classPath + "/" + allMatFile[0]);
                    QString chartTitle="Temporary Title";
                    if(clickedType=="HRRP") chartTitle="HRRP(Ephi),Polarization HP(1)[Magnitude in dB]";
                    else if (clickedType=="RADIO") chartTitle="RADIO Temporary Title";
                    else if (clickedType=="FEATURE") chartTitle="Feture Temporary Title";
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

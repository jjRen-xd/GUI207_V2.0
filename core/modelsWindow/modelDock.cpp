#include "modelDock.h"

#include <QStandardItemModel>
#include <QFileDialog>
#include <QMessageBox>

using namespace std;

ModelDock::ModelDock(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, ModelInfo *globalModelInfo):
    ui(main_ui),
    terminal(bash_terminal),
    modelInfo(globalModelInfo)
{
    // 模型导入事件
    connect(ui->action_importModel_TRA_DL, &QAction::triggered, this, [this]{importModel("TRA_DL");});
    connect(ui->action_importModel_FEA_RELE, &QAction::triggered, this, [this]{importModel("FEA_RELE");});
    connect(ui->action_importModel_FEA_OPTI, &QAction::triggered, this, [this]{importModel("FEA_OPTI");});
    connect(ui->action_importModel_INCRE, &QAction::triggered, this, [this]{importModel("INCRE");});
    // 模型删除事件
    connect(ui->action_dele_model, &QAction::triggered, this, &ModelDock::deleteModel);

    // 预览属性标签成组
    attriLabelGroup["name"] = ui->label_modelDock_name;
    attriLabelGroup["algorithm"] = ui->label_modelDock_algorithm;
    attriLabelGroup["accuracy"] = ui->label_modelDock_accuracy;
    attriLabelGroup["trainDataset"] = ui->label_modelDock_trainDataset;
    attriLabelGroup["trainEpoch"] = ui->label_modelDock_trainEpoch;
    attriLabelGroup["trainLR"] = ui->label_modelDock_trainLR;
    attriLabelGroup["framework"] = ui->label_modelDock_framework;
    attriLabelGroup["PATH"] = ui->label_modelDock_PATH;
    attriLabelGroup["note"] = ui->label_modelDock_note;

    // TreeView
    this->modelTreeView = ui->TreeView_modelDock;
    // 初始化TreeView
    reloadTreeView();
}


void ModelDock::reloadTreeView(){
    // 不可编辑节点
    modelTreeView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    modelTreeView->setHeaderHidden(true);
    // 构建节点
    vector<string> modelTypes = modelInfo->getTypes();
    QStandardItemModel *typeTreeModel = new QStandardItemModel(modelTypes.size(),1);
    int idx = 0;
    for(auto &type: modelTypes){
        QStandardItem *typeItem = new QStandardItem(modelInfo->var2TypeName[type].c_str());
        typeTreeModel->setItem(idx, 0, typeItem);   // 节点链接

        // 构建子节点
        vector<string> modelNames = modelInfo->getNamesInType(type);
        for(auto &name: modelNames){
            QStandardItem *nameItem = new QStandardItem(QString::fromStdString(name));
            typeItem->appendRow(nameItem);
        }

        idx += 1;
    }
    modelTreeView->setModel(typeTreeModel);
    //链接节点点击事件
    connect(modelTreeView, SIGNAL(clicked(QModelIndex)), this, SLOT(treeItemClicked(QModelIndex)));
    terminal->print("i am here!");
}


void ModelDock::treeItemClicked(const QModelIndex &index){
    // 获取点击预览模型的类型和名称
    QStandardItem *currItem = static_cast<QStandardItemModel*>(modelTreeView->model())->itemFromIndex(index);     //返回给定index的条目
    string clickedName = currItem->data(0).toString().toStdString();             //获取该条目的值
    if(currItem->hasChildren()){    // 点击父节点直接返回
        return;
    }
    string clickedType = modelInfo->typeName2Var[currItem->parent()->data(0).toString().toStdString()];
    this->previewName = clickedName;
    this->previewType = clickedType;
    terminal->print(QString::fromStdString(clickedName));
    terminal->print(QString::fromStdString(clickedType));

    // 更新预览属性参数
    map<string,string> attriContents = modelInfo->getAllAttri(previewType, previewName);
    for(auto &currAttriLabel: attriLabelGroup){
        currAttriLabel.second->setText(QString::fromStdString(attriContents[currAttriLabel.first]));
    }
}


void ModelDock::importModel(string type){
    QString modelPath = QFileDialog::getOpenFileName(NULL, "打开网络模型文件", "../../db/models/");
    if(modelPath == ""){
        QMessageBox::warning(NULL, "提示", "文件打开失败!");
        return;
    }
    QString modelName = modelPath.split('/').last();
    string savePath = modelPath.toStdString();
    QString rootPath = modelPath.remove(modelPath.length()-modelName.length()-1, modelPath.length());
    QString xmlPath;

    vector<string> allXmlNames;
    bool existXml = false;
    dirTools->getFiles(allXmlNames, ".xml",rootPath.toStdString());
    // 寻找与.pt文件相同的.xml文件
    for(auto &xmlName: allXmlNames){
        if(QString::fromStdString(xmlName).split(".").first() == modelName.split(".").first()){
            existXml = true;
            xmlPath = rootPath + "/" + QString::fromStdString(xmlName);
        }
    }
    if(existXml){
        modelInfo->addItemFromXML(xmlPath.toStdString());

        terminal->print("添加模型成功:"+xmlPath);
        QMessageBox::information(NULL, "添加模型", "添加模型成功！");
    }
    else{
        terminal->print("添加模型成功，但该模型没有说明文件.xml！");
        QMessageBox::warning(NULL, "添加模型", "添加模型成功，但该模型没有说明文件.xml！");
    }

    this->modelInfo->modifyAttri(type, modelName.toStdString(), "PATH", savePath);
    this->reloadTreeView();
    this->modelInfo->writeToXML(modelInfo->defaultXmlPath);
}


void ModelDock::deleteModel(){
    QMessageBox confirmMsg;
    confirmMsg.setText(QString::fromStdString("确认要删除模型："+previewType+"->"+previewName));
    confirmMsg.setStandardButtons(QMessageBox::No | QMessageBox::Yes);
    if(confirmMsg.exec() == QMessageBox::Yes){
        this->modelInfo->deleteItem(previewType,previewName);
        this->reloadTreeView();
        this->modelInfo->writeToXML(modelInfo->defaultXmlPath);
        terminal->print(QString::fromStdString("模型删除成功:"+previewName));
        QMessageBox::information(NULL, "删除模型", "模型删除成功！");
    }
    else{}

    return;
}

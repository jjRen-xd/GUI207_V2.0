#include <QShortcut>

#include "MainWindow.h"
#include "ui_MainWindow.h"

MainWindow::MainWindow(QWidget *parent): QMainWindow(parent){
    ui = new Ui::MainWindow();
	ui->setupUi(this);
	QRibbon::install(this);

    // 全局数据记录设置
    this->globalDatasetInfo = new DatasetInfo("../../conf/datasetInfoCache.xml");
    this->globalModelInfo = new ModelInfo("../../conf/modelInfoCache.xml");

	// 悬浮窗设置
	setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
	setCorner(Qt::BottomRightCorner, Qt::RightDockWidgetArea);

	// 功能模块:切换页面
	connect(ui->action_SceneSetting, &QAction::triggered, this, &MainWindow::switchPage);
    connect(ui->action_ModelChoice, &QAction::triggered, this, &MainWindow::switchPage);
    connect(ui->action_Evaluate, &QAction::triggered, this, &MainWindow::switchPage);
    connect(ui->action_ModelTrain, &QAction::triggered, this, &MainWindow::switchPage);
    connect(ui->action_Monitor, &QAction::triggered, this, &MainWindow::switchPage);

    connect(ui->action_ModelVis, &QAction::triggered, this, &MainWindow::switchPage);
    connect(ui->action_ModelCAM, &QAction::triggered, this, &MainWindow::switchPage);


    // 视图设置
	connect(ui->actionFullScreen, &QAction::triggered, this, &MainWindow::fullScreen);

    // 调试控制台设置
    terminal = new BashTerminal(ui->lineEdit, ui->textEdit);
    connect(ui->pushButton_bashCommit, &QPushButton::clicked, terminal, &BashTerminal::commitBash);
    connect(ui->pushButton_bashClean, &QPushButton::clicked, terminal, &BashTerminal::cleanBash);

    // 数据集悬浮窗设置
    this->datasetDock = new DatasetDock(this->ui, this->terminal, this->globalDatasetInfo);
    this->modelDock = new ModelDock(this->ui, this->terminal, this->globalModelInfo);

    // 场景选择页面
    this->senseSetPage = new SenseSetPage(this->ui, this->terminal, this->globalDatasetInfo);

    this->modelChoicePage = new ModelChoicePage(this->ui, this->terminal, this->globalModelInfo);

    this->modelEvalPage = new ModelEvalPage(this->ui, this->terminal,this->globalDatasetInfo, this->globalModelInfo);

    this->modelTrainPage = new ModelTrainPage(this->ui, this->terminal,this->globalDatasetInfo, this->globalModelInfo, this->modelDock);

    this->monitorPage = new MonitorPage(this->ui, this->terminal,this->globalDatasetInfo,this->globalModelInfo);

    this->modelVisPage = new ModelVisPage(this->ui, this->terminal, this->globalDatasetInfo, this->globalModelInfo);
    this->modelCAMPage = new ModelCAMPage(this->ui, this->terminal, this->globalDatasetInfo, this->globalModelInfo);

}


MainWindow::~MainWindow(){
	delete ui;
}


void MainWindow::switchPage(){
    QAction *action = qobject_cast<QAction*>(sender());
    if(action==ui->action_SceneSetting)
        ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_senseSet);
    else if(action==ui->action_ModelChoice)
        ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_modelChoice);
    else if(action==ui->action_Evaluate){
        ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_modelEval);
        this->modelEvalPage->refreshGlobalInfo();
    }
    else if(action==ui->action_ModelTrain){
        ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_modelTrain);
        this->modelTrainPage->refreshGlobalInfo();
    }
    else if(action==ui->action_Monitor){
        ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_monitor);
        this->monitorPage->refresh();
    }
    
    else if(action==ui->action_ModelVis){
        ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_modelVis);
        this->modelVisPage->refreshGlobalInfo();
    }
    else if(action==ui->action_ModelCAM){
        ui->stackedWidget_MultiPage->setCurrentWidget(ui->page_modelCAM);
        this->modelCAMPage->refreshGlobalInfo();
    }
}


void MainWindow::fullScreen(){
	auto full = ui->actionFullScreen->isChecked();
	menuBar()->setVisible(!full);
	ui->actionFullScreen->setShortcut(full ? QKeySequence("Esc") : QKeySequence("Ctrl+F"));
	
	static bool maximized = false;// 记录当前状态
	if (full){
		maximized = isMaximized();
	}
	else if ( maximized && isMaximized() ){
		return;
	}

    if ((full && !isMaximized()) || (!full && isMaximized())){
		if (isMaximized()){
			showNormal();
		}
		else
			showMaximized();
	}
}

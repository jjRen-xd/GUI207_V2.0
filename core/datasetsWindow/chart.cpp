#include "chart.h"
#include "qapplication.h"
#include "qpushbutton.h"
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>

Chart::Chart(QWidget* parent, QString _chartname, QString _filename){
    // setParent(parent);
    chartname = _chartname;
    filename = _filename;
    series = new QSplineSeries(this);
    qchart = new QChart;
    chartview = new QChartView(qchart);
    layout = new QHBoxLayout(this);
    axisX = new QValueAxis(this);
    axisY = new QValueAxis(this);
    zoom_btn = new QPushButton("放\n大");
    download_btn = new QPushButton("下\n载");

    connect(zoom_btn,&QPushButton::clicked,this,&Chart::ShowBigPic);
    connect(download_btn,&QPushButton::clicked,this,&Chart::SaveBigPic);

    layout->addWidget(chartview);
    layout->setContentsMargins(0,0,0,0);
    setLayout(layout);
    chartview->setRenderHint(QPainter::Antialiasing);//防止图形走样
}


Chart::~Chart(){
    delete qchart;
    delete zoom_btn;
    delete download_btn;
//    qchart=NULL;
//    zoom_btn=NULL;
//    download_btn=NULL;
}


void Chart::drawHRRPimage(QLabel* chartLabel){
    readHRRPtxt();
    setAxis("Range/cm",xmin,xmax,10, "dB(V/m)",ymin,ymax,10);
    buildChart(points);
    showChart(chartLabel);
}


void Chart::readHRRPtxt(){
    float x_min = 200,x_max = -200,y_min = 200,y_max = -200;
    //=======================================================
    //             文件读操作，后续可更换
    //=======================================================
    QFile file(filename);
    if(file.open(QIODevice::ReadOnly)){
        QByteArray line = file.readLine();
        QString str(line);
        QStringList strList = str.split(" ");
//        if(!(strList.filter("Range").length()&&strList.filter("HRRP").length()))
//            return;
        file.readLine();
        points.clear();
        while(!file.atEnd()){
            QByteArray line = file.readLine();
            QString str(line);
            QStringList strList = str.split("\t");
            QStringList result = strList.filter(".");
            if(result.length()==2){
                float x=result[0].toFloat();
                float y=result[1].toFloat();
                points.append(QPointF(x,y));
                x_min = fmin(x_min,x);
                y_min = fmin(y_min,y);
                x_max = fmax(x_max,x);
                y_max = fmax(y_max,y);
            }
        }
        xmin = x_min-3; xmax = x_max+3;
        ymin = y_min-3; ymax = y_max+3;
    }
    else{
    }
}


void Chart::setAxis(QString _xname, qreal _xmin, qreal _xmax, int _xtickc, \
             QString _yname, qreal _ymin, qreal _ymax, int _ytickc){
    xname = _xname; xmin = _xmin; xmax = _xmax; xtickc = _xtickc;
    yname = _yname; ymin = _ymin; ymax = _ymax; ytickc = _ytickc;

    axisX->setRange(xmin, xmax);    //设置范围
    axisX->setLabelsVisible(false);   //设置刻度的格式
    axisX->setGridLineVisible(true);   //网格线可见
    axisX->setTickCount(xtickc);       //设置多少个大格
    axisX->setMinorTickCount(1);   //设置每个大格里面小刻度线的数目
    axisX->setTitleText(xname);  //设置描述
    axisX->setTitleVisible(false);
    axisY->setRange(ymin, ymax);
    axisY->setLabelsVisible(false);
    axisY->setGridLineVisible(true);
    axisY->setTickCount(ytickc);
    axisY->setMinorTickCount(1);
    axisY->setTitleText(yname);
    axisY->setTitleVisible(false);
    qchart->addAxis(axisX, Qt::AlignBottom); //下：Qt::AlignBottom  上：Qt::AlignTop
    qchart->addAxis(axisY, Qt::AlignLeft);   //左：Qt::AlignLeft    右：Qt::AlignRight
    qchart->setContentsMargins(-10, -10, -10, -10);  //设置外边界全部为0
    qchart->setMargins(QMargins(-35, 0, -5, -15));
}


void Chart::buildChart(QList<QPointF> pointlist)
{
    //创建数据源
    series->setPen(QPen(Qt::blue,0.5,Qt::SolidLine));
    series->clear();
    points.clear();
    for(int i=0; i<pointlist.size();i++){
        series->append(pointlist.at(i).x(), pointlist.at(i).y());
        points.append(QPointF(pointlist.at(i).x(), pointlist.at(i).y()));
    }

    qchart->setAnimationOptions(QChart::SeriesAnimations);//设置曲线动画模式
    qchart->legend()->hide(); //隐藏图例
    qchart->addSeries(series);//输入数据
    qchart->setAxisX(axisX, series);
    qchart->setAxisY(axisY, series);
}


void Chart::showChart(QLabel *imagelabel){
    QHBoxLayout *pHLayout = (QHBoxLayout *)imagelabel->layout();
    if(!imagelabel->layout()){
        pHLayout = new QHBoxLayout(imagelabel);
    }
    else{
        QLayoutItem *child;
        while ((child = pHLayout->takeAt(0)) != 0){
            if(child->widget()){
                child->widget()->setParent(NULL);
            }
            delete child;
         }
    }
    QVBoxLayout *subqvLayout = new QVBoxLayout();

    zoom_btn->setFixedSize(20,35);
    download_btn->setFixedSize(20,35);

    subqvLayout->addWidget(zoom_btn);
    subqvLayout->addWidget(download_btn);
    subqvLayout->setAlignment(Qt::AlignCenter);
    subqvLayout->setContentsMargins(0, 0, 0, 0);

    QWidget* Widget = new QWidget;
    Widget->setLayout(subqvLayout);
    Widget->setContentsMargins(0, 0, 0, 0);

    pHLayout->addWidget(this, 20);
    pHLayout->addWidget(Widget, 1);
    pHLayout->setContentsMargins(0, 0, 0, 0);
}


void Chart::Show_infor(){
    QStringList spilited_names = filename.split('/');
    int id = spilited_names.length()-2;
    QString cls = spilited_names[id];
    QString content = "文件路径:   "+filename+"\n"+"类别标签:   "+cls;
    QMessageBox::about(NULL, "文件信息", content);
}


void Chart::ShowBigPic(){
    ShoworSave = 1;
    Show_Save();
}


void Chart::SaveBigPic(){
    ShoworSave = 2;
    Show_Save();
}


void Chart::Show_Save(){
    QChart *newchart = new QChart();
    QValueAxis *newaxisX = new QValueAxis();
    QValueAxis *newaxisY = new QValueAxis();
    newaxisX->setRange(xmin, xmax);    //设置范围
    newaxisX->setLabelFormat("%d");   //设置刻度的格式
    newaxisX->setGridLineVisible(true);   //网格线可见
    newaxisX->setTickCount(xtickc);       //设置多少个大格
    newaxisX->setMinorTickCount(1);   //设置每个大格里面小刻度线的数目
    newaxisX->setTitleText(xname);  //设置描述
    newaxisX->setTitleVisible(true);
    newaxisY->setRange(ymin, ymax);
    newaxisY->setLabelFormat("%d");
    newaxisY->setGridLineVisible(true);
    newaxisY->setTickCount(ytickc);
    newaxisY->setMinorTickCount(1);
    newaxisY->setTitleText(yname);
    newaxisY->setTitleVisible(true);
    newchart->addAxis(newaxisX, Qt::AlignBottom); //下：Qt::AlignBottom  上：Qt::AlignTop
    newchart->addAxis(newaxisY, Qt::AlignLeft);   //左：Qt::AlignLeft    右：Qt::AlignRight
    newchart->setContentsMargins(0, 0, 0, 0);  //设置外边界全部为0
    newchart->setMargins(QMargins(0, 0, 0, 0));
    newchart->setAnimationOptions(QChart::SeriesAnimations);//设置曲线动画模式

    QSplineSeries *newseries = new QSplineSeries();
    newseries->setPen(QPen(Qt::blue,1,Qt::SolidLine));
    newseries->clear();
    for(int i=0; i<points.size();i++)
        newseries->append(points.at(i).x(), points.at(i).y());

    newchart->setTitle(chartname);
    newchart->legend()->hide(); //隐藏图例
    newchart->addSeries(newseries);//输入数据
    newchart->setAxisX(newaxisX, newseries);
    newchart->setAxisY(newaxisY, newseries);

    newchart->resize(800,600);
    QChartView *bigView = new QChartView(newchart);
    bigView->setRenderHint(QPainter::Antialiasing);

    if(ShoworSave==1){
        bigView->show();
    }
    else{
        QPixmap p = bigView->grab();
        QImage image = p.toImage();
        QString fileName = QFileDialog::getSaveFileName(this,tr("保存文件"),"",tr("chart(*.png)"));
        if(!fileName.isNull()){
            image.save(fileName);
            QFileInfo file(fileName);
            if(file.exists()){
                QMessageBox::about(this, "操作成功", "文件保存成功!");
            }
            else{
                QMessageBox::about(this, "操作失败", "文件保存失败，请重试!");
            }
        }
    }
};

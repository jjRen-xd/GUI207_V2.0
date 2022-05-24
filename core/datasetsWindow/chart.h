#ifndef CHART_H
#define CHART_H

#include "qlabel.h"
#include "qpushbutton.h"
#include <QChartView>
#include <QChart>
#include <QSplineSeries>
#include <QHBoxLayout>
#include <QValueAxis>

class Chart : public QWidget{
    Q_OBJECT

    public:
        QChart *qchart;
        QChartView *chartview;
        QSplineSeries *series;

        QHBoxLayout *layout;
        QValueAxis *axisX;
        QValueAxis *axisY;

        QString chartname;
        QString filename;
        //坐标轴参数
        QString xname;
        qreal xmin;
        qreal xmax;
        int xtickc;
        QString yname;
        qreal ymin;
        qreal ymax;
        int ytickc;

        QList<QPointF> points;
        QPushButton *zoom_btn;
        QPushButton *download_btn;
        int ShoworSave = 1;

    public:
        Chart(QWidget* parent, QString _chartname, QString filename);
        ~Chart();
        void setAxis(QString _xname, qreal _xmin, qreal _xmax, int _xtickc, \
                     QString _yname, qreal _ymin, qreal _ymax, int _ytickc);
        void readHRRPtxt();
        void buildChart(QList<QPointF> pointlist);
        void drawHRRPimage(QLabel* chartLabel);
        void showChart(QLabel *imagelabel);
        void Show_Save();

    private slots:
        void ShowBigPic();
        void SaveBigPic();
        void Show_infor();
};

#endif // CHART_H

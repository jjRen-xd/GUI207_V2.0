#ifndef GUITHREADRUN_H
#define GUITHREADRUN_H
#pragma once
#include <QObject>
#include <mutex>
#include <memory>
#include <thread>
#include <future>
#include <functional>
#include <QThread>
#include <QCoreApplication>

class GuiThreadRun : public QObject {
    Q_OBJECT

public:
    static GuiThreadRun *inst() {
        static GuiThreadRun *s_this = new GuiThreadRun();
        return s_this;
    }

    /**
     * @brief 在gui线程中执行指定函数
     * @param f 要执行函数的指针
     * @param args 要执行函数的参数
     * @return 同执行函数的返回类型一致
     */
    template<typename F, typename... Args>
    static auto excute(F &&f, Args&&... args)
        -> typename std::result_of<F(Args...)>::type
    {
        using return_type = typename std::result_of<F(Args...)>::type;
        // 如果当前线程为主线程，直接调用，否则通过信号槽方式调用
        if (QThread::currentThread() == GuiThreadRun::inst()->thread()) {
            auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
            return task();
        }
        else {
            auto task = std::make_shared< std::packaged_task<return_type()> >(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...));
            std::future<return_type> res = task->get_future();
            GuiThreadRun::inst()->m_runLock.lock();
            GuiThreadRun::inst()->m_func = [task]() { (*task)(); };
            emit GuiThreadRun::inst()->run();
            GuiThreadRun::inst()->m_runLock.unlock();
            return res.get();
        }
    }

    std::mutex m_runLock;
    std::function<void()> m_func;
signals:
    void run();

private slots:
    void onRun() { m_func(); }

private:
    // 异常提示类
    class can_not_create_mainThreadRun_in_sub_thread {};

    GuiThreadRun() {
        if (this->thread() != QCoreApplication::instance()->thread()) {
            throw can_not_create_mainThreadRun_in_sub_thread();
        }
        connect(this, &GuiThreadRun::run, this, &GuiThreadRun::onRun, Qt::BlockingQueuedConnection);
    }
    GuiThreadRun(const GuiThreadRun &) = delete;
    GuiThreadRun operator =(const GuiThreadRun &) = delete;
};

#endif // GUITHREADRUN_H

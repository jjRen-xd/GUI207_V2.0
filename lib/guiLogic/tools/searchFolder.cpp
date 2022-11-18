#include "searchFolder.h"
#include "qdebug.h"
#include <string.h>
#include <string>
#include <io.h>
#include<algorithm>
#include <fstream>
#include <QObject>
using namespace std;

// 为了兼容win与linux双平台
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

// Windows平台
bool SearchFolder::getFiles(vector<string> &files, string filesType, string folderPath){
    intptr_t hFile = 0;
    struct _finddata_t fileInfo;

    if ((hFile = _findfirst((folderPath+"/*"+filesType).c_str(), &fileInfo)) != -1){
        do{
            files.push_back(fileInfo.name);
        } while(_findnext(hFile, &fileInfo) == 0);
    }
    else{
        return false;
    }
    return true;
}

bool SearchFolder::getAllFiles(vector<string> &files, string folderPath){
    intptr_t hFile = 0;
    struct _finddata_t fileInfo;
    int a=0;
    if ((hFile = _findfirst((folderPath+"/*").c_str(), &fileInfo)) != -1){
        do{
            files.push_back(fileInfo.name);
        } while(_findnext(hFile, &fileInfo) == 0);
    }
    else{
        return false;
    }
    return true;
}

bool SearchFolder::getDirs(vector<string> &dirs, string folderPath){
    intptr_t hFile = 0;
    struct _finddata_t fileInfo;

    if ((hFile = _findfirst((folderPath+"/*").c_str(), &fileInfo)) != -1){
        do{
            if ((fileInfo.attrib & _A_SUBDIR) && strcmp(fileInfo.name, ".") != 0 && strcmp(fileInfo.name, "..") != 0) {  //比较文件类型是否是文件夹
                dirs.push_back(fileInfo.name);
            }
        } while(_findnext(hFile, &fileInfo) == 0);
    }
    else{
        return false;
    }
    std::sort(dirs.begin(),dirs.end());
    return true;
}

#else
#include <dirent.h>
// Linux平台
bool SearchFolder::getFiles(vector<string> &files, string filesType, string folderPath){
    DIR *dir;
    struct dirent *ptr;

    if ((dir=opendir(folderPath.c_str())) == NULL)
        return false;

    while ((ptr=readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir
            continue;
        else if (ptr->d_type == 8){    //file
            // 判断是否是指定类型

            string sFilename(ptr->d_name);
            string suffixStr = sFilename.substr(sFilename.find_last_of('.'));//获取文件后缀
            if (suffixStr.compare(filesType) == 0) {//根据后缀筛选文件
                files.push_back(ptr->d_name);
            }
        }
        else if (ptr->d_type == 10)    //link file
            continue;
        else if (ptr->d_type == 4)    //dir
            continue;
    }
    closedir(dir);
    return true;
}

bool SearchFolder::getDirs(vector<string> &dirs, string folderPath){
    struct dirent *ptr;
    DIR *dir;

    if ((dir=opendir(folderPath.c_str())) == NULL)
        return false;

    while ((ptr=readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir
            continue;
        else if (ptr->d_type == 8)    ///file
            continue;
        else if (ptr->d_type == 10)    ///link file
            continue;
        else if (ptr->d_type == 4)    ///dir
            dirs.push_back(ptr->d_name);
    }
    closedir(dir);
    return true;
}

#endif



bool SearchFolder::exist(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}
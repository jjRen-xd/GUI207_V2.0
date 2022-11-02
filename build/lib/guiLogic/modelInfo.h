#ifndef MODELINFO_H
#define MODELINFO_H

#include <iostream>
#include <map>
#include <vector>
#include <string>


class ModelInfo{
    public:
        ModelInfo(std::string xmlPath);
        ~ModelInfo();


        std::string defaultXmlPath;                      // 默认.xml路径
        int writeToXML(std::string xmlPath);             // 将载入的模型信息保存至.xml文件
        int loadFromXML(std::string xmlPath);            // 从.xml文件中读取所载入模型的信息
        int addItemFromXML(std::string xmlPath);         // 从.xml文件中导入新模型

        void deleteItem(std::string type, std::string name);  // 删除模型

        void print();
        void clear();   // 清空所有模型

        size_t typeNum();
        std::vector<std::string> getTypes();                  // 获取所有的模型类型
        std::vector<std::string> getNamesInType(std::string type); // 获取特定类型下的模型名称
        std::map<std::string,std::string> getAllAttri(std::string Type, std::string Name);   // 获取指定模型的所有属性
        std::string getAttri(std::string type, std::string name, std::string attri);    // 获取指定模型指定属性的值

        void modifyAttri(std::string Type, std::string Name, std::string Attri, std::string AttriValue);   //修改指定数据集的指定属性
        bool checkMap(std::string type, std::string name="NULL", std::string attri="NULL");


        std::string selectedType;
        std::string selectedName;
        std::vector<std::string> selectedClassNames;

        std::map<std::string, std::string> var2TypeName;
        std::map<std::string, std::string> typeName2Var;



    private:
        // 模型核心数据Map
        std::map<std::string, std::map<std::string, std::map<std::string,std::string>>> infoMap;
        // map<datasetType, map<datasetName, map<datasetAttri, attriValue>>>
};


#endif // MODELINFO_H

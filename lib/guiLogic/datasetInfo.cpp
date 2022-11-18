#include "datasetInfo.h"


using namespace std;

#include "./lib/guiLogic/tinyXml/tinyxml.h"
/*
    TiXmlDocument：文档类，它代表了整个xml文件
    TiXmlDeclaration：声明类，它表示文件的声明部分
    TiXmlComment：注释类，它表示文件的注释部分
    TiXmlElement：元素类，它是文件的主要部分，并且支持嵌套结构，一般使用这种结构来分类的存储信息，它可以包含属性类和文本类
    TiXmlAttribute/TiXmlAttributeSet：元素属性，它一般嵌套在元素中，用于记录此元素的一些属性
    TiXmlText：文本对象，它嵌套在某个元素内部
*/

DatasetInfo::DatasetInfo(string xmlPath):
    defaultXmlPath(xmlPath)
{
    loadFromXML(this->defaultXmlPath);
    this->selectedType = "";
    this->selectedName = "";
//    this->print();
}

DatasetInfo::~DatasetInfo(){
    this->clear();
}


vector<string> DatasetInfo::getTypes(){
    vector<string> types;
    for(auto& it : infoMap) {
        types.push_back(it.first);
    }
    return types;
}


size_t DatasetInfo::typeNum(){
    return infoMap.size();
}


vector<string> DatasetInfo::getNamesInType(string type){
    vector<string> names;
    for(auto &item: this->infoMap[type]){
        names.push_back(item.first);
    }
    return names;
}


map<string,string> DatasetInfo::getAllAttri(string Type, string Name){
    return infoMap[Type][Name];
}


string DatasetInfo::getAttri(string type, string name, string attri){
    if (!checkMap(type,name,attri)) return "";
    return this->infoMap[type][name][attri];
}

void DatasetInfo::modifyAttri(string Type, string Name, string Attri, string AttriValue){
    this->infoMap[Type][Name][Attri] = AttriValue;
}


void DatasetInfo::print(){
    for(auto &datasetType: infoMap){
        cout<<"-->"<<datasetType.first<<endl;
        for(auto &datasetName: datasetType.second){
            cout<<"---->"<<datasetName.first<<endl;
            for(auto &datasetAttr: datasetName.second){
                cout<<"------>"<<datasetAttr.first<<"->"<<datasetAttr.second<<endl;
            }
        }
    }
}

void DatasetInfo::clear(){
    infoMap.clear();
}


int DatasetInfo::writeToXML(string xmlPath){
    TiXmlDocument *writeDoc = new TiXmlDocument; //xml文档指针
    TiXmlDeclaration *decl = new TiXmlDeclaration("1.0", "UTF-8", "yes");       //文档格式声明
    writeDoc->LinkEndChild(decl); //写入文档

    TiXmlElement *RootElement = new TiXmlElement("globalDatasetInfo");          //根元素
    RootElement->SetAttribute("datasetTypeNum", this->typeNum());  //属性
    writeDoc->LinkEndChild(RootElement);

    int typeID = 0;
    int nameID = 0;
    for(auto &datasetType: this->infoMap){  //n个父节点,即n个数据类型
        /* 对每个数据集类型建立节点 */
        typeID += 1;
        TiXmlElement *currTypeEle = new TiXmlElement(datasetType.first.c_str());
        currTypeEle->SetAttribute("typeID",typeID);         //设置节点属性
        RootElement->LinkEndChild(currTypeEle);             //父节点根节点

        //子元素
        for(auto &datasetName: datasetType.second){
            /* 对每个数据集建立节点 */
            nameID += 1;
            TiXmlElement *currNameEle = new TiXmlElement(datasetName.first.c_str());
            currTypeEle->LinkEndChild(currNameEle);
            currNameEle->SetAttribute("nameID",nameID);

            for(auto &datasetAttr: datasetName.second){
                /* 对每个属性建立节点 */
                TiXmlElement *currAttrEle = new TiXmlElement(datasetAttr.first.c_str());
                currNameEle->LinkEndChild(currAttrEle);

                TiXmlText *attrContent = new TiXmlText(datasetAttr.second.c_str());
                currAttrEle->LinkEndChild(attrContent);
            }
        }
    }

    writeDoc->SaveFile(xmlPath.c_str());
    delete writeDoc;

    return 1;
}

//解析xml文件
int DatasetInfo::loadFromXML(string xmlPath){
    TiXmlDocument datasetInfoDoc(xmlPath.c_str());   //xml文档对象
    bool loadOk=datasetInfoDoc.LoadFile();                  //加载文档
    if(!loadOk){
        cout<<"Could not load the datasetInfo file.Error:"<<datasetInfoDoc.ErrorDesc()<<endl;
        exit(1);
    }

    TiXmlElement *RootElement = datasetInfoDoc.RootElement();	//根元素, Info

    //遍历该结点
    for(TiXmlElement *currTypeEle = RootElement->FirstChildElement(); currTypeEle != NULL; currTypeEle = currTypeEle->NextSiblingElement()){
        map<string, map<string,string>> datasetNameMap;
        // 遍历节点属性
        TiXmlAttribute *pAttr=currTypeEle->FirstAttribute();
        while( NULL != pAttr){
            pAttr=pAttr->Next();
        }
        //遍历子节点
        for(TiXmlElement *currNameEle=currTypeEle->FirstChildElement(); currNameEle != NULL; currNameEle=currNameEle->NextSiblingElement()){
            map<string,string> datasetAttrMap;
            // 遍历节点属性
            TiXmlAttribute *pAttr=currNameEle->FirstAttribute();
            while( NULL != pAttr){
                pAttr=pAttr->Next();
            }
            //遍历子子节点
            for(TiXmlElement *currAttrEle=currNameEle->FirstChildElement(); currAttrEle != NULL; currAttrEle=currAttrEle->NextSiblingElement()){
                datasetAttrMap[currAttrEle->Value()] = currAttrEle->FirstChild()->Value();
                // 遍历节点属性
                TiXmlAttribute *pAttr=currAttrEle->FirstAttribute();
                while( NULL != pAttr){
                    pAttr=pAttr->Next();
                }
            }
            datasetNameMap[currNameEle->Value()] = datasetAttrMap;
        }
        this->infoMap[currTypeEle->Value()] = datasetNameMap;
    }
    return 1;
}


int DatasetInfo::addItemFromXML(string xmlPath){    //根据对应的xml将数据集的信息写入datasetInfoCache
    TiXmlDocument datasetInfoDoc(xmlPath.c_str());   //xml文档对象
    bool loadOk=datasetInfoDoc.LoadFile();                  //加载文档
    if(!loadOk){
        cout<<"Could not load the datasetInfo file.Error:"<<datasetInfoDoc.ErrorDesc()<<endl;
        return 0;
    }

    TiXmlElement *RootElement = datasetInfoDoc.RootElement();	//根元素, Info

    //遍历Type结点
    for(TiXmlElement *currTypeEle = RootElement->FirstChildElement(); currTypeEle != NULL; currTypeEle = currTypeEle->NextSiblingElement()){

        // 遍历节点属性
        TiXmlAttribute *pAttr=currTypeEle->FirstAttribute();
        while( NULL != pAttr){
            pAttr=pAttr->Next();
        }
        //遍历Name节点
        for(TiXmlElement *currNameEle=currTypeEle->FirstChildElement(); currNameEle != NULL; currNameEle=currNameEle->NextSiblingElement()){
            map<string,string> datasetAttrMap;
            // 遍历节点属性
            TiXmlAttribute *pAttr=currNameEle->FirstAttribute();
            while( NULL != pAttr){
                pAttr=pAttr->Next();
            }
            //遍历子子节点
            for(TiXmlElement *currAttrEle=currNameEle->FirstChildElement(); currAttrEle != NULL; currAttrEle=currAttrEle->NextSiblingElement()){
                datasetAttrMap[currAttrEle->Value()] = currAttrEle->FirstChild()->Value();
                // 遍历节点属性
                TiXmlAttribute *pAttr=currAttrEle->FirstAttribute();
                while( NULL != pAttr){
                    pAttr=pAttr->Next();
                }
            }
            this->infoMap[currTypeEle->Value()][currNameEle->Value()] = datasetAttrMap;
        }
    }
    return 1;
}


void DatasetInfo::deleteItem(string type, string name){
    if(checkMap(type, name)){
        this->infoMap[type].erase(name);
    }
}


bool DatasetInfo::checkMap(string type, string name, string attri){
    if(!this->infoMap.count(type)){
        return false;
    }
    else{
        if(name!="NULL" && !this->infoMap[type].count(name)){
            return false;
        }
        else{
            if(attri!="NULL" && !this->infoMap[type][name].count(attri)){
                return false;
            }
        }
    }
    return true;
}

//infoMap example
//map<QString, map<QString, map<QString,QString>>> infoMap = {
//    {"HRRP",{
//            {"HRRP_001",
//             {{"PATH","../../testPath"},{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            },
//            {"HRRP_002",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            }
//        },
//    },
//    {"RCS",{
//            {"RCS_001",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            },
//            {"RCS_002",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            }
//        },
//    },
//    {"Radia",{
//            {"Radia_001",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            },
//            {"Radia_002",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            }
//        },
//    },
//    {"Image",{
//            {"Image_001",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            },
//            {"Image_002",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            }
//        },
//    },
//};

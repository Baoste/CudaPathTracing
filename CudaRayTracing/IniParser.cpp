#include "IniParser.h"

void IniParser::Parse(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file) 
    {
        std::cerr << "Unable to open file " << filename << std::endl;
        return;
    }

    std::string line;
    std::string section;
    while (std::getline(file, line)) {
        // 跳过空行和注释
        if (line.empty() || line[0] == ';') continue;

        // 检测节（[Section]）
        if (line[0] == '[') 
        {
            section = line.substr(1, line.find(']') - 1);
            if (section == "Light") lights.push_back({});
            else if (section == "Sphere") spheres.push_back({});
            else if (section == "Floor") floors.push_back({});
            else if (section == "Mesh") meshes.push_back({});
            continue;
        }

        // 处理键值对
        size_t equalPos = line.find('=');
        if (equalPos == std::string::npos) continue;

        std::string key = line.substr(0, equalPos);
        std::string value = line.substr(equalPos + 1);

        // 根据节来处理不同的配置
        if (section == "Camera") 
        {
            if (key == "background") camera.background = parseVec3(value);
            else if (key == "width") camera.width = std::stoi(value);
            else if (key == "lookFrom") camera.lookFrom = parseVec3(value);
            else if (key == "lookAt") camera.lookAt = parseVec3(value);
            else if (key == "vFov") camera.vFov = std::stod(value);
        }
        else if (section == "Light") 
        {
            if (key == "center") lights.back().center = parseVec3(value);
            else if (key == "width") lights.back().width = std::stod(value);
            else if (key == "height") lights.back().height = std::stod(value);
            else if (key == "normal") lights.back().normal = parseVec3(value);
            else if (key == "color") lights.back().color = parseVec3(value);
        }
        else if (section == "Sphere") 
        {
            if (key == "center") spheres.back().center = parseVec3(value);
            else if (key == "radius") spheres.back().radius = std::stod(value);
            else if (key == "color") spheres.back().color = parseVec3(value);
            else if (key == "alphaX") spheres.back().alphaX = std::stod(value);
            else if (key == "alphaY") spheres.back().alphaY = std::stod(value);
        }
        else if (section == "Floor") 
        {
            if (key == "lt") floors.back().lt = parseVec3(value);
            else if (key == "rt") floors.back().rt = parseVec3(value);
            else if (key == "lb") floors.back().lb = parseVec3(value);
            else if (key == "rb") floors.back().rb = parseVec3(value);
            else if (key == "color") floors.back().color = parseVec3(value);
            else if (key == "alphaX") floors.back().alphaX = std::stod(value);
            else if (key == "alphaY") floors.back().alphaY = std::stod(value);
        }
        else if (section == "Mesh") 
        {
            if (key == "path") meshes.back().path = value;
            else if (key == "center") meshes.back().center = parseVec3(value);
            else if (key == "rotation") meshes.back().rotation = std::stod(value);
            else if (key == "color") meshes.back().color = parseVec3(value);
            else if (key == "glass") meshes.back().glass = std::stoi(value);
            else if (key == "scale") meshes.back().scale = std::stod(value);
            else if (key == "alphaX") meshes.back().alphaX = std::stod(value);
            else if (key == "alphaY") meshes.back().alphaY = std::stod(value);
        }
        else if (section == "Cloth")
        {
            hasCloth = true;
            if (key == "center") cloth.center = parseVec3(value);
            else if (key == "width") cloth.width = std::stod(value);
            else if (key == "color") cloth.color = parseVec3(value);
        }
    }
}

double3 IniParser::parseVec3(const std::string& str)
{
    double3 vec;
    std::istringstream ss(str);
    char discard;
    ss >> discard >> vec.x >> discard >> vec.y >> discard >> vec.z >> discard;
    return vec;
}
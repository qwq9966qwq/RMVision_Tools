// file_analyzer.cpp
// 文件结构分析工具 - 递归遍历文件夹并生成结构描述

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <sstream>
#include <algorithm>

namespace fs = std::filesystem;

// 定义配置结构体
struct Config {
    std::string rootPath;              // 要分析的根目录路径
    std::string outputFile;            // 输出文件路径
    bool includeHiddenFiles = false;   // 是否包含隐藏文件
    bool includeEmptyFolders = true;   // 是否包含空文件夹
    std::vector<std::string> excludeExtensions; // 要排除的文件扩展名
    std::vector<std::string> excludeFolders;    // 要排除的文件夹名
    int maxDepth = -1;                 // 最大遍历深度，-1表示无限制
    std::string outputFormat = "text"; // 输出格式：text, json, xml, html
    bool useColors = true;             // 是否使用彩色输出（仅用于控制台）
    bool useUnicodeChars = true;       // 是否使用Unicode字符表示树结构
};

// 定义统计信息结构体
struct Statistics {
    int totalDirs = 0;
    int totalFiles = 0;
    std::map<std::string, int> extensionCounts;
};

// 读取配置文件
Config readConfig(const std::string& configFile) {
    Config config;
    std::ifstream file(configFile);
    
    if (!file.is_open()) {
        std::cerr << "无法打开配置文件: " << configFile << std::endl;
        return config;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // 跳过注释和空行
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        
        size_t delimPos = line.find('=');
        if (delimPos != std::string::npos) {
            std::string key = line.substr(0, delimPos);
            std::string value = line.substr(delimPos + 1);
            
            // 去除首尾空格
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (key == "rootPath") {
                config.rootPath = value;
            } else if (key == "outputFile") {
                config.outputFile = value;
            } else if (key == "includeHiddenFiles") {
                config.includeHiddenFiles = (value == "true" || value == "1" || value == "yes");
            } else if (key == "includeEmptyFolders") {
                config.includeEmptyFolders = (value == "true" || value == "1" || value == "yes");
            } else if (key == "excludeExtensions") {
                std::stringstream ss(value);
                std::string ext;
                while (std::getline(ss, ext, ',')) {
                    ext.erase(0, ext.find_first_not_of(" \t"));
                    ext.erase(ext.find_last_not_of(" \t") + 1);
                    config.excludeExtensions.push_back(ext);
                }
            } else if (key == "excludeFolders") {
                std::stringstream ss(value);
                std::string folder;
                while (std::getline(ss, folder, ',')) {
                    folder.erase(0, folder.find_first_not_of(" \t"));
                    folder.erase(folder.find_last_not_of(" \t") + 1);
                    config.excludeFolders.push_back(folder);
                }
            } else if (key == "maxDepth") {
                try {
                    config.maxDepth = std::stoi(value);
                } catch (...) {
                    std::cerr << "警告: 无效的maxDepth值: " << value << std::endl;
                }
            } else if (key == "outputFormat") {
                config.outputFormat = value;
            } else if (key == "useColors") {
                config.useColors = (value == "true" || value == "1" || value == "yes");
            } else if (key == "useUnicodeChars") {
                config.useUnicodeChars = (value == "true" || value == "1" || value == "yes");
            }
        }
    }
    
    return config;
}

// 检查文件/文件夹是否应被排除
bool shouldExclude(const fs::path& path, const Config& config, int currentDepth) {
    // 检查隐藏文件
    if (!config.includeHiddenFiles) {
        std::string filename = path.filename().string();
        if (!filename.empty() && filename[0] == '.') {
            return true;
        }
    }
    
    // 检查最大深度
    if (config.maxDepth >= 0 && currentDepth > config.maxDepth) {
        return true;
    }
    
    // 检查是否是要排除的文件夹
    if (fs::is_directory(path)) {
        std::string folderName = path.filename().string();
        for (const auto& excludeFolder : config.excludeFolders) {
            if (folderName == excludeFolder) {
                return true;
            }
        }
    } 
    // 检查是否是要排除的文件扩展名
    else if (fs::is_regular_file(path)) {
        std::string ext = path.extension().string();
        if (!ext.empty()) {
            // 移除前导点号
            ext = ext.substr(1);
            for (const auto& excludeExt : config.excludeExtensions) {
                if (ext == excludeExt) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

// 表示文件系统中的项
struct FileSystemItem {
    std::string name;
    bool isDirectory;
    std::vector<FileSystemItem> children;
    std::string extension; // 文件扩展名
};

// 递归分析文件夹结构
FileSystemItem analyzeDirectory(const fs::path& path, const Config& config, Statistics& stats, int currentDepth = 0) {
    FileSystemItem item;
    item.name = path.filename().string();
    if (item.name.empty()) { // 如果是根目录
        item.name = path.string();
    }
    item.isDirectory = true;
    
    bool hasChildren = false;
    
    try {
        // 收集子项
        std::vector<fs::directory_entry> entries;
        for (const auto& entry : fs::directory_iterator(path)) {
            if (!shouldExclude(entry.path(), config, currentDepth + 1)) {
                entries.push_back(entry);
            }
        }
        
        // 排序：先文件夹，再文件，每种类型内部按名称排序
        std::sort(entries.begin(), entries.end(), [](const fs::directory_entry& a, const fs::directory_entry& b) {
            bool aIsDir = fs::is_directory(a.path());
            bool bIsDir = fs::is_directory(b.path());
            
            if (aIsDir && !bIsDir) return true;
            if (!aIsDir && bIsDir) return false;
            
            return a.path().filename() < b.path().filename();
        });
        
        // 处理子项
        for (const auto& entry : entries) {
            if (fs::is_directory(entry.path())) {
                stats.totalDirs++;
                FileSystemItem childDir = analyzeDirectory(entry.path(), config, stats, currentDepth + 1);
                if (config.includeEmptyFolders || !childDir.children.empty()) {
                    item.children.push_back(childDir);
                    hasChildren = true;
                }
            } else if (fs::is_regular_file(entry.path())) {
                stats.totalFiles++;
                FileSystemItem file;
                file.name = entry.path().filename().string();
                file.isDirectory = false;
                file.extension = entry.path().extension().string();
                
                // 更新统计信息
                if (!file.extension.empty()) {
                    // 移除前导点号
                    std::string ext = file.extension.substr(1);
                    stats.extensionCounts[ext]++;
                }
                
                item.children.push_back(file);
                hasChildren = true;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "错误: 无法访问目录 " << path << ": " << e.what() << std::endl;
    }
    
    return item;
}

// ANSI颜色代码
namespace Color {
    const std::string Reset = "\033[0m";
    const std::string Bold = "\033[1m";
    const std::string Blue = "\033[34m";
    const std::string Green = "\033[32m";
    const std::string Yellow = "\033[33m";
    const std::string Cyan = "\033[36m";
    const std::string Magenta = "\033[35m";
    const std::string Red = "\033[31m";
}

// 生成文本格式的输出
void generateTextOutput(const FileSystemItem& item, std::ostream& out, const Config& config, 
                      const Statistics& stats, const std::vector<bool>& isLastChild = {}) {
    // 生成标题
    out << "===================================================" << std::endl;
    out << "  文件结构分析报告" << std::endl;
    out << "===================================================" << std::endl;
    out << "分析根目录: " << item.name << std::endl;
    out << "===================================================" << std::endl << std::endl;
    
    // 定义树形图符号
    std::string verticalLine, turnCorner, horizontalLine, verticalAndRight, lastItem;
    
    if (config.useUnicodeChars) {
        verticalLine = "│   ";
        turnCorner = "└── ";
        horizontalLine = "─";
        verticalAndRight = "├── ";
        lastItem = "└── ";
    } else {
        verticalLine = "|   ";
        turnCorner = "`-- ";
        horizontalLine = "-";
        verticalAndRight = "|-- ";
        lastItem = "`-- ";
    }
    
    // 递归生成目录树的辅助函数
    std::function<void(const FileSystemItem&, const std::vector<bool>&)> printTree;
    
    printTree = [&](const FileSystemItem& currentItem, const std::vector<bool>& levels) {
        // 打印当前项的前缀线条
        for (size_t i = 0; i < levels.size() - 1; ++i) {
            if (levels[i]) {
                out << "    ";
            } else {
                out << verticalLine;
            }
        }
        
        // 打印当前项的连接线和名称
        if (!levels.empty()) {
            out << (levels.back() ? lastItem : verticalAndRight);
        }
        
        // 根据是否是目录，使用不同的颜色和样式
        if (config.useColors && &out == &std::cout) {
            if (currentItem.isDirectory) {
                out << Color::Bold << Color::Blue << currentItem.name << Color::Reset;
            } else {
                // 根据扩展名选择颜色
                std::string ext = currentItem.extension;
                if (!ext.empty()) ext = ext.substr(1); // 去掉点号
                
                if (ext == "cpp" || ext == "h" || ext == "hpp" || ext == "c") {
                    out << Color::Green << currentItem.name << Color::Reset;
                } else if (ext == "py" || ext == "js" || ext == "java") {
                    out << Color::Yellow << currentItem.name << Color::Reset;
                } else if (ext == "md" || ext == "txt" || ext == "pdf" || ext == "doc" || ext == "docx") {
                    out << Color::Cyan << currentItem.name << Color::Reset;
                } else if (ext == "jpg" || ext == "png" || ext == "gif" || ext == "bmp") {
                    out << Color::Magenta << currentItem.name << Color::Reset;
                } else if (ext == "exe" || ext == "dll" || ext == "so" || ext == "o") {
                    out << Color::Red << currentItem.name << Color::Reset;
                } else {
                    out << currentItem.name;
                }
            }
        } else {
            // 不使用颜色时，为目录添加括号区分
            if (currentItem.isDirectory) {
                out << "[" << currentItem.name << "]";
            } else {
                out << currentItem.name;
            }
        }
        
        out << std::endl;
        
        // 递归处理子项
        for (size_t i = 0; i < currentItem.children.size(); ++i) {
            std::vector<bool> newLevels = levels;
            newLevels.push_back(i == currentItem.children.size() - 1);
            printTree(currentItem.children[i], newLevels);
        }
    };
    
    // 根目录不需要前缀线条
    out << (config.useColors && &out == &std::cout ? Color::Bold + Color::Blue : "");
    out << item.name;
    out << (config.useColors && &out == &std::cout ? Color::Reset : "");
    out << std::endl;
    
    // 打印子项
    for (size_t i = 0; i < item.children.size(); ++i) {
        std::vector<bool> levels = {i == item.children.size() - 1};
        printTree(item.children[i], levels);
    }
    
    // 打印统计信息
    out << std::endl << "===================================================" << std::endl;
    out << "统计信息:" << std::endl;
    out << "===================================================" << std::endl;
    out << "目录数: " << stats.totalDirs << std::endl;
    out << "文件数: " << stats.totalFiles << std::endl;
    
    // 打印扩展名统计
    if (!stats.extensionCounts.empty()) {
        out << std::endl << "文件类型统计:" << std::endl;
        for (const auto& [ext, count] : stats.extensionCounts) {
            out << "." << ext << ": " << count << " 个文件" << std::endl;
        }
    }
    
    out << "===================================================" << std::endl;
}

// 生成JSON格式的输出
void generateJsonOutput(const FileSystemItem& item, std::ostream& out, const Statistics& stats) {
    // 打印JSON头部和元数据
    out << "{" << std::endl;
    out << "  \"metadata\": {" << std::endl;
    out << "    \"totalDirectories\": " << stats.totalDirs << "," << std::endl;
    out << "    \"totalFiles\": " << stats.totalFiles << "," << std::endl;
    out << "    \"extensions\": {" << std::endl;
    
    // 输出扩展名统计
    size_t extCount = 0;
    for (const auto& [ext, count] : stats.extensionCounts) {
        out << "      \"" << ext << "\": " << count;
        if (++extCount < stats.extensionCounts.size()) {
            out << ",";
        }
        out << std::endl;
    }
    
    out << "    }" << std::endl;
    out << "  }," << std::endl;
    
    // 打印文件结构
    out << "  \"fileStructure\": ";
    
    // 递归打印项的辅助函数
    std::function<void(const FileSystemItem&, int, bool)> printItem;
    
    printItem = [&](const FileSystemItem& currentItem, int indent, bool isLast) {
        std::string indentStr(indent * 2, ' ');
        
        out << indentStr << "{" << std::endl;
        out << indentStr << "  \"name\": \"" << currentItem.name << "\"," << std::endl;
        out << indentStr << "  \"type\": \"" << (currentItem.isDirectory ? "directory" : "file") << "\"";
        
        if (!currentItem.isDirectory) {
            out << "," << std::endl;
            out << indentStr << "  \"extension\": \"" << currentItem.extension << "\"";
        }
        
        if (currentItem.isDirectory && !currentItem.children.empty()) {
            out << "," << std::endl;
            out << indentStr << "  \"children\": [" << std::endl;
            
            for (size_t i = 0; i < currentItem.children.size(); ++i) {
                printItem(currentItem.children[i], indent + 2, i == currentItem.children.size() - 1);
            }
            
            out << indentStr << "  ]" << std::endl;
        } else {
            out << std::endl;
        }
        
        out << indentStr << "}" << (isLast ? "" : ",") << std::endl;
    };
    
    printItem(item, 1, true);
    
    out << "}" << std::endl;
}

// 生成XML格式的输出
void generateXmlOutput(const FileSystemItem& item, std::ostream& out, const Statistics& stats) {
    // XML头部
    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
    out << "<fileStructure>" << std::endl;
    
    // 生成元数据
    out << "  <metadata>" << std::endl;
    out << "    <statistics>" << std::endl;
    out << "      <totalDirectories>" << stats.totalDirs << "</totalDirectories>" << std::endl;
    out << "      <totalFiles>" << stats.totalFiles << "</totalFiles>" << std::endl;
    
    // 扩展名统计
    if (!stats.extensionCounts.empty()) {
        out << "      <extensions>" << std::endl;
        for (const auto& [ext, count] : stats.extensionCounts) {
            out << "        <extension type=\"" << ext << "\" count=\"" << count << "\" />" << std::endl;
        }
        out << "      </extensions>" << std::endl;
    }
    
    out << "    </statistics>" << std::endl;
    out << "  </metadata>" << std::endl;
    
    // 递归生成XML的辅助函数
    std::function<void(const FileSystemItem&, int)> generateXml;
    
    generateXml = [&](const FileSystemItem& currentItem, int indent) {
        std::string indentStr(indent * 2, ' ');
        
        if (currentItem.isDirectory) {
            out << indentStr << "<directory name=\"" << currentItem.name << "\">" << std::endl;
            
            for (const auto& child : currentItem.children) {
                generateXml(child, indent + 1);
            }
            
            out << indentStr << "</directory>" << std::endl;
        } else {
            out << indentStr << "<file name=\"" << currentItem.name 
                << "\" extension=\"" << currentItem.extension << "\" />" << std::endl;
        }
    };
    
    // 生成根目录
    generateXml(item, 1);
    
    out << "</fileStructure>" << std::endl;
}

// 生成HTML格式的输出
void generateHtmlOutput(const FileSystemItem& item, std::ostream& out, const Statistics& stats) {
    // HTML头部
    out << "<!DOCTYPE html>" << std::endl;
    out << "<html lang=\"zh-CN\">" << std::endl;
    out << "<head>" << std::endl;
    out << "  <meta charset=\"UTF-8\">" << std::endl;
    out << "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">" << std::endl;
    out << "  <title>文件结构分析报告</title>" << std::endl;
    out << "  <style>" << std::endl;
    out << "    body { font-family: Arial, sans-serif; margin: 20px; }" << std::endl;
    out << "    h1, h2 { color: #333; }" << std::endl;
    out << "    .container { max-width: 1200px; margin: 0 auto; }" << std::endl;
    out << "    .header { background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }" << std::endl;
    out << "    .stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }" << std::endl;
    out << "    .stat-card { background-color: #f9f9f9; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }" << std::endl;
    out << "    .extensions { margin-bottom: 20px; }" << std::endl;
    out << "    .ext-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }" << std::endl;
    out << "    .ext-item { background-color: #e9f7fe; padding: 8px; border-radius: 3px; }" << std::endl;
    out << "    .tree { font-family: monospace; }" << std::endl;
    out << "    .tree ul { list-style-type: none; padding-left: 20px; }" << std::endl;
    out << "    .directory { color: #0066cc; font-weight: bold; }" << std::endl;
    out << "    .file { color: #333; }" << std::endl;
    out << "    .ext-cpp, .ext-h, .ext-hpp, .ext-c { color: #5cb85c; }" << std::endl;
    out << "    .ext-py, .ext-js, .ext-java { color: #f0ad4e; }" << std::endl;
    out << "    .ext-md, .ext-txt, .ext-pdf, .ext-doc, .ext-docx { color: #5bc0de; }" << std::endl;
    out << "    .ext-jpg, .ext-png, .ext-gif, .ext-bmp { color: #d9534f; }" << std::endl;
    out << "    .ext-exe, .ext-dll, .ext-so, .ext-o { color: #9c27b0; }" << std::endl;
    out << "    footer { margin-top: 30px; text-align: center; color: #777; font-size: 0.9em; }" << std::endl;
    out << "    .toggle-btn { cursor: pointer; user-select: none; }" << std::endl;
    out << "    .hidden { display: none; }" << std::endl;
    out << "  </style>" << std::endl;
    out << "</head>" << std::endl;
    out << "<body>" << std::endl;
    out << "  <div class=\"container\">" << std::endl;
    
    // 页面头部
    out << "    <div class=\"header\">" << std::endl;
    out << "      <h1>文件结构分析报告</h1>" << std::endl;
    out << "      <p>根目录: <strong>" << item.name << "</strong></p>" << std::endl;
    out << "    </div>" << std::endl;
    
    // 统计信息
    out << "    <h2>统计信息</h2>" << std::endl;
    out << "    <div class=\"stats\">" << std::endl;
    out << "      <div class=\"stat-card\">" << std::endl;
    out << "        <h3>目录数</h3>" << std::endl;
    out << "        <p>" << stats.totalDirs << "</p>" << std::endl;
    out << "      </div>" << std::endl;
    out << "      <div class=\"stat-card\">" << std::endl;
    out << "        <h3>文件数</h3>" << std::endl;
    out << "        <p>" << stats.totalFiles << "</p>" << std::endl;
    out << "      </div>" << std::endl;
    out << "    </div>" << std::endl;
    
    // 扩展名统计
    if (!stats.extensionCounts.empty()) {
        out << "    <div class=\"extensions\">" << std::endl;
        out << "      <h2>文件类型统计</h2>" << std::endl;
        out << "      <div class=\"ext-grid\">" << std::endl;
        
        for (const auto& [ext, count] : stats.extensionCounts) {
            out << "        <div class=\"ext-item\">." << ext << ": " << count << " 个文件</div>" << std::endl;
        }
        
        out << "      </div>" << std::endl;
        out << "    </div>" << std::endl;
    }
    
    // 文件结构树
    out << "    <h2>文件结构</h2>" << std::endl;
    out << "    <div class=\"tree\">" << std::endl;
    
    // 递归生成HTML树的辅助函数
    std::function<void(const FileSystemItem&)> generateTree;
    
    generateTree = [&](const FileSystemItem& currentItem) {
        if (currentItem.isDirectory) {
            // 输出目录节点
            if (currentItem.children.empty()) {
                out << "      <div><span class=\"directory\">" << currentItem.name << "</span> (空目录)</div>" << std::endl;
            } else {
                out << "      <div><span class=\"toggle-btn directory\" onclick=\"this.parentElement.querySelector('.children').classList.toggle('hidden');\">+ " 
                    << currentItem.name << "</span></div>" << std::endl;
                out << "      <ul class=\"children\">" << std::endl;
                
                for (const auto& child : currentItem.children) {
                    out << "        <li>";
                    generateTree(child);
                    out << "</li>" << std::endl;
                }
                
                out << "      </ul>" << std::endl;
            }
        } else {
            // 输出文件节点，根据扩展名设置不同的样式
            std::string extClass = "";
            std::string ext = currentItem.extension;
            if (!ext.empty()) {
                ext = ext.substr(1); // 去掉点号
                
                if (ext == "cpp" || ext == "h" || ext == "hpp" || ext == "c") {
                    extClass = "ext-cpp";
                } else if (ext == "py" || ext == "js" || ext == "java") {
                    extClass = "ext-py";
                } else if (ext == "md" || ext == "txt" || ext == "pdf" || ext == "doc" || ext == "docx") {
                    extClass = "ext-md";
                } else if (ext == "jpg" || ext == "png" || ext == "gif" || ext == "bmp") {
                    extClass = "ext-jpg";
                } else if (ext == "exe" || ext == "dll" || ext == "so" || ext == "o") {
                    extClass = "ext-exe";
                }
            }
            
            out << "<span class=\"file " << extClass << "\">" << currentItem.name << "</span>";
        }
    };
    
    // 生成根目录节点
    generateTree(item);
    
    out << "    </div>" << std::endl;
    
    // 页面脚部
    out << "    <footer>" << std::endl;
    out << "      <p>文件结构分析报告</p>" << std::endl;
    out << "    </footer>" << std::endl;
    
    // JavaScript脚本用于交互性
    out << "    <script>" << std::endl;
    out << "      // 初始化时展开第一级目录" << std::endl;
    out << "      document.addEventListener('DOMContentLoaded', function() {" << std::endl;
    out << "        var rootChildren = document.querySelector('.children');" << std::endl;
    out << "        if(rootChildren) {" << std::endl;
    out << "          rootChildren.classList.remove('hidden');" << std::endl;
    out << "        }" << std::endl;
    out << "      });" << std::endl;
    out << "    </script>" << std::endl;
    
    out << "  </div>" << std::endl;
    out << "</body>" << std::endl;
    out << "</html>" << std::endl;
}

// 生成输出
void generateOutput(const FileSystemItem& rootItem, const Config& config, const Statistics& stats) {
    std::ofstream outFile;
    std::ostream* out = &std::cout; // 默认输出到标准输出
    
    if (!config.outputFile.empty()) {
        outFile.open(config.outputFile);
        if (outFile.is_open()) {
            out = &outFile;
        } else {
            std::cerr << "警告: 无法创建输出文件: " << config.outputFile << "，将输出到控制台" << std::endl;
        }
    }
    
    if (config.outputFormat == "json") {
        generateJsonOutput(rootItem, *out, stats);
    } else if (config.outputFormat == "xml") {
        generateXmlOutput(rootItem, *out, stats);
    } else if (config.outputFormat == "html") {
        generateHtmlOutput(rootItem, *out, stats);
    } else { // 默认为文本格式
        generateTextOutput(rootItem, *out, config, stats);
    }
    
    if (outFile.is_open()) {
        outFile.close();
        std::cout << "文件结构已保存到: " << config.outputFile << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string configFile = "file_analyzer.conf";
    
    // 检查命令行参数
    if (argc > 1) {
        configFile = argv[1];
    }
    
    // 读取配置
    Config config = readConfig(configFile);
    
    // 检查根路径是否有效
    if (config.rootPath.empty()) {
        std::cerr << "错误: 未指定根路径，请在配置文件中设置rootPath" << std::endl;
        return 1;
    }
    
    if (!fs::exists(config.rootPath)) {
        std::cerr << "错误: 指定的根路径不存在: " << config.rootPath << std::endl;
        return 1;
    }
    
    std::cout << "正在分析目录: " << config.rootPath << std::endl;
    
    // 初始化统计信息
    Statistics stats;
    
    // 分析目录结构
    FileSystemItem rootItem = analyzeDirectory(config.rootPath, config, stats);
    
    // 生成输出
    generateOutput(rootItem, config, stats);
    
    return 0;
}

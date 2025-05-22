cmake_minimum_required(VERSION 3.10)
project(FileStructureAnalyzer VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 添加源文件
add_executable(file_analyzer src/file_analyzer.cpp)

# 设置包含目录
target_include_directories(file_analyzer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 根据操作系统设置链接库
if(UNIX AND NOT APPLE)
    # Linux需要链接stdc++fs库以支持std::filesystem (取决于编译器版本)
    target_link_libraries(file_analyzer PRIVATE stdc++fs)
endif()

# 将配置文件复制到构建目录
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/file_analyzer.conf 
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/file_analyzer.conf
               COPYONLY)

# 安装配置
install(TARGETS file_analyzer DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/file_analyzer.conf DESTINATION etc)

# 添加测试目标
enable_testing()
add_test(NAME DefaultTest 
         COMMAND file_analyzer ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/file_analyzer.conf
         WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

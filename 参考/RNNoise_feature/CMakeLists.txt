cmake_minimum_required(VERSION 3.10)
project(RNNoise_feature C)
# C++编译器
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fexec-charset=GBK")
#
## C语言编译器
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fexec-charset=GBK")
set(CMAKE_C_STANDARD 11)

include_directories(./inc)    # 添添加 头文件 的搜索路径
aux_source_directory(./src src_folder)  # 将路径的 源文件 收集到变量列表

#add_compile_options(-lm)
LINK_LIBRARIES(m)

#add_subdirectory(src)       # 指定子目录
add_executable(C_demo
        main.c ${src_folder})


file(GLOB A1_SRC src/*.cpp)
add_library(A1 STATIC ${A1_SRC})
target_include_directories(A1 PUBLIC include)
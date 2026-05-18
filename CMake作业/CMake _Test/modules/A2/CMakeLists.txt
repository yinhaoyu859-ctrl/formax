file(GLOB A2_SRC src/*.cpp)
add_library(A2 STATIC ${A2_SRC})
target_include_directories(A2 PUBLIC include)
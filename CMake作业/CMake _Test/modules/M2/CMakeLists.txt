file(GLOB M2_SRC src/*.cpp)
add_library(M2 STATIC ${M2_SRC})
target_include_directories(M2 PUBLIC include)
target_link_libraries(M2 PUBLIC A1 A2 kalman)
file(GLOB math_SRC src/*.cpp)
add_library(math STATIC ${math_SRC})
target_include_directories(math PUBLIC include)
target_link_libraries(math PUBLIC ${OpenCV_LIBS})
add_executable(
    mlp_example
    main.cpp
    mlp.cpp
)

target_include_directories(
    mlp_example
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
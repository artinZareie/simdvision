set(SIMD_CLI_SOURCES
    src/main.c
)

add_executable(simdcnn-cli
    ${SIMD_CLI_SOURCES}
)

target_link_libraries(simdcnn-cli PRIVATE simdcnn)

target_compile_options(simdcnn-cli PRIVATE
    -Wall -Wextra -pedantic
)

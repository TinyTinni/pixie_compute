add_executable(pixie_template_tests
    test_vector_add.cpp
)

target_link_libraries(pixie_template_tests
    PRIVATE
        pixie_template_core
        Catch2::Catch2WithMain
)

catch_discover_tests(pixie_template_tests)

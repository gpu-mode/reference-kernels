#ifndef POPCORN_UTILS_H
#define POPCORN_UTILS_H

#include <iostream>
#include <string>
#include <sstream>

enum ExitCodes: int {
    USAGE_ERROR = 2,            // (standard?) exit code for wrong command line
    EXIT_CUDA_API = 110,        // cuda API call returned an error
    EXIT_PIPE_FAIL = 111,       // could not set up communication with runner
    EXIT_TEST_FAIL = 112,       // a test case failed
    EXIT_TEST_SPEC = 113        // error when trying to construct a test case
};

// checks that a CUDA API call returned successfully, otherwise prints an error message and exits.
static inline void cuda_check_(cudaError_t status, const char* expr, const char* file, int line, const char* function)
{
    if(status != cudaSuccess) {
        std::cerr << "CUDA error (" << (int)status << ") while evaluating expression "
                  << expr << " at "
                  << file << '('
                  << line << ") in `"
                  << function << "`: "
                  << cudaGetErrorString(status) << std::endl;
        std::exit(ExitCodes::EXIT_CUDA_API);
    }
}

// Convenience macro, automatically logs expression, file, line, and function name
// of the error.
#define CUDA_CHECK(expr) cuda_check_(expr, #expr, __FILE__, __LINE__, __FUNCTION__)


struct TestVerdict {
    bool Pass;
    std::string Message = "";
};

class TestReporter {
public:
    TestReporter() = default;

    void pass() {
        if(m_State != NONE) {
            std::cerr << "Trying to mark result of test twice."
                         " This indicates an error in the task definition, please report."
                      << std::endl;
            std::exit(EXIT_TEST_SPEC);
        }
        m_State = PASS;
    }
    std::stringstream& fail() {
        if(m_State != NONE) {
            std::cerr << "Trying to mark result of test twice."
                         " This indicates an error in the task definition, please report."
                      << std::endl;
            std::exit(EXIT_TEST_SPEC);
        }
        m_State = FAIL;
        return m_Message;
    }

    bool has_passed() const {
        if(m_State == NONE) {
            std::cerr << "Trying to query result of unfinished test."
                         " This indicates an error in the task definition, please report."
                      << std::endl;
            std::exit(EXIT_TEST_SPEC);
        }
        return m_State == PASS;
    }

    template<class T>
    bool check_equal(const char* message, const T& value, const T& expected) {
        if(value == expected) return true;
        fail() << message << ": " << expected << "`" << expected << "`, got `" << value << "`";
        return false;
    }

    std::string message() const {
        return m_Message.str();
    }
private:
    enum State {
        NONE, PASS, FAIL
    };
    State m_State = NONE;
    std::stringstream m_Message;
};
#endif

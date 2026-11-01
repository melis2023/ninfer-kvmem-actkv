#pragma once

#include <stdexcept>

namespace ninfer {

// A positioned read of a model file that failed while the Engine ran. The round that needed the
// bytes cannot finish, but the reader keeps nothing resident that the failed read would have
// supplied, so an Engine may fail the requests of that round and keep serving.
class FileReadError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

} // namespace ninfer

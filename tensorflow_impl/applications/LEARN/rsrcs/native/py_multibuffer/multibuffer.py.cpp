/**
 * @file   multibuffer.py.cpp
 * @author Sébastien Rouault <sebastien.rouault@epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 Sébastien ROUAULT.
 *
 * @section DESCRIPTION
 *
 * MRMW read-only/write-only atomic register with optional "blocking producer-consumer" semantic, specialized for 'Buffer', native interface.
**/

// External headers
#include <memory>
#include <new>
#include <utility>
extern "C" {
#include <stdlib.h>
}

// Internal headers
#include <common.hpp>
#include <buffer.hpp>
#include <multiregister.hpp>

// -------------------------------------------------------------------------- //
// MultiBuffer class aliases

using MultiBuffer = MultiRegister::MultiRegister<Buffer>;

// -------------------------------------------------------------------------- //
// Buffer native interfaces

/** Buffer allocation and initialization.
 * @param align Alignment of the following buffer
 * @param size  Size of the following buffer
 * @return Allocated and initialized Buffer instance, nullptr on failure
**/
extern "C" Buffer* buffer_create(size_t align, size_t size) noexcept {
    try {
        return new(::std::nothrow) Buffer(align, size);
    } catch (...) {
        return nullptr;
    }
}

/** Buffer clean-up and freeing.
 * @param buffer MultiBuffer instance to destroy
**/
extern "C" void buffer_destroy(Buffer* buffer) noexcept {
    try {
        delete buffer;
    } catch (...) {}
}

// -------------------------------------------------------------------------- //
// MultiBuffer native interfaces

/** MultiBuffer allocation and initialization.
 * @param length  Size of the above array
 * @param consume Whether there is only one reader per written buffer (optional)
 * @return Allocated and initialized MultiBuffer instance, nullptr on failure
**/
extern "C" MultiBuffer* multibuffer_create(size_t length, bool consume) noexcept {
    try {
        return new(::std::nothrow) MultiBuffer(length, consume);
    } catch (...) {
        return nullptr;
    }
}

/** MultiBuffer clean-up and freeing.
 * @param mb MultiBuffer instance to destroy
**/
extern "C" void multibuffer_destroy(MultiBuffer* mb) noexcept {
    try {
        delete mb;
    } catch (...) {}
}

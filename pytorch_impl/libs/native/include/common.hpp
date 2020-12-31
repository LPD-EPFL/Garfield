/**
 * @file   common.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * All rights reserved.
 *
 * @section DESCRIPTION
 *
 * Common declarations.
**/

#pragma once

// Compiler version check
#if __cplusplus < 201103L
    #error This translation unit requires at least a C++11 compiler
#endif
#ifndef __GNUC__
    #error This translation unit requires a GNU C++ compiler
#endif

// External headers
#ifndef NDEBUG
    #include <c10/util/Exception.h>
#endif
#include <cstdint>
#include <memory>
#include <type_traits>

// -------------------------------------------------------------------------- //
// Macro helpers

/** Define a proposition as likely true.
 * @param prop Proposition
**/
#undef likely
#define likely(prop) \
    __builtin_expect((prop) ? 1 : 0, 1)

/** Define a proposition as likely false.
 * @param prop Proposition
**/
#undef unlikely
#define unlikely(prop) \
    __builtin_expect((prop) ? 1 : 0, 0)

/** Assertion macros relying on <torch/extension.h>.
**/
#ifndef NDEBUG
    #define ASSERT(x, ...) TORCH_CHECK(static_cast<bool>(x), __VA_ARGS__)
#else
    #define ASSERT(x, ...)
#endif

// -------------------------------------------------------------------------- //
// Class helpers

/** Non copyable class.
**/
class NonCopyable {
public:
    /** Deleted copy constructor/assignment.
    **/
    NonCopyable(NonCopyable const&) = delete;
    NonCopyable& operator=(NonCopyable const&) = delete;
    /** Defaulted copy constructor/assignment.
    **/
    NonCopyable(NonCopyable&&) = default;
    NonCopyable& operator=(NonCopyable&&) = default;
    /** Default constructor.
    **/
    NonCopyable() = default;
};

/** Non copyable and movable class.
**/
class NonMoveable {
public:
    /** Defaulted copy constructor/assignment.
    **/
    NonMoveable(NonMoveable const&) = default;
    NonMoveable& operator=(NonMoveable const&) = default;
    /** Deleted copy constructor/assignment.
    **/
    NonMoveable(NonMoveable&&) = delete;
    NonMoveable& operator=(NonMoveable&&) = delete;
    /** Default constructor.
    **/
    NonMoveable() = default;
};

/** Non constructible class.
**/
class NonConstructible {
public:
    /** Deleted default constructor.
    **/
    NonConstructible() = delete;
};

/** Non instantiable class.
**/
class NonInstantiable: protected virtual NonCopyable, protected virtual NonMoveable, protected virtual NonConstructible {};
using Static = NonInstantiable;

// -------------------------------------------------------------------------- //
// Function helpers

/** Simple 'free'-calling deleter class.
**/
class VLFree final {
public:
    /** Free-calling deleter.
     * @param ptr Pointer to the memory to free
    **/
    void operator()(void* ptr) const noexcept {
        ::free(ptr);
    }
};

/** Variable-length (as in "non-constexpr const size") array class alias.
 * @param Type Element class
**/
template<class Type> using VLArray = ::std::unique_ptr<Type[], VLFree>;

/** Allocate a "variable" (as in "non-constexpr const size") length array of the given size.
 * @param Type Element class
 * @param size Number of elements
 * @return Pointer on the first element of the array
**/
template<class Type> static VLArray<Type> vlarray(size_t size) {
    auto align = alignof(typename ::std::remove_extent<Type>::type[size]); // Always a power of 2
    if (align < sizeof(void*))
        align = sizeof(void*);
    auto alloc = ::aligned_alloc(align, size * sizeof(Type));
    if (unlikely(!alloc))
        throw ::std::bad_alloc{};
    return VLArray<Type>{reinterpret_cast<typename VLArray<Type>::pointer>(alloc)}; // Constructors are all marked noexcept
}

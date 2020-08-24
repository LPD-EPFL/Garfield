/**
 * @file   exception.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * Exception base class and derived.
**/

#pragma once

// External headers
#include <cstdio>
#include <cstring>
#include <exception>

// Internal headers
#include <common.hpp>
#include <string_view.hpp>

namespace Exception {
// -------------------------------------------------------------------------- //
// Configuration

constexpr static size_t text_size = 256; // Max explanatory string size (not including '\0'), might be null

// -------------------------------------------------------------------------- //
// Helpers

/** Fixed-size string builder class.
 * @param ... Types of each argument
**/
template<class...> class Builder;
template<class Arg, class... Args> class Builder<Arg, Args...> final: private virtual NonInstantiable {
public:
    /** Push forwarder.
     * @param text   Text array
     * @param length Free length (not including the null-terminating character)
     * @param arg    Blob to push
     * @param args   Following blobs to push
     * @return Free length (not including the null-terminating character)
    **/
    static size_t build(char* text, size_t length, Arg&& arg, Args&&... args) noexcept {
        auto remaining = Builder<Arg>::build(text, length, arg);
        return Builder<Args...>::build(text + length - remaining, remaining, ::std::forward<Args>(args)...);
    }
};
template<class Arg> class Builder<Arg> final: private virtual NonInstantiable {
public:
    /** Push a string/boolean/integral/float/pointer.
     * @param text   Text array
     * @param length Free length (not including the null-terminating character)
     * @param value  Instance to (convert to string and) copy
     * @return Free length (not including the null-terminating character)
    **/
    static size_t build(char* text, size_t length, ::std::string_view const& value) noexcept {
        auto len = value.size();
        if (len > length) // Bound checking
            len = length;
        ::std::memcpy(text, value.data(), len); // Memory copy
        return length - len;
    }
    template<class Type,
             class Base = typename ::std::remove_cv<typename ::std::remove_reference<Type>::type>::type,
             typename ::std::enable_if<!::std::is_constructible<::std::string_view, Base>::value && ::std::is_same<Base, bool>::value>::type* = nullptr>
    static size_t build(char* text, size_t length, Type&& value) noexcept {
        return build(text, length, ::std::string_view{value ? "true" : "false"});
    }
    template<class Type,
             class Base = typename ::std::remove_cv<typename ::std::remove_reference<Type>::type>::type,
             typename ::std::enable_if<!::std::is_constructible<::std::string_view, Base>::value && !::std::is_same<Base, bool>::value && (::std::is_integral<Base>::value || ::std::is_floating_point<Base>::value || ::std::is_pointer<Base>::value)>::type* = nullptr>
    static size_t build(char* text, size_t length, Type&& value) noexcept {
        // Select format string
        char const* format;
        if (::std::is_integral<Base>::value) {
            if (::std::is_signed<Base>::value) { // Signed integral
                format = (sizeof(Base) > 4 ? "%ld" : "%d");
            } else { // Unsigned integral
                format = (sizeof(Base) > 4 ? "%lu" : "%u");
            }
        } else if (::std::is_floating_point<Base>::value) {
            format = "%f";
        } else { // is_pointer
            format = "%p";
        }
        // Format string
        auto written = ::std::snprintf(text, length + 1, format, value); // 'length + 1' since 'snprintf' includes the null-terminating character
        if (written < 0)
            return 0;
        return (static_cast<decltype(length)>(written) > length ? 0 : length - written); // Return the max
    }
};

// -------------------------------------------------------------------------- //
// Exception classes

/** Root exception class.
**/
class Exception: public ::std::exception {
private:
    char text[text_size + 1]; // Optional explanatory null-terminated string
private:
    /** Copy the given explanatory string.
     * @param explanatory Explanatory string to copy
    **/
    void copy(decltype(text) const& explanatory) noexcept {
        auto const* source = explanatory;
        auto*  destination = text;
        while (true) { // Partial string copy
            auto c = *(source++);
            *(destination++) = c;
            if (c == '\0')
                break;
        }
    }
public:
    /** Copy/move constructor/assignment.
     * @param exception Exception to copy
     * @return Current instance
    **/
    Exception(Exception const& exception) noexcept {
        copy(exception.text);
    }
    Exception(Exception&& exception) noexcept: Exception{const_cast<Exception const&>(exception)} {} // Necessary because of the template constructor
    Exception& operator=(Exception const& exception) noexcept {
        copy(exception.text);
        return *this;
    }
    /** No explanatory string constructor.
    **/
    explicit Exception() noexcept {
        text[0] = '\0';
    }
    /** String copy initialization string.
     * @param args Parameters to copy
    **/
    template<class... Args> explicit Exception(Args&&... args) noexcept {
        text[text_size - Builder<Args...>::build(text, text_size, ::std::forward<Args>(args)...)] = '\0';
    }
public:
    /** Returns the null-terminated explanatory string.
     * @return Null-terminated explanatory string
    **/
    virtual char const* what() const noexcept {
        return text;
    }
};

/** Derived exception classes.
 * @param PRNT Parent class
 * @param NAME Name of the derived exception class
 * @param WHAT Default description string
**/
#define DECLARE_EXCEPTION(PRNT, NAME, WHAT) \
    class NAME: public PRNT { \
    public: \
        explicit NAME() noexcept: PRNT{"[" #NAME "] ", WHAT} {} \
        template<class... Args> explicit NAME(Args&&... args) noexcept: PRNT{"[" #NAME "] ", ::std::forward<Args>(args)...} {} \
    }
DECLARE_EXCEPTION(Exception, Unreachable, "Unreachable code reached (bug!)");
DECLARE_EXCEPTION(Exception, Capacity,    "Unspecified capacity limit reached");
DECLARE_EXCEPTION(Exception, External,    "Unspecified exception using external library");

// -------------------------------------------------------------------------- //
}

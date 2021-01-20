/**
 * @file   buffer.hpp
 * @author Sébastien Rouault <sebastien.rouault@epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 Sébastien ROUAULT.
 *
 * @section DESCRIPTION
 *
 * Buffer and helper declarations.
**/

#pragma once

#include <cstdint>
#include <cstring>
#include <utility>
extern "C" {
#include <stdlib.h>
}

#include "common.hpp"

// -------------------------------------------------------------------------- //
// Buffer class

/** Simple aligned allocation management class.
**/
class Buffer final {
private:
    /** Header class.
    **/
    class Header final {
    public:
        size_t align; // Alignment of the following buffer
        size_t size;  // Size (in bytes) of the following buffer
    public:
        /** Initialization constructor.
         * @param align Alignment of the following buffer
         * @param size  Size of the following buffer
        **/
        constexpr Header(size_t align, size_t size) noexcept: align{align}, size{size} {}
    private:
        /** Compute the address of the following buffer.
         * @return Address of the following buffer
        **/
        uintptr_t get_buffer_address() const noexcept {
            auto addr = reinterpret_cast<uintptr_t>(this);
            addr += sizeof(Header);
            return (addr + align - 1) / align * align;
        }
    public:
        /** Compute a pointer to the following buffer, forward constness.
         * @return Pointer to the following buffer
        **/
        void* get_buffer() noexcept {
            return reinterpret_cast<void*>(get_buffer_address());
        }
        void const* get_buffer() const noexcept {
            return reinterpret_cast<void const*>(get_buffer_address());
        }
    };
private:
    Header* header; // Header followed by the buffer, nullptr if none
private:
    /** Allocated the associated buffer, none must exist.
     * @param align Alignment requirement, must be a power of 2
     * @param size  Size (in bytes) of the buffer, must be a multiple of 'align'
    **/
    void allocate(size_t align, size_t size) {
        // Full size computation (i.e. including the header then re-aligned)
        auto full_size = (size + sizeof(Header) + align - 1) / align * align;
        // Alignment correction
        if (align < alignof(Header))
            align = alignof(Header);
        if (align < sizeof(void*))
            align = sizeof(void*);
        // Aligned allocation
        auto alloc = ::aligned_alloc(align, full_size);
        if (unlikely(!alloc))
            throw ::std::bad_alloc{};
        // Final construction (noexcept, so no need to protect pointer 'alloc')
        header = new(alloc) Header{align, size};
    }
    /** Free the associated buffer, if any.
    **/
    void free() noexcept {
        if (header) {
            ::free(header);
            header = nullptr;
        }
    }
public:
    /** Explicit deep-copy constructor.
     * @param instance Instance to deep-copy
    **/
    explicit Buffer(Buffer const& instance) {
        if (instance.header) { // Buffer to deep-copy
            allocate(instance.header->align, instance.header->size);
            ::std::memcpy(header->get_buffer(), instance.header->get_buffer(), header->size);
        } else { // Buffer is empty
            header = nullptr;
        }
    }
    /** Deleted copy assignment.
    **/
    Buffer& operator=(Buffer const&) = delete;
    /** Move constructor/assignment.
     * @param instance Instance to move
     * @return This instance
    **/
    Buffer(Buffer&& instance) noexcept: header{::std::move(instance.header)} {
        instance.header = nullptr;
    }
    Buffer& operator=(Buffer&& instance) noexcept {
        free();
        header = instance.header;
        instance.header = nullptr;
        return *this;
    }
    /** Empty constructor.
    **/
    Buffer(): Buffer{nullptr} {};
    Buffer(decltype(nullptr)) noexcept: header{nullptr} {};
    /** Explicit aligned allocation constructor.
     * @param align Alignment requirement, must be a power of 2
     * @param size  Size (in bytes) of the buffer, must be a multiple of 'align'
    **/
    Buffer(size_t align, size_t size) {
        allocate(align, size);
    }
    /** Free destructor.
    **/
    ~Buffer() noexcept {
        free();
    }
public:
    /** Get a pointer to the header, forward constness.
     * @return Pointer to the header
    **/
    Header* get_header() noexcept {
        return header;
    }
    Header const* get_header() const noexcept {
        return const_cast<Header const*>(header);
    }
};

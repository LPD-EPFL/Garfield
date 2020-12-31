/**
 * @file   multiregister.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * All rights reserved.
 *
 * @section DESCRIPTION
 *
 * MRMW read-only/write-only atomic register with optional "blocking producer-consumer" semantic.
**/

#pragma once

// -------------------------------------------------------------------------- //
// Compiler version check

#if __cplusplus < 201103L
    #error This translation unit requires at least a C++11 compiler
#endif
#ifndef __GNUC__
    #error This translation unit requires a GNU C++ compiler
#endif

// -------------------------------------------------------------------------- //
// Includes

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

#include <common.hpp>
#include <exception.hpp>

// -------------------------------------------------------------------------- //
// Exception tree
namespace Exception {

/** Local exception branches.
**/
DECLARE_EXCEPTION(Exception, MultiRegister, "multi-replica exception");
    DECLARE_EXCEPTION(MultiRegister, MultiRegisterClosed, "multi-replica was/has been closed");
    DECLARE_EXCEPTION(MultiRegister, MultiRegisterTimeout, "wait for multi-replica timed out");
    DECLARE_EXCEPTION(MultiRegister, MultiRegisterUnreachable, "unreachable code reached (i.e. bug)");

}
// -------------------------------------------------------------------------- //
// Helper classes
namespace MultiRegister {

/** Counter class.
**/
using Count = uint_fast32_t;

/** Mutex class.
**/
using Lock = ::std::mutex;

/** Condition variable class.
**/
using Cond = ::std::condition_variable;

/** Replica status enum class.
**/
enum class Status {
    empty,    // No reader, no writer
    writing,  // Being written by 1 writer
    written,  // Wrote by 1 writer, no reader
    readable, // Being read by >= 0 reader(s), accept more readers
    reading   // Being read by >= 1 reader(s), does not accept more readers
};

/** Timeout duration (in ns) class.
**/
using Timeout = ::std::chrono::nanoseconds;

/** Dynamically-size, fixed array class.
 * @param Type Element class
**/
template<class Type> class Array final {
private:
    /** This class instance.
    **/
    using This = Array<Type>;
    /** Storage for the replica instances.
    **/
    using Store = typename ::std::aligned_storage<sizeof(Type), alignof(Type)>::type;
private:
    ::std::unique_ptr<Store[]> storage; // Storage array
public:
    size_t const length; // Length of the storage array
public:
    /** Deleted copy constructor/assignment.
    **/
    Array(This const&) = delete;
    This& operator=(This const&) = delete;
    /** Default move constructor/assignment.
    **/
    Array(This&&) = default;
    This& operator=(This&&) = default;
    /** Allocation and default constructor.
     * @param length Length of the array
    **/
    Array(size_t length): storage{new Store[length]}, length{length} {
        for (size_t i = 0; i < length; ++i)
            new(&storage[i]) Type();
    }
    /** Allocation and construction constructor.
     * @param length Length of the array
     * @param ...    Forwarded arguments to each element's constructor
    **/
    template<class... Args> Array(size_t length, Args const&... args): storage{new Store[length]}, length{length} {
        for (size_t i = 0; i < length; ++i)
            new(&storage[i]) Type(args...);
    }
    /** Element destructor.
    **/
    ~Array() noexcept(::std::is_nothrow_destructible<Type>::value) {
        if (storage) {
            for (size_t i = 0; i < length; ++i)
                reinterpret_cast<Type&>(storage[i]).~Type();
        }
    }
public:
    /** Get one element of the array (no bound check).
     * @param index Element index
    **/
    Type& get(size_t index) noexcept {
        return reinterpret_cast<Type&>(storage[index]);
    }
    Type const& get(size_t index) const noexcept {
        return reinterpret_cast<Type const&>(storage[index]);
    }
    Type& operator[](size_t index) noexcept {
        return get(index);
    }
    Type const& operator[](size_t index) const noexcept {
        return get(index);
    }
};

/** Optional class.
 * @param Type Underlying type
**/
template<class Type> class Optional final {
private:
    /** This class instance.
    **/
    using This = Optional<Type>;
    /** Storage class.
    **/
    using Storage = typename ::std::aligned_storage<sizeof(Type), alignof(Type)>::type;
private:
    Storage storage; // Storage for the optional value
    bool    valued;  // Whether the value is set
public:
    /** Deleted copy constructor/assignment.
    **/
    Optional(This const&) = delete;
    This& operator=(This const&) = delete;
    /** Move constructor/assignment.
     * @param instance Optional to swap with
     * @return Current instance
    **/
    Optional(This&& instance): valued{instance.valued} {
        if (valued)
            new(&storage) Type(::std::move(instance.storage));
    }
    This& operator=(This&& instance) {
        if (instance.valued) {
            operator=(::std::move(instance.get()));
        } else if (valued) {
            reset();
        }
        return *this;
    }
    /** Unvalued constructor.
    **/
    Optional(): valued{false} {}
    /** Valued constructor/assignment.
     * @param value Value to move
    **/
    Optional(Type&& value): valued{true} {
        new(&storage) Type(::std::move(value));
    }
    This& operator=(Type&& value) {
        if (valued) {
            get() = ::std::move(value);
        } else {
            new(&storage) Type(::std::move(value));
            valued = true;
        }
        return *this;
    }
public:
    /** Check whether a value is set.
     * @return Whether a value is set
    **/
    operator bool() const noexcept {
        return valued;
    }
    /** Get the current value (which must be set).
     * @return Current value
    **/
    Type& get() noexcept {
        return reinterpret_cast<Type&>(storage);
    }
    Type const& get() const noexcept {
        return reinterpret_cast<Type const&>(storage);
    }
    /** Reset contained value (which must exist), mark as containing no value.
    **/
    void reset() {
        reinterpret_cast<Type&>(storage).~Type();
        valued = false;
    }
};

}
// -------------------------------------------------------------------------- //
// Base class
namespace MultiRegister {

/** MRMW read-only/write-only atomic register class.
 * @param Type Underlying register type
**/
template<class Type> class MultiRegister {
protected:
    /** Stateful, reference-counted replica class.
    **/
    class Replica final {
    public:
        Status status; // Current status
        Count readers; // Number of readers
        Type     data; // Stored data
    public:
        /** Deleted copy constructor/assignment.
        **/
        Replica(Replica const&) = delete;
        Replica& operator=(Replica const&) = delete;
        /** Forwarding constructor.
         * @param ... Forwarded arguments
        **/
        template<class... Args> Replica(Args&&... args): status{Status::empty}, readers{0}, data{::std::forward<Args>(args)...} {}
    };
    /** Array of replicas class.
    **/
    using Replicas = Array<Replica>;
public:
    /** Replica reader management class.
    **/
    class Read final {
    private:
        MultiRegister* multi; // Bound multi-replica, 'nullptr' for unbound
        Replica*     replica; // Readable replica
    public:
        /** Deleted copy constructor/assignment.
        **/
        Read(Read const&) = delete;
        Read& operator=(Read const&) = delete;
        /** Move constructor/assignment.
         * @param instance Instance to move from
         * @return Current instance
        **/
        Read(Read&& instance): multi{instance.multi}, replica{instance.replica} {
            instance.multi = nullptr;
        }
        Read& operator=(Read&& instance) {
            if (multi)
                release();
            multi = instance.multi;
            replica = instance.replica;
            instance.multi = nullptr;
            return *this;
        }
        /** Bind constructor.
         * @param multi  Bound multi-replica
         * @param replica Readable replica
        **/
        Read(MultiRegister& multi, Replica& replica): multi{::std::addressof(multi)}, replica{::std::addressof(replica)} {}
        /** Release destructor.
        **/
        ~Read() noexcept(false) {
            if (!multi) // Unbound, nothing to do
                return;
            release();
        }
    private:
        /** Release the replica, must be bound.
        **/
        void release() {
            ::std::unique_lock<Lock> locked{multi->lock};
            if (--multi->nbused == 0 && multi->closed)
                multi->condwr.notify_all();
            auto remain = --replica->readers;
            switch (replica->status) {
            case Status::readable:
                break;
            case Status::reading:
                if (remain == 0) {
                    replica->status = Status::empty;
                    multi->condwr.notify_one();
                }
                break;
            default:
                throw Exception::MultiRegisterUnreachable{};
            }
        }
    public:
        /** Access the underlying data, undefined behavior if unbound instance.
         * @return Underlying data
        **/
        Type const& get() const noexcept {
            return replica->data;
        }
        /** Move-access the underlying data, undefined behavior if unbound instance.
         * @return Underlying data
        **/
        Type&& move() const noexcept {
            return ::std::move(replica->data);
        }
        /** Overwrite the underlying data, behavior is undefined if the register is not in "consume" mode, undefined behavior if unbound instance.
         * @param data New data
        **/
        void reset(Type&& data) {
            replica->data = ::std::move(data);
        }
    };
    /** Replica writer management class.
    **/
    class Write final {
    private:
        MultiRegister* multi; // Bound multi-replica, 'nullptr' if unbound
        Replica*     replica; // Writable replica
        size_t        offset; // Offset of this replica
        bool       validated; // Commit has been validated
    public:
        /** Deleted copy constructor/assignment.
        **/
        Write(Write const&) = delete;
        Write& operator=(Write const&) = delete;
        /** Move constructor/assignment.
         * @param instance Instance to move from
         * @return Current instance
        **/
        Write(Write&& instance): multi{instance.multi}, replica{instance.replica}, offset{instance.offset}, validated{instance.validated} {
            instance.multi = nullptr;
        }
        Write& operator=(Write&& instance) {
            if (multi)
                release();
            multi = instance.multi;
            replica = instance.replica;
            offset = instance.offset;
            validated = instance.validated;
            instance.multi = nullptr;
            return *this;
        }
        /** Bind constructor.
         * @param multi  Bound multi-replica
         * @param replica Writable replica
         * @param offset Offset of this replica
        **/
        Write(MultiRegister& multi, Replica& replica, size_t offset): multi{::std::addressof(multi)}, replica{::std::addressof(replica)}, offset{offset}, validated{false} {}
        /** Commit/abort destructor.
        **/
        ~Write() noexcept(false) {
            if (!multi) // Unbound, nothing to do
                return;
            release();
        }
    private:
        /** Release the replica, must be bound.
        **/
        void release() {
            ::std::unique_lock<Lock> locked{multi->lock};
            if (--multi->nbused == 0 && multi->closed)
                multi->condwr.notify_all();
            switch (replica->status) {
            case Status::writing:
                if (validated) {
                    replica->status = Status::written;
                    if (multi->offset < offset)
                        multi->offset = offset;
                    if (multi->consume) {
                        multi->condrd.notify_one();
                    } else {
                        multi->condrd.notify_all();
                    }
                } else {
                    replica->status = Status::empty;
                }
                break;
            default:
                throw Exception::MultiRegisterUnreachable{};
            }
        }
    public:
        /** Access the underlying data, can be any value, undefined behavior if unbound instance.
         * @return Underlying data
        **/
        Type& peek() const noexcept {
            return replica->data;
        }
        /** Overwrite the underlying data, undefined behavior if unbound instance.
         * @param data New data
        **/
        void set(Type&& data) {
            replica->data = ::std::move(data);
        }
        /** Validate the write, undefined behavior if unbound instance.
        **/
        void validate() noexcept {
            validated = true;
        }
    };
protected:
    Lock lock;   // Access lock
    Cond condrd; // To notify read-ability
    Cond condwr; // To notify write-ability and 'nbused' reaching 0
    Array<Replica> replicas; // Array of replicas
    size_t offset; // Array offset of last written (except when all empty), readers read it, writers write after
    size_t nbused; // Number of threads using this instance
    bool   closed; // This instance is closed
public:
    bool const consume; // Whether there is only one reader per written replica
public:
    /** Close the multi-register, if not already closed.
    **/
    void close() {
        ::std::unique_lock<Lock> locked{lock};
        if (closed)
            return;
        closed = true;
        condrd.notify_all();
        condwr.notify_all();
    }
public:
    /** Deleted copy constructor/assignment.
    **/
    MultiRegister(MultiRegister const&) = delete;
    MultiRegister& operator=(MultiRegister const&) = delete;
    /** Bind constructor.
     * @param length  Number of replicas
     * @param consume Whether there is only one reader per written replica (optional)
     * @param ...     Forwarded arguments
    **/
    template<class... Args> MultiRegister(size_t length, bool consume = false, Args&&... args): lock{}, condrd{}, condwr{}, replicas{length, ::std::forward<Args>(args)...}, offset{0}, nbused{0}, closed{false}, consume{consume} {}
    /** Wake and wait destructor.
    **/
    ~MultiRegister() noexcept(false) {
        close();
        { // Wait for the concurrent threads
            ::std::unique_lock<Lock> locked{lock};
            while (nbused > 0)
                condwr.wait(locked);
        }
    }
protected:
    /** Assert that this instance is open.
    **/
    void assert_open() const {
        if (unlikely(closed))
            throw Exception::MultiRegisterClosed{};
    }
    /** Wait for a condition variable, must be open, assert that still open.
     * @param locked  Owned lock to use
     * @param cond    Condition variable to wait for
     * @param timeout Optional timeout to satisfy
    **/
    void wait(::std::unique_lock<Lock>& locked, Cond& cond, Optional<Timeout> const& timeout) {
        ++nbused;
        if (timeout) {
            if (unlikely(cond.wait_for(locked, timeout.get()) == ::std::cv_status::timeout))
                throw Exception::MultiRegisterTimeout{};
        } else {
            cond.wait(locked);
        }
        if (--nbused == 0 && closed)
            condwr.notify_all();
        assert_open();
    }
public:
    /** [thread-safe] Get a read replica, wait for it if none available.
     * @param timeout Maximal duration to wait, in ns (optional)
     * @return Replica reader management instance
    **/
    Read read(Optional<Timeout> const& timeout = Optional<Timeout>{}) {
        ::std::unique_lock<Lock> locked{lock};
        assert_open();
        while (true) {
            auto& replica = replicas[offset % replicas.length];
            switch (replica.status) {
            case Status::written:
                replica.status = (consume ? Status::reading : Status::readable);
                [[fallthrough]];
            case Status::readable:
                ++replica.readers;
                ++nbused;
                return Read{*this, replica};
            default: // Empty, writing or reading
                wait(locked, condrd, timeout);
            }
        }
    }
    /** [thread-safe] Get a write replica, wait for it if none available.
     * @param timeout Maximal duration to wait, in ns (optional)
     * @return Replica writer management instance
    **/
    Write write(Optional<Timeout> const& timeout = Optional<Timeout>{}) {
        ::std::unique_lock<Lock> locked{lock};
        assert_open();
        decltype(offset) position = offset;
        decltype(offset) readable = 0; // Offset of the first readable (with readers) to switch to 'reading' if no slot available
        bool   found_readable = false; // Whether a readable slot (with readers) has been found
        while (true) {
            ++position;
            auto& replica = replicas[position % replicas.length];
            switch (replica.status) {
            case Status::empty:
                [[fallthrough]];
            case Status::written:
            begin_write:
                replica.status = Status::writing;
                ++nbused;
                return Write{*this, replica, position};
            case Status::readable:
                if (replica.readers == 0)
                    goto begin_write;
                if (!found_readable) {
                    readable = position;
                    found_readable = true;
                }
                [[fallthrough]];
            default: // Writing or reading
                if (position == offset + replicas.length) { // Should wait since no slot available
                    if (found_readable) // A slot was being read, flush readers
                        replicas[readable % replicas.length].status = Status::reading;
                    wait(locked, condwr, timeout);
                    // Reset to re-check every slot
                    found_readable = false;
                    position = offset;
                }
            }
        }
    }
};

}

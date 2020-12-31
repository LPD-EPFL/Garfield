/**
 * @file   threadpool.cpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * All rights reserved.
 *
 * @section DESCRIPTION
 *
 * Another thread pool management class, with parallel for-loop helper.
**/

// Compiler version check
#if __cplusplus < 201103L
    #error This translation unit requires at least a C++11 compiler
#endif
#ifndef __GNUC__
    #error This translation unit requires a GNU C++ compiler
#endif

// External headers
#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
extern "C" {
#include <unistd.h>
}

// Internal headers
#include <common.hpp>
#include <threadpool.hpp>

// -------------------------------------------------------------------------- //

/** Lock guard class.
**/
template<class Lock> using Guard = ::std::unique_lock<Lock>;

/** Lock "reverse" guard class.
**/
template<class Lock> class Forsake {
private:
    Lock& lock; // Bound lock
public:
    /** Deleted copy constructor/assignment.
    **/
    Forsake(Forsake const&) = delete;
    Forsake& operator=(Forsake const&) = delete;
    /** Lock release constructor.
     * @param lock Lock to bind
    **/
    Forsake(Lock& lock): lock{lock} {
        lock.unlock();
    }
    /** Lock acquire destructor.
    **/
    ~Forsake() {
        lock.lock();
    }
};

/** Worker thread entry point.
 * @param self Bound thread pool
**/
void ThreadPool::entry_point(ThreadPool& self) {
    Guard<Mutex> guard{self.lock};
    while (true) { // Main loop
        if (self.status == Status::detached) // Must terminate
            return;
        if (!self.jobs.empty()) { // Pick and run job
            auto& job = *self.jobs.front(); // Get job info
            self.jobs.pop(); // Remove job
            { // Run job
                Forsake<Mutex> forsake{self.lock};
                job();
            }
            continue;
        }
        self.cv.wait(guard); // Wait for event
    }
}

/** Get the default number of worker threads to use.
 * @return Default number of worker threads
**/
size_t ThreadPool::get_default_nbworker() noexcept {
    auto res = ::sysconf(_SC_NPROCESSORS_ONLN);
    if (unlikely(res <= 0))
        return 8; // Arbitrary default
    return res;
}

// -------------------------------------------------------------------------- //

/** Thread pool begin constructor.
 * @param nbworkers Number of worker threads to use (optional, 0 for auto)
**/
ThreadPool::ThreadPool(size_t nbworkers): size{nbworkers}, status{Status::running}, lock{}, cv{}, jobs{}, threads{} {
    if (nbworkers == 0)
        size = get_default_nbworker();
    { // Worker threads creation
        threads.reserve(size);
        for (decltype(size) i = 0; i < size; ++i)
            threads.emplace_back(entry_point, ::std::ref(*this));
    }
}

/** Notify then detach worker threads destructor.
**/
ThreadPool::~ThreadPool() {
    { // Mark ending
        Guard<Mutex> guard{lock};
        status = Status::detached;
    }
    cv.notify_all();
    for (auto&& thread: threads) // Detach workers
        thread.detach();
}

/** Submit a new job.
 * @param job Job to submit
**/
void ThreadPool::submit(AbstractJob& job) {
    { // Submit one job
        Guard<Mutex> guard{lock};
        jobs.push(::std::addressof(job));
    }
    cv.notify_one(); // Notify one worker
}

// -------------------------------------------------------------------------- //

// Shared thread pool
ThreadPool pool;

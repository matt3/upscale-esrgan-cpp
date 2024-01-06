#pragma once
#include <utility>

template<class T>
struct TE{};

template<class T, int T_Size, int Wrapper_Size>
static constexpr void checkSize() {
    static_assert(Wrapper_Size >= T_Size , "Size too small");
}

// Non-movable object of size less than or equal to Size
template<int Size>
class TE0 {
public:
    template<class T, class... Args>
    TE0(TE<T> type, Args&&... args) {
        checkSize<T, sizeof(T), Size>();
        // Instantiate object of type T with arguments args...
        new(data) T(std::forward<Args>(args)...);
        destructor = [](void* ptr){ static_cast<T*>(ptr)->~T(); };
    }
    ~TE0() {
        destructor(data);
    }
    template<class T>
    T& get() {
        return *reinterpret_cast<T*>(data);
    }
    template<class T>
    const T& get() const{
        return *reinterpret_cast<const T*>(data);
    }
private:
    void (*destructor)(void*);
    char data[Size];
};

// Movable wrapper for object on the heap
class TE1 {
public:
    template<class T, class... Args>
    TE1(TE<T> type, Args&&... args) {
        // Instantiate object of type T with arguments args...
        data = new T(std::forward<Args>(args)...);
        destructor = [](void* ptr){ delete static_cast<T*>(ptr); };
    }
    TE1(TE1&& other): data(other.data) {
        other.data = nullptr;
    }
    ~TE1() {
        if(data) {
            destructor(data);
        }
    }
    template<class T>
    T& get() {
        return *reinterpret_cast<T*>(data);
    }
    template<class T>
    const T& get() const{
        return *reinterpret_cast<const T*>(data);
    }
private:
    void (*destructor)(void*);
    void* data;
};

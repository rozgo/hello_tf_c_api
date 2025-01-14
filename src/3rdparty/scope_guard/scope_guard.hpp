//   _____                         _____                     _    _____
//  / ____|                       / ____|                   | |  / ____|_     _
// | (___   ___ ___  _ __   ___  | |  __ _   _  __ _ _ __ __| | | |   _| |_ _| |_
//  \___ \ / __/ _ \| '_ \ / _ \ | | |_ | | | |/ _` | '__/ _` | | |  |_   _|_   _|
//  ____) | (_| (_) | |_) |  __/ | |__| | |_| | (_| | | | (_| | | |____|_|   |_|
// |_____/ \___\___/| .__/ \___|  \_____|\__,_|\__,_|_|  \__,_|  \_____|
//                  | | https://github.com/Neargye/scope_guard
//                  |_| vesion 0.5.0
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018 - 2019 Daniil Goncharov <neargye@gmail.com>.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef NEARGYE_SCOPE_GUARD_HPP
#define NEARGYE_SCOPE_GUARD_HPP

#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>
#if (defined(_MSC_VER) && _MSC_VER >= 1900) || ((defined(__clang__) || defined(__GNUC__)) && __cplusplus >= 201700L)
#include <exception>
#endif

// scope_guard throwable settings:
// SCOPE_GUARD_MAY_EXCEPTIONS action may throw exceptions.
// SCOPE_GUARD_NO_EXCEPTIONS require noexcept action.
// SCOPE_GUARD_SUPPRESS_EXCEPTIONS exceptions during action will be suppressed.

#if !defined(SCOPE_GUARD_MAY_EXCEPTIONS) && !defined(SCOPE_GUARD_NO_EXCEPTIONS) && !defined(SCOPE_GUARD_SUPPRESS_EXCEPTIONS)
#  define SCOPE_GUARD_MAY_EXCEPTIONS
#elif (defined(SCOPE_GUARD_MAY_EXCEPTIONS) + defined(SCOPE_GUARD_NO_EXCEPTIONS) + defined(SCOPE_GUARD_SUPPRESS_EXCEPTIONS)) > 1
#  error Only one of SCOPE_GUARD_MAY_EXCEPTIONS and SCOPE_GUARD_NO_EXCEPTIONS and SCOPE_GUARD_SUPPRESS_EXCEPTIONS may be defined.
#endif

#if defined(SCOPE_GUARD_NO_EXCEPTIONS)
#  define SCOPE_GUARD_ACTION_NOEXCEPT noexcept
#else
#  define SCOPE_GUARD_ACTION_NOEXCEPT
#endif

#if defined(SCOPE_GUARD_SUPPRESS_EXCEPTIONS)
#  define SCOPE_GUARD_NOEXCEPT(...) noexcept
#  define SCOPE_GUARD_TRY try {
#  define SCOPE_GUARD_CATCH } catch (...) {}
#else
#  define SCOPE_GUARD_NOEXCEPT(...) noexcept(__VA_ARGS__)
#  define SCOPE_GUARD_TRY
#  define SCOPE_GUARD_CATCH
#endif

namespace scope_guard {

namespace detail {

#if defined(_MSC_VER) && _MSC_VER < 1900
inline int uncaught_exceptions() noexcept {
  return *(reinterpret_cast<int*>(static_cast<char*>(static_cast<void*>(_getptd())) + (sizeof(void*) == 8 ? 0x100 : 0x90)));
}
#elif (defined(__clang__) || defined(__GNUC__)) && __cplusplus < 201700L
struct __cxa_eh_globals;
extern "C" __cxa_eh_globals* __cxa_get_globals() noexcept;
inline int uncaught_exceptions() noexcept {
  return *(reinterpret_cast<unsigned int*>(static_cast<char*>(static_cast<void*>(__cxa_get_globals())) + sizeof(void*)));
}
#else
inline int uncaught_exceptions() noexcept {
  return std::uncaught_exceptions();
}
#endif

struct on_exit_policy final {
  bool execute_;

  explicit on_exit_policy(bool execute) noexcept : execute_(execute) {}

  void dismiss() noexcept {
    execute_ = false;
  }

  bool should_execute() const noexcept {
    return execute_;
  }
};

struct on_fail_policy final {
  int ec_;

  explicit on_fail_policy(bool execute) noexcept : ec_(execute ? uncaught_exceptions() : -1) {}

  void dismiss() noexcept {
    ec_ = -1;
  }

  bool should_execute() const noexcept {
    return ec_ != -1 && ec_ < uncaught_exceptions();
  }
};

struct on_success_policy final {
  int ec_;

  explicit on_success_policy(bool execute) noexcept : ec_(execute ? uncaught_exceptions() : -1) {}

  void dismiss() noexcept {
    ec_ = -1;
  }

  bool should_execute() const noexcept {
    return ec_ != -1 && ec_ >= uncaught_exceptions();
  }
};

template <typename F, typename P>
class scope_guard final {
  using A = typename std::decay<F>::type;
  using invoke_action_result_t = decltype((std::declval<A>())());
  using is_nothrow_invocable_action = std::integral_constant<bool, noexcept((std::declval<A>())())>;

  static_assert(std::is_same<void, invoke_action_result_t>::value,
                "scope_guard require no-argument action returns void.");
  static_assert(std::is_same<P, on_exit_policy>::value ||
                    std::is_same<P, on_fail_policy>::value ||
                    std::is_same<P, on_success_policy>::value,
                "scope_guard require on_exit_policy, on_fail_policy or on_success_policy.");

 public:
  scope_guard() = delete;
  scope_guard(const scope_guard&) = delete;
  scope_guard& operator=(const scope_guard&) = delete;
  scope_guard& operator=(scope_guard&&) = delete;

  scope_guard(scope_guard&& other) noexcept(std::is_nothrow_move_constructible<A>::value || std::is_nothrow_copy_constructible<A>::value)
      : policy_(false),
        action_(std::move_if_noexcept(other.action_)) {
    policy_ = std::move(other.policy_);
    other.policy_.dismiss();
  }

  explicit scope_guard(A&& action) noexcept(std::is_nothrow_move_constructible<A>::value)
      : policy_(true),
        action_(std::move(action)) {}

  explicit scope_guard(const A& action) noexcept(std::is_nothrow_copy_constructible<A>::value)
      : policy_(true),
        action_(action) {}

  void dismiss() noexcept {
    policy_.dismiss();
  }

  ~scope_guard() SCOPE_GUARD_NOEXCEPT(is_nothrow_invocable_action::value) {
#if defined(SCOPE_GUARD_NO_EXCEPTIONS)
    static_assert(is_nothrow_invocable_action::value, "scope_guard require noexcept action");
#endif
    if (policy_.should_execute()) {
      SCOPE_GUARD_TRY
        action_();
      SCOPE_GUARD_CATCH
    }
  }

 private:
  P policy_;
  A action_;

  void* operator new(std::size_t) = delete;
  void operator delete(void*) = delete;
};

} // namespace scope_guard::detail

template <typename F>
using scope_exit = detail::scope_guard<F, detail::on_exit_policy>;

template <typename F>
using scope_fail = detail::scope_guard<F, detail::on_fail_policy>;

template <typename F>
using scope_succes = detail::scope_guard<F, detail::on_success_policy>;

// ATTR_NODISCARD encourages the compiler to issue a warning if the return value is discarded.
#if !defined(ATTR_NODISCARD)
#  if defined(__clang__)
#    if (__clang_major__ * 10 + __clang_minor__) >= 39 && __cplusplus >= 201703L
#      define ATTR_NODISCARD [[nodiscard]]
#    else
#      define ATTR_NODISCARD __attribute__((__warn_unused_result__))
#    endif
#  elif defined(__GNUC__)
#    if __GNUC__ >= 7 && __cplusplus >= 201703L
#      define ATTR_NODISCARD [[nodiscard]]
#    else
#      define ATTR_NODISCARD __attribute__((__warn_unused_result__))
#    endif
#  elif defined(_MSC_VER)
#    if _MSC_VER >= 1911 && defined(_MSVC_LANG) && _MSVC_LANG >= 201703L
#      define ATTR_NODISCARD [[nodiscard]]
#    elif defined(_Check_return_)
#      define ATTR_NODISCARD _Check_return_
#    else
#      define ATTR_NODISCARD
#    endif
#  else
#    define ATTR_NODISCARD
#  endif
#endif

template <typename F>
ATTR_NODISCARD scope_exit<F> make_scope_exit(F&& action) noexcept(noexcept(scope_exit<F>(std::forward<F>(action)))) {
  return scope_exit<F>(std::forward<F>(action));
}

template <typename F>
ATTR_NODISCARD scope_fail<F> make_scope_fail(F&& action) noexcept(noexcept(scope_fail<F>(std::forward<F>(action)))) {
  return scope_fail<F>(std::forward<F>(action));
}

template <typename F>
ATTR_NODISCARD scope_succes<F> make_scope_succes(F&& action) noexcept(noexcept(scope_succes<F>(std::forward<F>(action)))) {
  return scope_succes<F>(std::forward<F>(action));
}

namespace detail {

struct scope_exit_tag {};

template <typename F>
scope_exit<F> operator+(scope_exit_tag, F&& action) noexcept(noexcept(scope_exit<F>(std::forward<F>(action)))) {
  return scope_exit<F>(std::forward<F>(action));
}

struct scope_fail_tag {};

template <typename F>
scope_fail<F> operator+(scope_fail_tag, F&& action) noexcept(noexcept(scope_fail<F>(std::forward<F>(action)))) {
  return scope_fail<F>(std::forward<F>(action));
}

struct scope_succes_tag {};

template <typename F>
scope_succes<F> operator+(scope_succes_tag, F&& action) noexcept(noexcept(scope_succes<F>(std::forward<F>(action)))) {
  return scope_succes<F>(std::forward<F>(action));
}

} // namespace scope_guard::detail

} // namespace scope_guard

// ATTR_MAYBE_UNUSED suppresses compiler warnings on unused entities, if any.
#if !defined(ATTR_MAYBE_UNUSED)
#  if defined(__clang__)
#    if (__clang_major__ * 10 + __clang_minor__) >= 39 && __cplusplus >= 201703L
#      define ATTR_MAYBE_UNUSED [[maybe_unused]]
#    else
#      define ATTR_MAYBE_UNUSED __attribute__((__unused__))
#    endif
#  elif defined(__GNUC__)
#    if __GNUC__ >= 7 && __cplusplus >= 201703L
#      define ATTR_MAYBE_UNUSED [[maybe_unused]]
#    else
#      define ATTR_MAYBE_UNUSED __attribute__((__unused__))
#    endif
#  elif defined(_MSC_VER)
#    if _MSC_VER >= 1911 && defined(_MSVC_LANG) && _MSVC_LANG >= 201703L
#      define ATTR_MAYBE_UNUSED [[maybe_unused]]
#    else
#      define ATTR_MAYBE_UNUSED __pragma(warning(suppress : 4100 4101 4189))
#    endif
#  else
#    define ATTR_MAYBE_UNUSED
#  endif
#endif

#define SCOPE_GUARD_STR_CONCAT_(s1, s2) s1##s2
#define SCOPE_GUARD_STR_CONCAT(s1, s2) SCOPE_GUARD_STR_CONCAT_(s1, s2)

#if defined(__COUNTER__)
#  define SCOPE_GUARD_COUNTER __COUNTER__
#elif defined(__LINE__)
#  define SCOPE_GUARD_COUNTER __LINE__
#endif

// SCOPE_EXIT executing action on scope exit.
#define MAKE_SCOPE_EXIT(name) auto name = ::scope_guard::detail::scope_exit_tag{} + [&]() SCOPE_GUARD_ACTION_NOEXCEPT -> void
#define SCOPE_EXIT        \
  ATTR_MAYBE_UNUSED const \
  MAKE_SCOPE_EXIT(SCOPE_GUARD_STR_CONCAT(__scope_exit__object_, SCOPE_GUARD_COUNTER))

// SCOPE_FAIL executing action on scope exit when an exception has been thrown before scope exit.
#define MAKE_SCOPE_FAIL(name) auto name = ::scope_guard::detail::scope_fail_tag{} + [&]() SCOPE_GUARD_ACTION_NOEXCEPT -> void
#define SCOPE_FAIL        \
  ATTR_MAYBE_UNUSED const \
  MAKE_SCOPE_FAIL(SCOPE_GUARD_STR_CONCAT(__scope_fail__object_, SCOPE_GUARD_COUNTER))

// SCOPE_SUCCESS executing action on scope exit when no exceptions have been thrown before scope exit.
#define MAKE_SCOPE_SUCCESS(name) auto name = ::scope_guard::detail::scope_succes_tag{} + [&]() SCOPE_GUARD_ACTION_NOEXCEPT -> void
#define SCOPE_SUCCESS     \
  ATTR_MAYBE_UNUSED const \
  MAKE_SCOPE_SUCCESS(SCOPE_GUARD_STR_CONCAT(__scope_succes__object_, SCOPE_GUARD_COUNTER))

#define MAKE_DEFER(name) MAKE_SCOPE_EXIT(name)
#define DEFER SCOPE_EXIT

#endif // NEARGYE_SCOPE_GUARD_HPP

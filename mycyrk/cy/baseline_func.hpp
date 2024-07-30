#pragma once

#include <memory>

#include "common.hpp"
#include "cysolverbase_class.hpp"
#include "rk.hpp"


std::shared_ptr<CySolverResult> baseline_cysolve_ivp(
    DiffeqFuncType diffeq_ptr,
    const double* t_span_ptr,
    const double* y0_ptr,
    const unsigned int num_y,
    const unsigned int method,
    // General optional arguments
    const size_t expected_size = 0,
    const unsigned int num_extra = 0,
    const void* args_ptr = nullptr,
    const size_t max_num_steps = 0,
    const size_t max_ram_MB = 2000,
    const bool dense_output = false,
    const double* t_eval = nullptr,
    const size_t len_t_eval = 0,
    PreEvalFunc pre_eval_func = nullptr,
    // rk optional arguments
    const double rtol = 1.0e-3,
    const double atol = 1.0e-6,
    const double* rtols_ptr = nullptr,
    const double* atols_ptr = nullptr,
    const double max_step_size = MAX_STEP,
    const double first_step_size = 0.0
);

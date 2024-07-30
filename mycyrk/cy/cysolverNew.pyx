# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
np.import_array()

# =====================================================================================================================
# Import CySolverResult (container for integration results)
# =====================================================================================================================
cdef class WrapCySolverResult:

    cdef void set_cyresult_pointer(self, shared_ptr[CySolverResult] cyresult_shptr):

        # Store c++ based result and pull out key information
        self.cyresult_shptr = cyresult_shptr
        self.cyresult_ptr   = cyresult_shptr.get()
        self.size           = self.cyresult_ptr[0].size
        self.num_dy         = self.cyresult_ptr[0].num_dy

        # Convert solution to pointers and views
        if self.cyresult_ptr.size > 0:
            self.time_ptr  = &self.cyresult_ptr.time_domain[0]
            self.y_ptr     = &self.cyresult_ptr.solution[0]
            self.time_view = <double[:self.size]>self.time_ptr
            self.y_view    = <double[:self.size * self.num_dy]>self.y_ptr
    
    def call(self, double t):
        """ Call the dense output interpolater and return y """

        if not self.cyresult_ptr.capture_dense_output:
            raise AttributeError("Can not call WrapCySolverResult when dense_output set to False.")

        y_interp_array = np.empty(self.cyresult_ptr.num_y, dtype=np.float64, order='C')
        cdef double[::1] y_interp_view = y_interp_array
        cdef double* y_interp_ptr      = &y_interp_view[0]

        self.cyresult_ptr.call(t, y_interp_ptr)
        return y_interp_array
    
    def call_vectorize(self, double[::1] t_view):
        """ Call the dense output interpolater and return y """

        if not self.cyresult_ptr.capture_dense_output:
            raise AttributeError("Can not call WrapCySolverResult when dense_output set to False.")

        cdef size_t len_t = len(t_view)

        y_interp_array = np.empty(self.cyresult_ptr.num_y * len_t, dtype=np.float64, order='C')
        cdef double[::1] y_interp_view = y_interp_array
        cdef double* y_interp_ptr = &y_interp_view[0]
        cdef double* t_array_ptr  = &t_view[0]

        self.cyresult_ptr.call_vectorize(t_array_ptr, len_t, y_interp_ptr)
        return y_interp_array.reshape(len_t, self.cyresult_ptr.num_y).T

    @property
    def success(self):
        return self.cyresult_ptr.success
        
    @property
    def message(self):
        return str(self.cyresult_ptr.message_ptr, 'UTF-8')
    
    @property
    def t(self):
        return np.asarray(self.time_view, dtype=np.float64, order='C')
    
    @property
    def y(self):
        return np.asarray(self.y_view, dtype=np.float64, order='C').reshape((self.size, self.num_dy)).T
    
    @property
    def size(self):
        return self.cyresult_ptr.size
    
    @property
    def error_code(self):
        return self.cyresult_ptr.error_code
    
    def __call__(self, t):

        if type(t) == np.ndarray:
            return self.call_vectorize(t)
        else:
            return self.call(t).reshape(self.cyresult_ptr.num_y, 1)

# =====================================================================================================================
# Create Wrapped cysolve_ivp (has various defaults)
# =====================================================================================================================

cdef CySolveOutput cysolve_ivp(
            DiffeqFuncType diffeq_ptr,
            const double* t_span_ptr,
            const double* y0_ptr,
            const unsigned int num_y,
            unsigned int method = 1,
            double rtol = 1.0e-3,
            double atol = 1.0e-6,
            void* args_ptr = NULL,
            unsigned int num_extra = 0,
            size_t max_num_steps = 0,
            size_t max_ram_MB = 2000,
            bint dense_output = False,
            double* t_eval = NULL,
            size_t len_t_eval = 0,
            PreEvalFunc pre_eval_func = NULL,
            double* rtols_ptr = NULL,
            double* atols_ptr = NULL,
            double max_step = MAX_STEP,
            double first_step = 0.0,
            size_t expected_size = 0
            ) noexcept nogil:
    
    cdef CySolveOutput result = baseline_cysolve_ivp(
        diffeq_ptr,
        t_span_ptr,
        y0_ptr,
        num_y,
        method,
        expected_size,
        num_extra,
        args_ptr,
        max_num_steps,
        max_ram_MB,
        dense_output,
        t_eval,
        len_t_eval,
        pre_eval_func,
        rtol,
        atol,
        rtols_ptr,
        atols_ptr,
        max_step,
        first_step
        )

    return result

cdef CySolveOutput cysolve_ivp_gil(
            DiffeqFuncType diffeq_ptr,
            const double* t_span_ptr,
            const double* y0_ptr,
            const unsigned int num_y,
            unsigned int method = 1,
            double rtol = 1.0e-3,
            double atol = 1.0e-6,
            void* args_ptr = NULL,
            unsigned int num_extra = 0,
            size_t max_num_steps = 0,
            size_t max_ram_MB = 2000,
            bint dense_output = False,
            double* t_eval = NULL,
            size_t len_t_eval = 0,
            PreEvalFunc pre_eval_func = NULL,
            double* rtols_ptr = NULL,
            double* atols_ptr = NULL,
            double max_step = MAX_STEP,
            double first_step = 0.0,
            size_t expected_size = 0
            ) noexcept:
    
    cdef CySolveOutput result = baseline_cysolve_ivp(
        diffeq_ptr,
        t_span_ptr,
        y0_ptr,
        num_y,
        method,
        expected_size,
        num_extra,
        args_ptr,
        max_num_steps,
        max_ram_MB,
        dense_output,
        t_eval,
        len_t_eval,
        pre_eval_func,
        rtol,
        atol,
        rtols_ptr,
        atols_ptr,
        max_step,
        first_step
        )

    return result

# =====================================================================================================================
# PySolver Class (holds the intergrator class and reference to the python diffeq function)
# =====================================================================================================================
cdef class WrapPyDiffeq:

    def __cinit__(
            self,
            object diffeq_func,
            tuple args,
            unsigned int num_y,
            unsigned int num_dy,
            bint pass_dy_as_arg = False
            ):
        
        # Install differential equation function and any additional args
        self.diffeq_func = diffeq_func
        if args is None:
            self.args     = None
            self.use_args = False
        else:
            if len(args) == 0:
                # Empty tuple provided. Don't use args.
                self.args     = None
                self.use_args = False
            else:
                self.args     = args
                self.use_args = True
        
        # Build python-safe arrays
        self.num_y  = num_y
        self.num_dy = num_dy

        if pass_dy_as_arg:
            self.pass_dy_as_arg = True
        else:
            self.pass_dy_as_arg = False
    
    cdef void set_state(self, double* dy_ptr, double* t_ptr, double* y_ptr) noexcept:
        self.dy_now_ptr = dy_ptr
        self.t_now_ptr  = t_ptr
        self.y_now_ptr  = y_ptr

        # Create memoryviews of the pointers
        self.y_now_view  = <double[:self.num_y]>self.y_now_ptr

        # Create numpy arrays which will be passed to the python diffeq.
        # We need to make sure that this is a not a new ndarray, but one that points to the same data. 
        # That is why we use `PyArray_SimpleNewFromData` instead of a more simple `asarray`.
        # Note that it is not safe to return these arrays outside of this class because they may get deallocated while
        # the numpy array still points to the underlying memory.
        cdef np.npy_intp[1] shape
        cdef np.npy_intp* shape_ptr = &shape[0]
        shape_ptr[0] = <np.npy_intp>self.num_y
        
        self.y_now_arr = np.PyArray_SimpleNewFromData(1, shape_ptr, np.NPY_DOUBLE, self.y_now_ptr)
        
        # Do the same for dy if the user provided the appropriate kind of differential equation.
        if self.pass_dy_as_arg:
            self.dy_now_view = <double[:self.num_dy]>self.dy_now_ptr
            shape[0]         = <np.npy_intp>self.num_dy  # dy may have a larger shape than y
            self.dy_now_arr  = np.PyArray_SimpleNewFromData(1, shape_ptr, np.NPY_DOUBLE, self.dy_now_ptr)   

    cdef void diffeq(self) noexcept:
        # Run python diffeq
        if self.pass_dy_as_arg:
            if self.use_args:
                self.diffeq_func(self.dy_now_arr, self.t_now_ptr[0], self.y_now_arr, *self.args)
            else:
                self.diffeq_func(self.dy_now_arr, self.t_now_ptr[0], self.y_now_arr)
        else:
            if self.use_args:
                self.dy_now_view = self.diffeq_func(self.t_now_ptr[0], self.y_now_arr, *self.args)
            else:
                self.dy_now_view = self.diffeq_func(self.t_now_ptr[0], self.y_now_arr)
            # Since we do not have a static dy that we can pass to the function and use in the solver we must copy over
            # the values from the newly created dy memory view
            # Note that num_dy may be larger than num_y if the user is capturing extra output during integration.
            memcpy(self.dy_now_ptr, &self.dy_now_view[0], sizeof(double) * self.num_dy)
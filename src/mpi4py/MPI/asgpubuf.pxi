#------------------------------------------------------------------------------

# CUDA array interface for interoperating Python CUDA GPU libraries
# See http://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html

cdef inline char* cuda_get_format(char typekind, Py_ssize_t itemsize) nogil:
   if typekind == c'b':
       if itemsize ==  1: return b"b1"
       if itemsize ==  2: return b"b2"
       if itemsize ==  4: return b"b4"
       if itemsize ==  8: return b"b8"
   if typekind == c'i':
       if itemsize ==  1: return b"i1"
       if itemsize ==  2: return b"i2"
       if itemsize ==  4: return b"i4"
       if itemsize ==  8: return b"i8"
   if typekind == c'u':
       if itemsize ==  1: return b"u1"
       if itemsize ==  2: return b"u2"
       if itemsize ==  4: return b"u4"
       if itemsize ==  8: return b"u8"
   if typekind == c'f':
       if itemsize ==  2: return b"f2"
       if itemsize ==  4: return b"f4"
       if itemsize ==  8: return b"f8"
       if itemsize == 12: return b"f12"
       if itemsize == 16: return b"f16"
   if typekind == c'c':
       if itemsize ==  4: return b"c4"
       if itemsize ==  8: return b"c8"
       if itemsize == 16: return b"c16"
       if itemsize == 24: return b"c24"
       if itemsize == 32: return b"c32"
   return BYTE_FMT

cdef inline Py_ssize_t cuda_get_size(object shape, object strides) except -1:
    cdef Py_ssize_t s, size = 1
    for s in shape: size *= s
    assert size >= 0
    if strides is not None:
        # TODO check strides
        pass
    return size

cdef int Py_CheckCUDABuffer(object obj):
    try: return <bint>hasattr(obj, '__cuda_array_interface__')
    except: return 0

cdef int Py_GetCUDABuffer(object obj, Py_buffer *view, int flags) except -1:
    cdef dict cuda_array_interface
    cdef object dev_ptr
    cdef object typestr
    cdef object shape
    cdef object strides
    cdef void *buf = NULL
    cdef bint readonly = 0
    cdef Py_ssize_t size = 1
    cdef Py_ssize_t itemsize = 1
    cdef char typekind = c'u'
    cdef bint fixnull = 0

    try:
        cuda_array_interface = obj.__cuda_array_interface__
    except AttributeError:
        raise NotImplementedError("missing CUDA array interface")

    dev_ptr, readonly = cuda_array_interface['data']
    typestr = cuda_array_interface['typestr']
    shape = cuda_array_interface['shape']
    strides = cuda_array_interface.get('strides', None)
    size = cuda_get_size(shape, strides)
    if dev_ptr is None and size == 0: dev_ptr = 0 # XXX

    buf = PyLong_AsVoidPtr(dev_ptr)
    typekind = ord(typestr[1])
    itemsize = int(typestr[2:])

    fixnull = (buf == NULL and size == 0)
    if fixnull: buf = &fixnull
    PyBuffer_FillInfo(view, obj, buf, size*itemsize, readonly, flags)
    if fixnull: view.buf = NULL

    if (flags & PyBUF_FORMAT) == PyBUF_FORMAT:
        view.format = cuda_get_format(typekind, itemsize)
        if view.format != BYTE_FMT:
            view.itemsize = itemsize
    return 0

#------------------------------------------------------------------------------

cdef int Py_CheckGPUBuffer(object obj):
    return Py_CheckCUDABuffer(obj)

cdef int Py_GetGPUBuffer(object obj, Py_buffer *view, int flags) except -1:
    return Py_GetCUDABuffer(obj, view, flags)

#------------------------------------------------------------------------------

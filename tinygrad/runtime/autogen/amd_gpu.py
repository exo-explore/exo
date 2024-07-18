# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/opt/rocm/include', '-x', 'c++']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, os


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass





SDMA_OP_COPY = 1 # Variable ctypes.c_uint32
SDMA_OP_FENCE = 5 # Variable ctypes.c_uint32
SDMA_OP_TRAP = 6 # Variable ctypes.c_uint32
SDMA_OP_POLL_REGMEM = 8 # Variable ctypes.c_uint32
SDMA_OP_ATOMIC = 10 # Variable ctypes.c_uint32
SDMA_OP_CONST_FILL = 11 # Variable ctypes.c_uint32
SDMA_OP_TIMESTAMP = 13 # Variable ctypes.c_uint32
SDMA_OP_GCR = 17 # Variable ctypes.c_uint32
SDMA_SUBOP_COPY_LINEAR = 0 # Variable ctypes.c_uint32
SDMA_SUBOP_COPY_LINEAR_RECT = 4 # Variable ctypes.c_uint32
SDMA_SUBOP_TIMESTAMP_GET_GLOBAL = 2 # Variable ctypes.c_uint32
SDMA_SUBOP_USER_GCR = 1 # Variable ctypes.c_uint32
SDMA_ATOMIC_ADD64 = 47 # Variable ctypes.c_uint32
class struct_SDMA_PKT_COPY_LINEAR_TAG(Structure):
    pass

class union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('extra_info', ctypes.c_uint32, 16),
]

union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_1_0._fields_ = [
    ('count', ctypes.c_uint32, 22),
    ('reserved_0', ctypes.c_uint32, 10),
]

union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_2_0._fields_ = [
    ('reserved_0', ctypes.c_uint32, 16),
    ('dst_swap', ctypes.c_uint32, 2),
    ('reserved_1', ctypes.c_uint32, 6),
    ('src_swap', ctypes.c_uint32, 2),
    ('reserved_2', ctypes.c_uint32, 6),
]

union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_3_0._fields_ = [
    ('src_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_4_0._fields_ = [
    ('src_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_5_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_5_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_5_0._fields_ = [
    ('dst_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_5_0),
    ('DW_5_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_6_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_6_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_6_0._fields_ = [
    ('dst_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_6_0),
    ('DW_6_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_COPY_LINEAR_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION),
    ('COUNT_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION),
    ('PARAMETER_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION),
    ('SRC_ADDR_LO_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION),
    ('SRC_ADDR_HI_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION),
    ('DST_ADDR_LO_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION),
    ('DST_ADDR_HI_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION),
]

SDMA_PKT_COPY_LINEAR = struct_SDMA_PKT_COPY_LINEAR_TAG
class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG(Structure):
    pass

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('reserved', ctypes.c_uint32, 13),
    ('element', ctypes.c_uint32, 3),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0._fields_ = [
    ('src_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0._fields_ = [
    ('src_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0._fields_ = [
    ('src_offset_x', ctypes.c_uint32, 14),
    ('reserved_1', ctypes.c_uint32, 2),
    ('src_offset_y', ctypes.c_uint32, 14),
    ('reserved_2', ctypes.c_uint32, 2),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0._fields_ = [
    ('src_offset_z', ctypes.c_uint32, 11),
    ('reserved_1', ctypes.c_uint32, 2),
    ('src_pitch', ctypes.c_uint32, 19),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0._fields_ = [
    ('src_slice_pitch', ctypes.c_uint32, 28),
    ('reserved_1', ctypes.c_uint32, 4),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0),
    ('DW_5_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0._fields_ = [
    ('dst_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0),
    ('DW_6_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0._fields_ = [
    ('dst_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0),
    ('DW_7_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0._fields_ = [
    ('dst_offset_x', ctypes.c_uint32, 14),
    ('reserved_1', ctypes.c_uint32, 2),
    ('dst_offset_y', ctypes.c_uint32, 14),
    ('reserved_2', ctypes.c_uint32, 2),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0),
    ('DW_8_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0._fields_ = [
    ('dst_offset_z', ctypes.c_uint32, 11),
    ('reserved_1', ctypes.c_uint32, 2),
    ('dst_pitch', ctypes.c_uint32, 19),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0),
    ('DW_9_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0._fields_ = [
    ('dst_slice_pitch', ctypes.c_uint32, 28),
    ('reserved_1', ctypes.c_uint32, 4),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0),
    ('DW_10_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0._fields_ = [
    ('rect_x', ctypes.c_uint32, 14),
    ('reserved_1', ctypes.c_uint32, 2),
    ('rect_y', ctypes.c_uint32, 14),
    ('reserved_2', ctypes.c_uint32, 2),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0),
    ('DW_11_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0._fields_ = [
    ('rect_z', ctypes.c_uint32, 11),
    ('reserved_1', ctypes.c_uint32, 5),
    ('dst_swap', ctypes.c_uint32, 2),
    ('reserved_2', ctypes.c_uint32, 6),
    ('src_swap', ctypes.c_uint32, 2),
    ('reserved_3', ctypes.c_uint32, 6),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0),
    ('DW_12_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION),
    ('SRC_ADDR_LO_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION),
    ('SRC_ADDR_HI_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION),
    ('SRC_PARAMETER_1_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION),
    ('SRC_PARAMETER_2_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION),
    ('SRC_PARAMETER_3_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION),
    ('DST_ADDR_LO_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION),
    ('DST_ADDR_HI_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION),
    ('DST_PARAMETER_1_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION),
    ('DST_PARAMETER_2_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION),
    ('DST_PARAMETER_3_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION),
    ('RECT_PARAMETER_1_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION),
    ('RECT_PARAMETER_2_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION),
]

SDMA_PKT_COPY_LINEAR_RECT = struct_SDMA_PKT_COPY_LINEAR_RECT_TAG
class struct_SDMA_PKT_CONSTANT_FILL_TAG(Structure):
    pass

class union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('sw', ctypes.c_uint32, 2),
    ('reserved_0', ctypes.c_uint32, 12),
    ('fillsize', ctypes.c_uint32, 2),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0._fields_ = [
    ('dst_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0._fields_ = [
    ('dst_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0._fields_ = [
    ('src_data_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0._fields_ = [
    ('count', ctypes.c_uint32, 22),
    ('reserved_0', ctypes.c_uint32, 10),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_CONSTANT_FILL_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION),
    ('DST_ADDR_LO_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION),
    ('DST_ADDR_HI_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION),
    ('DATA_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION),
    ('COUNT_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION),
]

SDMA_PKT_CONSTANT_FILL = struct_SDMA_PKT_CONSTANT_FILL_TAG
class struct_SDMA_PKT_FENCE_TAG(Structure):
    pass

class union_SDMA_PKT_FENCE_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_FENCE_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_FENCE_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('mtype', ctypes.c_uint32, 3),
    ('gcc', ctypes.c_uint32, 1),
    ('sys', ctypes.c_uint32, 1),
    ('pad1', ctypes.c_uint32, 1),
    ('snp', ctypes.c_uint32, 1),
    ('gpa', ctypes.c_uint32, 1),
    ('l2_policy', ctypes.c_uint32, 2),
    ('reserved_0', ctypes.c_uint32, 6),
]

union_SDMA_PKT_FENCE_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_FENCE_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_FENCE_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_FENCE_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_FENCE_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_FENCE_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG_1_0._fields_ = [
    ('addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_FENCE_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_FENCE_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_FENCE_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG_2_0._fields_ = [
    ('addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_FENCE_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_FENCE_TAG_DATA_UNION(Union):
    pass

class struct_SDMA_PKT_FENCE_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_FENCE_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG_3_0._fields_ = [
    ('data', ctypes.c_uint32, 32),
]

union_SDMA_PKT_FENCE_TAG_DATA_UNION._pack_ = 1 # source:False
union_SDMA_PKT_FENCE_TAG_DATA_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_FENCE_TAG_DATA_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_FENCE_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_FENCE_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_FENCE_TAG_HEADER_UNION),
    ('ADDR_LO_UNION', union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION),
    ('ADDR_HI_UNION', union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION),
    ('DATA_UNION', union_SDMA_PKT_FENCE_TAG_DATA_UNION),
]

SDMA_PKT_FENCE = struct_SDMA_PKT_FENCE_TAG
class struct_SDMA_PKT_POLL_REGMEM_TAG(Structure):
    pass

class union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('reserved_0', ctypes.c_uint32, 10),
    ('hdp_flush', ctypes.c_uint32, 1),
    ('reserved_1', ctypes.c_uint32, 1),
    ('func', ctypes.c_uint32, 3),
    ('mem_poll', ctypes.c_uint32, 1),
]

union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_1_0._fields_ = [
    ('addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_2_0._fields_ = [
    ('addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_3_0._fields_ = [
    ('value', ctypes.c_uint32, 32),
]

union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_4_0._fields_ = [
    ('mask', ctypes.c_uint32, 32),
]

union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_5_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_5_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_5_0._fields_ = [
    ('interval', ctypes.c_uint32, 16),
    ('retry_count', ctypes.c_uint32, 12),
    ('reserved_0', ctypes.c_uint32, 4),
]

union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_5_0),
    ('DW_5_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_POLL_REGMEM_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION),
    ('ADDR_LO_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION),
    ('ADDR_HI_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION),
    ('VALUE_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION),
    ('MASK_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION),
    ('DW5_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION),
]

SDMA_PKT_POLL_REGMEM = struct_SDMA_PKT_POLL_REGMEM_TAG
class struct_SDMA_PKT_ATOMIC_TAG(Structure):
    pass

class union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('l', ctypes.c_uint32, 1),
    ('reserved_0', ctypes.c_uint32, 8),
    ('operation', ctypes.c_uint32, 7),
]

union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_1_0._fields_ = [
    ('addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_2_0._fields_ = [
    ('addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_3_0._fields_ = [
    ('src_data_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_4_0._fields_ = [
    ('src_data_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_5_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_5_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_5_0._fields_ = [
    ('cmp_data_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_5_0),
    ('DW_5_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_6_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_6_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_6_0._fields_ = [
    ('cmp_data_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_6_0),
    ('DW_6_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_7_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_7_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_7_0._fields_ = [
    ('loop_interval', ctypes.c_uint32, 13),
    ('reserved_0', ctypes.c_uint32, 19),
]

union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_7_0),
    ('DW_7_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_ATOMIC_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION),
    ('ADDR_LO_UNION', union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION),
    ('ADDR_HI_UNION', union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION),
    ('SRC_DATA_LO_UNION', union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION),
    ('SRC_DATA_HI_UNION', union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION),
    ('CMP_DATA_LO_UNION', union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION),
    ('CMP_DATA_HI_UNION', union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION),
    ('LOOP_UNION', union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION),
]

SDMA_PKT_ATOMIC = struct_SDMA_PKT_ATOMIC_TAG
class struct_SDMA_PKT_TIMESTAMP_TAG(Structure):
    pass

class union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_TIMESTAMP_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_TIMESTAMP_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_TIMESTAMP_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('reserved_0', ctypes.c_uint32, 16),
]

union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TIMESTAMP_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_TIMESTAMP_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_TIMESTAMP_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_TIMESTAMP_TAG_1_0._fields_ = [
    ('addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TIMESTAMP_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_TIMESTAMP_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_TIMESTAMP_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_TIMESTAMP_TAG_2_0._fields_ = [
    ('addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TIMESTAMP_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_TIMESTAMP_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_TIMESTAMP_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION),
    ('ADDR_LO_UNION', union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION),
    ('ADDR_HI_UNION', union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION),
]

SDMA_PKT_TIMESTAMP = struct_SDMA_PKT_TIMESTAMP_TAG
class struct_SDMA_PKT_TRAP_TAG(Structure):
    pass

class union_SDMA_PKT_TRAP_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_TRAP_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_TRAP_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_TRAP_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('reserved_0', ctypes.c_uint32, 16),
]

union_SDMA_PKT_TRAP_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TRAP_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TRAP_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TRAP_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION(Union):
    pass

class struct_SDMA_PKT_TRAP_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_TRAP_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_TRAP_TAG_1_0._fields_ = [
    ('int_ctx', ctypes.c_uint32, 28),
    ('reserved_1', ctypes.c_uint32, 4),
]

union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TRAP_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_TRAP_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_TRAP_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_TRAP_TAG_HEADER_UNION),
    ('INT_CONTEXT_UNION', union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION),
]

SDMA_PKT_TRAP = struct_SDMA_PKT_TRAP_TAG
class struct_SDMA_PKT_HDP_FLUSH_TAG(Structure):
    pass

struct_SDMA_PKT_HDP_FLUSH_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_HDP_FLUSH_TAG._fields_ = [
    ('DW_0_DATA', ctypes.c_uint32),
    ('DW_1_DATA', ctypes.c_uint32),
    ('DW_2_DATA', ctypes.c_uint32),
    ('DW_3_DATA', ctypes.c_uint32),
    ('DW_4_DATA', ctypes.c_uint32),
    ('DW_5_DATA', ctypes.c_uint32),
]

SDMA_PKT_HDP_FLUSH = struct_SDMA_PKT_HDP_FLUSH_TAG
hdp_flush_cmd = struct_SDMA_PKT_HDP_FLUSH_TAG # Variable struct_SDMA_PKT_HDP_FLUSH_TAG
class struct_SDMA_PKT_GCR_TAG(Structure):
    pass

class union_SDMA_PKT_GCR_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('_2', ctypes.c_uint32, 16),
]

union_SDMA_PKT_GCR_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_GCR_TAG_WORD1_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_1_0._fields_ = [
    ('_0', ctypes.c_uint32, 7),
    ('BaseVA_LO', ctypes.c_uint32, 25),
]

union_SDMA_PKT_GCR_TAG_WORD1_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_WORD1_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_WORD1_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_GCR_TAG_WORD2_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_2_0._fields_ = [
    ('BaseVA_HI', ctypes.c_uint32, 16),
    ('GCR_CONTROL_GLI_INV', ctypes.c_uint32, 2),
    ('GCR_CONTROL_GL1_RANGE', ctypes.c_uint32, 2),
    ('GCR_CONTROL_GLM_WB', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GLM_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GLK_WB', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GLK_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GLV_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL1_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL2_US', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL2_RANGE', ctypes.c_uint32, 2),
    ('GCR_CONTROL_GL2_DISCARD', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL2_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL2_WB', ctypes.c_uint32, 1),
]

union_SDMA_PKT_GCR_TAG_WORD2_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_WORD2_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_WORD2_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_GCR_TAG_WORD3_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_3_0._fields_ = [
    ('GCR_CONTROL_RANGE_IS_PA', ctypes.c_uint32, 1),
    ('GCR_CONTROL_SEQ', ctypes.c_uint32, 2),
    ('_2', ctypes.c_uint32, 4),
    ('LimitVA_LO', ctypes.c_uint32, 25),
]

union_SDMA_PKT_GCR_TAG_WORD3_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_WORD3_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_WORD3_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_GCR_TAG_WORD4_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_4_0._fields_ = [
    ('LimitVA_HI', ctypes.c_uint32, 16),
    ('_1', ctypes.c_uint32, 8),
    ('VMID', ctypes.c_uint32, 4),
    ('_3', ctypes.c_uint32, 4),
]

union_SDMA_PKT_GCR_TAG_WORD4_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_WORD4_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_WORD4_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_GCR_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_GCR_TAG_HEADER_UNION),
    ('WORD1_UNION', union_SDMA_PKT_GCR_TAG_WORD1_UNION),
    ('WORD2_UNION', union_SDMA_PKT_GCR_TAG_WORD2_UNION),
    ('WORD3_UNION', union_SDMA_PKT_GCR_TAG_WORD3_UNION),
    ('WORD4_UNION', union_SDMA_PKT_GCR_TAG_WORD4_UNION),
]

SDMA_PKT_GCR = struct_SDMA_PKT_GCR_TAG
__all__ = \
    ['SDMA_ATOMIC_ADD64', 'SDMA_OP_ATOMIC', 'SDMA_OP_CONST_FILL',
    'SDMA_OP_COPY', 'SDMA_OP_FENCE', 'SDMA_OP_GCR',
    'SDMA_OP_POLL_REGMEM', 'SDMA_OP_TIMESTAMP', 'SDMA_OP_TRAP',
    'SDMA_PKT_ATOMIC', 'SDMA_PKT_CONSTANT_FILL',
    'SDMA_PKT_COPY_LINEAR', 'SDMA_PKT_COPY_LINEAR_RECT',
    'SDMA_PKT_FENCE', 'SDMA_PKT_GCR', 'SDMA_PKT_HDP_FLUSH',
    'SDMA_PKT_POLL_REGMEM', 'SDMA_PKT_TIMESTAMP', 'SDMA_PKT_TRAP',
    'SDMA_SUBOP_COPY_LINEAR', 'SDMA_SUBOP_COPY_LINEAR_RECT',
    'SDMA_SUBOP_TIMESTAMP_GET_GLOBAL', 'SDMA_SUBOP_USER_GCR',
    'hdp_flush_cmd', 'struct_SDMA_PKT_ATOMIC_TAG',
    'struct_SDMA_PKT_ATOMIC_TAG_0_0',
    'struct_SDMA_PKT_ATOMIC_TAG_1_0',
    'struct_SDMA_PKT_ATOMIC_TAG_2_0',
    'struct_SDMA_PKT_ATOMIC_TAG_3_0',
    'struct_SDMA_PKT_ATOMIC_TAG_4_0',
    'struct_SDMA_PKT_ATOMIC_TAG_5_0',
    'struct_SDMA_PKT_ATOMIC_TAG_6_0',
    'struct_SDMA_PKT_ATOMIC_TAG_7_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_0_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_1_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_2_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_3_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_4_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_5_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_6_0',
    'struct_SDMA_PKT_FENCE_TAG', 'struct_SDMA_PKT_FENCE_TAG_0_0',
    'struct_SDMA_PKT_FENCE_TAG_1_0', 'struct_SDMA_PKT_FENCE_TAG_2_0',
    'struct_SDMA_PKT_FENCE_TAG_3_0', 'struct_SDMA_PKT_GCR_TAG',
    'struct_SDMA_PKT_GCR_TAG_0_0', 'struct_SDMA_PKT_GCR_TAG_1_0',
    'struct_SDMA_PKT_GCR_TAG_2_0', 'struct_SDMA_PKT_GCR_TAG_3_0',
    'struct_SDMA_PKT_GCR_TAG_4_0', 'struct_SDMA_PKT_HDP_FLUSH_TAG',
    'struct_SDMA_PKT_POLL_REGMEM_TAG',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_0_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_1_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_2_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_3_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_4_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_5_0',
    'struct_SDMA_PKT_TIMESTAMP_TAG',
    'struct_SDMA_PKT_TIMESTAMP_TAG_0_0',
    'struct_SDMA_PKT_TIMESTAMP_TAG_1_0',
    'struct_SDMA_PKT_TIMESTAMP_TAG_2_0', 'struct_SDMA_PKT_TRAP_TAG',
    'struct_SDMA_PKT_TRAP_TAG_0_0', 'struct_SDMA_PKT_TRAP_TAG_1_0',
    'union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION',
    'union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION',
    'union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION',
    'union_SDMA_PKT_FENCE_TAG_DATA_UNION',
    'union_SDMA_PKT_FENCE_TAG_HEADER_UNION',
    'union_SDMA_PKT_GCR_TAG_HEADER_UNION',
    'union_SDMA_PKT_GCR_TAG_WORD1_UNION',
    'union_SDMA_PKT_GCR_TAG_WORD2_UNION',
    'union_SDMA_PKT_GCR_TAG_WORD3_UNION',
    'union_SDMA_PKT_GCR_TAG_WORD4_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION',
    'union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION',
    'union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION',
    'union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION',
    'union_SDMA_PKT_TRAP_TAG_HEADER_UNION',
    'union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION']
 #/*
# * Copyright 2019 Advanced Micro Devices, Inc.
# *
# * Permission is hereby granted, free of charge, to any person obtaining a
# * copy of this software and associated documentation files (the "Software"),
# * to deal in the Software without restriction, including without limitation
# * the rights to use, copy, modify, merge, publish, distribute, sublicense,
# * and/or sell copies of the Software, and to permit persons to whom the
# * Software is furnished to do so, subject to the following conditions:
# *
# * The above copyright notice and this permission notice shall be included in
# * all copies or substantial portions of the Software.
# *
# * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
# * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# * OTHER DEALINGS IN THE SOFTWARE.
# *
# */

#ifndef NVD_H
#define NVD_H

 #/**
# * Navi's PM4 definitions
# */
PACKET_TYPE0 = 0
PACKET_TYPE1 = 1
PACKET_TYPE2 = 2
PACKET_TYPE3 = 3

def CP_PACKET_GET_TYPE(h): return (((h) >> 30) & 3)
def CP_PACKET_GET_COUNT(h): return (((h) >> 16) & 0x3FFF)
def CP_PACKET0_GET_REG(h): return ((h) & 0xFFFF)
def CP_PACKET3_GET_OPCODE(h): return (((h) >> 8) & 0xFF)
def PACKET0(reg, n): return ((PACKET_TYPE0 << 30) |				\
			 ((reg) & 0xFFFF) |			\
			 ((n) & 0x3FFF) << 16)
CP_PACKET2 = 0x80000000
PACKET2_PAD_SHIFT = 0
PACKET2_PAD_MASK = (0x3fffffff << 0)

def PACKET2(v): return (CP_PACKET2 | REG_SET(PACKET2_PAD, (v)))

def PACKET3(op, n): return ((PACKET_TYPE3 << 30) |				\
			 (((op) & 0xFF) << 8) |				\
			 ((n) & 0x3FFF) << 16)

def PACKET3_COMPUTE(op, n): return (PACKET3(op, n) | 1 << 1)

 #/* Packet 3 types */
PACKET3_NOP = 0x10
PACKET3_SET_BASE = 0x11
def PACKET3_BASE_INDEX(x): return ((x) << 0)
CE_PARTITION_BASE = 3
PACKET3_CLEAR_STATE = 0x12
PACKET3_INDEX_BUFFER_SIZE = 0x13
PACKET3_DISPATCH_DIRECT = 0x15
PACKET3_DISPATCH_INDIRECT = 0x16
PACKET3_INDIRECT_BUFFER_END = 0x17
PACKET3_INDIRECT_BUFFER_CNST_END = 0x19
PACKET3_ATOMIC_GDS = 0x1D
PACKET3_ATOMIC_MEM = 0x1E
PACKET3_OCCLUSION_QUERY = 0x1F
PACKET3_SET_PREDICATION = 0x20
PACKET3_REG_RMW = 0x21
PACKET3_COND_EXEC = 0x22
PACKET3_PRED_EXEC = 0x23
PACKET3_DRAW_INDIRECT = 0x24
PACKET3_DRAW_INDEX_INDIRECT = 0x25
PACKET3_INDEX_BASE = 0x26
PACKET3_DRAW_INDEX_2 = 0x27
PACKET3_CONTEXT_CONTROL = 0x28
PACKET3_INDEX_TYPE = 0x2A
PACKET3_DRAW_INDIRECT_MULTI = 0x2C
PACKET3_DRAW_INDEX_AUTO = 0x2D
PACKET3_NUM_INSTANCES = 0x2F
PACKET3_DRAW_INDEX_MULTI_AUTO = 0x30
PACKET3_INDIRECT_BUFFER_PRIV = 0x32
PACKET3_INDIRECT_BUFFER_CNST = 0x33
PACKET3_COND_INDIRECT_BUFFER_CNST = 0x33
PACKET3_STRMOUT_BUFFER_UPDATE = 0x34
PACKET3_DRAW_INDEX_OFFSET_2 = 0x35
PACKET3_DRAW_PREAMBLE = 0x36
PACKET3_WRITE_DATA = 0x37
def WRITE_DATA_DST_SEL(x): return ((x) << 8)
		 #/* 0 - register
#		 * 1 - memory (sync - via GRBM)
#		 * 2 - gl2
#		 * 3 - gds
#		 * 4 - reserved
#		 * 5 - memory (async - direct)
#		 */
WR_ONE_ADDR = (1 << 16)
WR_CONFIRM = (1 << 20)
def WRITE_DATA_CACHE_POLICY(x): return ((x) << 25)
		 #/* 0 - LRU
#		 * 1 - Stream
#		 */
def WRITE_DATA_ENGINE_SEL(x): return ((x) << 30)
		 #/* 0 - me
#		 * 1 - pfp
#		 * 2 - ce
#		 */
PACKET3_DRAW_INDEX_INDIRECT_MULTI = 0x38
PACKET3_MEM_SEMAPHORE = 0x39
PACKET3_SEM_USE_MAILBOX = (0x1 << 16)
PACKET3_SEM_SEL_SIGNAL_TYPE = (0x1 << 20)  #/* 0 = increment, 1 = write 1 */
PACKET3_SEM_SEL_SIGNAL = (0x6 << 29)
PACKET3_SEM_SEL_WAIT = (0x7 << 29)
PACKET3_DRAW_INDEX_MULTI_INST = 0x3A
PACKET3_COPY_DW = 0x3B
PACKET3_WAIT_REG_MEM = 0x3C
def WAIT_REG_MEM_FUNCTION(x): return ((x) << 0)
		 #/* 0 - always
#		 * 1 - <
#		 * 2 - <=
#		 * 3 - ==
#		 * 4 - !=
#		 * 5 - >=
#		 * 6 - >
#		 */
def WAIT_REG_MEM_MEM_SPACE(x): return ((x) << 4)
		 #/* 0 - reg
#		 * 1 - mem
#		 */
def WAIT_REG_MEM_OPERATION(x): return ((x) << 6)
		 #/* 0 - wait_reg_mem
#		 * 1 - wr_wait_wr_reg
#		 */
def WAIT_REG_MEM_ENGINE(x): return ((x) << 8)
		 #/* 0 - me
#		 * 1 - pfp
#		 */
PACKET3_INDIRECT_BUFFER = 0x3F
INDIRECT_BUFFER_VALID = (1 << 23)
def INDIRECT_BUFFER_CACHE_POLICY(x): return ((x) << 28)
		 #/* 0 - LRU
#		 * 1 - Stream
#		 * 2 - Bypass
#		 */
def INDIRECT_BUFFER_PRE_ENB(x): return ((x) << 21)
def INDIRECT_BUFFER_PRE_RESUME(x): return ((x) << 30)
PACKET3_COND_INDIRECT_BUFFER = 0x3F
PACKET3_COPY_DATA = 0x40
PACKET3_CP_DMA = 0x41
PACKET3_PFP_SYNC_ME = 0x42
PACKET3_SURFACE_SYNC = 0x43
PACKET3_ME_INITIALIZE = 0x44
PACKET3_COND_WRITE = 0x45
PACKET3_EVENT_WRITE = 0x46
def EVENT_TYPE(x): return ((x) << 0)
def EVENT_INDEX(x): return ((x) << 8)
		 #/* 0 - any non-TS event
#		 * 1 - ZPASS_DONE, PIXEL_PIPE_STAT_*
#		 * 2 - SAMPLE_PIPELINESTAT
#		 * 3 - SAMPLE_STREAMOUTSTAT*
#		 * 4 - *S_PARTIAL_FLUSH
#		 */
PACKET3_EVENT_WRITE_EOP = 0x47
PACKET3_EVENT_WRITE_EOS = 0x48
PACKET3_RELEASE_MEM = 0x49
def PACKET3_RELEASE_MEM_EVENT_TYPE(x): return ((x) << 0)
def PACKET3_RELEASE_MEM_EVENT_INDEX(x): return ((x) << 8)
PACKET3_RELEASE_MEM_GCR_GLM_WB = (1 << 12)
PACKET3_RELEASE_MEM_GCR_GLM_INV = (1 << 13)
PACKET3_RELEASE_MEM_GCR_GLV_INV = (1 << 14)
PACKET3_RELEASE_MEM_GCR_GL1_INV = (1 << 15)
PACKET3_RELEASE_MEM_GCR_GL2_US = (1 << 16)
PACKET3_RELEASE_MEM_GCR_GL2_RANGE = (1 << 17)
PACKET3_RELEASE_MEM_GCR_GL2_DISCARD = (1 << 19)
PACKET3_RELEASE_MEM_GCR_GL2_INV = (1 << 20)
PACKET3_RELEASE_MEM_GCR_GL2_WB = (1 << 21)
PACKET3_RELEASE_MEM_GCR_SEQ = (1 << 22)
def PACKET3_RELEASE_MEM_CACHE_POLICY(x): return ((x) << 25)
		 #/* 0 - cache_policy__me_release_mem__lru
#		 * 1 - cache_policy__me_release_mem__stream
#		 * 2 - cache_policy__me_release_mem__noa
#		 * 3 - cache_policy__me_release_mem__bypass
#		 */
PACKET3_RELEASE_MEM_EXECUTE = (1 << 28)

def PACKET3_RELEASE_MEM_DATA_SEL(x): return ((x) << 29)
		 #/* 0 - discard
#		 * 1 - send low 32bit data
#		 * 2 - send 64bit data
#		 * 3 - send 64bit GPU counter value
#		 * 4 - send 64bit sys counter value
#		 */
def PACKET3_RELEASE_MEM_INT_SEL(x): return ((x) << 24)
		 #/* 0 - none
#		 * 1 - interrupt only (DATA_SEL = 0)
#		 * 2 - interrupt when data write is confirmed
#		 */
def PACKET3_RELEASE_MEM_DST_SEL(x): return ((x) << 16)
		 #/* 0 - MC
#		 * 1 - TC/L2
#		 */



PACKET3_PREAMBLE_CNTL = 0x4A
PACKET3_PREAMBLE_BEGIN_CLEAR_STATE = (2 << 28)
PACKET3_PREAMBLE_END_CLEAR_STATE = (3 << 28)
PACKET3_DMA_DATA = 0x50
 #/* 1. header
# * 2. CONTROL
# * 3. SRC_ADDR_LO or DATA [31:0]
# * 4. SRC_ADDR_HI [31:0]
# * 5. DST_ADDR_LO [31:0]
# * 6. DST_ADDR_HI [7:0]
# * 7. COMMAND [31:26] | BYTE_COUNT [25:0]
# */
 #/* CONTROL */
def PACKET3_DMA_DATA_ENGINE(x): return ((x) << 0)
		 #/* 0 - ME
#		 * 1 - PFP
#		 */
def PACKET3_DMA_DATA_SRC_CACHE_POLICY(x): return ((x) << 13)
		 #/* 0 - LRU
#		 * 1 - Stream
#		 */
def PACKET3_DMA_DATA_DST_SEL(x): return ((x) << 20)
		 #/* 0 - DST_ADDR using DAS
#		 * 1 - GDS
#		 * 3 - DST_ADDR using L2
#		 */
def PACKET3_DMA_DATA_DST_CACHE_POLICY(x): return ((x) << 25)
		 #/* 0 - LRU
#		 * 1 - Stream
#		 */
def PACKET3_DMA_DATA_SRC_SEL(x): return ((x) << 29)
		 #/* 0 - SRC_ADDR using SAS
#		 * 1 - GDS
#		 * 2 - DATA
#		 * 3 - SRC_ADDR using L2
#		 */
PACKET3_DMA_DATA_CP_SYNC = (1 << 31)
 #/* COMMAND */
PACKET3_DMA_DATA_CMD_SAS = (1 << 26)
		 #/* 0 - memory
#		 * 1 - register
#		 */
PACKET3_DMA_DATA_CMD_DAS = (1 << 27)
		 #/* 0 - memory
#		 * 1 - register
#		 */
PACKET3_DMA_DATA_CMD_SAIC = (1 << 28)
PACKET3_DMA_DATA_CMD_DAIC = (1 << 29)
PACKET3_DMA_DATA_CMD_RAW_WAIT = (1 << 30)
PACKET3_CONTEXT_REG_RMW = 0x51
PACKET3_GFX_CNTX_UPDATE = 0x52
PACKET3_BLK_CNTX_UPDATE = 0x53
PACKET3_INCR_UPDT_STATE = 0x55
PACKET3_ACQUIRE_MEM = 0x58
 #/* 1.  HEADER
# * 2.  COHER_CNTL [30:0]
# * 2.1 ENGINE_SEL [31:31]
# * 2.  COHER_SIZE [31:0]
# * 3.  COHER_SIZE_HI [7:0]
# * 4.  COHER_BASE_LO [31:0]
# * 5.  COHER_BASE_HI [23:0]
# * 7.  POLL_INTERVAL [15:0]
# * 8.  GCR_CNTL [18:0]
# */
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(x): return ((x) << 0)
		 #/*
#		 * 0:NOP
#		 * 1:ALL
#		 * 2:RANGE
#		 * 3:FIRST_LAST
#		 */
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_RANGE(x): return ((x) << 2)
		 #/*
#		 * 0:ALL
#		 * 1:reserved
#		 * 2:RANGE
#		 * 3:FIRST_LAST
#		 */
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(x): return ((x) << 4)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(x): return ((x) << 5)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(x): return ((x) << 6)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(x): return ((x) << 7)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(x): return ((x) << 8)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(x): return ((x) << 9)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_US(x): return ((x) << 10)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_RANGE(x): return ((x) << 11)
		 #/*
#		 * 0:ALL
#		 * 1:VOL
#		 * 2:RANGE
#		 * 3:FIRST_LAST
#		 */
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_DISCARD(x): return ((x) << 13)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(x): return ((x) << 14)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(x): return ((x) << 15)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_SEQ(x): return ((x) << 16)
		 #/*
#		 * 0: PARALLEL
#		 * 1: FORWARD
#		 * 2: REVERSE
#		 */
PACKET3_ACQUIRE_MEM_GCR_RANGE_IS_PA = (1 << 18)
PACKET3_REWIND = 0x59
PACKET3_INTERRUPT = 0x5A
PACKET3_GEN_PDEPTE = 0x5B
PACKET3_INDIRECT_BUFFER_PASID = 0x5C
PACKET3_PRIME_UTCL2 = 0x5D
PACKET3_LOAD_UCONFIG_REG = 0x5E
PACKET3_LOAD_SH_REG = 0x5F
PACKET3_LOAD_CONFIG_REG = 0x60
PACKET3_LOAD_CONTEXT_REG = 0x61
PACKET3_LOAD_COMPUTE_STATE = 0x62
PACKET3_LOAD_SH_REG_INDEX = 0x63
PACKET3_SET_CONFIG_REG = 0x68
PACKET3_SET_CONFIG_REG_START = 0x00002000
PACKET3_SET_CONFIG_REG_END = 0x00002c00
PACKET3_SET_CONTEXT_REG = 0x69
PACKET3_SET_CONTEXT_REG_START = 0x0000a000
PACKET3_SET_CONTEXT_REG_END = 0x0000a400
PACKET3_SET_CONTEXT_REG_INDEX = 0x6A
PACKET3_SET_VGPR_REG_DI_MULTI = 0x71
PACKET3_SET_SH_REG_DI = 0x72
PACKET3_SET_CONTEXT_REG_INDIRECT = 0x73
PACKET3_SET_SH_REG_DI_MULTI = 0x74
PACKET3_GFX_PIPE_LOCK = 0x75
PACKET3_SET_SH_REG = 0x76
PACKET3_SET_SH_REG_START = 0x00002c00
PACKET3_SET_SH_REG_END = 0x00003000
PACKET3_SET_SH_REG_OFFSET = 0x77
PACKET3_SET_QUEUE_REG = 0x78
PACKET3_SET_UCONFIG_REG = 0x79
PACKET3_SET_UCONFIG_REG_START = 0x0000c000
PACKET3_SET_UCONFIG_REG_END = 0x0000c400
PACKET3_SET_UCONFIG_REG_INDEX = 0x7A
PACKET3_FORWARD_HEADER = 0x7C
PACKET3_SCRATCH_RAM_WRITE = 0x7D
PACKET3_SCRATCH_RAM_READ = 0x7E
PACKET3_LOAD_CONST_RAM = 0x80
PACKET3_WRITE_CONST_RAM = 0x81
PACKET3_DUMP_CONST_RAM = 0x83
PACKET3_INCREMENT_CE_COUNTER = 0x84
PACKET3_INCREMENT_DE_COUNTER = 0x85
PACKET3_WAIT_ON_CE_COUNTER = 0x86
PACKET3_WAIT_ON_DE_COUNTER_DIFF = 0x88
PACKET3_SWITCH_BUFFER = 0x8B
PACKET3_DISPATCH_DRAW_PREAMBLE = 0x8C
PACKET3_DISPATCH_DRAW_PREAMBLE_ACE = 0x8C
PACKET3_DISPATCH_DRAW = 0x8D
PACKET3_DISPATCH_DRAW_ACE = 0x8D
PACKET3_GET_LOD_STATS = 0x8E
PACKET3_DRAW_MULTI_PREAMBLE = 0x8F
PACKET3_FRAME_CONTROL = 0x90
FRAME_TMZ = (1 << 0)
def FRAME_CMD(x): return ((x) << 28)
			 #/*
#			 * x=0: tmz_begin
#			 * x=1: tmz_end
#			 */
PACKET3_INDEX_ATTRIBUTES_INDIRECT = 0x91
PACKET3_WAIT_REG_MEM64 = 0x93
PACKET3_COND_PREEMPT = 0x94
PACKET3_HDP_FLUSH = 0x95
PACKET3_COPY_DATA_RB = 0x96
PACKET3_INVALIDATE_TLBS = 0x98
def PACKET3_INVALIDATE_TLBS_DST_SEL(x): return ((x) << 0)
def PACKET3_INVALIDATE_TLBS_ALL_HUB(x): return ((x) << 4)
def PACKET3_INVALIDATE_TLBS_PASID(x): return ((x) << 5)
PACKET3_AQL_PACKET = 0x99
PACKET3_DMA_DATA_FILL_MULTI = 0x9A
PACKET3_SET_SH_REG_INDEX = 0x9B
PACKET3_DRAW_INDIRECT_COUNT_MULTI = 0x9C
PACKET3_DRAW_INDEX_INDIRECT_COUNT_MULTI = 0x9D
PACKET3_DUMP_CONST_RAM_OFFSET = 0x9E
PACKET3_LOAD_CONTEXT_REG_INDEX = 0x9F
PACKET3_SET_RESOURCES = 0xA0
 #/* 1. header
# * 2. CONTROL
# * 3. QUEUE_MASK_LO [31:0]
# * 4. QUEUE_MASK_HI [31:0]
# * 5. GWS_MASK_LO [31:0]
# * 6. GWS_MASK_HI [31:0]
# * 7. OAC_MASK [15:0]
# * 8. GDS_HEAP_SIZE [16:11] | GDS_HEAP_BASE [5:0]
# */
def PACKET3_SET_RESOURCES_VMID_MASK(x): return ((x) << 0)
def PACKET3_SET_RESOURCES_UNMAP_LATENTY(x): return ((x) << 16)
def PACKET3_SET_RESOURCES_QUEUE_TYPE(x): return ((x) << 29)
PACKET3_MAP_PROCESS = 0xA1
PACKET3_MAP_QUEUES = 0xA2
 #/* 1. header
# * 2. CONTROL
# * 3. CONTROL2
# * 4. MQD_ADDR_LO [31:0]
# * 5. MQD_ADDR_HI [31:0]
# * 6. WPTR_ADDR_LO [31:0]
# * 7. WPTR_ADDR_HI [31:0]
# */
 #/* CONTROL */
def PACKET3_MAP_QUEUES_QUEUE_SEL(x): return ((x) << 4)
def PACKET3_MAP_QUEUES_VMID(x): return ((x) << 8)
def PACKET3_MAP_QUEUES_QUEUE(x): return ((x) << 13)
def PACKET3_MAP_QUEUES_PIPE(x): return ((x) << 16)
def PACKET3_MAP_QUEUES_ME(x): return ((x) << 18)
def PACKET3_MAP_QUEUES_QUEUE_TYPE(x): return ((x) << 21)
def PACKET3_MAP_QUEUES_ALLOC_FORMAT(x): return ((x) << 24)
def PACKET3_MAP_QUEUES_ENGINE_SEL(x): return ((x) << 26)
def PACKET3_MAP_QUEUES_NUM_QUEUES(x): return ((x) << 29)
 #/* CONTROL2 */
def PACKET3_MAP_QUEUES_CHECK_DISABLE(x): return ((x) << 1)
def PACKET3_MAP_QUEUES_DOORBELL_OFFSET(x): return ((x) << 2)
PACKET3_UNMAP_QUEUES = 0xA3
 #/* 1. header
# * 2. CONTROL
# * 3. CONTROL2
# * 4. CONTROL3
# * 5. CONTROL4
# * 6. CONTROL5
# */
 #/* CONTROL */
def PACKET3_UNMAP_QUEUES_ACTION(x): return ((x) << 0)
		 #/* 0 - PREEMPT_QUEUES
#		 * 1 - RESET_QUEUES
#		 * 2 - DISABLE_PROCESS_QUEUES
#		 * 3 - PREEMPT_QUEUES_NO_UNMAP
#		 */
def PACKET3_UNMAP_QUEUES_QUEUE_SEL(x): return ((x) << 4)
def PACKET3_UNMAP_QUEUES_ENGINE_SEL(x): return ((x) << 26)
def PACKET3_UNMAP_QUEUES_NUM_QUEUES(x): return ((x) << 29)
 #/* CONTROL2a */
def PACKET3_UNMAP_QUEUES_PASID(x): return ((x) << 0)
 #/* CONTROL2b */
def PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET0(x): return ((x) << 2)
 #/* CONTROL3a */
def PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET1(x): return ((x) << 2)
 #/* CONTROL3b */
def PACKET3_UNMAP_QUEUES_RB_WPTR(x): return ((x) << 0)
 #/* CONTROL4 */
def PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET2(x): return ((x) << 2)
 #/* CONTROL5 */
def PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET3(x): return ((x) << 2)
PACKET3_QUERY_STATUS = 0xA4
 #/* 1. header
# * 2. CONTROL
# * 3. CONTROL2
# * 4. ADDR_LO [31:0]
# * 5. ADDR_HI [31:0]
# * 6. DATA_LO [31:0]
# * 7. DATA_HI [31:0]
# */
 #/* CONTROL */
def PACKET3_QUERY_STATUS_CONTEXT_ID(x): return ((x) << 0)
def PACKET3_QUERY_STATUS_INTERRUPT_SEL(x): return ((x) << 28)
def PACKET3_QUERY_STATUS_COMMAND(x): return ((x) << 30)
 #/* CONTROL2a */
def PACKET3_QUERY_STATUS_PASID(x): return ((x) << 0)
 #/* CONTROL2b */
def PACKET3_QUERY_STATUS_DOORBELL_OFFSET(x): return ((x) << 2)
def PACKET3_QUERY_STATUS_ENG_SEL(x): return ((x) << 25)
PACKET3_RUN_LIST = 0xA5
PACKET3_MAP_PROCESS_VM = 0xA6
 #/* GFX11 */
PACKET3_SET_Q_PREEMPTION_MODE = 0xF0
def PACKET3_SET_Q_PREEMPTION_MODE_IB_VMID(x): return ((x) << 0)
PACKET3_SET_Q_PREEMPTION_MODE_INIT_SHADOW_MEM = (1 << 0)

#endif
 #/*
# * Copyright 2021 Advanced Micro Devices, Inc.
# *
# * Permission is hereby granted, free of charge, to any person obtaining a
# * copy of this software and associated documentation files (the "Software"),
# * to deal in the Software without restriction, including without limitation
# * the rights to use, copy, modify, merge, publish, distribute, sublicense,
# * and/or sell copies of the Software, and to permit persons to whom the
# * Software is furnished to do so, subject to the following conditions:
# *
# * The above copyright notice and this permission notice shall be included in
# * all copies or substantial portions of the Software.
# *
# * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
# * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# * OTHER DEALINGS IN THE SOFTWARE.
# *
# */
#ifndef __SDMA_V6_0_0_PKT_OPEN_H_
#define __SDMA_V6_0_0_PKT_OPEN_H_

SDMA_OP_NOP = 0
SDMA_OP_COPY = 1
SDMA_OP_WRITE = 2
SDMA_OP_INDIRECT = 4
SDMA_OP_FENCE = 5
SDMA_OP_TRAP = 6
SDMA_OP_SEM = 7
SDMA_OP_POLL_REGMEM = 8
SDMA_OP_COND_EXE = 9
SDMA_OP_ATOMIC = 10
SDMA_OP_CONST_FILL = 11
SDMA_OP_PTEPDE = 12
SDMA_OP_TIMESTAMP = 13
SDMA_OP_SRBM_WRITE = 14
SDMA_OP_PRE_EXE = 15
SDMA_OP_GPUVM_INV = 16
SDMA_OP_GCR_REQ = 17
SDMA_OP_DUMMY_TRAP = 32
SDMA_SUBOP_TIMESTAMP_SET = 0
SDMA_SUBOP_TIMESTAMP_GET = 1
SDMA_SUBOP_TIMESTAMP_GET_GLOBAL = 2
SDMA_SUBOP_COPY_LINEAR = 0
SDMA_SUBOP_COPY_LINEAR_SUB_WIND = 4
SDMA_SUBOP_COPY_TILED = 1
SDMA_SUBOP_COPY_TILED_SUB_WIND = 5
SDMA_SUBOP_COPY_T2T_SUB_WIND = 6
SDMA_SUBOP_COPY_SOA = 3
SDMA_SUBOP_COPY_DIRTY_PAGE = 7
SDMA_SUBOP_COPY_LINEAR_PHY = 8
SDMA_SUBOP_COPY_LINEAR_SUB_WIND_LARGE = 36
SDMA_SUBOP_COPY_LINEAR_BC = 16
SDMA_SUBOP_COPY_TILED_BC = 17
SDMA_SUBOP_COPY_LINEAR_SUB_WIND_BC = 20
SDMA_SUBOP_COPY_TILED_SUB_WIND_BC = 21
SDMA_SUBOP_COPY_T2T_SUB_WIND_BC = 22
SDMA_SUBOP_WRITE_LINEAR = 0
SDMA_SUBOP_WRITE_TILED = 1
SDMA_SUBOP_WRITE_TILED_BC = 17
SDMA_SUBOP_PTEPDE_GEN = 0
SDMA_SUBOP_PTEPDE_COPY = 1
SDMA_SUBOP_PTEPDE_RMW = 2
SDMA_SUBOP_PTEPDE_COPY_BACKWARDS = 3
SDMA_SUBOP_MEM_INCR = 1
SDMA_SUBOP_DATA_FILL_MULTI = 1
SDMA_SUBOP_POLL_REG_WRITE_MEM = 1
SDMA_SUBOP_POLL_DBIT_WRITE_MEM = 2
SDMA_SUBOP_POLL_MEM_VERIFY = 3
SDMA_SUBOP_VM_INVALIDATION = 4
HEADER_AGENT_DISPATCH = 4
HEADER_BARRIER = 5
SDMA_OP_AQL_COPY = 0
SDMA_OP_AQL_BARRIER_OR = 0

SDMA_GCR_RANGE_IS_PA = (1 << 18)
def SDMA_GCR_SEQ(x): return (((x) & 0x3) << 16)
SDMA_GCR_GL2_WB = (1 << 15)
SDMA_GCR_GL2_INV = (1 << 14)
SDMA_GCR_GL2_DISCARD = (1 << 13)
def SDMA_GCR_GL2_RANGE(x): return (((x) & 0x3) << 11)
SDMA_GCR_GL2_US = (1 << 10)
SDMA_GCR_GL1_INV = (1 << 9)
SDMA_GCR_GLV_INV = (1 << 8)
SDMA_GCR_GLK_INV = (1 << 7)
SDMA_GCR_GLK_WB = (1 << 6)
SDMA_GCR_GLM_INV = (1 << 5)
SDMA_GCR_GLM_WB = (1 << 4)
def SDMA_GCR_GL1_RANGE(x): return (((x) & 0x3) << 2)
def SDMA_GCR_GLI_INV(x): return (((x) & 0x3) << 0)
 #/*
#** Definitions for SDMA_PKT_COPY_LINEAR packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_LINEAR_HEADER_op_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_HEADER_op_shift = 0
def SDMA_PKT_COPY_LINEAR_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_LINEAR_HEADER_op_mask) << SDMA_PKT_COPY_LINEAR_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_LINEAR_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_LINEAR_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_LINEAR_HEADER_sub_op_mask) << SDMA_PKT_COPY_LINEAR_HEADER_sub_op_shift)

 #/*define for encrypt field*/
SDMA_PKT_COPY_LINEAR_HEADER_encrypt_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_encrypt_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_HEADER_encrypt_shift = 16
def SDMA_PKT_COPY_LINEAR_HEADER_ENCRYPT(x): return (((x) & SDMA_PKT_COPY_LINEAR_HEADER_encrypt_mask) << SDMA_PKT_COPY_LINEAR_HEADER_encrypt_shift)

 #/*define for tmz field*/
SDMA_PKT_COPY_LINEAR_HEADER_tmz_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_HEADER_tmz_shift = 18
def SDMA_PKT_COPY_LINEAR_HEADER_TMZ(x): return (((x) & SDMA_PKT_COPY_LINEAR_HEADER_tmz_mask) << SDMA_PKT_COPY_LINEAR_HEADER_tmz_shift)

 #/*define for cpv field*/
SDMA_PKT_COPY_LINEAR_HEADER_cpv_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_HEADER_cpv_shift = 19
def SDMA_PKT_COPY_LINEAR_HEADER_CPV(x): return (((x) & SDMA_PKT_COPY_LINEAR_HEADER_cpv_mask) << SDMA_PKT_COPY_LINEAR_HEADER_cpv_shift)

 #/*define for backwards field*/
SDMA_PKT_COPY_LINEAR_HEADER_backwards_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_backwards_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_HEADER_backwards_shift = 25
def SDMA_PKT_COPY_LINEAR_HEADER_BACKWARDS(x): return (((x) & SDMA_PKT_COPY_LINEAR_HEADER_backwards_mask) << SDMA_PKT_COPY_LINEAR_HEADER_backwards_shift)

 #/*define for broadcast field*/
SDMA_PKT_COPY_LINEAR_HEADER_broadcast_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_broadcast_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_HEADER_broadcast_shift = 27
def SDMA_PKT_COPY_LINEAR_HEADER_BROADCAST(x): return (((x) & SDMA_PKT_COPY_LINEAR_HEADER_broadcast_mask) << SDMA_PKT_COPY_LINEAR_HEADER_broadcast_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_COPY_LINEAR_COUNT_count_offset = 1
SDMA_PKT_COPY_LINEAR_COUNT_count_mask = 0x3FFFFFFF
SDMA_PKT_COPY_LINEAR_COUNT_count_shift = 0
def SDMA_PKT_COPY_LINEAR_COUNT_COUNT(x): return (((x) & SDMA_PKT_COPY_LINEAR_COUNT_count_mask) << SDMA_PKT_COPY_LINEAR_COUNT_count_shift)

 #/*define for PARAMETER word*/
 #/*define for dst_sw field*/
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_offset = 2
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_shift = 16
def SDMA_PKT_COPY_LINEAR_PARAMETER_DST_SW(x): return (((x) & SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_mask) << SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_shift)

 #/*define for dst_cache_policy field*/
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_offset = 2
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_shift = 18
def SDMA_PKT_COPY_LINEAR_PARAMETER_DST_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_mask) << SDMA_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_shift)

 #/*define for src_sw field*/
SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_offset = 2
SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_shift = 24
def SDMA_PKT_COPY_LINEAR_PARAMETER_SRC_SW(x): return (((x) & SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_mask) << SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_shift)

 #/*define for src_cache_policy field*/
SDMA_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_offset = 2
SDMA_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_shift = 26
def SDMA_PKT_COPY_LINEAR_PARAMETER_SRC_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_mask) << SDMA_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_shift)

 #/*define for SRC_ADDR_LO word*/
 #/*define for src_addr_31_0 field*/
SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset = 3
SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift = 0
def SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_SRC_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift)

 #/*define for SRC_ADDR_HI word*/
 #/*define for src_addr_63_32 field*/
SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset = 4
SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift = 0
def SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_SRC_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_offset = 5
SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_offset = 6
SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_LINEAR_BC packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_LINEAR_BC_HEADER_op_offset = 0
SDMA_PKT_COPY_LINEAR_BC_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_BC_HEADER_op_shift = 0
def SDMA_PKT_COPY_LINEAR_BC_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_LINEAR_BC_HEADER_op_mask) << SDMA_PKT_COPY_LINEAR_BC_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_LINEAR_BC_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_mask) << SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_COPY_LINEAR_BC_COUNT_count_offset = 1
SDMA_PKT_COPY_LINEAR_BC_COUNT_count_mask = 0x003FFFFF
SDMA_PKT_COPY_LINEAR_BC_COUNT_count_shift = 0
def SDMA_PKT_COPY_LINEAR_BC_COUNT_COUNT(x): return (((x) & SDMA_PKT_COPY_LINEAR_BC_COUNT_count_mask) << SDMA_PKT_COPY_LINEAR_BC_COUNT_count_shift)

 #/*define for PARAMETER word*/
 #/*define for dst_sw field*/
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_offset = 2
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_shift = 16
def SDMA_PKT_COPY_LINEAR_BC_PARAMETER_DST_SW(x): return (((x) & SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_mask) << SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_shift)

 #/*define for dst_ha field*/
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_offset = 2
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_shift = 19
def SDMA_PKT_COPY_LINEAR_BC_PARAMETER_DST_HA(x): return (((x) & SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_mask) << SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_shift)

 #/*define for src_sw field*/
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_offset = 2
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_shift = 24
def SDMA_PKT_COPY_LINEAR_BC_PARAMETER_SRC_SW(x): return (((x) & SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_mask) << SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_shift)

 #/*define for src_ha field*/
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_offset = 2
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_shift = 27
def SDMA_PKT_COPY_LINEAR_BC_PARAMETER_SRC_HA(x): return (((x) & SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_mask) << SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_shift)

 #/*define for SRC_ADDR_LO word*/
 #/*define for src_addr_31_0 field*/
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_offset = 3
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_shift = 0
def SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_SRC_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_shift)

 #/*define for SRC_ADDR_HI word*/
 #/*define for src_addr_63_32 field*/
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_offset = 4
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_shift = 0
def SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_SRC_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_offset = 5
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_offset = 6
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_DIRTY_PAGE packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_offset = 0
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_shift = 0
def SDMA_PKT_COPY_DIRTY_PAGE_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_mask) << SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_DIRTY_PAGE_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_mask) << SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_shift)

 #/*define for tmz field*/
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_offset = 0
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_shift = 18
def SDMA_PKT_COPY_DIRTY_PAGE_HEADER_TMZ(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_mask) << SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_shift)

 #/*define for cpv field*/
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_cpv_offset = 0
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_cpv_shift = 19
def SDMA_PKT_COPY_DIRTY_PAGE_HEADER_CPV(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_HEADER_cpv_mask) << SDMA_PKT_COPY_DIRTY_PAGE_HEADER_cpv_shift)

 #/*define for all field*/
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_offset = 0
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_shift = 31
def SDMA_PKT_COPY_DIRTY_PAGE_HEADER_ALL(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_mask) << SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_offset = 1
SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_mask = 0x003FFFFF
SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_shift = 0
def SDMA_PKT_COPY_DIRTY_PAGE_COUNT_COUNT(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_mask) << SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_shift)

 #/*define for PARAMETER word*/
 #/*define for dst_mtype field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_mask = 0x00000007
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_shift = 3
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_MTYPE(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_shift)

 #/*define for dst_l2_policy field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_mask = 0x00000003
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_shift = 6
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_L2_POLICY(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_shift)

 #/*define for dst_llc field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_llc_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_llc_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_llc_shift = 8
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_LLC(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_llc_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_llc_shift)

 #/*define for src_mtype field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_mask = 0x00000007
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_shift = 11
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_MTYPE(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_shift)

 #/*define for src_l2_policy field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_mask = 0x00000003
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_shift = 14
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_L2_POLICY(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_shift)

 #/*define for src_llc field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_llc_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_llc_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_llc_shift = 16
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_LLC(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_llc_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_llc_shift)

 #/*define for dst_sw field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_shift = 17
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_SW(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_shift)

 #/*define for dst_gcc field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_shift = 19
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_GCC(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_shift)

 #/*define for dst_sys field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_shift = 20
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_SYS(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_shift)

 #/*define for dst_snoop field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_shift = 22
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_SNOOP(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_shift)

 #/*define for dst_gpa field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_shift = 23
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_GPA(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_shift)

 #/*define for src_sw field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_mask = 0x00000003
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_shift = 24
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_SW(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_shift)

 #/*define for src_sys field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_shift = 28
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_SYS(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_shift)

 #/*define for src_snoop field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_shift = 30
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_SNOOP(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_shift)

 #/*define for src_gpa field*/
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_shift = 31
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_GPA(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_shift)

 #/*define for SRC_ADDR_LO word*/
 #/*define for src_addr_31_0 field*/
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_offset = 3
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_shift = 0
def SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_SRC_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_shift)

 #/*define for SRC_ADDR_HI word*/
 #/*define for src_addr_63_32 field*/
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_offset = 4
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_shift = 0
def SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_SRC_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_offset = 5
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_offset = 6
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_PHYSICAL_LINEAR packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_offset = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_shift = 0
def SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_shift)

 #/*define for tmz field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_offset = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_shift = 18
def SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_TMZ(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_shift)

 #/*define for cpv field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_cpv_offset = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_cpv_shift = 19
def SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_CPV(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_cpv_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_cpv_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_offset = 1
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_mask = 0x003FFFFF
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_shift = 0
def SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_COUNT(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_shift)

 #/*define for addr_pair_num field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_addr_pair_num_offset = 1
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_addr_pair_num_mask = 0x000000FF
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_addr_pair_num_shift = 24
def SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_ADDR_PAIR_NUM(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_addr_pair_num_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_addr_pair_num_shift)

 #/*define for PARAMETER word*/
 #/*define for dst_mtype field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_mask = 0x00000007
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_shift = 3
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_MTYPE(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_shift)

 #/*define for dst_l2_policy field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_mask = 0x00000003
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_shift = 6
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_L2_POLICY(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_shift)

 #/*define for dst_llc field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_llc_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_llc_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_llc_shift = 8
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_LLC(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_llc_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_llc_shift)

 #/*define for src_mtype field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_mask = 0x00000007
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_shift = 11
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_MTYPE(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_shift)

 #/*define for src_l2_policy field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_mask = 0x00000003
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_shift = 14
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_L2_POLICY(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_shift)

 #/*define for src_llc field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_llc_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_llc_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_llc_shift = 16
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_LLC(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_llc_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_llc_shift)

 #/*define for dst_sw field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_shift = 17
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_SW(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_shift)

 #/*define for dst_gcc field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_shift = 19
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_GCC(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_shift)

 #/*define for dst_sys field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_shift = 20
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_SYS(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_shift)

 #/*define for dst_log field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_shift = 21
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_LOG(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_shift)

 #/*define for dst_snoop field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_shift = 22
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_SNOOP(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_shift)

 #/*define for dst_gpa field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_shift = 23
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_GPA(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_shift)

 #/*define for src_sw field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_mask = 0x00000003
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_shift = 24
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_SW(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_shift)

 #/*define for src_gcc field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_shift = 27
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_GCC(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_shift)

 #/*define for src_sys field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_shift = 28
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_SYS(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_shift)

 #/*define for src_snoop field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_shift = 30
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_SNOOP(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_shift)

 #/*define for src_gpa field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_shift = 31
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_GPA(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_shift)

 #/*define for SRC_ADDR_LO word*/
 #/*define for src_addr_31_0 field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset = 3
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift = 0
def SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_SRC_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift)

 #/*define for SRC_ADDR_HI word*/
 #/*define for src_addr_63_32 field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset = 4
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift = 0
def SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_SRC_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_offset = 5
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_offset = 6
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_BROADCAST_LINEAR packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_offset = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_shift = 0
def SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_shift)

 #/*define for encrypt field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_offset = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_mask = 0x00000001
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_shift = 16
def SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_ENCRYPT(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_shift)

 #/*define for tmz field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_offset = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_shift = 18
def SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_TMZ(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_shift)

 #/*define for cpv field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_cpv_offset = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_cpv_shift = 19
def SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_CPV(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_cpv_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_cpv_shift)

 #/*define for broadcast field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_offset = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_mask = 0x00000001
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_shift = 27
def SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_BROADCAST(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_offset = 1
SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_mask = 0x3FFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_shift = 0
def SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_COUNT(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_shift)

 #/*define for PARAMETER word*/
 #/*define for dst2_sw field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_offset = 2
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_mask = 0x00000003
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_shift = 8
def SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_DST2_SW(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_shift)

 #/*define for dst2_cache_policy field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_cache_policy_offset = 2
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_cache_policy_shift = 10
def SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_DST2_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_cache_policy_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_cache_policy_shift)

 #/*define for dst1_sw field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_offset = 2
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_mask = 0x00000003
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_shift = 16
def SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_DST1_SW(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_shift)

 #/*define for dst1_cache_policy field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_cache_policy_offset = 2
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_cache_policy_shift = 18
def SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_DST1_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_cache_policy_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_cache_policy_shift)

 #/*define for src_sw field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_offset = 2
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_mask = 0x00000003
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_shift = 24
def SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_SRC_SW(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_shift)

 #/*define for src_cache_policy field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_cache_policy_offset = 2
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_cache_policy_shift = 26
def SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_SRC_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_cache_policy_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_cache_policy_shift)

 #/*define for SRC_ADDR_LO word*/
 #/*define for src_addr_31_0 field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset = 3
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift = 0
def SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_SRC_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift)

 #/*define for SRC_ADDR_HI word*/
 #/*define for src_addr_63_32 field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset = 4
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift = 0
def SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_SRC_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift)

 #/*define for DST1_ADDR_LO word*/
 #/*define for dst1_addr_31_0 field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_offset = 5
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_shift = 0
def SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_DST1_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_shift)

 #/*define for DST1_ADDR_HI word*/
 #/*define for dst1_addr_63_32 field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_offset = 6
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_shift = 0
def SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_DST1_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_shift)

 #/*define for DST2_ADDR_LO word*/
 #/*define for dst2_addr_31_0 field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_offset = 7
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_shift = 0
def SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_DST2_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_shift)

 #/*define for DST2_ADDR_HI word*/
 #/*define for dst2_addr_63_32 field*/
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_offset = 8
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_shift = 0
def SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_DST2_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_LINEAR_SUBWIN packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_shift)

 #/*define for tmz field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_shift = 18
def SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_TMZ(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_shift)

 #/*define for cpv field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_cpv_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_cpv_shift = 19
def SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_CPV(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_cpv_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_cpv_shift)

 #/*define for elementsize field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_shift = 29
def SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_ELEMENTSIZE(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_shift)

 #/*define for SRC_ADDR_LO word*/
 #/*define for src_addr_31_0 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_SRC_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_shift)

 #/*define for SRC_ADDR_HI word*/
 #/*define for src_addr_63_32 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_SRC_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_shift)

 #/*define for DW_3 word*/
 #/*define for src_x field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_offset = 3
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_SRC_X(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_shift)

 #/*define for src_y field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_offset = 3
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_shift = 16
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_SRC_Y(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_shift)

 #/*define for DW_4 word*/
 #/*define for src_z field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_offset = 4
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_mask = 0x00001FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_SRC_Z(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_shift)

 #/*define for src_pitch field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_offset = 4
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_mask = 0x0007FFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_shift = 13
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_SRC_PITCH(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_shift)

 #/*define for DW_5 word*/
 #/*define for src_slice_pitch field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_offset = 5
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_mask = 0x0FFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_SRC_SLICE_PITCH(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_offset = 6
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_offset = 7
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for DW_8 word*/
 #/*define for dst_x field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_offset = 8
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_DST_X(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_shift)

 #/*define for dst_y field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_offset = 8
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_shift = 16
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_DST_Y(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_shift)

 #/*define for DW_9 word*/
 #/*define for dst_z field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_offset = 9
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_mask = 0x00001FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_DST_Z(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_shift)

 #/*define for dst_pitch field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_offset = 9
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_mask = 0x0007FFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_shift = 13
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_DST_PITCH(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_shift)

 #/*define for DW_10 word*/
 #/*define for dst_slice_pitch field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_offset = 10
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_mask = 0x0FFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_DST_SLICE_PITCH(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_shift)

 #/*define for DW_11 word*/
 #/*define for rect_x field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_offset = 11
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_RECT_X(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_shift)

 #/*define for rect_y field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_offset = 11
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_shift = 16
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_RECT_Y(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_shift)

 #/*define for DW_12 word*/
 #/*define for rect_z field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_mask = 0x00001FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_RECT_Z(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_shift)

 #/*define for dst_sw field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_shift = 16
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_DST_SW(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_shift)

 #/*define for dst_cache_policy field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_cache_policy_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_cache_policy_shift = 18
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_DST_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_cache_policy_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_cache_policy_shift)

 #/*define for src_sw field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_shift = 24
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_SRC_SW(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_shift)

 #/*define for src_cache_policy field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_cache_policy_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_cache_policy_shift = 26
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_SRC_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_cache_policy_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_cache_policy_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_op_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_op_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_op_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_sub_op_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_sub_op_shift)

 #/*define for tmz field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_tmz_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_tmz_shift = 18
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_TMZ(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_tmz_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_tmz_shift)

 #/*define for cpv field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_cpv_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_cpv_shift = 19
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_CPV(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_cpv_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_cpv_shift)

 #/*define for SRC_ADDR_LO word*/
 #/*define for src_addr_31_0 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_LO_src_addr_31_0_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_LO_SRC_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_LO_src_addr_31_0_shift)

 #/*define for SRC_ADDR_HI word*/
 #/*define for src_addr_63_32 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_HI_src_addr_63_32_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_HI_SRC_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_HI_src_addr_63_32_shift)

 #/*define for DW_3 word*/
 #/*define for src_x field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_3_src_x_offset = 3
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_3_src_x_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_3_src_x_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_3_SRC_X(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_3_src_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_3_src_x_shift)

 #/*define for DW_4 word*/
 #/*define for src_y field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_4_src_y_offset = 4
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_4_src_y_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_4_src_y_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_4_SRC_Y(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_4_src_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_4_src_y_shift)

 #/*define for DW_5 word*/
 #/*define for src_z field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_5_src_z_offset = 5
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_5_src_z_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_5_src_z_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_5_SRC_Z(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_5_src_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_5_src_z_shift)

 #/*define for DW_6 word*/
 #/*define for src_pitch field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_6_src_pitch_offset = 6
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_6_src_pitch_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_6_src_pitch_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_6_SRC_PITCH(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_6_src_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_6_src_pitch_shift)

 #/*define for DW_7 word*/
 #/*define for src_slice_pitch_31_0 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_7_src_slice_pitch_31_0_offset = 7
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_7_src_slice_pitch_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_7_src_slice_pitch_31_0_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_7_SRC_SLICE_PITCH_31_0(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_7_src_slice_pitch_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_7_src_slice_pitch_31_0_shift)

 #/*define for DW_8 word*/
 #/*define for src_slice_pitch_47_32 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_8_src_slice_pitch_47_32_offset = 8
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_8_src_slice_pitch_47_32_mask = 0x0000FFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_8_src_slice_pitch_47_32_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_8_SRC_SLICE_PITCH_47_32(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_8_src_slice_pitch_47_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_8_src_slice_pitch_47_32_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_LO_dst_addr_31_0_offset = 9
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_HI_dst_addr_63_32_offset = 10
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for DW_11 word*/
 #/*define for dst_x field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_11_dst_x_offset = 11
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_11_dst_x_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_11_dst_x_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_11_DST_X(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_11_dst_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_11_dst_x_shift)

 #/*define for DW_12 word*/
 #/*define for dst_y field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_12_dst_y_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_12_dst_y_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_12_dst_y_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_12_DST_Y(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_12_dst_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_12_dst_y_shift)

 #/*define for DW_13 word*/
 #/*define for dst_z field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_13_dst_z_offset = 13
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_13_dst_z_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_13_dst_z_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_13_DST_Z(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_13_dst_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_13_dst_z_shift)

 #/*define for DW_14 word*/
 #/*define for dst_pitch field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_14_dst_pitch_offset = 14
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_14_dst_pitch_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_14_dst_pitch_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_14_DST_PITCH(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_14_dst_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_14_dst_pitch_shift)

 #/*define for DW_15 word*/
 #/*define for dst_slice_pitch_31_0 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_15_dst_slice_pitch_31_0_offset = 15
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_15_dst_slice_pitch_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_15_dst_slice_pitch_31_0_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_15_DST_SLICE_PITCH_31_0(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_15_dst_slice_pitch_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_15_dst_slice_pitch_31_0_shift)

 #/*define for DW_16 word*/
 #/*define for dst_slice_pitch_47_32 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_slice_pitch_47_32_offset = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_slice_pitch_47_32_mask = 0x0000FFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_slice_pitch_47_32_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_DST_SLICE_PITCH_47_32(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_slice_pitch_47_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_slice_pitch_47_32_shift)

 #/*define for dst_sw field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_sw_offset = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_sw_shift = 16
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_DST_SW(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_sw_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_sw_shift)

 #/*define for dst_policy field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_policy_offset = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_policy_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_policy_shift = 18
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_DST_POLICY(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_policy_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_policy_shift)

 #/*define for src_sw field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_sw_offset = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_sw_shift = 24
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_SRC_SW(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_sw_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_sw_shift)

 #/*define for src_policy field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_policy_offset = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_policy_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_policy_shift = 26
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_SRC_POLICY(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_policy_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_policy_shift)

 #/*define for DW_17 word*/
 #/*define for rect_x field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_17_rect_x_offset = 17
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_17_rect_x_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_17_rect_x_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_17_RECT_X(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_17_rect_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_17_rect_x_shift)

 #/*define for DW_18 word*/
 #/*define for rect_y field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_18_rect_y_offset = 18
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_18_rect_y_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_18_rect_y_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_18_RECT_Y(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_18_rect_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_18_rect_y_shift)

 #/*define for DW_19 word*/
 #/*define for rect_z field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_19_rect_z_offset = 19
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_19_rect_z_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_19_rect_z_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_19_RECT_Z(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_19_rect_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_19_rect_z_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_LINEAR_SUBWIN_BC packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_shift)

 #/*define for elementsize field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_shift = 29
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_ELEMENTSIZE(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_shift)

 #/*define for SRC_ADDR_LO word*/
 #/*define for src_addr_31_0 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_SRC_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_shift)

 #/*define for SRC_ADDR_HI word*/
 #/*define for src_addr_63_32 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_SRC_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_shift)

 #/*define for DW_3 word*/
 #/*define for src_x field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_offset = 3
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_SRC_X(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_shift)

 #/*define for src_y field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_offset = 3
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_shift = 16
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_SRC_Y(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_shift)

 #/*define for DW_4 word*/
 #/*define for src_z field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_offset = 4
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_mask = 0x000007FF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_SRC_Z(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_shift)

 #/*define for src_pitch field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_offset = 4
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_shift = 13
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_SRC_PITCH(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_shift)

 #/*define for DW_5 word*/
 #/*define for src_slice_pitch field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_offset = 5
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_mask = 0x0FFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_SRC_SLICE_PITCH(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_offset = 6
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_offset = 7
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for DW_8 word*/
 #/*define for dst_x field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_offset = 8
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_DST_X(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_shift)

 #/*define for dst_y field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_offset = 8
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_shift = 16
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_DST_Y(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_shift)

 #/*define for DW_9 word*/
 #/*define for dst_z field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_offset = 9
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_mask = 0x000007FF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_DST_Z(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_shift)

 #/*define for dst_pitch field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_offset = 9
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_shift = 13
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_DST_PITCH(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_shift)

 #/*define for DW_10 word*/
 #/*define for dst_slice_pitch field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_offset = 10
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_mask = 0x0FFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_DST_SLICE_PITCH(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_shift)

 #/*define for DW_11 word*/
 #/*define for rect_x field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_offset = 11
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_RECT_X(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_shift)

 #/*define for rect_y field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_offset = 11
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_shift = 16
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_RECT_Y(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_shift)

 #/*define for DW_12 word*/
 #/*define for rect_z field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_mask = 0x000007FF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_shift = 0
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_RECT_Z(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_shift)

 #/*define for dst_sw field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_shift = 16
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_DST_SW(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_shift)

 #/*define for dst_ha field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_shift = 19
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_DST_HA(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_shift)

 #/*define for src_sw field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_shift = 24
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_SRC_SW(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_shift)

 #/*define for src_ha field*/
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_shift = 27
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_SRC_HA(x): return (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_TILED packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_TILED_HEADER_op_offset = 0
SDMA_PKT_COPY_TILED_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_HEADER_op_shift = 0
def SDMA_PKT_COPY_TILED_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_TILED_HEADER_op_mask) << SDMA_PKT_COPY_TILED_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_TILED_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_TILED_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_TILED_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_TILED_HEADER_sub_op_mask) << SDMA_PKT_COPY_TILED_HEADER_sub_op_shift)

 #/*define for encrypt field*/
SDMA_PKT_COPY_TILED_HEADER_encrypt_offset = 0
SDMA_PKT_COPY_TILED_HEADER_encrypt_mask = 0x00000001
SDMA_PKT_COPY_TILED_HEADER_encrypt_shift = 16
def SDMA_PKT_COPY_TILED_HEADER_ENCRYPT(x): return (((x) & SDMA_PKT_COPY_TILED_HEADER_encrypt_mask) << SDMA_PKT_COPY_TILED_HEADER_encrypt_shift)

 #/*define for tmz field*/
SDMA_PKT_COPY_TILED_HEADER_tmz_offset = 0
SDMA_PKT_COPY_TILED_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_TILED_HEADER_tmz_shift = 18
def SDMA_PKT_COPY_TILED_HEADER_TMZ(x): return (((x) & SDMA_PKT_COPY_TILED_HEADER_tmz_mask) << SDMA_PKT_COPY_TILED_HEADER_tmz_shift)

 #/*define for cpv field*/
SDMA_PKT_COPY_TILED_HEADER_cpv_offset = 0
SDMA_PKT_COPY_TILED_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_TILED_HEADER_cpv_shift = 19
def SDMA_PKT_COPY_TILED_HEADER_CPV(x): return (((x) & SDMA_PKT_COPY_TILED_HEADER_cpv_mask) << SDMA_PKT_COPY_TILED_HEADER_cpv_shift)

 #/*define for detile field*/
SDMA_PKT_COPY_TILED_HEADER_detile_offset = 0
SDMA_PKT_COPY_TILED_HEADER_detile_mask = 0x00000001
SDMA_PKT_COPY_TILED_HEADER_detile_shift = 31
def SDMA_PKT_COPY_TILED_HEADER_DETILE(x): return (((x) & SDMA_PKT_COPY_TILED_HEADER_detile_mask) << SDMA_PKT_COPY_TILED_HEADER_detile_shift)

 #/*define for TILED_ADDR_LO word*/
 #/*define for tiled_addr_31_0 field*/
SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_offset = 1
SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_shift = 0
def SDMA_PKT_COPY_TILED_TILED_ADDR_LO_TILED_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_mask) << SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_shift)

 #/*define for TILED_ADDR_HI word*/
 #/*define for tiled_addr_63_32 field*/
SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_offset = 2
SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_shift = 0
def SDMA_PKT_COPY_TILED_TILED_ADDR_HI_TILED_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_mask) << SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_shift)

 #/*define for DW_3 word*/
 #/*define for width field*/
SDMA_PKT_COPY_TILED_DW_3_width_offset = 3
SDMA_PKT_COPY_TILED_DW_3_width_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_DW_3_width_shift = 0
def SDMA_PKT_COPY_TILED_DW_3_WIDTH(x): return (((x) & SDMA_PKT_COPY_TILED_DW_3_width_mask) << SDMA_PKT_COPY_TILED_DW_3_width_shift)

 #/*define for DW_4 word*/
 #/*define for height field*/
SDMA_PKT_COPY_TILED_DW_4_height_offset = 4
SDMA_PKT_COPY_TILED_DW_4_height_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_DW_4_height_shift = 0
def SDMA_PKT_COPY_TILED_DW_4_HEIGHT(x): return (((x) & SDMA_PKT_COPY_TILED_DW_4_height_mask) << SDMA_PKT_COPY_TILED_DW_4_height_shift)

 #/*define for depth field*/
SDMA_PKT_COPY_TILED_DW_4_depth_offset = 4
SDMA_PKT_COPY_TILED_DW_4_depth_mask = 0x00001FFF
SDMA_PKT_COPY_TILED_DW_4_depth_shift = 16
def SDMA_PKT_COPY_TILED_DW_4_DEPTH(x): return (((x) & SDMA_PKT_COPY_TILED_DW_4_depth_mask) << SDMA_PKT_COPY_TILED_DW_4_depth_shift)

 #/*define for DW_5 word*/
 #/*define for element_size field*/
SDMA_PKT_COPY_TILED_DW_5_element_size_offset = 5
SDMA_PKT_COPY_TILED_DW_5_element_size_mask = 0x00000007
SDMA_PKT_COPY_TILED_DW_5_element_size_shift = 0
def SDMA_PKT_COPY_TILED_DW_5_ELEMENT_SIZE(x): return (((x) & SDMA_PKT_COPY_TILED_DW_5_element_size_mask) << SDMA_PKT_COPY_TILED_DW_5_element_size_shift)

 #/*define for swizzle_mode field*/
SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_offset = 5
SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_mask = 0x0000001F
SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_shift = 3
def SDMA_PKT_COPY_TILED_DW_5_SWIZZLE_MODE(x): return (((x) & SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_mask) << SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_shift)

 #/*define for dimension field*/
SDMA_PKT_COPY_TILED_DW_5_dimension_offset = 5
SDMA_PKT_COPY_TILED_DW_5_dimension_mask = 0x00000003
SDMA_PKT_COPY_TILED_DW_5_dimension_shift = 9
def SDMA_PKT_COPY_TILED_DW_5_DIMENSION(x): return (((x) & SDMA_PKT_COPY_TILED_DW_5_dimension_mask) << SDMA_PKT_COPY_TILED_DW_5_dimension_shift)

 #/*define for mip_max field*/
SDMA_PKT_COPY_TILED_DW_5_mip_max_offset = 5
SDMA_PKT_COPY_TILED_DW_5_mip_max_mask = 0x0000000F
SDMA_PKT_COPY_TILED_DW_5_mip_max_shift = 16
def SDMA_PKT_COPY_TILED_DW_5_MIP_MAX(x): return (((x) & SDMA_PKT_COPY_TILED_DW_5_mip_max_mask) << SDMA_PKT_COPY_TILED_DW_5_mip_max_shift)

 #/*define for DW_6 word*/
 #/*define for x field*/
SDMA_PKT_COPY_TILED_DW_6_x_offset = 6
SDMA_PKT_COPY_TILED_DW_6_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_DW_6_x_shift = 0
def SDMA_PKT_COPY_TILED_DW_6_X(x): return (((x) & SDMA_PKT_COPY_TILED_DW_6_x_mask) << SDMA_PKT_COPY_TILED_DW_6_x_shift)

 #/*define for y field*/
SDMA_PKT_COPY_TILED_DW_6_y_offset = 6
SDMA_PKT_COPY_TILED_DW_6_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_DW_6_y_shift = 16
def SDMA_PKT_COPY_TILED_DW_6_Y(x): return (((x) & SDMA_PKT_COPY_TILED_DW_6_y_mask) << SDMA_PKT_COPY_TILED_DW_6_y_shift)

 #/*define for DW_7 word*/
 #/*define for z field*/
SDMA_PKT_COPY_TILED_DW_7_z_offset = 7
SDMA_PKT_COPY_TILED_DW_7_z_mask = 0x00001FFF
SDMA_PKT_COPY_TILED_DW_7_z_shift = 0
def SDMA_PKT_COPY_TILED_DW_7_Z(x): return (((x) & SDMA_PKT_COPY_TILED_DW_7_z_mask) << SDMA_PKT_COPY_TILED_DW_7_z_shift)

 #/*define for linear_sw field*/
SDMA_PKT_COPY_TILED_DW_7_linear_sw_offset = 7
SDMA_PKT_COPY_TILED_DW_7_linear_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_DW_7_linear_sw_shift = 16
def SDMA_PKT_COPY_TILED_DW_7_LINEAR_SW(x): return (((x) & SDMA_PKT_COPY_TILED_DW_7_linear_sw_mask) << SDMA_PKT_COPY_TILED_DW_7_linear_sw_shift)

 #/*define for linear_cache_policy field*/
SDMA_PKT_COPY_TILED_DW_7_linear_cache_policy_offset = 7
SDMA_PKT_COPY_TILED_DW_7_linear_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_TILED_DW_7_linear_cache_policy_shift = 18
def SDMA_PKT_COPY_TILED_DW_7_LINEAR_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_TILED_DW_7_linear_cache_policy_mask) << SDMA_PKT_COPY_TILED_DW_7_linear_cache_policy_shift)

 #/*define for tile_sw field*/
SDMA_PKT_COPY_TILED_DW_7_tile_sw_offset = 7
SDMA_PKT_COPY_TILED_DW_7_tile_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_DW_7_tile_sw_shift = 24
def SDMA_PKT_COPY_TILED_DW_7_TILE_SW(x): return (((x) & SDMA_PKT_COPY_TILED_DW_7_tile_sw_mask) << SDMA_PKT_COPY_TILED_DW_7_tile_sw_shift)

 #/*define for tile_cache_policy field*/
SDMA_PKT_COPY_TILED_DW_7_tile_cache_policy_offset = 7
SDMA_PKT_COPY_TILED_DW_7_tile_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_TILED_DW_7_tile_cache_policy_shift = 26
def SDMA_PKT_COPY_TILED_DW_7_TILE_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_TILED_DW_7_tile_cache_policy_mask) << SDMA_PKT_COPY_TILED_DW_7_tile_cache_policy_shift)

 #/*define for LINEAR_ADDR_LO word*/
 #/*define for linear_addr_31_0 field*/
SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_offset = 8
SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0
def SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_LINEAR_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_mask) << SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_shift)

 #/*define for LINEAR_ADDR_HI word*/
 #/*define for linear_addr_63_32 field*/
SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_offset = 9
SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0
def SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_LINEAR_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_mask) << SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_shift)

 #/*define for LINEAR_PITCH word*/
 #/*define for linear_pitch field*/
SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_offset = 10
SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_mask = 0x0007FFFF
SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_shift = 0
def SDMA_PKT_COPY_TILED_LINEAR_PITCH_LINEAR_PITCH(x): return (((x) & SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_mask) << SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_shift)

 #/*define for LINEAR_SLICE_PITCH word*/
 #/*define for linear_slice_pitch field*/
SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_offset = 11
SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_shift = 0
def SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_LINEAR_SLICE_PITCH(x): return (((x) & SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_mask) << SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_COPY_TILED_COUNT_count_offset = 12
SDMA_PKT_COPY_TILED_COUNT_count_mask = 0x3FFFFFFF
SDMA_PKT_COPY_TILED_COUNT_count_shift = 0
def SDMA_PKT_COPY_TILED_COUNT_COUNT(x): return (((x) & SDMA_PKT_COPY_TILED_COUNT_count_mask) << SDMA_PKT_COPY_TILED_COUNT_count_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_TILED_BC packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_TILED_BC_HEADER_op_offset = 0
SDMA_PKT_COPY_TILED_BC_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_BC_HEADER_op_shift = 0
def SDMA_PKT_COPY_TILED_BC_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_TILED_BC_HEADER_op_mask) << SDMA_PKT_COPY_TILED_BC_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_TILED_BC_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_mask) << SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_shift)

 #/*define for detile field*/
SDMA_PKT_COPY_TILED_BC_HEADER_detile_offset = 0
SDMA_PKT_COPY_TILED_BC_HEADER_detile_mask = 0x00000001
SDMA_PKT_COPY_TILED_BC_HEADER_detile_shift = 31
def SDMA_PKT_COPY_TILED_BC_HEADER_DETILE(x): return (((x) & SDMA_PKT_COPY_TILED_BC_HEADER_detile_mask) << SDMA_PKT_COPY_TILED_BC_HEADER_detile_shift)

 #/*define for TILED_ADDR_LO word*/
 #/*define for tiled_addr_31_0 field*/
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_offset = 1
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_shift = 0
def SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_TILED_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_mask) << SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_shift)

 #/*define for TILED_ADDR_HI word*/
 #/*define for tiled_addr_63_32 field*/
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_offset = 2
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_shift = 0
def SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_TILED_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_mask) << SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_shift)

 #/*define for DW_3 word*/
 #/*define for width field*/
SDMA_PKT_COPY_TILED_BC_DW_3_width_offset = 3
SDMA_PKT_COPY_TILED_BC_DW_3_width_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_BC_DW_3_width_shift = 0
def SDMA_PKT_COPY_TILED_BC_DW_3_WIDTH(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_3_width_mask) << SDMA_PKT_COPY_TILED_BC_DW_3_width_shift)

 #/*define for DW_4 word*/
 #/*define for height field*/
SDMA_PKT_COPY_TILED_BC_DW_4_height_offset = 4
SDMA_PKT_COPY_TILED_BC_DW_4_height_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_BC_DW_4_height_shift = 0
def SDMA_PKT_COPY_TILED_BC_DW_4_HEIGHT(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_4_height_mask) << SDMA_PKT_COPY_TILED_BC_DW_4_height_shift)

 #/*define for depth field*/
SDMA_PKT_COPY_TILED_BC_DW_4_depth_offset = 4
SDMA_PKT_COPY_TILED_BC_DW_4_depth_mask = 0x000007FF
SDMA_PKT_COPY_TILED_BC_DW_4_depth_shift = 16
def SDMA_PKT_COPY_TILED_BC_DW_4_DEPTH(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_4_depth_mask) << SDMA_PKT_COPY_TILED_BC_DW_4_depth_shift)

 #/*define for DW_5 word*/
 #/*define for element_size field*/
SDMA_PKT_COPY_TILED_BC_DW_5_element_size_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_element_size_mask = 0x00000007
SDMA_PKT_COPY_TILED_BC_DW_5_element_size_shift = 0
def SDMA_PKT_COPY_TILED_BC_DW_5_ELEMENT_SIZE(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_element_size_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_element_size_shift)

 #/*define for array_mode field*/
SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_mask = 0x0000000F
SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_shift = 3
def SDMA_PKT_COPY_TILED_BC_DW_5_ARRAY_MODE(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_shift)

 #/*define for mit_mode field*/
SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_mask = 0x00000007
SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_shift = 8
def SDMA_PKT_COPY_TILED_BC_DW_5_MIT_MODE(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_shift)

 #/*define for tilesplit_size field*/
SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_mask = 0x00000007
SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_shift = 11
def SDMA_PKT_COPY_TILED_BC_DW_5_TILESPLIT_SIZE(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_shift)

 #/*define for bank_w field*/
SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_mask = 0x00000003
SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_shift = 15
def SDMA_PKT_COPY_TILED_BC_DW_5_BANK_W(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_shift)

 #/*define for bank_h field*/
SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_mask = 0x00000003
SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_shift = 18
def SDMA_PKT_COPY_TILED_BC_DW_5_BANK_H(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_shift)

 #/*define for num_bank field*/
SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_mask = 0x00000003
SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_shift = 21
def SDMA_PKT_COPY_TILED_BC_DW_5_NUM_BANK(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_shift)

 #/*define for mat_aspt field*/
SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_mask = 0x00000003
SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_shift = 24
def SDMA_PKT_COPY_TILED_BC_DW_5_MAT_ASPT(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_shift)

 #/*define for pipe_config field*/
SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_mask = 0x0000001F
SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_shift = 26
def SDMA_PKT_COPY_TILED_BC_DW_5_PIPE_CONFIG(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_shift)

 #/*define for DW_6 word*/
 #/*define for x field*/
SDMA_PKT_COPY_TILED_BC_DW_6_x_offset = 6
SDMA_PKT_COPY_TILED_BC_DW_6_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_BC_DW_6_x_shift = 0
def SDMA_PKT_COPY_TILED_BC_DW_6_X(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_6_x_mask) << SDMA_PKT_COPY_TILED_BC_DW_6_x_shift)

 #/*define for y field*/
SDMA_PKT_COPY_TILED_BC_DW_6_y_offset = 6
SDMA_PKT_COPY_TILED_BC_DW_6_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_BC_DW_6_y_shift = 16
def SDMA_PKT_COPY_TILED_BC_DW_6_Y(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_6_y_mask) << SDMA_PKT_COPY_TILED_BC_DW_6_y_shift)

 #/*define for DW_7 word*/
 #/*define for z field*/
SDMA_PKT_COPY_TILED_BC_DW_7_z_offset = 7
SDMA_PKT_COPY_TILED_BC_DW_7_z_mask = 0x000007FF
SDMA_PKT_COPY_TILED_BC_DW_7_z_shift = 0
def SDMA_PKT_COPY_TILED_BC_DW_7_Z(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_7_z_mask) << SDMA_PKT_COPY_TILED_BC_DW_7_z_shift)

 #/*define for linear_sw field*/
SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_offset = 7
SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_shift = 16
def SDMA_PKT_COPY_TILED_BC_DW_7_LINEAR_SW(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_mask) << SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_shift)

 #/*define for tile_sw field*/
SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_offset = 7
SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_shift = 24
def SDMA_PKT_COPY_TILED_BC_DW_7_TILE_SW(x): return (((x) & SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_mask) << SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_shift)

 #/*define for LINEAR_ADDR_LO word*/
 #/*define for linear_addr_31_0 field*/
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_offset = 8
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0
def SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_LINEAR_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_mask) << SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_shift)

 #/*define for LINEAR_ADDR_HI word*/
 #/*define for linear_addr_63_32 field*/
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_offset = 9
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0
def SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_LINEAR_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_mask) << SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_shift)

 #/*define for LINEAR_PITCH word*/
 #/*define for linear_pitch field*/
SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_offset = 10
SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_mask = 0x0007FFFF
SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_shift = 0
def SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_LINEAR_PITCH(x): return (((x) & SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_mask) << SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_shift)

 #/*define for LINEAR_SLICE_PITCH word*/
 #/*define for linear_slice_pitch field*/
SDMA_PKT_COPY_TILED_BC_LINEAR_SLICE_PITCH_linear_slice_pitch_offset = 11
SDMA_PKT_COPY_TILED_BC_LINEAR_SLICE_PITCH_linear_slice_pitch_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_BC_LINEAR_SLICE_PITCH_linear_slice_pitch_shift = 0
def SDMA_PKT_COPY_TILED_BC_LINEAR_SLICE_PITCH_LINEAR_SLICE_PITCH(x): return (((x) & SDMA_PKT_COPY_TILED_BC_LINEAR_SLICE_PITCH_linear_slice_pitch_mask) << SDMA_PKT_COPY_TILED_BC_LINEAR_SLICE_PITCH_linear_slice_pitch_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_COPY_TILED_BC_COUNT_count_offset = 12
SDMA_PKT_COPY_TILED_BC_COUNT_count_mask = 0x000FFFFF
SDMA_PKT_COPY_TILED_BC_COUNT_count_shift = 2
def SDMA_PKT_COPY_TILED_BC_COUNT_COUNT(x): return (((x) & SDMA_PKT_COPY_TILED_BC_COUNT_count_mask) << SDMA_PKT_COPY_TILED_BC_COUNT_count_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_L2T_BROADCAST packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_L2T_BROADCAST_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_shift)

 #/*define for encrypt field*/
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_mask = 0x00000001
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_shift = 16
def SDMA_PKT_COPY_L2T_BROADCAST_HEADER_ENCRYPT(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_shift)

 #/*define for tmz field*/
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_shift = 18
def SDMA_PKT_COPY_L2T_BROADCAST_HEADER_TMZ(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_shift)

 #/*define for cpv field*/
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_cpv_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_cpv_shift = 19
def SDMA_PKT_COPY_L2T_BROADCAST_HEADER_CPV(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_cpv_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_cpv_shift)

 #/*define for videocopy field*/
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_mask = 0x00000001
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_shift = 26
def SDMA_PKT_COPY_L2T_BROADCAST_HEADER_VIDEOCOPY(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_shift)

 #/*define for broadcast field*/
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_mask = 0x00000001
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_shift = 27
def SDMA_PKT_COPY_L2T_BROADCAST_HEADER_BROADCAST(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_shift)

 #/*define for TILED_ADDR_LO_0 word*/
 #/*define for tiled_addr0_31_0 field*/
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_offset = 1
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_TILED_ADDR0_31_0(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_mask) << SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_shift)

 #/*define for TILED_ADDR_HI_0 word*/
 #/*define for tiled_addr0_63_32 field*/
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_offset = 2
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_TILED_ADDR0_63_32(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_mask) << SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_shift)

 #/*define for TILED_ADDR_LO_1 word*/
 #/*define for tiled_addr1_31_0 field*/
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_offset = 3
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_TILED_ADDR1_31_0(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_mask) << SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_shift)

 #/*define for TILED_ADDR_HI_1 word*/
 #/*define for tiled_addr1_63_32 field*/
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_offset = 4
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_TILED_ADDR1_63_32(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_mask) << SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_shift)

 #/*define for DW_5 word*/
 #/*define for width field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_offset = 5
SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_mask = 0x00003FFF
SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_DW_5_WIDTH(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_shift)

 #/*define for DW_6 word*/
 #/*define for height field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_offset = 6
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_mask = 0x00003FFF
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_DW_6_HEIGHT(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_shift)

 #/*define for depth field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_offset = 6
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_mask = 0x00001FFF
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_shift = 16
def SDMA_PKT_COPY_L2T_BROADCAST_DW_6_DEPTH(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_shift)

 #/*define for DW_7 word*/
 #/*define for element_size field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_offset = 7
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_mask = 0x00000007
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_DW_7_ELEMENT_SIZE(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_shift)

 #/*define for swizzle_mode field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_offset = 7
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_mask = 0x0000001F
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_shift = 3
def SDMA_PKT_COPY_L2T_BROADCAST_DW_7_SWIZZLE_MODE(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_shift)

 #/*define for dimension field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_offset = 7
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_mask = 0x00000003
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_shift = 9
def SDMA_PKT_COPY_L2T_BROADCAST_DW_7_DIMENSION(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_shift)

 #/*define for mip_max field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_offset = 7
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_mask = 0x0000000F
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_shift = 16
def SDMA_PKT_COPY_L2T_BROADCAST_DW_7_MIP_MAX(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_shift)

 #/*define for DW_8 word*/
 #/*define for x field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_offset = 8
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_mask = 0x00003FFF
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_DW_8_X(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_shift)

 #/*define for y field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_offset = 8
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_mask = 0x00003FFF
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_shift = 16
def SDMA_PKT_COPY_L2T_BROADCAST_DW_8_Y(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_shift)

 #/*define for DW_9 word*/
 #/*define for z field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_offset = 9
SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_mask = 0x00001FFF
SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_DW_9_Z(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_shift)

 #/*define for DW_10 word*/
 #/*define for dst2_sw field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_offset = 10
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_mask = 0x00000003
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_shift = 8
def SDMA_PKT_COPY_L2T_BROADCAST_DW_10_DST2_SW(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_shift)

 #/*define for dst2_cache_policy field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_cache_policy_offset = 10
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_cache_policy_shift = 10
def SDMA_PKT_COPY_L2T_BROADCAST_DW_10_DST2_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_cache_policy_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_cache_policy_shift)

 #/*define for linear_sw field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_offset = 10
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_mask = 0x00000003
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_shift = 16
def SDMA_PKT_COPY_L2T_BROADCAST_DW_10_LINEAR_SW(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_shift)

 #/*define for linear_cache_policy field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_cache_policy_offset = 10
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_cache_policy_shift = 18
def SDMA_PKT_COPY_L2T_BROADCAST_DW_10_LINEAR_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_cache_policy_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_cache_policy_shift)

 #/*define for tile_sw field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_offset = 10
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_mask = 0x00000003
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_shift = 24
def SDMA_PKT_COPY_L2T_BROADCAST_DW_10_TILE_SW(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_shift)

 #/*define for tile_cache_policy field*/
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_cache_policy_offset = 10
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_cache_policy_shift = 26
def SDMA_PKT_COPY_L2T_BROADCAST_DW_10_TILE_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_cache_policy_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_cache_policy_shift)

 #/*define for LINEAR_ADDR_LO word*/
 #/*define for linear_addr_31_0 field*/
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_offset = 11
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_LINEAR_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_mask) << SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_shift)

 #/*define for LINEAR_ADDR_HI word*/
 #/*define for linear_addr_63_32 field*/
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_offset = 12
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_LINEAR_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_mask) << SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_shift)

 #/*define for LINEAR_PITCH word*/
 #/*define for linear_pitch field*/
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_offset = 13
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_mask = 0x0007FFFF
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_LINEAR_PITCH(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_mask) << SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_shift)

 #/*define for LINEAR_SLICE_PITCH word*/
 #/*define for linear_slice_pitch field*/
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_offset = 14
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_LINEAR_SLICE_PITCH(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_mask) << SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_offset = 15
SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_mask = 0x3FFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_shift = 0
def SDMA_PKT_COPY_L2T_BROADCAST_COUNT_COUNT(x): return (((x) & SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_mask) << SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_T2T packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_T2T_HEADER_op_offset = 0
SDMA_PKT_COPY_T2T_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_T2T_HEADER_op_shift = 0
def SDMA_PKT_COPY_T2T_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_T2T_HEADER_op_mask) << SDMA_PKT_COPY_T2T_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_T2T_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_T2T_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_T2T_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_T2T_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_T2T_HEADER_sub_op_mask) << SDMA_PKT_COPY_T2T_HEADER_sub_op_shift)

 #/*define for tmz field*/
SDMA_PKT_COPY_T2T_HEADER_tmz_offset = 0
SDMA_PKT_COPY_T2T_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_T2T_HEADER_tmz_shift = 18
def SDMA_PKT_COPY_T2T_HEADER_TMZ(x): return (((x) & SDMA_PKT_COPY_T2T_HEADER_tmz_mask) << SDMA_PKT_COPY_T2T_HEADER_tmz_shift)

 #/*define for dcc field*/
SDMA_PKT_COPY_T2T_HEADER_dcc_offset = 0
SDMA_PKT_COPY_T2T_HEADER_dcc_mask = 0x00000001
SDMA_PKT_COPY_T2T_HEADER_dcc_shift = 19
def SDMA_PKT_COPY_T2T_HEADER_DCC(x): return (((x) & SDMA_PKT_COPY_T2T_HEADER_dcc_mask) << SDMA_PKT_COPY_T2T_HEADER_dcc_shift)

 #/*define for cpv field*/
SDMA_PKT_COPY_T2T_HEADER_cpv_offset = 0
SDMA_PKT_COPY_T2T_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_T2T_HEADER_cpv_shift = 28
def SDMA_PKT_COPY_T2T_HEADER_CPV(x): return (((x) & SDMA_PKT_COPY_T2T_HEADER_cpv_mask) << SDMA_PKT_COPY_T2T_HEADER_cpv_shift)

 #/*define for dcc_dir field*/
SDMA_PKT_COPY_T2T_HEADER_dcc_dir_offset = 0
SDMA_PKT_COPY_T2T_HEADER_dcc_dir_mask = 0x00000001
SDMA_PKT_COPY_T2T_HEADER_dcc_dir_shift = 31
def SDMA_PKT_COPY_T2T_HEADER_DCC_DIR(x): return (((x) & SDMA_PKT_COPY_T2T_HEADER_dcc_dir_mask) << SDMA_PKT_COPY_T2T_HEADER_dcc_dir_shift)

 #/*define for SRC_ADDR_LO word*/
 #/*define for src_addr_31_0 field*/
SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_shift = 0
def SDMA_PKT_COPY_T2T_SRC_ADDR_LO_SRC_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_shift)

 #/*define for SRC_ADDR_HI word*/
 #/*define for src_addr_63_32 field*/
SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_shift = 0
def SDMA_PKT_COPY_T2T_SRC_ADDR_HI_SRC_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_shift)

 #/*define for DW_3 word*/
 #/*define for src_x field*/
SDMA_PKT_COPY_T2T_DW_3_src_x_offset = 3
SDMA_PKT_COPY_T2T_DW_3_src_x_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_3_src_x_shift = 0
def SDMA_PKT_COPY_T2T_DW_3_SRC_X(x): return (((x) & SDMA_PKT_COPY_T2T_DW_3_src_x_mask) << SDMA_PKT_COPY_T2T_DW_3_src_x_shift)

 #/*define for src_y field*/
SDMA_PKT_COPY_T2T_DW_3_src_y_offset = 3
SDMA_PKT_COPY_T2T_DW_3_src_y_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_3_src_y_shift = 16
def SDMA_PKT_COPY_T2T_DW_3_SRC_Y(x): return (((x) & SDMA_PKT_COPY_T2T_DW_3_src_y_mask) << SDMA_PKT_COPY_T2T_DW_3_src_y_shift)

 #/*define for DW_4 word*/
 #/*define for src_z field*/
SDMA_PKT_COPY_T2T_DW_4_src_z_offset = 4
SDMA_PKT_COPY_T2T_DW_4_src_z_mask = 0x00001FFF
SDMA_PKT_COPY_T2T_DW_4_src_z_shift = 0
def SDMA_PKT_COPY_T2T_DW_4_SRC_Z(x): return (((x) & SDMA_PKT_COPY_T2T_DW_4_src_z_mask) << SDMA_PKT_COPY_T2T_DW_4_src_z_shift)

 #/*define for src_width field*/
SDMA_PKT_COPY_T2T_DW_4_src_width_offset = 4
SDMA_PKT_COPY_T2T_DW_4_src_width_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_4_src_width_shift = 16
def SDMA_PKT_COPY_T2T_DW_4_SRC_WIDTH(x): return (((x) & SDMA_PKT_COPY_T2T_DW_4_src_width_mask) << SDMA_PKT_COPY_T2T_DW_4_src_width_shift)

 #/*define for DW_5 word*/
 #/*define for src_height field*/
SDMA_PKT_COPY_T2T_DW_5_src_height_offset = 5
SDMA_PKT_COPY_T2T_DW_5_src_height_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_5_src_height_shift = 0
def SDMA_PKT_COPY_T2T_DW_5_SRC_HEIGHT(x): return (((x) & SDMA_PKT_COPY_T2T_DW_5_src_height_mask) << SDMA_PKT_COPY_T2T_DW_5_src_height_shift)

 #/*define for src_depth field*/
SDMA_PKT_COPY_T2T_DW_5_src_depth_offset = 5
SDMA_PKT_COPY_T2T_DW_5_src_depth_mask = 0x00001FFF
SDMA_PKT_COPY_T2T_DW_5_src_depth_shift = 16
def SDMA_PKT_COPY_T2T_DW_5_SRC_DEPTH(x): return (((x) & SDMA_PKT_COPY_T2T_DW_5_src_depth_mask) << SDMA_PKT_COPY_T2T_DW_5_src_depth_shift)

 #/*define for DW_6 word*/
 #/*define for src_element_size field*/
SDMA_PKT_COPY_T2T_DW_6_src_element_size_offset = 6
SDMA_PKT_COPY_T2T_DW_6_src_element_size_mask = 0x00000007
SDMA_PKT_COPY_T2T_DW_6_src_element_size_shift = 0
def SDMA_PKT_COPY_T2T_DW_6_SRC_ELEMENT_SIZE(x): return (((x) & SDMA_PKT_COPY_T2T_DW_6_src_element_size_mask) << SDMA_PKT_COPY_T2T_DW_6_src_element_size_shift)

 #/*define for src_swizzle_mode field*/
SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_offset = 6
SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_mask = 0x0000001F
SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_shift = 3
def SDMA_PKT_COPY_T2T_DW_6_SRC_SWIZZLE_MODE(x): return (((x) & SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_mask) << SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_shift)

 #/*define for src_dimension field*/
SDMA_PKT_COPY_T2T_DW_6_src_dimension_offset = 6
SDMA_PKT_COPY_T2T_DW_6_src_dimension_mask = 0x00000003
SDMA_PKT_COPY_T2T_DW_6_src_dimension_shift = 9
def SDMA_PKT_COPY_T2T_DW_6_SRC_DIMENSION(x): return (((x) & SDMA_PKT_COPY_T2T_DW_6_src_dimension_mask) << SDMA_PKT_COPY_T2T_DW_6_src_dimension_shift)

 #/*define for src_mip_max field*/
SDMA_PKT_COPY_T2T_DW_6_src_mip_max_offset = 6
SDMA_PKT_COPY_T2T_DW_6_src_mip_max_mask = 0x0000000F
SDMA_PKT_COPY_T2T_DW_6_src_mip_max_shift = 16
def SDMA_PKT_COPY_T2T_DW_6_SRC_MIP_MAX(x): return (((x) & SDMA_PKT_COPY_T2T_DW_6_src_mip_max_mask) << SDMA_PKT_COPY_T2T_DW_6_src_mip_max_shift)

 #/*define for src_mip_id field*/
SDMA_PKT_COPY_T2T_DW_6_src_mip_id_offset = 6
SDMA_PKT_COPY_T2T_DW_6_src_mip_id_mask = 0x0000000F
SDMA_PKT_COPY_T2T_DW_6_src_mip_id_shift = 20
def SDMA_PKT_COPY_T2T_DW_6_SRC_MIP_ID(x): return (((x) & SDMA_PKT_COPY_T2T_DW_6_src_mip_id_mask) << SDMA_PKT_COPY_T2T_DW_6_src_mip_id_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_offset = 7
SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_COPY_T2T_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_offset = 8
SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_COPY_T2T_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for DW_9 word*/
 #/*define for dst_x field*/
SDMA_PKT_COPY_T2T_DW_9_dst_x_offset = 9
SDMA_PKT_COPY_T2T_DW_9_dst_x_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_9_dst_x_shift = 0
def SDMA_PKT_COPY_T2T_DW_9_DST_X(x): return (((x) & SDMA_PKT_COPY_T2T_DW_9_dst_x_mask) << SDMA_PKT_COPY_T2T_DW_9_dst_x_shift)

 #/*define for dst_y field*/
SDMA_PKT_COPY_T2T_DW_9_dst_y_offset = 9
SDMA_PKT_COPY_T2T_DW_9_dst_y_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_9_dst_y_shift = 16
def SDMA_PKT_COPY_T2T_DW_9_DST_Y(x): return (((x) & SDMA_PKT_COPY_T2T_DW_9_dst_y_mask) << SDMA_PKT_COPY_T2T_DW_9_dst_y_shift)

 #/*define for DW_10 word*/
 #/*define for dst_z field*/
SDMA_PKT_COPY_T2T_DW_10_dst_z_offset = 10
SDMA_PKT_COPY_T2T_DW_10_dst_z_mask = 0x00001FFF
SDMA_PKT_COPY_T2T_DW_10_dst_z_shift = 0
def SDMA_PKT_COPY_T2T_DW_10_DST_Z(x): return (((x) & SDMA_PKT_COPY_T2T_DW_10_dst_z_mask) << SDMA_PKT_COPY_T2T_DW_10_dst_z_shift)

 #/*define for dst_width field*/
SDMA_PKT_COPY_T2T_DW_10_dst_width_offset = 10
SDMA_PKT_COPY_T2T_DW_10_dst_width_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_10_dst_width_shift = 16
def SDMA_PKT_COPY_T2T_DW_10_DST_WIDTH(x): return (((x) & SDMA_PKT_COPY_T2T_DW_10_dst_width_mask) << SDMA_PKT_COPY_T2T_DW_10_dst_width_shift)

 #/*define for DW_11 word*/
 #/*define for dst_height field*/
SDMA_PKT_COPY_T2T_DW_11_dst_height_offset = 11
SDMA_PKT_COPY_T2T_DW_11_dst_height_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_11_dst_height_shift = 0
def SDMA_PKT_COPY_T2T_DW_11_DST_HEIGHT(x): return (((x) & SDMA_PKT_COPY_T2T_DW_11_dst_height_mask) << SDMA_PKT_COPY_T2T_DW_11_dst_height_shift)

 #/*define for dst_depth field*/
SDMA_PKT_COPY_T2T_DW_11_dst_depth_offset = 11
SDMA_PKT_COPY_T2T_DW_11_dst_depth_mask = 0x00001FFF
SDMA_PKT_COPY_T2T_DW_11_dst_depth_shift = 16
def SDMA_PKT_COPY_T2T_DW_11_DST_DEPTH(x): return (((x) & SDMA_PKT_COPY_T2T_DW_11_dst_depth_mask) << SDMA_PKT_COPY_T2T_DW_11_dst_depth_shift)

 #/*define for DW_12 word*/
 #/*define for dst_element_size field*/
SDMA_PKT_COPY_T2T_DW_12_dst_element_size_offset = 12
SDMA_PKT_COPY_T2T_DW_12_dst_element_size_mask = 0x00000007
SDMA_PKT_COPY_T2T_DW_12_dst_element_size_shift = 0
def SDMA_PKT_COPY_T2T_DW_12_DST_ELEMENT_SIZE(x): return (((x) & SDMA_PKT_COPY_T2T_DW_12_dst_element_size_mask) << SDMA_PKT_COPY_T2T_DW_12_dst_element_size_shift)

 #/*define for dst_swizzle_mode field*/
SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_offset = 12
SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_mask = 0x0000001F
SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_shift = 3
def SDMA_PKT_COPY_T2T_DW_12_DST_SWIZZLE_MODE(x): return (((x) & SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_mask) << SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_shift)

 #/*define for dst_dimension field*/
SDMA_PKT_COPY_T2T_DW_12_dst_dimension_offset = 12
SDMA_PKT_COPY_T2T_DW_12_dst_dimension_mask = 0x00000003
SDMA_PKT_COPY_T2T_DW_12_dst_dimension_shift = 9
def SDMA_PKT_COPY_T2T_DW_12_DST_DIMENSION(x): return (((x) & SDMA_PKT_COPY_T2T_DW_12_dst_dimension_mask) << SDMA_PKT_COPY_T2T_DW_12_dst_dimension_shift)

 #/*define for dst_mip_max field*/
SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_offset = 12
SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_mask = 0x0000000F
SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_shift = 16
def SDMA_PKT_COPY_T2T_DW_12_DST_MIP_MAX(x): return (((x) & SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_mask) << SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_shift)

 #/*define for dst_mip_id field*/
SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_offset = 12
SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_mask = 0x0000000F
SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_shift = 20
def SDMA_PKT_COPY_T2T_DW_12_DST_MIP_ID(x): return (((x) & SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_mask) << SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_shift)

 #/*define for DW_13 word*/
 #/*define for rect_x field*/
SDMA_PKT_COPY_T2T_DW_13_rect_x_offset = 13
SDMA_PKT_COPY_T2T_DW_13_rect_x_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_13_rect_x_shift = 0
def SDMA_PKT_COPY_T2T_DW_13_RECT_X(x): return (((x) & SDMA_PKT_COPY_T2T_DW_13_rect_x_mask) << SDMA_PKT_COPY_T2T_DW_13_rect_x_shift)

 #/*define for rect_y field*/
SDMA_PKT_COPY_T2T_DW_13_rect_y_offset = 13
SDMA_PKT_COPY_T2T_DW_13_rect_y_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_13_rect_y_shift = 16
def SDMA_PKT_COPY_T2T_DW_13_RECT_Y(x): return (((x) & SDMA_PKT_COPY_T2T_DW_13_rect_y_mask) << SDMA_PKT_COPY_T2T_DW_13_rect_y_shift)

 #/*define for DW_14 word*/
 #/*define for rect_z field*/
SDMA_PKT_COPY_T2T_DW_14_rect_z_offset = 14
SDMA_PKT_COPY_T2T_DW_14_rect_z_mask = 0x00001FFF
SDMA_PKT_COPY_T2T_DW_14_rect_z_shift = 0
def SDMA_PKT_COPY_T2T_DW_14_RECT_Z(x): return (((x) & SDMA_PKT_COPY_T2T_DW_14_rect_z_mask) << SDMA_PKT_COPY_T2T_DW_14_rect_z_shift)

 #/*define for dst_sw field*/
SDMA_PKT_COPY_T2T_DW_14_dst_sw_offset = 14
SDMA_PKT_COPY_T2T_DW_14_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_T2T_DW_14_dst_sw_shift = 16
def SDMA_PKT_COPY_T2T_DW_14_DST_SW(x): return (((x) & SDMA_PKT_COPY_T2T_DW_14_dst_sw_mask) << SDMA_PKT_COPY_T2T_DW_14_dst_sw_shift)

 #/*define for dst_cache_policy field*/
SDMA_PKT_COPY_T2T_DW_14_dst_cache_policy_offset = 14
SDMA_PKT_COPY_T2T_DW_14_dst_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_T2T_DW_14_dst_cache_policy_shift = 18
def SDMA_PKT_COPY_T2T_DW_14_DST_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_T2T_DW_14_dst_cache_policy_mask) << SDMA_PKT_COPY_T2T_DW_14_dst_cache_policy_shift)

 #/*define for src_sw field*/
SDMA_PKT_COPY_T2T_DW_14_src_sw_offset = 14
SDMA_PKT_COPY_T2T_DW_14_src_sw_mask = 0x00000003
SDMA_PKT_COPY_T2T_DW_14_src_sw_shift = 24
def SDMA_PKT_COPY_T2T_DW_14_SRC_SW(x): return (((x) & SDMA_PKT_COPY_T2T_DW_14_src_sw_mask) << SDMA_PKT_COPY_T2T_DW_14_src_sw_shift)

 #/*define for src_cache_policy field*/
SDMA_PKT_COPY_T2T_DW_14_src_cache_policy_offset = 14
SDMA_PKT_COPY_T2T_DW_14_src_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_T2T_DW_14_src_cache_policy_shift = 26
def SDMA_PKT_COPY_T2T_DW_14_SRC_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_T2T_DW_14_src_cache_policy_mask) << SDMA_PKT_COPY_T2T_DW_14_src_cache_policy_shift)

 #/*define for META_ADDR_LO word*/
 #/*define for meta_addr_31_0 field*/
SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_offset = 15
SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_shift = 0
def SDMA_PKT_COPY_T2T_META_ADDR_LO_META_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_mask) << SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_shift)

 #/*define for META_ADDR_HI word*/
 #/*define for meta_addr_63_32 field*/
SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_offset = 16
SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_shift = 0
def SDMA_PKT_COPY_T2T_META_ADDR_HI_META_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_mask) << SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_shift)

 #/*define for META_CONFIG word*/
 #/*define for data_format field*/
SDMA_PKT_COPY_T2T_META_CONFIG_data_format_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_data_format_mask = 0x0000007F
SDMA_PKT_COPY_T2T_META_CONFIG_data_format_shift = 0
def SDMA_PKT_COPY_T2T_META_CONFIG_DATA_FORMAT(x): return (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_data_format_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_data_format_shift)

 #/*define for color_transform_disable field*/
SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_mask = 0x00000001
SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_shift = 7
def SDMA_PKT_COPY_T2T_META_CONFIG_COLOR_TRANSFORM_DISABLE(x): return (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_shift)

 #/*define for alpha_is_on_msb field*/
SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_mask = 0x00000001
SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_shift = 8
def SDMA_PKT_COPY_T2T_META_CONFIG_ALPHA_IS_ON_MSB(x): return (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_shift)

 #/*define for number_type field*/
SDMA_PKT_COPY_T2T_META_CONFIG_number_type_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_number_type_mask = 0x00000007
SDMA_PKT_COPY_T2T_META_CONFIG_number_type_shift = 9
def SDMA_PKT_COPY_T2T_META_CONFIG_NUMBER_TYPE(x): return (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_number_type_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_number_type_shift)

 #/*define for surface_type field*/
SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_mask = 0x00000003
SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_shift = 12
def SDMA_PKT_COPY_T2T_META_CONFIG_SURFACE_TYPE(x): return (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_shift)

 #/*define for meta_llc field*/
SDMA_PKT_COPY_T2T_META_CONFIG_meta_llc_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_meta_llc_mask = 0x00000001
SDMA_PKT_COPY_T2T_META_CONFIG_meta_llc_shift = 14
def SDMA_PKT_COPY_T2T_META_CONFIG_META_LLC(x): return (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_meta_llc_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_meta_llc_shift)

 #/*define for max_comp_block_size field*/
SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_mask = 0x00000003
SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_shift = 24
def SDMA_PKT_COPY_T2T_META_CONFIG_MAX_COMP_BLOCK_SIZE(x): return (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_shift)

 #/*define for max_uncomp_block_size field*/
SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_mask = 0x00000003
SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_shift = 26
def SDMA_PKT_COPY_T2T_META_CONFIG_MAX_UNCOMP_BLOCK_SIZE(x): return (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_shift)

 #/*define for write_compress_enable field*/
SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_mask = 0x00000001
SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_shift = 28
def SDMA_PKT_COPY_T2T_META_CONFIG_WRITE_COMPRESS_ENABLE(x): return (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_shift)

 #/*define for meta_tmz field*/
SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_mask = 0x00000001
SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_shift = 29
def SDMA_PKT_COPY_T2T_META_CONFIG_META_TMZ(x): return (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_shift)

 #/*define for pipe_aligned field*/
SDMA_PKT_COPY_T2T_META_CONFIG_pipe_aligned_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_pipe_aligned_mask = 0x00000001
SDMA_PKT_COPY_T2T_META_CONFIG_pipe_aligned_shift = 31
def SDMA_PKT_COPY_T2T_META_CONFIG_PIPE_ALIGNED(x): return (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_pipe_aligned_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_pipe_aligned_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_T2T_BC packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_T2T_BC_HEADER_op_offset = 0
SDMA_PKT_COPY_T2T_BC_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_T2T_BC_HEADER_op_shift = 0
def SDMA_PKT_COPY_T2T_BC_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_T2T_BC_HEADER_op_mask) << SDMA_PKT_COPY_T2T_BC_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_T2T_BC_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_mask) << SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_shift)

 #/*define for SRC_ADDR_LO word*/
 #/*define for src_addr_31_0 field*/
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_shift = 0
def SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_SRC_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_shift)

 #/*define for SRC_ADDR_HI word*/
 #/*define for src_addr_63_32 field*/
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_shift = 0
def SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_SRC_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_shift)

 #/*define for DW_3 word*/
 #/*define for src_x field*/
SDMA_PKT_COPY_T2T_BC_DW_3_src_x_offset = 3
SDMA_PKT_COPY_T2T_BC_DW_3_src_x_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_3_src_x_shift = 0
def SDMA_PKT_COPY_T2T_BC_DW_3_SRC_X(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_3_src_x_mask) << SDMA_PKT_COPY_T2T_BC_DW_3_src_x_shift)

 #/*define for src_y field*/
SDMA_PKT_COPY_T2T_BC_DW_3_src_y_offset = 3
SDMA_PKT_COPY_T2T_BC_DW_3_src_y_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_3_src_y_shift = 16
def SDMA_PKT_COPY_T2T_BC_DW_3_SRC_Y(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_3_src_y_mask) << SDMA_PKT_COPY_T2T_BC_DW_3_src_y_shift)

 #/*define for DW_4 word*/
 #/*define for src_z field*/
SDMA_PKT_COPY_T2T_BC_DW_4_src_z_offset = 4
SDMA_PKT_COPY_T2T_BC_DW_4_src_z_mask = 0x000007FF
SDMA_PKT_COPY_T2T_BC_DW_4_src_z_shift = 0
def SDMA_PKT_COPY_T2T_BC_DW_4_SRC_Z(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_4_src_z_mask) << SDMA_PKT_COPY_T2T_BC_DW_4_src_z_shift)

 #/*define for src_width field*/
SDMA_PKT_COPY_T2T_BC_DW_4_src_width_offset = 4
SDMA_PKT_COPY_T2T_BC_DW_4_src_width_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_4_src_width_shift = 16
def SDMA_PKT_COPY_T2T_BC_DW_4_SRC_WIDTH(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_4_src_width_mask) << SDMA_PKT_COPY_T2T_BC_DW_4_src_width_shift)

 #/*define for DW_5 word*/
 #/*define for src_height field*/
SDMA_PKT_COPY_T2T_BC_DW_5_src_height_offset = 5
SDMA_PKT_COPY_T2T_BC_DW_5_src_height_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_5_src_height_shift = 0
def SDMA_PKT_COPY_T2T_BC_DW_5_SRC_HEIGHT(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_5_src_height_mask) << SDMA_PKT_COPY_T2T_BC_DW_5_src_height_shift)

 #/*define for src_depth field*/
SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_offset = 5
SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_mask = 0x000007FF
SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_shift = 16
def SDMA_PKT_COPY_T2T_BC_DW_5_SRC_DEPTH(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_mask) << SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_shift)

 #/*define for DW_6 word*/
 #/*define for src_element_size field*/
SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_mask = 0x00000007
SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_shift = 0
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_ELEMENT_SIZE(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_shift)

 #/*define for src_array_mode field*/
SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_mask = 0x0000000F
SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_shift = 3
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_ARRAY_MODE(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_shift)

 #/*define for src_mit_mode field*/
SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_mask = 0x00000007
SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_shift = 8
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_MIT_MODE(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_shift)

 #/*define for src_tilesplit_size field*/
SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_mask = 0x00000007
SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_shift = 11
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_TILESPLIT_SIZE(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_shift)

 #/*define for src_bank_w field*/
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_shift = 15
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_BANK_W(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_shift)

 #/*define for src_bank_h field*/
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_shift = 18
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_BANK_H(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_shift)

 #/*define for src_num_bank field*/
SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_shift = 21
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_NUM_BANK(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_shift)

 #/*define for src_mat_aspt field*/
SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_shift = 24
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_MAT_ASPT(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_shift)

 #/*define for src_pipe_config field*/
SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_mask = 0x0000001F
SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_shift = 26
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_PIPE_CONFIG(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_offset = 7
SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_offset = 8
SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for DW_9 word*/
 #/*define for dst_x field*/
SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_offset = 9
SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_shift = 0
def SDMA_PKT_COPY_T2T_BC_DW_9_DST_X(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_mask) << SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_shift)

 #/*define for dst_y field*/
SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_offset = 9
SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_shift = 16
def SDMA_PKT_COPY_T2T_BC_DW_9_DST_Y(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_mask) << SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_shift)

 #/*define for DW_10 word*/
 #/*define for dst_z field*/
SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_offset = 10
SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_mask = 0x000007FF
SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_shift = 0
def SDMA_PKT_COPY_T2T_BC_DW_10_DST_Z(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_mask) << SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_shift)

 #/*define for dst_width field*/
SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_offset = 10
SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_shift = 16
def SDMA_PKT_COPY_T2T_BC_DW_10_DST_WIDTH(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_mask) << SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_shift)

 #/*define for DW_11 word*/
 #/*define for dst_height field*/
SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_offset = 11
SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_shift = 0
def SDMA_PKT_COPY_T2T_BC_DW_11_DST_HEIGHT(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_mask) << SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_shift)

 #/*define for dst_depth field*/
SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_offset = 11
SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_mask = 0x00000FFF
SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_shift = 16
def SDMA_PKT_COPY_T2T_BC_DW_11_DST_DEPTH(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_mask) << SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_shift)

 #/*define for DW_12 word*/
 #/*define for dst_element_size field*/
SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_mask = 0x00000007
SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_shift = 0
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_ELEMENT_SIZE(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_shift)

 #/*define for dst_array_mode field*/
SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_mask = 0x0000000F
SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_shift = 3
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_ARRAY_MODE(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_shift)

 #/*define for dst_mit_mode field*/
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_mask = 0x00000007
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_shift = 8
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_MIT_MODE(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_shift)

 #/*define for dst_tilesplit_size field*/
SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_mask = 0x00000007
SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_shift = 11
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_TILESPLIT_SIZE(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_shift)

 #/*define for dst_bank_w field*/
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_shift = 15
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_BANK_W(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_shift)

 #/*define for dst_bank_h field*/
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_shift = 18
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_BANK_H(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_shift)

 #/*define for dst_num_bank field*/
SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_shift = 21
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_NUM_BANK(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_shift)

 #/*define for dst_mat_aspt field*/
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_shift = 24
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_MAT_ASPT(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_shift)

 #/*define for dst_pipe_config field*/
SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_mask = 0x0000001F
SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_shift = 26
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_PIPE_CONFIG(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_shift)

 #/*define for DW_13 word*/
 #/*define for rect_x field*/
SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_offset = 13
SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_shift = 0
def SDMA_PKT_COPY_T2T_BC_DW_13_RECT_X(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_mask) << SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_shift)

 #/*define for rect_y field*/
SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_offset = 13
SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_shift = 16
def SDMA_PKT_COPY_T2T_BC_DW_13_RECT_Y(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_mask) << SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_shift)

 #/*define for DW_14 word*/
 #/*define for rect_z field*/
SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_offset = 14
SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_mask = 0x000007FF
SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_shift = 0
def SDMA_PKT_COPY_T2T_BC_DW_14_RECT_Z(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_mask) << SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_shift)

 #/*define for dst_sw field*/
SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_offset = 14
SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_shift = 16
def SDMA_PKT_COPY_T2T_BC_DW_14_DST_SW(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_mask) << SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_shift)

 #/*define for src_sw field*/
SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_offset = 14
SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_shift = 24
def SDMA_PKT_COPY_T2T_BC_DW_14_SRC_SW(x): return (((x) & SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_mask) << SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_TILED_SUBWIN packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_mask) << SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_TILED_SUBWIN_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_mask) << SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_shift)

 #/*define for tmz field*/
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_shift = 18
def SDMA_PKT_COPY_TILED_SUBWIN_HEADER_TMZ(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_mask) << SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_shift)

 #/*define for dcc field*/
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_shift = 19
def SDMA_PKT_COPY_TILED_SUBWIN_HEADER_DCC(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_mask) << SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_shift)

 #/*define for cpv field*/
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_cpv_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_cpv_shift = 28
def SDMA_PKT_COPY_TILED_SUBWIN_HEADER_CPV(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_HEADER_cpv_mask) << SDMA_PKT_COPY_TILED_SUBWIN_HEADER_cpv_shift)

 #/*define for detile field*/
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_shift = 31
def SDMA_PKT_COPY_TILED_SUBWIN_HEADER_DETILE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_mask) << SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_shift)

 #/*define for TILED_ADDR_LO word*/
 #/*define for tiled_addr_31_0 field*/
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_offset = 1
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_TILED_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_mask) << SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_shift)

 #/*define for TILED_ADDR_HI word*/
 #/*define for tiled_addr_63_32 field*/
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_offset = 2
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_TILED_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_mask) << SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_shift)

 #/*define for DW_3 word*/
 #/*define for tiled_x field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_offset = 3
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_DW_3_TILED_X(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_shift)

 #/*define for tiled_y field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_offset = 3
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_DW_3_TILED_Y(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_shift)

 #/*define for DW_4 word*/
 #/*define for tiled_z field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_offset = 4
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_mask = 0x00001FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_DW_4_TILED_Z(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_shift)

 #/*define for width field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_offset = 4
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_DW_4_WIDTH(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_shift)

 #/*define for DW_5 word*/
 #/*define for height field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_offset = 5
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_DW_5_HEIGHT(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_shift)

 #/*define for depth field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_offset = 5
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_mask = 0x00001FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_DW_5_DEPTH(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_shift)

 #/*define for DW_6 word*/
 #/*define for element_size field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_DW_6_ELEMENT_SIZE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_shift)

 #/*define for swizzle_mode field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_mask = 0x0000001F
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_shift = 3
def SDMA_PKT_COPY_TILED_SUBWIN_DW_6_SWIZZLE_MODE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_shift)

 #/*define for dimension field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_shift = 9
def SDMA_PKT_COPY_TILED_SUBWIN_DW_6_DIMENSION(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_shift)

 #/*define for mip_max field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_mask = 0x0000000F
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_DW_6_MIP_MAX(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_shift)

 #/*define for mip_id field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_mask = 0x0000000F
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_shift = 20
def SDMA_PKT_COPY_TILED_SUBWIN_DW_6_MIP_ID(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_shift)

 #/*define for LINEAR_ADDR_LO word*/
 #/*define for linear_addr_31_0 field*/
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_offset = 7
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_LINEAR_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_mask) << SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_shift)

 #/*define for LINEAR_ADDR_HI word*/
 #/*define for linear_addr_63_32 field*/
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_offset = 8
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_LINEAR_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_mask) << SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_shift)

 #/*define for DW_9 word*/
 #/*define for linear_x field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_offset = 9
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_DW_9_LINEAR_X(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_shift)

 #/*define for linear_y field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_offset = 9
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_DW_9_LINEAR_Y(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_shift)

 #/*define for DW_10 word*/
 #/*define for linear_z field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_offset = 10
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_mask = 0x00001FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_DW_10_LINEAR_Z(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_shift)

 #/*define for linear_pitch field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_offset = 10
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_DW_10_LINEAR_PITCH(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_shift)

 #/*define for DW_11 word*/
 #/*define for linear_slice_pitch field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_offset = 11
SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_mask = 0x0FFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_DW_11_LINEAR_SLICE_PITCH(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_shift)

 #/*define for DW_12 word*/
 #/*define for rect_x field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_offset = 12
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_DW_12_RECT_X(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_shift)

 #/*define for rect_y field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_offset = 12
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_DW_12_RECT_Y(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_shift)

 #/*define for DW_13 word*/
 #/*define for rect_z field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_mask = 0x00001FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_DW_13_RECT_Z(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_shift)

 #/*define for linear_sw field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_DW_13_LINEAR_SW(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_shift)

 #/*define for linear_cache_policy field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_cache_policy_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_cache_policy_shift = 18
def SDMA_PKT_COPY_TILED_SUBWIN_DW_13_LINEAR_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_cache_policy_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_cache_policy_shift)

 #/*define for tile_sw field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_shift = 24
def SDMA_PKT_COPY_TILED_SUBWIN_DW_13_TILE_SW(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_shift)

 #/*define for tile_cache_policy field*/
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_cache_policy_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_cache_policy_shift = 26
def SDMA_PKT_COPY_TILED_SUBWIN_DW_13_TILE_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_cache_policy_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_cache_policy_shift)

 #/*define for META_ADDR_LO word*/
 #/*define for meta_addr_31_0 field*/
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_offset = 14
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_META_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_shift)

 #/*define for META_ADDR_HI word*/
 #/*define for meta_addr_63_32 field*/
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_offset = 15
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_META_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_shift)

 #/*define for META_CONFIG word*/
 #/*define for data_format field*/
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_mask = 0x0000007F
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_DATA_FORMAT(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_shift)

 #/*define for color_transform_disable field*/
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_shift = 7
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_COLOR_TRANSFORM_DISABLE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_shift)

 #/*define for alpha_is_on_msb field*/
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_shift = 8
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_ALPHA_IS_ON_MSB(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_shift)

 #/*define for number_type field*/
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_shift = 9
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_NUMBER_TYPE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_shift)

 #/*define for surface_type field*/
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_shift = 12
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_SURFACE_TYPE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_shift)

 #/*define for meta_llc field*/
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_llc_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_llc_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_llc_shift = 14
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_META_LLC(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_llc_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_llc_shift)

 #/*define for max_comp_block_size field*/
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_shift = 24
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_MAX_COMP_BLOCK_SIZE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_shift)

 #/*define for max_uncomp_block_size field*/
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_shift = 26
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_MAX_UNCOMP_BLOCK_SIZE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_shift)

 #/*define for write_compress_enable field*/
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_shift = 28
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_WRITE_COMPRESS_ENABLE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_shift)

 #/*define for meta_tmz field*/
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_shift = 29
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_META_TMZ(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_shift)

 #/*define for pipe_aligned field*/
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_pipe_aligned_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_pipe_aligned_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_pipe_aligned_shift = 31
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_PIPE_ALIGNED(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_pipe_aligned_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_pipe_aligned_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_TILED_SUBWIN_BC packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_shift)

 #/*define for detile field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_shift = 31
def SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_DETILE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_shift)

 #/*define for TILED_ADDR_LO word*/
 #/*define for tiled_addr_31_0 field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_offset = 1
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_TILED_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_shift)

 #/*define for TILED_ADDR_HI word*/
 #/*define for tiled_addr_63_32 field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_offset = 2
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_TILED_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_shift)

 #/*define for DW_3 word*/
 #/*define for tiled_x field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_offset = 3
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_TILED_X(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_shift)

 #/*define for tiled_y field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_offset = 3
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_TILED_Y(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_shift)

 #/*define for DW_4 word*/
 #/*define for tiled_z field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_offset = 4
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_mask = 0x000007FF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_TILED_Z(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_shift)

 #/*define for width field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_offset = 4
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_WIDTH(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_shift)

 #/*define for DW_5 word*/
 #/*define for height field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_offset = 5
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_HEIGHT(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_shift)

 #/*define for depth field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_offset = 5
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_mask = 0x000007FF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_DEPTH(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_shift)

 #/*define for DW_6 word*/
 #/*define for element_size field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_ELEMENT_SIZE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_shift)

 #/*define for array_mode field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_mask = 0x0000000F
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_shift = 3
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_ARRAY_MODE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_shift)

 #/*define for mit_mode field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_shift = 8
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_MIT_MODE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_shift)

 #/*define for tilesplit_size field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_shift = 11
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_TILESPLIT_SIZE(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_shift)

 #/*define for bank_w field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_shift = 15
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_BANK_W(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_shift)

 #/*define for bank_h field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_shift = 18
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_BANK_H(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_shift)

 #/*define for num_bank field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_shift = 21
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_NUM_BANK(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_shift)

 #/*define for mat_aspt field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_shift = 24
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_MAT_ASPT(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_shift)

 #/*define for pipe_config field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_mask = 0x0000001F
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_shift = 26
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_PIPE_CONFIG(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_shift)

 #/*define for LINEAR_ADDR_LO word*/
 #/*define for linear_addr_31_0 field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_offset = 7
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_LINEAR_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_shift)

 #/*define for LINEAR_ADDR_HI word*/
 #/*define for linear_addr_63_32 field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_offset = 8
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_LINEAR_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_shift)

 #/*define for DW_9 word*/
 #/*define for linear_x field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_offset = 9
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_LINEAR_X(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_shift)

 #/*define for linear_y field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_offset = 9
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_LINEAR_Y(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_shift)

 #/*define for DW_10 word*/
 #/*define for linear_z field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_offset = 10
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_mask = 0x000007FF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_LINEAR_Z(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_shift)

 #/*define for linear_pitch field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_offset = 10
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_LINEAR_PITCH(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_shift)

 #/*define for DW_11 word*/
 #/*define for linear_slice_pitch field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_offset = 11
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_mask = 0x0FFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_LINEAR_SLICE_PITCH(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_shift)

 #/*define for DW_12 word*/
 #/*define for rect_x field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_offset = 12
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_RECT_X(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_shift)

 #/*define for rect_y field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_offset = 12
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_RECT_Y(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_shift)

 #/*define for DW_13 word*/
 #/*define for rect_z field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_mask = 0x000007FF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_shift = 0
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_RECT_Z(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_shift)

 #/*define for linear_sw field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_shift = 16
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_LINEAR_SW(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_shift)

 #/*define for tile_sw field*/
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_shift = 24
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_TILE_SW(x): return (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_shift)


 #/*
#** Definitions for SDMA_PKT_COPY_STRUCT packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COPY_STRUCT_HEADER_op_offset = 0
SDMA_PKT_COPY_STRUCT_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_STRUCT_HEADER_op_shift = 0
def SDMA_PKT_COPY_STRUCT_HEADER_OP(x): return (((x) & SDMA_PKT_COPY_STRUCT_HEADER_op_mask) << SDMA_PKT_COPY_STRUCT_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COPY_STRUCT_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_STRUCT_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_STRUCT_HEADER_sub_op_shift = 8
def SDMA_PKT_COPY_STRUCT_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COPY_STRUCT_HEADER_sub_op_mask) << SDMA_PKT_COPY_STRUCT_HEADER_sub_op_shift)

 #/*define for tmz field*/
SDMA_PKT_COPY_STRUCT_HEADER_tmz_offset = 0
SDMA_PKT_COPY_STRUCT_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_STRUCT_HEADER_tmz_shift = 18
def SDMA_PKT_COPY_STRUCT_HEADER_TMZ(x): return (((x) & SDMA_PKT_COPY_STRUCT_HEADER_tmz_mask) << SDMA_PKT_COPY_STRUCT_HEADER_tmz_shift)

 #/*define for cpv field*/
SDMA_PKT_COPY_STRUCT_HEADER_cpv_offset = 0
SDMA_PKT_COPY_STRUCT_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_STRUCT_HEADER_cpv_shift = 28
def SDMA_PKT_COPY_STRUCT_HEADER_CPV(x): return (((x) & SDMA_PKT_COPY_STRUCT_HEADER_cpv_mask) << SDMA_PKT_COPY_STRUCT_HEADER_cpv_shift)

 #/*define for detile field*/
SDMA_PKT_COPY_STRUCT_HEADER_detile_offset = 0
SDMA_PKT_COPY_STRUCT_HEADER_detile_mask = 0x00000001
SDMA_PKT_COPY_STRUCT_HEADER_detile_shift = 31
def SDMA_PKT_COPY_STRUCT_HEADER_DETILE(x): return (((x) & SDMA_PKT_COPY_STRUCT_HEADER_detile_mask) << SDMA_PKT_COPY_STRUCT_HEADER_detile_shift)

 #/*define for SB_ADDR_LO word*/
 #/*define for sb_addr_31_0 field*/
SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_offset = 1
SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_shift = 0
def SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_SB_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_mask) << SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_shift)

 #/*define for SB_ADDR_HI word*/
 #/*define for sb_addr_63_32 field*/
SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_offset = 2
SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_shift = 0
def SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_SB_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_mask) << SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_shift)

 #/*define for START_INDEX word*/
 #/*define for start_index field*/
SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_offset = 3
SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_mask = 0xFFFFFFFF
SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_shift = 0
def SDMA_PKT_COPY_STRUCT_START_INDEX_START_INDEX(x): return (((x) & SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_mask) << SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_COPY_STRUCT_COUNT_count_offset = 4
SDMA_PKT_COPY_STRUCT_COUNT_count_mask = 0xFFFFFFFF
SDMA_PKT_COPY_STRUCT_COUNT_count_shift = 0
def SDMA_PKT_COPY_STRUCT_COUNT_COUNT(x): return (((x) & SDMA_PKT_COPY_STRUCT_COUNT_count_mask) << SDMA_PKT_COPY_STRUCT_COUNT_count_shift)

 #/*define for DW_5 word*/
 #/*define for stride field*/
SDMA_PKT_COPY_STRUCT_DW_5_stride_offset = 5
SDMA_PKT_COPY_STRUCT_DW_5_stride_mask = 0x000007FF
SDMA_PKT_COPY_STRUCT_DW_5_stride_shift = 0
def SDMA_PKT_COPY_STRUCT_DW_5_STRIDE(x): return (((x) & SDMA_PKT_COPY_STRUCT_DW_5_stride_mask) << SDMA_PKT_COPY_STRUCT_DW_5_stride_shift)

 #/*define for linear_sw field*/
SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_offset = 5
SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_mask = 0x00000003
SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_shift = 16
def SDMA_PKT_COPY_STRUCT_DW_5_LINEAR_SW(x): return (((x) & SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_mask) << SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_shift)

 #/*define for linear_cache_policy field*/
SDMA_PKT_COPY_STRUCT_DW_5_linear_cache_policy_offset = 5
SDMA_PKT_COPY_STRUCT_DW_5_linear_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_STRUCT_DW_5_linear_cache_policy_shift = 18
def SDMA_PKT_COPY_STRUCT_DW_5_LINEAR_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_STRUCT_DW_5_linear_cache_policy_mask) << SDMA_PKT_COPY_STRUCT_DW_5_linear_cache_policy_shift)

 #/*define for struct_sw field*/
SDMA_PKT_COPY_STRUCT_DW_5_struct_sw_offset = 5
SDMA_PKT_COPY_STRUCT_DW_5_struct_sw_mask = 0x00000003
SDMA_PKT_COPY_STRUCT_DW_5_struct_sw_shift = 24
def SDMA_PKT_COPY_STRUCT_DW_5_STRUCT_SW(x): return (((x) & SDMA_PKT_COPY_STRUCT_DW_5_struct_sw_mask) << SDMA_PKT_COPY_STRUCT_DW_5_struct_sw_shift)

 #/*define for struct_cache_policy field*/
SDMA_PKT_COPY_STRUCT_DW_5_struct_cache_policy_offset = 5
SDMA_PKT_COPY_STRUCT_DW_5_struct_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_STRUCT_DW_5_struct_cache_policy_shift = 26
def SDMA_PKT_COPY_STRUCT_DW_5_STRUCT_CACHE_POLICY(x): return (((x) & SDMA_PKT_COPY_STRUCT_DW_5_struct_cache_policy_mask) << SDMA_PKT_COPY_STRUCT_DW_5_struct_cache_policy_shift)

 #/*define for LINEAR_ADDR_LO word*/
 #/*define for linear_addr_31_0 field*/
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_offset = 6
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0
def SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_LINEAR_ADDR_31_0(x): return (((x) & SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_mask) << SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_shift)

 #/*define for LINEAR_ADDR_HI word*/
 #/*define for linear_addr_63_32 field*/
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_offset = 7
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0
def SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_LINEAR_ADDR_63_32(x): return (((x) & SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_mask) << SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_shift)


 #/*
#** Definitions for SDMA_PKT_WRITE_UNTILED packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_WRITE_UNTILED_HEADER_op_offset = 0
SDMA_PKT_WRITE_UNTILED_HEADER_op_mask = 0x000000FF
SDMA_PKT_WRITE_UNTILED_HEADER_op_shift = 0
def SDMA_PKT_WRITE_UNTILED_HEADER_OP(x): return (((x) & SDMA_PKT_WRITE_UNTILED_HEADER_op_mask) << SDMA_PKT_WRITE_UNTILED_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_offset = 0
SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_shift = 8
def SDMA_PKT_WRITE_UNTILED_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_mask) << SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_shift)

 #/*define for encrypt field*/
SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_offset = 0
SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_mask = 0x00000001
SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_shift = 16
def SDMA_PKT_WRITE_UNTILED_HEADER_ENCRYPT(x): return (((x) & SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_mask) << SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_shift)

 #/*define for tmz field*/
SDMA_PKT_WRITE_UNTILED_HEADER_tmz_offset = 0
SDMA_PKT_WRITE_UNTILED_HEADER_tmz_mask = 0x00000001
SDMA_PKT_WRITE_UNTILED_HEADER_tmz_shift = 18
def SDMA_PKT_WRITE_UNTILED_HEADER_TMZ(x): return (((x) & SDMA_PKT_WRITE_UNTILED_HEADER_tmz_mask) << SDMA_PKT_WRITE_UNTILED_HEADER_tmz_shift)

 #/*define for cpv field*/
SDMA_PKT_WRITE_UNTILED_HEADER_cpv_offset = 0
SDMA_PKT_WRITE_UNTILED_HEADER_cpv_mask = 0x00000001
SDMA_PKT_WRITE_UNTILED_HEADER_cpv_shift = 28
def SDMA_PKT_WRITE_UNTILED_HEADER_CPV(x): return (((x) & SDMA_PKT_WRITE_UNTILED_HEADER_cpv_mask) << SDMA_PKT_WRITE_UNTILED_HEADER_cpv_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_offset = 1
SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_offset = 2
SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for DW_3 word*/
 #/*define for count field*/
SDMA_PKT_WRITE_UNTILED_DW_3_count_offset = 3
SDMA_PKT_WRITE_UNTILED_DW_3_count_mask = 0x000FFFFF
SDMA_PKT_WRITE_UNTILED_DW_3_count_shift = 0
def SDMA_PKT_WRITE_UNTILED_DW_3_COUNT(x): return (((x) & SDMA_PKT_WRITE_UNTILED_DW_3_count_mask) << SDMA_PKT_WRITE_UNTILED_DW_3_count_shift)

 #/*define for sw field*/
SDMA_PKT_WRITE_UNTILED_DW_3_sw_offset = 3
SDMA_PKT_WRITE_UNTILED_DW_3_sw_mask = 0x00000003
SDMA_PKT_WRITE_UNTILED_DW_3_sw_shift = 24
def SDMA_PKT_WRITE_UNTILED_DW_3_SW(x): return (((x) & SDMA_PKT_WRITE_UNTILED_DW_3_sw_mask) << SDMA_PKT_WRITE_UNTILED_DW_3_sw_shift)

 #/*define for cache_policy field*/
SDMA_PKT_WRITE_UNTILED_DW_3_cache_policy_offset = 3
SDMA_PKT_WRITE_UNTILED_DW_3_cache_policy_mask = 0x00000007
SDMA_PKT_WRITE_UNTILED_DW_3_cache_policy_shift = 26
def SDMA_PKT_WRITE_UNTILED_DW_3_CACHE_POLICY(x): return (((x) & SDMA_PKT_WRITE_UNTILED_DW_3_cache_policy_mask) << SDMA_PKT_WRITE_UNTILED_DW_3_cache_policy_shift)

 #/*define for DATA0 word*/
 #/*define for data0 field*/
SDMA_PKT_WRITE_UNTILED_DATA0_data0_offset = 4
SDMA_PKT_WRITE_UNTILED_DATA0_data0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_UNTILED_DATA0_data0_shift = 0
def SDMA_PKT_WRITE_UNTILED_DATA0_DATA0(x): return (((x) & SDMA_PKT_WRITE_UNTILED_DATA0_data0_mask) << SDMA_PKT_WRITE_UNTILED_DATA0_data0_shift)


 #/*
#** Definitions for SDMA_PKT_WRITE_TILED packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_WRITE_TILED_HEADER_op_offset = 0
SDMA_PKT_WRITE_TILED_HEADER_op_mask = 0x000000FF
SDMA_PKT_WRITE_TILED_HEADER_op_shift = 0
def SDMA_PKT_WRITE_TILED_HEADER_OP(x): return (((x) & SDMA_PKT_WRITE_TILED_HEADER_op_mask) << SDMA_PKT_WRITE_TILED_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_WRITE_TILED_HEADER_sub_op_offset = 0
SDMA_PKT_WRITE_TILED_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_WRITE_TILED_HEADER_sub_op_shift = 8
def SDMA_PKT_WRITE_TILED_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_WRITE_TILED_HEADER_sub_op_mask) << SDMA_PKT_WRITE_TILED_HEADER_sub_op_shift)

 #/*define for encrypt field*/
SDMA_PKT_WRITE_TILED_HEADER_encrypt_offset = 0
SDMA_PKT_WRITE_TILED_HEADER_encrypt_mask = 0x00000001
SDMA_PKT_WRITE_TILED_HEADER_encrypt_shift = 16
def SDMA_PKT_WRITE_TILED_HEADER_ENCRYPT(x): return (((x) & SDMA_PKT_WRITE_TILED_HEADER_encrypt_mask) << SDMA_PKT_WRITE_TILED_HEADER_encrypt_shift)

 #/*define for tmz field*/
SDMA_PKT_WRITE_TILED_HEADER_tmz_offset = 0
SDMA_PKT_WRITE_TILED_HEADER_tmz_mask = 0x00000001
SDMA_PKT_WRITE_TILED_HEADER_tmz_shift = 18
def SDMA_PKT_WRITE_TILED_HEADER_TMZ(x): return (((x) & SDMA_PKT_WRITE_TILED_HEADER_tmz_mask) << SDMA_PKT_WRITE_TILED_HEADER_tmz_shift)

 #/*define for cpv field*/
SDMA_PKT_WRITE_TILED_HEADER_cpv_offset = 0
SDMA_PKT_WRITE_TILED_HEADER_cpv_mask = 0x00000001
SDMA_PKT_WRITE_TILED_HEADER_cpv_shift = 28
def SDMA_PKT_WRITE_TILED_HEADER_CPV(x): return (((x) & SDMA_PKT_WRITE_TILED_HEADER_cpv_mask) << SDMA_PKT_WRITE_TILED_HEADER_cpv_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_offset = 1
SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_WRITE_TILED_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_offset = 2
SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_WRITE_TILED_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for DW_3 word*/
 #/*define for width field*/
SDMA_PKT_WRITE_TILED_DW_3_width_offset = 3
SDMA_PKT_WRITE_TILED_DW_3_width_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_DW_3_width_shift = 0
def SDMA_PKT_WRITE_TILED_DW_3_WIDTH(x): return (((x) & SDMA_PKT_WRITE_TILED_DW_3_width_mask) << SDMA_PKT_WRITE_TILED_DW_3_width_shift)

 #/*define for DW_4 word*/
 #/*define for height field*/
SDMA_PKT_WRITE_TILED_DW_4_height_offset = 4
SDMA_PKT_WRITE_TILED_DW_4_height_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_DW_4_height_shift = 0
def SDMA_PKT_WRITE_TILED_DW_4_HEIGHT(x): return (((x) & SDMA_PKT_WRITE_TILED_DW_4_height_mask) << SDMA_PKT_WRITE_TILED_DW_4_height_shift)

 #/*define for depth field*/
SDMA_PKT_WRITE_TILED_DW_4_depth_offset = 4
SDMA_PKT_WRITE_TILED_DW_4_depth_mask = 0x00001FFF
SDMA_PKT_WRITE_TILED_DW_4_depth_shift = 16
def SDMA_PKT_WRITE_TILED_DW_4_DEPTH(x): return (((x) & SDMA_PKT_WRITE_TILED_DW_4_depth_mask) << SDMA_PKT_WRITE_TILED_DW_4_depth_shift)

 #/*define for DW_5 word*/
 #/*define for element_size field*/
SDMA_PKT_WRITE_TILED_DW_5_element_size_offset = 5
SDMA_PKT_WRITE_TILED_DW_5_element_size_mask = 0x00000007
SDMA_PKT_WRITE_TILED_DW_5_element_size_shift = 0
def SDMA_PKT_WRITE_TILED_DW_5_ELEMENT_SIZE(x): return (((x) & SDMA_PKT_WRITE_TILED_DW_5_element_size_mask) << SDMA_PKT_WRITE_TILED_DW_5_element_size_shift)

 #/*define for swizzle_mode field*/
SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_offset = 5
SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_mask = 0x0000001F
SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_shift = 3
def SDMA_PKT_WRITE_TILED_DW_5_SWIZZLE_MODE(x): return (((x) & SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_mask) << SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_shift)

 #/*define for dimension field*/
SDMA_PKT_WRITE_TILED_DW_5_dimension_offset = 5
SDMA_PKT_WRITE_TILED_DW_5_dimension_mask = 0x00000003
SDMA_PKT_WRITE_TILED_DW_5_dimension_shift = 9
def SDMA_PKT_WRITE_TILED_DW_5_DIMENSION(x): return (((x) & SDMA_PKT_WRITE_TILED_DW_5_dimension_mask) << SDMA_PKT_WRITE_TILED_DW_5_dimension_shift)

 #/*define for mip_max field*/
SDMA_PKT_WRITE_TILED_DW_5_mip_max_offset = 5
SDMA_PKT_WRITE_TILED_DW_5_mip_max_mask = 0x0000000F
SDMA_PKT_WRITE_TILED_DW_5_mip_max_shift = 16
def SDMA_PKT_WRITE_TILED_DW_5_MIP_MAX(x): return (((x) & SDMA_PKT_WRITE_TILED_DW_5_mip_max_mask) << SDMA_PKT_WRITE_TILED_DW_5_mip_max_shift)

 #/*define for DW_6 word*/
 #/*define for x field*/
SDMA_PKT_WRITE_TILED_DW_6_x_offset = 6
SDMA_PKT_WRITE_TILED_DW_6_x_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_DW_6_x_shift = 0
def SDMA_PKT_WRITE_TILED_DW_6_X(x): return (((x) & SDMA_PKT_WRITE_TILED_DW_6_x_mask) << SDMA_PKT_WRITE_TILED_DW_6_x_shift)

 #/*define for y field*/
SDMA_PKT_WRITE_TILED_DW_6_y_offset = 6
SDMA_PKT_WRITE_TILED_DW_6_y_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_DW_6_y_shift = 16
def SDMA_PKT_WRITE_TILED_DW_6_Y(x): return (((x) & SDMA_PKT_WRITE_TILED_DW_6_y_mask) << SDMA_PKT_WRITE_TILED_DW_6_y_shift)

 #/*define for DW_7 word*/
 #/*define for z field*/
SDMA_PKT_WRITE_TILED_DW_7_z_offset = 7
SDMA_PKT_WRITE_TILED_DW_7_z_mask = 0x00001FFF
SDMA_PKT_WRITE_TILED_DW_7_z_shift = 0
def SDMA_PKT_WRITE_TILED_DW_7_Z(x): return (((x) & SDMA_PKT_WRITE_TILED_DW_7_z_mask) << SDMA_PKT_WRITE_TILED_DW_7_z_shift)

 #/*define for sw field*/
SDMA_PKT_WRITE_TILED_DW_7_sw_offset = 7
SDMA_PKT_WRITE_TILED_DW_7_sw_mask = 0x00000003
SDMA_PKT_WRITE_TILED_DW_7_sw_shift = 24
def SDMA_PKT_WRITE_TILED_DW_7_SW(x): return (((x) & SDMA_PKT_WRITE_TILED_DW_7_sw_mask) << SDMA_PKT_WRITE_TILED_DW_7_sw_shift)

 #/*define for cache_policy field*/
SDMA_PKT_WRITE_TILED_DW_7_cache_policy_offset = 7
SDMA_PKT_WRITE_TILED_DW_7_cache_policy_mask = 0x00000007
SDMA_PKT_WRITE_TILED_DW_7_cache_policy_shift = 26
def SDMA_PKT_WRITE_TILED_DW_7_CACHE_POLICY(x): return (((x) & SDMA_PKT_WRITE_TILED_DW_7_cache_policy_mask) << SDMA_PKT_WRITE_TILED_DW_7_cache_policy_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_WRITE_TILED_COUNT_count_offset = 8
SDMA_PKT_WRITE_TILED_COUNT_count_mask = 0x000FFFFF
SDMA_PKT_WRITE_TILED_COUNT_count_shift = 0
def SDMA_PKT_WRITE_TILED_COUNT_COUNT(x): return (((x) & SDMA_PKT_WRITE_TILED_COUNT_count_mask) << SDMA_PKT_WRITE_TILED_COUNT_count_shift)

 #/*define for DATA0 word*/
 #/*define for data0 field*/
SDMA_PKT_WRITE_TILED_DATA0_data0_offset = 9
SDMA_PKT_WRITE_TILED_DATA0_data0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_TILED_DATA0_data0_shift = 0
def SDMA_PKT_WRITE_TILED_DATA0_DATA0(x): return (((x) & SDMA_PKT_WRITE_TILED_DATA0_data0_mask) << SDMA_PKT_WRITE_TILED_DATA0_data0_shift)


 #/*
#** Definitions for SDMA_PKT_WRITE_TILED_BC packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_WRITE_TILED_BC_HEADER_op_offset = 0
SDMA_PKT_WRITE_TILED_BC_HEADER_op_mask = 0x000000FF
SDMA_PKT_WRITE_TILED_BC_HEADER_op_shift = 0
def SDMA_PKT_WRITE_TILED_BC_HEADER_OP(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_HEADER_op_mask) << SDMA_PKT_WRITE_TILED_BC_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_offset = 0
SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_shift = 8
def SDMA_PKT_WRITE_TILED_BC_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_mask) << SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_offset = 1
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_offset = 2
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for DW_3 word*/
 #/*define for width field*/
SDMA_PKT_WRITE_TILED_BC_DW_3_width_offset = 3
SDMA_PKT_WRITE_TILED_BC_DW_3_width_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_BC_DW_3_width_shift = 0
def SDMA_PKT_WRITE_TILED_BC_DW_3_WIDTH(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_3_width_mask) << SDMA_PKT_WRITE_TILED_BC_DW_3_width_shift)

 #/*define for DW_4 word*/
 #/*define for height field*/
SDMA_PKT_WRITE_TILED_BC_DW_4_height_offset = 4
SDMA_PKT_WRITE_TILED_BC_DW_4_height_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_BC_DW_4_height_shift = 0
def SDMA_PKT_WRITE_TILED_BC_DW_4_HEIGHT(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_4_height_mask) << SDMA_PKT_WRITE_TILED_BC_DW_4_height_shift)

 #/*define for depth field*/
SDMA_PKT_WRITE_TILED_BC_DW_4_depth_offset = 4
SDMA_PKT_WRITE_TILED_BC_DW_4_depth_mask = 0x000007FF
SDMA_PKT_WRITE_TILED_BC_DW_4_depth_shift = 16
def SDMA_PKT_WRITE_TILED_BC_DW_4_DEPTH(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_4_depth_mask) << SDMA_PKT_WRITE_TILED_BC_DW_4_depth_shift)

 #/*define for DW_5 word*/
 #/*define for element_size field*/
SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_mask = 0x00000007
SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_shift = 0
def SDMA_PKT_WRITE_TILED_BC_DW_5_ELEMENT_SIZE(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_shift)

 #/*define for array_mode field*/
SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_mask = 0x0000000F
SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_shift = 3
def SDMA_PKT_WRITE_TILED_BC_DW_5_ARRAY_MODE(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_shift)

 #/*define for mit_mode field*/
SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_mask = 0x00000007
SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_shift = 8
def SDMA_PKT_WRITE_TILED_BC_DW_5_MIT_MODE(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_shift)

 #/*define for tilesplit_size field*/
SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_mask = 0x00000007
SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_shift = 11
def SDMA_PKT_WRITE_TILED_BC_DW_5_TILESPLIT_SIZE(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_shift)

 #/*define for bank_w field*/
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_mask = 0x00000003
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_shift = 15
def SDMA_PKT_WRITE_TILED_BC_DW_5_BANK_W(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_shift)

 #/*define for bank_h field*/
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_mask = 0x00000003
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_shift = 18
def SDMA_PKT_WRITE_TILED_BC_DW_5_BANK_H(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_shift)

 #/*define for num_bank field*/
SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_mask = 0x00000003
SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_shift = 21
def SDMA_PKT_WRITE_TILED_BC_DW_5_NUM_BANK(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_shift)

 #/*define for mat_aspt field*/
SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_mask = 0x00000003
SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_shift = 24
def SDMA_PKT_WRITE_TILED_BC_DW_5_MAT_ASPT(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_shift)

 #/*define for pipe_config field*/
SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_mask = 0x0000001F
SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_shift = 26
def SDMA_PKT_WRITE_TILED_BC_DW_5_PIPE_CONFIG(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_shift)

 #/*define for DW_6 word*/
 #/*define for x field*/
SDMA_PKT_WRITE_TILED_BC_DW_6_x_offset = 6
SDMA_PKT_WRITE_TILED_BC_DW_6_x_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_BC_DW_6_x_shift = 0
def SDMA_PKT_WRITE_TILED_BC_DW_6_X(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_6_x_mask) << SDMA_PKT_WRITE_TILED_BC_DW_6_x_shift)

 #/*define for y field*/
SDMA_PKT_WRITE_TILED_BC_DW_6_y_offset = 6
SDMA_PKT_WRITE_TILED_BC_DW_6_y_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_BC_DW_6_y_shift = 16
def SDMA_PKT_WRITE_TILED_BC_DW_6_Y(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_6_y_mask) << SDMA_PKT_WRITE_TILED_BC_DW_6_y_shift)

 #/*define for DW_7 word*/
 #/*define for z field*/
SDMA_PKT_WRITE_TILED_BC_DW_7_z_offset = 7
SDMA_PKT_WRITE_TILED_BC_DW_7_z_mask = 0x000007FF
SDMA_PKT_WRITE_TILED_BC_DW_7_z_shift = 0
def SDMA_PKT_WRITE_TILED_BC_DW_7_Z(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_7_z_mask) << SDMA_PKT_WRITE_TILED_BC_DW_7_z_shift)

 #/*define for sw field*/
SDMA_PKT_WRITE_TILED_BC_DW_7_sw_offset = 7
SDMA_PKT_WRITE_TILED_BC_DW_7_sw_mask = 0x00000003
SDMA_PKT_WRITE_TILED_BC_DW_7_sw_shift = 24
def SDMA_PKT_WRITE_TILED_BC_DW_7_SW(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DW_7_sw_mask) << SDMA_PKT_WRITE_TILED_BC_DW_7_sw_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_WRITE_TILED_BC_COUNT_count_offset = 8
SDMA_PKT_WRITE_TILED_BC_COUNT_count_mask = 0x000FFFFF
SDMA_PKT_WRITE_TILED_BC_COUNT_count_shift = 2
def SDMA_PKT_WRITE_TILED_BC_COUNT_COUNT(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_COUNT_count_mask) << SDMA_PKT_WRITE_TILED_BC_COUNT_count_shift)

 #/*define for DATA0 word*/
 #/*define for data0 field*/
SDMA_PKT_WRITE_TILED_BC_DATA0_data0_offset = 9
SDMA_PKT_WRITE_TILED_BC_DATA0_data0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_TILED_BC_DATA0_data0_shift = 0
def SDMA_PKT_WRITE_TILED_BC_DATA0_DATA0(x): return (((x) & SDMA_PKT_WRITE_TILED_BC_DATA0_data0_mask) << SDMA_PKT_WRITE_TILED_BC_DATA0_data0_shift)


 #/*
#** Definitions for SDMA_PKT_PTEPDE_COPY packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_PTEPDE_COPY_HEADER_op_offset = 0
SDMA_PKT_PTEPDE_COPY_HEADER_op_mask = 0x000000FF
SDMA_PKT_PTEPDE_COPY_HEADER_op_shift = 0
def SDMA_PKT_PTEPDE_COPY_HEADER_OP(x): return (((x) & SDMA_PKT_PTEPDE_COPY_HEADER_op_mask) << SDMA_PKT_PTEPDE_COPY_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_offset = 0
SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_shift = 8
def SDMA_PKT_PTEPDE_COPY_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_mask) << SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_shift)

 #/*define for tmz field*/
SDMA_PKT_PTEPDE_COPY_HEADER_tmz_offset = 0
SDMA_PKT_PTEPDE_COPY_HEADER_tmz_mask = 0x00000001
SDMA_PKT_PTEPDE_COPY_HEADER_tmz_shift = 18
def SDMA_PKT_PTEPDE_COPY_HEADER_TMZ(x): return (((x) & SDMA_PKT_PTEPDE_COPY_HEADER_tmz_mask) << SDMA_PKT_PTEPDE_COPY_HEADER_tmz_shift)

 #/*define for cpv field*/
SDMA_PKT_PTEPDE_COPY_HEADER_cpv_offset = 0
SDMA_PKT_PTEPDE_COPY_HEADER_cpv_mask = 0x00000001
SDMA_PKT_PTEPDE_COPY_HEADER_cpv_shift = 28
def SDMA_PKT_PTEPDE_COPY_HEADER_CPV(x): return (((x) & SDMA_PKT_PTEPDE_COPY_HEADER_cpv_mask) << SDMA_PKT_PTEPDE_COPY_HEADER_cpv_shift)

 #/*define for ptepde_op field*/
SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_offset = 0
SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_mask = 0x00000001
SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_shift = 31
def SDMA_PKT_PTEPDE_COPY_HEADER_PTEPDE_OP(x): return (((x) & SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_mask) << SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_shift)

 #/*define for SRC_ADDR_LO word*/
 #/*define for src_addr_31_0 field*/
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_shift = 0
def SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_SRC_ADDR_31_0(x): return (((x) & SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_shift)

 #/*define for SRC_ADDR_HI word*/
 #/*define for src_addr_63_32 field*/
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_shift = 0
def SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_SRC_ADDR_63_32(x): return (((x) & SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_offset = 3
SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_offset = 4
SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for MASK_DW0 word*/
 #/*define for mask_dw0 field*/
SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_offset = 5
SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_shift = 0
def SDMA_PKT_PTEPDE_COPY_MASK_DW0_MASK_DW0(x): return (((x) & SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_mask) << SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_shift)

 #/*define for MASK_DW1 word*/
 #/*define for mask_dw1 field*/
SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_offset = 6
SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_shift = 0
def SDMA_PKT_PTEPDE_COPY_MASK_DW1_MASK_DW1(x): return (((x) & SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_mask) << SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_PTEPDE_COPY_COUNT_count_offset = 7
SDMA_PKT_PTEPDE_COPY_COUNT_count_mask = 0x0007FFFF
SDMA_PKT_PTEPDE_COPY_COUNT_count_shift = 0
def SDMA_PKT_PTEPDE_COPY_COUNT_COUNT(x): return (((x) & SDMA_PKT_PTEPDE_COPY_COUNT_count_mask) << SDMA_PKT_PTEPDE_COPY_COUNT_count_shift)

 #/*define for dst_cache_policy field*/
SDMA_PKT_PTEPDE_COPY_COUNT_dst_cache_policy_offset = 7
SDMA_PKT_PTEPDE_COPY_COUNT_dst_cache_policy_mask = 0x00000007
SDMA_PKT_PTEPDE_COPY_COUNT_dst_cache_policy_shift = 22
def SDMA_PKT_PTEPDE_COPY_COUNT_DST_CACHE_POLICY(x): return (((x) & SDMA_PKT_PTEPDE_COPY_COUNT_dst_cache_policy_mask) << SDMA_PKT_PTEPDE_COPY_COUNT_dst_cache_policy_shift)

 #/*define for src_cache_policy field*/
SDMA_PKT_PTEPDE_COPY_COUNT_src_cache_policy_offset = 7
SDMA_PKT_PTEPDE_COPY_COUNT_src_cache_policy_mask = 0x00000007
SDMA_PKT_PTEPDE_COPY_COUNT_src_cache_policy_shift = 29
def SDMA_PKT_PTEPDE_COPY_COUNT_SRC_CACHE_POLICY(x): return (((x) & SDMA_PKT_PTEPDE_COPY_COUNT_src_cache_policy_mask) << SDMA_PKT_PTEPDE_COPY_COUNT_src_cache_policy_shift)


 #/*
#** Definitions for SDMA_PKT_PTEPDE_COPY_BACKWARDS packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_offset = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_mask = 0x000000FF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_shift = 0
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_OP(x): return (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_offset = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_shift = 8
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_shift)

 #/*define for pte_size field*/
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_offset = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_mask = 0x00000003
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_shift = 28
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_PTE_SIZE(x): return (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_shift)

 #/*define for direction field*/
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_offset = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_mask = 0x00000001
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_shift = 30
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_DIRECTION(x): return (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_shift)

 #/*define for ptepde_op field*/
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_offset = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_mask = 0x00000001
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_shift = 31
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_PTEPDE_OP(x): return (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_shift)

 #/*define for SRC_ADDR_LO word*/
 #/*define for src_addr_31_0 field*/
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_shift = 0
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_SRC_ADDR_31_0(x): return (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_shift)

 #/*define for SRC_ADDR_HI word*/
 #/*define for src_addr_63_32 field*/
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_shift = 0
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_SRC_ADDR_63_32(x): return (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_offset = 3
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_offset = 4
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for MASK_BIT_FOR_DW word*/
 #/*define for mask_first_xfer field*/
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_offset = 5
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_mask = 0x000000FF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_shift = 0
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_MASK_FIRST_XFER(x): return (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_shift)

 #/*define for mask_last_xfer field*/
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_offset = 5
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_mask = 0x000000FF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_shift = 8
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_MASK_LAST_XFER(x): return (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_shift)

 #/*define for COUNT_IN_32B_XFER word*/
 #/*define for count field*/
SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_offset = 6
SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_mask = 0x0001FFFF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_shift = 0
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_COUNT(x): return (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_shift)


 #/*
#** Definitions for SDMA_PKT_PTEPDE_RMW packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_PTEPDE_RMW_HEADER_op_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_op_mask = 0x000000FF
SDMA_PKT_PTEPDE_RMW_HEADER_op_shift = 0
def SDMA_PKT_PTEPDE_RMW_HEADER_OP(x): return (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_op_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_shift = 8
def SDMA_PKT_PTEPDE_RMW_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_shift)

 #/*define for mtype field*/
SDMA_PKT_PTEPDE_RMW_HEADER_mtype_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_mtype_mask = 0x00000007
SDMA_PKT_PTEPDE_RMW_HEADER_mtype_shift = 16
def SDMA_PKT_PTEPDE_RMW_HEADER_MTYPE(x): return (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_mtype_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_mtype_shift)

 #/*define for gcc field*/
SDMA_PKT_PTEPDE_RMW_HEADER_gcc_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_gcc_mask = 0x00000001
SDMA_PKT_PTEPDE_RMW_HEADER_gcc_shift = 19
def SDMA_PKT_PTEPDE_RMW_HEADER_GCC(x): return (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_gcc_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_gcc_shift)

 #/*define for sys field*/
SDMA_PKT_PTEPDE_RMW_HEADER_sys_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_sys_mask = 0x00000001
SDMA_PKT_PTEPDE_RMW_HEADER_sys_shift = 20
def SDMA_PKT_PTEPDE_RMW_HEADER_SYS(x): return (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_sys_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_sys_shift)

 #/*define for snp field*/
SDMA_PKT_PTEPDE_RMW_HEADER_snp_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_snp_mask = 0x00000001
SDMA_PKT_PTEPDE_RMW_HEADER_snp_shift = 22
def SDMA_PKT_PTEPDE_RMW_HEADER_SNP(x): return (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_snp_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_snp_shift)

 #/*define for gpa field*/
SDMA_PKT_PTEPDE_RMW_HEADER_gpa_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_gpa_mask = 0x00000001
SDMA_PKT_PTEPDE_RMW_HEADER_gpa_shift = 23
def SDMA_PKT_PTEPDE_RMW_HEADER_GPA(x): return (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_gpa_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_gpa_shift)

 #/*define for l2_policy field*/
SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_mask = 0x00000003
SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_shift = 24
def SDMA_PKT_PTEPDE_RMW_HEADER_L2_POLICY(x): return (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_shift)

 #/*define for llc_policy field*/
SDMA_PKT_PTEPDE_RMW_HEADER_llc_policy_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_llc_policy_mask = 0x00000001
SDMA_PKT_PTEPDE_RMW_HEADER_llc_policy_shift = 26
def SDMA_PKT_PTEPDE_RMW_HEADER_LLC_POLICY(x): return (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_llc_policy_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_llc_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_PTEPDE_RMW_HEADER_cpv_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_cpv_mask = 0x00000001
SDMA_PKT_PTEPDE_RMW_HEADER_cpv_shift = 28
def SDMA_PKT_PTEPDE_RMW_HEADER_CPV(x): return (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_cpv_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_cpv_shift)

 #/*define for ADDR_LO word*/
 #/*define for addr_31_0 field*/
SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_shift = 0
def SDMA_PKT_PTEPDE_RMW_ADDR_LO_ADDR_31_0(x): return (((x) & SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_mask) << SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_shift)

 #/*define for ADDR_HI word*/
 #/*define for addr_63_32 field*/
SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_shift = 0
def SDMA_PKT_PTEPDE_RMW_ADDR_HI_ADDR_63_32(x): return (((x) & SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_mask) << SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_shift)

 #/*define for MASK_LO word*/
 #/*define for mask_31_0 field*/
SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_offset = 3
SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_shift = 0
def SDMA_PKT_PTEPDE_RMW_MASK_LO_MASK_31_0(x): return (((x) & SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_mask) << SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_shift)

 #/*define for MASK_HI word*/
 #/*define for mask_63_32 field*/
SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_offset = 4
SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_shift = 0
def SDMA_PKT_PTEPDE_RMW_MASK_HI_MASK_63_32(x): return (((x) & SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_mask) << SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_shift)

 #/*define for VALUE_LO word*/
 #/*define for value_31_0 field*/
SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_offset = 5
SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_shift = 0
def SDMA_PKT_PTEPDE_RMW_VALUE_LO_VALUE_31_0(x): return (((x) & SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_mask) << SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_shift)

 #/*define for VALUE_HI word*/
 #/*define for value_63_32 field*/
SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_offset = 6
SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_shift = 0
def SDMA_PKT_PTEPDE_RMW_VALUE_HI_VALUE_63_32(x): return (((x) & SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_mask) << SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_shift)

 #/*define for COUNT word*/
 #/*define for num_of_pte field*/
SDMA_PKT_PTEPDE_RMW_COUNT_num_of_pte_offset = 7
SDMA_PKT_PTEPDE_RMW_COUNT_num_of_pte_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_COUNT_num_of_pte_shift = 0
def SDMA_PKT_PTEPDE_RMW_COUNT_NUM_OF_PTE(x): return (((x) & SDMA_PKT_PTEPDE_RMW_COUNT_num_of_pte_mask) << SDMA_PKT_PTEPDE_RMW_COUNT_num_of_pte_shift)


 #/*
#** Definitions for SDMA_PKT_REGISTER_RMW packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_REGISTER_RMW_HEADER_op_offset = 0
SDMA_PKT_REGISTER_RMW_HEADER_op_mask = 0x000000FF
SDMA_PKT_REGISTER_RMW_HEADER_op_shift = 0
def SDMA_PKT_REGISTER_RMW_HEADER_OP(x): return (((x) & SDMA_PKT_REGISTER_RMW_HEADER_op_mask) << SDMA_PKT_REGISTER_RMW_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_REGISTER_RMW_HEADER_sub_op_offset = 0
SDMA_PKT_REGISTER_RMW_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_REGISTER_RMW_HEADER_sub_op_shift = 8
def SDMA_PKT_REGISTER_RMW_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_REGISTER_RMW_HEADER_sub_op_mask) << SDMA_PKT_REGISTER_RMW_HEADER_sub_op_shift)

 #/*define for ADDR word*/
 #/*define for addr field*/
SDMA_PKT_REGISTER_RMW_ADDR_addr_offset = 1
SDMA_PKT_REGISTER_RMW_ADDR_addr_mask = 0x000FFFFF
SDMA_PKT_REGISTER_RMW_ADDR_addr_shift = 0
def SDMA_PKT_REGISTER_RMW_ADDR_ADDR(x): return (((x) & SDMA_PKT_REGISTER_RMW_ADDR_addr_mask) << SDMA_PKT_REGISTER_RMW_ADDR_addr_shift)

 #/*define for aperture_id field*/
SDMA_PKT_REGISTER_RMW_ADDR_aperture_id_offset = 1
SDMA_PKT_REGISTER_RMW_ADDR_aperture_id_mask = 0x00000FFF
SDMA_PKT_REGISTER_RMW_ADDR_aperture_id_shift = 20
def SDMA_PKT_REGISTER_RMW_ADDR_APERTURE_ID(x): return (((x) & SDMA_PKT_REGISTER_RMW_ADDR_aperture_id_mask) << SDMA_PKT_REGISTER_RMW_ADDR_aperture_id_shift)

 #/*define for MASK word*/
 #/*define for mask field*/
SDMA_PKT_REGISTER_RMW_MASK_mask_offset = 2
SDMA_PKT_REGISTER_RMW_MASK_mask_mask = 0xFFFFFFFF
SDMA_PKT_REGISTER_RMW_MASK_mask_shift = 0
def SDMA_PKT_REGISTER_RMW_MASK_MASK(x): return (((x) & SDMA_PKT_REGISTER_RMW_MASK_mask_mask) << SDMA_PKT_REGISTER_RMW_MASK_mask_shift)

 #/*define for VALUE word*/
 #/*define for value field*/
SDMA_PKT_REGISTER_RMW_VALUE_value_offset = 3
SDMA_PKT_REGISTER_RMW_VALUE_value_mask = 0xFFFFFFFF
SDMA_PKT_REGISTER_RMW_VALUE_value_shift = 0
def SDMA_PKT_REGISTER_RMW_VALUE_VALUE(x): return (((x) & SDMA_PKT_REGISTER_RMW_VALUE_value_mask) << SDMA_PKT_REGISTER_RMW_VALUE_value_shift)

 #/*define for MISC word*/
 #/*define for stride field*/
SDMA_PKT_REGISTER_RMW_MISC_stride_offset = 4
SDMA_PKT_REGISTER_RMW_MISC_stride_mask = 0x000FFFFF
SDMA_PKT_REGISTER_RMW_MISC_stride_shift = 0
def SDMA_PKT_REGISTER_RMW_MISC_STRIDE(x): return (((x) & SDMA_PKT_REGISTER_RMW_MISC_stride_mask) << SDMA_PKT_REGISTER_RMW_MISC_stride_shift)

 #/*define for num_of_reg field*/
SDMA_PKT_REGISTER_RMW_MISC_num_of_reg_offset = 4
SDMA_PKT_REGISTER_RMW_MISC_num_of_reg_mask = 0x00000FFF
SDMA_PKT_REGISTER_RMW_MISC_num_of_reg_shift = 20
def SDMA_PKT_REGISTER_RMW_MISC_NUM_OF_REG(x): return (((x) & SDMA_PKT_REGISTER_RMW_MISC_num_of_reg_mask) << SDMA_PKT_REGISTER_RMW_MISC_num_of_reg_shift)


 #/*
#** Definitions for SDMA_PKT_WRITE_INCR packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_WRITE_INCR_HEADER_op_offset = 0
SDMA_PKT_WRITE_INCR_HEADER_op_mask = 0x000000FF
SDMA_PKT_WRITE_INCR_HEADER_op_shift = 0
def SDMA_PKT_WRITE_INCR_HEADER_OP(x): return (((x) & SDMA_PKT_WRITE_INCR_HEADER_op_mask) << SDMA_PKT_WRITE_INCR_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_WRITE_INCR_HEADER_sub_op_offset = 0
SDMA_PKT_WRITE_INCR_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_WRITE_INCR_HEADER_sub_op_shift = 8
def SDMA_PKT_WRITE_INCR_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_WRITE_INCR_HEADER_sub_op_mask) << SDMA_PKT_WRITE_INCR_HEADER_sub_op_shift)

 #/*define for cache_policy field*/
SDMA_PKT_WRITE_INCR_HEADER_cache_policy_offset = 0
SDMA_PKT_WRITE_INCR_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_WRITE_INCR_HEADER_cache_policy_shift = 24
def SDMA_PKT_WRITE_INCR_HEADER_CACHE_POLICY(x): return (((x) & SDMA_PKT_WRITE_INCR_HEADER_cache_policy_mask) << SDMA_PKT_WRITE_INCR_HEADER_cache_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_WRITE_INCR_HEADER_cpv_offset = 0
SDMA_PKT_WRITE_INCR_HEADER_cpv_mask = 0x00000001
SDMA_PKT_WRITE_INCR_HEADER_cpv_shift = 28
def SDMA_PKT_WRITE_INCR_HEADER_CPV(x): return (((x) & SDMA_PKT_WRITE_INCR_HEADER_cpv_mask) << SDMA_PKT_WRITE_INCR_HEADER_cpv_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_offset = 1
SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_WRITE_INCR_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_offset = 2
SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_WRITE_INCR_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for MASK_DW0 word*/
 #/*define for mask_dw0 field*/
SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_offset = 3
SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_shift = 0
def SDMA_PKT_WRITE_INCR_MASK_DW0_MASK_DW0(x): return (((x) & SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_mask) << SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_shift)

 #/*define for MASK_DW1 word*/
 #/*define for mask_dw1 field*/
SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_offset = 4
SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_shift = 0
def SDMA_PKT_WRITE_INCR_MASK_DW1_MASK_DW1(x): return (((x) & SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_mask) << SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_shift)

 #/*define for INIT_DW0 word*/
 #/*define for init_dw0 field*/
SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_offset = 5
SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_shift = 0
def SDMA_PKT_WRITE_INCR_INIT_DW0_INIT_DW0(x): return (((x) & SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_mask) << SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_shift)

 #/*define for INIT_DW1 word*/
 #/*define for init_dw1 field*/
SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_offset = 6
SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_shift = 0
def SDMA_PKT_WRITE_INCR_INIT_DW1_INIT_DW1(x): return (((x) & SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_mask) << SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_shift)

 #/*define for INCR_DW0 word*/
 #/*define for incr_dw0 field*/
SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_offset = 7
SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_shift = 0
def SDMA_PKT_WRITE_INCR_INCR_DW0_INCR_DW0(x): return (((x) & SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_mask) << SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_shift)

 #/*define for INCR_DW1 word*/
 #/*define for incr_dw1 field*/
SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_offset = 8
SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_shift = 0
def SDMA_PKT_WRITE_INCR_INCR_DW1_INCR_DW1(x): return (((x) & SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_mask) << SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_WRITE_INCR_COUNT_count_offset = 9
SDMA_PKT_WRITE_INCR_COUNT_count_mask = 0x0007FFFF
SDMA_PKT_WRITE_INCR_COUNT_count_shift = 0
def SDMA_PKT_WRITE_INCR_COUNT_COUNT(x): return (((x) & SDMA_PKT_WRITE_INCR_COUNT_count_mask) << SDMA_PKT_WRITE_INCR_COUNT_count_shift)


 #/*
#** Definitions for SDMA_PKT_INDIRECT packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_INDIRECT_HEADER_op_offset = 0
SDMA_PKT_INDIRECT_HEADER_op_mask = 0x000000FF
SDMA_PKT_INDIRECT_HEADER_op_shift = 0
def SDMA_PKT_INDIRECT_HEADER_OP(x): return (((x) & SDMA_PKT_INDIRECT_HEADER_op_mask) << SDMA_PKT_INDIRECT_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_INDIRECT_HEADER_sub_op_offset = 0
SDMA_PKT_INDIRECT_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_INDIRECT_HEADER_sub_op_shift = 8
def SDMA_PKT_INDIRECT_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_INDIRECT_HEADER_sub_op_mask) << SDMA_PKT_INDIRECT_HEADER_sub_op_shift)

 #/*define for vmid field*/
SDMA_PKT_INDIRECT_HEADER_vmid_offset = 0
SDMA_PKT_INDIRECT_HEADER_vmid_mask = 0x0000000F
SDMA_PKT_INDIRECT_HEADER_vmid_shift = 16
def SDMA_PKT_INDIRECT_HEADER_VMID(x): return (((x) & SDMA_PKT_INDIRECT_HEADER_vmid_mask) << SDMA_PKT_INDIRECT_HEADER_vmid_shift)

 #/*define for priv field*/
SDMA_PKT_INDIRECT_HEADER_priv_offset = 0
SDMA_PKT_INDIRECT_HEADER_priv_mask = 0x00000001
SDMA_PKT_INDIRECT_HEADER_priv_shift = 31
def SDMA_PKT_INDIRECT_HEADER_PRIV(x): return (((x) & SDMA_PKT_INDIRECT_HEADER_priv_mask) << SDMA_PKT_INDIRECT_HEADER_priv_shift)

 #/*define for BASE_LO word*/
 #/*define for ib_base_31_0 field*/
SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_offset = 1
SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_mask = 0xFFFFFFFF
SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_shift = 0
def SDMA_PKT_INDIRECT_BASE_LO_IB_BASE_31_0(x): return (((x) & SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_mask) << SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_shift)

 #/*define for BASE_HI word*/
 #/*define for ib_base_63_32 field*/
SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_offset = 2
SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_mask = 0xFFFFFFFF
SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_shift = 0
def SDMA_PKT_INDIRECT_BASE_HI_IB_BASE_63_32(x): return (((x) & SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_mask) << SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_shift)

 #/*define for IB_SIZE word*/
 #/*define for ib_size field*/
SDMA_PKT_INDIRECT_IB_SIZE_ib_size_offset = 3
SDMA_PKT_INDIRECT_IB_SIZE_ib_size_mask = 0x000FFFFF
SDMA_PKT_INDIRECT_IB_SIZE_ib_size_shift = 0
def SDMA_PKT_INDIRECT_IB_SIZE_IB_SIZE(x): return (((x) & SDMA_PKT_INDIRECT_IB_SIZE_ib_size_mask) << SDMA_PKT_INDIRECT_IB_SIZE_ib_size_shift)

 #/*define for CSA_ADDR_LO word*/
 #/*define for csa_addr_31_0 field*/
SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_offset = 4
SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_shift = 0
def SDMA_PKT_INDIRECT_CSA_ADDR_LO_CSA_ADDR_31_0(x): return (((x) & SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_mask) << SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_shift)

 #/*define for CSA_ADDR_HI word*/
 #/*define for csa_addr_63_32 field*/
SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_offset = 5
SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_shift = 0
def SDMA_PKT_INDIRECT_CSA_ADDR_HI_CSA_ADDR_63_32(x): return (((x) & SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_mask) << SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_shift)


 #/*
#** Definitions for SDMA_PKT_SEMAPHORE packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_SEMAPHORE_HEADER_op_offset = 0
SDMA_PKT_SEMAPHORE_HEADER_op_mask = 0x000000FF
SDMA_PKT_SEMAPHORE_HEADER_op_shift = 0
def SDMA_PKT_SEMAPHORE_HEADER_OP(x): return (((x) & SDMA_PKT_SEMAPHORE_HEADER_op_mask) << SDMA_PKT_SEMAPHORE_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_SEMAPHORE_HEADER_sub_op_offset = 0
SDMA_PKT_SEMAPHORE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_SEMAPHORE_HEADER_sub_op_shift = 8
def SDMA_PKT_SEMAPHORE_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_SEMAPHORE_HEADER_sub_op_mask) << SDMA_PKT_SEMAPHORE_HEADER_sub_op_shift)

 #/*define for write_one field*/
SDMA_PKT_SEMAPHORE_HEADER_write_one_offset = 0
SDMA_PKT_SEMAPHORE_HEADER_write_one_mask = 0x00000001
SDMA_PKT_SEMAPHORE_HEADER_write_one_shift = 29
def SDMA_PKT_SEMAPHORE_HEADER_WRITE_ONE(x): return (((x) & SDMA_PKT_SEMAPHORE_HEADER_write_one_mask) << SDMA_PKT_SEMAPHORE_HEADER_write_one_shift)

 #/*define for signal field*/
SDMA_PKT_SEMAPHORE_HEADER_signal_offset = 0
SDMA_PKT_SEMAPHORE_HEADER_signal_mask = 0x00000001
SDMA_PKT_SEMAPHORE_HEADER_signal_shift = 30
def SDMA_PKT_SEMAPHORE_HEADER_SIGNAL(x): return (((x) & SDMA_PKT_SEMAPHORE_HEADER_signal_mask) << SDMA_PKT_SEMAPHORE_HEADER_signal_shift)

 #/*define for mailbox field*/
SDMA_PKT_SEMAPHORE_HEADER_mailbox_offset = 0
SDMA_PKT_SEMAPHORE_HEADER_mailbox_mask = 0x00000001
SDMA_PKT_SEMAPHORE_HEADER_mailbox_shift = 31
def SDMA_PKT_SEMAPHORE_HEADER_MAILBOX(x): return (((x) & SDMA_PKT_SEMAPHORE_HEADER_mailbox_mask) << SDMA_PKT_SEMAPHORE_HEADER_mailbox_shift)

 #/*define for ADDR_LO word*/
 #/*define for addr_31_0 field*/
SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_shift = 0
def SDMA_PKT_SEMAPHORE_ADDR_LO_ADDR_31_0(x): return (((x) & SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_mask) << SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_shift)

 #/*define for ADDR_HI word*/
 #/*define for addr_63_32 field*/
SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_shift = 0
def SDMA_PKT_SEMAPHORE_ADDR_HI_ADDR_63_32(x): return (((x) & SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_mask) << SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_shift)


 #/*
#** Definitions for SDMA_PKT_MEM_INCR packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_MEM_INCR_HEADER_op_offset = 0
SDMA_PKT_MEM_INCR_HEADER_op_mask = 0x000000FF
SDMA_PKT_MEM_INCR_HEADER_op_shift = 0
def SDMA_PKT_MEM_INCR_HEADER_OP(x): return (((x) & SDMA_PKT_MEM_INCR_HEADER_op_mask) << SDMA_PKT_MEM_INCR_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_MEM_INCR_HEADER_sub_op_offset = 0
SDMA_PKT_MEM_INCR_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_MEM_INCR_HEADER_sub_op_shift = 8
def SDMA_PKT_MEM_INCR_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_MEM_INCR_HEADER_sub_op_mask) << SDMA_PKT_MEM_INCR_HEADER_sub_op_shift)

 #/*define for l2_policy field*/
SDMA_PKT_MEM_INCR_HEADER_l2_policy_offset = 0
SDMA_PKT_MEM_INCR_HEADER_l2_policy_mask = 0x00000003
SDMA_PKT_MEM_INCR_HEADER_l2_policy_shift = 24
def SDMA_PKT_MEM_INCR_HEADER_L2_POLICY(x): return (((x) & SDMA_PKT_MEM_INCR_HEADER_l2_policy_mask) << SDMA_PKT_MEM_INCR_HEADER_l2_policy_shift)

 #/*define for llc_policy field*/
SDMA_PKT_MEM_INCR_HEADER_llc_policy_offset = 0
SDMA_PKT_MEM_INCR_HEADER_llc_policy_mask = 0x00000001
SDMA_PKT_MEM_INCR_HEADER_llc_policy_shift = 26
def SDMA_PKT_MEM_INCR_HEADER_LLC_POLICY(x): return (((x) & SDMA_PKT_MEM_INCR_HEADER_llc_policy_mask) << SDMA_PKT_MEM_INCR_HEADER_llc_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_MEM_INCR_HEADER_cpv_offset = 0
SDMA_PKT_MEM_INCR_HEADER_cpv_mask = 0x00000001
SDMA_PKT_MEM_INCR_HEADER_cpv_shift = 28
def SDMA_PKT_MEM_INCR_HEADER_CPV(x): return (((x) & SDMA_PKT_MEM_INCR_HEADER_cpv_mask) << SDMA_PKT_MEM_INCR_HEADER_cpv_shift)

 #/*define for ADDR_LO word*/
 #/*define for addr_31_0 field*/
SDMA_PKT_MEM_INCR_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_MEM_INCR_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_MEM_INCR_ADDR_LO_addr_31_0_shift = 0
def SDMA_PKT_MEM_INCR_ADDR_LO_ADDR_31_0(x): return (((x) & SDMA_PKT_MEM_INCR_ADDR_LO_addr_31_0_mask) << SDMA_PKT_MEM_INCR_ADDR_LO_addr_31_0_shift)

 #/*define for ADDR_HI word*/
 #/*define for addr_63_32 field*/
SDMA_PKT_MEM_INCR_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_MEM_INCR_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_MEM_INCR_ADDR_HI_addr_63_32_shift = 0
def SDMA_PKT_MEM_INCR_ADDR_HI_ADDR_63_32(x): return (((x) & SDMA_PKT_MEM_INCR_ADDR_HI_addr_63_32_mask) << SDMA_PKT_MEM_INCR_ADDR_HI_addr_63_32_shift)


 #/*
#** Definitions for SDMA_PKT_VM_INVALIDATION packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_VM_INVALIDATION_HEADER_op_offset = 0
SDMA_PKT_VM_INVALIDATION_HEADER_op_mask = 0x000000FF
SDMA_PKT_VM_INVALIDATION_HEADER_op_shift = 0
def SDMA_PKT_VM_INVALIDATION_HEADER_OP(x): return (((x) & SDMA_PKT_VM_INVALIDATION_HEADER_op_mask) << SDMA_PKT_VM_INVALIDATION_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_offset = 0
SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_shift = 8
def SDMA_PKT_VM_INVALIDATION_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_mask) << SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_shift)

 #/*define for gfx_eng_id field*/
SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_offset = 0
SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_mask = 0x0000001F
SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_shift = 16
def SDMA_PKT_VM_INVALIDATION_HEADER_GFX_ENG_ID(x): return (((x) & SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_mask) << SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_shift)

 #/*define for mm_eng_id field*/
SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_offset = 0
SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_mask = 0x0000001F
SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_shift = 24
def SDMA_PKT_VM_INVALIDATION_HEADER_MM_ENG_ID(x): return (((x) & SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_mask) << SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_shift)

 #/*define for INVALIDATEREQ word*/
 #/*define for invalidatereq field*/
SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_offset = 1
SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_mask = 0xFFFFFFFF
SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_shift = 0
def SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_INVALIDATEREQ(x): return (((x) & SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_mask) << SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_shift)

 #/*define for ADDRESSRANGELO word*/
 #/*define for addressrangelo field*/
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_offset = 2
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_mask = 0xFFFFFFFF
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_shift = 0
def SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_ADDRESSRANGELO(x): return (((x) & SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_mask) << SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_shift)

 #/*define for ADDRESSRANGEHI word*/
 #/*define for invalidateack field*/
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_offset = 3
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_mask = 0x0000FFFF
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_shift = 0
def SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_INVALIDATEACK(x): return (((x) & SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_mask) << SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_shift)

 #/*define for addressrangehi field*/
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_offset = 3
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_mask = 0x0000001F
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_shift = 16
def SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_ADDRESSRANGEHI(x): return (((x) & SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_mask) << SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_shift)

 #/*define for reserved field*/
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_offset = 3
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_mask = 0x000001FF
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_shift = 23
def SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_RESERVED(x): return (((x) & SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_mask) << SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_shift)


 #/*
#** Definitions for SDMA_PKT_FENCE packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_FENCE_HEADER_op_offset = 0
SDMA_PKT_FENCE_HEADER_op_mask = 0x000000FF
SDMA_PKT_FENCE_HEADER_op_shift = 0
def SDMA_PKT_FENCE_HEADER_OP(x): return (((x) & SDMA_PKT_FENCE_HEADER_op_mask) << SDMA_PKT_FENCE_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_FENCE_HEADER_sub_op_offset = 0
SDMA_PKT_FENCE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_FENCE_HEADER_sub_op_shift = 8
def SDMA_PKT_FENCE_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_FENCE_HEADER_sub_op_mask) << SDMA_PKT_FENCE_HEADER_sub_op_shift)

 #/*define for mtype field*/
SDMA_PKT_FENCE_HEADER_mtype_offset = 0
SDMA_PKT_FENCE_HEADER_mtype_mask = 0x00000007
SDMA_PKT_FENCE_HEADER_mtype_shift = 16
def SDMA_PKT_FENCE_HEADER_MTYPE(x): return (((x) & SDMA_PKT_FENCE_HEADER_mtype_mask) << SDMA_PKT_FENCE_HEADER_mtype_shift)

 #/*define for gcc field*/
SDMA_PKT_FENCE_HEADER_gcc_offset = 0
SDMA_PKT_FENCE_HEADER_gcc_mask = 0x00000001
SDMA_PKT_FENCE_HEADER_gcc_shift = 19
def SDMA_PKT_FENCE_HEADER_GCC(x): return (((x) & SDMA_PKT_FENCE_HEADER_gcc_mask) << SDMA_PKT_FENCE_HEADER_gcc_shift)

 #/*define for sys field*/
SDMA_PKT_FENCE_HEADER_sys_offset = 0
SDMA_PKT_FENCE_HEADER_sys_mask = 0x00000001
SDMA_PKT_FENCE_HEADER_sys_shift = 20
def SDMA_PKT_FENCE_HEADER_SYS(x): return (((x) & SDMA_PKT_FENCE_HEADER_sys_mask) << SDMA_PKT_FENCE_HEADER_sys_shift)

 #/*define for snp field*/
SDMA_PKT_FENCE_HEADER_snp_offset = 0
SDMA_PKT_FENCE_HEADER_snp_mask = 0x00000001
SDMA_PKT_FENCE_HEADER_snp_shift = 22
def SDMA_PKT_FENCE_HEADER_SNP(x): return (((x) & SDMA_PKT_FENCE_HEADER_snp_mask) << SDMA_PKT_FENCE_HEADER_snp_shift)

 #/*define for gpa field*/
SDMA_PKT_FENCE_HEADER_gpa_offset = 0
SDMA_PKT_FENCE_HEADER_gpa_mask = 0x00000001
SDMA_PKT_FENCE_HEADER_gpa_shift = 23
def SDMA_PKT_FENCE_HEADER_GPA(x): return (((x) & SDMA_PKT_FENCE_HEADER_gpa_mask) << SDMA_PKT_FENCE_HEADER_gpa_shift)

 #/*define for l2_policy field*/
SDMA_PKT_FENCE_HEADER_l2_policy_offset = 0
SDMA_PKT_FENCE_HEADER_l2_policy_mask = 0x00000003
SDMA_PKT_FENCE_HEADER_l2_policy_shift = 24
def SDMA_PKT_FENCE_HEADER_L2_POLICY(x): return (((x) & SDMA_PKT_FENCE_HEADER_l2_policy_mask) << SDMA_PKT_FENCE_HEADER_l2_policy_shift)

 #/*define for llc_policy field*/
SDMA_PKT_FENCE_HEADER_llc_policy_offset = 0
SDMA_PKT_FENCE_HEADER_llc_policy_mask = 0x00000001
SDMA_PKT_FENCE_HEADER_llc_policy_shift = 26
def SDMA_PKT_FENCE_HEADER_LLC_POLICY(x): return (((x) & SDMA_PKT_FENCE_HEADER_llc_policy_mask) << SDMA_PKT_FENCE_HEADER_llc_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_FENCE_HEADER_cpv_offset = 0
SDMA_PKT_FENCE_HEADER_cpv_mask = 0x00000001
SDMA_PKT_FENCE_HEADER_cpv_shift = 28
def SDMA_PKT_FENCE_HEADER_CPV(x): return (((x) & SDMA_PKT_FENCE_HEADER_cpv_mask) << SDMA_PKT_FENCE_HEADER_cpv_shift)

 #/*define for ADDR_LO word*/
 #/*define for addr_31_0 field*/
SDMA_PKT_FENCE_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_FENCE_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_FENCE_ADDR_LO_addr_31_0_shift = 0
def SDMA_PKT_FENCE_ADDR_LO_ADDR_31_0(x): return (((x) & SDMA_PKT_FENCE_ADDR_LO_addr_31_0_mask) << SDMA_PKT_FENCE_ADDR_LO_addr_31_0_shift)

 #/*define for ADDR_HI word*/
 #/*define for addr_63_32 field*/
SDMA_PKT_FENCE_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_FENCE_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_FENCE_ADDR_HI_addr_63_32_shift = 0
def SDMA_PKT_FENCE_ADDR_HI_ADDR_63_32(x): return (((x) & SDMA_PKT_FENCE_ADDR_HI_addr_63_32_mask) << SDMA_PKT_FENCE_ADDR_HI_addr_63_32_shift)

 #/*define for DATA word*/
 #/*define for data field*/
SDMA_PKT_FENCE_DATA_data_offset = 3
SDMA_PKT_FENCE_DATA_data_mask = 0xFFFFFFFF
SDMA_PKT_FENCE_DATA_data_shift = 0
def SDMA_PKT_FENCE_DATA_DATA(x): return (((x) & SDMA_PKT_FENCE_DATA_data_mask) << SDMA_PKT_FENCE_DATA_data_shift)


 #/*
#** Definitions for SDMA_PKT_SRBM_WRITE packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_SRBM_WRITE_HEADER_op_offset = 0
SDMA_PKT_SRBM_WRITE_HEADER_op_mask = 0x000000FF
SDMA_PKT_SRBM_WRITE_HEADER_op_shift = 0
def SDMA_PKT_SRBM_WRITE_HEADER_OP(x): return (((x) & SDMA_PKT_SRBM_WRITE_HEADER_op_mask) << SDMA_PKT_SRBM_WRITE_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_SRBM_WRITE_HEADER_sub_op_offset = 0
SDMA_PKT_SRBM_WRITE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_SRBM_WRITE_HEADER_sub_op_shift = 8
def SDMA_PKT_SRBM_WRITE_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_SRBM_WRITE_HEADER_sub_op_mask) << SDMA_PKT_SRBM_WRITE_HEADER_sub_op_shift)

 #/*define for byte_en field*/
SDMA_PKT_SRBM_WRITE_HEADER_byte_en_offset = 0
SDMA_PKT_SRBM_WRITE_HEADER_byte_en_mask = 0x0000000F
SDMA_PKT_SRBM_WRITE_HEADER_byte_en_shift = 28
def SDMA_PKT_SRBM_WRITE_HEADER_BYTE_EN(x): return (((x) & SDMA_PKT_SRBM_WRITE_HEADER_byte_en_mask) << SDMA_PKT_SRBM_WRITE_HEADER_byte_en_shift)

 #/*define for ADDR word*/
 #/*define for addr field*/
SDMA_PKT_SRBM_WRITE_ADDR_addr_offset = 1
SDMA_PKT_SRBM_WRITE_ADDR_addr_mask = 0x0003FFFF
SDMA_PKT_SRBM_WRITE_ADDR_addr_shift = 0
def SDMA_PKT_SRBM_WRITE_ADDR_ADDR(x): return (((x) & SDMA_PKT_SRBM_WRITE_ADDR_addr_mask) << SDMA_PKT_SRBM_WRITE_ADDR_addr_shift)

 #/*define for apertureid field*/
SDMA_PKT_SRBM_WRITE_ADDR_apertureid_offset = 1
SDMA_PKT_SRBM_WRITE_ADDR_apertureid_mask = 0x00000FFF
SDMA_PKT_SRBM_WRITE_ADDR_apertureid_shift = 20
def SDMA_PKT_SRBM_WRITE_ADDR_APERTUREID(x): return (((x) & SDMA_PKT_SRBM_WRITE_ADDR_apertureid_mask) << SDMA_PKT_SRBM_WRITE_ADDR_apertureid_shift)

 #/*define for DATA word*/
 #/*define for data field*/
SDMA_PKT_SRBM_WRITE_DATA_data_offset = 2
SDMA_PKT_SRBM_WRITE_DATA_data_mask = 0xFFFFFFFF
SDMA_PKT_SRBM_WRITE_DATA_data_shift = 0
def SDMA_PKT_SRBM_WRITE_DATA_DATA(x): return (((x) & SDMA_PKT_SRBM_WRITE_DATA_data_mask) << SDMA_PKT_SRBM_WRITE_DATA_data_shift)


 #/*
#** Definitions for SDMA_PKT_PRE_EXE packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_PRE_EXE_HEADER_op_offset = 0
SDMA_PKT_PRE_EXE_HEADER_op_mask = 0x000000FF
SDMA_PKT_PRE_EXE_HEADER_op_shift = 0
def SDMA_PKT_PRE_EXE_HEADER_OP(x): return (((x) & SDMA_PKT_PRE_EXE_HEADER_op_mask) << SDMA_PKT_PRE_EXE_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_PRE_EXE_HEADER_sub_op_offset = 0
SDMA_PKT_PRE_EXE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_PRE_EXE_HEADER_sub_op_shift = 8
def SDMA_PKT_PRE_EXE_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_PRE_EXE_HEADER_sub_op_mask) << SDMA_PKT_PRE_EXE_HEADER_sub_op_shift)

 #/*define for dev_sel field*/
SDMA_PKT_PRE_EXE_HEADER_dev_sel_offset = 0
SDMA_PKT_PRE_EXE_HEADER_dev_sel_mask = 0x000000FF
SDMA_PKT_PRE_EXE_HEADER_dev_sel_shift = 16
def SDMA_PKT_PRE_EXE_HEADER_DEV_SEL(x): return (((x) & SDMA_PKT_PRE_EXE_HEADER_dev_sel_mask) << SDMA_PKT_PRE_EXE_HEADER_dev_sel_shift)

 #/*define for EXEC_COUNT word*/
 #/*define for exec_count field*/
SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_offset = 1
SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_mask = 0x00003FFF
SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_shift = 0
def SDMA_PKT_PRE_EXE_EXEC_COUNT_EXEC_COUNT(x): return (((x) & SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_mask) << SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_shift)


 #/*
#** Definitions for SDMA_PKT_COND_EXE packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_COND_EXE_HEADER_op_offset = 0
SDMA_PKT_COND_EXE_HEADER_op_mask = 0x000000FF
SDMA_PKT_COND_EXE_HEADER_op_shift = 0
def SDMA_PKT_COND_EXE_HEADER_OP(x): return (((x) & SDMA_PKT_COND_EXE_HEADER_op_mask) << SDMA_PKT_COND_EXE_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_COND_EXE_HEADER_sub_op_offset = 0
SDMA_PKT_COND_EXE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COND_EXE_HEADER_sub_op_shift = 8
def SDMA_PKT_COND_EXE_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_COND_EXE_HEADER_sub_op_mask) << SDMA_PKT_COND_EXE_HEADER_sub_op_shift)

 #/*define for cache_policy field*/
SDMA_PKT_COND_EXE_HEADER_cache_policy_offset = 0
SDMA_PKT_COND_EXE_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_COND_EXE_HEADER_cache_policy_shift = 24
def SDMA_PKT_COND_EXE_HEADER_CACHE_POLICY(x): return (((x) & SDMA_PKT_COND_EXE_HEADER_cache_policy_mask) << SDMA_PKT_COND_EXE_HEADER_cache_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_COND_EXE_HEADER_cpv_offset = 0
SDMA_PKT_COND_EXE_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COND_EXE_HEADER_cpv_shift = 28
def SDMA_PKT_COND_EXE_HEADER_CPV(x): return (((x) & SDMA_PKT_COND_EXE_HEADER_cpv_mask) << SDMA_PKT_COND_EXE_HEADER_cpv_shift)

 #/*define for ADDR_LO word*/
 #/*define for addr_31_0 field*/
SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_shift = 0
def SDMA_PKT_COND_EXE_ADDR_LO_ADDR_31_0(x): return (((x) & SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_mask) << SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_shift)

 #/*define for ADDR_HI word*/
 #/*define for addr_63_32 field*/
SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_shift = 0
def SDMA_PKT_COND_EXE_ADDR_HI_ADDR_63_32(x): return (((x) & SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_mask) << SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_shift)

 #/*define for REFERENCE word*/
 #/*define for reference field*/
SDMA_PKT_COND_EXE_REFERENCE_reference_offset = 3
SDMA_PKT_COND_EXE_REFERENCE_reference_mask = 0xFFFFFFFF
SDMA_PKT_COND_EXE_REFERENCE_reference_shift = 0
def SDMA_PKT_COND_EXE_REFERENCE_REFERENCE(x): return (((x) & SDMA_PKT_COND_EXE_REFERENCE_reference_mask) << SDMA_PKT_COND_EXE_REFERENCE_reference_shift)

 #/*define for EXEC_COUNT word*/
 #/*define for exec_count field*/
SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_offset = 4
SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_mask = 0x00003FFF
SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_shift = 0
def SDMA_PKT_COND_EXE_EXEC_COUNT_EXEC_COUNT(x): return (((x) & SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_mask) << SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_shift)


 #/*
#** Definitions for SDMA_PKT_CONSTANT_FILL packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_CONSTANT_FILL_HEADER_op_offset = 0
SDMA_PKT_CONSTANT_FILL_HEADER_op_mask = 0x000000FF
SDMA_PKT_CONSTANT_FILL_HEADER_op_shift = 0
def SDMA_PKT_CONSTANT_FILL_HEADER_OP(x): return (((x) & SDMA_PKT_CONSTANT_FILL_HEADER_op_mask) << SDMA_PKT_CONSTANT_FILL_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_offset = 0
SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_shift = 8
def SDMA_PKT_CONSTANT_FILL_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_mask) << SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_shift)

 #/*define for sw field*/
SDMA_PKT_CONSTANT_FILL_HEADER_sw_offset = 0
SDMA_PKT_CONSTANT_FILL_HEADER_sw_mask = 0x00000003
SDMA_PKT_CONSTANT_FILL_HEADER_sw_shift = 16
def SDMA_PKT_CONSTANT_FILL_HEADER_SW(x): return (((x) & SDMA_PKT_CONSTANT_FILL_HEADER_sw_mask) << SDMA_PKT_CONSTANT_FILL_HEADER_sw_shift)

 #/*define for cache_policy field*/
SDMA_PKT_CONSTANT_FILL_HEADER_cache_policy_offset = 0
SDMA_PKT_CONSTANT_FILL_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_CONSTANT_FILL_HEADER_cache_policy_shift = 24
def SDMA_PKT_CONSTANT_FILL_HEADER_CACHE_POLICY(x): return (((x) & SDMA_PKT_CONSTANT_FILL_HEADER_cache_policy_mask) << SDMA_PKT_CONSTANT_FILL_HEADER_cache_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_CONSTANT_FILL_HEADER_cpv_offset = 0
SDMA_PKT_CONSTANT_FILL_HEADER_cpv_mask = 0x00000001
SDMA_PKT_CONSTANT_FILL_HEADER_cpv_shift = 28
def SDMA_PKT_CONSTANT_FILL_HEADER_CPV(x): return (((x) & SDMA_PKT_CONSTANT_FILL_HEADER_cpv_mask) << SDMA_PKT_CONSTANT_FILL_HEADER_cpv_shift)

 #/*define for fillsize field*/
SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_offset = 0
SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_mask = 0x00000003
SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_shift = 30
def SDMA_PKT_CONSTANT_FILL_HEADER_FILLSIZE(x): return (((x) & SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_mask) << SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_offset = 1
SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_offset = 2
SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for DATA word*/
 #/*define for src_data_31_0 field*/
SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_offset = 3
SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_mask = 0xFFFFFFFF
SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_shift = 0
def SDMA_PKT_CONSTANT_FILL_DATA_SRC_DATA_31_0(x): return (((x) & SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_mask) << SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_PKT_CONSTANT_FILL_COUNT_count_offset = 4
SDMA_PKT_CONSTANT_FILL_COUNT_count_mask = 0x3FFFFFFF
SDMA_PKT_CONSTANT_FILL_COUNT_count_shift = 0
def SDMA_PKT_CONSTANT_FILL_COUNT_COUNT(x): return (((x) & SDMA_PKT_CONSTANT_FILL_COUNT_count_mask) << SDMA_PKT_CONSTANT_FILL_COUNT_count_shift)


 #/*
#** Definitions for SDMA_PKT_DATA_FILL_MULTI packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_DATA_FILL_MULTI_HEADER_op_offset = 0
SDMA_PKT_DATA_FILL_MULTI_HEADER_op_mask = 0x000000FF
SDMA_PKT_DATA_FILL_MULTI_HEADER_op_shift = 0
def SDMA_PKT_DATA_FILL_MULTI_HEADER_OP(x): return (((x) & SDMA_PKT_DATA_FILL_MULTI_HEADER_op_mask) << SDMA_PKT_DATA_FILL_MULTI_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_offset = 0
SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_shift = 8
def SDMA_PKT_DATA_FILL_MULTI_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_mask) << SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_shift)

 #/*define for cache_policy field*/
SDMA_PKT_DATA_FILL_MULTI_HEADER_cache_policy_offset = 0
SDMA_PKT_DATA_FILL_MULTI_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_DATA_FILL_MULTI_HEADER_cache_policy_shift = 24
def SDMA_PKT_DATA_FILL_MULTI_HEADER_CACHE_POLICY(x): return (((x) & SDMA_PKT_DATA_FILL_MULTI_HEADER_cache_policy_mask) << SDMA_PKT_DATA_FILL_MULTI_HEADER_cache_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_DATA_FILL_MULTI_HEADER_cpv_offset = 0
SDMA_PKT_DATA_FILL_MULTI_HEADER_cpv_mask = 0x00000001
SDMA_PKT_DATA_FILL_MULTI_HEADER_cpv_shift = 28
def SDMA_PKT_DATA_FILL_MULTI_HEADER_CPV(x): return (((x) & SDMA_PKT_DATA_FILL_MULTI_HEADER_cpv_mask) << SDMA_PKT_DATA_FILL_MULTI_HEADER_cpv_shift)

 #/*define for memlog_clr field*/
SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_offset = 0
SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_mask = 0x00000001
SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_shift = 31
def SDMA_PKT_DATA_FILL_MULTI_HEADER_MEMLOG_CLR(x): return (((x) & SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_mask) << SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_shift)

 #/*define for BYTE_STRIDE word*/
 #/*define for byte_stride field*/
SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_offset = 1
SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_mask = 0xFFFFFFFF
SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_shift = 0
def SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_BYTE_STRIDE(x): return (((x) & SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_mask) << SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_shift)

 #/*define for DMA_COUNT word*/
 #/*define for dma_count field*/
SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_offset = 2
SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_mask = 0xFFFFFFFF
SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_shift = 0
def SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_DMA_COUNT(x): return (((x) & SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_mask) << SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_offset = 3
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_offset = 4
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for BYTE_COUNT word*/
 #/*define for count field*/
SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_offset = 5
SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_mask = 0x03FFFFFF
SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_shift = 0
def SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_COUNT(x): return (((x) & SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_mask) << SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_shift)


 #/*
#** Definitions for SDMA_PKT_POLL_REGMEM packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_POLL_REGMEM_HEADER_op_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_op_mask = 0x000000FF
SDMA_PKT_POLL_REGMEM_HEADER_op_shift = 0
def SDMA_PKT_POLL_REGMEM_HEADER_OP(x): return (((x) & SDMA_PKT_POLL_REGMEM_HEADER_op_mask) << SDMA_PKT_POLL_REGMEM_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_POLL_REGMEM_HEADER_sub_op_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_POLL_REGMEM_HEADER_sub_op_shift = 8
def SDMA_PKT_POLL_REGMEM_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_POLL_REGMEM_HEADER_sub_op_mask) << SDMA_PKT_POLL_REGMEM_HEADER_sub_op_shift)

 #/*define for cache_policy field*/
SDMA_PKT_POLL_REGMEM_HEADER_cache_policy_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_POLL_REGMEM_HEADER_cache_policy_shift = 20
def SDMA_PKT_POLL_REGMEM_HEADER_CACHE_POLICY(x): return (((x) & SDMA_PKT_POLL_REGMEM_HEADER_cache_policy_mask) << SDMA_PKT_POLL_REGMEM_HEADER_cache_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_POLL_REGMEM_HEADER_cpv_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_cpv_mask = 0x00000001
SDMA_PKT_POLL_REGMEM_HEADER_cpv_shift = 24
def SDMA_PKT_POLL_REGMEM_HEADER_CPV(x): return (((x) & SDMA_PKT_POLL_REGMEM_HEADER_cpv_mask) << SDMA_PKT_POLL_REGMEM_HEADER_cpv_shift)

 #/*define for hdp_flush field*/
SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_mask = 0x00000001
SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_shift = 26
def SDMA_PKT_POLL_REGMEM_HEADER_HDP_FLUSH(x): return (((x) & SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_mask) << SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_shift)

 #/*define for func field*/
SDMA_PKT_POLL_REGMEM_HEADER_func_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_func_mask = 0x00000007
SDMA_PKT_POLL_REGMEM_HEADER_func_shift = 28
def SDMA_PKT_POLL_REGMEM_HEADER_FUNC(x): return (((x) & SDMA_PKT_POLL_REGMEM_HEADER_func_mask) << SDMA_PKT_POLL_REGMEM_HEADER_func_shift)

 #/*define for mem_poll field*/
SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_mask = 0x00000001
SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_shift = 31
def SDMA_PKT_POLL_REGMEM_HEADER_MEM_POLL(x): return (((x) & SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_mask) << SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_shift)

 #/*define for ADDR_LO word*/
 #/*define for addr_31_0 field*/
SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_shift = 0
def SDMA_PKT_POLL_REGMEM_ADDR_LO_ADDR_31_0(x): return (((x) & SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_mask) << SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_shift)

 #/*define for ADDR_HI word*/
 #/*define for addr_63_32 field*/
SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_shift = 0
def SDMA_PKT_POLL_REGMEM_ADDR_HI_ADDR_63_32(x): return (((x) & SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_mask) << SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_shift)

 #/*define for VALUE word*/
 #/*define for value field*/
SDMA_PKT_POLL_REGMEM_VALUE_value_offset = 3
SDMA_PKT_POLL_REGMEM_VALUE_value_mask = 0xFFFFFFFF
SDMA_PKT_POLL_REGMEM_VALUE_value_shift = 0
def SDMA_PKT_POLL_REGMEM_VALUE_VALUE(x): return (((x) & SDMA_PKT_POLL_REGMEM_VALUE_value_mask) << SDMA_PKT_POLL_REGMEM_VALUE_value_shift)

 #/*define for MASK word*/
 #/*define for mask field*/
SDMA_PKT_POLL_REGMEM_MASK_mask_offset = 4
SDMA_PKT_POLL_REGMEM_MASK_mask_mask = 0xFFFFFFFF
SDMA_PKT_POLL_REGMEM_MASK_mask_shift = 0
def SDMA_PKT_POLL_REGMEM_MASK_MASK(x): return (((x) & SDMA_PKT_POLL_REGMEM_MASK_mask_mask) << SDMA_PKT_POLL_REGMEM_MASK_mask_shift)

 #/*define for DW5 word*/
 #/*define for interval field*/
SDMA_PKT_POLL_REGMEM_DW5_interval_offset = 5
SDMA_PKT_POLL_REGMEM_DW5_interval_mask = 0x0000FFFF
SDMA_PKT_POLL_REGMEM_DW5_interval_shift = 0
def SDMA_PKT_POLL_REGMEM_DW5_INTERVAL(x): return (((x) & SDMA_PKT_POLL_REGMEM_DW5_interval_mask) << SDMA_PKT_POLL_REGMEM_DW5_interval_shift)

 #/*define for retry_count field*/
SDMA_PKT_POLL_REGMEM_DW5_retry_count_offset = 5
SDMA_PKT_POLL_REGMEM_DW5_retry_count_mask = 0x00000FFF
SDMA_PKT_POLL_REGMEM_DW5_retry_count_shift = 16
def SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT(x): return (((x) & SDMA_PKT_POLL_REGMEM_DW5_retry_count_mask) << SDMA_PKT_POLL_REGMEM_DW5_retry_count_shift)


 #/*
#** Definitions for SDMA_PKT_POLL_REG_WRITE_MEM packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_offset = 0
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_mask = 0x000000FF
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_shift = 0
def SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_OP(x): return (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_offset = 0
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_shift = 8
def SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_shift)

 #/*define for cache_policy field*/
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cache_policy_offset = 0
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cache_policy_shift = 24
def SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_CACHE_POLICY(x): return (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cache_policy_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cache_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cpv_offset = 0
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cpv_mask = 0x00000001
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cpv_shift = 28
def SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_CPV(x): return (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cpv_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cpv_shift)

 #/*define for SRC_ADDR word*/
 #/*define for addr_31_2 field*/
SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_offset = 1
SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_mask = 0x3FFFFFFF
SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_shift = 2
def SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_ADDR_31_2(x): return (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for addr_31_0 field*/
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_offset = 2
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_shift = 0
def SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_ADDR_31_0(x): return (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for addr_63_32 field*/
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_offset = 3
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_shift = 0
def SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_ADDR_63_32(x): return (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_shift)


 #/*
#** Definitions for SDMA_PKT_POLL_DBIT_WRITE_MEM packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_offset = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_mask = 0x000000FF
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_shift = 0
def SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_OP(x): return (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_offset = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_shift = 8
def SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_shift)

 #/*define for ea field*/
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_offset = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_mask = 0x00000003
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_shift = 16
def SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_EA(x): return (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_shift)

 #/*define for cache_policy field*/
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cache_policy_offset = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cache_policy_shift = 24
def SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_CACHE_POLICY(x): return (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cache_policy_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cache_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cpv_offset = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cpv_mask = 0x00000001
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cpv_shift = 28
def SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_CPV(x): return (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cpv_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cpv_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for addr_31_0 field*/
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_shift = 0
def SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_ADDR_31_0(x): return (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for addr_63_32 field*/
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_shift = 0
def SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_ADDR_63_32(x): return (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_shift)

 #/*define for START_PAGE word*/
 #/*define for addr_31_4 field*/
SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_offset = 3
SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_mask = 0x0FFFFFFF
SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_shift = 4
def SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_ADDR_31_4(x): return (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_shift)

 #/*define for PAGE_NUM word*/
 #/*define for page_num_31_0 field*/
SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_offset = 4
SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_shift = 0
def SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_PAGE_NUM_31_0(x): return (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_shift)


 #/*
#** Definitions for SDMA_PKT_POLL_MEM_VERIFY packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_offset = 0
SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_mask = 0x000000FF
SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_shift = 0
def SDMA_PKT_POLL_MEM_VERIFY_HEADER_OP(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_mask) << SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_offset = 0
SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_shift = 8
def SDMA_PKT_POLL_MEM_VERIFY_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_mask) << SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_shift)

 #/*define for cache_policy field*/
SDMA_PKT_POLL_MEM_VERIFY_HEADER_cache_policy_offset = 0
SDMA_PKT_POLL_MEM_VERIFY_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_POLL_MEM_VERIFY_HEADER_cache_policy_shift = 24
def SDMA_PKT_POLL_MEM_VERIFY_HEADER_CACHE_POLICY(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_HEADER_cache_policy_mask) << SDMA_PKT_POLL_MEM_VERIFY_HEADER_cache_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_POLL_MEM_VERIFY_HEADER_cpv_offset = 0
SDMA_PKT_POLL_MEM_VERIFY_HEADER_cpv_mask = 0x00000001
SDMA_PKT_POLL_MEM_VERIFY_HEADER_cpv_shift = 28
def SDMA_PKT_POLL_MEM_VERIFY_HEADER_CPV(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_HEADER_cpv_mask) << SDMA_PKT_POLL_MEM_VERIFY_HEADER_cpv_shift)

 #/*define for mode field*/
SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_offset = 0
SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_mask = 0x00000001
SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_shift = 31
def SDMA_PKT_POLL_MEM_VERIFY_HEADER_MODE(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_mask) << SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_shift)

 #/*define for PATTERN word*/
 #/*define for pattern field*/
SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_offset = 1
SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_shift = 0
def SDMA_PKT_POLL_MEM_VERIFY_PATTERN_PATTERN(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_mask) << SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_shift)

 #/*define for CMP0_ADDR_START_LO word*/
 #/*define for cmp0_start_31_0 field*/
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_offset = 2
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_shift = 0
def SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_CMP0_START_31_0(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_shift)

 #/*define for CMP0_ADDR_START_HI word*/
 #/*define for cmp0_start_63_32 field*/
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_offset = 3
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_shift = 0
def SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_CMP0_START_63_32(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_shift)

 #/*define for CMP0_ADDR_END_LO word*/
 #/*define for cmp0_end_31_0 field*/
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp0_end_31_0_offset = 4
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp0_end_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp0_end_31_0_shift = 0
def SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_CMP0_END_31_0(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp0_end_31_0_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp0_end_31_0_shift)

 #/*define for CMP0_ADDR_END_HI word*/
 #/*define for cmp0_end_63_32 field*/
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp0_end_63_32_offset = 5
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp0_end_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp0_end_63_32_shift = 0
def SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_CMP0_END_63_32(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp0_end_63_32_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp0_end_63_32_shift)

 #/*define for CMP1_ADDR_START_LO word*/
 #/*define for cmp1_start_31_0 field*/
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_offset = 6
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_shift = 0
def SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_CMP1_START_31_0(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_shift)

 #/*define for CMP1_ADDR_START_HI word*/
 #/*define for cmp1_start_63_32 field*/
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_offset = 7
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_shift = 0
def SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_CMP1_START_63_32(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_shift)

 #/*define for CMP1_ADDR_END_LO word*/
 #/*define for cmp1_end_31_0 field*/
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_offset = 8
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_shift = 0
def SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_CMP1_END_31_0(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_shift)

 #/*define for CMP1_ADDR_END_HI word*/
 #/*define for cmp1_end_63_32 field*/
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_offset = 9
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_shift = 0
def SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_CMP1_END_63_32(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_shift)

 #/*define for REC_ADDR_LO word*/
 #/*define for rec_31_0 field*/
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_offset = 10
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_shift = 0
def SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_REC_31_0(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_mask) << SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_shift)

 #/*define for REC_ADDR_HI word*/
 #/*define for rec_63_32 field*/
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_offset = 11
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_shift = 0
def SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_REC_63_32(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_mask) << SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_shift)

 #/*define for RESERVED word*/
 #/*define for reserved field*/
SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_offset = 12
SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_shift = 0
def SDMA_PKT_POLL_MEM_VERIFY_RESERVED_RESERVED(x): return (((x) & SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_mask) << SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_shift)


 #/*
#** Definitions for SDMA_PKT_ATOMIC packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_ATOMIC_HEADER_op_offset = 0
SDMA_PKT_ATOMIC_HEADER_op_mask = 0x000000FF
SDMA_PKT_ATOMIC_HEADER_op_shift = 0
def SDMA_PKT_ATOMIC_HEADER_OP(x): return (((x) & SDMA_PKT_ATOMIC_HEADER_op_mask) << SDMA_PKT_ATOMIC_HEADER_op_shift)

 #/*define for loop field*/
SDMA_PKT_ATOMIC_HEADER_loop_offset = 0
SDMA_PKT_ATOMIC_HEADER_loop_mask = 0x00000001
SDMA_PKT_ATOMIC_HEADER_loop_shift = 16
def SDMA_PKT_ATOMIC_HEADER_LOOP(x): return (((x) & SDMA_PKT_ATOMIC_HEADER_loop_mask) << SDMA_PKT_ATOMIC_HEADER_loop_shift)

 #/*define for tmz field*/
SDMA_PKT_ATOMIC_HEADER_tmz_offset = 0
SDMA_PKT_ATOMIC_HEADER_tmz_mask = 0x00000001
SDMA_PKT_ATOMIC_HEADER_tmz_shift = 18
def SDMA_PKT_ATOMIC_HEADER_TMZ(x): return (((x) & SDMA_PKT_ATOMIC_HEADER_tmz_mask) << SDMA_PKT_ATOMIC_HEADER_tmz_shift)

 #/*define for cache_policy field*/
SDMA_PKT_ATOMIC_HEADER_cache_policy_offset = 0
SDMA_PKT_ATOMIC_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_ATOMIC_HEADER_cache_policy_shift = 20
def SDMA_PKT_ATOMIC_HEADER_CACHE_POLICY(x): return (((x) & SDMA_PKT_ATOMIC_HEADER_cache_policy_mask) << SDMA_PKT_ATOMIC_HEADER_cache_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_ATOMIC_HEADER_cpv_offset = 0
SDMA_PKT_ATOMIC_HEADER_cpv_mask = 0x00000001
SDMA_PKT_ATOMIC_HEADER_cpv_shift = 24
def SDMA_PKT_ATOMIC_HEADER_CPV(x): return (((x) & SDMA_PKT_ATOMIC_HEADER_cpv_mask) << SDMA_PKT_ATOMIC_HEADER_cpv_shift)

 #/*define for atomic_op field*/
SDMA_PKT_ATOMIC_HEADER_atomic_op_offset = 0
SDMA_PKT_ATOMIC_HEADER_atomic_op_mask = 0x0000007F
SDMA_PKT_ATOMIC_HEADER_atomic_op_shift = 25
def SDMA_PKT_ATOMIC_HEADER_ATOMIC_OP(x): return (((x) & SDMA_PKT_ATOMIC_HEADER_atomic_op_mask) << SDMA_PKT_ATOMIC_HEADER_atomic_op_shift)

 #/*define for ADDR_LO word*/
 #/*define for addr_31_0 field*/
SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_shift = 0
def SDMA_PKT_ATOMIC_ADDR_LO_ADDR_31_0(x): return (((x) & SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_mask) << SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_shift)

 #/*define for ADDR_HI word*/
 #/*define for addr_63_32 field*/
SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_shift = 0
def SDMA_PKT_ATOMIC_ADDR_HI_ADDR_63_32(x): return (((x) & SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_mask) << SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_shift)

 #/*define for SRC_DATA_LO word*/
 #/*define for src_data_31_0 field*/
SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_offset = 3
SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_mask = 0xFFFFFFFF
SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_shift = 0
def SDMA_PKT_ATOMIC_SRC_DATA_LO_SRC_DATA_31_0(x): return (((x) & SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_mask) << SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_shift)

 #/*define for SRC_DATA_HI word*/
 #/*define for src_data_63_32 field*/
SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_offset = 4
SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_mask = 0xFFFFFFFF
SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_shift = 0
def SDMA_PKT_ATOMIC_SRC_DATA_HI_SRC_DATA_63_32(x): return (((x) & SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_mask) << SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_shift)

 #/*define for CMP_DATA_LO word*/
 #/*define for cmp_data_31_0 field*/
SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_offset = 5
SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_mask = 0xFFFFFFFF
SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_shift = 0
def SDMA_PKT_ATOMIC_CMP_DATA_LO_CMP_DATA_31_0(x): return (((x) & SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_mask) << SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_shift)

 #/*define for CMP_DATA_HI word*/
 #/*define for cmp_data_63_32 field*/
SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_offset = 6
SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_mask = 0xFFFFFFFF
SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_shift = 0
def SDMA_PKT_ATOMIC_CMP_DATA_HI_CMP_DATA_63_32(x): return (((x) & SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_mask) << SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_shift)

 #/*define for LOOP_INTERVAL word*/
 #/*define for loop_interval field*/
SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_offset = 7
SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_mask = 0x00001FFF
SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_shift = 0
def SDMA_PKT_ATOMIC_LOOP_INTERVAL_LOOP_INTERVAL(x): return (((x) & SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_mask) << SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_shift)


 #/*
#** Definitions for SDMA_PKT_TIMESTAMP_SET packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_TIMESTAMP_SET_HEADER_op_offset = 0
SDMA_PKT_TIMESTAMP_SET_HEADER_op_mask = 0x000000FF
SDMA_PKT_TIMESTAMP_SET_HEADER_op_shift = 0
def SDMA_PKT_TIMESTAMP_SET_HEADER_OP(x): return (((x) & SDMA_PKT_TIMESTAMP_SET_HEADER_op_mask) << SDMA_PKT_TIMESTAMP_SET_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_offset = 0
SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_shift = 8
def SDMA_PKT_TIMESTAMP_SET_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_mask) << SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_shift)

 #/*define for INIT_DATA_LO word*/
 #/*define for init_data_31_0 field*/
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_offset = 1
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_mask = 0xFFFFFFFF
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_shift = 0
def SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_INIT_DATA_31_0(x): return (((x) & SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_mask) << SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_shift)

 #/*define for INIT_DATA_HI word*/
 #/*define for init_data_63_32 field*/
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_offset = 2
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_mask = 0xFFFFFFFF
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_shift = 0
def SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_INIT_DATA_63_32(x): return (((x) & SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_mask) << SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_shift)


 #/*
#** Definitions for SDMA_PKT_TIMESTAMP_GET packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_TIMESTAMP_GET_HEADER_op_offset = 0
SDMA_PKT_TIMESTAMP_GET_HEADER_op_mask = 0x000000FF
SDMA_PKT_TIMESTAMP_GET_HEADER_op_shift = 0
def SDMA_PKT_TIMESTAMP_GET_HEADER_OP(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_HEADER_op_mask) << SDMA_PKT_TIMESTAMP_GET_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_offset = 0
SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_shift = 8
def SDMA_PKT_TIMESTAMP_GET_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_mask) << SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_shift)

 #/*define for l2_policy field*/
SDMA_PKT_TIMESTAMP_GET_HEADER_l2_policy_offset = 0
SDMA_PKT_TIMESTAMP_GET_HEADER_l2_policy_mask = 0x00000003
SDMA_PKT_TIMESTAMP_GET_HEADER_l2_policy_shift = 24
def SDMA_PKT_TIMESTAMP_GET_HEADER_L2_POLICY(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_HEADER_l2_policy_mask) << SDMA_PKT_TIMESTAMP_GET_HEADER_l2_policy_shift)

 #/*define for llc_policy field*/
SDMA_PKT_TIMESTAMP_GET_HEADER_llc_policy_offset = 0
SDMA_PKT_TIMESTAMP_GET_HEADER_llc_policy_mask = 0x00000001
SDMA_PKT_TIMESTAMP_GET_HEADER_llc_policy_shift = 26
def SDMA_PKT_TIMESTAMP_GET_HEADER_LLC_POLICY(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_HEADER_llc_policy_mask) << SDMA_PKT_TIMESTAMP_GET_HEADER_llc_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_TIMESTAMP_GET_HEADER_cpv_offset = 0
SDMA_PKT_TIMESTAMP_GET_HEADER_cpv_mask = 0x00000001
SDMA_PKT_TIMESTAMP_GET_HEADER_cpv_shift = 28
def SDMA_PKT_TIMESTAMP_GET_HEADER_CPV(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_HEADER_cpv_mask) << SDMA_PKT_TIMESTAMP_GET_HEADER_cpv_shift)

 #/*define for WRITE_ADDR_LO word*/
 #/*define for write_addr_31_3 field*/
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_offset = 1
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_mask = 0x1FFFFFFF
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_shift = 3
def SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_WRITE_ADDR_31_3(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_mask) << SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_shift)

 #/*define for WRITE_ADDR_HI word*/
 #/*define for write_addr_63_32 field*/
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_offset = 2
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_shift = 0
def SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_WRITE_ADDR_63_32(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_mask) << SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_shift)


 #/*
#** Definitions for SDMA_PKT_TIMESTAMP_GET_GLOBAL packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_offset = 0
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_mask = 0x000000FF
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_shift = 0
def SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_OP(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_offset = 0
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_shift = 8
def SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_shift)

 #/*define for l2_policy field*/
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_l2_policy_offset = 0
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_l2_policy_mask = 0x00000003
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_l2_policy_shift = 24
def SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_L2_POLICY(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_l2_policy_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_l2_policy_shift)

 #/*define for llc_policy field*/
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_llc_policy_offset = 0
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_llc_policy_mask = 0x00000001
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_llc_policy_shift = 26
def SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_LLC_POLICY(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_llc_policy_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_llc_policy_shift)

 #/*define for cpv field*/
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_cpv_offset = 0
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_cpv_mask = 0x00000001
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_cpv_shift = 28
def SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_CPV(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_cpv_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_cpv_shift)

 #/*define for WRITE_ADDR_LO word*/
 #/*define for write_addr_31_3 field*/
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_offset = 1
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_mask = 0x1FFFFFFF
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_shift = 3
def SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_WRITE_ADDR_31_3(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_shift)

 #/*define for WRITE_ADDR_HI word*/
 #/*define for write_addr_63_32 field*/
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_offset = 2
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_shift = 0
def SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_WRITE_ADDR_63_32(x): return (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_shift)


 #/*
#** Definitions for SDMA_PKT_TRAP packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_TRAP_HEADER_op_offset = 0
SDMA_PKT_TRAP_HEADER_op_mask = 0x000000FF
SDMA_PKT_TRAP_HEADER_op_shift = 0
def SDMA_PKT_TRAP_HEADER_OP(x): return (((x) & SDMA_PKT_TRAP_HEADER_op_mask) << SDMA_PKT_TRAP_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_TRAP_HEADER_sub_op_offset = 0
SDMA_PKT_TRAP_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_TRAP_HEADER_sub_op_shift = 8
def SDMA_PKT_TRAP_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_TRAP_HEADER_sub_op_mask) << SDMA_PKT_TRAP_HEADER_sub_op_shift)

 #/*define for INT_CONTEXT word*/
 #/*define for int_context field*/
SDMA_PKT_TRAP_INT_CONTEXT_int_context_offset = 1
SDMA_PKT_TRAP_INT_CONTEXT_int_context_mask = 0x0FFFFFFF
SDMA_PKT_TRAP_INT_CONTEXT_int_context_shift = 0
def SDMA_PKT_TRAP_INT_CONTEXT_INT_CONTEXT(x): return (((x) & SDMA_PKT_TRAP_INT_CONTEXT_int_context_mask) << SDMA_PKT_TRAP_INT_CONTEXT_int_context_shift)


 #/*
#** Definitions for SDMA_PKT_DUMMY_TRAP packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_DUMMY_TRAP_HEADER_op_offset = 0
SDMA_PKT_DUMMY_TRAP_HEADER_op_mask = 0x000000FF
SDMA_PKT_DUMMY_TRAP_HEADER_op_shift = 0
def SDMA_PKT_DUMMY_TRAP_HEADER_OP(x): return (((x) & SDMA_PKT_DUMMY_TRAP_HEADER_op_mask) << SDMA_PKT_DUMMY_TRAP_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_offset = 0
SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_shift = 8
def SDMA_PKT_DUMMY_TRAP_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_mask) << SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_shift)

 #/*define for INT_CONTEXT word*/
 #/*define for int_context field*/
SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_offset = 1
SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_mask = 0x0FFFFFFF
SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_shift = 0
def SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_INT_CONTEXT(x): return (((x) & SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_mask) << SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_shift)


 #/*
#** Definitions for SDMA_PKT_GPUVM_INV packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_GPUVM_INV_HEADER_op_offset = 0
SDMA_PKT_GPUVM_INV_HEADER_op_mask = 0x000000FF
SDMA_PKT_GPUVM_INV_HEADER_op_shift = 0
def SDMA_PKT_GPUVM_INV_HEADER_OP(x): return (((x) & SDMA_PKT_GPUVM_INV_HEADER_op_mask) << SDMA_PKT_GPUVM_INV_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_GPUVM_INV_HEADER_sub_op_offset = 0
SDMA_PKT_GPUVM_INV_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_GPUVM_INV_HEADER_sub_op_shift = 8
def SDMA_PKT_GPUVM_INV_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_GPUVM_INV_HEADER_sub_op_mask) << SDMA_PKT_GPUVM_INV_HEADER_sub_op_shift)

 #/*define for PAYLOAD1 word*/
 #/*define for per_vmid_inv_req field*/
SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_mask = 0x0000FFFF
SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_shift = 0
def SDMA_PKT_GPUVM_INV_PAYLOAD1_PER_VMID_INV_REQ(x): return (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_shift)

 #/*define for flush_type field*/
SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_mask = 0x00000007
SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_shift = 16
def SDMA_PKT_GPUVM_INV_PAYLOAD1_FLUSH_TYPE(x): return (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_shift)

 #/*define for l2_ptes field*/
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_shift = 19
def SDMA_PKT_GPUVM_INV_PAYLOAD1_L2_PTES(x): return (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_shift)

 #/*define for l2_pde0 field*/
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_shift = 20
def SDMA_PKT_GPUVM_INV_PAYLOAD1_L2_PDE0(x): return (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_shift)

 #/*define for l2_pde1 field*/
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_shift = 21
def SDMA_PKT_GPUVM_INV_PAYLOAD1_L2_PDE1(x): return (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_shift)

 #/*define for l2_pde2 field*/
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_shift = 22
def SDMA_PKT_GPUVM_INV_PAYLOAD1_L2_PDE2(x): return (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_shift)

 #/*define for l1_ptes field*/
SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_shift = 23
def SDMA_PKT_GPUVM_INV_PAYLOAD1_L1_PTES(x): return (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_shift)

 #/*define for clr_protection_fault_status_addr field*/
SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_shift = 24
def SDMA_PKT_GPUVM_INV_PAYLOAD1_CLR_PROTECTION_FAULT_STATUS_ADDR(x): return (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_shift)

 #/*define for log_request field*/
SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_shift = 25
def SDMA_PKT_GPUVM_INV_PAYLOAD1_LOG_REQUEST(x): return (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_shift)

 #/*define for four_kilobytes field*/
SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_shift = 26
def SDMA_PKT_GPUVM_INV_PAYLOAD1_FOUR_KILOBYTES(x): return (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_shift)

 #/*define for PAYLOAD2 word*/
 #/*define for s field*/
SDMA_PKT_GPUVM_INV_PAYLOAD2_s_offset = 2
SDMA_PKT_GPUVM_INV_PAYLOAD2_s_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD2_s_shift = 0
def SDMA_PKT_GPUVM_INV_PAYLOAD2_S(x): return (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD2_s_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD2_s_shift)

 #/*define for page_va_42_12 field*/
SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_offset = 2
SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_mask = 0x7FFFFFFF
SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_shift = 1
def SDMA_PKT_GPUVM_INV_PAYLOAD2_PAGE_VA_42_12(x): return (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_shift)

 #/*define for PAYLOAD3 word*/
 #/*define for page_va_47_43 field*/
SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_offset = 3
SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_mask = 0x0000003F
SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_shift = 0
def SDMA_PKT_GPUVM_INV_PAYLOAD3_PAGE_VA_47_43(x): return (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_shift)


 #/*
#** Definitions for SDMA_PKT_GCR_REQ packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_GCR_REQ_HEADER_op_offset = 0
SDMA_PKT_GCR_REQ_HEADER_op_mask = 0x000000FF
SDMA_PKT_GCR_REQ_HEADER_op_shift = 0
def SDMA_PKT_GCR_REQ_HEADER_OP(x): return (((x) & SDMA_PKT_GCR_REQ_HEADER_op_mask) << SDMA_PKT_GCR_REQ_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_GCR_REQ_HEADER_sub_op_offset = 0
SDMA_PKT_GCR_REQ_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_GCR_REQ_HEADER_sub_op_shift = 8
def SDMA_PKT_GCR_REQ_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_GCR_REQ_HEADER_sub_op_mask) << SDMA_PKT_GCR_REQ_HEADER_sub_op_shift)

 #/*define for PAYLOAD1 word*/
 #/*define for base_va_31_7 field*/
SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_offset = 1
SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_mask = 0x01FFFFFF
SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_shift = 7
def SDMA_PKT_GCR_REQ_PAYLOAD1_BASE_VA_31_7(x): return (((x) & SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_mask) << SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_shift)

 #/*define for PAYLOAD2 word*/
 #/*define for base_va_47_32 field*/
SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_offset = 2
SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_mask = 0x0000FFFF
SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_shift = 0
def SDMA_PKT_GCR_REQ_PAYLOAD2_BASE_VA_47_32(x): return (((x) & SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_mask) << SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_shift)

 #/*define for gcr_control_15_0 field*/
SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_offset = 2
SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_mask = 0x0000FFFF
SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_shift = 16
def SDMA_PKT_GCR_REQ_PAYLOAD2_GCR_CONTROL_15_0(x): return (((x) & SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_mask) << SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_shift)

 #/*define for PAYLOAD3 word*/
 #/*define for gcr_control_18_16 field*/
SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_offset = 3
SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_mask = 0x00000007
SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_shift = 0
def SDMA_PKT_GCR_REQ_PAYLOAD3_GCR_CONTROL_18_16(x): return (((x) & SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_mask) << SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_shift)

 #/*define for limit_va_31_7 field*/
SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_offset = 3
SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_mask = 0x01FFFFFF
SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_shift = 7
def SDMA_PKT_GCR_REQ_PAYLOAD3_LIMIT_VA_31_7(x): return (((x) & SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_mask) << SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_shift)

 #/*define for PAYLOAD4 word*/
 #/*define for limit_va_47_32 field*/
SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_offset = 4
SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_mask = 0x0000FFFF
SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_shift = 0
def SDMA_PKT_GCR_REQ_PAYLOAD4_LIMIT_VA_47_32(x): return (((x) & SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_mask) << SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_shift)

 #/*define for vmid field*/
SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_offset = 4
SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_mask = 0x0000000F
SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_shift = 24
def SDMA_PKT_GCR_REQ_PAYLOAD4_VMID(x): return (((x) & SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_mask) << SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_shift)


 #/*
#** Definitions for SDMA_PKT_NOP packet
#*/

 #/*define for HEADER word*/
 #/*define for op field*/
SDMA_PKT_NOP_HEADER_op_offset = 0
SDMA_PKT_NOP_HEADER_op_mask = 0x000000FF
SDMA_PKT_NOP_HEADER_op_shift = 0
def SDMA_PKT_NOP_HEADER_OP(x): return (((x) & SDMA_PKT_NOP_HEADER_op_mask) << SDMA_PKT_NOP_HEADER_op_shift)

 #/*define for sub_op field*/
SDMA_PKT_NOP_HEADER_sub_op_offset = 0
SDMA_PKT_NOP_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_NOP_HEADER_sub_op_shift = 8
def SDMA_PKT_NOP_HEADER_SUB_OP(x): return (((x) & SDMA_PKT_NOP_HEADER_sub_op_mask) << SDMA_PKT_NOP_HEADER_sub_op_shift)

 #/*define for count field*/
SDMA_PKT_NOP_HEADER_count_offset = 0
SDMA_PKT_NOP_HEADER_count_mask = 0x00003FFF
SDMA_PKT_NOP_HEADER_count_shift = 16
def SDMA_PKT_NOP_HEADER_COUNT(x): return (((x) & SDMA_PKT_NOP_HEADER_count_mask) << SDMA_PKT_NOP_HEADER_count_shift)

 #/*define for DATA0 word*/
 #/*define for data0 field*/
SDMA_PKT_NOP_DATA0_data0_offset = 1
SDMA_PKT_NOP_DATA0_data0_mask = 0xFFFFFFFF
SDMA_PKT_NOP_DATA0_data0_shift = 0
def SDMA_PKT_NOP_DATA0_DATA0(x): return (((x) & SDMA_PKT_NOP_DATA0_data0_mask) << SDMA_PKT_NOP_DATA0_data0_shift)


 #/*
#** Definitions for SDMA_AQL_PKT_HEADER packet
#*/

 #/*define for HEADER word*/
 #/*define for format field*/
SDMA_AQL_PKT_HEADER_HEADER_format_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_format_mask = 0x000000FF
SDMA_AQL_PKT_HEADER_HEADER_format_shift = 0
def SDMA_AQL_PKT_HEADER_HEADER_FORMAT(x): return (((x) & SDMA_AQL_PKT_HEADER_HEADER_format_mask) << SDMA_AQL_PKT_HEADER_HEADER_format_shift)

 #/*define for barrier field*/
SDMA_AQL_PKT_HEADER_HEADER_barrier_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_barrier_mask = 0x00000001
SDMA_AQL_PKT_HEADER_HEADER_barrier_shift = 8
def SDMA_AQL_PKT_HEADER_HEADER_BARRIER(x): return (((x) & SDMA_AQL_PKT_HEADER_HEADER_barrier_mask) << SDMA_AQL_PKT_HEADER_HEADER_barrier_shift)

 #/*define for acquire_fence_scope field*/
SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_mask = 0x00000003
SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_shift = 9
def SDMA_AQL_PKT_HEADER_HEADER_ACQUIRE_FENCE_SCOPE(x): return (((x) & SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_mask) << SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_shift)

 #/*define for release_fence_scope field*/
SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_mask = 0x00000003
SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_shift = 11
def SDMA_AQL_PKT_HEADER_HEADER_RELEASE_FENCE_SCOPE(x): return (((x) & SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_mask) << SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_shift)

 #/*define for reserved field*/
SDMA_AQL_PKT_HEADER_HEADER_reserved_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_reserved_mask = 0x00000007
SDMA_AQL_PKT_HEADER_HEADER_reserved_shift = 13
def SDMA_AQL_PKT_HEADER_HEADER_RESERVED(x): return (((x) & SDMA_AQL_PKT_HEADER_HEADER_reserved_mask) << SDMA_AQL_PKT_HEADER_HEADER_reserved_shift)

 #/*define for op field*/
SDMA_AQL_PKT_HEADER_HEADER_op_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_op_mask = 0x0000000F
SDMA_AQL_PKT_HEADER_HEADER_op_shift = 16
def SDMA_AQL_PKT_HEADER_HEADER_OP(x): return (((x) & SDMA_AQL_PKT_HEADER_HEADER_op_mask) << SDMA_AQL_PKT_HEADER_HEADER_op_shift)

 #/*define for subop field*/
SDMA_AQL_PKT_HEADER_HEADER_subop_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_subop_mask = 0x00000007
SDMA_AQL_PKT_HEADER_HEADER_subop_shift = 20
def SDMA_AQL_PKT_HEADER_HEADER_SUBOP(x): return (((x) & SDMA_AQL_PKT_HEADER_HEADER_subop_mask) << SDMA_AQL_PKT_HEADER_HEADER_subop_shift)

 #/*define for cpv field*/
SDMA_AQL_PKT_HEADER_HEADER_cpv_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_cpv_mask = 0x00000001
SDMA_AQL_PKT_HEADER_HEADER_cpv_shift = 28
def SDMA_AQL_PKT_HEADER_HEADER_CPV(x): return (((x) & SDMA_AQL_PKT_HEADER_HEADER_cpv_mask) << SDMA_AQL_PKT_HEADER_HEADER_cpv_shift)


 #/*
#** Definitions for SDMA_AQL_PKT_COPY_LINEAR packet
#*/

 #/*define for HEADER word*/
 #/*define for format field*/
SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_mask = 0x000000FF
SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_FORMAT(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_shift)

 #/*define for barrier field*/
SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_mask = 0x00000001
SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_shift = 8
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_BARRIER(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_shift)

 #/*define for acquire_fence_scope field*/
SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_mask = 0x00000003
SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_shift = 9
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_ACQUIRE_FENCE_SCOPE(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_shift)

 #/*define for release_fence_scope field*/
SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_mask = 0x00000003
SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_shift = 11
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_RELEASE_FENCE_SCOPE(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_shift)

 #/*define for reserved field*/
SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_mask = 0x00000007
SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_shift = 13
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_RESERVED(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_shift)

 #/*define for op field*/
SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_mask = 0x0000000F
SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_shift = 16
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_OP(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_shift)

 #/*define for subop field*/
SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_mask = 0x00000007
SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_shift = 20
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_SUBOP(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_shift)

 #/*define for cpv field*/
SDMA_AQL_PKT_COPY_LINEAR_HEADER_cpv_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_cpv_mask = 0x00000001
SDMA_AQL_PKT_COPY_LINEAR_HEADER_cpv_shift = 28
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_CPV(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_cpv_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_cpv_shift)

 #/*define for RESERVED_DW1 word*/
 #/*define for reserved_dw1 field*/
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_offset = 1
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_RESERVED_DW1(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_mask) << SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_shift)

 #/*define for RETURN_ADDR_LO word*/
 #/*define for return_addr_31_0 field*/
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_offset = 2
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_RETURN_ADDR_31_0(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_mask) << SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_shift)

 #/*define for RETURN_ADDR_HI word*/
 #/*define for return_addr_63_32 field*/
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_offset = 3
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_RETURN_ADDR_63_32(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_mask) << SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_shift)

 #/*define for COUNT word*/
 #/*define for count field*/
SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_offset = 4
SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_mask = 0x003FFFFF
SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_COUNT_COUNT(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_mask) << SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_shift)

 #/*define for PARAMETER word*/
 #/*define for dst_sw field*/
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_offset = 5
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_mask = 0x00000003
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_shift = 16
def SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_DST_SW(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_mask) << SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_shift)

 #/*define for dst_cache_policy field*/
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_offset = 5
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_mask = 0x00000007
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_shift = 18
def SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_DST_CACHE_POLICY(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_mask) << SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_shift)

 #/*define for src_sw field*/
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_offset = 5
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_mask = 0x00000003
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_shift = 24
def SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_SRC_SW(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_mask) << SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_shift)

 #/*define for src_cache_policy field*/
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_offset = 5
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_mask = 0x00000007
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_shift = 26
def SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_SRC_CACHE_POLICY(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_mask) << SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_shift)

 #/*define for SRC_ADDR_LO word*/
 #/*define for src_addr_31_0 field*/
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset = 6
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_SRC_ADDR_31_0(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift)

 #/*define for SRC_ADDR_HI word*/
 #/*define for src_addr_63_32 field*/
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset = 7
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_SRC_ADDR_63_32(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift)

 #/*define for DST_ADDR_LO word*/
 #/*define for dst_addr_31_0 field*/
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_offset = 8
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_DST_ADDR_31_0(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift)

 #/*define for DST_ADDR_HI word*/
 #/*define for dst_addr_63_32 field*/
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_offset = 9
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_DST_ADDR_63_32(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift)

 #/*define for RESERVED_DW10 word*/
 #/*define for reserved_dw10 field*/
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_offset = 10
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_RESERVED_DW10(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_mask) << SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_shift)

 #/*define for RESERVED_DW11 word*/
 #/*define for reserved_dw11 field*/
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_offset = 11
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_RESERVED_DW11(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_mask) << SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_shift)

 #/*define for RESERVED_DW12 word*/
 #/*define for reserved_dw12 field*/
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_offset = 12
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_RESERVED_DW12(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_mask) << SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_shift)

 #/*define for RESERVED_DW13 word*/
 #/*define for reserved_dw13 field*/
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_offset = 13
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_RESERVED_DW13(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_mask) << SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_shift)

 #/*define for COMPLETION_SIGNAL_LO word*/
 #/*define for completion_signal_31_0 field*/
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_offset = 14
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_COMPLETION_SIGNAL_31_0(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_mask) << SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_shift)

 #/*define for COMPLETION_SIGNAL_HI word*/
 #/*define for completion_signal_63_32 field*/
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_offset = 15
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_shift = 0
def SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_COMPLETION_SIGNAL_63_32(x): return (((x) & SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_mask) << SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_shift)


 #/*
#** Definitions for SDMA_AQL_PKT_BARRIER_OR packet
#*/

 #/*define for HEADER word*/
 #/*define for format field*/
SDMA_AQL_PKT_BARRIER_OR_HEADER_format_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_format_mask = 0x000000FF
SDMA_AQL_PKT_BARRIER_OR_HEADER_format_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_HEADER_FORMAT(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_format_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_format_shift)

 #/*define for barrier field*/
SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_mask = 0x00000001
SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_shift = 8
def SDMA_AQL_PKT_BARRIER_OR_HEADER_BARRIER(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_shift)

 #/*define for acquire_fence_scope field*/
SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_mask = 0x00000003
SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_shift = 9
def SDMA_AQL_PKT_BARRIER_OR_HEADER_ACQUIRE_FENCE_SCOPE(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_shift)

 #/*define for release_fence_scope field*/
SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_mask = 0x00000003
SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_shift = 11
def SDMA_AQL_PKT_BARRIER_OR_HEADER_RELEASE_FENCE_SCOPE(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_shift)

 #/*define for reserved field*/
SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_shift = 13
def SDMA_AQL_PKT_BARRIER_OR_HEADER_RESERVED(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_shift)

 #/*define for op field*/
SDMA_AQL_PKT_BARRIER_OR_HEADER_op_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_op_mask = 0x0000000F
SDMA_AQL_PKT_BARRIER_OR_HEADER_op_shift = 16
def SDMA_AQL_PKT_BARRIER_OR_HEADER_OP(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_op_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_op_shift)

 #/*define for subop field*/
SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_shift = 20
def SDMA_AQL_PKT_BARRIER_OR_HEADER_SUBOP(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_shift)

 #/*define for cpv field*/
SDMA_AQL_PKT_BARRIER_OR_HEADER_cpv_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_cpv_mask = 0x00000001
SDMA_AQL_PKT_BARRIER_OR_HEADER_cpv_shift = 28
def SDMA_AQL_PKT_BARRIER_OR_HEADER_CPV(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_cpv_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_cpv_shift)

 #/*define for RESERVED_DW1 word*/
 #/*define for reserved_dw1 field*/
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_offset = 1
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_RESERVED_DW1(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_mask) << SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_shift)

 #/*define for DEPENDENT_ADDR_0_LO word*/
 #/*define for dependent_addr_0_31_0 field*/
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_offset = 2
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_DEPENDENT_ADDR_0_31_0(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_shift)

 #/*define for DEPENDENT_ADDR_0_HI word*/
 #/*define for dependent_addr_0_63_32 field*/
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_offset = 3
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_DEPENDENT_ADDR_0_63_32(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_shift)

 #/*define for DEPENDENT_ADDR_1_LO word*/
 #/*define for dependent_addr_1_31_0 field*/
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_offset = 4
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_DEPENDENT_ADDR_1_31_0(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_shift)

 #/*define for DEPENDENT_ADDR_1_HI word*/
 #/*define for dependent_addr_1_63_32 field*/
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_offset = 5
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_DEPENDENT_ADDR_1_63_32(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_shift)

 #/*define for DEPENDENT_ADDR_2_LO word*/
 #/*define for dependent_addr_2_31_0 field*/
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_offset = 6
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_DEPENDENT_ADDR_2_31_0(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_shift)

 #/*define for DEPENDENT_ADDR_2_HI word*/
 #/*define for dependent_addr_2_63_32 field*/
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_offset = 7
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_DEPENDENT_ADDR_2_63_32(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_shift)

 #/*define for DEPENDENT_ADDR_3_LO word*/
 #/*define for dependent_addr_3_31_0 field*/
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_offset = 8
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_DEPENDENT_ADDR_3_31_0(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_shift)

 #/*define for DEPENDENT_ADDR_3_HI word*/
 #/*define for dependent_addr_3_63_32 field*/
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_offset = 9
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_DEPENDENT_ADDR_3_63_32(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_shift)

 #/*define for DEPENDENT_ADDR_4_LO word*/
 #/*define for dependent_addr_4_31_0 field*/
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_offset = 10
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_DEPENDENT_ADDR_4_31_0(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_shift)

 #/*define for DEPENDENT_ADDR_4_HI word*/
 #/*define for dependent_addr_4_63_32 field*/
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_offset = 11
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_DEPENDENT_ADDR_4_63_32(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_shift)

 #/*define for CACHE_POLICY word*/
 #/*define for cache_policy0 field*/
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy0_offset = 12
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy0_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy0_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_CACHE_POLICY0(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy0_mask) << SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy0_shift)

 #/*define for cache_policy1 field*/
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy1_offset = 12
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy1_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy1_shift = 5
def SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_CACHE_POLICY1(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy1_mask) << SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy1_shift)

 #/*define for cache_policy2 field*/
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy2_offset = 12
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy2_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy2_shift = 10
def SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_CACHE_POLICY2(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy2_mask) << SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy2_shift)

 #/*define for cache_policy3 field*/
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy3_offset = 12
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy3_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy3_shift = 15
def SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_CACHE_POLICY3(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy3_mask) << SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy3_shift)

 #/*define for cache_policy4 field*/
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy4_offset = 12
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy4_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy4_shift = 20
def SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_CACHE_POLICY4(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy4_mask) << SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy4_shift)

 #/*define for RESERVED_DW13 word*/
 #/*define for reserved_dw13 field*/
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_offset = 13
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_RESERVED_DW13(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_mask) << SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_shift)

 #/*define for COMPLETION_SIGNAL_LO word*/
 #/*define for completion_signal_31_0 field*/
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_offset = 14
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_COMPLETION_SIGNAL_31_0(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_mask) << SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_shift)

 #/*define for COMPLETION_SIGNAL_HI word*/
 #/*define for completion_signal_63_32 field*/
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_offset = 15
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_shift = 0
def SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_COMPLETION_SIGNAL_63_32(x): return (((x) & SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_mask) << SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_shift)


#endif  #/* __SDMA_V6_0_0_PKT_OPEN_H_ */
5838
ixFIXED_PATTERN_PERF_COUNTER_1 = 0x10c
ixFIXED_PATTERN_PERF_COUNTER_10 = 0x115
ixFIXED_PATTERN_PERF_COUNTER_2 = 0x10d
ixFIXED_PATTERN_PERF_COUNTER_3 = 0x10e
ixFIXED_PATTERN_PERF_COUNTER_4 = 0x10f
ixFIXED_PATTERN_PERF_COUNTER_5 = 0x110
ixFIXED_PATTERN_PERF_COUNTER_6 = 0x111
ixFIXED_PATTERN_PERF_COUNTER_7 = 0x112
ixFIXED_PATTERN_PERF_COUNTER_8 = 0x113
ixFIXED_PATTERN_PERF_COUNTER_9 = 0x114
ixGC_CAC_ACC_CHC0 = 0x61
ixGC_CAC_ACC_CHC1 = 0x62
ixGC_CAC_ACC_CHC2 = 0x63
ixGC_CAC_ACC_CP0 = 0x10
ixGC_CAC_ACC_CP1 = 0x11
ixGC_CAC_ACC_CP2 = 0x12
ixGC_CAC_ACC_EA0 = 0x13
ixGC_CAC_ACC_EA1 = 0x14
ixGC_CAC_ACC_EA2 = 0x15
ixGC_CAC_ACC_EA3 = 0x16
ixGC_CAC_ACC_EA4 = 0x17
ixGC_CAC_ACC_EA5 = 0x18
ixGC_CAC_ACC_GDS0 = 0x2d
ixGC_CAC_ACC_GDS1 = 0x2e
ixGC_CAC_ACC_GDS2 = 0x2f
ixGC_CAC_ACC_GDS3 = 0x30
ixGC_CAC_ACC_GDS4 = 0x31
ixGC_CAC_ACC_GE0 = 0x32
ixGC_CAC_ACC_GE1 = 0x33
ixGC_CAC_ACC_GE10 = 0x3c
ixGC_CAC_ACC_GE11 = 0x3d
ixGC_CAC_ACC_GE12 = 0x3e
ixGC_CAC_ACC_GE13 = 0x3f
ixGC_CAC_ACC_GE14 = 0x40
ixGC_CAC_ACC_GE15 = 0x41
ixGC_CAC_ACC_GE16 = 0x42
ixGC_CAC_ACC_GE17 = 0x43
ixGC_CAC_ACC_GE18 = 0x44
ixGC_CAC_ACC_GE19 = 0x45
ixGC_CAC_ACC_GE2 = 0x34
ixGC_CAC_ACC_GE20 = 0x46
ixGC_CAC_ACC_GE3 = 0x35
ixGC_CAC_ACC_GE4 = 0x36
ixGC_CAC_ACC_GE5 = 0x37
ixGC_CAC_ACC_GE6 = 0x38
ixGC_CAC_ACC_GE7 = 0x39
ixGC_CAC_ACC_GE8 = 0x3a
ixGC_CAC_ACC_GE9 = 0x3b
ixGC_CAC_ACC_GL2C0 = 0x48
ixGC_CAC_ACC_GL2C1 = 0x49
ixGC_CAC_ACC_GL2C2 = 0x4a
ixGC_CAC_ACC_GL2C3 = 0x4b
ixGC_CAC_ACC_GL2C4 = 0x4c
ixGC_CAC_ACC_GUS0 = 0x64
ixGC_CAC_ACC_GUS1 = 0x65
ixGC_CAC_ACC_GUS2 = 0x66
ixGC_CAC_ACC_PH0 = 0x4d
ixGC_CAC_ACC_PH1 = 0x4e
ixGC_CAC_ACC_PH2 = 0x4f
ixGC_CAC_ACC_PH3 = 0x50
ixGC_CAC_ACC_PH4 = 0x51
ixGC_CAC_ACC_PH5 = 0x52
ixGC_CAC_ACC_PH6 = 0x53
ixGC_CAC_ACC_PH7 = 0x54
ixGC_CAC_ACC_PMM0 = 0x47
ixGC_CAC_ACC_RLC0 = 0x67
ixGC_CAC_ACC_SDMA0 = 0x55
ixGC_CAC_ACC_SDMA1 = 0x56
ixGC_CAC_ACC_SDMA10 = 0x5f
ixGC_CAC_ACC_SDMA11 = 0x60
ixGC_CAC_ACC_SDMA2 = 0x57
ixGC_CAC_ACC_SDMA3 = 0x58
ixGC_CAC_ACC_SDMA4 = 0x59
ixGC_CAC_ACC_SDMA5 = 0x5a
ixGC_CAC_ACC_SDMA6 = 0x5b
ixGC_CAC_ACC_SDMA7 = 0x5c
ixGC_CAC_ACC_SDMA8 = 0x5d
ixGC_CAC_ACC_SDMA9 = 0x5e
ixGC_CAC_ACC_UTCL2_ROUTER0 = 0x19
ixGC_CAC_ACC_UTCL2_ROUTER1 = 0x1a
ixGC_CAC_ACC_UTCL2_ROUTER2 = 0x1b
ixGC_CAC_ACC_UTCL2_ROUTER3 = 0x1c
ixGC_CAC_ACC_UTCL2_ROUTER4 = 0x1d
ixGC_CAC_ACC_UTCL2_ROUTER5 = 0x1e
ixGC_CAC_ACC_UTCL2_ROUTER6 = 0x1f
ixGC_CAC_ACC_UTCL2_ROUTER7 = 0x20
ixGC_CAC_ACC_UTCL2_ROUTER8 = 0x21
ixGC_CAC_ACC_UTCL2_ROUTER9 = 0x22
ixGC_CAC_ACC_UTCL2_VML20 = 0x23
ixGC_CAC_ACC_UTCL2_VML21 = 0x24
ixGC_CAC_ACC_UTCL2_VML22 = 0x25
ixGC_CAC_ACC_UTCL2_VML23 = 0x26
ixGC_CAC_ACC_UTCL2_VML24 = 0x27
ixGC_CAC_ACC_UTCL2_WALKER0 = 0x28
ixGC_CAC_ACC_UTCL2_WALKER1 = 0x29
ixGC_CAC_ACC_UTCL2_WALKER2 = 0x2a
ixGC_CAC_ACC_UTCL2_WALKER3 = 0x2b
ixGC_CAC_ACC_UTCL2_WALKER4 = 0x2c
ixGC_CAC_CNTL = 0x1
ixGC_CAC_ID = 0x0
ixHW_LUT_UPDATE_STATUS = 0x116
ixPWRBRK_RELEASE_TO_STALL_LUT_17_20 = 0x10b
ixPWRBRK_RELEASE_TO_STALL_LUT_1_8 = 0x109
ixPWRBRK_RELEASE_TO_STALL_LUT_9_16 = 0x10a
ixPWRBRK_STALL_TO_RELEASE_LUT_1_4 = 0x107
ixPWRBRK_STALL_TO_RELEASE_LUT_5_7 = 0x108
ixRELEASE_TO_STALL_LUT_17_20 = 0x102
ixRELEASE_TO_STALL_LUT_1_8 = 0x100
ixRELEASE_TO_STALL_LUT_9_16 = 0x101
ixRTAVFS_REG0 = 0x0
ixRTAVFS_REG1 = 0x1
ixRTAVFS_REG10 = 0xa
ixRTAVFS_REG100 = 0x64
ixRTAVFS_REG101 = 0x65
ixRTAVFS_REG102 = 0x66
ixRTAVFS_REG103 = 0x67
ixRTAVFS_REG104 = 0x68
ixRTAVFS_REG105 = 0x69
ixRTAVFS_REG106 = 0x6a
ixRTAVFS_REG107 = 0x6b
ixRTAVFS_REG108 = 0x6c
ixRTAVFS_REG109 = 0x6d
ixRTAVFS_REG11 = 0xb
ixRTAVFS_REG110 = 0x6e
ixRTAVFS_REG111 = 0x6f
ixRTAVFS_REG112 = 0x70
ixRTAVFS_REG113 = 0x71
ixRTAVFS_REG114 = 0x72
ixRTAVFS_REG115 = 0x73
ixRTAVFS_REG116 = 0x74
ixRTAVFS_REG117 = 0x75
ixRTAVFS_REG118 = 0x76
ixRTAVFS_REG119 = 0x77
ixRTAVFS_REG12 = 0xc
ixRTAVFS_REG120 = 0x78
ixRTAVFS_REG121 = 0x79
ixRTAVFS_REG122 = 0x7a
ixRTAVFS_REG123 = 0x7b
ixRTAVFS_REG124 = 0x7c
ixRTAVFS_REG125 = 0x7d
ixRTAVFS_REG126 = 0x7e
ixRTAVFS_REG127 = 0x7f
ixRTAVFS_REG128 = 0x80
ixRTAVFS_REG129 = 0x81
ixRTAVFS_REG13 = 0xd
ixRTAVFS_REG130 = 0x82
ixRTAVFS_REG131 = 0x83
ixRTAVFS_REG132 = 0x84
ixRTAVFS_REG133 = 0x85
ixRTAVFS_REG134 = 0x86
ixRTAVFS_REG135 = 0x87
ixRTAVFS_REG136 = 0x88
ixRTAVFS_REG137 = 0x89
ixRTAVFS_REG138 = 0x8a
ixRTAVFS_REG139 = 0x8b
ixRTAVFS_REG14 = 0xe
ixRTAVFS_REG140 = 0x8c
ixRTAVFS_REG141 = 0x8d
ixRTAVFS_REG142 = 0x8e
ixRTAVFS_REG143 = 0x8f
ixRTAVFS_REG144 = 0x90
ixRTAVFS_REG145 = 0x91
ixRTAVFS_REG146 = 0x92
ixRTAVFS_REG147 = 0x93
ixRTAVFS_REG148 = 0x94
ixRTAVFS_REG149 = 0x95
ixRTAVFS_REG15 = 0xf
ixRTAVFS_REG150 = 0x96
ixRTAVFS_REG151 = 0x97
ixRTAVFS_REG152 = 0x98
ixRTAVFS_REG153 = 0x99
ixRTAVFS_REG154 = 0x9a
ixRTAVFS_REG155 = 0x9b
ixRTAVFS_REG156 = 0x9c
ixRTAVFS_REG157 = 0x9d
ixRTAVFS_REG158 = 0x9e
ixRTAVFS_REG159 = 0x9f
ixRTAVFS_REG16 = 0x10
ixRTAVFS_REG160 = 0xa0
ixRTAVFS_REG161 = 0xa1
ixRTAVFS_REG162 = 0xa2
ixRTAVFS_REG163 = 0xa3
ixRTAVFS_REG164 = 0xa4
ixRTAVFS_REG165 = 0xa5
ixRTAVFS_REG166 = 0xa6
ixRTAVFS_REG167 = 0xa7
ixRTAVFS_REG168 = 0xa8
ixRTAVFS_REG169 = 0xa9
ixRTAVFS_REG17 = 0x11
ixRTAVFS_REG170 = 0xaa
ixRTAVFS_REG171 = 0xab
ixRTAVFS_REG172 = 0xac
ixRTAVFS_REG173 = 0xad
ixRTAVFS_REG174 = 0xae
ixRTAVFS_REG175 = 0xaf
ixRTAVFS_REG176 = 0xb0
ixRTAVFS_REG177 = 0xb1
ixRTAVFS_REG178 = 0xb2
ixRTAVFS_REG179 = 0xb3
ixRTAVFS_REG18 = 0x12
ixRTAVFS_REG180 = 0xb4
ixRTAVFS_REG181 = 0xb5
ixRTAVFS_REG182 = 0xb6
ixRTAVFS_REG183 = 0xb7
ixRTAVFS_REG184 = 0xb8
ixRTAVFS_REG185 = 0xb9
ixRTAVFS_REG186 = 0xba
ixRTAVFS_REG187 = 0xbb
ixRTAVFS_REG188 = 0xbc
ixRTAVFS_REG189 = 0xbd
ixRTAVFS_REG19 = 0x13
ixRTAVFS_REG190 = 0xbe
ixRTAVFS_REG191 = 0xbf
ixRTAVFS_REG192 = 0xc0
ixRTAVFS_REG193 = 0xc1
ixRTAVFS_REG194 = 0xc2
ixRTAVFS_REG2 = 0x2
ixRTAVFS_REG20 = 0x14
ixRTAVFS_REG21 = 0x15
ixRTAVFS_REG22 = 0x16
ixRTAVFS_REG23 = 0x17
ixRTAVFS_REG24 = 0x18
ixRTAVFS_REG25 = 0x19
ixRTAVFS_REG26 = 0x1a
ixRTAVFS_REG27 = 0x1b
ixRTAVFS_REG28 = 0x1c
ixRTAVFS_REG29 = 0x1d
ixRTAVFS_REG3 = 0x3
ixRTAVFS_REG30 = 0x1e
ixRTAVFS_REG31 = 0x1f
ixRTAVFS_REG32 = 0x20
ixRTAVFS_REG33 = 0x21
ixRTAVFS_REG34 = 0x22
ixRTAVFS_REG35 = 0x23
ixRTAVFS_REG36 = 0x24
ixRTAVFS_REG37 = 0x25
ixRTAVFS_REG38 = 0x26
ixRTAVFS_REG39 = 0x27
ixRTAVFS_REG4 = 0x4
ixRTAVFS_REG40 = 0x28
ixRTAVFS_REG41 = 0x29
ixRTAVFS_REG42 = 0x2a
ixRTAVFS_REG43 = 0x2b
ixRTAVFS_REG44 = 0x2c
ixRTAVFS_REG45 = 0x2d
ixRTAVFS_REG46 = 0x2e
ixRTAVFS_REG47 = 0x2f
ixRTAVFS_REG48 = 0x30
ixRTAVFS_REG49 = 0x31
ixRTAVFS_REG5 = 0x5
ixRTAVFS_REG50 = 0x32
ixRTAVFS_REG51 = 0x33
ixRTAVFS_REG52 = 0x34
ixRTAVFS_REG53 = 0x35
ixRTAVFS_REG54 = 0x36
ixRTAVFS_REG55 = 0x37
ixRTAVFS_REG56 = 0x38
ixRTAVFS_REG57 = 0x39
ixRTAVFS_REG58 = 0x3a
ixRTAVFS_REG59 = 0x3b
ixRTAVFS_REG6 = 0x6
ixRTAVFS_REG60 = 0x3c
ixRTAVFS_REG61 = 0x3d
ixRTAVFS_REG62 = 0x3e
ixRTAVFS_REG63 = 0x3f
ixRTAVFS_REG64 = 0x40
ixRTAVFS_REG65 = 0x41
ixRTAVFS_REG66 = 0x42
ixRTAVFS_REG67 = 0x43
ixRTAVFS_REG68 = 0x44
ixRTAVFS_REG69 = 0x45
ixRTAVFS_REG7 = 0x7
ixRTAVFS_REG70 = 0x46
ixRTAVFS_REG71 = 0x47
ixRTAVFS_REG72 = 0x48
ixRTAVFS_REG73 = 0x49
ixRTAVFS_REG74 = 0x4a
ixRTAVFS_REG75 = 0x4b
ixRTAVFS_REG76 = 0x4c
ixRTAVFS_REG77 = 0x4d
ixRTAVFS_REG78 = 0x4e
ixRTAVFS_REG79 = 0x4f
ixRTAVFS_REG8 = 0x8
ixRTAVFS_REG80 = 0x50
ixRTAVFS_REG81 = 0x51
ixRTAVFS_REG82 = 0x52
ixRTAVFS_REG83 = 0x53
ixRTAVFS_REG84 = 0x54
ixRTAVFS_REG85 = 0x55
ixRTAVFS_REG86 = 0x56
ixRTAVFS_REG87 = 0x57
ixRTAVFS_REG88 = 0x58
ixRTAVFS_REG89 = 0x59
ixRTAVFS_REG9 = 0x9
ixRTAVFS_REG90 = 0x5a
ixRTAVFS_REG91 = 0x5b
ixRTAVFS_REG92 = 0x5c
ixRTAVFS_REG93 = 0x5d
ixRTAVFS_REG94 = 0x5e
ixRTAVFS_REG95 = 0x5f
ixRTAVFS_REG96 = 0x60
ixRTAVFS_REG97 = 0x61
ixRTAVFS_REG98 = 0x62
ixRTAVFS_REG99 = 0x63
ixSE_CAC_CNTL = 0x1
ixSE_CAC_ID = 0x0
ixSQ_DEBUG_CTRL_LOCAL = 0x9
ixSQ_DEBUG_STS_LOCAL = 0x8
ixSQ_WAVE_ACTIVE = 0xa
ixSQ_WAVE_EXEC_HI = 0x27f
ixSQ_WAVE_EXEC_LO = 0x27e
ixSQ_WAVE_FLAT_SCRATCH_HI = 0x115
ixSQ_WAVE_FLAT_SCRATCH_LO = 0x114
ixSQ_WAVE_FLUSH_IB = 0x10e
ixSQ_WAVE_GPR_ALLOC = 0x105
ixSQ_WAVE_HW_ID1 = 0x117
ixSQ_WAVE_HW_ID2 = 0x118
ixSQ_WAVE_IB_DBG1 = 0x10d
ixSQ_WAVE_IB_STS = 0x107
ixSQ_WAVE_IB_STS2 = 0x11c
ixSQ_WAVE_LDS_ALLOC = 0x106
ixSQ_WAVE_M0 = 0x27d
ixSQ_WAVE_MODE = 0x101
ixSQ_WAVE_PC_HI = 0x109
ixSQ_WAVE_PC_LO = 0x108
ixSQ_WAVE_POPS_PACKER = 0x119
ixSQ_WAVE_SCHED_MODE = 0x11a
ixSQ_WAVE_SHADER_CYCLES = 0x11d
ixSQ_WAVE_STATUS = 0x102
ixSQ_WAVE_TRAPSTS = 0x103
ixSQ_WAVE_TTMP0 = 0x26c
ixSQ_WAVE_TTMP1 = 0x26d
ixSQ_WAVE_TTMP10 = 0x276
ixSQ_WAVE_TTMP11 = 0x277
ixSQ_WAVE_TTMP12 = 0x278
ixSQ_WAVE_TTMP13 = 0x279
ixSQ_WAVE_TTMP14 = 0x27a
ixSQ_WAVE_TTMP15 = 0x27b
ixSQ_WAVE_TTMP3 = 0x26f
ixSQ_WAVE_TTMP4 = 0x270
ixSQ_WAVE_TTMP5 = 0x271
ixSQ_WAVE_TTMP6 = 0x272
ixSQ_WAVE_TTMP7 = 0x273
ixSQ_WAVE_TTMP8 = 0x274
ixSQ_WAVE_TTMP9 = 0x275
ixSQ_WAVE_VALID_AND_IDLE = 0xb
ixSTALL_TO_PWRBRK_LUT_1_4 = 0x105
ixSTALL_TO_PWRBRK_LUT_5_7 = 0x106
ixSTALL_TO_RELEASE_LUT_1_4 = 0x103
ixSTALL_TO_RELEASE_LUT_5_7 = 0x104
regCB_BLEND0_CONTROL = 0x1e0
regCB_BLEND1_CONTROL = 0x1e1
regCB_BLEND2_CONTROL = 0x1e2
regCB_BLEND3_CONTROL = 0x1e3
regCB_BLEND4_CONTROL = 0x1e4
regCB_BLEND5_CONTROL = 0x1e5
regCB_BLEND6_CONTROL = 0x1e6
regCB_BLEND7_CONTROL = 0x1e7
regCB_BLEND_ALPHA = 0x108
regCB_BLEND_BLUE = 0x107
regCB_BLEND_GREEN = 0x106
regCB_BLEND_RED = 0x105
regCB_CACHE_EVICT_POINTS = 0x142e
regCB_CGTT_SCLK_CTRL = 0x50a8
regCB_COLOR0_ATTRIB = 0x31d
regCB_COLOR0_ATTRIB2 = 0x3b0
regCB_COLOR0_ATTRIB3 = 0x3b8
regCB_COLOR0_BASE = 0x318
regCB_COLOR0_BASE_EXT = 0x390
regCB_COLOR0_DCC_BASE = 0x325
regCB_COLOR0_DCC_BASE_EXT = 0x3a8
regCB_COLOR0_FDCC_CONTROL = 0x31e
regCB_COLOR0_INFO = 0x31c
regCB_COLOR0_VIEW = 0x31b
regCB_COLOR1_ATTRIB = 0x32c
regCB_COLOR1_ATTRIB2 = 0x3b1
regCB_COLOR1_ATTRIB3 = 0x3b9
regCB_COLOR1_BASE = 0x327
regCB_COLOR1_BASE_EXT = 0x391
regCB_COLOR1_DCC_BASE = 0x334
regCB_COLOR1_DCC_BASE_EXT = 0x3a9
regCB_COLOR1_FDCC_CONTROL = 0x32d
regCB_COLOR1_INFO = 0x32b
regCB_COLOR1_VIEW = 0x32a
regCB_COLOR2_ATTRIB = 0x33b
regCB_COLOR2_ATTRIB2 = 0x3b2
regCB_COLOR2_ATTRIB3 = 0x3ba
regCB_COLOR2_BASE = 0x336
regCB_COLOR2_BASE_EXT = 0x392
regCB_COLOR2_DCC_BASE = 0x343
regCB_COLOR2_DCC_BASE_EXT = 0x3aa
regCB_COLOR2_FDCC_CONTROL = 0x33c
regCB_COLOR2_INFO = 0x33a
regCB_COLOR2_VIEW = 0x339
regCB_COLOR3_ATTRIB = 0x34a
regCB_COLOR3_ATTRIB2 = 0x3b3
regCB_COLOR3_ATTRIB3 = 0x3bb
regCB_COLOR3_BASE = 0x345
regCB_COLOR3_BASE_EXT = 0x393
regCB_COLOR3_DCC_BASE = 0x352
regCB_COLOR3_DCC_BASE_EXT = 0x3ab
regCB_COLOR3_FDCC_CONTROL = 0x34b
regCB_COLOR3_INFO = 0x349
regCB_COLOR3_VIEW = 0x348
regCB_COLOR4_ATTRIB = 0x359
regCB_COLOR4_ATTRIB2 = 0x3b4
regCB_COLOR4_ATTRIB3 = 0x3bc
regCB_COLOR4_BASE = 0x354
regCB_COLOR4_BASE_EXT = 0x394
regCB_COLOR4_DCC_BASE = 0x361
regCB_COLOR4_DCC_BASE_EXT = 0x3ac
regCB_COLOR4_FDCC_CONTROL = 0x35a
regCB_COLOR4_INFO = 0x358
regCB_COLOR4_VIEW = 0x357
regCB_COLOR5_ATTRIB = 0x368
regCB_COLOR5_ATTRIB2 = 0x3b5
regCB_COLOR5_ATTRIB3 = 0x3bd
regCB_COLOR5_BASE = 0x363
regCB_COLOR5_BASE_EXT = 0x395
regCB_COLOR5_DCC_BASE = 0x370
regCB_COLOR5_DCC_BASE_EXT = 0x3ad
regCB_COLOR5_FDCC_CONTROL = 0x369
regCB_COLOR5_INFO = 0x367
regCB_COLOR5_VIEW = 0x366
regCB_COLOR6_ATTRIB = 0x377
regCB_COLOR6_ATTRIB2 = 0x3b6
regCB_COLOR6_ATTRIB3 = 0x3be
regCB_COLOR6_BASE = 0x372
regCB_COLOR6_BASE_EXT = 0x396
regCB_COLOR6_DCC_BASE = 0x37f
regCB_COLOR6_DCC_BASE_EXT = 0x3ae
regCB_COLOR6_FDCC_CONTROL = 0x378
regCB_COLOR6_INFO = 0x376
regCB_COLOR6_VIEW = 0x375
regCB_COLOR7_ATTRIB = 0x386
regCB_COLOR7_ATTRIB2 = 0x3b7
regCB_COLOR7_ATTRIB3 = 0x3bf
regCB_COLOR7_BASE = 0x381
regCB_COLOR7_BASE_EXT = 0x397
regCB_COLOR7_DCC_BASE = 0x38e
regCB_COLOR7_DCC_BASE_EXT = 0x3af
regCB_COLOR7_FDCC_CONTROL = 0x387
regCB_COLOR7_INFO = 0x385
regCB_COLOR7_VIEW = 0x384
regCB_COLOR_CONTROL = 0x202
regCB_COVERAGE_OUT_CONTROL = 0x10a
regCB_DCC_CONFIG = 0x1427
regCB_DCC_CONFIG2 = 0x142b
regCB_FDCC_CONTROL = 0x109
regCB_FGCG_SRAM_OVERRIDE = 0x142a
regCB_HW_CONTROL = 0x1424
regCB_HW_CONTROL_1 = 0x1425
regCB_HW_CONTROL_2 = 0x1426
regCB_HW_CONTROL_3 = 0x1423
regCB_HW_CONTROL_4 = 0x1422
regCB_HW_MEM_ARBITER_RD = 0x1428
regCB_HW_MEM_ARBITER_WR = 0x1429
regCB_PERFCOUNTER0_HI = 0x3407
regCB_PERFCOUNTER0_LO = 0x3406
regCB_PERFCOUNTER0_SELECT = 0x3c01
regCB_PERFCOUNTER0_SELECT1 = 0x3c02
regCB_PERFCOUNTER1_HI = 0x3409
regCB_PERFCOUNTER1_LO = 0x3408
regCB_PERFCOUNTER1_SELECT = 0x3c03
regCB_PERFCOUNTER2_HI = 0x340b
regCB_PERFCOUNTER2_LO = 0x340a
regCB_PERFCOUNTER2_SELECT = 0x3c04
regCB_PERFCOUNTER3_HI = 0x340d
regCB_PERFCOUNTER3_LO = 0x340c
regCB_PERFCOUNTER3_SELECT = 0x3c05
regCB_PERFCOUNTER_FILTER = 0x3c00
regCB_RMI_GL2_CACHE_CONTROL = 0x104
regCB_SHADER_MASK = 0x8f
regCB_TARGET_MASK = 0x8e
regCC_GC_EDC_CONFIG = 0x1e38
regCC_GC_PRIM_CONFIG = 0xfe0
regCC_GC_SA_UNIT_DISABLE = 0xfe9
regCC_GC_SHADER_ARRAY_CONFIG = 0x100f
regCC_GC_SHADER_RATE_CONFIG = 0x10bc
regCC_RB_BACKEND_DISABLE = 0x13dd
regCC_RB_DAISY_CHAIN = 0x13e1
regCC_RB_REDUNDANCY = 0x13dc
regCC_RMI_REDUNDANCY = 0x18a2
regCGTS_TCC_DISABLE = 0x5006
regCGTS_USER_TCC_DISABLE = 0x5b96
regCGTT_CPC_CLK_CTRL = 0x50b2
regCGTT_CPF_CLK_CTRL = 0x50b1
regCGTT_CP_CLK_CTRL = 0x50b0
regCGTT_GS_NGG_CLK_CTRL = 0x5087
regCGTT_PA_CLK_CTRL = 0x5088
regCGTT_PH_CLK_CTRL0 = 0x50f8
regCGTT_PH_CLK_CTRL1 = 0x50f9
regCGTT_PH_CLK_CTRL2 = 0x50fa
regCGTT_PH_CLK_CTRL3 = 0x50fb
regCGTT_RLC_CLK_CTRL = 0x50b5
regCGTT_SC_CLK_CTRL0 = 0x5089
regCGTT_SC_CLK_CTRL1 = 0x508a
regCGTT_SC_CLK_CTRL2 = 0x508b
regCGTT_SC_CLK_CTRL3 = 0x50bc
regCGTT_SC_CLK_CTRL4 = 0x50bd
regCGTT_SQG_CLK_CTRL = 0x508d
regCHA_CHC_CREDITS = 0x2d88
regCHA_CLIENT_FREE_DELAY = 0x2d89
regCHA_PERFCOUNTER0_HI = 0x3601
regCHA_PERFCOUNTER0_LO = 0x3600
regCHA_PERFCOUNTER0_SELECT = 0x3de0
regCHA_PERFCOUNTER0_SELECT1 = 0x3de1
regCHA_PERFCOUNTER1_HI = 0x3603
regCHA_PERFCOUNTER1_LO = 0x3602
regCHA_PERFCOUNTER1_SELECT = 0x3de2
regCHA_PERFCOUNTER2_HI = 0x3605
regCHA_PERFCOUNTER2_LO = 0x3604
regCHA_PERFCOUNTER2_SELECT = 0x3de3
regCHA_PERFCOUNTER3_HI = 0x3607
regCHA_PERFCOUNTER3_LO = 0x3606
regCHA_PERFCOUNTER3_SELECT = 0x3de4
regCHCG_CTRL = 0x2dc2
regCHCG_PERFCOUNTER0_HI = 0x33c9
regCHCG_PERFCOUNTER0_LO = 0x33c8
regCHCG_PERFCOUNTER0_SELECT = 0x3bc6
regCHCG_PERFCOUNTER0_SELECT1 = 0x3bc7
regCHCG_PERFCOUNTER1_HI = 0x33cb
regCHCG_PERFCOUNTER1_LO = 0x33ca
regCHCG_PERFCOUNTER1_SELECT = 0x3bc8
regCHCG_PERFCOUNTER2_HI = 0x33cd
regCHCG_PERFCOUNTER2_LO = 0x33cc
regCHCG_PERFCOUNTER2_SELECT = 0x3bc9
regCHCG_PERFCOUNTER3_HI = 0x33cf
regCHCG_PERFCOUNTER3_LO = 0x33ce
regCHCG_PERFCOUNTER3_SELECT = 0x3bca
regCHCG_STATUS = 0x2dc3
regCHC_CTRL = 0x2dc0
regCHC_PERFCOUNTER0_HI = 0x33c1
regCHC_PERFCOUNTER0_LO = 0x33c0
regCHC_PERFCOUNTER0_SELECT = 0x3bc0
regCHC_PERFCOUNTER0_SELECT1 = 0x3bc1
regCHC_PERFCOUNTER1_HI = 0x33c3
regCHC_PERFCOUNTER1_LO = 0x33c2
regCHC_PERFCOUNTER1_SELECT = 0x3bc2
regCHC_PERFCOUNTER2_HI = 0x33c5
regCHC_PERFCOUNTER2_LO = 0x33c4
regCHC_PERFCOUNTER2_SELECT = 0x3bc3
regCHC_PERFCOUNTER3_HI = 0x33c7
regCHC_PERFCOUNTER3_LO = 0x33c6
regCHC_PERFCOUNTER3_SELECT = 0x3bc4
regCHC_STATUS = 0x2dc1
regCHICKEN_BITS = 0x142d
regCHI_CHR_MGCG_OVERRIDE = 0x50e9
regCHI_CHR_REP_FGCG_OVERRIDE = 0x2d8c
regCH_ARB_CTRL = 0x2d80
regCH_ARB_STATUS = 0x2d83
regCH_DRAM_BURST_CTRL = 0x2d84
regCH_DRAM_BURST_MASK = 0x2d82
regCH_PIPE_STEER = 0x5b88
regCH_VC5_ENABLE = 0x2d94
regCOHER_DEST_BASE_0 = 0x92
regCOHER_DEST_BASE_1 = 0x93
regCOHER_DEST_BASE_2 = 0x7e
regCOHER_DEST_BASE_3 = 0x7f
regCOHER_DEST_BASE_HI_0 = 0x7a
regCOHER_DEST_BASE_HI_1 = 0x7b
regCOHER_DEST_BASE_HI_2 = 0x7c
regCOHER_DEST_BASE_HI_3 = 0x7d
regCOMPUTE_DDID_INDEX = 0x1bc9
regCOMPUTE_DESTINATION_EN_SE0 = 0x1bb6
regCOMPUTE_DESTINATION_EN_SE1 = 0x1bb7
regCOMPUTE_DESTINATION_EN_SE2 = 0x1bb9
regCOMPUTE_DESTINATION_EN_SE3 = 0x1bba
regCOMPUTE_DIM_X = 0x1ba1
regCOMPUTE_DIM_Y = 0x1ba2
regCOMPUTE_DIM_Z = 0x1ba3
regCOMPUTE_DISPATCH_END = 0x1c1e
regCOMPUTE_DISPATCH_ID = 0x1bc0
regCOMPUTE_DISPATCH_INITIATOR = 0x1ba0
regCOMPUTE_DISPATCH_INTERLEAVE = 0x1bcf
regCOMPUTE_DISPATCH_PKT_ADDR_HI = 0x1baf
regCOMPUTE_DISPATCH_PKT_ADDR_LO = 0x1bae
regCOMPUTE_DISPATCH_SCRATCH_BASE_HI = 0x1bb1
regCOMPUTE_DISPATCH_SCRATCH_BASE_LO = 0x1bb0
regCOMPUTE_DISPATCH_TUNNEL = 0x1c1d
regCOMPUTE_MISC_RESERVED = 0x1bbf
regCOMPUTE_NOWHERE = 0x1c1f
regCOMPUTE_NUM_THREAD_X = 0x1ba7
regCOMPUTE_NUM_THREAD_Y = 0x1ba8
regCOMPUTE_NUM_THREAD_Z = 0x1ba9
regCOMPUTE_PERFCOUNT_ENABLE = 0x1bab
regCOMPUTE_PGM_HI = 0x1bad
regCOMPUTE_PGM_LO = 0x1bac
regCOMPUTE_PGM_RSRC1 = 0x1bb2
regCOMPUTE_PGM_RSRC2 = 0x1bb3
regCOMPUTE_PGM_RSRC3 = 0x1bc8
regCOMPUTE_PIPELINESTAT_ENABLE = 0x1baa
regCOMPUTE_RELAUNCH = 0x1bd0
regCOMPUTE_RELAUNCH2 = 0x1bd3
regCOMPUTE_REQ_CTRL = 0x1bc2
regCOMPUTE_RESOURCE_LIMITS = 0x1bb5
regCOMPUTE_RESTART_X = 0x1bbb
regCOMPUTE_RESTART_Y = 0x1bbc
regCOMPUTE_RESTART_Z = 0x1bbd
regCOMPUTE_SHADER_CHKSUM = 0x1bca
regCOMPUTE_START_X = 0x1ba4
regCOMPUTE_START_Y = 0x1ba5
regCOMPUTE_START_Z = 0x1ba6
regCOMPUTE_STATIC_THREAD_MGMT_SE0 = 0x1bb6
regCOMPUTE_STATIC_THREAD_MGMT_SE1 = 0x1bb7
regCOMPUTE_STATIC_THREAD_MGMT_SE2 = 0x1bb9
regCOMPUTE_STATIC_THREAD_MGMT_SE3 = 0x1bba
regCOMPUTE_STATIC_THREAD_MGMT_SE4 = 0x1bcb
regCOMPUTE_STATIC_THREAD_MGMT_SE5 = 0x1bcc
regCOMPUTE_STATIC_THREAD_MGMT_SE6 = 0x1bcd
regCOMPUTE_STATIC_THREAD_MGMT_SE7 = 0x1bce
regCOMPUTE_THREADGROUP_ID = 0x1bc1
regCOMPUTE_THREAD_TRACE_ENABLE = 0x1bbe
regCOMPUTE_TMPRING_SIZE = 0x1bb8
regCOMPUTE_USER_ACCUM_0 = 0x1bc4
regCOMPUTE_USER_ACCUM_1 = 0x1bc5
regCOMPUTE_USER_ACCUM_2 = 0x1bc6
regCOMPUTE_USER_ACCUM_3 = 0x1bc7
regCOMPUTE_USER_DATA_0 = 0x1be0
regCOMPUTE_USER_DATA_1 = 0x1be1
regCOMPUTE_USER_DATA_10 = 0x1bea
regCOMPUTE_USER_DATA_11 = 0x1beb
regCOMPUTE_USER_DATA_12 = 0x1bec
regCOMPUTE_USER_DATA_13 = 0x1bed
regCOMPUTE_USER_DATA_14 = 0x1bee
regCOMPUTE_USER_DATA_15 = 0x1bef
regCOMPUTE_USER_DATA_2 = 0x1be2
regCOMPUTE_USER_DATA_3 = 0x1be3
regCOMPUTE_USER_DATA_4 = 0x1be4
regCOMPUTE_USER_DATA_5 = 0x1be5
regCOMPUTE_USER_DATA_6 = 0x1be6
regCOMPUTE_USER_DATA_7 = 0x1be7
regCOMPUTE_USER_DATA_8 = 0x1be8
regCOMPUTE_USER_DATA_9 = 0x1be9
regCOMPUTE_VMID = 0x1bb4
regCOMPUTE_WAVE_RESTORE_ADDR_HI = 0x1bd2
regCOMPUTE_WAVE_RESTORE_ADDR_LO = 0x1bd1
regCONFIG_RESERVED_REG0 = 0x800
regCONFIG_RESERVED_REG1 = 0x801
regCONTEXT_RESERVED_REG0 = 0xdb
regCONTEXT_RESERVED_REG1 = 0xdc
regCPC_DDID_BASE_ADDR_HI = 0x1e6c
regCPC_DDID_BASE_ADDR_LO = 0x1e6b
regCPC_DDID_CNTL = 0x1e6d
regCPC_INT_ADDR = 0x1dd9
regCPC_INT_CNTL = 0x1e54
regCPC_INT_CNTX_ID = 0x1e57
regCPC_INT_INFO = 0x1dd7
regCPC_INT_PASID = 0x1dda
regCPC_INT_STATUS = 0x1e55
regCPC_LATENCY_STATS_DATA = 0x300e
regCPC_LATENCY_STATS_SELECT = 0x380e
regCPC_OS_PIPES = 0x1e67
regCPC_PERFCOUNTER0_HI = 0x3007
regCPC_PERFCOUNTER0_LO = 0x3006
regCPC_PERFCOUNTER0_SELECT = 0x3809
regCPC_PERFCOUNTER0_SELECT1 = 0x3804
regCPC_PERFCOUNTER1_HI = 0x3005
regCPC_PERFCOUNTER1_LO = 0x3004
regCPC_PERFCOUNTER1_SELECT = 0x3803
regCPC_PSP_DEBUG = 0x5c11
regCPC_SUSPEND_CNTL_STACK_OFFSET = 0x1e63
regCPC_SUSPEND_CNTL_STACK_SIZE = 0x1e64
regCPC_SUSPEND_CTX_SAVE_BASE_ADDR_HI = 0x1e61
regCPC_SUSPEND_CTX_SAVE_BASE_ADDR_LO = 0x1e60
regCPC_SUSPEND_CTX_SAVE_CONTROL = 0x1e62
regCPC_SUSPEND_CTX_SAVE_SIZE = 0x1e66
regCPC_SUSPEND_WG_STATE_OFFSET = 0x1e65
regCPC_TC_PERF_COUNTER_WINDOW_SELECT = 0x380f
regCPC_UTCL1_CNTL = 0x1ddd
regCPC_UTCL1_ERROR = 0x1dff
regCPC_UTCL1_STATUS = 0x1f55
regCPF_GCR_CNTL = 0x1f53
regCPF_LATENCY_STATS_DATA = 0x300c
regCPF_LATENCY_STATS_SELECT = 0x380c
regCPF_PERFCOUNTER0_HI = 0x300b
regCPF_PERFCOUNTER0_LO = 0x300a
regCPF_PERFCOUNTER0_SELECT = 0x3807
regCPF_PERFCOUNTER0_SELECT1 = 0x3806
regCPF_PERFCOUNTER1_HI = 0x3009
regCPF_PERFCOUNTER1_LO = 0x3008
regCPF_PERFCOUNTER1_SELECT = 0x3805
regCPF_TC_PERF_COUNTER_WINDOW_SELECT = 0x380a
regCPF_UTCL1_CNTL = 0x1dde
regCPF_UTCL1_STATUS = 0x1f56
regCPG_LATENCY_STATS_DATA = 0x300d
regCPG_LATENCY_STATS_SELECT = 0x380d
regCPG_PERFCOUNTER0_HI = 0x3003
regCPG_PERFCOUNTER0_LO = 0x3002
regCPG_PERFCOUNTER0_SELECT = 0x3802
regCPG_PERFCOUNTER0_SELECT1 = 0x3801
regCPG_PERFCOUNTER1_HI = 0x3001
regCPG_PERFCOUNTER1_LO = 0x3000
regCPG_PERFCOUNTER1_SELECT = 0x3800
regCPG_PSP_DEBUG = 0x5c10
regCPG_RCIU_CAM_DATA = 0x1f45
regCPG_RCIU_CAM_DATA_PHASE0 = 0x1f45
regCPG_RCIU_CAM_DATA_PHASE1 = 0x1f45
regCPG_RCIU_CAM_DATA_PHASE2 = 0x1f45
regCPG_RCIU_CAM_INDEX = 0x1f44
regCPG_TC_PERF_COUNTER_WINDOW_SELECT = 0x380b
regCPG_UTCL1_CNTL = 0x1ddc
regCPG_UTCL1_ERROR = 0x1dfe
regCPG_UTCL1_STATUS = 0x1f54
regCP_APPEND_ADDR_HI = 0x2059
regCP_APPEND_ADDR_LO = 0x2058
regCP_APPEND_CMD_ADDR_HI = 0x20a1
regCP_APPEND_CMD_ADDR_LO = 0x20a0
regCP_APPEND_DATA = 0x205a
regCP_APPEND_DATA_HI = 0x204c
regCP_APPEND_DATA_LO = 0x205a
regCP_APPEND_DDID_CNT = 0x204b
regCP_APPEND_LAST_CS_FENCE = 0x205b
regCP_APPEND_LAST_CS_FENCE_HI = 0x204d
regCP_APPEND_LAST_CS_FENCE_LO = 0x205b
regCP_APPEND_LAST_PS_FENCE = 0x205c
regCP_APPEND_LAST_PS_FENCE_HI = 0x204e
regCP_APPEND_LAST_PS_FENCE_LO = 0x205c
regCP_AQL_SMM_STATUS = 0x1ddf
regCP_ATOMIC_PREOP_HI = 0x205e
regCP_ATOMIC_PREOP_LO = 0x205d
regCP_BUSY_STAT = 0xf3f
regCP_CMD_DATA = 0xf7f
regCP_CMD_INDEX = 0xf7e
regCP_CNTX_STAT = 0xf58
regCP_CONTEXT_CNTL = 0x1e4d
regCP_CPC_BUSY_HYSTERESIS = 0x1edb
regCP_CPC_BUSY_STAT = 0xe25
regCP_CPC_BUSY_STAT2 = 0xe2a
regCP_CPC_DEBUG = 0x1e21
regCP_CPC_DEBUG_CNTL = 0xe20
regCP_CPC_DEBUG_DATA = 0xe21
regCP_CPC_GFX_CNTL = 0x1f5a
regCP_CPC_GRBM_FREE_COUNT = 0xe2b
regCP_CPC_HALT_HYST_COUNT = 0xe47
regCP_CPC_IC_BASE_CNTL = 0x584e
regCP_CPC_IC_BASE_HI = 0x584d
regCP_CPC_IC_BASE_LO = 0x584c
regCP_CPC_IC_OP_CNTL = 0x297a
regCP_CPC_MGCG_SYNC_CNTL = 0x1dd6
regCP_CPC_PRIV_VIOLATION_ADDR = 0xe2c
regCP_CPC_SCRATCH_DATA = 0xe31
regCP_CPC_SCRATCH_INDEX = 0xe30
regCP_CPC_STALLED_STAT1 = 0xe26
regCP_CPC_STATUS = 0xe24
regCP_CPF_BUSY_HYSTERESIS1 = 0x1edc
regCP_CPF_BUSY_HYSTERESIS2 = 0x1edd
regCP_CPF_BUSY_STAT = 0xe28
regCP_CPF_BUSY_STAT2 = 0xe33
regCP_CPF_GRBM_FREE_COUNT = 0xe32
regCP_CPF_STALLED_STAT1 = 0xe29
regCP_CPF_STATUS = 0xe27
regCP_CPG_BUSY_HYSTERESIS1 = 0x1ede
regCP_CPG_BUSY_HYSTERESIS2 = 0x1edf
regCP_CSF_STAT = 0xf54
regCP_CU_MASK_ADDR_HI = 0x1dd3
regCP_CU_MASK_ADDR_LO = 0x1dd2
regCP_CU_MASK_CNTL = 0x1dd4
regCP_DB_BASE_HI = 0x20d9
regCP_DB_BASE_LO = 0x20d8
regCP_DB_BUFSZ = 0x20da
regCP_DB_CMD_BUFSZ = 0x20db
regCP_DDID_BASE_ADDR_HI = 0x1e6c
regCP_DDID_BASE_ADDR_LO = 0x1e6b
regCP_DDID_CNTL = 0x1e6d
regCP_DEBUG = 0x1e1f
regCP_DEBUG_2 = 0x1800
regCP_DEBUG_CNTL = 0xf98
regCP_DEBUG_DATA = 0xf99
regCP_DEVICE_ID = 0x1deb
regCP_DISPATCH_INDR_ADDR = 0x20f6
regCP_DISPATCH_INDR_ADDR_HI = 0x20f7
regCP_DMA_CNTL = 0x208a
regCP_DMA_ME_CMD_ADDR_HI = 0x209d
regCP_DMA_ME_CMD_ADDR_LO = 0x209c
regCP_DMA_ME_COMMAND = 0x2084
regCP_DMA_ME_CONTROL = 0x2078
regCP_DMA_ME_DST_ADDR = 0x2082
regCP_DMA_ME_DST_ADDR_HI = 0x2083
regCP_DMA_ME_SRC_ADDR = 0x2080
regCP_DMA_ME_SRC_ADDR_HI = 0x2081
regCP_DMA_PFP_CMD_ADDR_HI = 0x209f
regCP_DMA_PFP_CMD_ADDR_LO = 0x209e
regCP_DMA_PFP_COMMAND = 0x2089
regCP_DMA_PFP_CONTROL = 0x2077
regCP_DMA_PFP_DST_ADDR = 0x2087
regCP_DMA_PFP_DST_ADDR_HI = 0x2088
regCP_DMA_PFP_SRC_ADDR = 0x2085
regCP_DMA_PFP_SRC_ADDR_HI = 0x2086
regCP_DMA_READ_TAGS = 0x208b
regCP_DMA_WATCH0_ADDR_HI = 0x1ec1
regCP_DMA_WATCH0_ADDR_LO = 0x1ec0
regCP_DMA_WATCH0_CNTL = 0x1ec3
regCP_DMA_WATCH0_MASK = 0x1ec2
regCP_DMA_WATCH1_ADDR_HI = 0x1ec5
regCP_DMA_WATCH1_ADDR_LO = 0x1ec4
regCP_DMA_WATCH1_CNTL = 0x1ec7
regCP_DMA_WATCH1_MASK = 0x1ec6
regCP_DMA_WATCH2_ADDR_HI = 0x1ec9
regCP_DMA_WATCH2_ADDR_LO = 0x1ec8
regCP_DMA_WATCH2_CNTL = 0x1ecb
regCP_DMA_WATCH2_MASK = 0x1eca
regCP_DMA_WATCH3_ADDR_HI = 0x1ecd
regCP_DMA_WATCH3_ADDR_LO = 0x1ecc
regCP_DMA_WATCH3_CNTL = 0x1ecf
regCP_DMA_WATCH3_MASK = 0x1ece
regCP_DMA_WATCH_STAT = 0x1ed2
regCP_DMA_WATCH_STAT_ADDR_HI = 0x1ed1
regCP_DMA_WATCH_STAT_ADDR_LO = 0x1ed0
regCP_DRAW_INDX_INDR_ADDR = 0x20f4
regCP_DRAW_INDX_INDR_ADDR_HI = 0x20f5
regCP_DRAW_OBJECT = 0x3810
regCP_DRAW_OBJECT_COUNTER = 0x3811
regCP_DRAW_WINDOW_CNTL = 0x3815
regCP_DRAW_WINDOW_HI = 0x3813
regCP_DRAW_WINDOW_LO = 0x3814
regCP_DRAW_WINDOW_MASK_HI = 0x3812
regCP_ECC_FIRSTOCCURRENCE = 0x1e1a
regCP_ECC_FIRSTOCCURRENCE_RING0 = 0x1e1b
regCP_ECC_FIRSTOCCURRENCE_RING1 = 0x1e1c
regCP_EOPQ_WAIT_TIME = 0x1dd5
regCP_EOP_DONE_ADDR_HI = 0x2001
regCP_EOP_DONE_ADDR_LO = 0x2000
regCP_EOP_DONE_CNTX_ID = 0x20d7
regCP_EOP_DONE_DATA_CNTL = 0x20d6
regCP_EOP_DONE_DATA_HI = 0x2003
regCP_EOP_DONE_DATA_LO = 0x2002
regCP_EOP_DONE_EVENT_CNTL = 0x20d5
regCP_EOP_LAST_FENCE_HI = 0x2005
regCP_EOP_LAST_FENCE_LO = 0x2004
regCP_FATAL_ERROR = 0x1df0
regCP_FETCHER_SOURCE = 0x1801
regCP_GDS_ATOMIC0_PREOP_HI = 0x2060
regCP_GDS_ATOMIC0_PREOP_LO = 0x205f
regCP_GDS_ATOMIC1_PREOP_HI = 0x2062
regCP_GDS_ATOMIC1_PREOP_LO = 0x2061
regCP_GDS_BKUP_ADDR = 0x20fb
regCP_GDS_BKUP_ADDR_HI = 0x20fc
regCP_GE_MSINVOC_COUNT_HI = 0x20a7
regCP_GE_MSINVOC_COUNT_LO = 0x20a6
regCP_GFX_CNTL = 0x2a00
regCP_GFX_DDID_DELTA_RPT_COUNT = 0x1e71
regCP_GFX_DDID_INFLIGHT_COUNT = 0x1e6e
regCP_GFX_DDID_RPTR = 0x1e70
regCP_GFX_DDID_WPTR = 0x1e6f
regCP_GFX_ERROR = 0x1ddb
regCP_GFX_HPD_CONTROL0 = 0x1e73
regCP_GFX_HPD_OSPRE_FENCE_ADDR_HI = 0x1e75
regCP_GFX_HPD_OSPRE_FENCE_ADDR_LO = 0x1e74
regCP_GFX_HPD_OSPRE_FENCE_DATA_HI = 0x1e77
regCP_GFX_HPD_OSPRE_FENCE_DATA_LO = 0x1e76
regCP_GFX_HPD_STATUS0 = 0x1e72
regCP_GFX_HQD_ACTIVE = 0x1e80
regCP_GFX_HQD_BASE = 0x1e86
regCP_GFX_HQD_BASE_HI = 0x1e87
regCP_GFX_HQD_CNTL = 0x1e8f
regCP_GFX_HQD_CSMD_RPTR = 0x1e90
regCP_GFX_HQD_DEQUEUE_REQUEST = 0x1e93
regCP_GFX_HQD_HQ_CONTROL0 = 0x1e99
regCP_GFX_HQD_HQ_STATUS0 = 0x1e98
regCP_GFX_HQD_IQ_TIMER = 0x1e96
regCP_GFX_HQD_MAPPED = 0x1e94
regCP_GFX_HQD_OFFSET = 0x1e8e
regCP_GFX_HQD_QUANTUM = 0x1e85
regCP_GFX_HQD_QUEUE_PRIORITY = 0x1e84
regCP_GFX_HQD_QUE_MGR_CONTROL = 0x1e95
regCP_GFX_HQD_RPTR = 0x1e88
regCP_GFX_HQD_RPTR_ADDR = 0x1e89
regCP_GFX_HQD_RPTR_ADDR_HI = 0x1e8a
regCP_GFX_HQD_VMID = 0x1e81
regCP_GFX_HQD_WPTR = 0x1e91
regCP_GFX_HQD_WPTR_HI = 0x1e92
regCP_GFX_INDEX_MUTEX = 0x1e78
regCP_GFX_MQD_BASE_ADDR = 0x1e7e
regCP_GFX_MQD_BASE_ADDR_HI = 0x1e7f
regCP_GFX_MQD_CONTROL = 0x1e9a
regCP_GFX_QUEUE_INDEX = 0x1e37
regCP_GFX_RS64_DC_APERTURE0_BASE0 = 0x2a49
regCP_GFX_RS64_DC_APERTURE0_BASE1 = 0x2a79
regCP_GFX_RS64_DC_APERTURE0_CNTL0 = 0x2a4b
regCP_GFX_RS64_DC_APERTURE0_CNTL1 = 0x2a7b
regCP_GFX_RS64_DC_APERTURE0_MASK0 = 0x2a4a
regCP_GFX_RS64_DC_APERTURE0_MASK1 = 0x2a7a
regCP_GFX_RS64_DC_APERTURE10_BASE0 = 0x2a67
regCP_GFX_RS64_DC_APERTURE10_BASE1 = 0x2a97
regCP_GFX_RS64_DC_APERTURE10_CNTL0 = 0x2a69
regCP_GFX_RS64_DC_APERTURE10_CNTL1 = 0x2a99
regCP_GFX_RS64_DC_APERTURE10_MASK0 = 0x2a68
regCP_GFX_RS64_DC_APERTURE10_MASK1 = 0x2a98
regCP_GFX_RS64_DC_APERTURE11_BASE0 = 0x2a6a
regCP_GFX_RS64_DC_APERTURE11_BASE1 = 0x2a9a
regCP_GFX_RS64_DC_APERTURE11_CNTL0 = 0x2a6c
regCP_GFX_RS64_DC_APERTURE11_CNTL1 = 0x2a9c
regCP_GFX_RS64_DC_APERTURE11_MASK0 = 0x2a6b
regCP_GFX_RS64_DC_APERTURE11_MASK1 = 0x2a9b
regCP_GFX_RS64_DC_APERTURE12_BASE0 = 0x2a6d
regCP_GFX_RS64_DC_APERTURE12_BASE1 = 0x2a9d
regCP_GFX_RS64_DC_APERTURE12_CNTL0 = 0x2a6f
regCP_GFX_RS64_DC_APERTURE12_CNTL1 = 0x2a9f
regCP_GFX_RS64_DC_APERTURE12_MASK0 = 0x2a6e
regCP_GFX_RS64_DC_APERTURE12_MASK1 = 0x2a9e
regCP_GFX_RS64_DC_APERTURE13_BASE0 = 0x2a70
regCP_GFX_RS64_DC_APERTURE13_BASE1 = 0x2aa0
regCP_GFX_RS64_DC_APERTURE13_CNTL0 = 0x2a72
regCP_GFX_RS64_DC_APERTURE13_CNTL1 = 0x2aa2
regCP_GFX_RS64_DC_APERTURE13_MASK0 = 0x2a71
regCP_GFX_RS64_DC_APERTURE13_MASK1 = 0x2aa1
regCP_GFX_RS64_DC_APERTURE14_BASE0 = 0x2a73
regCP_GFX_RS64_DC_APERTURE14_BASE1 = 0x2aa3
regCP_GFX_RS64_DC_APERTURE14_CNTL0 = 0x2a75
regCP_GFX_RS64_DC_APERTURE14_CNTL1 = 0x2aa5
regCP_GFX_RS64_DC_APERTURE14_MASK0 = 0x2a74
regCP_GFX_RS64_DC_APERTURE14_MASK1 = 0x2aa4
regCP_GFX_RS64_DC_APERTURE15_BASE0 = 0x2a76
regCP_GFX_RS64_DC_APERTURE15_BASE1 = 0x2aa6
regCP_GFX_RS64_DC_APERTURE15_CNTL0 = 0x2a78
regCP_GFX_RS64_DC_APERTURE15_CNTL1 = 0x2aa8
regCP_GFX_RS64_DC_APERTURE15_MASK0 = 0x2a77
regCP_GFX_RS64_DC_APERTURE15_MASK1 = 0x2aa7
regCP_GFX_RS64_DC_APERTURE1_BASE0 = 0x2a4c
regCP_GFX_RS64_DC_APERTURE1_BASE1 = 0x2a7c
regCP_GFX_RS64_DC_APERTURE1_CNTL0 = 0x2a4e
regCP_GFX_RS64_DC_APERTURE1_CNTL1 = 0x2a7e
regCP_GFX_RS64_DC_APERTURE1_MASK0 = 0x2a4d
regCP_GFX_RS64_DC_APERTURE1_MASK1 = 0x2a7d
regCP_GFX_RS64_DC_APERTURE2_BASE0 = 0x2a4f
regCP_GFX_RS64_DC_APERTURE2_BASE1 = 0x2a7f
regCP_GFX_RS64_DC_APERTURE2_CNTL0 = 0x2a51
regCP_GFX_RS64_DC_APERTURE2_CNTL1 = 0x2a81
regCP_GFX_RS64_DC_APERTURE2_MASK0 = 0x2a50
regCP_GFX_RS64_DC_APERTURE2_MASK1 = 0x2a80
regCP_GFX_RS64_DC_APERTURE3_BASE0 = 0x2a52
regCP_GFX_RS64_DC_APERTURE3_BASE1 = 0x2a82
regCP_GFX_RS64_DC_APERTURE3_CNTL0 = 0x2a54
regCP_GFX_RS64_DC_APERTURE3_CNTL1 = 0x2a84
regCP_GFX_RS64_DC_APERTURE3_MASK0 = 0x2a53
regCP_GFX_RS64_DC_APERTURE3_MASK1 = 0x2a83
regCP_GFX_RS64_DC_APERTURE4_BASE0 = 0x2a55
regCP_GFX_RS64_DC_APERTURE4_BASE1 = 0x2a85
regCP_GFX_RS64_DC_APERTURE4_CNTL0 = 0x2a57
regCP_GFX_RS64_DC_APERTURE4_CNTL1 = 0x2a87
regCP_GFX_RS64_DC_APERTURE4_MASK0 = 0x2a56
regCP_GFX_RS64_DC_APERTURE4_MASK1 = 0x2a86
regCP_GFX_RS64_DC_APERTURE5_BASE0 = 0x2a58
regCP_GFX_RS64_DC_APERTURE5_BASE1 = 0x2a88
regCP_GFX_RS64_DC_APERTURE5_CNTL0 = 0x2a5a
regCP_GFX_RS64_DC_APERTURE5_CNTL1 = 0x2a8a
regCP_GFX_RS64_DC_APERTURE5_MASK0 = 0x2a59
regCP_GFX_RS64_DC_APERTURE5_MASK1 = 0x2a89
regCP_GFX_RS64_DC_APERTURE6_BASE0 = 0x2a5b
regCP_GFX_RS64_DC_APERTURE6_BASE1 = 0x2a8b
regCP_GFX_RS64_DC_APERTURE6_CNTL0 = 0x2a5d
regCP_GFX_RS64_DC_APERTURE6_CNTL1 = 0x2a8d
regCP_GFX_RS64_DC_APERTURE6_MASK0 = 0x2a5c
regCP_GFX_RS64_DC_APERTURE6_MASK1 = 0x2a8c
regCP_GFX_RS64_DC_APERTURE7_BASE0 = 0x2a5e
regCP_GFX_RS64_DC_APERTURE7_BASE1 = 0x2a8e
regCP_GFX_RS64_DC_APERTURE7_CNTL0 = 0x2a60
regCP_GFX_RS64_DC_APERTURE7_CNTL1 = 0x2a90
regCP_GFX_RS64_DC_APERTURE7_MASK0 = 0x2a5f
regCP_GFX_RS64_DC_APERTURE7_MASK1 = 0x2a8f
regCP_GFX_RS64_DC_APERTURE8_BASE0 = 0x2a61
regCP_GFX_RS64_DC_APERTURE8_BASE1 = 0x2a91
regCP_GFX_RS64_DC_APERTURE8_CNTL0 = 0x2a63
regCP_GFX_RS64_DC_APERTURE8_CNTL1 = 0x2a93
regCP_GFX_RS64_DC_APERTURE8_MASK0 = 0x2a62
regCP_GFX_RS64_DC_APERTURE8_MASK1 = 0x2a92
regCP_GFX_RS64_DC_APERTURE9_BASE0 = 0x2a64
regCP_GFX_RS64_DC_APERTURE9_BASE1 = 0x2a94
regCP_GFX_RS64_DC_APERTURE9_CNTL0 = 0x2a66
regCP_GFX_RS64_DC_APERTURE9_CNTL1 = 0x2a96
regCP_GFX_RS64_DC_APERTURE9_MASK0 = 0x2a65
regCP_GFX_RS64_DC_APERTURE9_MASK1 = 0x2a95
regCP_GFX_RS64_DC_BASE0_HI = 0x5865
regCP_GFX_RS64_DC_BASE0_LO = 0x5863
regCP_GFX_RS64_DC_BASE1_HI = 0x5866
regCP_GFX_RS64_DC_BASE1_LO = 0x5864
regCP_GFX_RS64_DC_BASE_CNTL = 0x2a08
regCP_GFX_RS64_DC_OP_CNTL = 0x2a09
regCP_GFX_RS64_DM_INDEX_ADDR = 0x5c04
regCP_GFX_RS64_DM_INDEX_DATA = 0x5c05
regCP_GFX_RS64_GP0_HI0 = 0x2a26
regCP_GFX_RS64_GP0_HI1 = 0x2a27
regCP_GFX_RS64_GP0_LO0 = 0x2a24
regCP_GFX_RS64_GP0_LO1 = 0x2a25
regCP_GFX_RS64_GP1_HI0 = 0x2a2a
regCP_GFX_RS64_GP1_HI1 = 0x2a2b
regCP_GFX_RS64_GP1_LO0 = 0x2a28
regCP_GFX_RS64_GP1_LO1 = 0x2a29
regCP_GFX_RS64_GP2_HI0 = 0x2a2e
regCP_GFX_RS64_GP2_HI1 = 0x2a2f
regCP_GFX_RS64_GP2_LO0 = 0x2a2c
regCP_GFX_RS64_GP2_LO1 = 0x2a2d
regCP_GFX_RS64_GP3_HI0 = 0x2a32
regCP_GFX_RS64_GP3_HI1 = 0x2a33
regCP_GFX_RS64_GP3_LO0 = 0x2a30
regCP_GFX_RS64_GP3_LO1 = 0x2a31
regCP_GFX_RS64_GP4_HI0 = 0x2a36
regCP_GFX_RS64_GP4_HI1 = 0x2a37
regCP_GFX_RS64_GP4_LO0 = 0x2a34
regCP_GFX_RS64_GP4_LO1 = 0x2a35
regCP_GFX_RS64_GP5_HI0 = 0x2a3a
regCP_GFX_RS64_GP5_HI1 = 0x2a3b
regCP_GFX_RS64_GP5_LO0 = 0x2a38
regCP_GFX_RS64_GP5_LO1 = 0x2a39
regCP_GFX_RS64_GP6_HI = 0x2a3d
regCP_GFX_RS64_GP6_LO = 0x2a3c
regCP_GFX_RS64_GP7_HI = 0x2a3f
regCP_GFX_RS64_GP7_LO = 0x2a3e
regCP_GFX_RS64_GP8_HI = 0x2a41
regCP_GFX_RS64_GP8_LO = 0x2a40
regCP_GFX_RS64_GP9_HI = 0x2a43
regCP_GFX_RS64_GP9_LO = 0x2a42
regCP_GFX_RS64_INSTR_PNTR0 = 0x2a44
regCP_GFX_RS64_INSTR_PNTR1 = 0x2a45
regCP_GFX_RS64_INTERRUPT0 = 0x2a01
regCP_GFX_RS64_INTERRUPT1 = 0x2aac
regCP_GFX_RS64_INTR_EN0 = 0x2a02
regCP_GFX_RS64_INTR_EN1 = 0x2a03
regCP_GFX_RS64_LOCAL_APERTURE = 0x2a0e
regCP_GFX_RS64_LOCAL_BASE0_HI = 0x2a0b
regCP_GFX_RS64_LOCAL_BASE0_LO = 0x2a0a
regCP_GFX_RS64_LOCAL_INSTR_APERTURE = 0x2a13
regCP_GFX_RS64_LOCAL_INSTR_BASE_HI = 0x2a10
regCP_GFX_RS64_LOCAL_INSTR_BASE_LO = 0x2a0f
regCP_GFX_RS64_LOCAL_INSTR_MASK_HI = 0x2a12
regCP_GFX_RS64_LOCAL_INSTR_MASK_LO = 0x2a11
regCP_GFX_RS64_LOCAL_MASK0_HI = 0x2a0d
regCP_GFX_RS64_LOCAL_MASK0_LO = 0x2a0c
regCP_GFX_RS64_LOCAL_SCRATCH_APERTURE = 0x2a14
regCP_GFX_RS64_LOCAL_SCRATCH_BASE_HI = 0x2a16
regCP_GFX_RS64_LOCAL_SCRATCH_BASE_LO = 0x2a15
regCP_GFX_RS64_MIBOUND_HI = 0x586d
regCP_GFX_RS64_MIBOUND_LO = 0x586c
regCP_GFX_RS64_MIP_HI0 = 0x2a1e
regCP_GFX_RS64_MIP_HI1 = 0x2a1f
regCP_GFX_RS64_MIP_LO0 = 0x2a1c
regCP_GFX_RS64_MIP_LO1 = 0x2a1d
regCP_GFX_RS64_MTIMECMP_HI0 = 0x2a22
regCP_GFX_RS64_MTIMECMP_HI1 = 0x2a23
regCP_GFX_RS64_MTIMECMP_LO0 = 0x2a20
regCP_GFX_RS64_MTIMECMP_LO1 = 0x2a21
regCP_GFX_RS64_PENDING_INTERRUPT0 = 0x2a46
regCP_GFX_RS64_PENDING_INTERRUPT1 = 0x2a47
regCP_GFX_RS64_PERFCOUNT_CNTL0 = 0x2a1a
regCP_GFX_RS64_PERFCOUNT_CNTL1 = 0x2a1b
regCP_GPU_TIMESTAMP_OFFSET_HI = 0x1f4d
regCP_GPU_TIMESTAMP_OFFSET_LO = 0x1f4c
regCP_GRBM_FREE_COUNT = 0xf43
regCP_HPD_MES_ROQ_OFFSETS = 0x1821
regCP_HPD_ROQ_OFFSETS = 0x1821
regCP_HPD_STATUS0 = 0x1822
regCP_HPD_UTCL1_CNTL = 0x1fa3
regCP_HPD_UTCL1_ERROR = 0x1fa7
regCP_HPD_UTCL1_ERROR_ADDR = 0x1fa8
regCP_HQD_ACTIVE = 0x1fab
regCP_HQD_AQL_CONTROL = 0x1fde
regCP_HQD_ATOMIC0_PREOP_HI = 0x1fc6
regCP_HQD_ATOMIC0_PREOP_LO = 0x1fc5
regCP_HQD_ATOMIC1_PREOP_HI = 0x1fc8
regCP_HQD_ATOMIC1_PREOP_LO = 0x1fc7
regCP_HQD_CNTL_STACK_OFFSET = 0x1fd7
regCP_HQD_CNTL_STACK_SIZE = 0x1fd8
regCP_HQD_CTX_SAVE_BASE_ADDR_HI = 0x1fd5
regCP_HQD_CTX_SAVE_BASE_ADDR_LO = 0x1fd4
regCP_HQD_CTX_SAVE_CONTROL = 0x1fd6
regCP_HQD_CTX_SAVE_SIZE = 0x1fda
regCP_HQD_DDID_DELTA_RPT_COUNT = 0x1fe7
regCP_HQD_DDID_INFLIGHT_COUNT = 0x1fe6
regCP_HQD_DDID_RPTR = 0x1fe4
regCP_HQD_DDID_WPTR = 0x1fe5
regCP_HQD_DEQUEUE_REQUEST = 0x1fc1
regCP_HQD_DEQUEUE_STATUS = 0x1fe8
regCP_HQD_DMA_OFFLOAD = 0x1fc2
regCP_HQD_EOP_BASE_ADDR = 0x1fce
regCP_HQD_EOP_BASE_ADDR_HI = 0x1fcf
regCP_HQD_EOP_CONTROL = 0x1fd0
regCP_HQD_EOP_EVENTS = 0x1fd3
regCP_HQD_EOP_RPTR = 0x1fd1
regCP_HQD_EOP_WPTR = 0x1fd2
regCP_HQD_EOP_WPTR_MEM = 0x1fdd
regCP_HQD_ERROR = 0x1fdc
regCP_HQD_GDS_RESOURCE_STATE = 0x1fdb
regCP_HQD_GFX_CONTROL = 0x1e9f
regCP_HQD_GFX_STATUS = 0x1ea0
regCP_HQD_HQ_CONTROL0 = 0x1fca
regCP_HQD_HQ_CONTROL1 = 0x1fcd
regCP_HQD_HQ_SCHEDULER0 = 0x1fc9
regCP_HQD_HQ_SCHEDULER1 = 0x1fca
regCP_HQD_HQ_STATUS0 = 0x1fc9
regCP_HQD_HQ_STATUS1 = 0x1fcc
regCP_HQD_IB_BASE_ADDR = 0x1fbb
regCP_HQD_IB_BASE_ADDR_HI = 0x1fbc
regCP_HQD_IB_CONTROL = 0x1fbe
regCP_HQD_IB_RPTR = 0x1fbd
regCP_HQD_IQ_RPTR = 0x1fc0
regCP_HQD_IQ_TIMER = 0x1fbf
regCP_HQD_MSG_TYPE = 0x1fc4
regCP_HQD_OFFLOAD = 0x1fc2
regCP_HQD_PERSISTENT_STATE = 0x1fad
regCP_HQD_PIPE_PRIORITY = 0x1fae
regCP_HQD_PQ_BASE = 0x1fb1
regCP_HQD_PQ_BASE_HI = 0x1fb2
regCP_HQD_PQ_CONTROL = 0x1fba
regCP_HQD_PQ_DOORBELL_CONTROL = 0x1fb8
regCP_HQD_PQ_RPTR = 0x1fb3
regCP_HQD_PQ_RPTR_REPORT_ADDR = 0x1fb4
regCP_HQD_PQ_RPTR_REPORT_ADDR_HI = 0x1fb5
regCP_HQD_PQ_WPTR_HI = 0x1fe0
regCP_HQD_PQ_WPTR_LO = 0x1fdf
regCP_HQD_PQ_WPTR_POLL_ADDR = 0x1fb6
regCP_HQD_PQ_WPTR_POLL_ADDR_HI = 0x1fb7
regCP_HQD_QUANTUM = 0x1fb0
regCP_HQD_QUEUE_PRIORITY = 0x1faf
regCP_HQD_SEMA_CMD = 0x1fc3
regCP_HQD_SUSPEND_CNTL_STACK_DW_CNT = 0x1fe2
regCP_HQD_SUSPEND_CNTL_STACK_OFFSET = 0x1fe1
regCP_HQD_SUSPEND_WG_STATE_OFFSET = 0x1fe3
regCP_HQD_VMID = 0x1fac
regCP_HQD_WG_STATE_OFFSET = 0x1fd9
regCP_HYP_MEC1_UCODE_ADDR = 0x581a
regCP_HYP_MEC1_UCODE_DATA = 0x581b
regCP_HYP_MEC2_UCODE_ADDR = 0x581c
regCP_HYP_MEC2_UCODE_DATA = 0x581d
regCP_HYP_ME_UCODE_ADDR = 0x5816
regCP_HYP_ME_UCODE_DATA = 0x5817
regCP_HYP_PFP_UCODE_ADDR = 0x5814
regCP_HYP_PFP_UCODE_DATA = 0x5815
regCP_IB2_BASE_HI = 0x20d0
regCP_IB2_BASE_LO = 0x20cf
regCP_IB2_BUFSZ = 0x20d1
regCP_IB2_CMD_BUFSZ = 0x20c1
regCP_IB2_OFFSET = 0x2093
regCP_IB2_PREAMBLE_BEGIN = 0x2096
regCP_IB2_PREAMBLE_END = 0x2097
regCP_INDEX_BASE_ADDR = 0x20f8
regCP_INDEX_BASE_ADDR_HI = 0x20f9
regCP_INDEX_TYPE = 0x20fa
regCP_INT_CNTL = 0x1de9
regCP_INT_CNTL_RING0 = 0x1e0a
regCP_INT_CNTL_RING1 = 0x1e0b
regCP_INT_STATUS = 0x1dea
regCP_INT_STATUS_RING0 = 0x1e0d
regCP_INT_STATUS_RING1 = 0x1e0e
regCP_IQ_WAIT_TIME1 = 0x1e4f
regCP_IQ_WAIT_TIME2 = 0x1e50
regCP_IQ_WAIT_TIME3 = 0x1e6a
regCP_MAX_CONTEXT = 0x1e4e
regCP_MAX_DRAW_COUNT = 0x1e5c
regCP_ME0_PIPE0_PRIORITY = 0x1ded
regCP_ME0_PIPE0_VMID = 0x1df2
regCP_ME0_PIPE1_PRIORITY = 0x1dee
regCP_ME0_PIPE1_VMID = 0x1df3
regCP_ME0_PIPE_PRIORITY_CNTS = 0x1dec
regCP_ME1_PIPE0_INT_CNTL = 0x1e25
regCP_ME1_PIPE0_INT_STATUS = 0x1e2d
regCP_ME1_PIPE0_PRIORITY = 0x1e3a
regCP_ME1_PIPE1_INT_CNTL = 0x1e26
regCP_ME1_PIPE1_INT_STATUS = 0x1e2e
regCP_ME1_PIPE1_PRIORITY = 0x1e3b
regCP_ME1_PIPE2_INT_CNTL = 0x1e27
regCP_ME1_PIPE2_INT_STATUS = 0x1e2f
regCP_ME1_PIPE2_PRIORITY = 0x1e3c
regCP_ME1_PIPE3_INT_CNTL = 0x1e28
regCP_ME1_PIPE3_INT_STATUS = 0x1e30
regCP_ME1_PIPE3_PRIORITY = 0x1e3d
regCP_ME1_PIPE_PRIORITY_CNTS = 0x1e39
regCP_ME2_PIPE0_INT_CNTL = 0x1e29
regCP_ME2_PIPE0_INT_STATUS = 0x1e31
regCP_ME2_PIPE0_PRIORITY = 0x1e3f
regCP_ME2_PIPE1_INT_CNTL = 0x1e2a
regCP_ME2_PIPE1_INT_STATUS = 0x1e32
regCP_ME2_PIPE1_PRIORITY = 0x1e40
regCP_ME2_PIPE2_INT_CNTL = 0x1e2b
regCP_ME2_PIPE2_INT_STATUS = 0x1e33
regCP_ME2_PIPE2_PRIORITY = 0x1e41
regCP_ME2_PIPE3_INT_CNTL = 0x1e2c
regCP_ME2_PIPE3_INT_STATUS = 0x1e34
regCP_ME2_PIPE3_PRIORITY = 0x1e42
regCP_ME2_PIPE_PRIORITY_CNTS = 0x1e3e
regCP_MEC1_F32_INTERRUPT = 0x1e16
regCP_MEC1_F32_INT_DIS = 0x1e5d
regCP_MEC1_INSTR_PNTR = 0xf48
regCP_MEC1_INTR_ROUTINE_START = 0x1e4b
regCP_MEC1_PRGRM_CNTR_START = 0x1e46
regCP_MEC2_F32_INTERRUPT = 0x1e17
regCP_MEC2_F32_INT_DIS = 0x1e5e
regCP_MEC2_INSTR_PNTR = 0xf49
regCP_MEC2_INTR_ROUTINE_START = 0x1e4c
regCP_MEC2_PRGRM_CNTR_START = 0x1e47
regCP_MEC_CNTL = 0x802
regCP_MEC_DC_APERTURE0_BASE = 0x294a
regCP_MEC_DC_APERTURE0_CNTL = 0x294c
regCP_MEC_DC_APERTURE0_MASK = 0x294b
regCP_MEC_DC_APERTURE10_BASE = 0x2968
regCP_MEC_DC_APERTURE10_CNTL = 0x296a
regCP_MEC_DC_APERTURE10_MASK = 0x2969
regCP_MEC_DC_APERTURE11_BASE = 0x296b
regCP_MEC_DC_APERTURE11_CNTL = 0x296d
regCP_MEC_DC_APERTURE11_MASK = 0x296c
regCP_MEC_DC_APERTURE12_BASE = 0x296e
regCP_MEC_DC_APERTURE12_CNTL = 0x2970
regCP_MEC_DC_APERTURE12_MASK = 0x296f
regCP_MEC_DC_APERTURE13_BASE = 0x2971
regCP_MEC_DC_APERTURE13_CNTL = 0x2973
regCP_MEC_DC_APERTURE13_MASK = 0x2972
regCP_MEC_DC_APERTURE14_BASE = 0x2974
regCP_MEC_DC_APERTURE14_CNTL = 0x2976
regCP_MEC_DC_APERTURE14_MASK = 0x2975
regCP_MEC_DC_APERTURE15_BASE = 0x2977
regCP_MEC_DC_APERTURE15_CNTL = 0x2979
regCP_MEC_DC_APERTURE15_MASK = 0x2978
regCP_MEC_DC_APERTURE1_BASE = 0x294d
regCP_MEC_DC_APERTURE1_CNTL = 0x294f
regCP_MEC_DC_APERTURE1_MASK = 0x294e
regCP_MEC_DC_APERTURE2_BASE = 0x2950
regCP_MEC_DC_APERTURE2_CNTL = 0x2952
regCP_MEC_DC_APERTURE2_MASK = 0x2951
regCP_MEC_DC_APERTURE3_BASE = 0x2953
regCP_MEC_DC_APERTURE3_CNTL = 0x2955
regCP_MEC_DC_APERTURE3_MASK = 0x2954
regCP_MEC_DC_APERTURE4_BASE = 0x2956
regCP_MEC_DC_APERTURE4_CNTL = 0x2958
regCP_MEC_DC_APERTURE4_MASK = 0x2957
regCP_MEC_DC_APERTURE5_BASE = 0x2959
regCP_MEC_DC_APERTURE5_CNTL = 0x295b
regCP_MEC_DC_APERTURE5_MASK = 0x295a
regCP_MEC_DC_APERTURE6_BASE = 0x295c
regCP_MEC_DC_APERTURE6_CNTL = 0x295e
regCP_MEC_DC_APERTURE6_MASK = 0x295d
regCP_MEC_DC_APERTURE7_BASE = 0x295f
regCP_MEC_DC_APERTURE7_CNTL = 0x2961
regCP_MEC_DC_APERTURE7_MASK = 0x2960
regCP_MEC_DC_APERTURE8_BASE = 0x2962
regCP_MEC_DC_APERTURE8_CNTL = 0x2964
regCP_MEC_DC_APERTURE8_MASK = 0x2963
regCP_MEC_DC_APERTURE9_BASE = 0x2965
regCP_MEC_DC_APERTURE9_CNTL = 0x2967
regCP_MEC_DC_APERTURE9_MASK = 0x2966
regCP_MEC_DC_BASE_CNTL = 0x290b
regCP_MEC_DC_BASE_HI = 0x5871
regCP_MEC_DC_BASE_LO = 0x5870
regCP_MEC_DC_OP_CNTL = 0x290c
regCP_MEC_DM_INDEX_ADDR = 0x5c02
regCP_MEC_DM_INDEX_DATA = 0x5c03
regCP_MEC_DOORBELL_RANGE_LOWER = 0x1dfc
regCP_MEC_DOORBELL_RANGE_UPPER = 0x1dfd
regCP_MEC_GP0_HI = 0x2911
regCP_MEC_GP0_LO = 0x2910
regCP_MEC_GP1_HI = 0x2913
regCP_MEC_GP1_LO = 0x2912
regCP_MEC_GP2_HI = 0x2915
regCP_MEC_GP2_LO = 0x2914
regCP_MEC_GP3_HI = 0x2917
regCP_MEC_GP3_LO = 0x2916
regCP_MEC_GP4_HI = 0x2919
regCP_MEC_GP4_LO = 0x2918
regCP_MEC_GP5_HI = 0x291b
regCP_MEC_GP5_LO = 0x291a
regCP_MEC_GP6_HI = 0x291d
regCP_MEC_GP6_LO = 0x291c
regCP_MEC_GP7_HI = 0x291f
regCP_MEC_GP7_LO = 0x291e
regCP_MEC_GP8_HI = 0x2921
regCP_MEC_GP8_LO = 0x2920
regCP_MEC_GP9_HI = 0x2923
regCP_MEC_GP9_LO = 0x2922
regCP_MEC_ISA_CNTL = 0x2903
regCP_MEC_JT_STAT = 0x1ed5
regCP_MEC_LOCAL_APERTURE = 0x292b
regCP_MEC_LOCAL_BASE0_HI = 0x2928
regCP_MEC_LOCAL_BASE0_LO = 0x2927
regCP_MEC_LOCAL_INSTR_APERTURE = 0x2930
regCP_MEC_LOCAL_INSTR_BASE_HI = 0x292d
regCP_MEC_LOCAL_INSTR_BASE_LO = 0x292c
regCP_MEC_LOCAL_INSTR_MASK_HI = 0x292f
regCP_MEC_LOCAL_INSTR_MASK_LO = 0x292e
regCP_MEC_LOCAL_MASK0_HI = 0x292a
regCP_MEC_LOCAL_MASK0_LO = 0x2929
regCP_MEC_LOCAL_SCRATCH_APERTURE = 0x2931
regCP_MEC_LOCAL_SCRATCH_BASE_HI = 0x2933
regCP_MEC_LOCAL_SCRATCH_BASE_LO = 0x2932
regCP_MEC_MDBASE_HI = 0x5871
regCP_MEC_MDBASE_LO = 0x5870
regCP_MEC_MDBOUND_HI = 0x5875
regCP_MEC_MDBOUND_LO = 0x5874
regCP_MEC_ME1_HEADER_DUMP = 0xe2e
regCP_MEC_ME1_UCODE_ADDR = 0x581a
regCP_MEC_ME1_UCODE_DATA = 0x581b
regCP_MEC_ME2_HEADER_DUMP = 0xe2f
regCP_MEC_ME2_UCODE_ADDR = 0x581c
regCP_MEC_ME2_UCODE_DATA = 0x581d
regCP_MEC_MIBOUND_HI = 0x5873
regCP_MEC_MIBOUND_LO = 0x5872
regCP_MEC_MIE_HI = 0x2906
regCP_MEC_MIE_LO = 0x2905
regCP_MEC_MIP_HI = 0x290a
regCP_MEC_MIP_LO = 0x2909
regCP_MEC_MTIMECMP_HI = 0x290e
regCP_MEC_MTIMECMP_LO = 0x290d
regCP_MEC_MTVEC_HI = 0x2902
regCP_MEC_MTVEC_LO = 0x2901
regCP_MEC_RS64_CNTL = 0x2904
regCP_MEC_RS64_INSTR_PNTR = 0x2908
regCP_MEC_RS64_INTERRUPT = 0x2907
regCP_MEC_RS64_INTERRUPT_DATA_16 = 0x293a
regCP_MEC_RS64_INTERRUPT_DATA_17 = 0x293b
regCP_MEC_RS64_INTERRUPT_DATA_18 = 0x293c
regCP_MEC_RS64_INTERRUPT_DATA_19 = 0x293d
regCP_MEC_RS64_INTERRUPT_DATA_20 = 0x293e
regCP_MEC_RS64_INTERRUPT_DATA_21 = 0x293f
regCP_MEC_RS64_INTERRUPT_DATA_22 = 0x2940
regCP_MEC_RS64_INTERRUPT_DATA_23 = 0x2941
regCP_MEC_RS64_INTERRUPT_DATA_24 = 0x2942
regCP_MEC_RS64_INTERRUPT_DATA_25 = 0x2943
regCP_MEC_RS64_INTERRUPT_DATA_26 = 0x2944
regCP_MEC_RS64_INTERRUPT_DATA_27 = 0x2945
regCP_MEC_RS64_INTERRUPT_DATA_28 = 0x2946
regCP_MEC_RS64_INTERRUPT_DATA_29 = 0x2947
regCP_MEC_RS64_INTERRUPT_DATA_30 = 0x2948
regCP_MEC_RS64_INTERRUPT_DATA_31 = 0x2949
regCP_MEC_RS64_PENDING_INTERRUPT = 0x2935
regCP_MEC_RS64_PERFCOUNT_CNTL = 0x2934
regCP_MEC_RS64_PRGRM_CNTR_START = 0x2900
regCP_MEC_RS64_PRGRM_CNTR_START_HI = 0x2938
regCP_MEQ_AVAIL = 0xf7d
regCP_MEQ_STAT = 0xf85
regCP_MEQ_THRESHOLDS = 0xf79
regCP_MES_CNTL = 0x2807
regCP_MES_DC_APERTURE0_BASE = 0x28af
regCP_MES_DC_APERTURE0_CNTL = 0x28b1
regCP_MES_DC_APERTURE0_MASK = 0x28b0
regCP_MES_DC_APERTURE10_BASE = 0x28cd
regCP_MES_DC_APERTURE10_CNTL = 0x28cf
regCP_MES_DC_APERTURE10_MASK = 0x28ce
regCP_MES_DC_APERTURE11_BASE = 0x28d0
regCP_MES_DC_APERTURE11_CNTL = 0x28d2
regCP_MES_DC_APERTURE11_MASK = 0x28d1
regCP_MES_DC_APERTURE12_BASE = 0x28d3
regCP_MES_DC_APERTURE12_CNTL = 0x28d5
regCP_MES_DC_APERTURE12_MASK = 0x28d4
regCP_MES_DC_APERTURE13_BASE = 0x28d6
regCP_MES_DC_APERTURE13_CNTL = 0x28d8
regCP_MES_DC_APERTURE13_MASK = 0x28d7
regCP_MES_DC_APERTURE14_BASE = 0x28d9
regCP_MES_DC_APERTURE14_CNTL = 0x28db
regCP_MES_DC_APERTURE14_MASK = 0x28da
regCP_MES_DC_APERTURE15_BASE = 0x28dc
regCP_MES_DC_APERTURE15_CNTL = 0x28de
regCP_MES_DC_APERTURE15_MASK = 0x28dd
regCP_MES_DC_APERTURE1_BASE = 0x28b2
regCP_MES_DC_APERTURE1_CNTL = 0x28b4
regCP_MES_DC_APERTURE1_MASK = 0x28b3
regCP_MES_DC_APERTURE2_BASE = 0x28b5
regCP_MES_DC_APERTURE2_CNTL = 0x28b7
regCP_MES_DC_APERTURE2_MASK = 0x28b6
regCP_MES_DC_APERTURE3_BASE = 0x28b8
regCP_MES_DC_APERTURE3_CNTL = 0x28ba
regCP_MES_DC_APERTURE3_MASK = 0x28b9
regCP_MES_DC_APERTURE4_BASE = 0x28bb
regCP_MES_DC_APERTURE4_CNTL = 0x28bd
regCP_MES_DC_APERTURE4_MASK = 0x28bc
regCP_MES_DC_APERTURE5_BASE = 0x28be
regCP_MES_DC_APERTURE5_CNTL = 0x28c0
regCP_MES_DC_APERTURE5_MASK = 0x28bf
regCP_MES_DC_APERTURE6_BASE = 0x28c1
regCP_MES_DC_APERTURE6_CNTL = 0x28c3
regCP_MES_DC_APERTURE6_MASK = 0x28c2
regCP_MES_DC_APERTURE7_BASE = 0x28c4
regCP_MES_DC_APERTURE7_CNTL = 0x28c6
regCP_MES_DC_APERTURE7_MASK = 0x28c5
regCP_MES_DC_APERTURE8_BASE = 0x28c7
regCP_MES_DC_APERTURE8_CNTL = 0x28c9
regCP_MES_DC_APERTURE8_MASK = 0x28c8
regCP_MES_DC_APERTURE9_BASE = 0x28ca
regCP_MES_DC_APERTURE9_CNTL = 0x28cc
regCP_MES_DC_APERTURE9_MASK = 0x28cb
regCP_MES_DC_BASE_CNTL = 0x2836
regCP_MES_DC_BASE_HI = 0x5855
regCP_MES_DC_BASE_LO = 0x5854
regCP_MES_DC_OP_CNTL = 0x2837
regCP_MES_DM_INDEX_ADDR = 0x5c00
regCP_MES_DM_INDEX_DATA = 0x5c01
regCP_MES_DOORBELL_CONTROL1 = 0x283c
regCP_MES_DOORBELL_CONTROL2 = 0x283d
regCP_MES_DOORBELL_CONTROL3 = 0x283e
regCP_MES_DOORBELL_CONTROL4 = 0x283f
regCP_MES_DOORBELL_CONTROL5 = 0x2840
regCP_MES_DOORBELL_CONTROL6 = 0x2841
regCP_MES_GP0_HI = 0x2844
regCP_MES_GP0_LO = 0x2843
regCP_MES_GP1_HI = 0x2846
regCP_MES_GP1_LO = 0x2845
regCP_MES_GP2_HI = 0x2848
regCP_MES_GP2_LO = 0x2847
regCP_MES_GP3_HI = 0x284a
regCP_MES_GP3_LO = 0x2849
regCP_MES_GP4_HI = 0x284c
regCP_MES_GP4_LO = 0x284b
regCP_MES_GP5_HI = 0x284e
regCP_MES_GP5_LO = 0x284d
regCP_MES_GP6_HI = 0x2850
regCP_MES_GP6_LO = 0x284f
regCP_MES_GP7_HI = 0x2852
regCP_MES_GP7_LO = 0x2851
regCP_MES_GP8_HI = 0x2854
regCP_MES_GP8_LO = 0x2853
regCP_MES_GP9_HI = 0x2856
regCP_MES_GP9_LO = 0x2855
regCP_MES_HEADER_DUMP = 0x280d
regCP_MES_IC_BASE_CNTL = 0x5852
regCP_MES_IC_BASE_HI = 0x5851
regCP_MES_IC_BASE_LO = 0x5850
regCP_MES_IC_OP_CNTL = 0x2820
regCP_MES_INSTR_PNTR = 0x2813
regCP_MES_INTERRUPT = 0x2810
regCP_MES_INTERRUPT_DATA_16 = 0x289f
regCP_MES_INTERRUPT_DATA_17 = 0x28a0
regCP_MES_INTERRUPT_DATA_18 = 0x28a1
regCP_MES_INTERRUPT_DATA_19 = 0x28a2
regCP_MES_INTERRUPT_DATA_20 = 0x28a3
regCP_MES_INTERRUPT_DATA_21 = 0x28a4
regCP_MES_INTERRUPT_DATA_22 = 0x28a5
regCP_MES_INTERRUPT_DATA_23 = 0x28a6
regCP_MES_INTERRUPT_DATA_24 = 0x28a7
regCP_MES_INTERRUPT_DATA_25 = 0x28a8
regCP_MES_INTERRUPT_DATA_26 = 0x28a9
regCP_MES_INTERRUPT_DATA_27 = 0x28aa
regCP_MES_INTERRUPT_DATA_28 = 0x28ab
regCP_MES_INTERRUPT_DATA_29 = 0x28ac
regCP_MES_INTERRUPT_DATA_30 = 0x28ad
regCP_MES_INTERRUPT_DATA_31 = 0x28ae
regCP_MES_INTR_ROUTINE_START = 0x2801
regCP_MES_INTR_ROUTINE_START_HI = 0x2802
regCP_MES_LOCAL_APERTURE = 0x2887
regCP_MES_LOCAL_BASE0_HI = 0x2884
regCP_MES_LOCAL_BASE0_LO = 0x2883
regCP_MES_LOCAL_INSTR_APERTURE = 0x288c
regCP_MES_LOCAL_INSTR_BASE_HI = 0x2889
regCP_MES_LOCAL_INSTR_BASE_LO = 0x2888
regCP_MES_LOCAL_INSTR_MASK_HI = 0x288b
regCP_MES_LOCAL_INSTR_MASK_LO = 0x288a
regCP_MES_LOCAL_MASK0_HI = 0x2886
regCP_MES_LOCAL_MASK0_LO = 0x2885
regCP_MES_LOCAL_SCRATCH_APERTURE = 0x288d
regCP_MES_LOCAL_SCRATCH_BASE_HI = 0x288f
regCP_MES_LOCAL_SCRATCH_BASE_LO = 0x288e
regCP_MES_MARCHID_HI = 0x2831
regCP_MES_MARCHID_LO = 0x2830
regCP_MES_MBADADDR_HI = 0x281d
regCP_MES_MBADADDR_LO = 0x281c
regCP_MES_MCAUSE_HI = 0x281b
regCP_MES_MCAUSE_LO = 0x281a
regCP_MES_MCYCLE_HI = 0x2827
regCP_MES_MCYCLE_LO = 0x2826
regCP_MES_MDBASE_HI = 0x5855
regCP_MES_MDBASE_LO = 0x5854
regCP_MES_MDBOUND_HI = 0x585e
regCP_MES_MDBOUND_LO = 0x585d
regCP_MES_MEPC_HI = 0x2819
regCP_MES_MEPC_LO = 0x2818
regCP_MES_MHARTID_HI = 0x2835
regCP_MES_MHARTID_LO = 0x2834
regCP_MES_MIBASE_HI = 0x5851
regCP_MES_MIBASE_LO = 0x5850
regCP_MES_MIBOUND_HI = 0x585c
regCP_MES_MIBOUND_LO = 0x585b
regCP_MES_MIE_HI = 0x280f
regCP_MES_MIE_LO = 0x280e
regCP_MES_MIMPID_HI = 0x2833
regCP_MES_MIMPID_LO = 0x2832
regCP_MES_MINSTRET_HI = 0x282b
regCP_MES_MINSTRET_LO = 0x282a
regCP_MES_MIP_HI = 0x281f
regCP_MES_MIP_LO = 0x281e
regCP_MES_MISA_HI = 0x282d
regCP_MES_MISA_LO = 0x282c
regCP_MES_MSCRATCH_HI = 0x2814
regCP_MES_MSCRATCH_LO = 0x2815
regCP_MES_MSTATUS_HI = 0x2817
regCP_MES_MSTATUS_LO = 0x2816
regCP_MES_MTIMECMP_HI = 0x2839
regCP_MES_MTIMECMP_LO = 0x2838
regCP_MES_MTIME_HI = 0x2829
regCP_MES_MTIME_LO = 0x2828
regCP_MES_MTVEC_HI = 0x2802
regCP_MES_MTVEC_LO = 0x2801
regCP_MES_MVENDORID_HI = 0x282f
regCP_MES_MVENDORID_LO = 0x282e
regCP_MES_PENDING_INTERRUPT = 0x289a
regCP_MES_PERFCOUNT_CNTL = 0x2899
regCP_MES_PIPE0_PRIORITY = 0x2809
regCP_MES_PIPE1_PRIORITY = 0x280a
regCP_MES_PIPE2_PRIORITY = 0x280b
regCP_MES_PIPE3_PRIORITY = 0x280c
regCP_MES_PIPE_PRIORITY_CNTS = 0x2808
regCP_MES_PRGRM_CNTR_START = 0x2800
regCP_MES_PRGRM_CNTR_START_HI = 0x289d
regCP_MES_PROCESS_QUANTUM_PIPE0 = 0x283a
regCP_MES_PROCESS_QUANTUM_PIPE1 = 0x283b
regCP_MES_SCRATCH_DATA = 0x2812
regCP_MES_SCRATCH_INDEX = 0x2811
regCP_ME_ATOMIC_PREOP_HI = 0x205e
regCP_ME_ATOMIC_PREOP_LO = 0x205d
regCP_ME_CNTL = 0x803
regCP_ME_COHER_BASE = 0x2101
regCP_ME_COHER_BASE_HI = 0x2102
regCP_ME_COHER_CNTL = 0x20fe
regCP_ME_COHER_SIZE = 0x20ff
regCP_ME_COHER_SIZE_HI = 0x2100
regCP_ME_COHER_STATUS = 0x2103
regCP_ME_F32_INTERRUPT = 0x1e13
regCP_ME_GDS_ATOMIC0_PREOP_HI = 0x2060
regCP_ME_GDS_ATOMIC0_PREOP_LO = 0x205f
regCP_ME_GDS_ATOMIC1_PREOP_HI = 0x2062
regCP_ME_GDS_ATOMIC1_PREOP_LO = 0x2061
regCP_ME_HEADER_DUMP = 0xf41
regCP_ME_IC_BASE_CNTL = 0x5846
regCP_ME_IC_BASE_HI = 0x5845
regCP_ME_IC_BASE_LO = 0x5844
regCP_ME_IC_OP_CNTL = 0x5847
regCP_ME_INSTR_PNTR = 0xf46
regCP_ME_INTR_ROUTINE_START = 0x1e4a
regCP_ME_INTR_ROUTINE_START_HI = 0x1e7b
regCP_ME_MC_RADDR_HI = 0x206e
regCP_ME_MC_RADDR_LO = 0x206d
regCP_ME_MC_WADDR_HI = 0x206a
regCP_ME_MC_WADDR_LO = 0x2069
regCP_ME_MC_WDATA_HI = 0x206c
regCP_ME_MC_WDATA_LO = 0x206b
regCP_ME_PREEMPTION = 0xf59
regCP_ME_PRGRM_CNTR_START = 0x1e45
regCP_ME_PRGRM_CNTR_START_HI = 0x1e79
regCP_ME_RAM_DATA = 0x5817
regCP_ME_RAM_RADDR = 0x5816
regCP_ME_RAM_WADDR = 0x5816
regCP_ME_SDMA_CS = 0x1f50
regCP_MQD_BASE_ADDR = 0x1fa9
regCP_MQD_BASE_ADDR_HI = 0x1faa
regCP_MQD_CONTROL = 0x1fcb
regCP_PA_CINVOC_COUNT_HI = 0x2029
regCP_PA_CINVOC_COUNT_LO = 0x2028
regCP_PA_CPRIM_COUNT_HI = 0x202b
regCP_PA_CPRIM_COUNT_LO = 0x202a
regCP_PA_MSPRIM_COUNT_HI = 0x20a5
regCP_PA_MSPRIM_COUNT_LO = 0x20a4
regCP_PERFMON_CNTL = 0x3808
regCP_PERFMON_CNTX_CNTL = 0xd8
regCP_PFP_ATOMIC_PREOP_HI = 0x2053
regCP_PFP_ATOMIC_PREOP_LO = 0x2052
regCP_PFP_COMPLETION_STATUS = 0x20ec
regCP_PFP_F32_INTERRUPT = 0x1e14
regCP_PFP_GDS_ATOMIC0_PREOP_HI = 0x2055
regCP_PFP_GDS_ATOMIC0_PREOP_LO = 0x2054
regCP_PFP_GDS_ATOMIC1_PREOP_HI = 0x2057
regCP_PFP_GDS_ATOMIC1_PREOP_LO = 0x2056
regCP_PFP_HEADER_DUMP = 0xf42
regCP_PFP_IB_CONTROL = 0x208d
regCP_PFP_IC_BASE_CNTL = 0x5842
regCP_PFP_IC_BASE_HI = 0x5841
regCP_PFP_IC_BASE_LO = 0x5840
regCP_PFP_IC_OP_CNTL = 0x5843
regCP_PFP_INSTR_PNTR = 0xf45
regCP_PFP_INTR_ROUTINE_START = 0x1e49
regCP_PFP_INTR_ROUTINE_START_HI = 0x1e7a
regCP_PFP_JT_STAT = 0x1ed3
regCP_PFP_LOAD_CONTROL = 0x208e
regCP_PFP_METADATA_BASE_ADDR = 0x20f0
regCP_PFP_METADATA_BASE_ADDR_HI = 0x20f1
regCP_PFP_PRGRM_CNTR_START = 0x1e44
regCP_PFP_PRGRM_CNTR_START_HI = 0x1e59
regCP_PFP_SDMA_CS = 0x1f4f
regCP_PFP_UCODE_ADDR = 0x5814
regCP_PFP_UCODE_DATA = 0x5815
regCP_PIPEID = 0xd9
regCP_PIPE_STATS_ADDR_HI = 0x2019
regCP_PIPE_STATS_ADDR_LO = 0x2018
regCP_PIPE_STATS_CONTROL = 0x203d
regCP_PQ_STATUS = 0x1e58
regCP_PQ_WPTR_POLL_CNTL = 0x1e23
regCP_PQ_WPTR_POLL_CNTL1 = 0x1e24
regCP_PRED_NOT_VISIBLE = 0x20ee
regCP_PRIV_VIOLATION_ADDR = 0xf9a
regCP_PROCESS_QUANTUM = 0x1df9
regCP_PWR_CNTL = 0x1e18
regCP_RB0_ACTIVE = 0x1f40
regCP_RB0_BASE = 0x1de0
regCP_RB0_BASE_HI = 0x1e51
regCP_RB0_BUFSZ_MASK = 0x1de5
regCP_RB0_CNTL = 0x1de1
regCP_RB0_RPTR = 0xf60
regCP_RB0_RPTR_ADDR = 0x1de3
regCP_RB0_RPTR_ADDR_HI = 0x1de4
regCP_RB0_WPTR = 0x1df4
regCP_RB0_WPTR_HI = 0x1df5
regCP_RB1_ACTIVE = 0x1f41
regCP_RB1_BASE = 0x1e00
regCP_RB1_BASE_HI = 0x1e52
regCP_RB1_BUFSZ_MASK = 0x1e04
regCP_RB1_CNTL = 0x1e01
regCP_RB1_RPTR = 0xf5f
regCP_RB1_RPTR_ADDR = 0x1e02
regCP_RB1_RPTR_ADDR_HI = 0x1e03
regCP_RB1_WPTR = 0x1df6
regCP_RB1_WPTR_HI = 0x1df7
regCP_RB_ACTIVE = 0x1f40
regCP_RB_BASE = 0x1de0
regCP_RB_BUFSZ_MASK = 0x1de5
regCP_RB_CNTL = 0x1de1
regCP_RB_DOORBELL_CLEAR = 0x1f28
regCP_RB_DOORBELL_CONTROL = 0x1e8d
regCP_RB_DOORBELL_RANGE_LOWER = 0x1dfa
regCP_RB_DOORBELL_RANGE_UPPER = 0x1dfb
regCP_RB_OFFSET = 0x2091
regCP_RB_RPTR = 0xf60
regCP_RB_RPTR_ADDR = 0x1de3
regCP_RB_RPTR_ADDR_HI = 0x1de4
regCP_RB_RPTR_WR = 0x1de2
regCP_RB_STATUS = 0x1f43
regCP_RB_VMID = 0x1df1
regCP_RB_WPTR = 0x1df4
regCP_RB_WPTR_DELAY = 0xf61
regCP_RB_WPTR_HI = 0x1df5
regCP_RB_WPTR_POLL_ADDR_HI = 0x1e8c
regCP_RB_WPTR_POLL_ADDR_LO = 0x1e8b
regCP_RB_WPTR_POLL_CNTL = 0xf62
regCP_RING0_PRIORITY = 0x1ded
regCP_RING1_PRIORITY = 0x1dee
regCP_RINGID = 0xd9
regCP_RING_PRIORITY_CNTS = 0x1dec
regCP_ROQ1_THRESHOLDS = 0xf75
regCP_ROQ2_AVAIL = 0xf7c
regCP_ROQ2_THRESHOLDS = 0xf76
regCP_ROQ3_THRESHOLDS = 0xf8c
regCP_ROQ_AVAIL = 0xf7a
regCP_ROQ_DB_STAT = 0xf8d
regCP_ROQ_IB1_STAT = 0xf81
regCP_ROQ_IB2_STAT = 0xf82
regCP_ROQ_RB_STAT = 0xf80
regCP_SAMPLE_STATUS = 0x20fd
regCP_SCRATCH_DATA = 0x2090
regCP_SCRATCH_INDEX = 0x208f
regCP_SC_PSINVOC_COUNT0_HI = 0x202d
regCP_SC_PSINVOC_COUNT0_LO = 0x202c
regCP_SC_PSINVOC_COUNT1_HI = 0x202f
regCP_SC_PSINVOC_COUNT1_LO = 0x202e
regCP_SDMA_DMA_DONE = 0x1f4e
regCP_SD_CNTL = 0x1f57
regCP_SEM_WAIT_TIMER = 0x206f
regCP_SIG_SEM_ADDR_HI = 0x2071
regCP_SIG_SEM_ADDR_LO = 0x2070
regCP_SOFT_RESET_CNTL = 0x1f59
regCP_STALLED_STAT1 = 0xf3d
regCP_STALLED_STAT2 = 0xf3e
regCP_STALLED_STAT3 = 0xf3c
regCP_STAT = 0xf40
regCP_STQ_AVAIL = 0xf7b
regCP_STQ_STAT = 0xf83
regCP_STQ_THRESHOLDS = 0xf77
regCP_STQ_WR_STAT = 0xf84
regCP_ST_BASE_HI = 0x20d3
regCP_ST_BASE_LO = 0x20d2
regCP_ST_BUFSZ = 0x20d4
regCP_ST_CMD_BUFSZ = 0x20c2
regCP_SUSPEND_CNTL = 0x1e69
regCP_SUSPEND_RESUME_REQ = 0x1e68
regCP_VGT_ASINVOC_COUNT_HI = 0x2033
regCP_VGT_ASINVOC_COUNT_LO = 0x2032
regCP_VGT_CSINVOC_COUNT_HI = 0x2031
regCP_VGT_CSINVOC_COUNT_LO = 0x2030
regCP_VGT_DSINVOC_COUNT_HI = 0x2027
regCP_VGT_DSINVOC_COUNT_LO = 0x2026
regCP_VGT_GSINVOC_COUNT_HI = 0x2023
regCP_VGT_GSINVOC_COUNT_LO = 0x2022
regCP_VGT_GSPRIM_COUNT_HI = 0x201f
regCP_VGT_GSPRIM_COUNT_LO = 0x201e
regCP_VGT_HSINVOC_COUNT_HI = 0x2025
regCP_VGT_HSINVOC_COUNT_LO = 0x2024
regCP_VGT_IAPRIM_COUNT_HI = 0x201d
regCP_VGT_IAPRIM_COUNT_LO = 0x201c
regCP_VGT_IAVERT_COUNT_HI = 0x201b
regCP_VGT_IAVERT_COUNT_LO = 0x201a
regCP_VGT_VSINVOC_COUNT_HI = 0x2021
regCP_VGT_VSINVOC_COUNT_LO = 0x2020
regCP_VIRT_STATUS = 0x1dd8
regCP_VMID = 0xda
regCP_VMID_PREEMPT = 0x1e56
regCP_VMID_RESET = 0x1e53
regCP_VMID_STATUS = 0x1e5f
regCP_WAIT_REG_MEM_TIMEOUT = 0x2074
regCP_WAIT_SEM_ADDR_HI = 0x2076
regCP_WAIT_SEM_ADDR_LO = 0x2075
regDB_ALPHA_TO_MASK = 0x2dc
regDB_CGTT_CLK_CTRL_0 = 0x50a4
regDB_COUNT_CONTROL = 0x1
regDB_CREDIT_LIMIT = 0x13b4
regDB_DEBUG = 0x13ac
regDB_DEBUG2 = 0x13ad
regDB_DEBUG3 = 0x13ae
regDB_DEBUG4 = 0x13af
regDB_DEBUG5 = 0x13d1
regDB_DEBUG6 = 0x13be
regDB_DEBUG7 = 0x13d0
regDB_DEPTH_BOUNDS_MAX = 0x9
regDB_DEPTH_BOUNDS_MIN = 0x8
regDB_DEPTH_CLEAR = 0xb
regDB_DEPTH_CONTROL = 0x200
regDB_DEPTH_SIZE_XY = 0x7
regDB_DEPTH_VIEW = 0x2
regDB_EQAA = 0x201
regDB_EQUAD_STUTTER_CONTROL = 0x13b2
regDB_ETILE_STUTTER_CONTROL = 0x13b0
regDB_EXCEPTION_CONTROL = 0x13bf
regDB_FGCG_INTERFACES_CLK_CTRL = 0x13d8
regDB_FGCG_SRAMS_CLK_CTRL = 0x13d7
regDB_FIFO_DEPTH1 = 0x13b8
regDB_FIFO_DEPTH2 = 0x13b9
regDB_FIFO_DEPTH3 = 0x13bd
regDB_FIFO_DEPTH4 = 0x13d9
regDB_FREE_CACHELINES = 0x13b7
regDB_HTILE_DATA_BASE = 0x5
regDB_HTILE_DATA_BASE_HI = 0x1e
regDB_HTILE_SURFACE = 0x2af
regDB_LAST_OF_BURST_CONFIG = 0x13ba
regDB_LQUAD_STUTTER_CONTROL = 0x13b3
regDB_LTILE_STUTTER_CONTROL = 0x13b1
regDB_MEM_ARB_WATERMARKS = 0x13bc
regDB_OCCLUSION_COUNT0_HI = 0x23c1
regDB_OCCLUSION_COUNT0_LOW = 0x23c0
regDB_OCCLUSION_COUNT1_HI = 0x23c3
regDB_OCCLUSION_COUNT1_LOW = 0x23c2
regDB_OCCLUSION_COUNT2_HI = 0x23c5
regDB_OCCLUSION_COUNT2_LOW = 0x23c4
regDB_OCCLUSION_COUNT3_HI = 0x23c7
regDB_OCCLUSION_COUNT3_LOW = 0x23c6
regDB_PERFCOUNTER0_HI = 0x3441
regDB_PERFCOUNTER0_LO = 0x3440
regDB_PERFCOUNTER0_SELECT = 0x3c40
regDB_PERFCOUNTER0_SELECT1 = 0x3c41
regDB_PERFCOUNTER1_HI = 0x3443
regDB_PERFCOUNTER1_LO = 0x3442
regDB_PERFCOUNTER1_SELECT = 0x3c42
regDB_PERFCOUNTER1_SELECT1 = 0x3c43
regDB_PERFCOUNTER2_HI = 0x3445
regDB_PERFCOUNTER2_LO = 0x3444
regDB_PERFCOUNTER2_SELECT = 0x3c44
regDB_PERFCOUNTER3_HI = 0x3447
regDB_PERFCOUNTER3_LO = 0x3446
regDB_PERFCOUNTER3_SELECT = 0x3c46
regDB_PRELOAD_CONTROL = 0x2b2
regDB_RENDER_CONTROL = 0x0
regDB_RENDER_OVERRIDE = 0x3
regDB_RENDER_OVERRIDE2 = 0x4
regDB_RESERVED_REG_1 = 0x16
regDB_RESERVED_REG_2 = 0xf
regDB_RESERVED_REG_3 = 0x17
regDB_RING_CONTROL = 0x13bb
regDB_RMI_L2_CACHE_CONTROL = 0x1f
regDB_SHADER_CONTROL = 0x203
regDB_SRESULTS_COMPARE_STATE0 = 0x2b0
regDB_SRESULTS_COMPARE_STATE1 = 0x2b1
regDB_STENCILREFMASK = 0x10c
regDB_STENCILREFMASK_BF = 0x10d
regDB_STENCIL_CLEAR = 0xa
regDB_STENCIL_CONTROL = 0x10b
regDB_STENCIL_INFO = 0x11
regDB_STENCIL_READ_BASE = 0x13
regDB_STENCIL_READ_BASE_HI = 0x1b
regDB_STENCIL_WRITE_BASE = 0x15
regDB_STENCIL_WRITE_BASE_HI = 0x1d
regDB_SUBTILE_CONTROL = 0x13b6
regDB_WATERMARKS = 0x13b5
regDB_Z_INFO = 0x10
regDB_Z_READ_BASE = 0x12
regDB_Z_READ_BASE_HI = 0x1a
regDB_Z_WRITE_BASE = 0x14
regDB_Z_WRITE_BASE_HI = 0x1c
regDIDT_EDC_CTRL = 0x1901
regDIDT_EDC_DYNAMIC_THRESHOLD_RO = 0x1909
regDIDT_EDC_OVERFLOW = 0x190a
regDIDT_EDC_ROLLING_POWER_DELTA = 0x190b
regDIDT_EDC_STALL_PATTERN_1_2 = 0x1904
regDIDT_EDC_STALL_PATTERN_3_4 = 0x1905
regDIDT_EDC_STALL_PATTERN_5_6 = 0x1906
regDIDT_EDC_STALL_PATTERN_7 = 0x1907
regDIDT_EDC_STATUS = 0x1908
regDIDT_EDC_THRESHOLD = 0x1903
regDIDT_EDC_THROTTLE_CTRL = 0x1902
regDIDT_INDEX_AUTO_INCR_EN = 0x1900
regDIDT_IND_DATA = 0x190d
regDIDT_IND_INDEX = 0x190c
regDIDT_STALL_PATTERN_1_2 = 0x1aff
regDIDT_STALL_PATTERN_3_4 = 0x1b00
regDIDT_STALL_PATTERN_5_6 = 0x1b01
regDIDT_STALL_PATTERN_7 = 0x1b02
regDIDT_STALL_PATTERN_CTRL = 0x1afe
regEDC_HYSTERESIS_CNTL = 0x1af1
regEDC_HYSTERESIS_STAT = 0x1b0e
regEDC_PERF_COUNTER = 0x1b0b
regEDC_STRETCH_NUM_PERF_COUNTER = 0x1b06
regEDC_STRETCH_PERF_COUNTER = 0x1b04
regEDC_UNSTRETCH_PERF_COUNTER = 0x1b05
regGB_ADDR_CONFIG = 0x13de
regGB_ADDR_CONFIG_READ = 0x13e2
regGB_BACKEND_MAP = 0x13df
regGB_EDC_MODE = 0x1e1e
regGB_GPU_ID = 0x13e0
regGCEA_DRAM_PAGE_BURST = 0x17aa
regGCEA_DRAM_RD_CAM_CNTL = 0x17a8
regGCEA_DRAM_RD_CLI2GRP_MAP0 = 0x17a0
regGCEA_DRAM_RD_CLI2GRP_MAP1 = 0x17a1
regGCEA_DRAM_RD_GRP2VC_MAP = 0x17a4
regGCEA_DRAM_RD_LAZY = 0x17a6
regGCEA_DRAM_RD_PRI_AGE = 0x17ab
regGCEA_DRAM_RD_PRI_FIXED = 0x17af
regGCEA_DRAM_RD_PRI_QUANT_PRI1 = 0x17b3
regGCEA_DRAM_RD_PRI_QUANT_PRI2 = 0x17b4
regGCEA_DRAM_RD_PRI_QUANT_PRI3 = 0x17b5
regGCEA_DRAM_RD_PRI_QUEUING = 0x17ad
regGCEA_DRAM_RD_PRI_URGENCY = 0x17b1
regGCEA_DRAM_WR_CAM_CNTL = 0x17a9
regGCEA_DRAM_WR_CLI2GRP_MAP0 = 0x17a2
regGCEA_DRAM_WR_CLI2GRP_MAP1 = 0x17a3
regGCEA_DRAM_WR_GRP2VC_MAP = 0x17a5
regGCEA_DRAM_WR_LAZY = 0x17a7
regGCEA_DRAM_WR_PRI_AGE = 0x17ac
regGCEA_DRAM_WR_PRI_FIXED = 0x17b0
regGCEA_DRAM_WR_PRI_QUANT_PRI1 = 0x17b6
regGCEA_DRAM_WR_PRI_QUANT_PRI2 = 0x17b7
regGCEA_DRAM_WR_PRI_QUANT_PRI3 = 0x17b8
regGCEA_DRAM_WR_PRI_QUEUING = 0x17ae
regGCEA_DRAM_WR_PRI_URGENCY = 0x17b2
regGCEA_DSM_CNTL = 0x14b4
regGCEA_DSM_CNTL2 = 0x14b7
regGCEA_DSM_CNTL2A = 0x14b8
regGCEA_DSM_CNTL2B = 0x14b9
regGCEA_DSM_CNTLA = 0x14b5
regGCEA_DSM_CNTLB = 0x14b6
regGCEA_EDC_CNT = 0x14b2
regGCEA_EDC_CNT2 = 0x14b3
regGCEA_EDC_CNT3 = 0x151a
regGCEA_ERR_STATUS = 0x14be
regGCEA_GL2C_XBR_CREDITS = 0x14ba
regGCEA_GL2C_XBR_MAXBURST = 0x14bb
regGCEA_ICG_CTRL = 0x50c4
regGCEA_IO_GROUP_BURST = 0x1883
regGCEA_IO_RD_CLI2GRP_MAP0 = 0x187d
regGCEA_IO_RD_CLI2GRP_MAP1 = 0x187e
regGCEA_IO_RD_COMBINE_FLUSH = 0x1881
regGCEA_IO_RD_PRI_AGE = 0x1884
regGCEA_IO_RD_PRI_FIXED = 0x1888
regGCEA_IO_RD_PRI_QUANT_PRI1 = 0x188e
regGCEA_IO_RD_PRI_QUANT_PRI2 = 0x188f
regGCEA_IO_RD_PRI_QUANT_PRI3 = 0x1890
regGCEA_IO_RD_PRI_QUEUING = 0x1886
regGCEA_IO_RD_PRI_URGENCY = 0x188a
regGCEA_IO_RD_PRI_URGENCY_MASKING = 0x188c
regGCEA_IO_WR_CLI2GRP_MAP0 = 0x187f
regGCEA_IO_WR_CLI2GRP_MAP1 = 0x1880
regGCEA_IO_WR_COMBINE_FLUSH = 0x1882
regGCEA_IO_WR_PRI_AGE = 0x1885
regGCEA_IO_WR_PRI_FIXED = 0x1889
regGCEA_IO_WR_PRI_QUANT_PRI1 = 0x1891
regGCEA_IO_WR_PRI_QUANT_PRI2 = 0x1892
regGCEA_IO_WR_PRI_QUANT_PRI3 = 0x1893
regGCEA_IO_WR_PRI_QUEUING = 0x1887
regGCEA_IO_WR_PRI_URGENCY = 0x188b
regGCEA_IO_WR_PRI_URGENCY_MASKING = 0x188d
regGCEA_LATENCY_SAMPLING = 0x14a3
regGCEA_MAM_CTRL = 0x14ab
regGCEA_MAM_CTRL2 = 0x14a9
regGCEA_MISC = 0x14a2
regGCEA_MISC2 = 0x14bf
regGCEA_PERFCOUNTER0_CFG = 0x3a03
regGCEA_PERFCOUNTER1_CFG = 0x3a04
regGCEA_PERFCOUNTER2_HI = 0x3261
regGCEA_PERFCOUNTER2_LO = 0x3260
regGCEA_PERFCOUNTER2_MODE = 0x3a02
regGCEA_PERFCOUNTER2_SELECT = 0x3a00
regGCEA_PERFCOUNTER2_SELECT1 = 0x3a01
regGCEA_PERFCOUNTER_HI = 0x3263
regGCEA_PERFCOUNTER_LO = 0x3262
regGCEA_PERFCOUNTER_RSLT_CNTL = 0x3a05
regGCEA_PROBE_CNTL = 0x14bc
regGCEA_PROBE_MAP = 0x14bd
regGCEA_RRET_MEM_RESERVE = 0x1518
regGCEA_SDP_ARB_FINAL = 0x1896
regGCEA_SDP_CREDITS = 0x189a
regGCEA_SDP_ENABLE = 0x151e
regGCEA_SDP_IO_PRIORITY = 0x1899
regGCEA_SDP_TAG_RESERVE0 = 0x189b
regGCEA_SDP_TAG_RESERVE1 = 0x189c
regGCEA_SDP_VCC_RESERVE0 = 0x189d
regGCEA_SDP_VCC_RESERVE1 = 0x189e
regGCMC_MEM_POWER_LS = 0x15ac
regGCMC_VM_AGP_BASE = 0x167c
regGCMC_VM_AGP_BOT = 0x167b
regGCMC_VM_AGP_TOP = 0x167a
regGCMC_VM_APT_CNTL = 0x15b1
regGCMC_VM_CACHEABLE_DRAM_ADDRESS_END = 0x15ae
regGCMC_VM_CACHEABLE_DRAM_ADDRESS_START = 0x15ad
regGCMC_VM_FB_LOCATION_BASE = 0x1678
regGCMC_VM_FB_LOCATION_TOP = 0x1679
regGCMC_VM_FB_NOALLOC_CNTL = 0x15b8
regGCMC_VM_FB_OFFSET = 0x15a7
regGCMC_VM_FB_SIZE_OFFSET_VF0 = 0x5a80
regGCMC_VM_FB_SIZE_OFFSET_VF1 = 0x5a81
regGCMC_VM_FB_SIZE_OFFSET_VF10 = 0x5a8a
regGCMC_VM_FB_SIZE_OFFSET_VF11 = 0x5a8b
regGCMC_VM_FB_SIZE_OFFSET_VF12 = 0x5a8c
regGCMC_VM_FB_SIZE_OFFSET_VF13 = 0x5a8d
regGCMC_VM_FB_SIZE_OFFSET_VF14 = 0x5a8e
regGCMC_VM_FB_SIZE_OFFSET_VF15 = 0x5a8f
regGCMC_VM_FB_SIZE_OFFSET_VF2 = 0x5a82
regGCMC_VM_FB_SIZE_OFFSET_VF3 = 0x5a83
regGCMC_VM_FB_SIZE_OFFSET_VF4 = 0x5a84
regGCMC_VM_FB_SIZE_OFFSET_VF5 = 0x5a85
regGCMC_VM_FB_SIZE_OFFSET_VF6 = 0x5a86
regGCMC_VM_FB_SIZE_OFFSET_VF7 = 0x5a87
regGCMC_VM_FB_SIZE_OFFSET_VF8 = 0x5a88
regGCMC_VM_FB_SIZE_OFFSET_VF9 = 0x5a89
regGCMC_VM_L2_PERFCOUNTER0_CFG = 0x3d30
regGCMC_VM_L2_PERFCOUNTER1_CFG = 0x3d31
regGCMC_VM_L2_PERFCOUNTER2_CFG = 0x3d32
regGCMC_VM_L2_PERFCOUNTER3_CFG = 0x3d33
regGCMC_VM_L2_PERFCOUNTER4_CFG = 0x3d34
regGCMC_VM_L2_PERFCOUNTER5_CFG = 0x3d35
regGCMC_VM_L2_PERFCOUNTER6_CFG = 0x3d36
regGCMC_VM_L2_PERFCOUNTER7_CFG = 0x3d37
regGCMC_VM_L2_PERFCOUNTER_HI = 0x34e5
regGCMC_VM_L2_PERFCOUNTER_LO = 0x34e4
regGCMC_VM_L2_PERFCOUNTER_RSLT_CNTL = 0x3d38
regGCMC_VM_LOCAL_FB_ADDRESS_END = 0x15b3
regGCMC_VM_LOCAL_FB_ADDRESS_LOCK_CNTL = 0x15b4
regGCMC_VM_LOCAL_FB_ADDRESS_START = 0x15b2
regGCMC_VM_LOCAL_SYSMEM_ADDRESS_END = 0x15b0
regGCMC_VM_LOCAL_SYSMEM_ADDRESS_START = 0x15af
regGCMC_VM_MARC_BASE_HI_0 = 0x5e58
regGCMC_VM_MARC_BASE_HI_1 = 0x5e59
regGCMC_VM_MARC_BASE_HI_10 = 0x5e62
regGCMC_VM_MARC_BASE_HI_11 = 0x5e63
regGCMC_VM_MARC_BASE_HI_12 = 0x5e64
regGCMC_VM_MARC_BASE_HI_13 = 0x5e65
regGCMC_VM_MARC_BASE_HI_14 = 0x5e66
regGCMC_VM_MARC_BASE_HI_15 = 0x5e67
regGCMC_VM_MARC_BASE_HI_2 = 0x5e5a
regGCMC_VM_MARC_BASE_HI_3 = 0x5e5b
regGCMC_VM_MARC_BASE_HI_4 = 0x5e5c
regGCMC_VM_MARC_BASE_HI_5 = 0x5e5d
regGCMC_VM_MARC_BASE_HI_6 = 0x5e5e
regGCMC_VM_MARC_BASE_HI_7 = 0x5e5f
regGCMC_VM_MARC_BASE_HI_8 = 0x5e60
regGCMC_VM_MARC_BASE_HI_9 = 0x5e61
regGCMC_VM_MARC_BASE_LO_0 = 0x5e48
regGCMC_VM_MARC_BASE_LO_1 = 0x5e49
regGCMC_VM_MARC_BASE_LO_10 = 0x5e52
regGCMC_VM_MARC_BASE_LO_11 = 0x5e53
regGCMC_VM_MARC_BASE_LO_12 = 0x5e54
regGCMC_VM_MARC_BASE_LO_13 = 0x5e55
regGCMC_VM_MARC_BASE_LO_14 = 0x5e56
regGCMC_VM_MARC_BASE_LO_15 = 0x5e57
regGCMC_VM_MARC_BASE_LO_2 = 0x5e4a
regGCMC_VM_MARC_BASE_LO_3 = 0x5e4b
regGCMC_VM_MARC_BASE_LO_4 = 0x5e4c
regGCMC_VM_MARC_BASE_LO_5 = 0x5e4d
regGCMC_VM_MARC_BASE_LO_6 = 0x5e4e
regGCMC_VM_MARC_BASE_LO_7 = 0x5e4f
regGCMC_VM_MARC_BASE_LO_8 = 0x5e50
regGCMC_VM_MARC_BASE_LO_9 = 0x5e51
regGCMC_VM_MARC_LEN_HI_0 = 0x5e98
regGCMC_VM_MARC_LEN_HI_1 = 0x5e99
regGCMC_VM_MARC_LEN_HI_10 = 0x5ea2
regGCMC_VM_MARC_LEN_HI_11 = 0x5ea3
regGCMC_VM_MARC_LEN_HI_12 = 0x5ea4
regGCMC_VM_MARC_LEN_HI_13 = 0x5ea5
regGCMC_VM_MARC_LEN_HI_14 = 0x5ea6
regGCMC_VM_MARC_LEN_HI_15 = 0x5ea7
regGCMC_VM_MARC_LEN_HI_2 = 0x5e9a
regGCMC_VM_MARC_LEN_HI_3 = 0x5e9b
regGCMC_VM_MARC_LEN_HI_4 = 0x5e9c
regGCMC_VM_MARC_LEN_HI_5 = 0x5e9d
regGCMC_VM_MARC_LEN_HI_6 = 0x5e9e
regGCMC_VM_MARC_LEN_HI_7 = 0x5e9f
regGCMC_VM_MARC_LEN_HI_8 = 0x5ea0
regGCMC_VM_MARC_LEN_HI_9 = 0x5ea1
regGCMC_VM_MARC_LEN_LO_0 = 0x5e88
regGCMC_VM_MARC_LEN_LO_1 = 0x5e89
regGCMC_VM_MARC_LEN_LO_10 = 0x5e92
regGCMC_VM_MARC_LEN_LO_11 = 0x5e93
regGCMC_VM_MARC_LEN_LO_12 = 0x5e94
regGCMC_VM_MARC_LEN_LO_13 = 0x5e95
regGCMC_VM_MARC_LEN_LO_14 = 0x5e96
regGCMC_VM_MARC_LEN_LO_15 = 0x5e97
regGCMC_VM_MARC_LEN_LO_2 = 0x5e8a
regGCMC_VM_MARC_LEN_LO_3 = 0x5e8b
regGCMC_VM_MARC_LEN_LO_4 = 0x5e8c
regGCMC_VM_MARC_LEN_LO_5 = 0x5e8d
regGCMC_VM_MARC_LEN_LO_6 = 0x5e8e
regGCMC_VM_MARC_LEN_LO_7 = 0x5e8f
regGCMC_VM_MARC_LEN_LO_8 = 0x5e90
regGCMC_VM_MARC_LEN_LO_9 = 0x5e91
regGCMC_VM_MARC_PFVF_MAPPING_0 = 0x5ea8
regGCMC_VM_MARC_PFVF_MAPPING_1 = 0x5ea9
regGCMC_VM_MARC_PFVF_MAPPING_10 = 0x5eb2
regGCMC_VM_MARC_PFVF_MAPPING_11 = 0x5eb3
regGCMC_VM_MARC_PFVF_MAPPING_12 = 0x5eb4
regGCMC_VM_MARC_PFVF_MAPPING_13 = 0x5eb5
regGCMC_VM_MARC_PFVF_MAPPING_14 = 0x5eb6
regGCMC_VM_MARC_PFVF_MAPPING_15 = 0x5eb7
regGCMC_VM_MARC_PFVF_MAPPING_2 = 0x5eaa
regGCMC_VM_MARC_PFVF_MAPPING_3 = 0x5eab
regGCMC_VM_MARC_PFVF_MAPPING_4 = 0x5eac
regGCMC_VM_MARC_PFVF_MAPPING_5 = 0x5ead
regGCMC_VM_MARC_PFVF_MAPPING_6 = 0x5eae
regGCMC_VM_MARC_PFVF_MAPPING_7 = 0x5eaf
regGCMC_VM_MARC_PFVF_MAPPING_8 = 0x5eb0
regGCMC_VM_MARC_PFVF_MAPPING_9 = 0x5eb1
regGCMC_VM_MARC_RELOC_HI_0 = 0x5e78
regGCMC_VM_MARC_RELOC_HI_1 = 0x5e79
regGCMC_VM_MARC_RELOC_HI_10 = 0x5e82
regGCMC_VM_MARC_RELOC_HI_11 = 0x5e83
regGCMC_VM_MARC_RELOC_HI_12 = 0x5e84
regGCMC_VM_MARC_RELOC_HI_13 = 0x5e85
regGCMC_VM_MARC_RELOC_HI_14 = 0x5e86
regGCMC_VM_MARC_RELOC_HI_15 = 0x5e87
regGCMC_VM_MARC_RELOC_HI_2 = 0x5e7a
regGCMC_VM_MARC_RELOC_HI_3 = 0x5e7b
regGCMC_VM_MARC_RELOC_HI_4 = 0x5e7c
regGCMC_VM_MARC_RELOC_HI_5 = 0x5e7d
regGCMC_VM_MARC_RELOC_HI_6 = 0x5e7e
regGCMC_VM_MARC_RELOC_HI_7 = 0x5e7f
regGCMC_VM_MARC_RELOC_HI_8 = 0x5e80
regGCMC_VM_MARC_RELOC_HI_9 = 0x5e81
regGCMC_VM_MARC_RELOC_LO_0 = 0x5e68
regGCMC_VM_MARC_RELOC_LO_1 = 0x5e69
regGCMC_VM_MARC_RELOC_LO_10 = 0x5e72
regGCMC_VM_MARC_RELOC_LO_11 = 0x5e73
regGCMC_VM_MARC_RELOC_LO_12 = 0x5e74
regGCMC_VM_MARC_RELOC_LO_13 = 0x5e75
regGCMC_VM_MARC_RELOC_LO_14 = 0x5e76
regGCMC_VM_MARC_RELOC_LO_15 = 0x5e77
regGCMC_VM_MARC_RELOC_LO_2 = 0x5e6a
regGCMC_VM_MARC_RELOC_LO_3 = 0x5e6b
regGCMC_VM_MARC_RELOC_LO_4 = 0x5e6c
regGCMC_VM_MARC_RELOC_LO_5 = 0x5e6d
regGCMC_VM_MARC_RELOC_LO_6 = 0x5e6e
regGCMC_VM_MARC_RELOC_LO_7 = 0x5e6f
regGCMC_VM_MARC_RELOC_LO_8 = 0x5e70
regGCMC_VM_MARC_RELOC_LO_9 = 0x5e71
regGCMC_VM_MX_L1_TLB_CNTL = 0x167f
regGCMC_VM_NB_LOWER_TOP_OF_DRAM2 = 0x15a5
regGCMC_VM_NB_TOP_OF_DRAM_SLOT1 = 0x15a4
regGCMC_VM_NB_UPPER_TOP_OF_DRAM2 = 0x15a6
regGCMC_VM_STEERING = 0x15aa
regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB = 0x15a8
regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB = 0x15a9
regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR = 0x167e
regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR = 0x167d
regGCRD_CREDIT_SAFE = 0x198a
regGCRD_SA0_TARGETS_DISABLE = 0x1987
regGCRD_SA1_TARGETS_DISABLE = 0x1989
regGCR_CMD_STATUS = 0x1992
regGCR_GENERAL_CNTL = 0x1990
regGCR_PERFCOUNTER0_HI = 0x3521
regGCR_PERFCOUNTER0_LO = 0x3520
regGCR_PERFCOUNTER0_SELECT = 0x3d60
regGCR_PERFCOUNTER0_SELECT1 = 0x3d61
regGCR_PERFCOUNTER1_HI = 0x3523
regGCR_PERFCOUNTER1_LO = 0x3522
regGCR_PERFCOUNTER1_SELECT = 0x3d62
regGCR_PIO_CNTL = 0x1580
regGCR_PIO_DATA = 0x1581
regGCR_SPARE = 0x1993
regGCUTCL2_CGTT_BUSY_CTRL = 0x15b7
regGCUTCL2_CREDIT_SAFETY_GROUP_CLIENTS_INVREQ_CDC = 0x15eb
regGCUTCL2_CREDIT_SAFETY_GROUP_CLIENTS_INVREQ_NOCDC = 0x15ec
regGCUTCL2_CREDIT_SAFETY_GROUP_RET_CDC = 0x15ea
regGCUTCL2_GROUP_RET_FAULT_STATUS = 0x15bb
regGCUTCL2_HARVEST_BYPASS_GROUPS = 0x15b9
regGCUTCL2_ICG_CTRL = 0x15b5
regGCUTCL2_PERFCOUNTER0_CFG = 0x3d39
regGCUTCL2_PERFCOUNTER1_CFG = 0x3d3a
regGCUTCL2_PERFCOUNTER2_CFG = 0x3d3b
regGCUTCL2_PERFCOUNTER3_CFG = 0x3d3c
regGCUTCL2_PERFCOUNTER_HI = 0x34e7
regGCUTCL2_PERFCOUNTER_LO = 0x34e6
regGCUTCL2_PERFCOUNTER_RSLT_CNTL = 0x3d3d
regGCUTCL2_TRANSLATION_BYPASS_BY_VMID = 0x5e41
regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_CNTL = 0x5e44
regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_REQUEST_HI = 0x15e6
regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_REQUEST_LO = 0x15e5
regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_RESPONSE_HI = 0x15e8
regGCUTC_GPUVA_VMID_TRANSLATION_ASSIST_RESPONSE_LO = 0x15e7
regGCUTC_TRANSLATION_FAULT_CNTL0 = 0x5eb8
regGCUTC_TRANSLATION_FAULT_CNTL1 = 0x5eb9
regGCVML2_CREDIT_SAFETY_IH_FAULT_INTERRUPT = 0x15ed
regGCVML2_PERFCOUNTER2_0_HI = 0x34e2
regGCVML2_PERFCOUNTER2_0_LO = 0x34e0
regGCVML2_PERFCOUNTER2_0_MODE = 0x3d24
regGCVML2_PERFCOUNTER2_0_SELECT = 0x3d20
regGCVML2_PERFCOUNTER2_0_SELECT1 = 0x3d22
regGCVML2_PERFCOUNTER2_1_HI = 0x34e3
regGCVML2_PERFCOUNTER2_1_LO = 0x34e1
regGCVML2_PERFCOUNTER2_1_MODE = 0x3d25
regGCVML2_PERFCOUNTER2_1_SELECT = 0x3d21
regGCVML2_PERFCOUNTER2_1_SELECT1 = 0x3d23
regGCVML2_WALKER_CREDIT_SAFETY_FETCH_RDREQ = 0x15ee
regGCVML2_WALKER_MACRO_THROTTLE_FETCH_LIMIT = 0x15dd
regGCVML2_WALKER_MACRO_THROTTLE_TIME = 0x15dc
regGCVML2_WALKER_MICRO_THROTTLE_FETCH_LIMIT = 0x15df
regGCVML2_WALKER_MICRO_THROTTLE_TIME = 0x15de
regGCVM_CONTEXT0_CNTL = 0x1688
regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 = 0x16f4
regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 = 0x16f3
regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32 = 0x1734
regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32 = 0x1733
regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32 = 0x1714
regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32 = 0x1713
regGCVM_CONTEXT10_CNTL = 0x1692
regGCVM_CONTEXT10_PAGE_TABLE_BASE_ADDR_HI32 = 0x1708
regGCVM_CONTEXT10_PAGE_TABLE_BASE_ADDR_LO32 = 0x1707
regGCVM_CONTEXT10_PAGE_TABLE_END_ADDR_HI32 = 0x1748
regGCVM_CONTEXT10_PAGE_TABLE_END_ADDR_LO32 = 0x1747
regGCVM_CONTEXT10_PAGE_TABLE_START_ADDR_HI32 = 0x1728
regGCVM_CONTEXT10_PAGE_TABLE_START_ADDR_LO32 = 0x1727
regGCVM_CONTEXT11_CNTL = 0x1693
regGCVM_CONTEXT11_PAGE_TABLE_BASE_ADDR_HI32 = 0x170a
regGCVM_CONTEXT11_PAGE_TABLE_BASE_ADDR_LO32 = 0x1709
regGCVM_CONTEXT11_PAGE_TABLE_END_ADDR_HI32 = 0x174a
regGCVM_CONTEXT11_PAGE_TABLE_END_ADDR_LO32 = 0x1749
regGCVM_CONTEXT11_PAGE_TABLE_START_ADDR_HI32 = 0x172a
regGCVM_CONTEXT11_PAGE_TABLE_START_ADDR_LO32 = 0x1729
regGCVM_CONTEXT12_CNTL = 0x1694
regGCVM_CONTEXT12_PAGE_TABLE_BASE_ADDR_HI32 = 0x170c
regGCVM_CONTEXT12_PAGE_TABLE_BASE_ADDR_LO32 = 0x170b
regGCVM_CONTEXT12_PAGE_TABLE_END_ADDR_HI32 = 0x174c
regGCVM_CONTEXT12_PAGE_TABLE_END_ADDR_LO32 = 0x174b
regGCVM_CONTEXT12_PAGE_TABLE_START_ADDR_HI32 = 0x172c
regGCVM_CONTEXT12_PAGE_TABLE_START_ADDR_LO32 = 0x172b
regGCVM_CONTEXT13_CNTL = 0x1695
regGCVM_CONTEXT13_PAGE_TABLE_BASE_ADDR_HI32 = 0x170e
regGCVM_CONTEXT13_PAGE_TABLE_BASE_ADDR_LO32 = 0x170d
regGCVM_CONTEXT13_PAGE_TABLE_END_ADDR_HI32 = 0x174e
regGCVM_CONTEXT13_PAGE_TABLE_END_ADDR_LO32 = 0x174d
regGCVM_CONTEXT13_PAGE_TABLE_START_ADDR_HI32 = 0x172e
regGCVM_CONTEXT13_PAGE_TABLE_START_ADDR_LO32 = 0x172d
regGCVM_CONTEXT14_CNTL = 0x1696
regGCVM_CONTEXT14_PAGE_TABLE_BASE_ADDR_HI32 = 0x1710
regGCVM_CONTEXT14_PAGE_TABLE_BASE_ADDR_LO32 = 0x170f
regGCVM_CONTEXT14_PAGE_TABLE_END_ADDR_HI32 = 0x1750
regGCVM_CONTEXT14_PAGE_TABLE_END_ADDR_LO32 = 0x174f
regGCVM_CONTEXT14_PAGE_TABLE_START_ADDR_HI32 = 0x1730
regGCVM_CONTEXT14_PAGE_TABLE_START_ADDR_LO32 = 0x172f
regGCVM_CONTEXT15_CNTL = 0x1697
regGCVM_CONTEXT15_PAGE_TABLE_BASE_ADDR_HI32 = 0x1712
regGCVM_CONTEXT15_PAGE_TABLE_BASE_ADDR_LO32 = 0x1711
regGCVM_CONTEXT15_PAGE_TABLE_END_ADDR_HI32 = 0x1752
regGCVM_CONTEXT15_PAGE_TABLE_END_ADDR_LO32 = 0x1751
regGCVM_CONTEXT15_PAGE_TABLE_START_ADDR_HI32 = 0x1732
regGCVM_CONTEXT15_PAGE_TABLE_START_ADDR_LO32 = 0x1731
regGCVM_CONTEXT1_CNTL = 0x1689
regGCVM_CONTEXT1_PAGE_TABLE_BASE_ADDR_HI32 = 0x16f6
regGCVM_CONTEXT1_PAGE_TABLE_BASE_ADDR_LO32 = 0x16f5
regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_HI32 = 0x1736
regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 = 0x1735
regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_HI32 = 0x1716
regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 = 0x1715
regGCVM_CONTEXT2_CNTL = 0x168a
regGCVM_CONTEXT2_PAGE_TABLE_BASE_ADDR_HI32 = 0x16f8
regGCVM_CONTEXT2_PAGE_TABLE_BASE_ADDR_LO32 = 0x16f7
regGCVM_CONTEXT2_PAGE_TABLE_END_ADDR_HI32 = 0x1738
regGCVM_CONTEXT2_PAGE_TABLE_END_ADDR_LO32 = 0x1737
regGCVM_CONTEXT2_PAGE_TABLE_START_ADDR_HI32 = 0x1718
regGCVM_CONTEXT2_PAGE_TABLE_START_ADDR_LO32 = 0x1717
regGCVM_CONTEXT3_CNTL = 0x168b
regGCVM_CONTEXT3_PAGE_TABLE_BASE_ADDR_HI32 = 0x16fa
regGCVM_CONTEXT3_PAGE_TABLE_BASE_ADDR_LO32 = 0x16f9
regGCVM_CONTEXT3_PAGE_TABLE_END_ADDR_HI32 = 0x173a
regGCVM_CONTEXT3_PAGE_TABLE_END_ADDR_LO32 = 0x1739
regGCVM_CONTEXT3_PAGE_TABLE_START_ADDR_HI32 = 0x171a
regGCVM_CONTEXT3_PAGE_TABLE_START_ADDR_LO32 = 0x1719
regGCVM_CONTEXT4_CNTL = 0x168c
regGCVM_CONTEXT4_PAGE_TABLE_BASE_ADDR_HI32 = 0x16fc
regGCVM_CONTEXT4_PAGE_TABLE_BASE_ADDR_LO32 = 0x16fb
regGCVM_CONTEXT4_PAGE_TABLE_END_ADDR_HI32 = 0x173c
regGCVM_CONTEXT4_PAGE_TABLE_END_ADDR_LO32 = 0x173b
regGCVM_CONTEXT4_PAGE_TABLE_START_ADDR_HI32 = 0x171c
regGCVM_CONTEXT4_PAGE_TABLE_START_ADDR_LO32 = 0x171b
regGCVM_CONTEXT5_CNTL = 0x168d
regGCVM_CONTEXT5_PAGE_TABLE_BASE_ADDR_HI32 = 0x16fe
regGCVM_CONTEXT5_PAGE_TABLE_BASE_ADDR_LO32 = 0x16fd
regGCVM_CONTEXT5_PAGE_TABLE_END_ADDR_HI32 = 0x173e
regGCVM_CONTEXT5_PAGE_TABLE_END_ADDR_LO32 = 0x173d
regGCVM_CONTEXT5_PAGE_TABLE_START_ADDR_HI32 = 0x171e
regGCVM_CONTEXT5_PAGE_TABLE_START_ADDR_LO32 = 0x171d
regGCVM_CONTEXT6_CNTL = 0x168e
regGCVM_CONTEXT6_PAGE_TABLE_BASE_ADDR_HI32 = 0x1700
regGCVM_CONTEXT6_PAGE_TABLE_BASE_ADDR_LO32 = 0x16ff
regGCVM_CONTEXT6_PAGE_TABLE_END_ADDR_HI32 = 0x1740
regGCVM_CONTEXT6_PAGE_TABLE_END_ADDR_LO32 = 0x173f
regGCVM_CONTEXT6_PAGE_TABLE_START_ADDR_HI32 = 0x1720
regGCVM_CONTEXT6_PAGE_TABLE_START_ADDR_LO32 = 0x171f
regGCVM_CONTEXT7_CNTL = 0x168f
regGCVM_CONTEXT7_PAGE_TABLE_BASE_ADDR_HI32 = 0x1702
regGCVM_CONTEXT7_PAGE_TABLE_BASE_ADDR_LO32 = 0x1701
regGCVM_CONTEXT7_PAGE_TABLE_END_ADDR_HI32 = 0x1742
regGCVM_CONTEXT7_PAGE_TABLE_END_ADDR_LO32 = 0x1741
regGCVM_CONTEXT7_PAGE_TABLE_START_ADDR_HI32 = 0x1722
regGCVM_CONTEXT7_PAGE_TABLE_START_ADDR_LO32 = 0x1721
regGCVM_CONTEXT8_CNTL = 0x1690
regGCVM_CONTEXT8_PAGE_TABLE_BASE_ADDR_HI32 = 0x1704
regGCVM_CONTEXT8_PAGE_TABLE_BASE_ADDR_LO32 = 0x1703
regGCVM_CONTEXT8_PAGE_TABLE_END_ADDR_HI32 = 0x1744
regGCVM_CONTEXT8_PAGE_TABLE_END_ADDR_LO32 = 0x1743
regGCVM_CONTEXT8_PAGE_TABLE_START_ADDR_HI32 = 0x1724
regGCVM_CONTEXT8_PAGE_TABLE_START_ADDR_LO32 = 0x1723
regGCVM_CONTEXT9_CNTL = 0x1691
regGCVM_CONTEXT9_PAGE_TABLE_BASE_ADDR_HI32 = 0x1706
regGCVM_CONTEXT9_PAGE_TABLE_BASE_ADDR_LO32 = 0x1705
regGCVM_CONTEXT9_PAGE_TABLE_END_ADDR_HI32 = 0x1746
regGCVM_CONTEXT9_PAGE_TABLE_END_ADDR_LO32 = 0x1745
regGCVM_CONTEXT9_PAGE_TABLE_START_ADDR_HI32 = 0x1726
regGCVM_CONTEXT9_PAGE_TABLE_START_ADDR_LO32 = 0x1725
regGCVM_CONTEXTS_DISABLE = 0x1698
regGCVM_DUMMY_PAGE_FAULT_ADDR_HI32 = 0x15c2
regGCVM_DUMMY_PAGE_FAULT_ADDR_LO32 = 0x15c1
regGCVM_DUMMY_PAGE_FAULT_CNTL = 0x15c0
regGCVM_INVALIDATE_CNTL = 0x15c3
regGCVM_INVALIDATE_ENG0_ACK = 0x16bd
regGCVM_INVALIDATE_ENG0_ADDR_RANGE_HI32 = 0x16d0
regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 = 0x16cf
regGCVM_INVALIDATE_ENG0_REQ = 0x16ab
regGCVM_INVALIDATE_ENG0_SEM = 0x1699
regGCVM_INVALIDATE_ENG10_ACK = 0x16c7
regGCVM_INVALIDATE_ENG10_ADDR_RANGE_HI32 = 0x16e4
regGCVM_INVALIDATE_ENG10_ADDR_RANGE_LO32 = 0x16e3
regGCVM_INVALIDATE_ENG10_REQ = 0x16b5
regGCVM_INVALIDATE_ENG10_SEM = 0x16a3
regGCVM_INVALIDATE_ENG11_ACK = 0x16c8
regGCVM_INVALIDATE_ENG11_ADDR_RANGE_HI32 = 0x16e6
regGCVM_INVALIDATE_ENG11_ADDR_RANGE_LO32 = 0x16e5
regGCVM_INVALIDATE_ENG11_REQ = 0x16b6
regGCVM_INVALIDATE_ENG11_SEM = 0x16a4
regGCVM_INVALIDATE_ENG12_ACK = 0x16c9
regGCVM_INVALIDATE_ENG12_ADDR_RANGE_HI32 = 0x16e8
regGCVM_INVALIDATE_ENG12_ADDR_RANGE_LO32 = 0x16e7
regGCVM_INVALIDATE_ENG12_REQ = 0x16b7
regGCVM_INVALIDATE_ENG12_SEM = 0x16a5
regGCVM_INVALIDATE_ENG13_ACK = 0x16ca
regGCVM_INVALIDATE_ENG13_ADDR_RANGE_HI32 = 0x16ea
regGCVM_INVALIDATE_ENG13_ADDR_RANGE_LO32 = 0x16e9
regGCVM_INVALIDATE_ENG13_REQ = 0x16b8
regGCVM_INVALIDATE_ENG13_SEM = 0x16a6
regGCVM_INVALIDATE_ENG14_ACK = 0x16cb
regGCVM_INVALIDATE_ENG14_ADDR_RANGE_HI32 = 0x16ec
regGCVM_INVALIDATE_ENG14_ADDR_RANGE_LO32 = 0x16eb
regGCVM_INVALIDATE_ENG14_REQ = 0x16b9
regGCVM_INVALIDATE_ENG14_SEM = 0x16a7
regGCVM_INVALIDATE_ENG15_ACK = 0x16cc
regGCVM_INVALIDATE_ENG15_ADDR_RANGE_HI32 = 0x16ee
regGCVM_INVALIDATE_ENG15_ADDR_RANGE_LO32 = 0x16ed
regGCVM_INVALIDATE_ENG15_REQ = 0x16ba
regGCVM_INVALIDATE_ENG15_SEM = 0x16a8
regGCVM_INVALIDATE_ENG16_ACK = 0x16cd
regGCVM_INVALIDATE_ENG16_ADDR_RANGE_HI32 = 0x16f0
regGCVM_INVALIDATE_ENG16_ADDR_RANGE_LO32 = 0x16ef
regGCVM_INVALIDATE_ENG16_REQ = 0x16bb
regGCVM_INVALIDATE_ENG16_SEM = 0x16a9
regGCVM_INVALIDATE_ENG17_ACK = 0x16ce
regGCVM_INVALIDATE_ENG17_ADDR_RANGE_HI32 = 0x16f2
regGCVM_INVALIDATE_ENG17_ADDR_RANGE_LO32 = 0x16f1
regGCVM_INVALIDATE_ENG17_REQ = 0x16bc
regGCVM_INVALIDATE_ENG17_SEM = 0x16aa
regGCVM_INVALIDATE_ENG1_ACK = 0x16be
regGCVM_INVALIDATE_ENG1_ADDR_RANGE_HI32 = 0x16d2
regGCVM_INVALIDATE_ENG1_ADDR_RANGE_LO32 = 0x16d1
regGCVM_INVALIDATE_ENG1_REQ = 0x16ac
regGCVM_INVALIDATE_ENG1_SEM = 0x169a
regGCVM_INVALIDATE_ENG2_ACK = 0x16bf
regGCVM_INVALIDATE_ENG2_ADDR_RANGE_HI32 = 0x16d4
regGCVM_INVALIDATE_ENG2_ADDR_RANGE_LO32 = 0x16d3
regGCVM_INVALIDATE_ENG2_REQ = 0x16ad
regGCVM_INVALIDATE_ENG2_SEM = 0x169b
regGCVM_INVALIDATE_ENG3_ACK = 0x16c0
regGCVM_INVALIDATE_ENG3_ADDR_RANGE_HI32 = 0x16d6
regGCVM_INVALIDATE_ENG3_ADDR_RANGE_LO32 = 0x16d5
regGCVM_INVALIDATE_ENG3_REQ = 0x16ae
regGCVM_INVALIDATE_ENG3_SEM = 0x169c
regGCVM_INVALIDATE_ENG4_ACK = 0x16c1
regGCVM_INVALIDATE_ENG4_ADDR_RANGE_HI32 = 0x16d8
regGCVM_INVALIDATE_ENG4_ADDR_RANGE_LO32 = 0x16d7
regGCVM_INVALIDATE_ENG4_REQ = 0x16af
regGCVM_INVALIDATE_ENG4_SEM = 0x169d
regGCVM_INVALIDATE_ENG5_ACK = 0x16c2
regGCVM_INVALIDATE_ENG5_ADDR_RANGE_HI32 = 0x16da
regGCVM_INVALIDATE_ENG5_ADDR_RANGE_LO32 = 0x16d9
regGCVM_INVALIDATE_ENG5_REQ = 0x16b0
regGCVM_INVALIDATE_ENG5_SEM = 0x169e
regGCVM_INVALIDATE_ENG6_ACK = 0x16c3
regGCVM_INVALIDATE_ENG6_ADDR_RANGE_HI32 = 0x16dc
regGCVM_INVALIDATE_ENG6_ADDR_RANGE_LO32 = 0x16db
regGCVM_INVALIDATE_ENG6_REQ = 0x16b1
regGCVM_INVALIDATE_ENG6_SEM = 0x169f
regGCVM_INVALIDATE_ENG7_ACK = 0x16c4
regGCVM_INVALIDATE_ENG7_ADDR_RANGE_HI32 = 0x16de
regGCVM_INVALIDATE_ENG7_ADDR_RANGE_LO32 = 0x16dd
regGCVM_INVALIDATE_ENG7_REQ = 0x16b2
regGCVM_INVALIDATE_ENG7_SEM = 0x16a0
regGCVM_INVALIDATE_ENG8_ACK = 0x16c5
regGCVM_INVALIDATE_ENG8_ADDR_RANGE_HI32 = 0x16e0
regGCVM_INVALIDATE_ENG8_ADDR_RANGE_LO32 = 0x16df
regGCVM_INVALIDATE_ENG8_REQ = 0x16b3
regGCVM_INVALIDATE_ENG8_SEM = 0x16a1
regGCVM_INVALIDATE_ENG9_ACK = 0x16c6
regGCVM_INVALIDATE_ENG9_ADDR_RANGE_HI32 = 0x16e2
regGCVM_INVALIDATE_ENG9_ADDR_RANGE_LO32 = 0x16e1
regGCVM_INVALIDATE_ENG9_REQ = 0x16b4
regGCVM_INVALIDATE_ENG9_SEM = 0x16a2
regGCVM_L2_BANK_SELECT_MASKS = 0x15e9
regGCVM_L2_BANK_SELECT_RESERVED_CID = 0x15d6
regGCVM_L2_BANK_SELECT_RESERVED_CID2 = 0x15d7
regGCVM_L2_CACHE_PARITY_CNTL = 0x15d8
regGCVM_L2_CGTT_BUSY_CTRL = 0x15e0
regGCVM_L2_CNTL = 0x15bc
regGCVM_L2_CNTL2 = 0x15bd
regGCVM_L2_CNTL3 = 0x15be
regGCVM_L2_CNTL4 = 0x15d4
regGCVM_L2_CNTL5 = 0x15da
regGCVM_L2_CONTEXT0_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1754
regGCVM_L2_CONTEXT10_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x175e
regGCVM_L2_CONTEXT11_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x175f
regGCVM_L2_CONTEXT12_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1760
regGCVM_L2_CONTEXT13_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1761
regGCVM_L2_CONTEXT14_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1762
regGCVM_L2_CONTEXT15_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1763
regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32 = 0x15d1
regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32 = 0x15d0
regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32 = 0x15cf
regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32 = 0x15ce
regGCVM_L2_CONTEXT1_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1755
regGCVM_L2_CONTEXT2_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1756
regGCVM_L2_CONTEXT3_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1757
regGCVM_L2_CONTEXT4_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1758
regGCVM_L2_CONTEXT5_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1759
regGCVM_L2_CONTEXT6_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x175a
regGCVM_L2_CONTEXT7_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x175b
regGCVM_L2_CONTEXT8_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x175c
regGCVM_L2_CONTEXT9_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x175d
regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32 = 0x15d3
regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32 = 0x15d2
regGCVM_L2_GCR_CNTL = 0x15db
regGCVM_L2_ICG_CTRL = 0x15d9
regGCVM_L2_MM_GROUP_RT_CLASSES = 0x15d5
regGCVM_L2_PER_PFVF_PTE_CACHE_FRAGMENT_SIZES = 0x1753
regGCVM_L2_PROTECTION_FAULT_ADDR_HI32 = 0x15ca
regGCVM_L2_PROTECTION_FAULT_ADDR_LO32 = 0x15c9
regGCVM_L2_PROTECTION_FAULT_CNTL = 0x15c4
regGCVM_L2_PROTECTION_FAULT_CNTL2 = 0x15c5
regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32 = 0x15cc
regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32 = 0x15cb
regGCVM_L2_PROTECTION_FAULT_MM_CNTL3 = 0x15c6
regGCVM_L2_PROTECTION_FAULT_MM_CNTL4 = 0x15c7
regGCVM_L2_PROTECTION_FAULT_STATUS = 0x15c8
regGCVM_L2_PTE_CACHE_DUMP_CNTL = 0x15e1
regGCVM_L2_PTE_CACHE_DUMP_READ = 0x15e2
regGCVM_L2_STATUS = 0x15bf
regGC_CAC_AGGR_GFXCLK_CYCLE = 0x1ae4
regGC_CAC_AGGR_LOWER = 0x1ad2
regGC_CAC_AGGR_UPPER = 0x1ad3
regGC_CAC_CTRL_1 = 0x1ad0
regGC_CAC_CTRL_2 = 0x1ad1
regGC_CAC_IND_DATA = 0x1b59
regGC_CAC_IND_INDEX = 0x1b58
regGC_CAC_WEIGHT_CHC_0 = 0x1b3c
regGC_CAC_WEIGHT_CHC_1 = 0x1b3d
regGC_CAC_WEIGHT_CP_0 = 0x1b10
regGC_CAC_WEIGHT_CP_1 = 0x1b11
regGC_CAC_WEIGHT_EA_0 = 0x1b12
regGC_CAC_WEIGHT_EA_1 = 0x1b13
regGC_CAC_WEIGHT_EA_2 = 0x1b14
regGC_CAC_WEIGHT_GDS_0 = 0x1b20
regGC_CAC_WEIGHT_GDS_1 = 0x1b21
regGC_CAC_WEIGHT_GDS_2 = 0x1b22
regGC_CAC_WEIGHT_GE_0 = 0x1b23
regGC_CAC_WEIGHT_GE_1 = 0x1b24
regGC_CAC_WEIGHT_GE_2 = 0x1b25
regGC_CAC_WEIGHT_GE_3 = 0x1b26
regGC_CAC_WEIGHT_GE_4 = 0x1b27
regGC_CAC_WEIGHT_GE_5 = 0x1b28
regGC_CAC_WEIGHT_GE_6 = 0x1b29
regGC_CAC_WEIGHT_GL2C_0 = 0x1b2f
regGC_CAC_WEIGHT_GL2C_1 = 0x1b30
regGC_CAC_WEIGHT_GL2C_2 = 0x1b31
regGC_CAC_WEIGHT_GRBM_0 = 0x1b44
regGC_CAC_WEIGHT_GUS_0 = 0x1b3e
regGC_CAC_WEIGHT_GUS_1 = 0x1b3f
regGC_CAC_WEIGHT_PH_0 = 0x1b32
regGC_CAC_WEIGHT_PH_1 = 0x1b33
regGC_CAC_WEIGHT_PH_2 = 0x1b34
regGC_CAC_WEIGHT_PH_3 = 0x1b35
regGC_CAC_WEIGHT_PMM_0 = 0x1b2e
regGC_CAC_WEIGHT_RLC_0 = 0x1b40
regGC_CAC_WEIGHT_SDMA_0 = 0x1b36
regGC_CAC_WEIGHT_SDMA_1 = 0x1b37
regGC_CAC_WEIGHT_SDMA_2 = 0x1b38
regGC_CAC_WEIGHT_SDMA_3 = 0x1b39
regGC_CAC_WEIGHT_SDMA_4 = 0x1b3a
regGC_CAC_WEIGHT_SDMA_5 = 0x1b3b
regGC_CAC_WEIGHT_UTCL2_ROUTER_0 = 0x1b15
regGC_CAC_WEIGHT_UTCL2_ROUTER_1 = 0x1b16
regGC_CAC_WEIGHT_UTCL2_ROUTER_2 = 0x1b17
regGC_CAC_WEIGHT_UTCL2_ROUTER_3 = 0x1b18
regGC_CAC_WEIGHT_UTCL2_ROUTER_4 = 0x1b19
regGC_CAC_WEIGHT_UTCL2_VML2_0 = 0x1b1a
regGC_CAC_WEIGHT_UTCL2_VML2_1 = 0x1b1b
regGC_CAC_WEIGHT_UTCL2_VML2_2 = 0x1b1c
regGC_CAC_WEIGHT_UTCL2_WALKER_0 = 0x1b1d
regGC_CAC_WEIGHT_UTCL2_WALKER_1 = 0x1b1e
regGC_CAC_WEIGHT_UTCL2_WALKER_2 = 0x1b1f
regGC_EDC_CLK_MONITOR_CTRL = 0x1b56
regGC_EDC_CTRL = 0x1aed
regGC_EDC_OVERFLOW = 0x1b08
regGC_EDC_ROLLING_POWER_DELTA = 0x1b09
regGC_EDC_STATUS = 0x1b07
regGC_EDC_STRETCH_CTRL = 0x1aef
regGC_EDC_STRETCH_THRESHOLD = 0x1af0
regGC_EDC_THRESHOLD = 0x1aee
regGC_IH_COOKIE_0_PTR = 0x5a07
regGC_THROTTLE_CTRL = 0x1af2
regGC_THROTTLE_CTRL1 = 0x1af3
regGC_THROTTLE_STATUS = 0x1b0a
regGC_USER_PRIM_CONFIG = 0x5b91
regGC_USER_RB_BACKEND_DISABLE = 0x5b94
regGC_USER_RB_REDUNDANCY = 0x5b93
regGC_USER_RMI_REDUNDANCY = 0x5b95
regGC_USER_SA_UNIT_DISABLE = 0x5b92
regGC_USER_SHADER_ARRAY_CONFIG = 0x5b90
regGC_USER_SHADER_RATE_CONFIG = 0x5b97
regGDS_ATOM_BASE = 0x240c
regGDS_ATOM_CNTL = 0x240a
regGDS_ATOM_COMPLETE = 0x240b
regGDS_ATOM_DST = 0x2410
regGDS_ATOM_OFFSET0 = 0x240e
regGDS_ATOM_OFFSET1 = 0x240f
regGDS_ATOM_OP = 0x2411
regGDS_ATOM_READ0 = 0x2416
regGDS_ATOM_READ0_U = 0x2417
regGDS_ATOM_READ1 = 0x2418
regGDS_ATOM_READ1_U = 0x2419
regGDS_ATOM_SIZE = 0x240d
regGDS_ATOM_SRC0 = 0x2412
regGDS_ATOM_SRC0_U = 0x2413
regGDS_ATOM_SRC1 = 0x2414
regGDS_ATOM_SRC1_U = 0x2415
regGDS_CNTL_STATUS = 0x1361
regGDS_COMPUTE_MAX_WAVE_ID = 0x20e8
regGDS_CONFIG = 0x1360
regGDS_CS_CTXSW_CNT0 = 0x20ee
regGDS_CS_CTXSW_CNT1 = 0x20ef
regGDS_CS_CTXSW_CNT2 = 0x20f0
regGDS_CS_CTXSW_CNT3 = 0x20f1
regGDS_CS_CTXSW_STATUS = 0x20ed
regGDS_DSM_CNTL = 0x136a
regGDS_DSM_CNTL2 = 0x136d
regGDS_EDC_CNT = 0x1365
regGDS_EDC_GRBM_CNT = 0x1366
regGDS_EDC_OA_DED = 0x1367
regGDS_EDC_OA_PHY_CNT = 0x136b
regGDS_EDC_OA_PIPE_CNT = 0x136c
regGDS_ENHANCE = 0x1362
regGDS_ENHANCE2 = 0x19b0
regGDS_GFX_CTXSW_STATUS = 0x20f2
regGDS_GS_0 = 0x2426
regGDS_GS_1 = 0x2427
regGDS_GS_2 = 0x2428
regGDS_GS_3 = 0x2429
regGDS_GS_CTXSW_CNT0 = 0x2117
regGDS_GS_CTXSW_CNT1 = 0x2118
regGDS_GS_CTXSW_CNT2 = 0x2119
regGDS_GS_CTXSW_CNT3 = 0x211a
regGDS_GWS_RESET0 = 0x20e4
regGDS_GWS_RESET1 = 0x20e5
regGDS_GWS_RESOURCE = 0x241b
regGDS_GWS_RESOURCE_CNT = 0x241c
regGDS_GWS_RESOURCE_CNTL = 0x241a
regGDS_GWS_RESOURCE_RESET = 0x20e6
regGDS_GWS_VMID0 = 0x20c0
regGDS_GWS_VMID1 = 0x20c1
regGDS_GWS_VMID10 = 0x20ca
regGDS_GWS_VMID11 = 0x20cb
regGDS_GWS_VMID12 = 0x20cc
regGDS_GWS_VMID13 = 0x20cd
regGDS_GWS_VMID14 = 0x20ce
regGDS_GWS_VMID15 = 0x20cf
regGDS_GWS_VMID2 = 0x20c2
regGDS_GWS_VMID3 = 0x20c3
regGDS_GWS_VMID4 = 0x20c4
regGDS_GWS_VMID5 = 0x20c5
regGDS_GWS_VMID6 = 0x20c6
regGDS_GWS_VMID7 = 0x20c7
regGDS_GWS_VMID8 = 0x20c8
regGDS_GWS_VMID9 = 0x20c9
regGDS_MEMORY_CLEAN = 0x211f
regGDS_OA_ADDRESS = 0x241f
regGDS_OA_CGPG_RESTORE = 0x19b1
regGDS_OA_CNTL = 0x241d
regGDS_OA_COUNTER = 0x241e
regGDS_OA_INCDEC = 0x2420
regGDS_OA_RESET = 0x20ea
regGDS_OA_RESET_MASK = 0x20e9
regGDS_OA_RING_SIZE = 0x2421
regGDS_OA_VMID0 = 0x20d0
regGDS_OA_VMID1 = 0x20d1
regGDS_OA_VMID10 = 0x20da
regGDS_OA_VMID11 = 0x20db
regGDS_OA_VMID12 = 0x20dc
regGDS_OA_VMID13 = 0x20dd
regGDS_OA_VMID14 = 0x20de
regGDS_OA_VMID15 = 0x20df
regGDS_OA_VMID2 = 0x20d2
regGDS_OA_VMID3 = 0x20d3
regGDS_OA_VMID4 = 0x20d4
regGDS_OA_VMID5 = 0x20d5
regGDS_OA_VMID6 = 0x20d6
regGDS_OA_VMID7 = 0x20d7
regGDS_OA_VMID8 = 0x20d8
regGDS_OA_VMID9 = 0x20d9
regGDS_PERFCOUNTER0_HI = 0x3281
regGDS_PERFCOUNTER0_LO = 0x3280
regGDS_PERFCOUNTER0_SELECT = 0x3a80
regGDS_PERFCOUNTER0_SELECT1 = 0x3a84
regGDS_PERFCOUNTER1_HI = 0x3283
regGDS_PERFCOUNTER1_LO = 0x3282
regGDS_PERFCOUNTER1_SELECT = 0x3a81
regGDS_PERFCOUNTER1_SELECT1 = 0x3a85
regGDS_PERFCOUNTER2_HI = 0x3285
regGDS_PERFCOUNTER2_LO = 0x3284
regGDS_PERFCOUNTER2_SELECT = 0x3a82
regGDS_PERFCOUNTER2_SELECT1 = 0x3a86
regGDS_PERFCOUNTER3_HI = 0x3287
regGDS_PERFCOUNTER3_LO = 0x3286
regGDS_PERFCOUNTER3_SELECT = 0x3a83
regGDS_PERFCOUNTER3_SELECT1 = 0x3a87
regGDS_PROTECTION_FAULT = 0x1363
regGDS_PS_CTXSW_CNT0 = 0x20f7
regGDS_PS_CTXSW_CNT1 = 0x20f8
regGDS_PS_CTXSW_CNT2 = 0x20f9
regGDS_PS_CTXSW_CNT3 = 0x20fa
regGDS_PS_CTXSW_IDX = 0x20fb
regGDS_RD_ADDR = 0x2400
regGDS_RD_BURST_ADDR = 0x2402
regGDS_RD_BURST_COUNT = 0x2403
regGDS_RD_BURST_DATA = 0x2404
regGDS_RD_DATA = 0x2401
regGDS_STRMOUT_DWORDS_WRITTEN_0 = 0x2422
regGDS_STRMOUT_DWORDS_WRITTEN_1 = 0x2423
regGDS_STRMOUT_DWORDS_WRITTEN_2 = 0x2424
regGDS_STRMOUT_DWORDS_WRITTEN_3 = 0x2425
regGDS_STRMOUT_PRIMS_NEEDED_0_HI = 0x242b
regGDS_STRMOUT_PRIMS_NEEDED_0_LO = 0x242a
regGDS_STRMOUT_PRIMS_NEEDED_1_HI = 0x242f
regGDS_STRMOUT_PRIMS_NEEDED_1_LO = 0x242e
regGDS_STRMOUT_PRIMS_NEEDED_2_HI = 0x2433
regGDS_STRMOUT_PRIMS_NEEDED_2_LO = 0x2432
regGDS_STRMOUT_PRIMS_NEEDED_3_HI = 0x2437
regGDS_STRMOUT_PRIMS_NEEDED_3_LO = 0x2436
regGDS_STRMOUT_PRIMS_WRITTEN_0_HI = 0x242d
regGDS_STRMOUT_PRIMS_WRITTEN_0_LO = 0x242c
regGDS_STRMOUT_PRIMS_WRITTEN_1_HI = 0x2431
regGDS_STRMOUT_PRIMS_WRITTEN_1_LO = 0x2430
regGDS_STRMOUT_PRIMS_WRITTEN_2_HI = 0x2435
regGDS_STRMOUT_PRIMS_WRITTEN_2_LO = 0x2434
regGDS_STRMOUT_PRIMS_WRITTEN_3_HI = 0x2439
regGDS_STRMOUT_PRIMS_WRITTEN_3_LO = 0x2438
regGDS_VMID0_BASE = 0x20a0
regGDS_VMID0_SIZE = 0x20a1
regGDS_VMID10_BASE = 0x20b4
regGDS_VMID10_SIZE = 0x20b5
regGDS_VMID11_BASE = 0x20b6
regGDS_VMID11_SIZE = 0x20b7
regGDS_VMID12_BASE = 0x20b8
regGDS_VMID12_SIZE = 0x20b9
regGDS_VMID13_BASE = 0x20ba
regGDS_VMID13_SIZE = 0x20bb
regGDS_VMID14_BASE = 0x20bc
regGDS_VMID14_SIZE = 0x20bd
regGDS_VMID15_BASE = 0x20be
regGDS_VMID15_SIZE = 0x20bf
regGDS_VMID1_BASE = 0x20a2
regGDS_VMID1_SIZE = 0x20a3
regGDS_VMID2_BASE = 0x20a4
regGDS_VMID2_SIZE = 0x20a5
regGDS_VMID3_BASE = 0x20a6
regGDS_VMID3_SIZE = 0x20a7
regGDS_VMID4_BASE = 0x20a8
regGDS_VMID4_SIZE = 0x20a9
regGDS_VMID5_BASE = 0x20aa
regGDS_VMID5_SIZE = 0x20ab
regGDS_VMID6_BASE = 0x20ac
regGDS_VMID6_SIZE = 0x20ad
regGDS_VMID7_BASE = 0x20ae
regGDS_VMID7_SIZE = 0x20af
regGDS_VMID8_BASE = 0x20b0
regGDS_VMID8_SIZE = 0x20b1
regGDS_VMID9_BASE = 0x20b2
regGDS_VMID9_SIZE = 0x20b3
regGDS_VM_PROTECTION_FAULT = 0x1364
regGDS_WRITE_COMPLETE = 0x2409
regGDS_WR_ADDR = 0x2405
regGDS_WR_BURST_ADDR = 0x2407
regGDS_WR_BURST_DATA = 0x2408
regGDS_WR_DATA = 0x2406
regGE1_PERFCOUNTER0_HI = 0x30a5
regGE1_PERFCOUNTER0_LO = 0x30a4
regGE1_PERFCOUNTER0_SELECT = 0x38a4
regGE1_PERFCOUNTER0_SELECT1 = 0x38a5
regGE1_PERFCOUNTER1_HI = 0x30a7
regGE1_PERFCOUNTER1_LO = 0x30a6
regGE1_PERFCOUNTER1_SELECT = 0x38a6
regGE1_PERFCOUNTER1_SELECT1 = 0x38a7
regGE1_PERFCOUNTER2_HI = 0x30a9
regGE1_PERFCOUNTER2_LO = 0x30a8
regGE1_PERFCOUNTER2_SELECT = 0x38a8
regGE1_PERFCOUNTER2_SELECT1 = 0x38a9
regGE1_PERFCOUNTER3_HI = 0x30ab
regGE1_PERFCOUNTER3_LO = 0x30aa
regGE1_PERFCOUNTER3_SELECT = 0x38aa
regGE1_PERFCOUNTER3_SELECT1 = 0x38ab
regGE2_DIST_PERFCOUNTER0_HI = 0x30ad
regGE2_DIST_PERFCOUNTER0_LO = 0x30ac
regGE2_DIST_PERFCOUNTER0_SELECT = 0x38ac
regGE2_DIST_PERFCOUNTER0_SELECT1 = 0x38ad
regGE2_DIST_PERFCOUNTER1_HI = 0x30af
regGE2_DIST_PERFCOUNTER1_LO = 0x30ae
regGE2_DIST_PERFCOUNTER1_SELECT = 0x38ae
regGE2_DIST_PERFCOUNTER1_SELECT1 = 0x38af
regGE2_DIST_PERFCOUNTER2_HI = 0x30b1
regGE2_DIST_PERFCOUNTER2_LO = 0x30b0
regGE2_DIST_PERFCOUNTER2_SELECT = 0x38b0
regGE2_DIST_PERFCOUNTER2_SELECT1 = 0x38b1
regGE2_DIST_PERFCOUNTER3_HI = 0x30b3
regGE2_DIST_PERFCOUNTER3_LO = 0x30b2
regGE2_DIST_PERFCOUNTER3_SELECT = 0x38b2
regGE2_DIST_PERFCOUNTER3_SELECT1 = 0x38b3
regGE2_SE_CNTL_STATUS = 0x1011
regGE2_SE_PERFCOUNTER0_HI = 0x30b5
regGE2_SE_PERFCOUNTER0_LO = 0x30b4
regGE2_SE_PERFCOUNTER0_SELECT = 0x38b4
regGE2_SE_PERFCOUNTER0_SELECT1 = 0x38b5
regGE2_SE_PERFCOUNTER1_HI = 0x30b7
regGE2_SE_PERFCOUNTER1_LO = 0x30b6
regGE2_SE_PERFCOUNTER1_SELECT = 0x38b6
regGE2_SE_PERFCOUNTER1_SELECT1 = 0x38b7
regGE2_SE_PERFCOUNTER2_HI = 0x30b9
regGE2_SE_PERFCOUNTER2_LO = 0x30b8
regGE2_SE_PERFCOUNTER2_SELECT = 0x38b8
regGE2_SE_PERFCOUNTER2_SELECT1 = 0x38b9
regGE2_SE_PERFCOUNTER3_HI = 0x30bb
regGE2_SE_PERFCOUNTER3_LO = 0x30ba
regGE2_SE_PERFCOUNTER3_SELECT = 0x38ba
regGE2_SE_PERFCOUNTER3_SELECT1 = 0x38bb
regGE_CNTL = 0x225b
regGE_GS_FAST_LAUNCH_WG_DIM = 0x2264
regGE_GS_FAST_LAUNCH_WG_DIM_1 = 0x2265
regGE_INDX_OFFSET = 0x224a
regGE_MAX_OUTPUT_PER_SUBGROUP = 0x1ff
regGE_MAX_VTX_INDX = 0x2259
regGE_MIN_VTX_INDX = 0x2249
regGE_MULTI_PRIM_IB_RESET_EN = 0x224b
regGE_NGG_SUBGRP_CNTL = 0x2d3
regGE_PA_IF_SAFE_REG = 0x1019
regGE_PC_ALLOC = 0x2260
regGE_PRIV_CONTROL = 0x1004
regGE_RATE_CNTL_1 = 0xff4
regGE_RATE_CNTL_2 = 0xff5
regGE_SPI_IF_SAFE_REG = 0x1018
regGE_STATUS = 0x1005
regGE_STEREO_CNTL = 0x225f
regGE_USER_VGPR1 = 0x225c
regGE_USER_VGPR2 = 0x225d
regGE_USER_VGPR3 = 0x225e
regGE_USER_VGPR_EN = 0x2262
regGFX_COPY_STATE = 0x1f4
regGFX_ICG_GL2C_CTRL = 0x50fc
regGFX_ICG_GL2C_CTRL1 = 0x50fd
regGFX_IMU_AEB_OVERRIDE = 0x40bd
regGFX_IMU_C2PMSG_0 = 0x4000
regGFX_IMU_C2PMSG_1 = 0x4001
regGFX_IMU_C2PMSG_10 = 0x400a
regGFX_IMU_C2PMSG_11 = 0x400b
regGFX_IMU_C2PMSG_12 = 0x400c
regGFX_IMU_C2PMSG_13 = 0x400d
regGFX_IMU_C2PMSG_14 = 0x400e
regGFX_IMU_C2PMSG_15 = 0x400f
regGFX_IMU_C2PMSG_16 = 0x4010
regGFX_IMU_C2PMSG_17 = 0x4011
regGFX_IMU_C2PMSG_18 = 0x4012
regGFX_IMU_C2PMSG_19 = 0x4013
regGFX_IMU_C2PMSG_2 = 0x4002
regGFX_IMU_C2PMSG_20 = 0x4014
regGFX_IMU_C2PMSG_21 = 0x4015
regGFX_IMU_C2PMSG_22 = 0x4016
regGFX_IMU_C2PMSG_23 = 0x4017
regGFX_IMU_C2PMSG_24 = 0x4018
regGFX_IMU_C2PMSG_25 = 0x4019
regGFX_IMU_C2PMSG_26 = 0x401a
regGFX_IMU_C2PMSG_27 = 0x401b
regGFX_IMU_C2PMSG_28 = 0x401c
regGFX_IMU_C2PMSG_29 = 0x401d
regGFX_IMU_C2PMSG_3 = 0x4003
regGFX_IMU_C2PMSG_30 = 0x401e
regGFX_IMU_C2PMSG_31 = 0x401f
regGFX_IMU_C2PMSG_32 = 0x4020
regGFX_IMU_C2PMSG_33 = 0x4021
regGFX_IMU_C2PMSG_34 = 0x4022
regGFX_IMU_C2PMSG_35 = 0x4023
regGFX_IMU_C2PMSG_36 = 0x4024
regGFX_IMU_C2PMSG_37 = 0x4025
regGFX_IMU_C2PMSG_38 = 0x4026
regGFX_IMU_C2PMSG_39 = 0x4027
regGFX_IMU_C2PMSG_4 = 0x4004
regGFX_IMU_C2PMSG_40 = 0x4028
regGFX_IMU_C2PMSG_41 = 0x4029
regGFX_IMU_C2PMSG_42 = 0x402a
regGFX_IMU_C2PMSG_43 = 0x402b
regGFX_IMU_C2PMSG_44 = 0x402c
regGFX_IMU_C2PMSG_45 = 0x402d
regGFX_IMU_C2PMSG_46 = 0x402e
regGFX_IMU_C2PMSG_47 = 0x402f
regGFX_IMU_C2PMSG_5 = 0x4005
regGFX_IMU_C2PMSG_6 = 0x4006
regGFX_IMU_C2PMSG_7 = 0x4007
regGFX_IMU_C2PMSG_8 = 0x4008
regGFX_IMU_C2PMSG_9 = 0x4009
regGFX_IMU_C2PMSG_ACCESS_CTRL0 = 0x4040
regGFX_IMU_C2PMSG_ACCESS_CTRL1 = 0x4041
regGFX_IMU_CLK_CTRL = 0x409d
regGFX_IMU_CORE_CTRL = 0x40b6
regGFX_IMU_CORE_INT_STATUS = 0x407f
regGFX_IMU_CORE_STATUS = 0x40b7
regGFX_IMU_DOORBELL_CONTROL = 0x409e
regGFX_IMU_DPM_ACC = 0x40a9
regGFX_IMU_DPM_CONTROL = 0x40a8
regGFX_IMU_DPM_REF_COUNTER = 0x40aa
regGFX_IMU_D_RAM_ADDR = 0x40fc
regGFX_IMU_D_RAM_DATA = 0x40fd
regGFX_IMU_FENCE_CTRL = 0x40b0
regGFX_IMU_FENCE_LOG_ADDR = 0x40b2
regGFX_IMU_FENCE_LOG_INIT = 0x40b1
regGFX_IMU_FUSE_CTRL = 0x40e0
regGFX_IMU_FW_GTS_HI = 0x4079
regGFX_IMU_FW_GTS_LO = 0x4078
regGFX_IMU_GAP_PWROK = 0x40ba
regGFX_IMU_GFXCLK_BYPASS_CTRL = 0x409c
regGFX_IMU_GFX_IH_GASKET_CTRL = 0x40ff
regGFX_IMU_GFX_ISO_CTRL = 0x40bf
regGFX_IMU_GFX_RESET_CTRL = 0x40bc
regGFX_IMU_GTS_OFFSET_HI = 0x407b
regGFX_IMU_GTS_OFFSET_LO = 0x407a
regGFX_IMU_IH_CTRL_1 = 0x4090
regGFX_IMU_IH_CTRL_2 = 0x4091
regGFX_IMU_IH_CTRL_3 = 0x4092
regGFX_IMU_IH_STATUS = 0x4093
regGFX_IMU_I_RAM_ADDR = 0x5f90
regGFX_IMU_I_RAM_DATA = 0x5f91
regGFX_IMU_MP1_MUTEX = 0x4043
regGFX_IMU_MSG_FLAGS = 0x403f
regGFX_IMU_PIC_INTR = 0x408c
regGFX_IMU_PIC_INTR_ID = 0x408d
regGFX_IMU_PIC_INT_EDGE = 0x4082
regGFX_IMU_PIC_INT_LVL = 0x4081
regGFX_IMU_PIC_INT_MASK = 0x4080
regGFX_IMU_PIC_INT_PRI_0 = 0x4083
regGFX_IMU_PIC_INT_PRI_1 = 0x4084
regGFX_IMU_PIC_INT_PRI_2 = 0x4085
regGFX_IMU_PIC_INT_PRI_3 = 0x4086
regGFX_IMU_PIC_INT_PRI_4 = 0x4087
regGFX_IMU_PIC_INT_PRI_5 = 0x4088
regGFX_IMU_PIC_INT_PRI_6 = 0x4089
regGFX_IMU_PIC_INT_PRI_7 = 0x408a
regGFX_IMU_PIC_INT_STATUS = 0x408b
regGFX_IMU_PROGRAM_CTR = 0x40b5
regGFX_IMU_PWRMGT_IRQ_CTRL = 0x4042
regGFX_IMU_PWROK = 0x40b9
regGFX_IMU_PWROKRAW = 0x40b8
regGFX_IMU_RESETn = 0x40bb
regGFX_IMU_RLC_BOOTLOADER_ADDR_HI = 0x5f81
regGFX_IMU_RLC_BOOTLOADER_ADDR_LO = 0x5f82
regGFX_IMU_RLC_BOOTLOADER_SIZE = 0x5f83
regGFX_IMU_RLC_CG_CTRL = 0x40a0
regGFX_IMU_RLC_CMD = 0x404b
regGFX_IMU_RLC_DATA_0 = 0x404a
regGFX_IMU_RLC_DATA_1 = 0x4049
regGFX_IMU_RLC_DATA_2 = 0x4048
regGFX_IMU_RLC_DATA_3 = 0x4047
regGFX_IMU_RLC_DATA_4 = 0x4046
regGFX_IMU_RLC_GTS_OFFSET_HI = 0x407d
regGFX_IMU_RLC_GTS_OFFSET_LO = 0x407c
regGFX_IMU_RLC_MSG_STATUS = 0x404f
regGFX_IMU_RLC_MUTEX = 0x404c
regGFX_IMU_RLC_OVERRIDE = 0x40a3
regGFX_IMU_RLC_RAM_ADDR_HIGH = 0x40ad
regGFX_IMU_RLC_RAM_ADDR_LOW = 0x40ae
regGFX_IMU_RLC_RAM_DATA = 0x40af
regGFX_IMU_RLC_RAM_INDEX = 0x40ac
regGFX_IMU_RLC_RESET_VECTOR = 0x40a2
regGFX_IMU_RLC_STATUS = 0x4054
regGFX_IMU_RLC_THROTTLE_GFX = 0x40a1
regGFX_IMU_SCRATCH_0 = 0x4068
regGFX_IMU_SCRATCH_1 = 0x4069
regGFX_IMU_SCRATCH_10 = 0x4072
regGFX_IMU_SCRATCH_11 = 0x4073
regGFX_IMU_SCRATCH_12 = 0x4074
regGFX_IMU_SCRATCH_13 = 0x4075
regGFX_IMU_SCRATCH_14 = 0x4076
regGFX_IMU_SCRATCH_15 = 0x4077
regGFX_IMU_SCRATCH_2 = 0x406a
regGFX_IMU_SCRATCH_3 = 0x406b
regGFX_IMU_SCRATCH_4 = 0x406c
regGFX_IMU_SCRATCH_5 = 0x406d
regGFX_IMU_SCRATCH_6 = 0x406e
regGFX_IMU_SCRATCH_7 = 0x406f
regGFX_IMU_SCRATCH_8 = 0x4070
regGFX_IMU_SCRATCH_9 = 0x4071
regGFX_IMU_SMUIO_VIDCHG_CTRL = 0x4098
regGFX_IMU_SOC_ADDR = 0x405a
regGFX_IMU_SOC_DATA = 0x4059
regGFX_IMU_SOC_REQ = 0x405b
regGFX_IMU_STATUS = 0x4055
regGFX_IMU_TELEMETRY = 0x4060
regGFX_IMU_TELEMETRY_DATA = 0x4061
regGFX_IMU_TELEMETRY_TEMPERATURE = 0x4062
regGFX_IMU_TIMER0_CMP0 = 0x40c4
regGFX_IMU_TIMER0_CMP1 = 0x40c5
regGFX_IMU_TIMER0_CMP3 = 0x40c7
regGFX_IMU_TIMER0_CMP_AUTOINC = 0x40c2
regGFX_IMU_TIMER0_CMP_INTEN = 0x40c3
regGFX_IMU_TIMER0_CTRL0 = 0x40c0
regGFX_IMU_TIMER0_CTRL1 = 0x40c1
regGFX_IMU_TIMER0_VALUE = 0x40c8
regGFX_IMU_TIMER1_CMP0 = 0x40cd
regGFX_IMU_TIMER1_CMP1 = 0x40ce
regGFX_IMU_TIMER1_CMP3 = 0x40d0
regGFX_IMU_TIMER1_CMP_AUTOINC = 0x40cb
regGFX_IMU_TIMER1_CMP_INTEN = 0x40cc
regGFX_IMU_TIMER1_CTRL0 = 0x40c9
regGFX_IMU_TIMER1_CTRL1 = 0x40ca
regGFX_IMU_TIMER1_VALUE = 0x40d1
regGFX_IMU_TIMER2_CMP0 = 0x40d6
regGFX_IMU_TIMER2_CMP1 = 0x40d7
regGFX_IMU_TIMER2_CMP3 = 0x40d9
regGFX_IMU_TIMER2_CMP_AUTOINC = 0x40d4
regGFX_IMU_TIMER2_CMP_INTEN = 0x40d5
regGFX_IMU_TIMER2_CTRL0 = 0x40d2
regGFX_IMU_TIMER2_CTRL1 = 0x40d3
regGFX_IMU_TIMER2_VALUE = 0x40da
regGFX_IMU_VDCI_RESET_CTRL = 0x40be
regGFX_IMU_VF_CTRL = 0x405c
regGFX_PIPE_CONTROL = 0x100d
regGFX_PIPE_PRIORITY = 0x587f
regGL1A_PERFCOUNTER0_HI = 0x35c1
regGL1A_PERFCOUNTER0_LO = 0x35c0
regGL1A_PERFCOUNTER0_SELECT = 0x3dc0
regGL1A_PERFCOUNTER0_SELECT1 = 0x3dc1
regGL1A_PERFCOUNTER1_HI = 0x35c3
regGL1A_PERFCOUNTER1_LO = 0x35c2
regGL1A_PERFCOUNTER1_SELECT = 0x3dc2
regGL1A_PERFCOUNTER2_HI = 0x35c5
regGL1A_PERFCOUNTER2_LO = 0x35c4
regGL1A_PERFCOUNTER2_SELECT = 0x3dc3
regGL1A_PERFCOUNTER3_HI = 0x35c7
regGL1A_PERFCOUNTER3_LO = 0x35c6
regGL1A_PERFCOUNTER3_SELECT = 0x3dc4
regGL1C_PERFCOUNTER0_HI = 0x33a1
regGL1C_PERFCOUNTER0_LO = 0x33a0
regGL1C_PERFCOUNTER0_SELECT = 0x3ba0
regGL1C_PERFCOUNTER0_SELECT1 = 0x3ba1
regGL1C_PERFCOUNTER1_HI = 0x33a3
regGL1C_PERFCOUNTER1_LO = 0x33a2
regGL1C_PERFCOUNTER1_SELECT = 0x3ba2
regGL1C_PERFCOUNTER2_HI = 0x33a5
regGL1C_PERFCOUNTER2_LO = 0x33a4
regGL1C_PERFCOUNTER2_SELECT = 0x3ba3
regGL1C_PERFCOUNTER3_HI = 0x33a7
regGL1C_PERFCOUNTER3_LO = 0x33a6
regGL1C_PERFCOUNTER3_SELECT = 0x3ba4
regGL1C_STATUS = 0x2d41
regGL1C_UTCL0_CNTL1 = 0x2d42
regGL1C_UTCL0_CNTL2 = 0x2d43
regGL1C_UTCL0_RETRY = 0x2d45
regGL1C_UTCL0_STATUS = 0x2d44
regGL1H_ARB_CTRL = 0x2e40
regGL1H_ARB_STATUS = 0x2e44
regGL1H_BURST_CTRL = 0x2e43
regGL1H_BURST_MASK = 0x2e42
regGL1H_GL1_CREDITS = 0x2e41
regGL1H_ICG_CTRL = 0x50e8
regGL1H_PERFCOUNTER0_HI = 0x35d1
regGL1H_PERFCOUNTER0_LO = 0x35d0
regGL1H_PERFCOUNTER0_SELECT = 0x3dd0
regGL1H_PERFCOUNTER0_SELECT1 = 0x3dd1
regGL1H_PERFCOUNTER1_HI = 0x35d3
regGL1H_PERFCOUNTER1_LO = 0x35d2
regGL1H_PERFCOUNTER1_SELECT = 0x3dd2
regGL1H_PERFCOUNTER2_HI = 0x35d5
regGL1H_PERFCOUNTER2_LO = 0x35d4
regGL1H_PERFCOUNTER2_SELECT = 0x3dd3
regGL1H_PERFCOUNTER3_HI = 0x35d7
regGL1H_PERFCOUNTER3_LO = 0x35d6
regGL1H_PERFCOUNTER3_SELECT = 0x3dd4
regGL1I_GL1R_MGCG_OVERRIDE = 0x50e4
regGL1I_GL1R_REP_FGCG_OVERRIDE = 0x2d05
regGL1_ARB_STATUS = 0x2d03
regGL1_DRAM_BURST_MASK = 0x2d02
regGL1_PIPE_STEER = 0x5b84
regGL2A_ADDR_MATCH_CTRL = 0x2e20
regGL2A_ADDR_MATCH_MASK = 0x2e21
regGL2A_ADDR_MATCH_SIZE = 0x2e22
regGL2A_PERFCOUNTER0_HI = 0x3391
regGL2A_PERFCOUNTER0_LO = 0x3390
regGL2A_PERFCOUNTER0_SELECT = 0x3b90
regGL2A_PERFCOUNTER0_SELECT1 = 0x3b91
regGL2A_PERFCOUNTER1_HI = 0x3393
regGL2A_PERFCOUNTER1_LO = 0x3392
regGL2A_PERFCOUNTER1_SELECT = 0x3b92
regGL2A_PERFCOUNTER1_SELECT1 = 0x3b93
regGL2A_PERFCOUNTER2_HI = 0x3395
regGL2A_PERFCOUNTER2_LO = 0x3394
regGL2A_PERFCOUNTER2_SELECT = 0x3b94
regGL2A_PERFCOUNTER3_HI = 0x3397
regGL2A_PERFCOUNTER3_LO = 0x3396
regGL2A_PERFCOUNTER3_SELECT = 0x3b95
regGL2A_PRIORITY_CTRL = 0x2e23
regGL2A_RESP_THROTTLE_CTRL = 0x2e2a
regGL2C_ADDR_MATCH_MASK = 0x2e03
regGL2C_ADDR_MATCH_SIZE = 0x2e04
regGL2C_CM_CTRL0 = 0x2e07
regGL2C_CM_CTRL1 = 0x2e08
regGL2C_CM_STALL = 0x2e09
regGL2C_CTRL = 0x2e00
regGL2C_CTRL2 = 0x2e01
regGL2C_CTRL3 = 0x2e0c
regGL2C_CTRL4 = 0x2e17
regGL2C_DISCARD_STALL_CTRL = 0x2e18
regGL2C_LB_CTR_CTRL = 0x2e0d
regGL2C_LB_CTR_SEL0 = 0x2e12
regGL2C_LB_CTR_SEL1 = 0x2e13
regGL2C_LB_DATA0 = 0x2e0e
regGL2C_LB_DATA1 = 0x2e0f
regGL2C_LB_DATA2 = 0x2e10
regGL2C_LB_DATA3 = 0x2e11
regGL2C_PERFCOUNTER0_HI = 0x3381
regGL2C_PERFCOUNTER0_LO = 0x3380
regGL2C_PERFCOUNTER0_SELECT = 0x3b80
regGL2C_PERFCOUNTER0_SELECT1 = 0x3b81
regGL2C_PERFCOUNTER1_HI = 0x3383
regGL2C_PERFCOUNTER1_LO = 0x3382
regGL2C_PERFCOUNTER1_SELECT = 0x3b82
regGL2C_PERFCOUNTER1_SELECT1 = 0x3b83
regGL2C_PERFCOUNTER2_HI = 0x3385
regGL2C_PERFCOUNTER2_LO = 0x3384
regGL2C_PERFCOUNTER2_SELECT = 0x3b84
regGL2C_PERFCOUNTER3_HI = 0x3387
regGL2C_PERFCOUNTER3_LO = 0x3386
regGL2C_PERFCOUNTER3_SELECT = 0x3b85
regGL2C_SOFT_RESET = 0x2e06
regGL2C_WBINVL2 = 0x2e05
regGL2_PIPE_STEER_0 = 0x5b80
regGL2_PIPE_STEER_1 = 0x5b81
regGL2_PIPE_STEER_2 = 0x5b82
regGL2_PIPE_STEER_3 = 0x5b83
regGRBM_CAM_DATA = 0x5e11
regGRBM_CAM_DATA_UPPER = 0x5e12
regGRBM_CAM_INDEX = 0x5e10
regGRBM_CHIP_REVISION = 0xdc1
regGRBM_CNTL = 0xda0
regGRBM_DSM_BYPASS = 0xdbe
regGRBM_FENCE_RANGE0 = 0xdca
regGRBM_FENCE_RANGE1 = 0xdcb
regGRBM_GFX_CLKEN_CNTL = 0xdac
regGRBM_GFX_CNTL = 0x900
regGRBM_GFX_CNTL_SR_DATA = 0x5a03
regGRBM_GFX_CNTL_SR_SELECT = 0x5a02
regGRBM_GFX_INDEX = 0x2200
regGRBM_GFX_INDEX_SR_DATA = 0x5a01
regGRBM_GFX_INDEX_SR_SELECT = 0x5a00
regGRBM_HYP_CAM_DATA = 0x5e11
regGRBM_HYP_CAM_DATA_UPPER = 0x5e12
regGRBM_HYP_CAM_INDEX = 0x5e10
regGRBM_IH_CREDIT = 0xdc4
regGRBM_INT_CNTL = 0xdb8
regGRBM_INVALID_PIPE = 0xdc9
regGRBM_NOWHERE = 0x901
regGRBM_PERFCOUNTER0_HI = 0x3041
regGRBM_PERFCOUNTER0_LO = 0x3040
regGRBM_PERFCOUNTER0_SELECT = 0x3840
regGRBM_PERFCOUNTER0_SELECT_HI = 0x384d
regGRBM_PERFCOUNTER1_HI = 0x3044
regGRBM_PERFCOUNTER1_LO = 0x3043
regGRBM_PERFCOUNTER1_SELECT = 0x3841
regGRBM_PERFCOUNTER1_SELECT_HI = 0x384e
regGRBM_PWR_CNTL = 0xda3
regGRBM_PWR_CNTL2 = 0xdc5
regGRBM_READ_ERROR = 0xdb6
regGRBM_READ_ERROR2 = 0xdb7
regGRBM_SCRATCH_REG0 = 0xde0
regGRBM_SCRATCH_REG1 = 0xde1
regGRBM_SCRATCH_REG2 = 0xde2
regGRBM_SCRATCH_REG3 = 0xde3
regGRBM_SCRATCH_REG4 = 0xde4
regGRBM_SCRATCH_REG5 = 0xde5
regGRBM_SCRATCH_REG6 = 0xde6
regGRBM_SCRATCH_REG7 = 0xde7
regGRBM_SE0_PERFCOUNTER_HI = 0x3046
regGRBM_SE0_PERFCOUNTER_LO = 0x3045
regGRBM_SE0_PERFCOUNTER_SELECT = 0x3842
regGRBM_SE1_PERFCOUNTER_HI = 0x3048
regGRBM_SE1_PERFCOUNTER_LO = 0x3047
regGRBM_SE1_PERFCOUNTER_SELECT = 0x3843
regGRBM_SE2_PERFCOUNTER_HI = 0x304a
regGRBM_SE2_PERFCOUNTER_LO = 0x3049
regGRBM_SE2_PERFCOUNTER_SELECT = 0x3844
regGRBM_SE3_PERFCOUNTER_HI = 0x304c
regGRBM_SE3_PERFCOUNTER_LO = 0x304b
regGRBM_SE3_PERFCOUNTER_SELECT = 0x3845
regGRBM_SE4_PERFCOUNTER_HI = 0x304e
regGRBM_SE4_PERFCOUNTER_LO = 0x304d
regGRBM_SE4_PERFCOUNTER_SELECT = 0x3846
regGRBM_SE5_PERFCOUNTER_HI = 0x3050
regGRBM_SE5_PERFCOUNTER_LO = 0x304f
regGRBM_SE5_PERFCOUNTER_SELECT = 0x3847
regGRBM_SE6_PERFCOUNTER_HI = 0x3052
regGRBM_SE6_PERFCOUNTER_LO = 0x3051
regGRBM_SE6_PERFCOUNTER_SELECT = 0x3848
regGRBM_SEC_CNTL = 0x5e0d
regGRBM_SE_REMAP_CNTL = 0x5a08
regGRBM_SKEW_CNTL = 0xda1
regGRBM_SOFT_RESET = 0xda8
regGRBM_STATUS = 0xda4
regGRBM_STATUS2 = 0xda2
regGRBM_STATUS3 = 0xda7
regGRBM_STATUS_SE0 = 0xda5
regGRBM_STATUS_SE1 = 0xda6
regGRBM_STATUS_SE2 = 0xdae
regGRBM_STATUS_SE3 = 0xdaf
regGRBM_STATUS_SE4 = 0xdb0
regGRBM_STATUS_SE5 = 0xdb1
regGRBM_TRAP_ADDR = 0xdba
regGRBM_TRAP_ADDR_MSK = 0xdbb
regGRBM_TRAP_OP = 0xdb9
regGRBM_TRAP_WD = 0xdbc
regGRBM_TRAP_WD_MSK = 0xdbd
regGRBM_UTCL2_INVAL_RANGE_END = 0xdc7
regGRBM_UTCL2_INVAL_RANGE_START = 0xdc6
regGRBM_WAIT_IDLE_CLOCKS = 0xdad
regGRBM_WRITE_ERROR = 0xdbf
regGRTAVFS_CLK_CNTL = 0x4b0e
regGRTAVFS_GENERAL_0 = 0x4b02
regGRTAVFS_PSM_CNTL = 0x4b0d
regGRTAVFS_RTAVFS_RD_DATA = 0x4b03
regGRTAVFS_RTAVFS_REG_ADDR = 0x4b00
regGRTAVFS_RTAVFS_REG_CTRL = 0x4b04
regGRTAVFS_RTAVFS_REG_STATUS = 0x4b05
regGRTAVFS_RTAVFS_WR_DATA = 0x4b01
regGRTAVFS_SE_CLK_CNTL = 0x4b4e
regGRTAVFS_SE_GENERAL_0 = 0x4b42
regGRTAVFS_SE_PSM_CNTL = 0x4b4d
regGRTAVFS_SE_RTAVFS_RD_DATA = 0x4b43
regGRTAVFS_SE_RTAVFS_REG_ADDR = 0x4b40
regGRTAVFS_SE_RTAVFS_REG_CTRL = 0x4b44
regGRTAVFS_SE_RTAVFS_REG_STATUS = 0x4b45
regGRTAVFS_SE_RTAVFS_WR_DATA = 0x4b41
regGRTAVFS_SE_SOFT_RESET = 0x4b4c
regGRTAVFS_SE_TARG_FREQ = 0x4b46
regGRTAVFS_SE_TARG_VOLT = 0x4b47
regGRTAVFS_SOFT_RESET = 0x4b0c
regGRTAVFS_TARG_FREQ = 0x4b06
regGRTAVFS_TARG_VOLT = 0x4b07
regGUS_DRAM_COMBINE_FLUSH = 0x2c1e
regGUS_DRAM_COMBINE_RD_WR_EN = 0x2c1f
regGUS_DRAM_GROUP_BURST = 0x2c31
regGUS_DRAM_PRI_AGE_COEFF = 0x2c21
regGUS_DRAM_PRI_AGE_RATE = 0x2c20
regGUS_DRAM_PRI_FIXED = 0x2c23
regGUS_DRAM_PRI_QUANT1_PRI1 = 0x2c2b
regGUS_DRAM_PRI_QUANT1_PRI2 = 0x2c2c
regGUS_DRAM_PRI_QUANT1_PRI3 = 0x2c2d
regGUS_DRAM_PRI_QUANT1_PRI4 = 0x2c2e
regGUS_DRAM_PRI_QUANT1_PRI5 = 0x2c2f
regGUS_DRAM_PRI_QUANT_PRI1 = 0x2c26
regGUS_DRAM_PRI_QUANT_PRI2 = 0x2c27
regGUS_DRAM_PRI_QUANT_PRI3 = 0x2c28
regGUS_DRAM_PRI_QUANT_PRI4 = 0x2c29
regGUS_DRAM_PRI_QUANT_PRI5 = 0x2c2a
regGUS_DRAM_PRI_QUEUING = 0x2c22
regGUS_DRAM_PRI_URGENCY_COEFF = 0x2c24
regGUS_DRAM_PRI_URGENCY_MODE = 0x2c25
regGUS_ERR_STATUS = 0x2c3e
regGUS_ICG_CTRL = 0x50f4
regGUS_IO_GROUP_BURST = 0x2c30
regGUS_IO_RD_COMBINE_FLUSH = 0x2c00
regGUS_IO_RD_PRI_AGE_COEFF = 0x2c04
regGUS_IO_RD_PRI_AGE_RATE = 0x2c02
regGUS_IO_RD_PRI_FIXED = 0x2c08
regGUS_IO_RD_PRI_QUANT1_PRI1 = 0x2c16
regGUS_IO_RD_PRI_QUANT1_PRI2 = 0x2c17
regGUS_IO_RD_PRI_QUANT1_PRI3 = 0x2c18
regGUS_IO_RD_PRI_QUANT1_PRI4 = 0x2c19
regGUS_IO_RD_PRI_QUANT_PRI1 = 0x2c0e
regGUS_IO_RD_PRI_QUANT_PRI2 = 0x2c0f
regGUS_IO_RD_PRI_QUANT_PRI3 = 0x2c10
regGUS_IO_RD_PRI_QUANT_PRI4 = 0x2c11
regGUS_IO_RD_PRI_QUEUING = 0x2c06
regGUS_IO_RD_PRI_URGENCY_COEFF = 0x2c0a
regGUS_IO_RD_PRI_URGENCY_MODE = 0x2c0c
regGUS_IO_WR_COMBINE_FLUSH = 0x2c01
regGUS_IO_WR_PRI_AGE_COEFF = 0x2c05
regGUS_IO_WR_PRI_AGE_RATE = 0x2c03
regGUS_IO_WR_PRI_FIXED = 0x2c09
regGUS_IO_WR_PRI_QUANT1_PRI1 = 0x2c1a
regGUS_IO_WR_PRI_QUANT1_PRI2 = 0x2c1b
regGUS_IO_WR_PRI_QUANT1_PRI3 = 0x2c1c
regGUS_IO_WR_PRI_QUANT1_PRI4 = 0x2c1d
regGUS_IO_WR_PRI_QUANT_PRI1 = 0x2c12
regGUS_IO_WR_PRI_QUANT_PRI2 = 0x2c13
regGUS_IO_WR_PRI_QUANT_PRI3 = 0x2c14
regGUS_IO_WR_PRI_QUANT_PRI4 = 0x2c15
regGUS_IO_WR_PRI_QUEUING = 0x2c07
regGUS_IO_WR_PRI_URGENCY_COEFF = 0x2c0b
regGUS_IO_WR_PRI_URGENCY_MODE = 0x2c0d
regGUS_L1_CH0_CMD_IN = 0x2c46
regGUS_L1_CH0_CMD_OUT = 0x2c47
regGUS_L1_CH0_DATA_IN = 0x2c48
regGUS_L1_CH0_DATA_OUT = 0x2c49
regGUS_L1_CH0_DATA_U_IN = 0x2c4a
regGUS_L1_CH0_DATA_U_OUT = 0x2c4b
regGUS_L1_CH1_CMD_IN = 0x2c4c
regGUS_L1_CH1_CMD_OUT = 0x2c4d
regGUS_L1_CH1_DATA_IN = 0x2c4e
regGUS_L1_CH1_DATA_OUT = 0x2c4f
regGUS_L1_CH1_DATA_U_IN = 0x2c50
regGUS_L1_CH1_DATA_U_OUT = 0x2c51
regGUS_L1_SA0_CMD_IN = 0x2c52
regGUS_L1_SA0_CMD_OUT = 0x2c53
regGUS_L1_SA0_DATA_IN = 0x2c54
regGUS_L1_SA0_DATA_OUT = 0x2c55
regGUS_L1_SA0_DATA_U_IN = 0x2c56
regGUS_L1_SA0_DATA_U_OUT = 0x2c57
regGUS_L1_SA1_CMD_IN = 0x2c58
regGUS_L1_SA1_CMD_OUT = 0x2c59
regGUS_L1_SA1_DATA_IN = 0x2c5a
regGUS_L1_SA1_DATA_OUT = 0x2c5b
regGUS_L1_SA1_DATA_U_IN = 0x2c5c
regGUS_L1_SA1_DATA_U_OUT = 0x2c5d
regGUS_L1_SA2_CMD_IN = 0x2c5e
regGUS_L1_SA2_CMD_OUT = 0x2c5f
regGUS_L1_SA2_DATA_IN = 0x2c60
regGUS_L1_SA2_DATA_OUT = 0x2c61
regGUS_L1_SA2_DATA_U_IN = 0x2c62
regGUS_L1_SA2_DATA_U_OUT = 0x2c63
regGUS_L1_SA3_CMD_IN = 0x2c64
regGUS_L1_SA3_CMD_OUT = 0x2c65
regGUS_L1_SA3_DATA_IN = 0x2c66
regGUS_L1_SA3_DATA_OUT = 0x2c67
regGUS_L1_SA3_DATA_U_IN = 0x2c68
regGUS_L1_SA3_DATA_U_OUT = 0x2c69
regGUS_LATENCY_SAMPLING = 0x2c3d
regGUS_MISC = 0x2c3c
regGUS_MISC2 = 0x2c3f
regGUS_MISC3 = 0x2c6a
regGUS_PERFCOUNTER0_CFG = 0x3e03
regGUS_PERFCOUNTER1_CFG = 0x3e04
regGUS_PERFCOUNTER2_HI = 0x3641
regGUS_PERFCOUNTER2_LO = 0x3640
regGUS_PERFCOUNTER2_MODE = 0x3e02
regGUS_PERFCOUNTER2_SELECT = 0x3e00
regGUS_PERFCOUNTER2_SELECT1 = 0x3e01
regGUS_PERFCOUNTER_HI = 0x3643
regGUS_PERFCOUNTER_LO = 0x3642
regGUS_PERFCOUNTER_RSLT_CNTL = 0x3e05
regGUS_SDP_ARB_FINAL = 0x2c32
regGUS_SDP_CREDITS = 0x2c34
regGUS_SDP_ENABLE = 0x2c45
regGUS_SDP_QOS_VC_PRIORITY = 0x2c33
regGUS_SDP_REQ_CNTL = 0x2c3b
regGUS_SDP_TAG_RESERVE0 = 0x2c35
regGUS_SDP_TAG_RESERVE1 = 0x2c36
regGUS_SDP_VCC_RESERVE0 = 0x2c37
regGUS_SDP_VCC_RESERVE1 = 0x2c38
regGUS_SDP_VCD_RESERVE0 = 0x2c39
regGUS_SDP_VCD_RESERVE1 = 0x2c3a
regGUS_WRRSP_FIFO_CNTL = 0x2c6b
regIA_ENHANCE = 0x29c
regIA_UTCL1_CNTL = 0xfe6
regIA_UTCL1_STATUS = 0xfe7
regIA_UTCL1_STATUS_2 = 0xfd7
regICG_CHA_CTRL = 0x50f1
regICG_CHCG_CLK_CTRL = 0x5144
regICG_CHC_CLK_CTRL = 0x5140
regICG_GL1A_CTRL = 0x50f0
regICG_GL1C_CLK_CTRL = 0x50ec
regICG_LDS_CLK_CTRL = 0x5114
regICG_SP_CLK_CTRL = 0x5093
regLDS_CONFIG = 0x10a2
regPA_CL_CLIP_CNTL = 0x204
regPA_CL_CNTL_STATUS = 0x1024
regPA_CL_ENHANCE = 0x1025
regPA_CL_GB_HORZ_CLIP_ADJ = 0x2fc
regPA_CL_GB_HORZ_DISC_ADJ = 0x2fd
regPA_CL_GB_VERT_CLIP_ADJ = 0x2fa
regPA_CL_GB_VERT_DISC_ADJ = 0x2fb
regPA_CL_NANINF_CNTL = 0x208
regPA_CL_NGG_CNTL = 0x20e
regPA_CL_POINT_CULL_RAD = 0x1f8
regPA_CL_POINT_SIZE = 0x1f7
regPA_CL_POINT_X_RAD = 0x1f5
regPA_CL_POINT_Y_RAD = 0x1f6
regPA_CL_PROG_NEAR_CLIP_Z = 0x187
regPA_CL_UCP_0_W = 0x172
regPA_CL_UCP_0_X = 0x16f
regPA_CL_UCP_0_Y = 0x170
regPA_CL_UCP_0_Z = 0x171
regPA_CL_UCP_1_W = 0x176
regPA_CL_UCP_1_X = 0x173
regPA_CL_UCP_1_Y = 0x174
regPA_CL_UCP_1_Z = 0x175
regPA_CL_UCP_2_W = 0x17a
regPA_CL_UCP_2_X = 0x177
regPA_CL_UCP_2_Y = 0x178
regPA_CL_UCP_2_Z = 0x179
regPA_CL_UCP_3_W = 0x17e
regPA_CL_UCP_3_X = 0x17b
regPA_CL_UCP_3_Y = 0x17c
regPA_CL_UCP_3_Z = 0x17d
regPA_CL_UCP_4_W = 0x182
regPA_CL_UCP_4_X = 0x17f
regPA_CL_UCP_4_Y = 0x180
regPA_CL_UCP_4_Z = 0x181
regPA_CL_UCP_5_W = 0x186
regPA_CL_UCP_5_X = 0x183
regPA_CL_UCP_5_Y = 0x184
regPA_CL_UCP_5_Z = 0x185
regPA_CL_VPORT_XOFFSET = 0x110
regPA_CL_VPORT_XOFFSET_1 = 0x116
regPA_CL_VPORT_XOFFSET_10 = 0x14c
regPA_CL_VPORT_XOFFSET_11 = 0x152
regPA_CL_VPORT_XOFFSET_12 = 0x158
regPA_CL_VPORT_XOFFSET_13 = 0x15e
regPA_CL_VPORT_XOFFSET_14 = 0x164
regPA_CL_VPORT_XOFFSET_15 = 0x16a
regPA_CL_VPORT_XOFFSET_2 = 0x11c
regPA_CL_VPORT_XOFFSET_3 = 0x122
regPA_CL_VPORT_XOFFSET_4 = 0x128
regPA_CL_VPORT_XOFFSET_5 = 0x12e
regPA_CL_VPORT_XOFFSET_6 = 0x134
regPA_CL_VPORT_XOFFSET_7 = 0x13a
regPA_CL_VPORT_XOFFSET_8 = 0x140
regPA_CL_VPORT_XOFFSET_9 = 0x146
regPA_CL_VPORT_XSCALE = 0x10f
regPA_CL_VPORT_XSCALE_1 = 0x115
regPA_CL_VPORT_XSCALE_10 = 0x14b
regPA_CL_VPORT_XSCALE_11 = 0x151
regPA_CL_VPORT_XSCALE_12 = 0x157
regPA_CL_VPORT_XSCALE_13 = 0x15d
regPA_CL_VPORT_XSCALE_14 = 0x163
regPA_CL_VPORT_XSCALE_15 = 0x169
regPA_CL_VPORT_XSCALE_2 = 0x11b
regPA_CL_VPORT_XSCALE_3 = 0x121
regPA_CL_VPORT_XSCALE_4 = 0x127
regPA_CL_VPORT_XSCALE_5 = 0x12d
regPA_CL_VPORT_XSCALE_6 = 0x133
regPA_CL_VPORT_XSCALE_7 = 0x139
regPA_CL_VPORT_XSCALE_8 = 0x13f
regPA_CL_VPORT_XSCALE_9 = 0x145
regPA_CL_VPORT_YOFFSET = 0x112
regPA_CL_VPORT_YOFFSET_1 = 0x118
regPA_CL_VPORT_YOFFSET_10 = 0x14e
regPA_CL_VPORT_YOFFSET_11 = 0x154
regPA_CL_VPORT_YOFFSET_12 = 0x15a
regPA_CL_VPORT_YOFFSET_13 = 0x160
regPA_CL_VPORT_YOFFSET_14 = 0x166
regPA_CL_VPORT_YOFFSET_15 = 0x16c
regPA_CL_VPORT_YOFFSET_2 = 0x11e
regPA_CL_VPORT_YOFFSET_3 = 0x124
regPA_CL_VPORT_YOFFSET_4 = 0x12a
regPA_CL_VPORT_YOFFSET_5 = 0x130
regPA_CL_VPORT_YOFFSET_6 = 0x136
regPA_CL_VPORT_YOFFSET_7 = 0x13c
regPA_CL_VPORT_YOFFSET_8 = 0x142
regPA_CL_VPORT_YOFFSET_9 = 0x148
regPA_CL_VPORT_YSCALE = 0x111
regPA_CL_VPORT_YSCALE_1 = 0x117
regPA_CL_VPORT_YSCALE_10 = 0x14d
regPA_CL_VPORT_YSCALE_11 = 0x153
regPA_CL_VPORT_YSCALE_12 = 0x159
regPA_CL_VPORT_YSCALE_13 = 0x15f
regPA_CL_VPORT_YSCALE_14 = 0x165
regPA_CL_VPORT_YSCALE_15 = 0x16b
regPA_CL_VPORT_YSCALE_2 = 0x11d
regPA_CL_VPORT_YSCALE_3 = 0x123
regPA_CL_VPORT_YSCALE_4 = 0x129
regPA_CL_VPORT_YSCALE_5 = 0x12f
regPA_CL_VPORT_YSCALE_6 = 0x135
regPA_CL_VPORT_YSCALE_7 = 0x13b
regPA_CL_VPORT_YSCALE_8 = 0x141
regPA_CL_VPORT_YSCALE_9 = 0x147
regPA_CL_VPORT_ZOFFSET = 0x114
regPA_CL_VPORT_ZOFFSET_1 = 0x11a
regPA_CL_VPORT_ZOFFSET_10 = 0x150
regPA_CL_VPORT_ZOFFSET_11 = 0x156
regPA_CL_VPORT_ZOFFSET_12 = 0x15c
regPA_CL_VPORT_ZOFFSET_13 = 0x162
regPA_CL_VPORT_ZOFFSET_14 = 0x168
regPA_CL_VPORT_ZOFFSET_15 = 0x16e
regPA_CL_VPORT_ZOFFSET_2 = 0x120
regPA_CL_VPORT_ZOFFSET_3 = 0x126
regPA_CL_VPORT_ZOFFSET_4 = 0x12c
regPA_CL_VPORT_ZOFFSET_5 = 0x132
regPA_CL_VPORT_ZOFFSET_6 = 0x138
regPA_CL_VPORT_ZOFFSET_7 = 0x13e
regPA_CL_VPORT_ZOFFSET_8 = 0x144
regPA_CL_VPORT_ZOFFSET_9 = 0x14a
regPA_CL_VPORT_ZSCALE = 0x113
regPA_CL_VPORT_ZSCALE_1 = 0x119
regPA_CL_VPORT_ZSCALE_10 = 0x14f
regPA_CL_VPORT_ZSCALE_11 = 0x155
regPA_CL_VPORT_ZSCALE_12 = 0x15b
regPA_CL_VPORT_ZSCALE_13 = 0x161
regPA_CL_VPORT_ZSCALE_14 = 0x167
regPA_CL_VPORT_ZSCALE_15 = 0x16d
regPA_CL_VPORT_ZSCALE_2 = 0x11f
regPA_CL_VPORT_ZSCALE_3 = 0x125
regPA_CL_VPORT_ZSCALE_4 = 0x12b
regPA_CL_VPORT_ZSCALE_5 = 0x131
regPA_CL_VPORT_ZSCALE_6 = 0x137
regPA_CL_VPORT_ZSCALE_7 = 0x13d
regPA_CL_VPORT_ZSCALE_8 = 0x143
regPA_CL_VPORT_ZSCALE_9 = 0x149
regPA_CL_VRS_CNTL = 0x212
regPA_CL_VS_OUT_CNTL = 0x207
regPA_CL_VTE_CNTL = 0x206
regPA_PH_ENHANCE = 0x95f
regPA_PH_INTERFACE_FIFO_SIZE = 0x95e
regPA_PH_PERFCOUNTER0_HI = 0x3581
regPA_PH_PERFCOUNTER0_LO = 0x3580
regPA_PH_PERFCOUNTER0_SELECT = 0x3d80
regPA_PH_PERFCOUNTER0_SELECT1 = 0x3d81
regPA_PH_PERFCOUNTER1_HI = 0x3583
regPA_PH_PERFCOUNTER1_LO = 0x3582
regPA_PH_PERFCOUNTER1_SELECT = 0x3d82
regPA_PH_PERFCOUNTER1_SELECT1 = 0x3d90
regPA_PH_PERFCOUNTER2_HI = 0x3585
regPA_PH_PERFCOUNTER2_LO = 0x3584
regPA_PH_PERFCOUNTER2_SELECT = 0x3d83
regPA_PH_PERFCOUNTER2_SELECT1 = 0x3d91
regPA_PH_PERFCOUNTER3_HI = 0x3587
regPA_PH_PERFCOUNTER3_LO = 0x3586
regPA_PH_PERFCOUNTER3_SELECT = 0x3d84
regPA_PH_PERFCOUNTER3_SELECT1 = 0x3d92
regPA_PH_PERFCOUNTER4_HI = 0x3589
regPA_PH_PERFCOUNTER4_LO = 0x3588
regPA_PH_PERFCOUNTER4_SELECT = 0x3d85
regPA_PH_PERFCOUNTER5_HI = 0x358b
regPA_PH_PERFCOUNTER5_LO = 0x358a
regPA_PH_PERFCOUNTER5_SELECT = 0x3d86
regPA_PH_PERFCOUNTER6_HI = 0x358d
regPA_PH_PERFCOUNTER6_LO = 0x358c
regPA_PH_PERFCOUNTER6_SELECT = 0x3d87
regPA_PH_PERFCOUNTER7_HI = 0x358f
regPA_PH_PERFCOUNTER7_LO = 0x358e
regPA_PH_PERFCOUNTER7_SELECT = 0x3d88
regPA_RATE_CNTL = 0x188
regPA_SC_AA_CONFIG = 0x2f8
regPA_SC_AA_MASK_X0Y0_X1Y0 = 0x30e
regPA_SC_AA_MASK_X0Y1_X1Y1 = 0x30f
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0 = 0x2fe
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_1 = 0x2ff
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_2 = 0x300
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_3 = 0x301
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y1_0 = 0x306
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y1_1 = 0x307
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y1_2 = 0x308
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y1_3 = 0x309
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y0_0 = 0x302
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y0_1 = 0x303
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y0_2 = 0x304
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y0_3 = 0x305
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y1_0 = 0x30a
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y1_1 = 0x30b
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y1_2 = 0x30c
regPA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y1_3 = 0x30d
regPA_SC_ATM_CNTL = 0x94d
regPA_SC_BINNER_CNTL_0 = 0x311
regPA_SC_BINNER_CNTL_1 = 0x312
regPA_SC_BINNER_CNTL_2 = 0x315
regPA_SC_BINNER_CNTL_OVERRIDE = 0x946
regPA_SC_BINNER_EVENT_CNTL_0 = 0x950
regPA_SC_BINNER_EVENT_CNTL_1 = 0x951
regPA_SC_BINNER_EVENT_CNTL_2 = 0x952
regPA_SC_BINNER_EVENT_CNTL_3 = 0x953
regPA_SC_BINNER_PERF_CNTL_0 = 0x955
regPA_SC_BINNER_PERF_CNTL_1 = 0x956
regPA_SC_BINNER_PERF_CNTL_2 = 0x957
regPA_SC_BINNER_PERF_CNTL_3 = 0x958
regPA_SC_BINNER_TIMEOUT_COUNTER = 0x954
regPA_SC_CENTROID_PRIORITY_0 = 0x2f5
regPA_SC_CENTROID_PRIORITY_1 = 0x2f6
regPA_SC_CLIPRECT_0_BR = 0x85
regPA_SC_CLIPRECT_0_TL = 0x84
regPA_SC_CLIPRECT_1_BR = 0x87
regPA_SC_CLIPRECT_1_TL = 0x86
regPA_SC_CLIPRECT_2_BR = 0x89
regPA_SC_CLIPRECT_2_TL = 0x88
regPA_SC_CLIPRECT_3_BR = 0x8b
regPA_SC_CLIPRECT_3_TL = 0x8a
regPA_SC_CLIPRECT_RULE = 0x83
regPA_SC_CONSERVATIVE_RASTERIZATION_CNTL = 0x313
regPA_SC_DSM_CNTL = 0x948
regPA_SC_EDGERULE = 0x8c
regPA_SC_ENHANCE = 0x941
regPA_SC_ENHANCE_1 = 0x942
regPA_SC_ENHANCE_2 = 0x943
regPA_SC_ENHANCE_3 = 0x944
regPA_SC_FIFO_DEPTH_CNTL = 0x1035
regPA_SC_FIFO_SIZE = 0x94a
regPA_SC_FORCE_EOV_MAX_CNTS = 0x94f
regPA_SC_GENERIC_SCISSOR_BR = 0x91
regPA_SC_GENERIC_SCISSOR_TL = 0x90
regPA_SC_HP3D_TRAP_SCREEN_COUNT = 0x22ac
regPA_SC_HP3D_TRAP_SCREEN_H = 0x22a9
regPA_SC_HP3D_TRAP_SCREEN_HV_EN = 0x22a8
regPA_SC_HP3D_TRAP_SCREEN_HV_LOCK = 0x95c
regPA_SC_HP3D_TRAP_SCREEN_OCCURRENCE = 0x22ab
regPA_SC_HP3D_TRAP_SCREEN_V = 0x22aa
regPA_SC_IF_FIFO_SIZE = 0x94b
regPA_SC_LINE_CNTL = 0x2f7
regPA_SC_LINE_STIPPLE = 0x283
regPA_SC_LINE_STIPPLE_STATE = 0x2281
regPA_SC_MODE_CNTL_0 = 0x292
regPA_SC_MODE_CNTL_1 = 0x293
regPA_SC_NGG_MODE_CNTL = 0x314
regPA_SC_P3D_TRAP_SCREEN_COUNT = 0x22a4
regPA_SC_P3D_TRAP_SCREEN_H = 0x22a1
regPA_SC_P3D_TRAP_SCREEN_HV_EN = 0x22a0
regPA_SC_P3D_TRAP_SCREEN_HV_LOCK = 0x95b
regPA_SC_P3D_TRAP_SCREEN_OCCURRENCE = 0x22a3
regPA_SC_P3D_TRAP_SCREEN_V = 0x22a2
regPA_SC_PACKER_WAVE_ID_CNTL = 0x94c
regPA_SC_PBB_OVERRIDE_FLAG = 0x947
regPA_SC_PERFCOUNTER0_HI = 0x3141
regPA_SC_PERFCOUNTER0_LO = 0x3140
regPA_SC_PERFCOUNTER0_SELECT = 0x3940
regPA_SC_PERFCOUNTER0_SELECT1 = 0x3941
regPA_SC_PERFCOUNTER1_HI = 0x3143
regPA_SC_PERFCOUNTER1_LO = 0x3142
regPA_SC_PERFCOUNTER1_SELECT = 0x3942
regPA_SC_PERFCOUNTER2_HI = 0x3145
regPA_SC_PERFCOUNTER2_LO = 0x3144
regPA_SC_PERFCOUNTER2_SELECT = 0x3943
regPA_SC_PERFCOUNTER3_HI = 0x3147
regPA_SC_PERFCOUNTER3_LO = 0x3146
regPA_SC_PERFCOUNTER3_SELECT = 0x3944
regPA_SC_PERFCOUNTER4_HI = 0x3149
regPA_SC_PERFCOUNTER4_LO = 0x3148
regPA_SC_PERFCOUNTER4_SELECT = 0x3945
regPA_SC_PERFCOUNTER5_HI = 0x314b
regPA_SC_PERFCOUNTER5_LO = 0x314a
regPA_SC_PERFCOUNTER5_SELECT = 0x3946
regPA_SC_PERFCOUNTER6_HI = 0x314d
regPA_SC_PERFCOUNTER6_LO = 0x314c
regPA_SC_PERFCOUNTER6_SELECT = 0x3947
regPA_SC_PERFCOUNTER7_HI = 0x314f
regPA_SC_PERFCOUNTER7_LO = 0x314e
regPA_SC_PERFCOUNTER7_SELECT = 0x3948
regPA_SC_PKR_WAVE_TABLE_CNTL = 0x94e
regPA_SC_RASTER_CONFIG = 0xd4
regPA_SC_RASTER_CONFIG_1 = 0xd5
regPA_SC_SCREEN_EXTENT_CONTROL = 0xd6
regPA_SC_SCREEN_EXTENT_MAX_0 = 0x2285
regPA_SC_SCREEN_EXTENT_MAX_1 = 0x228b
regPA_SC_SCREEN_EXTENT_MIN_0 = 0x2284
regPA_SC_SCREEN_EXTENT_MIN_1 = 0x2286
regPA_SC_SCREEN_SCISSOR_BR = 0xd
regPA_SC_SCREEN_SCISSOR_TL = 0xc
regPA_SC_SHADER_CONTROL = 0x310
regPA_SC_TILE_STEERING_CREST_OVERRIDE = 0x949
regPA_SC_TILE_STEERING_OVERRIDE = 0xd7
regPA_SC_TRAP_SCREEN_COUNT = 0x22b4
regPA_SC_TRAP_SCREEN_H = 0x22b1
regPA_SC_TRAP_SCREEN_HV_EN = 0x22b0
regPA_SC_TRAP_SCREEN_HV_LOCK = 0x95d
regPA_SC_TRAP_SCREEN_OCCURRENCE = 0x22b3
regPA_SC_TRAP_SCREEN_V = 0x22b2
regPA_SC_VPORT_SCISSOR_0_BR = 0x95
regPA_SC_VPORT_SCISSOR_0_TL = 0x94
regPA_SC_VPORT_SCISSOR_10_BR = 0xa9
regPA_SC_VPORT_SCISSOR_10_TL = 0xa8
regPA_SC_VPORT_SCISSOR_11_BR = 0xab
regPA_SC_VPORT_SCISSOR_11_TL = 0xaa
regPA_SC_VPORT_SCISSOR_12_BR = 0xad
regPA_SC_VPORT_SCISSOR_12_TL = 0xac
regPA_SC_VPORT_SCISSOR_13_BR = 0xaf
regPA_SC_VPORT_SCISSOR_13_TL = 0xae
regPA_SC_VPORT_SCISSOR_14_BR = 0xb1
regPA_SC_VPORT_SCISSOR_14_TL = 0xb0
regPA_SC_VPORT_SCISSOR_15_BR = 0xb3
regPA_SC_VPORT_SCISSOR_15_TL = 0xb2
regPA_SC_VPORT_SCISSOR_1_BR = 0x97
regPA_SC_VPORT_SCISSOR_1_TL = 0x96
regPA_SC_VPORT_SCISSOR_2_BR = 0x99
regPA_SC_VPORT_SCISSOR_2_TL = 0x98
regPA_SC_VPORT_SCISSOR_3_BR = 0x9b
regPA_SC_VPORT_SCISSOR_3_TL = 0x9a
regPA_SC_VPORT_SCISSOR_4_BR = 0x9d
regPA_SC_VPORT_SCISSOR_4_TL = 0x9c
regPA_SC_VPORT_SCISSOR_5_BR = 0x9f
regPA_SC_VPORT_SCISSOR_5_TL = 0x9e
regPA_SC_VPORT_SCISSOR_6_BR = 0xa1
regPA_SC_VPORT_SCISSOR_6_TL = 0xa0
regPA_SC_VPORT_SCISSOR_7_BR = 0xa3
regPA_SC_VPORT_SCISSOR_7_TL = 0xa2
regPA_SC_VPORT_SCISSOR_8_BR = 0xa5
regPA_SC_VPORT_SCISSOR_8_TL = 0xa4
regPA_SC_VPORT_SCISSOR_9_BR = 0xa7
regPA_SC_VPORT_SCISSOR_9_TL = 0xa6
regPA_SC_VPORT_ZMAX_0 = 0xb5
regPA_SC_VPORT_ZMAX_1 = 0xb7
regPA_SC_VPORT_ZMAX_10 = 0xc9
regPA_SC_VPORT_ZMAX_11 = 0xcb
regPA_SC_VPORT_ZMAX_12 = 0xcd
regPA_SC_VPORT_ZMAX_13 = 0xcf
regPA_SC_VPORT_ZMAX_14 = 0xd1
regPA_SC_VPORT_ZMAX_15 = 0xd3
regPA_SC_VPORT_ZMAX_2 = 0xb9
regPA_SC_VPORT_ZMAX_3 = 0xbb
regPA_SC_VPORT_ZMAX_4 = 0xbd
regPA_SC_VPORT_ZMAX_5 = 0xbf
regPA_SC_VPORT_ZMAX_6 = 0xc1
regPA_SC_VPORT_ZMAX_7 = 0xc3
regPA_SC_VPORT_ZMAX_8 = 0xc5
regPA_SC_VPORT_ZMAX_9 = 0xc7
regPA_SC_VPORT_ZMIN_0 = 0xb4
regPA_SC_VPORT_ZMIN_1 = 0xb6
regPA_SC_VPORT_ZMIN_10 = 0xc8
regPA_SC_VPORT_ZMIN_11 = 0xca
regPA_SC_VPORT_ZMIN_12 = 0xcc
regPA_SC_VPORT_ZMIN_13 = 0xce
regPA_SC_VPORT_ZMIN_14 = 0xd0
regPA_SC_VPORT_ZMIN_15 = 0xd2
regPA_SC_VPORT_ZMIN_2 = 0xb8
regPA_SC_VPORT_ZMIN_3 = 0xba
regPA_SC_VPORT_ZMIN_4 = 0xbc
regPA_SC_VPORT_ZMIN_5 = 0xbe
regPA_SC_VPORT_ZMIN_6 = 0xc0
regPA_SC_VPORT_ZMIN_7 = 0xc2
regPA_SC_VPORT_ZMIN_8 = 0xc4
regPA_SC_VPORT_ZMIN_9 = 0xc6
regPA_SC_VRS_OVERRIDE_CNTL = 0xf4
regPA_SC_VRS_RATE_BASE = 0xfc
regPA_SC_VRS_RATE_BASE_EXT = 0xfd
regPA_SC_VRS_RATE_CACHE_CNTL = 0xf9
regPA_SC_VRS_RATE_FEEDBACK_BASE = 0xf5
regPA_SC_VRS_RATE_FEEDBACK_BASE_EXT = 0xf6
regPA_SC_VRS_RATE_FEEDBACK_SIZE_XY = 0xf7
regPA_SC_VRS_RATE_SIZE_XY = 0xfe
regPA_SC_VRS_SURFACE_CNTL = 0x940
regPA_SC_VRS_SURFACE_CNTL_1 = 0x960
regPA_SC_WINDOW_OFFSET = 0x80
regPA_SC_WINDOW_SCISSOR_BR = 0x82
regPA_SC_WINDOW_SCISSOR_TL = 0x81
regPA_STATE_STEREO_X = 0x211
regPA_STEREO_CNTL = 0x210
regPA_SU_CNTL_STATUS = 0x1034
regPA_SU_HARDWARE_SCREEN_OFFSET = 0x8d
regPA_SU_LINE_CNTL = 0x282
regPA_SU_LINE_STIPPLE_CNTL = 0x209
regPA_SU_LINE_STIPPLE_SCALE = 0x20a
regPA_SU_LINE_STIPPLE_VALUE = 0x2280
regPA_SU_OVER_RASTERIZATION_CNTL = 0x20f
regPA_SU_PERFCOUNTER0_HI = 0x3101
regPA_SU_PERFCOUNTER0_LO = 0x3100
regPA_SU_PERFCOUNTER0_SELECT = 0x3900
regPA_SU_PERFCOUNTER0_SELECT1 = 0x3901
regPA_SU_PERFCOUNTER1_HI = 0x3103
regPA_SU_PERFCOUNTER1_LO = 0x3102
regPA_SU_PERFCOUNTER1_SELECT = 0x3902
regPA_SU_PERFCOUNTER1_SELECT1 = 0x3903
regPA_SU_PERFCOUNTER2_HI = 0x3105
regPA_SU_PERFCOUNTER2_LO = 0x3104
regPA_SU_PERFCOUNTER2_SELECT = 0x3904
regPA_SU_PERFCOUNTER2_SELECT1 = 0x3905
regPA_SU_PERFCOUNTER3_HI = 0x3107
regPA_SU_PERFCOUNTER3_LO = 0x3106
regPA_SU_PERFCOUNTER3_SELECT = 0x3906
regPA_SU_PERFCOUNTER3_SELECT1 = 0x3907
regPA_SU_POINT_MINMAX = 0x281
regPA_SU_POINT_SIZE = 0x280
regPA_SU_POLY_OFFSET_BACK_OFFSET = 0x2e3
regPA_SU_POLY_OFFSET_BACK_SCALE = 0x2e2
regPA_SU_POLY_OFFSET_CLAMP = 0x2df
regPA_SU_POLY_OFFSET_DB_FMT_CNTL = 0x2de
regPA_SU_POLY_OFFSET_FRONT_OFFSET = 0x2e1
regPA_SU_POLY_OFFSET_FRONT_SCALE = 0x2e0
regPA_SU_PRIM_FILTER_CNTL = 0x20b
regPA_SU_SC_MODE_CNTL = 0x205
regPA_SU_SMALL_PRIM_FILTER_CNTL = 0x20c
regPA_SU_VTX_CNTL = 0x2f9
regPCC_PERF_COUNTER = 0x1b0c
regPCC_PWRBRK_HYSTERESIS_CTRL = 0x1b03
regPCC_STALL_PATTERN_1_2 = 0x1af6
regPCC_STALL_PATTERN_3_4 = 0x1af7
regPCC_STALL_PATTERN_5_6 = 0x1af8
regPCC_STALL_PATTERN_7 = 0x1af9
regPCC_STALL_PATTERN_CTRL = 0x1af4
regPC_PERFCOUNTER0_HI = 0x318c
regPC_PERFCOUNTER0_LO = 0x318d
regPC_PERFCOUNTER0_SELECT = 0x398c
regPC_PERFCOUNTER0_SELECT1 = 0x3990
regPC_PERFCOUNTER1_HI = 0x318e
regPC_PERFCOUNTER1_LO = 0x318f
regPC_PERFCOUNTER1_SELECT = 0x398d
regPC_PERFCOUNTER1_SELECT1 = 0x3991
regPC_PERFCOUNTER2_HI = 0x3190
regPC_PERFCOUNTER2_LO = 0x3191
regPC_PERFCOUNTER2_SELECT = 0x398e
regPC_PERFCOUNTER2_SELECT1 = 0x3992
regPC_PERFCOUNTER3_HI = 0x3192
regPC_PERFCOUNTER3_LO = 0x3193
regPC_PERFCOUNTER3_SELECT = 0x398f
regPC_PERFCOUNTER3_SELECT1 = 0x3993
regPMM_CNTL = 0x1582
regPMM_CNTL2 = 0x1999
regPMM_STATUS = 0x1583
regPWRBRK_PERF_COUNTER = 0x1b0d
regPWRBRK_STALL_PATTERN_1_2 = 0x1afa
regPWRBRK_STALL_PATTERN_3_4 = 0x1afb
regPWRBRK_STALL_PATTERN_5_6 = 0x1afc
regPWRBRK_STALL_PATTERN_7 = 0x1afd
regPWRBRK_STALL_PATTERN_CTRL = 0x1af5
regRLC_AUTO_PG_CTRL = 0x4c55
regRLC_BUSY_CLK_CNTL = 0x5b30
regRLC_CAC_MASK_CNTL = 0x4d45
regRLC_CAPTURE_GPU_CLOCK_COUNT = 0x4c26
regRLC_CAPTURE_GPU_CLOCK_COUNT_1 = 0x4cea
regRLC_CAPTURE_GPU_CLOCK_COUNT_2 = 0x4cef
regRLC_CGCG_CGLS_CTRL = 0x4c49
regRLC_CGCG_CGLS_CTRL_3D = 0x4cc5
regRLC_CGCG_RAMP_CTRL = 0x4c4a
regRLC_CGCG_RAMP_CTRL_3D = 0x4cc6
regRLC_CGTT_MGCG_OVERRIDE = 0x4c48
regRLC_CLK_CNTL = 0x5b31
regRLC_CLK_COUNT_CTRL = 0x4c34
regRLC_CLK_COUNT_GFXCLK_LSB = 0x4c30
regRLC_CLK_COUNT_GFXCLK_MSB = 0x4c31
regRLC_CLK_COUNT_REFCLK_LSB = 0x4c32
regRLC_CLK_COUNT_REFCLK_MSB = 0x4c33
regRLC_CLK_COUNT_STAT = 0x4c35
regRLC_CLK_RESIDENCY_CNTR_CTRL = 0x4d49
regRLC_CLK_RESIDENCY_EVENT_CNTR = 0x4d51
regRLC_CLK_RESIDENCY_REF_CNTR = 0x4d59
regRLC_CNTL = 0x4c00
regRLC_CP_EOF_INT = 0x98b
regRLC_CP_EOF_INT_CNT = 0x98c
regRLC_CP_SCHEDULERS = 0x98a
regRLC_CP_STAT_INVAL_CTRL = 0x4d0a
regRLC_CP_STAT_INVAL_STAT = 0x4d09
regRLC_CSIB_ADDR_HI = 0x988
regRLC_CSIB_ADDR_LO = 0x987
regRLC_CSIB_LENGTH = 0x989
regRLC_DS_RESIDENCY_CNTR_CTRL = 0x4d4a
regRLC_DS_RESIDENCY_EVENT_CNTR = 0x4d52
regRLC_DS_RESIDENCY_REF_CNTR = 0x4d5a
regRLC_DYN_PG_REQUEST = 0x4c4c
regRLC_DYN_PG_STATUS = 0x4c4b
regRLC_F32_UCODE_VERSION = 0x4c03
regRLC_FWL_FIRST_VIOL_ADDR = 0x5f26
regRLC_GENERAL_RESIDENCY_CNTR_CTRL = 0x4d4d
regRLC_GENERAL_RESIDENCY_EVENT_CNTR = 0x4d55
regRLC_GENERAL_RESIDENCY_REF_CNTR = 0x4d5d
regRLC_GFX_IH_ARBITER_STAT = 0x4d5f
regRLC_GFX_IH_CLIENT_CTRL = 0x4d5e
regRLC_GFX_IH_CLIENT_OTHER_STAT = 0x4d63
regRLC_GFX_IH_CLIENT_SDMA_STAT = 0x4d62
regRLC_GFX_IH_CLIENT_SE_STAT_H = 0x4d61
regRLC_GFX_IH_CLIENT_SE_STAT_L = 0x4d60
regRLC_GFX_IMU_CMD = 0x4053
regRLC_GFX_IMU_DATA_0 = 0x4052
regRLC_GPM_CP_DMA_COMPLETE_T0 = 0x4c29
regRLC_GPM_CP_DMA_COMPLETE_T1 = 0x4c2a
regRLC_GPM_GENERAL_0 = 0x4c63
regRLC_GPM_GENERAL_1 = 0x4c64
regRLC_GPM_GENERAL_10 = 0x4caf
regRLC_GPM_GENERAL_11 = 0x4cb0
regRLC_GPM_GENERAL_12 = 0x4cb1
regRLC_GPM_GENERAL_13 = 0x4cdd
regRLC_GPM_GENERAL_14 = 0x4cde
regRLC_GPM_GENERAL_15 = 0x4cdf
regRLC_GPM_GENERAL_16 = 0x4c76
regRLC_GPM_GENERAL_2 = 0x4c65
regRLC_GPM_GENERAL_3 = 0x4c66
regRLC_GPM_GENERAL_4 = 0x4c67
regRLC_GPM_GENERAL_5 = 0x4c68
regRLC_GPM_GENERAL_6 = 0x4c69
regRLC_GPM_GENERAL_7 = 0x4c6a
regRLC_GPM_GENERAL_8 = 0x4cad
regRLC_GPM_GENERAL_9 = 0x4cae
regRLC_GPM_INT_DISABLE_TH0 = 0x4c7c
regRLC_GPM_INT_FORCE_TH0 = 0x4c7e
regRLC_GPM_INT_STAT_TH0 = 0x4cdc
regRLC_GPM_IRAM_ADDR = 0x5b62
regRLC_GPM_IRAM_DATA = 0x5b63
regRLC_GPM_LEGACY_INT_CLEAR = 0x4c17
regRLC_GPM_LEGACY_INT_DISABLE = 0x4c7d
regRLC_GPM_LEGACY_INT_STAT = 0x4c16
regRLC_GPM_PERF_COUNT_0 = 0x2140
regRLC_GPM_PERF_COUNT_1 = 0x2141
regRLC_GPM_SCRATCH_ADDR = 0x5b6e
regRLC_GPM_SCRATCH_DATA = 0x5b6f
regRLC_GPM_STAT = 0x4e6b
regRLC_GPM_THREAD_ENABLE = 0x4c45
regRLC_GPM_THREAD_INVALIDATE_CACHE = 0x4c2b
regRLC_GPM_THREAD_PRIORITY = 0x4c44
regRLC_GPM_THREAD_RESET = 0x4c28
regRLC_GPM_TIMER_CTRL = 0x4c13
regRLC_GPM_TIMER_INT_0 = 0x4c0e
regRLC_GPM_TIMER_INT_1 = 0x4c0f
regRLC_GPM_TIMER_INT_2 = 0x4c10
regRLC_GPM_TIMER_INT_3 = 0x4c11
regRLC_GPM_TIMER_INT_4 = 0x4c12
regRLC_GPM_TIMER_STAT = 0x4c14
regRLC_GPM_UCODE_ADDR = 0x5b60
regRLC_GPM_UCODE_DATA = 0x5b61
regRLC_GPM_UTCL1_CNTL_0 = 0x4cb2
regRLC_GPM_UTCL1_CNTL_1 = 0x4cb3
regRLC_GPM_UTCL1_CNTL_2 = 0x4cb4
regRLC_GPM_UTCL1_TH0_ERROR_1 = 0x4cbe
regRLC_GPM_UTCL1_TH0_ERROR_2 = 0x4cc0
regRLC_GPM_UTCL1_TH1_ERROR_1 = 0x4cc1
regRLC_GPM_UTCL1_TH1_ERROR_2 = 0x4cc2
regRLC_GPM_UTCL1_TH2_ERROR_1 = 0x4cc3
regRLC_GPM_UTCL1_TH2_ERROR_2 = 0x4cc4
regRLC_GPR_REG1 = 0x4c79
regRLC_GPR_REG2 = 0x4c7a
regRLC_GPU_CLOCK_32 = 0x4c42
regRLC_GPU_CLOCK_32_RES_SEL = 0x4c41
regRLC_GPU_CLOCK_COUNT_LSB = 0x4c24
regRLC_GPU_CLOCK_COUNT_LSB_1 = 0x4cfb
regRLC_GPU_CLOCK_COUNT_LSB_2 = 0x4ceb
regRLC_GPU_CLOCK_COUNT_MSB = 0x4c25
regRLC_GPU_CLOCK_COUNT_MSB_1 = 0x4cfc
regRLC_GPU_CLOCK_COUNT_MSB_2 = 0x4cec
regRLC_GPU_CLOCK_COUNT_SPM_LSB = 0x4de4
regRLC_GPU_CLOCK_COUNT_SPM_MSB = 0x4de5
regRLC_GPU_IOV_CFG_REG1 = 0x5b35
regRLC_GPU_IOV_CFG_REG2 = 0x5b36
regRLC_GPU_IOV_CFG_REG6 = 0x5b06
regRLC_GPU_IOV_CFG_REG8 = 0x5b20
regRLC_GPU_IOV_F32_CNTL = 0x5b46
regRLC_GPU_IOV_F32_INVALIDATE_CACHE = 0x5b4b
regRLC_GPU_IOV_F32_RESET = 0x5b47
regRLC_GPU_IOV_INT_DISABLE = 0x5b4e
regRLC_GPU_IOV_INT_FORCE = 0x5b4f
regRLC_GPU_IOV_INT_STAT = 0x5b3f
regRLC_GPU_IOV_PERF_CNT_CNTL = 0x3cc3
regRLC_GPU_IOV_PERF_CNT_RD_ADDR = 0x3cc6
regRLC_GPU_IOV_PERF_CNT_RD_DATA = 0x3cc7
regRLC_GPU_IOV_PERF_CNT_WR_ADDR = 0x3cc4
regRLC_GPU_IOV_PERF_CNT_WR_DATA = 0x3cc5
regRLC_GPU_IOV_RLC_RESPONSE = 0x5b4d
regRLC_GPU_IOV_SCH_0 = 0x5b38
regRLC_GPU_IOV_SCH_1 = 0x5b3b
regRLC_GPU_IOV_SCH_2 = 0x5b3c
regRLC_GPU_IOV_SCH_3 = 0x5b3a
regRLC_GPU_IOV_SCH_BLOCK = 0x5b34
regRLC_GPU_IOV_SCRATCH_ADDR = 0x5b50
regRLC_GPU_IOV_SCRATCH_DATA = 0x5b51
regRLC_GPU_IOV_SDMA0_BUSY_STATUS = 0x5bc8
regRLC_GPU_IOV_SDMA0_STATUS = 0x5bc0
regRLC_GPU_IOV_SDMA1_BUSY_STATUS = 0x5bc9
regRLC_GPU_IOV_SDMA1_STATUS = 0x5bc1
regRLC_GPU_IOV_SDMA2_BUSY_STATUS = 0x5bca
regRLC_GPU_IOV_SDMA2_STATUS = 0x5bc2
regRLC_GPU_IOV_SDMA3_BUSY_STATUS = 0x5bcb
regRLC_GPU_IOV_SDMA3_STATUS = 0x5bc3
regRLC_GPU_IOV_SDMA4_BUSY_STATUS = 0x5bcc
regRLC_GPU_IOV_SDMA4_STATUS = 0x5bc4
regRLC_GPU_IOV_SDMA5_BUSY_STATUS = 0x5bcd
regRLC_GPU_IOV_SDMA5_STATUS = 0x5bc5
regRLC_GPU_IOV_SDMA6_BUSY_STATUS = 0x5bce
regRLC_GPU_IOV_SDMA6_STATUS = 0x5bc6
regRLC_GPU_IOV_SDMA7_BUSY_STATUS = 0x5bcf
regRLC_GPU_IOV_SDMA7_STATUS = 0x5bc7
regRLC_GPU_IOV_SMU_RESPONSE = 0x5b4a
regRLC_GPU_IOV_UCODE_ADDR = 0x5b48
regRLC_GPU_IOV_UCODE_DATA = 0x5b49
regRLC_GPU_IOV_VF_DOORBELL_STATUS = 0x5b2a
regRLC_GPU_IOV_VF_DOORBELL_STATUS_CLR = 0x5b2c
regRLC_GPU_IOV_VF_DOORBELL_STATUS_SET = 0x5b2b
regRLC_GPU_IOV_VF_ENABLE = 0x5b00
regRLC_GPU_IOV_VF_MASK = 0x5b2d
regRLC_GPU_IOV_VM_BUSY_STATUS = 0x5b37
regRLC_GTS_OFFSET_LSB = 0x5b79
regRLC_GTS_OFFSET_MSB = 0x5b7a
regRLC_HYP_RLCG_UCODE_CHKSUM = 0x5b43
regRLC_HYP_RLCP_UCODE_CHKSUM = 0x5b44
regRLC_HYP_RLCV_UCODE_CHKSUM = 0x5b45
regRLC_HYP_SEMAPHORE_0 = 0x5b2e
regRLC_HYP_SEMAPHORE_1 = 0x5b2f
regRLC_HYP_SEMAPHORE_2 = 0x5b52
regRLC_HYP_SEMAPHORE_3 = 0x5b53
regRLC_IH_COOKIE = 0x5b41
regRLC_IH_COOKIE_CNTL = 0x5b42
regRLC_IMU_BOOTLOAD_ADDR_HI = 0x4e10
regRLC_IMU_BOOTLOAD_ADDR_LO = 0x4e11
regRLC_IMU_BOOTLOAD_SIZE = 0x4e12
regRLC_IMU_MISC = 0x4e16
regRLC_IMU_RESET_VECTOR = 0x4e17
regRLC_INT_STAT = 0x4c18
regRLC_JUMP_TABLE_RESTORE = 0x4c1e
regRLC_LX6_CNTL = 0x4d80
regRLC_LX6_DRAM_ADDR = 0x5b68
regRLC_LX6_DRAM_DATA = 0x5b69
regRLC_LX6_IRAM_ADDR = 0x5b6a
regRLC_LX6_IRAM_DATA = 0x5b6b
regRLC_MAX_PG_WGP = 0x4c54
regRLC_MEM_SLP_CNTL = 0x4e00
regRLC_MGCG_CTRL = 0x4c1a
regRLC_PACE_INT_CLEAR = 0x5b3e
regRLC_PACE_INT_DISABLE = 0x4ced
regRLC_PACE_INT_FORCE = 0x5b3d
regRLC_PACE_INT_STAT = 0x4ccc
regRLC_PACE_SCRATCH_ADDR = 0x5b77
regRLC_PACE_SCRATCH_DATA = 0x5b78
regRLC_PACE_SPARE_INT = 0x990
regRLC_PACE_SPARE_INT_1 = 0x991
regRLC_PACE_TIMER_CTRL = 0x4d06
regRLC_PACE_TIMER_INT_0 = 0x4d04
regRLC_PACE_TIMER_INT_1 = 0x4d05
regRLC_PACE_TIMER_STAT = 0x5b33
regRLC_PACE_UCODE_ADDR = 0x5b6c
regRLC_PACE_UCODE_DATA = 0x5b6d
regRLC_PCC_RESIDENCY_CNTR_CTRL = 0x4d4c
regRLC_PCC_RESIDENCY_EVENT_CNTR = 0x4d54
regRLC_PCC_RESIDENCY_REF_CNTR = 0x4d5c
regRLC_PERFCOUNTER0_HI = 0x3481
regRLC_PERFCOUNTER0_LO = 0x3480
regRLC_PERFCOUNTER0_SELECT = 0x3cc1
regRLC_PERFCOUNTER1_HI = 0x3483
regRLC_PERFCOUNTER1_LO = 0x3482
regRLC_PERFCOUNTER1_SELECT = 0x3cc2
regRLC_PERFMON_CNTL = 0x3cc0
regRLC_PG_ALWAYS_ON_WGP_MASK = 0x4c53
regRLC_PG_CNTL = 0x4c43
regRLC_PG_DELAY = 0x4c4d
regRLC_PG_DELAY_2 = 0x4c1f
regRLC_PG_DELAY_3 = 0x4c78
regRLC_POWER_RESIDENCY_CNTR_CTRL = 0x4d48
regRLC_POWER_RESIDENCY_EVENT_CNTR = 0x4d50
regRLC_POWER_RESIDENCY_REF_CNTR = 0x4d58
regRLC_R2I_CNTL_0 = 0x4cd5
regRLC_R2I_CNTL_1 = 0x4cd6
regRLC_R2I_CNTL_2 = 0x4cd7
regRLC_R2I_CNTL_3 = 0x4cd8
regRLC_REFCLOCK_TIMESTAMP_LSB = 0x4c0c
regRLC_REFCLOCK_TIMESTAMP_MSB = 0x4c0d
regRLC_RLCG_DOORBELL_0_DATA_HI = 0x4c39
regRLC_RLCG_DOORBELL_0_DATA_LO = 0x4c38
regRLC_RLCG_DOORBELL_1_DATA_HI = 0x4c3b
regRLC_RLCG_DOORBELL_1_DATA_LO = 0x4c3a
regRLC_RLCG_DOORBELL_2_DATA_HI = 0x4c3d
regRLC_RLCG_DOORBELL_2_DATA_LO = 0x4c3c
regRLC_RLCG_DOORBELL_3_DATA_HI = 0x4c3f
regRLC_RLCG_DOORBELL_3_DATA_LO = 0x4c3e
regRLC_RLCG_DOORBELL_CNTL = 0x4c36
regRLC_RLCG_DOORBELL_RANGE = 0x4c47
regRLC_RLCG_DOORBELL_STAT = 0x4c37
regRLC_RLCP_DOORBELL_0_DATA_HI = 0x4d2a
regRLC_RLCP_DOORBELL_0_DATA_LO = 0x4d29
regRLC_RLCP_DOORBELL_1_DATA_HI = 0x4d2c
regRLC_RLCP_DOORBELL_1_DATA_LO = 0x4d2b
regRLC_RLCP_DOORBELL_2_DATA_HI = 0x4d2e
regRLC_RLCP_DOORBELL_2_DATA_LO = 0x4d2d
regRLC_RLCP_DOORBELL_3_DATA_HI = 0x4d30
regRLC_RLCP_DOORBELL_3_DATA_LO = 0x4d2f
regRLC_RLCP_DOORBELL_CNTL = 0x4d27
regRLC_RLCP_DOORBELL_RANGE = 0x4d26
regRLC_RLCP_DOORBELL_STAT = 0x4d28
regRLC_RLCP_IRAM_ADDR = 0x5b64
regRLC_RLCP_IRAM_DATA = 0x5b65
regRLC_RLCS_ABORTED_PD_SEQUENCE = 0x4e6c
regRLC_RLCS_AUXILIARY_REG_1 = 0x4ec5
regRLC_RLCS_AUXILIARY_REG_2 = 0x4ec6
regRLC_RLCS_AUXILIARY_REG_3 = 0x4ec7
regRLC_RLCS_AUXILIARY_REG_4 = 0x4ec8
regRLC_RLCS_BOOTLOAD_ID_STATUS1 = 0x4ecb
regRLC_RLCS_BOOTLOAD_ID_STATUS2 = 0x4ecc
regRLC_RLCS_BOOTLOAD_STATUS = 0x4e82
regRLC_RLCS_CGCG_REQUEST = 0x4e66
regRLC_RLCS_CGCG_STATUS = 0x4e67
regRLC_RLCS_CMP_IDLE_CNTL = 0x4e87
regRLC_RLCS_CP_DMA_SRCID_OVER = 0x4eca
regRLC_RLCS_CP_INT_CTRL_1 = 0x4e7a
regRLC_RLCS_CP_INT_CTRL_2 = 0x4e7b
regRLC_RLCS_CP_INT_INFO_1 = 0x4e7c
regRLC_RLCS_CP_INT_INFO_2 = 0x4e7d
regRLC_RLCS_DEC_DUMP_ADDR = 0x4e61
regRLC_RLCS_DEC_END = 0x4fff
regRLC_RLCS_DEC_START = 0x4e60
regRLC_RLCS_DIDT_FORCE_STALL = 0x4e6d
regRLC_RLCS_DSM_TRIG = 0x4e81
regRLC_RLCS_EDC_INT_CNTL = 0x4ece
regRLC_RLCS_EXCEPTION_REG_1 = 0x4e62
regRLC_RLCS_EXCEPTION_REG_2 = 0x4e63
regRLC_RLCS_EXCEPTION_REG_3 = 0x4e64
regRLC_RLCS_EXCEPTION_REG_4 = 0x4e65
regRLC_RLCS_GCR_DATA_0 = 0x4ed4
regRLC_RLCS_GCR_DATA_1 = 0x4ed5
regRLC_RLCS_GCR_DATA_2 = 0x4ed6
regRLC_RLCS_GCR_DATA_3 = 0x4ed7
regRLC_RLCS_GCR_STATUS = 0x4ed8
regRLC_RLCS_GENERAL_0 = 0x4e88
regRLC_RLCS_GENERAL_1 = 0x4e89
regRLC_RLCS_GENERAL_10 = 0x4e92
regRLC_RLCS_GENERAL_11 = 0x4e93
regRLC_RLCS_GENERAL_12 = 0x4e94
regRLC_RLCS_GENERAL_13 = 0x4e95
regRLC_RLCS_GENERAL_14 = 0x4e96
regRLC_RLCS_GENERAL_15 = 0x4e97
regRLC_RLCS_GENERAL_16 = 0x4e98
regRLC_RLCS_GENERAL_2 = 0x4e8a
regRLC_RLCS_GENERAL_3 = 0x4e8b
regRLC_RLCS_GENERAL_4 = 0x4e8c
regRLC_RLCS_GENERAL_5 = 0x4e8d
regRLC_RLCS_GENERAL_6 = 0x4e8e
regRLC_RLCS_GENERAL_7 = 0x4e8f
regRLC_RLCS_GENERAL_8 = 0x4e90
regRLC_RLCS_GENERAL_9 = 0x4e91
regRLC_RLCS_GFX_DS_ALLOW_MASK_CNTL = 0x4e6a
regRLC_RLCS_GFX_DS_CNTL = 0x4e69
regRLC_RLCS_GFX_MEM_POWER_CTRL_LO = 0x4ef8
regRLC_RLCS_GFX_RM_CNTL = 0x4efa
regRLC_RLCS_GPM_LEGACY_INT_DISABLE = 0x4ed2
regRLC_RLCS_GPM_LEGACY_INT_STAT = 0x4ed1
regRLC_RLCS_GPM_STAT = 0x4e6b
regRLC_RLCS_GPM_STAT_2 = 0x4e72
regRLC_RLCS_GRBM_IDLE_BUSY_INT_CNTL = 0x4e86
regRLC_RLCS_GRBM_IDLE_BUSY_STAT = 0x4e85
regRLC_RLCS_GRBM_SOFT_RESET = 0x4e73
regRLC_RLCS_IH_COOKIE_SEMAPHORE = 0x4e77
regRLC_RLCS_IH_SEMAPHORE = 0x4e76
regRLC_RLCS_IMU_GFX_DOORBELL_FENCE = 0x4ef1
regRLC_RLCS_IMU_RAM_ADDR_0_LSB = 0x4eee
regRLC_RLCS_IMU_RAM_ADDR_0_MSB = 0x4eef
regRLC_RLCS_IMU_RAM_ADDR_1_LSB = 0x4eeb
regRLC_RLCS_IMU_RAM_ADDR_1_MSB = 0x4eec
regRLC_RLCS_IMU_RAM_CNTL = 0x4ef0
regRLC_RLCS_IMU_RAM_DATA_0 = 0x4eed
regRLC_RLCS_IMU_RAM_DATA_1 = 0x4eea
regRLC_RLCS_IMU_RLC_MSG_CNTL = 0x4ee1
regRLC_RLCS_IMU_RLC_MSG_CONTROL = 0x4ee0
regRLC_RLCS_IMU_RLC_MSG_DATA0 = 0x4edb
regRLC_RLCS_IMU_RLC_MSG_DATA1 = 0x4edc
regRLC_RLCS_IMU_RLC_MSG_DATA2 = 0x4edd
regRLC_RLCS_IMU_RLC_MSG_DATA3 = 0x4ede
regRLC_RLCS_IMU_RLC_MSG_DATA4 = 0x4edf
regRLC_RLCS_IMU_RLC_MUTEX_CNTL = 0x4ee7
regRLC_RLCS_IMU_RLC_STATUS = 0x4ee8
regRLC_RLCS_IMU_RLC_TELEMETRY_DATA_0 = 0x4ee5
regRLC_RLCS_IMU_RLC_TELEMETRY_DATA_1 = 0x4ee6
regRLC_RLCS_IMU_VIDCHG_CNTL = 0x4ecd
regRLC_RLCS_IOV_CMD_STATUS = 0x4e6e
regRLC_RLCS_IOV_CNTX_LOC_SIZE = 0x4e6f
regRLC_RLCS_IOV_SCH_BLOCK = 0x4e70
regRLC_RLCS_IOV_VM_BUSY_STATUS = 0x4e71
regRLC_RLCS_KMD_LOG_CNTL1 = 0x4ecf
regRLC_RLCS_KMD_LOG_CNTL2 = 0x4ed0
regRLC_RLCS_PERFMON_CLK_CNTL_UCODE = 0x4ed9
regRLC_RLCS_PG_CHANGE_READ = 0x4e75
regRLC_RLCS_PG_CHANGE_STATUS = 0x4e74
regRLC_RLCS_PMM_CGCG_CNTL = 0x4ef7
regRLC_RLCS_POWER_BRAKE_CNTL = 0x4e83
regRLC_RLCS_POWER_BRAKE_CNTL_TH1 = 0x4e84
regRLC_RLCS_RLC_IMU_MSG_CNTL = 0x4ee4
regRLC_RLCS_RLC_IMU_MSG_CONTROL = 0x4ee3
regRLC_RLCS_RLC_IMU_MSG_DATA0 = 0x4ee2
regRLC_RLCS_RLC_IMU_STATUS = 0x4ee9
regRLC_RLCS_SDMA_INT_CNTL_1 = 0x4ef3
regRLC_RLCS_SDMA_INT_CNTL_2 = 0x4ef4
regRLC_RLCS_SDMA_INT_INFO = 0x4ef6
regRLC_RLCS_SDMA_INT_STAT = 0x4ef5
regRLC_RLCS_SOC_DS_CNTL = 0x4e68
regRLC_RLCS_SPM_INT_CTRL = 0x4e7e
regRLC_RLCS_SPM_INT_INFO_1 = 0x4e7f
regRLC_RLCS_SPM_INT_INFO_2 = 0x4e80
regRLC_RLCS_SPM_SQTT_MODE = 0x4ec9
regRLC_RLCS_SRM_SRCID_CNTL = 0x4ed3
regRLC_RLCS_UTCL2_CNTL = 0x4eda
regRLC_RLCS_WGP_READ = 0x4e79
regRLC_RLCS_WGP_STATUS = 0x4e78
regRLC_RLCV_COMMAND = 0x4e04
regRLC_RLCV_DOORBELL_0_DATA_HI = 0x4cf4
regRLC_RLCV_DOORBELL_0_DATA_LO = 0x4cf3
regRLC_RLCV_DOORBELL_1_DATA_HI = 0x4cf6
regRLC_RLCV_DOORBELL_1_DATA_LO = 0x4cf5
regRLC_RLCV_DOORBELL_2_DATA_HI = 0x4cf8
regRLC_RLCV_DOORBELL_2_DATA_LO = 0x4cf7
regRLC_RLCV_DOORBELL_3_DATA_HI = 0x4cfa
regRLC_RLCV_DOORBELL_3_DATA_LO = 0x4cf9
regRLC_RLCV_DOORBELL_CNTL = 0x4cf1
regRLC_RLCV_DOORBELL_RANGE = 0x4cf0
regRLC_RLCV_DOORBELL_STAT = 0x4cf2
regRLC_RLCV_IRAM_ADDR = 0x5b66
regRLC_RLCV_IRAM_DATA = 0x5b67
regRLC_RLCV_SAFE_MODE = 0x4e02
regRLC_RLCV_SPARE_INT = 0x4d00
regRLC_RLCV_SPARE_INT_1 = 0x992
regRLC_RLCV_TIMER_CTRL = 0x5b27
regRLC_RLCV_TIMER_INT_0 = 0x5b25
regRLC_RLCV_TIMER_INT_1 = 0x5b26
regRLC_RLCV_TIMER_STAT = 0x5b28
regRLC_SAFE_MODE = 0x980
regRLC_SDMA0_BUSY_STATUS = 0x5b1c
regRLC_SDMA0_STATUS = 0x5b18
regRLC_SDMA1_BUSY_STATUS = 0x5b1d
regRLC_SDMA1_STATUS = 0x5b19
regRLC_SDMA2_BUSY_STATUS = 0x5b1e
regRLC_SDMA2_STATUS = 0x5b1a
regRLC_SDMA3_BUSY_STATUS = 0x5b1f
regRLC_SDMA3_STATUS = 0x5b1b
regRLC_SEMAPHORE_0 = 0x4cc7
regRLC_SEMAPHORE_1 = 0x4cc8
regRLC_SEMAPHORE_2 = 0x4cc9
regRLC_SEMAPHORE_3 = 0x4cca
regRLC_SERDES_BUSY = 0x4c61
regRLC_SERDES_CTRL = 0x4c5f
regRLC_SERDES_DATA = 0x4c60
regRLC_SERDES_MASK = 0x4c5e
regRLC_SERDES_RD_DATA_0 = 0x4c5a
regRLC_SERDES_RD_DATA_1 = 0x4c5b
regRLC_SERDES_RD_DATA_2 = 0x4c5c
regRLC_SERDES_RD_DATA_3 = 0x4c5d
regRLC_SERDES_RD_INDEX = 0x4c59
regRLC_SMU_ARGUMENT_1 = 0x4e0b
regRLC_SMU_ARGUMENT_2 = 0x4e0c
regRLC_SMU_ARGUMENT_3 = 0x4e0d
regRLC_SMU_ARGUMENT_4 = 0x4e0e
regRLC_SMU_ARGUMENT_5 = 0x4e0f
regRLC_SMU_CLK_REQ = 0x4d08
regRLC_SMU_COMMAND = 0x4e0a
regRLC_SMU_MESSAGE = 0x4e05
regRLC_SMU_MESSAGE_1 = 0x4e06
regRLC_SMU_MESSAGE_2 = 0x4e07
regRLC_SMU_SAFE_MODE = 0x4e03
regRLC_SPARE = 0x4d0b
regRLC_SPARE_INT_0 = 0x98d
regRLC_SPARE_INT_1 = 0x98e
regRLC_SPARE_INT_2 = 0x98f
regRLC_SPM_ACCUM_CTRL = 0x3c9a
regRLC_SPM_ACCUM_CTRLRAM_ADDR = 0x3c96
regRLC_SPM_ACCUM_CTRLRAM_ADDR_OFFSET = 0x3c98
regRLC_SPM_ACCUM_CTRLRAM_DATA = 0x3c97
regRLC_SPM_ACCUM_DATARAM_32BITCNTRS_REGIONS = 0x3c9f
regRLC_SPM_ACCUM_DATARAM_ADDR = 0x3c92
regRLC_SPM_ACCUM_DATARAM_DATA = 0x3c93
regRLC_SPM_ACCUM_DATARAM_WRCOUNT = 0x3c9e
regRLC_SPM_ACCUM_MODE = 0x3c9b
regRLC_SPM_ACCUM_SAMPLES_REQUESTED = 0x3c9d
regRLC_SPM_ACCUM_STATUS = 0x3c99
regRLC_SPM_ACCUM_SWA_DATARAM_ADDR = 0x3c94
regRLC_SPM_ACCUM_SWA_DATARAM_DATA = 0x3c95
regRLC_SPM_ACCUM_THRESHOLD = 0x3c9c
regRLC_SPM_GFXCLOCK_HIGHCOUNT = 0x3ca5
regRLC_SPM_GFXCLOCK_LOWCOUNT = 0x3ca4
regRLC_SPM_GLOBAL_DELAY_IND_ADDR = 0x4d64
regRLC_SPM_GLOBAL_DELAY_IND_DATA = 0x4d65
regRLC_SPM_GLOBAL_MUXSEL_ADDR = 0x3c88
regRLC_SPM_GLOBAL_MUXSEL_DATA = 0x3c89
regRLC_SPM_INT_CNTL = 0x983
regRLC_SPM_INT_INFO_1 = 0x985
regRLC_SPM_INT_INFO_2 = 0x986
regRLC_SPM_INT_STATUS = 0x984
regRLC_SPM_MC_CNTL = 0x982
regRLC_SPM_MODE = 0x3cad
regRLC_SPM_PAUSE = 0x3ca2
regRLC_SPM_PERFMON_CNTL = 0x3c80
regRLC_SPM_PERFMON_RING_BASE_HI = 0x3c82
regRLC_SPM_PERFMON_RING_BASE_LO = 0x3c81
regRLC_SPM_PERFMON_RING_SIZE = 0x3c83
regRLC_SPM_PERFMON_SEGMENT_SIZE = 0x3c87
regRLC_SPM_RING_RDPTR = 0x3c85
regRLC_SPM_RING_WRPTR = 0x3c84
regRLC_SPM_RSPM_CMD = 0x3cb8
regRLC_SPM_RSPM_CMD_ACK = 0x3cb9
regRLC_SPM_RSPM_REQ_DATA_HI = 0x3caf
regRLC_SPM_RSPM_REQ_DATA_LO = 0x3cae
regRLC_SPM_RSPM_REQ_OP = 0x3cb0
regRLC_SPM_RSPM_RET_DATA = 0x3cb1
regRLC_SPM_RSPM_RET_OP = 0x3cb2
regRLC_SPM_SAMPLE_CNT = 0x981
regRLC_SPM_SEGMENT_THRESHOLD = 0x3c86
regRLC_SPM_SE_DELAY_IND_ADDR = 0x4d66
regRLC_SPM_SE_DELAY_IND_DATA = 0x4d67
regRLC_SPM_SE_MUXSEL_ADDR = 0x3c8a
regRLC_SPM_SE_MUXSEL_DATA = 0x3c8b
regRLC_SPM_SE_RSPM_REQ_DATA_HI = 0x3cb4
regRLC_SPM_SE_RSPM_REQ_DATA_LO = 0x3cb3
regRLC_SPM_SE_RSPM_REQ_OP = 0x3cb5
regRLC_SPM_SE_RSPM_RET_DATA = 0x3cb6
regRLC_SPM_SE_RSPM_RET_OP = 0x3cb7
regRLC_SPM_SPARE = 0x3cbf
regRLC_SPM_STATUS = 0x3ca3
regRLC_SPM_THREAD_TRACE_CTRL = 0x4de6
regRLC_SPM_UTCL1_CNTL = 0x4cb5
regRLC_SPM_UTCL1_ERROR_1 = 0x4cbc
regRLC_SPM_UTCL1_ERROR_2 = 0x4cbd
regRLC_SPP_CAM_ADDR = 0x4de8
regRLC_SPP_CAM_DATA = 0x4de9
regRLC_SPP_CAM_EXT_ADDR = 0x4dea
regRLC_SPP_CAM_EXT_DATA = 0x4deb
regRLC_SPP_CTRL = 0x4d0c
regRLC_SPP_GLOBAL_SH_ID = 0x4d1a
regRLC_SPP_GLOBAL_SH_ID_VALID = 0x4d1b
regRLC_SPP_INFLIGHT_RD_ADDR = 0x4d12
regRLC_SPP_INFLIGHT_RD_DATA = 0x4d13
regRLC_SPP_PBB_INFO = 0x4d23
regRLC_SPP_PROF_INFO_1 = 0x4d18
regRLC_SPP_PROF_INFO_2 = 0x4d19
regRLC_SPP_PVT_LEVEL_MAX = 0x4d21
regRLC_SPP_PVT_STAT_0 = 0x4d1d
regRLC_SPP_PVT_STAT_1 = 0x4d1e
regRLC_SPP_PVT_STAT_2 = 0x4d1f
regRLC_SPP_PVT_STAT_3 = 0x4d20
regRLC_SPP_RESET = 0x4d24
regRLC_SPP_SHADER_PROFILE_EN = 0x4d0d
regRLC_SPP_SSF_CAPTURE_EN = 0x4d0e
regRLC_SPP_SSF_THRESHOLD_0 = 0x4d0f
regRLC_SPP_SSF_THRESHOLD_1 = 0x4d10
regRLC_SPP_SSF_THRESHOLD_2 = 0x4d11
regRLC_SPP_STALL_STATE_UPDATE = 0x4d22
regRLC_SPP_STATUS = 0x4d1c
regRLC_SRM_ARAM_ADDR = 0x5b73
regRLC_SRM_ARAM_DATA = 0x5b74
regRLC_SRM_CNTL = 0x4c80
regRLC_SRM_DRAM_ADDR = 0x5b71
regRLC_SRM_DRAM_DATA = 0x5b72
regRLC_SRM_GPM_ABORT = 0x4e09
regRLC_SRM_GPM_COMMAND = 0x4e08
regRLC_SRM_GPM_COMMAND_STATUS = 0x4c88
regRLC_SRM_INDEX_CNTL_ADDR_0 = 0x4c8b
regRLC_SRM_INDEX_CNTL_ADDR_1 = 0x4c8c
regRLC_SRM_INDEX_CNTL_ADDR_2 = 0x4c8d
regRLC_SRM_INDEX_CNTL_ADDR_3 = 0x4c8e
regRLC_SRM_INDEX_CNTL_ADDR_4 = 0x4c8f
regRLC_SRM_INDEX_CNTL_ADDR_5 = 0x4c90
regRLC_SRM_INDEX_CNTL_ADDR_6 = 0x4c91
regRLC_SRM_INDEX_CNTL_ADDR_7 = 0x4c92
regRLC_SRM_INDEX_CNTL_DATA_0 = 0x4c93
regRLC_SRM_INDEX_CNTL_DATA_1 = 0x4c94
regRLC_SRM_INDEX_CNTL_DATA_2 = 0x4c95
regRLC_SRM_INDEX_CNTL_DATA_3 = 0x4c96
regRLC_SRM_INDEX_CNTL_DATA_4 = 0x4c97
regRLC_SRM_INDEX_CNTL_DATA_5 = 0x4c98
regRLC_SRM_INDEX_CNTL_DATA_6 = 0x4c99
regRLC_SRM_INDEX_CNTL_DATA_7 = 0x4c9a
regRLC_SRM_STAT = 0x4c9b
regRLC_STAT = 0x4c04
regRLC_STATIC_PG_STATUS = 0x4c6e
regRLC_UCODE_CNTL = 0x4c27
regRLC_ULV_RESIDENCY_CNTR_CTRL = 0x4d4b
regRLC_ULV_RESIDENCY_EVENT_CNTR = 0x4d53
regRLC_ULV_RESIDENCY_REF_CNTR = 0x4d5b
regRLC_UTCL1_STATUS = 0x4cd4
regRLC_UTCL1_STATUS_2 = 0x4cb6
regRLC_WGP_STATUS = 0x4c4e
regRLC_XT_CORE_ALT_RESET_VEC = 0x4dd7
regRLC_XT_CORE_FAULT_INFO = 0x4dd6
regRLC_XT_CORE_INTERRUPT = 0x4dd5
regRLC_XT_CORE_RESERVED = 0x4dd8
regRLC_XT_CORE_STATUS = 0x4dd4
regRLC_XT_DOORBELL_0_DATA_HI = 0x4df9
regRLC_XT_DOORBELL_0_DATA_LO = 0x4df8
regRLC_XT_DOORBELL_1_DATA_HI = 0x4dfb
regRLC_XT_DOORBELL_1_DATA_LO = 0x4dfa
regRLC_XT_DOORBELL_2_DATA_HI = 0x4dfd
regRLC_XT_DOORBELL_2_DATA_LO = 0x4dfc
regRLC_XT_DOORBELL_3_DATA_HI = 0x4dff
regRLC_XT_DOORBELL_3_DATA_LO = 0x4dfe
regRLC_XT_DOORBELL_CNTL = 0x4df6
regRLC_XT_DOORBELL_RANGE = 0x4df5
regRLC_XT_DOORBELL_STAT = 0x4df7
regRLC_XT_INT_VEC_CLEAR = 0x4dda
regRLC_XT_INT_VEC_FORCE = 0x4dd9
regRLC_XT_INT_VEC_MUX_INT_SEL = 0x4ddc
regRLC_XT_INT_VEC_MUX_SEL = 0x4ddb
regRMI_CLOCK_CNTRL = 0x1896
regRMI_DEMUX_CNTL = 0x188a
regRMI_GENERAL_CNTL = 0x1880
regRMI_GENERAL_CNTL1 = 0x1881
regRMI_GENERAL_STATUS = 0x1882
regRMI_PERFCOUNTER0_HI = 0x34c1
regRMI_PERFCOUNTER0_LO = 0x34c0
regRMI_PERFCOUNTER0_SELECT = 0x3d00
regRMI_PERFCOUNTER0_SELECT1 = 0x3d01
regRMI_PERFCOUNTER1_HI = 0x34c3
regRMI_PERFCOUNTER1_LO = 0x34c2
regRMI_PERFCOUNTER1_SELECT = 0x3d02
regRMI_PERFCOUNTER2_HI = 0x34c5
regRMI_PERFCOUNTER2_LO = 0x34c4
regRMI_PERFCOUNTER2_SELECT = 0x3d03
regRMI_PERFCOUNTER2_SELECT1 = 0x3d04
regRMI_PERFCOUNTER3_HI = 0x34c7
regRMI_PERFCOUNTER3_LO = 0x34c6
regRMI_PERFCOUNTER3_SELECT = 0x3d05
regRMI_PERF_COUNTER_CNTL = 0x3d06
regRMI_PROBE_POP_LOGIC_CNTL = 0x1888
regRMI_RB_GLX_CID_MAP = 0x1898
regRMI_SCOREBOARD_CNTL = 0x1890
regRMI_SCOREBOARD_STATUS0 = 0x1891
regRMI_SCOREBOARD_STATUS1 = 0x1892
regRMI_SCOREBOARD_STATUS2 = 0x1893
regRMI_SPARE = 0x189f
regRMI_SPARE_1 = 0x18a0
regRMI_SPARE_2 = 0x18a1
regRMI_SUBBLOCK_STATUS0 = 0x1883
regRMI_SUBBLOCK_STATUS1 = 0x1884
regRMI_SUBBLOCK_STATUS2 = 0x1885
regRMI_SUBBLOCK_STATUS3 = 0x1886
regRMI_TCIW_FORMATTER0_CNTL = 0x188e
regRMI_TCIW_FORMATTER1_CNTL = 0x188f
regRMI_UTCL1_CNTL1 = 0x188b
regRMI_UTCL1_CNTL2 = 0x188c
regRMI_UTCL1_STATUS = 0x1897
regRMI_UTC_UNIT_CONFIG = 0x188d
regRMI_UTC_XNACK_N_MISC_CNTL = 0x1889
regRMI_XBAR_ARBITER_CONFIG = 0x1894
regRMI_XBAR_ARBITER_CONFIG_1 = 0x1895
regRMI_XBAR_CONFIG = 0x1887
regRTAVFS_RTAVFS_REG_ADDR = 0x4b00
regRTAVFS_RTAVFS_WR_DATA = 0x4b01
regSCRATCH_REG0 = 0x2040
regSCRATCH_REG1 = 0x2041
regSCRATCH_REG2 = 0x2042
regSCRATCH_REG3 = 0x2043
regSCRATCH_REG4 = 0x2044
regSCRATCH_REG5 = 0x2045
regSCRATCH_REG6 = 0x2046
regSCRATCH_REG7 = 0x2047
regSCRATCH_REG_ATOMIC = 0x2048
regSCRATCH_REG_CMPSWAP_ATOMIC = 0x2048
regSDMA0_AQL_STATUS = 0x5f
regSDMA0_ATOMIC_CNTL = 0x39
regSDMA0_ATOMIC_PREOP_HI = 0x3b
regSDMA0_ATOMIC_PREOP_LO = 0x3a
regSDMA0_BA_THRESHOLD = 0x33
regSDMA0_BROADCAST_UCODE_ADDR = 0x5886
regSDMA0_BROADCAST_UCODE_DATA = 0x5887
regSDMA0_CE_CTRL = 0x7e
regSDMA0_CHICKEN_BITS = 0x1d
regSDMA0_CHICKEN_BITS_2 = 0x4b
regSDMA0_CLOCK_GATING_STATUS = 0x75
regSDMA0_CNTL = 0x1c
regSDMA0_CNTL1 = 0x27
regSDMA0_CRD_CNTL = 0x5b
regSDMA0_DEC_START = 0x0
regSDMA0_EA_DBIT_ADDR_DATA = 0x60
regSDMA0_EA_DBIT_ADDR_INDEX = 0x61
regSDMA0_EDC_CONFIG = 0x32
regSDMA0_EDC_COUNTER = 0x36
regSDMA0_EDC_COUNTER_CLEAR = 0x37
regSDMA0_ERROR_LOG = 0x50
regSDMA0_F32_CNTL = 0x589a
regSDMA0_F32_COUNTER = 0x55
regSDMA0_F32_MISC_CNTL = 0xb
regSDMA0_FED_STATUS = 0x7f
regSDMA0_FREEZE = 0x2b
regSDMA0_GB_ADDR_CONFIG = 0x1e
regSDMA0_GB_ADDR_CONFIG_READ = 0x1f
regSDMA0_GLOBAL_QUANTUM = 0x4f
regSDMA0_GLOBAL_TIMESTAMP_HI = 0x10
regSDMA0_GLOBAL_TIMESTAMP_LO = 0xf
regSDMA0_HBM_PAGE_CONFIG = 0x28
regSDMA0_HOLE_ADDR_HI = 0x73
regSDMA0_HOLE_ADDR_LO = 0x72
regSDMA0_IB_OFFSET_FETCH = 0x23
regSDMA0_ID = 0x34
regSDMA0_INT_STATUS = 0x70
regSDMA0_PERFCNT_MISC_CNTL = 0x3e23
regSDMA0_PERFCNT_PERFCOUNTER0_CFG = 0x3e20
regSDMA0_PERFCNT_PERFCOUNTER1_CFG = 0x3e21
regSDMA0_PERFCNT_PERFCOUNTER_HI = 0x3661
regSDMA0_PERFCNT_PERFCOUNTER_LO = 0x3660
regSDMA0_PERFCNT_PERFCOUNTER_RSLT_CNTL = 0x3e22
regSDMA0_PERFCOUNTER0_HI = 0x3663
regSDMA0_PERFCOUNTER0_LO = 0x3662
regSDMA0_PERFCOUNTER0_SELECT = 0x3e24
regSDMA0_PERFCOUNTER0_SELECT1 = 0x3e25
regSDMA0_PERFCOUNTER1_HI = 0x3665
regSDMA0_PERFCOUNTER1_LO = 0x3664
regSDMA0_PERFCOUNTER1_SELECT = 0x3e26
regSDMA0_PERFCOUNTER1_SELECT1 = 0x3e27
regSDMA0_PHYSICAL_ADDR_HI = 0x4e
regSDMA0_PHYSICAL_ADDR_LO = 0x4d
regSDMA0_POWER_CNTL = 0x1a
regSDMA0_PROCESS_QUANTUM0 = 0x2c
regSDMA0_PROCESS_QUANTUM1 = 0x2d
regSDMA0_PROGRAM = 0x24
regSDMA0_PUB_DUMMY_REG0 = 0x51
regSDMA0_PUB_DUMMY_REG1 = 0x52
regSDMA0_PUB_DUMMY_REG2 = 0x53
regSDMA0_PUB_DUMMY_REG3 = 0x54
regSDMA0_QUEUE0_CONTEXT_STATUS = 0x91
regSDMA0_QUEUE0_CSA_ADDR_HI = 0xad
regSDMA0_QUEUE0_CSA_ADDR_LO = 0xac
regSDMA0_QUEUE0_DOORBELL = 0x92
regSDMA0_QUEUE0_DOORBELL_LOG = 0xa9
regSDMA0_QUEUE0_DOORBELL_OFFSET = 0xab
regSDMA0_QUEUE0_DUMMY_REG = 0xb1
regSDMA0_QUEUE0_IB_BASE_HI = 0x8e
regSDMA0_QUEUE0_IB_BASE_LO = 0x8d
regSDMA0_QUEUE0_IB_CNTL = 0x8a
regSDMA0_QUEUE0_IB_OFFSET = 0x8c
regSDMA0_QUEUE0_IB_RPTR = 0x8b
regSDMA0_QUEUE0_IB_SIZE = 0x8f
regSDMA0_QUEUE0_IB_SUB_REMAIN = 0xaf
regSDMA0_QUEUE0_MIDCMD_CNTL = 0xcb
regSDMA0_QUEUE0_MIDCMD_DATA0 = 0xc0
regSDMA0_QUEUE0_MIDCMD_DATA1 = 0xc1
regSDMA0_QUEUE0_MIDCMD_DATA10 = 0xca
regSDMA0_QUEUE0_MIDCMD_DATA2 = 0xc2
regSDMA0_QUEUE0_MIDCMD_DATA3 = 0xc3
regSDMA0_QUEUE0_MIDCMD_DATA4 = 0xc4
regSDMA0_QUEUE0_MIDCMD_DATA5 = 0xc5
regSDMA0_QUEUE0_MIDCMD_DATA6 = 0xc6
regSDMA0_QUEUE0_MIDCMD_DATA7 = 0xc7
regSDMA0_QUEUE0_MIDCMD_DATA8 = 0xc8
regSDMA0_QUEUE0_MIDCMD_DATA9 = 0xc9
regSDMA0_QUEUE0_MINOR_PTR_UPDATE = 0xb5
regSDMA0_QUEUE0_PREEMPT = 0xb0
regSDMA0_QUEUE0_RB_AQL_CNTL = 0xb4
regSDMA0_QUEUE0_RB_BASE = 0x81
regSDMA0_QUEUE0_RB_BASE_HI = 0x82
regSDMA0_QUEUE0_RB_CNTL = 0x80
regSDMA0_QUEUE0_RB_PREEMPT = 0xb6
regSDMA0_QUEUE0_RB_RPTR = 0x83
regSDMA0_QUEUE0_RB_RPTR_ADDR_HI = 0x88
regSDMA0_QUEUE0_RB_RPTR_ADDR_LO = 0x89
regSDMA0_QUEUE0_RB_RPTR_HI = 0x84
regSDMA0_QUEUE0_RB_WPTR = 0x85
regSDMA0_QUEUE0_RB_WPTR_HI = 0x86
regSDMA0_QUEUE0_RB_WPTR_POLL_ADDR_HI = 0xb2
regSDMA0_QUEUE0_RB_WPTR_POLL_ADDR_LO = 0xb3
regSDMA0_QUEUE0_SCHEDULE_CNTL = 0xae
regSDMA0_QUEUE0_SKIP_CNTL = 0x90
regSDMA0_QUEUE1_CONTEXT_STATUS = 0xe9
regSDMA0_QUEUE1_CSA_ADDR_HI = 0x105
regSDMA0_QUEUE1_CSA_ADDR_LO = 0x104
regSDMA0_QUEUE1_DOORBELL = 0xea
regSDMA0_QUEUE1_DOORBELL_LOG = 0x101
regSDMA0_QUEUE1_DOORBELL_OFFSET = 0x103
regSDMA0_QUEUE1_DUMMY_REG = 0x109
regSDMA0_QUEUE1_IB_BASE_HI = 0xe6
regSDMA0_QUEUE1_IB_BASE_LO = 0xe5
regSDMA0_QUEUE1_IB_CNTL = 0xe2
regSDMA0_QUEUE1_IB_OFFSET = 0xe4
regSDMA0_QUEUE1_IB_RPTR = 0xe3
regSDMA0_QUEUE1_IB_SIZE = 0xe7
regSDMA0_QUEUE1_IB_SUB_REMAIN = 0x107
regSDMA0_QUEUE1_MIDCMD_CNTL = 0x123
regSDMA0_QUEUE1_MIDCMD_DATA0 = 0x118
regSDMA0_QUEUE1_MIDCMD_DATA1 = 0x119
regSDMA0_QUEUE1_MIDCMD_DATA10 = 0x122
regSDMA0_QUEUE1_MIDCMD_DATA2 = 0x11a
regSDMA0_QUEUE1_MIDCMD_DATA3 = 0x11b
regSDMA0_QUEUE1_MIDCMD_DATA4 = 0x11c
regSDMA0_QUEUE1_MIDCMD_DATA5 = 0x11d
regSDMA0_QUEUE1_MIDCMD_DATA6 = 0x11e
regSDMA0_QUEUE1_MIDCMD_DATA7 = 0x11f
regSDMA0_QUEUE1_MIDCMD_DATA8 = 0x120
regSDMA0_QUEUE1_MIDCMD_DATA9 = 0x121
regSDMA0_QUEUE1_MINOR_PTR_UPDATE = 0x10d
regSDMA0_QUEUE1_PREEMPT = 0x108
regSDMA0_QUEUE1_RB_AQL_CNTL = 0x10c
regSDMA0_QUEUE1_RB_BASE = 0xd9
regSDMA0_QUEUE1_RB_BASE_HI = 0xda
regSDMA0_QUEUE1_RB_CNTL = 0xd8
regSDMA0_QUEUE1_RB_PREEMPT = 0x10e
regSDMA0_QUEUE1_RB_RPTR = 0xdb
regSDMA0_QUEUE1_RB_RPTR_ADDR_HI = 0xe0
regSDMA0_QUEUE1_RB_RPTR_ADDR_LO = 0xe1
regSDMA0_QUEUE1_RB_RPTR_HI = 0xdc
regSDMA0_QUEUE1_RB_WPTR = 0xdd
regSDMA0_QUEUE1_RB_WPTR_HI = 0xde
regSDMA0_QUEUE1_RB_WPTR_POLL_ADDR_HI = 0x10a
regSDMA0_QUEUE1_RB_WPTR_POLL_ADDR_LO = 0x10b
regSDMA0_QUEUE1_SCHEDULE_CNTL = 0x106
regSDMA0_QUEUE1_SKIP_CNTL = 0xe8
regSDMA0_QUEUE2_CONTEXT_STATUS = 0x141
regSDMA0_QUEUE2_CSA_ADDR_HI = 0x15d
regSDMA0_QUEUE2_CSA_ADDR_LO = 0x15c
regSDMA0_QUEUE2_DOORBELL = 0x142
regSDMA0_QUEUE2_DOORBELL_LOG = 0x159
regSDMA0_QUEUE2_DOORBELL_OFFSET = 0x15b
regSDMA0_QUEUE2_DUMMY_REG = 0x161
regSDMA0_QUEUE2_IB_BASE_HI = 0x13e
regSDMA0_QUEUE2_IB_BASE_LO = 0x13d
regSDMA0_QUEUE2_IB_CNTL = 0x13a
regSDMA0_QUEUE2_IB_OFFSET = 0x13c
regSDMA0_QUEUE2_IB_RPTR = 0x13b
regSDMA0_QUEUE2_IB_SIZE = 0x13f
regSDMA0_QUEUE2_IB_SUB_REMAIN = 0x15f
regSDMA0_QUEUE2_MIDCMD_CNTL = 0x17b
regSDMA0_QUEUE2_MIDCMD_DATA0 = 0x170
regSDMA0_QUEUE2_MIDCMD_DATA1 = 0x171
regSDMA0_QUEUE2_MIDCMD_DATA10 = 0x17a
regSDMA0_QUEUE2_MIDCMD_DATA2 = 0x172
regSDMA0_QUEUE2_MIDCMD_DATA3 = 0x173
regSDMA0_QUEUE2_MIDCMD_DATA4 = 0x174
regSDMA0_QUEUE2_MIDCMD_DATA5 = 0x175
regSDMA0_QUEUE2_MIDCMD_DATA6 = 0x176
regSDMA0_QUEUE2_MIDCMD_DATA7 = 0x177
regSDMA0_QUEUE2_MIDCMD_DATA8 = 0x178
regSDMA0_QUEUE2_MIDCMD_DATA9 = 0x179
regSDMA0_QUEUE2_MINOR_PTR_UPDATE = 0x165
regSDMA0_QUEUE2_PREEMPT = 0x160
regSDMA0_QUEUE2_RB_AQL_CNTL = 0x164
regSDMA0_QUEUE2_RB_BASE = 0x131
regSDMA0_QUEUE2_RB_BASE_HI = 0x132
regSDMA0_QUEUE2_RB_CNTL = 0x130
regSDMA0_QUEUE2_RB_PREEMPT = 0x166
regSDMA0_QUEUE2_RB_RPTR = 0x133
regSDMA0_QUEUE2_RB_RPTR_ADDR_HI = 0x138
regSDMA0_QUEUE2_RB_RPTR_ADDR_LO = 0x139
regSDMA0_QUEUE2_RB_RPTR_HI = 0x134
regSDMA0_QUEUE2_RB_WPTR = 0x135
regSDMA0_QUEUE2_RB_WPTR_HI = 0x136
regSDMA0_QUEUE2_RB_WPTR_POLL_ADDR_HI = 0x162
regSDMA0_QUEUE2_RB_WPTR_POLL_ADDR_LO = 0x163
regSDMA0_QUEUE2_SCHEDULE_CNTL = 0x15e
regSDMA0_QUEUE2_SKIP_CNTL = 0x140
regSDMA0_QUEUE3_CONTEXT_STATUS = 0x199
regSDMA0_QUEUE3_CSA_ADDR_HI = 0x1b5
regSDMA0_QUEUE3_CSA_ADDR_LO = 0x1b4
regSDMA0_QUEUE3_DOORBELL = 0x19a
regSDMA0_QUEUE3_DOORBELL_LOG = 0x1b1
regSDMA0_QUEUE3_DOORBELL_OFFSET = 0x1b3
regSDMA0_QUEUE3_DUMMY_REG = 0x1b9
regSDMA0_QUEUE3_IB_BASE_HI = 0x196
regSDMA0_QUEUE3_IB_BASE_LO = 0x195
regSDMA0_QUEUE3_IB_CNTL = 0x192
regSDMA0_QUEUE3_IB_OFFSET = 0x194
regSDMA0_QUEUE3_IB_RPTR = 0x193
regSDMA0_QUEUE3_IB_SIZE = 0x197
regSDMA0_QUEUE3_IB_SUB_REMAIN = 0x1b7
regSDMA0_QUEUE3_MIDCMD_CNTL = 0x1d3
regSDMA0_QUEUE3_MIDCMD_DATA0 = 0x1c8
regSDMA0_QUEUE3_MIDCMD_DATA1 = 0x1c9
regSDMA0_QUEUE3_MIDCMD_DATA10 = 0x1d2
regSDMA0_QUEUE3_MIDCMD_DATA2 = 0x1ca
regSDMA0_QUEUE3_MIDCMD_DATA3 = 0x1cb
regSDMA0_QUEUE3_MIDCMD_DATA4 = 0x1cc
regSDMA0_QUEUE3_MIDCMD_DATA5 = 0x1cd
regSDMA0_QUEUE3_MIDCMD_DATA6 = 0x1ce
regSDMA0_QUEUE3_MIDCMD_DATA7 = 0x1cf
regSDMA0_QUEUE3_MIDCMD_DATA8 = 0x1d0
regSDMA0_QUEUE3_MIDCMD_DATA9 = 0x1d1
regSDMA0_QUEUE3_MINOR_PTR_UPDATE = 0x1bd
regSDMA0_QUEUE3_PREEMPT = 0x1b8
regSDMA0_QUEUE3_RB_AQL_CNTL = 0x1bc
regSDMA0_QUEUE3_RB_BASE = 0x189
regSDMA0_QUEUE3_RB_BASE_HI = 0x18a
regSDMA0_QUEUE3_RB_CNTL = 0x188
regSDMA0_QUEUE3_RB_PREEMPT = 0x1be
regSDMA0_QUEUE3_RB_RPTR = 0x18b
regSDMA0_QUEUE3_RB_RPTR_ADDR_HI = 0x190
regSDMA0_QUEUE3_RB_RPTR_ADDR_LO = 0x191
regSDMA0_QUEUE3_RB_RPTR_HI = 0x18c
regSDMA0_QUEUE3_RB_WPTR = 0x18d
regSDMA0_QUEUE3_RB_WPTR_HI = 0x18e
regSDMA0_QUEUE3_RB_WPTR_POLL_ADDR_HI = 0x1ba
regSDMA0_QUEUE3_RB_WPTR_POLL_ADDR_LO = 0x1bb
regSDMA0_QUEUE3_SCHEDULE_CNTL = 0x1b6
regSDMA0_QUEUE3_SKIP_CNTL = 0x198
regSDMA0_QUEUE4_CONTEXT_STATUS = 0x1f1
regSDMA0_QUEUE4_CSA_ADDR_HI = 0x20d
regSDMA0_QUEUE4_CSA_ADDR_LO = 0x20c
regSDMA0_QUEUE4_DOORBELL = 0x1f2
regSDMA0_QUEUE4_DOORBELL_LOG = 0x209
regSDMA0_QUEUE4_DOORBELL_OFFSET = 0x20b
regSDMA0_QUEUE4_DUMMY_REG = 0x211
regSDMA0_QUEUE4_IB_BASE_HI = 0x1ee
regSDMA0_QUEUE4_IB_BASE_LO = 0x1ed
regSDMA0_QUEUE4_IB_CNTL = 0x1ea
regSDMA0_QUEUE4_IB_OFFSET = 0x1ec
regSDMA0_QUEUE4_IB_RPTR = 0x1eb
regSDMA0_QUEUE4_IB_SIZE = 0x1ef
regSDMA0_QUEUE4_IB_SUB_REMAIN = 0x20f
regSDMA0_QUEUE4_MIDCMD_CNTL = 0x22b
regSDMA0_QUEUE4_MIDCMD_DATA0 = 0x220
regSDMA0_QUEUE4_MIDCMD_DATA1 = 0x221
regSDMA0_QUEUE4_MIDCMD_DATA10 = 0x22a
regSDMA0_QUEUE4_MIDCMD_DATA2 = 0x222
regSDMA0_QUEUE4_MIDCMD_DATA3 = 0x223
regSDMA0_QUEUE4_MIDCMD_DATA4 = 0x224
regSDMA0_QUEUE4_MIDCMD_DATA5 = 0x225
regSDMA0_QUEUE4_MIDCMD_DATA6 = 0x226
regSDMA0_QUEUE4_MIDCMD_DATA7 = 0x227
regSDMA0_QUEUE4_MIDCMD_DATA8 = 0x228
regSDMA0_QUEUE4_MIDCMD_DATA9 = 0x229
regSDMA0_QUEUE4_MINOR_PTR_UPDATE = 0x215
regSDMA0_QUEUE4_PREEMPT = 0x210
regSDMA0_QUEUE4_RB_AQL_CNTL = 0x214
regSDMA0_QUEUE4_RB_BASE = 0x1e1
regSDMA0_QUEUE4_RB_BASE_HI = 0x1e2
regSDMA0_QUEUE4_RB_CNTL = 0x1e0
regSDMA0_QUEUE4_RB_PREEMPT = 0x216
regSDMA0_QUEUE4_RB_RPTR = 0x1e3
regSDMA0_QUEUE4_RB_RPTR_ADDR_HI = 0x1e8
regSDMA0_QUEUE4_RB_RPTR_ADDR_LO = 0x1e9
regSDMA0_QUEUE4_RB_RPTR_HI = 0x1e4
regSDMA0_QUEUE4_RB_WPTR = 0x1e5
regSDMA0_QUEUE4_RB_WPTR_HI = 0x1e6
regSDMA0_QUEUE4_RB_WPTR_POLL_ADDR_HI = 0x212
regSDMA0_QUEUE4_RB_WPTR_POLL_ADDR_LO = 0x213
regSDMA0_QUEUE4_SCHEDULE_CNTL = 0x20e
regSDMA0_QUEUE4_SKIP_CNTL = 0x1f0
regSDMA0_QUEUE5_CONTEXT_STATUS = 0x249
regSDMA0_QUEUE5_CSA_ADDR_HI = 0x265
regSDMA0_QUEUE5_CSA_ADDR_LO = 0x264
regSDMA0_QUEUE5_DOORBELL = 0x24a
regSDMA0_QUEUE5_DOORBELL_LOG = 0x261
regSDMA0_QUEUE5_DOORBELL_OFFSET = 0x263
regSDMA0_QUEUE5_DUMMY_REG = 0x269
regSDMA0_QUEUE5_IB_BASE_HI = 0x246
regSDMA0_QUEUE5_IB_BASE_LO = 0x245
regSDMA0_QUEUE5_IB_CNTL = 0x242
regSDMA0_QUEUE5_IB_OFFSET = 0x244
regSDMA0_QUEUE5_IB_RPTR = 0x243
regSDMA0_QUEUE5_IB_SIZE = 0x247
regSDMA0_QUEUE5_IB_SUB_REMAIN = 0x267
regSDMA0_QUEUE5_MIDCMD_CNTL = 0x283
regSDMA0_QUEUE5_MIDCMD_DATA0 = 0x278
regSDMA0_QUEUE5_MIDCMD_DATA1 = 0x279
regSDMA0_QUEUE5_MIDCMD_DATA10 = 0x282
regSDMA0_QUEUE5_MIDCMD_DATA2 = 0x27a
regSDMA0_QUEUE5_MIDCMD_DATA3 = 0x27b
regSDMA0_QUEUE5_MIDCMD_DATA4 = 0x27c
regSDMA0_QUEUE5_MIDCMD_DATA5 = 0x27d
regSDMA0_QUEUE5_MIDCMD_DATA6 = 0x27e
regSDMA0_QUEUE5_MIDCMD_DATA7 = 0x27f
regSDMA0_QUEUE5_MIDCMD_DATA8 = 0x280
regSDMA0_QUEUE5_MIDCMD_DATA9 = 0x281
regSDMA0_QUEUE5_MINOR_PTR_UPDATE = 0x26d
regSDMA0_QUEUE5_PREEMPT = 0x268
regSDMA0_QUEUE5_RB_AQL_CNTL = 0x26c
regSDMA0_QUEUE5_RB_BASE = 0x239
regSDMA0_QUEUE5_RB_BASE_HI = 0x23a
regSDMA0_QUEUE5_RB_CNTL = 0x238
regSDMA0_QUEUE5_RB_PREEMPT = 0x26e
regSDMA0_QUEUE5_RB_RPTR = 0x23b
regSDMA0_QUEUE5_RB_RPTR_ADDR_HI = 0x240
regSDMA0_QUEUE5_RB_RPTR_ADDR_LO = 0x241
regSDMA0_QUEUE5_RB_RPTR_HI = 0x23c
regSDMA0_QUEUE5_RB_WPTR = 0x23d
regSDMA0_QUEUE5_RB_WPTR_HI = 0x23e
regSDMA0_QUEUE5_RB_WPTR_POLL_ADDR_HI = 0x26a
regSDMA0_QUEUE5_RB_WPTR_POLL_ADDR_LO = 0x26b
regSDMA0_QUEUE5_SCHEDULE_CNTL = 0x266
regSDMA0_QUEUE5_SKIP_CNTL = 0x248
regSDMA0_QUEUE6_CONTEXT_STATUS = 0x2a1
regSDMA0_QUEUE6_CSA_ADDR_HI = 0x2bd
regSDMA0_QUEUE6_CSA_ADDR_LO = 0x2bc
regSDMA0_QUEUE6_DOORBELL = 0x2a2
regSDMA0_QUEUE6_DOORBELL_LOG = 0x2b9
regSDMA0_QUEUE6_DOORBELL_OFFSET = 0x2bb
regSDMA0_QUEUE6_DUMMY_REG = 0x2c1
regSDMA0_QUEUE6_IB_BASE_HI = 0x29e
regSDMA0_QUEUE6_IB_BASE_LO = 0x29d
regSDMA0_QUEUE6_IB_CNTL = 0x29a
regSDMA0_QUEUE6_IB_OFFSET = 0x29c
regSDMA0_QUEUE6_IB_RPTR = 0x29b
regSDMA0_QUEUE6_IB_SIZE = 0x29f
regSDMA0_QUEUE6_IB_SUB_REMAIN = 0x2bf
regSDMA0_QUEUE6_MIDCMD_CNTL = 0x2db
regSDMA0_QUEUE6_MIDCMD_DATA0 = 0x2d0
regSDMA0_QUEUE6_MIDCMD_DATA1 = 0x2d1
regSDMA0_QUEUE6_MIDCMD_DATA10 = 0x2da
regSDMA0_QUEUE6_MIDCMD_DATA2 = 0x2d2
regSDMA0_QUEUE6_MIDCMD_DATA3 = 0x2d3
regSDMA0_QUEUE6_MIDCMD_DATA4 = 0x2d4
regSDMA0_QUEUE6_MIDCMD_DATA5 = 0x2d5
regSDMA0_QUEUE6_MIDCMD_DATA6 = 0x2d6
regSDMA0_QUEUE6_MIDCMD_DATA7 = 0x2d7
regSDMA0_QUEUE6_MIDCMD_DATA8 = 0x2d8
regSDMA0_QUEUE6_MIDCMD_DATA9 = 0x2d9
regSDMA0_QUEUE6_MINOR_PTR_UPDATE = 0x2c5
regSDMA0_QUEUE6_PREEMPT = 0x2c0
regSDMA0_QUEUE6_RB_AQL_CNTL = 0x2c4
regSDMA0_QUEUE6_RB_BASE = 0x291
regSDMA0_QUEUE6_RB_BASE_HI = 0x292
regSDMA0_QUEUE6_RB_CNTL = 0x290
regSDMA0_QUEUE6_RB_PREEMPT = 0x2c6
regSDMA0_QUEUE6_RB_RPTR = 0x293
regSDMA0_QUEUE6_RB_RPTR_ADDR_HI = 0x298
regSDMA0_QUEUE6_RB_RPTR_ADDR_LO = 0x299
regSDMA0_QUEUE6_RB_RPTR_HI = 0x294
regSDMA0_QUEUE6_RB_WPTR = 0x295
regSDMA0_QUEUE6_RB_WPTR_HI = 0x296
regSDMA0_QUEUE6_RB_WPTR_POLL_ADDR_HI = 0x2c2
regSDMA0_QUEUE6_RB_WPTR_POLL_ADDR_LO = 0x2c3
regSDMA0_QUEUE6_SCHEDULE_CNTL = 0x2be
regSDMA0_QUEUE6_SKIP_CNTL = 0x2a0
regSDMA0_QUEUE7_CONTEXT_STATUS = 0x2f9
regSDMA0_QUEUE7_CSA_ADDR_HI = 0x315
regSDMA0_QUEUE7_CSA_ADDR_LO = 0x314
regSDMA0_QUEUE7_DOORBELL = 0x2fa
regSDMA0_QUEUE7_DOORBELL_LOG = 0x311
regSDMA0_QUEUE7_DOORBELL_OFFSET = 0x313
regSDMA0_QUEUE7_DUMMY_REG = 0x319
regSDMA0_QUEUE7_IB_BASE_HI = 0x2f6
regSDMA0_QUEUE7_IB_BASE_LO = 0x2f5
regSDMA0_QUEUE7_IB_CNTL = 0x2f2
regSDMA0_QUEUE7_IB_OFFSET = 0x2f4
regSDMA0_QUEUE7_IB_RPTR = 0x2f3
regSDMA0_QUEUE7_IB_SIZE = 0x2f7
regSDMA0_QUEUE7_IB_SUB_REMAIN = 0x317
regSDMA0_QUEUE7_MIDCMD_CNTL = 0x333
regSDMA0_QUEUE7_MIDCMD_DATA0 = 0x328
regSDMA0_QUEUE7_MIDCMD_DATA1 = 0x329
regSDMA0_QUEUE7_MIDCMD_DATA10 = 0x332
regSDMA0_QUEUE7_MIDCMD_DATA2 = 0x32a
regSDMA0_QUEUE7_MIDCMD_DATA3 = 0x32b
regSDMA0_QUEUE7_MIDCMD_DATA4 = 0x32c
regSDMA0_QUEUE7_MIDCMD_DATA5 = 0x32d
regSDMA0_QUEUE7_MIDCMD_DATA6 = 0x32e
regSDMA0_QUEUE7_MIDCMD_DATA7 = 0x32f
regSDMA0_QUEUE7_MIDCMD_DATA8 = 0x330
regSDMA0_QUEUE7_MIDCMD_DATA9 = 0x331
regSDMA0_QUEUE7_MINOR_PTR_UPDATE = 0x31d
regSDMA0_QUEUE7_PREEMPT = 0x318
regSDMA0_QUEUE7_RB_AQL_CNTL = 0x31c
regSDMA0_QUEUE7_RB_BASE = 0x2e9
regSDMA0_QUEUE7_RB_BASE_HI = 0x2ea
regSDMA0_QUEUE7_RB_CNTL = 0x2e8
regSDMA0_QUEUE7_RB_PREEMPT = 0x31e
regSDMA0_QUEUE7_RB_RPTR = 0x2eb
regSDMA0_QUEUE7_RB_RPTR_ADDR_HI = 0x2f0
regSDMA0_QUEUE7_RB_RPTR_ADDR_LO = 0x2f1
regSDMA0_QUEUE7_RB_RPTR_HI = 0x2ec
regSDMA0_QUEUE7_RB_WPTR = 0x2ed
regSDMA0_QUEUE7_RB_WPTR_HI = 0x2ee
regSDMA0_QUEUE7_RB_WPTR_POLL_ADDR_HI = 0x31a
regSDMA0_QUEUE7_RB_WPTR_POLL_ADDR_LO = 0x31b
regSDMA0_QUEUE7_SCHEDULE_CNTL = 0x316
regSDMA0_QUEUE7_SKIP_CNTL = 0x2f8
regSDMA0_QUEUE_RESET_REQ = 0x7b
regSDMA0_QUEUE_STATUS0 = 0x2f
regSDMA0_RB_RPTR_FETCH = 0x20
regSDMA0_RB_RPTR_FETCH_HI = 0x21
regSDMA0_RELAX_ORDERING_LUT = 0x4a
regSDMA0_RLC_CGCG_CTRL = 0x5c
regSDMA0_SCRATCH_RAM_ADDR = 0x78
regSDMA0_SCRATCH_RAM_DATA = 0x77
regSDMA0_SEM_WAIT_FAIL_TIMER_CNTL = 0x22
regSDMA0_STATUS1_REG = 0x26
regSDMA0_STATUS2_REG = 0x38
regSDMA0_STATUS3_REG = 0x4c
regSDMA0_STATUS4_REG = 0x76
regSDMA0_STATUS5_REG = 0x7a
regSDMA0_STATUS6_REG = 0x7c
regSDMA0_STATUS_REG = 0x25
regSDMA0_TILING_CONFIG = 0x63
regSDMA0_TIMESTAMP_CNTL = 0x79
regSDMA0_TLBI_GCR_CNTL = 0x62
regSDMA0_UCODE1_CHECKSUM = 0x7d
regSDMA0_UCODE_ADDR = 0x5880
regSDMA0_UCODE_CHECKSUM = 0x29
regSDMA0_UCODE_DATA = 0x5881
regSDMA0_UCODE_SELFLOAD_CONTROL = 0x5882
regSDMA0_UTCL1_CNTL = 0x3c
regSDMA0_UTCL1_INV0 = 0x42
regSDMA0_UTCL1_INV1 = 0x43
regSDMA0_UTCL1_INV2 = 0x44
regSDMA0_UTCL1_PAGE = 0x3f
regSDMA0_UTCL1_RD_STATUS = 0x40
regSDMA0_UTCL1_RD_XNACK0 = 0x45
regSDMA0_UTCL1_RD_XNACK1 = 0x46
regSDMA0_UTCL1_TIMEOUT = 0x3e
regSDMA0_UTCL1_WATERMK = 0x3d
regSDMA0_UTCL1_WR_STATUS = 0x41
regSDMA0_UTCL1_WR_XNACK0 = 0x47
regSDMA0_UTCL1_WR_XNACK1 = 0x48
regSDMA0_VERSION = 0x35
regSDMA0_WATCHDOG_CNTL = 0x2e
regSDMA1_AQL_STATUS = 0x65f
regSDMA1_ATOMIC_CNTL = 0x639
regSDMA1_ATOMIC_PREOP_HI = 0x63b
regSDMA1_ATOMIC_PREOP_LO = 0x63a
regSDMA1_BA_THRESHOLD = 0x633
regSDMA1_BROADCAST_UCODE_ADDR = 0x58a6
regSDMA1_BROADCAST_UCODE_DATA = 0x58a7
regSDMA1_CE_CTRL = 0x67e
regSDMA1_CHICKEN_BITS = 0x61d
regSDMA1_CHICKEN_BITS_2 = 0x64b
regSDMA1_CLOCK_GATING_STATUS = 0x675
regSDMA1_CNTL = 0x61c
regSDMA1_CNTL1 = 0x627
regSDMA1_CRD_CNTL = 0x65b
regSDMA1_DEC_START = 0x600
regSDMA1_EA_DBIT_ADDR_DATA = 0x660
regSDMA1_EA_DBIT_ADDR_INDEX = 0x661
regSDMA1_EDC_CONFIG = 0x632
regSDMA1_EDC_COUNTER = 0x636
regSDMA1_EDC_COUNTER_CLEAR = 0x637
regSDMA1_ERROR_LOG = 0x650
regSDMA1_F32_CNTL = 0x58ba
regSDMA1_F32_COUNTER = 0x655
regSDMA1_F32_MISC_CNTL = 0x60b
regSDMA1_FED_STATUS = 0x67f
regSDMA1_FREEZE = 0x62b
regSDMA1_GB_ADDR_CONFIG = 0x61e
regSDMA1_GB_ADDR_CONFIG_READ = 0x61f
regSDMA1_GLOBAL_QUANTUM = 0x64f
regSDMA1_GLOBAL_TIMESTAMP_HI = 0x610
regSDMA1_GLOBAL_TIMESTAMP_LO = 0x60f
regSDMA1_HBM_PAGE_CONFIG = 0x628
regSDMA1_HOLE_ADDR_HI = 0x673
regSDMA1_HOLE_ADDR_LO = 0x672
regSDMA1_IB_OFFSET_FETCH = 0x623
regSDMA1_ID = 0x634
regSDMA1_INT_STATUS = 0x670
regSDMA1_PERFCNT_MISC_CNTL = 0x3e2f
regSDMA1_PERFCNT_PERFCOUNTER0_CFG = 0x3e2c
regSDMA1_PERFCNT_PERFCOUNTER1_CFG = 0x3e2d
regSDMA1_PERFCNT_PERFCOUNTER_HI = 0x366d
regSDMA1_PERFCNT_PERFCOUNTER_LO = 0x366c
regSDMA1_PERFCNT_PERFCOUNTER_RSLT_CNTL = 0x3e2e
regSDMA1_PERFCOUNTER0_HI = 0x366f
regSDMA1_PERFCOUNTER0_LO = 0x366e
regSDMA1_PERFCOUNTER0_SELECT = 0x3e30
regSDMA1_PERFCOUNTER0_SELECT1 = 0x3e31
regSDMA1_PERFCOUNTER1_HI = 0x3671
regSDMA1_PERFCOUNTER1_LO = 0x3670
regSDMA1_PERFCOUNTER1_SELECT = 0x3e32
regSDMA1_PERFCOUNTER1_SELECT1 = 0x3e33
regSDMA1_PHYSICAL_ADDR_HI = 0x64e
regSDMA1_PHYSICAL_ADDR_LO = 0x64d
regSDMA1_POWER_CNTL = 0x61a
regSDMA1_PROCESS_QUANTUM0 = 0x62c
regSDMA1_PROCESS_QUANTUM1 = 0x62d
regSDMA1_PROGRAM = 0x624
regSDMA1_PUB_DUMMY_REG0 = 0x651
regSDMA1_PUB_DUMMY_REG1 = 0x652
regSDMA1_PUB_DUMMY_REG2 = 0x653
regSDMA1_PUB_DUMMY_REG3 = 0x654
regSDMA1_QUEUE0_CONTEXT_STATUS = 0x691
regSDMA1_QUEUE0_CSA_ADDR_HI = 0x6ad
regSDMA1_QUEUE0_CSA_ADDR_LO = 0x6ac
regSDMA1_QUEUE0_DOORBELL = 0x692
regSDMA1_QUEUE0_DOORBELL_LOG = 0x6a9
regSDMA1_QUEUE0_DOORBELL_OFFSET = 0x6ab
regSDMA1_QUEUE0_DUMMY_REG = 0x6b1
regSDMA1_QUEUE0_IB_BASE_HI = 0x68e
regSDMA1_QUEUE0_IB_BASE_LO = 0x68d
regSDMA1_QUEUE0_IB_CNTL = 0x68a
regSDMA1_QUEUE0_IB_OFFSET = 0x68c
regSDMA1_QUEUE0_IB_RPTR = 0x68b
regSDMA1_QUEUE0_IB_SIZE = 0x68f
regSDMA1_QUEUE0_IB_SUB_REMAIN = 0x6af
regSDMA1_QUEUE0_MIDCMD_CNTL = 0x6cb
regSDMA1_QUEUE0_MIDCMD_DATA0 = 0x6c0
regSDMA1_QUEUE0_MIDCMD_DATA1 = 0x6c1
regSDMA1_QUEUE0_MIDCMD_DATA10 = 0x6ca
regSDMA1_QUEUE0_MIDCMD_DATA2 = 0x6c2
regSDMA1_QUEUE0_MIDCMD_DATA3 = 0x6c3
regSDMA1_QUEUE0_MIDCMD_DATA4 = 0x6c4
regSDMA1_QUEUE0_MIDCMD_DATA5 = 0x6c5
regSDMA1_QUEUE0_MIDCMD_DATA6 = 0x6c6
regSDMA1_QUEUE0_MIDCMD_DATA7 = 0x6c7
regSDMA1_QUEUE0_MIDCMD_DATA8 = 0x6c8
regSDMA1_QUEUE0_MIDCMD_DATA9 = 0x6c9
regSDMA1_QUEUE0_MINOR_PTR_UPDATE = 0x6b5
regSDMA1_QUEUE0_PREEMPT = 0x6b0
regSDMA1_QUEUE0_RB_AQL_CNTL = 0x6b4
regSDMA1_QUEUE0_RB_BASE = 0x681
regSDMA1_QUEUE0_RB_BASE_HI = 0x682
regSDMA1_QUEUE0_RB_CNTL = 0x680
regSDMA1_QUEUE0_RB_PREEMPT = 0x6b6
regSDMA1_QUEUE0_RB_RPTR = 0x683
regSDMA1_QUEUE0_RB_RPTR_ADDR_HI = 0x688
regSDMA1_QUEUE0_RB_RPTR_ADDR_LO = 0x689
regSDMA1_QUEUE0_RB_RPTR_HI = 0x684
regSDMA1_QUEUE0_RB_WPTR = 0x685
regSDMA1_QUEUE0_RB_WPTR_HI = 0x686
regSDMA1_QUEUE0_RB_WPTR_POLL_ADDR_HI = 0x6b2
regSDMA1_QUEUE0_RB_WPTR_POLL_ADDR_LO = 0x6b3
regSDMA1_QUEUE0_SCHEDULE_CNTL = 0x6ae
regSDMA1_QUEUE0_SKIP_CNTL = 0x690
regSDMA1_QUEUE1_CONTEXT_STATUS = 0x6e9
regSDMA1_QUEUE1_CSA_ADDR_HI = 0x705
regSDMA1_QUEUE1_CSA_ADDR_LO = 0x704
regSDMA1_QUEUE1_DOORBELL = 0x6ea
regSDMA1_QUEUE1_DOORBELL_LOG = 0x701
regSDMA1_QUEUE1_DOORBELL_OFFSET = 0x703
regSDMA1_QUEUE1_DUMMY_REG = 0x709
regSDMA1_QUEUE1_IB_BASE_HI = 0x6e6
regSDMA1_QUEUE1_IB_BASE_LO = 0x6e5
regSDMA1_QUEUE1_IB_CNTL = 0x6e2
regSDMA1_QUEUE1_IB_OFFSET = 0x6e4
regSDMA1_QUEUE1_IB_RPTR = 0x6e3
regSDMA1_QUEUE1_IB_SIZE = 0x6e7
regSDMA1_QUEUE1_IB_SUB_REMAIN = 0x707
regSDMA1_QUEUE1_MIDCMD_CNTL = 0x723
regSDMA1_QUEUE1_MIDCMD_DATA0 = 0x718
regSDMA1_QUEUE1_MIDCMD_DATA1 = 0x719
regSDMA1_QUEUE1_MIDCMD_DATA10 = 0x722
regSDMA1_QUEUE1_MIDCMD_DATA2 = 0x71a
regSDMA1_QUEUE1_MIDCMD_DATA3 = 0x71b
regSDMA1_QUEUE1_MIDCMD_DATA4 = 0x71c
regSDMA1_QUEUE1_MIDCMD_DATA5 = 0x71d
regSDMA1_QUEUE1_MIDCMD_DATA6 = 0x71e
regSDMA1_QUEUE1_MIDCMD_DATA7 = 0x71f
regSDMA1_QUEUE1_MIDCMD_DATA8 = 0x720
regSDMA1_QUEUE1_MIDCMD_DATA9 = 0x721
regSDMA1_QUEUE1_MINOR_PTR_UPDATE = 0x70d
regSDMA1_QUEUE1_PREEMPT = 0x708
regSDMA1_QUEUE1_RB_AQL_CNTL = 0x70c
regSDMA1_QUEUE1_RB_BASE = 0x6d9
regSDMA1_QUEUE1_RB_BASE_HI = 0x6da
regSDMA1_QUEUE1_RB_CNTL = 0x6d8
regSDMA1_QUEUE1_RB_PREEMPT = 0x70e
regSDMA1_QUEUE1_RB_RPTR = 0x6db
regSDMA1_QUEUE1_RB_RPTR_ADDR_HI = 0x6e0
regSDMA1_QUEUE1_RB_RPTR_ADDR_LO = 0x6e1
regSDMA1_QUEUE1_RB_RPTR_HI = 0x6dc
regSDMA1_QUEUE1_RB_WPTR = 0x6dd
regSDMA1_QUEUE1_RB_WPTR_HI = 0x6de
regSDMA1_QUEUE1_RB_WPTR_POLL_ADDR_HI = 0x70a
regSDMA1_QUEUE1_RB_WPTR_POLL_ADDR_LO = 0x70b
regSDMA1_QUEUE1_SCHEDULE_CNTL = 0x706
regSDMA1_QUEUE1_SKIP_CNTL = 0x6e8
regSDMA1_QUEUE2_CONTEXT_STATUS = 0x741
regSDMA1_QUEUE2_CSA_ADDR_HI = 0x75d
regSDMA1_QUEUE2_CSA_ADDR_LO = 0x75c
regSDMA1_QUEUE2_DOORBELL = 0x742
regSDMA1_QUEUE2_DOORBELL_LOG = 0x759
regSDMA1_QUEUE2_DOORBELL_OFFSET = 0x75b
regSDMA1_QUEUE2_DUMMY_REG = 0x761
regSDMA1_QUEUE2_IB_BASE_HI = 0x73e
regSDMA1_QUEUE2_IB_BASE_LO = 0x73d
regSDMA1_QUEUE2_IB_CNTL = 0x73a
regSDMA1_QUEUE2_IB_OFFSET = 0x73c
regSDMA1_QUEUE2_IB_RPTR = 0x73b
regSDMA1_QUEUE2_IB_SIZE = 0x73f
regSDMA1_QUEUE2_IB_SUB_REMAIN = 0x75f
regSDMA1_QUEUE2_MIDCMD_CNTL = 0x77b
regSDMA1_QUEUE2_MIDCMD_DATA0 = 0x770
regSDMA1_QUEUE2_MIDCMD_DATA1 = 0x771
regSDMA1_QUEUE2_MIDCMD_DATA10 = 0x77a
regSDMA1_QUEUE2_MIDCMD_DATA2 = 0x772
regSDMA1_QUEUE2_MIDCMD_DATA3 = 0x773
regSDMA1_QUEUE2_MIDCMD_DATA4 = 0x774
regSDMA1_QUEUE2_MIDCMD_DATA5 = 0x775
regSDMA1_QUEUE2_MIDCMD_DATA6 = 0x776
regSDMA1_QUEUE2_MIDCMD_DATA7 = 0x777
regSDMA1_QUEUE2_MIDCMD_DATA8 = 0x778
regSDMA1_QUEUE2_MIDCMD_DATA9 = 0x779
regSDMA1_QUEUE2_MINOR_PTR_UPDATE = 0x765
regSDMA1_QUEUE2_PREEMPT = 0x760
regSDMA1_QUEUE2_RB_AQL_CNTL = 0x764
regSDMA1_QUEUE2_RB_BASE = 0x731
regSDMA1_QUEUE2_RB_BASE_HI = 0x732
regSDMA1_QUEUE2_RB_CNTL = 0x730
regSDMA1_QUEUE2_RB_PREEMPT = 0x766
regSDMA1_QUEUE2_RB_RPTR = 0x733
regSDMA1_QUEUE2_RB_RPTR_ADDR_HI = 0x738
regSDMA1_QUEUE2_RB_RPTR_ADDR_LO = 0x739
regSDMA1_QUEUE2_RB_RPTR_HI = 0x734
regSDMA1_QUEUE2_RB_WPTR = 0x735
regSDMA1_QUEUE2_RB_WPTR_HI = 0x736
regSDMA1_QUEUE2_RB_WPTR_POLL_ADDR_HI = 0x762
regSDMA1_QUEUE2_RB_WPTR_POLL_ADDR_LO = 0x763
regSDMA1_QUEUE2_SCHEDULE_CNTL = 0x75e
regSDMA1_QUEUE2_SKIP_CNTL = 0x740
regSDMA1_QUEUE3_CONTEXT_STATUS = 0x799
regSDMA1_QUEUE3_CSA_ADDR_HI = 0x7b5
regSDMA1_QUEUE3_CSA_ADDR_LO = 0x7b4
regSDMA1_QUEUE3_DOORBELL = 0x79a
regSDMA1_QUEUE3_DOORBELL_LOG = 0x7b1
regSDMA1_QUEUE3_DOORBELL_OFFSET = 0x7b3
regSDMA1_QUEUE3_DUMMY_REG = 0x7b9
regSDMA1_QUEUE3_IB_BASE_HI = 0x796
regSDMA1_QUEUE3_IB_BASE_LO = 0x795
regSDMA1_QUEUE3_IB_CNTL = 0x792
regSDMA1_QUEUE3_IB_OFFSET = 0x794
regSDMA1_QUEUE3_IB_RPTR = 0x793
regSDMA1_QUEUE3_IB_SIZE = 0x797
regSDMA1_QUEUE3_IB_SUB_REMAIN = 0x7b7
regSDMA1_QUEUE3_MIDCMD_CNTL = 0x7d3
regSDMA1_QUEUE3_MIDCMD_DATA0 = 0x7c8
regSDMA1_QUEUE3_MIDCMD_DATA1 = 0x7c9
regSDMA1_QUEUE3_MIDCMD_DATA10 = 0x7d2
regSDMA1_QUEUE3_MIDCMD_DATA2 = 0x7ca
regSDMA1_QUEUE3_MIDCMD_DATA3 = 0x7cb
regSDMA1_QUEUE3_MIDCMD_DATA4 = 0x7cc
regSDMA1_QUEUE3_MIDCMD_DATA5 = 0x7cd
regSDMA1_QUEUE3_MIDCMD_DATA6 = 0x7ce
regSDMA1_QUEUE3_MIDCMD_DATA7 = 0x7cf
regSDMA1_QUEUE3_MIDCMD_DATA8 = 0x7d0
regSDMA1_QUEUE3_MIDCMD_DATA9 = 0x7d1
regSDMA1_QUEUE3_MINOR_PTR_UPDATE = 0x7bd
regSDMA1_QUEUE3_PREEMPT = 0x7b8
regSDMA1_QUEUE3_RB_AQL_CNTL = 0x7bc
regSDMA1_QUEUE3_RB_BASE = 0x789
regSDMA1_QUEUE3_RB_BASE_HI = 0x78a
regSDMA1_QUEUE3_RB_CNTL = 0x788
regSDMA1_QUEUE3_RB_PREEMPT = 0x7be
regSDMA1_QUEUE3_RB_RPTR = 0x78b
regSDMA1_QUEUE3_RB_RPTR_ADDR_HI = 0x790
regSDMA1_QUEUE3_RB_RPTR_ADDR_LO = 0x791
regSDMA1_QUEUE3_RB_RPTR_HI = 0x78c
regSDMA1_QUEUE3_RB_WPTR = 0x78d
regSDMA1_QUEUE3_RB_WPTR_HI = 0x78e
regSDMA1_QUEUE3_RB_WPTR_POLL_ADDR_HI = 0x7ba
regSDMA1_QUEUE3_RB_WPTR_POLL_ADDR_LO = 0x7bb
regSDMA1_QUEUE3_SCHEDULE_CNTL = 0x7b6
regSDMA1_QUEUE3_SKIP_CNTL = 0x798
regSDMA1_QUEUE4_CONTEXT_STATUS = 0x7f1
regSDMA1_QUEUE4_CSA_ADDR_HI = 0x80d
regSDMA1_QUEUE4_CSA_ADDR_LO = 0x80c
regSDMA1_QUEUE4_DOORBELL = 0x7f2
regSDMA1_QUEUE4_DOORBELL_LOG = 0x809
regSDMA1_QUEUE4_DOORBELL_OFFSET = 0x80b
regSDMA1_QUEUE4_DUMMY_REG = 0x811
regSDMA1_QUEUE4_IB_BASE_HI = 0x7ee
regSDMA1_QUEUE4_IB_BASE_LO = 0x7ed
regSDMA1_QUEUE4_IB_CNTL = 0x7ea
regSDMA1_QUEUE4_IB_OFFSET = 0x7ec
regSDMA1_QUEUE4_IB_RPTR = 0x7eb
regSDMA1_QUEUE4_IB_SIZE = 0x7ef
regSDMA1_QUEUE4_IB_SUB_REMAIN = 0x80f
regSDMA1_QUEUE4_MIDCMD_CNTL = 0x82b
regSDMA1_QUEUE4_MIDCMD_DATA0 = 0x820
regSDMA1_QUEUE4_MIDCMD_DATA1 = 0x821
regSDMA1_QUEUE4_MIDCMD_DATA10 = 0x82a
regSDMA1_QUEUE4_MIDCMD_DATA2 = 0x822
regSDMA1_QUEUE4_MIDCMD_DATA3 = 0x823
regSDMA1_QUEUE4_MIDCMD_DATA4 = 0x824
regSDMA1_QUEUE4_MIDCMD_DATA5 = 0x825
regSDMA1_QUEUE4_MIDCMD_DATA6 = 0x826
regSDMA1_QUEUE4_MIDCMD_DATA7 = 0x827
regSDMA1_QUEUE4_MIDCMD_DATA8 = 0x828
regSDMA1_QUEUE4_MIDCMD_DATA9 = 0x829
regSDMA1_QUEUE4_MINOR_PTR_UPDATE = 0x815
regSDMA1_QUEUE4_PREEMPT = 0x810
regSDMA1_QUEUE4_RB_AQL_CNTL = 0x814
regSDMA1_QUEUE4_RB_BASE = 0x7e1
regSDMA1_QUEUE4_RB_BASE_HI = 0x7e2
regSDMA1_QUEUE4_RB_CNTL = 0x7e0
regSDMA1_QUEUE4_RB_PREEMPT = 0x816
regSDMA1_QUEUE4_RB_RPTR = 0x7e3
regSDMA1_QUEUE4_RB_RPTR_ADDR_HI = 0x7e8
regSDMA1_QUEUE4_RB_RPTR_ADDR_LO = 0x7e9
regSDMA1_QUEUE4_RB_RPTR_HI = 0x7e4
regSDMA1_QUEUE4_RB_WPTR = 0x7e5
regSDMA1_QUEUE4_RB_WPTR_HI = 0x7e6
regSDMA1_QUEUE4_RB_WPTR_POLL_ADDR_HI = 0x812
regSDMA1_QUEUE4_RB_WPTR_POLL_ADDR_LO = 0x813
regSDMA1_QUEUE4_SCHEDULE_CNTL = 0x80e
regSDMA1_QUEUE4_SKIP_CNTL = 0x7f0
regSDMA1_QUEUE5_CONTEXT_STATUS = 0x849
regSDMA1_QUEUE5_CSA_ADDR_HI = 0x865
regSDMA1_QUEUE5_CSA_ADDR_LO = 0x864
regSDMA1_QUEUE5_DOORBELL = 0x84a
regSDMA1_QUEUE5_DOORBELL_LOG = 0x861
regSDMA1_QUEUE5_DOORBELL_OFFSET = 0x863
regSDMA1_QUEUE5_DUMMY_REG = 0x869
regSDMA1_QUEUE5_IB_BASE_HI = 0x846
regSDMA1_QUEUE5_IB_BASE_LO = 0x845
regSDMA1_QUEUE5_IB_CNTL = 0x842
regSDMA1_QUEUE5_IB_OFFSET = 0x844
regSDMA1_QUEUE5_IB_RPTR = 0x843
regSDMA1_QUEUE5_IB_SIZE = 0x847
regSDMA1_QUEUE5_IB_SUB_REMAIN = 0x867
regSDMA1_QUEUE5_MIDCMD_CNTL = 0x883
regSDMA1_QUEUE5_MIDCMD_DATA0 = 0x878
regSDMA1_QUEUE5_MIDCMD_DATA1 = 0x879
regSDMA1_QUEUE5_MIDCMD_DATA10 = 0x882
regSDMA1_QUEUE5_MIDCMD_DATA2 = 0x87a
regSDMA1_QUEUE5_MIDCMD_DATA3 = 0x87b
regSDMA1_QUEUE5_MIDCMD_DATA4 = 0x87c
regSDMA1_QUEUE5_MIDCMD_DATA5 = 0x87d
regSDMA1_QUEUE5_MIDCMD_DATA6 = 0x87e
regSDMA1_QUEUE5_MIDCMD_DATA7 = 0x87f
regSDMA1_QUEUE5_MIDCMD_DATA8 = 0x880
regSDMA1_QUEUE5_MIDCMD_DATA9 = 0x881
regSDMA1_QUEUE5_MINOR_PTR_UPDATE = 0x86d
regSDMA1_QUEUE5_PREEMPT = 0x868
regSDMA1_QUEUE5_RB_AQL_CNTL = 0x86c
regSDMA1_QUEUE5_RB_BASE = 0x839
regSDMA1_QUEUE5_RB_BASE_HI = 0x83a
regSDMA1_QUEUE5_RB_CNTL = 0x838
regSDMA1_QUEUE5_RB_PREEMPT = 0x86e
regSDMA1_QUEUE5_RB_RPTR = 0x83b
regSDMA1_QUEUE5_RB_RPTR_ADDR_HI = 0x840
regSDMA1_QUEUE5_RB_RPTR_ADDR_LO = 0x841
regSDMA1_QUEUE5_RB_RPTR_HI = 0x83c
regSDMA1_QUEUE5_RB_WPTR = 0x83d
regSDMA1_QUEUE5_RB_WPTR_HI = 0x83e
regSDMA1_QUEUE5_RB_WPTR_POLL_ADDR_HI = 0x86a
regSDMA1_QUEUE5_RB_WPTR_POLL_ADDR_LO = 0x86b
regSDMA1_QUEUE5_SCHEDULE_CNTL = 0x866
regSDMA1_QUEUE5_SKIP_CNTL = 0x848
regSDMA1_QUEUE6_CONTEXT_STATUS = 0x8a1
regSDMA1_QUEUE6_CSA_ADDR_HI = 0x8bd
regSDMA1_QUEUE6_CSA_ADDR_LO = 0x8bc
regSDMA1_QUEUE6_DOORBELL = 0x8a2
regSDMA1_QUEUE6_DOORBELL_LOG = 0x8b9
regSDMA1_QUEUE6_DOORBELL_OFFSET = 0x8bb
regSDMA1_QUEUE6_DUMMY_REG = 0x8c1
regSDMA1_QUEUE6_IB_BASE_HI = 0x89e
regSDMA1_QUEUE6_IB_BASE_LO = 0x89d
regSDMA1_QUEUE6_IB_CNTL = 0x89a
regSDMA1_QUEUE6_IB_OFFSET = 0x89c
regSDMA1_QUEUE6_IB_RPTR = 0x89b
regSDMA1_QUEUE6_IB_SIZE = 0x89f
regSDMA1_QUEUE6_IB_SUB_REMAIN = 0x8bf
regSDMA1_QUEUE6_MIDCMD_CNTL = 0x8db
regSDMA1_QUEUE6_MIDCMD_DATA0 = 0x8d0
regSDMA1_QUEUE6_MIDCMD_DATA1 = 0x8d1
regSDMA1_QUEUE6_MIDCMD_DATA10 = 0x8da
regSDMA1_QUEUE6_MIDCMD_DATA2 = 0x8d2
regSDMA1_QUEUE6_MIDCMD_DATA3 = 0x8d3
regSDMA1_QUEUE6_MIDCMD_DATA4 = 0x8d4
regSDMA1_QUEUE6_MIDCMD_DATA5 = 0x8d5
regSDMA1_QUEUE6_MIDCMD_DATA6 = 0x8d6
regSDMA1_QUEUE6_MIDCMD_DATA7 = 0x8d7
regSDMA1_QUEUE6_MIDCMD_DATA8 = 0x8d8
regSDMA1_QUEUE6_MIDCMD_DATA9 = 0x8d9
regSDMA1_QUEUE6_MINOR_PTR_UPDATE = 0x8c5
regSDMA1_QUEUE6_PREEMPT = 0x8c0
regSDMA1_QUEUE6_RB_AQL_CNTL = 0x8c4
regSDMA1_QUEUE6_RB_BASE = 0x891
regSDMA1_QUEUE6_RB_BASE_HI = 0x892
regSDMA1_QUEUE6_RB_CNTL = 0x890
regSDMA1_QUEUE6_RB_PREEMPT = 0x8c6
regSDMA1_QUEUE6_RB_RPTR = 0x893
regSDMA1_QUEUE6_RB_RPTR_ADDR_HI = 0x898
regSDMA1_QUEUE6_RB_RPTR_ADDR_LO = 0x899
regSDMA1_QUEUE6_RB_RPTR_HI = 0x894
regSDMA1_QUEUE6_RB_WPTR = 0x895
regSDMA1_QUEUE6_RB_WPTR_HI = 0x896
regSDMA1_QUEUE6_RB_WPTR_POLL_ADDR_HI = 0x8c2
regSDMA1_QUEUE6_RB_WPTR_POLL_ADDR_LO = 0x8c3
regSDMA1_QUEUE6_SCHEDULE_CNTL = 0x8be
regSDMA1_QUEUE6_SKIP_CNTL = 0x8a0
regSDMA1_QUEUE7_CONTEXT_STATUS = 0x8f9
regSDMA1_QUEUE7_CSA_ADDR_HI = 0x915
regSDMA1_QUEUE7_CSA_ADDR_LO = 0x914
regSDMA1_QUEUE7_DOORBELL = 0x8fa
regSDMA1_QUEUE7_DOORBELL_LOG = 0x911
regSDMA1_QUEUE7_DOORBELL_OFFSET = 0x913
regSDMA1_QUEUE7_DUMMY_REG = 0x919
regSDMA1_QUEUE7_IB_BASE_HI = 0x8f6
regSDMA1_QUEUE7_IB_BASE_LO = 0x8f5
regSDMA1_QUEUE7_IB_CNTL = 0x8f2
regSDMA1_QUEUE7_IB_OFFSET = 0x8f4
regSDMA1_QUEUE7_IB_RPTR = 0x8f3
regSDMA1_QUEUE7_IB_SIZE = 0x8f7
regSDMA1_QUEUE7_IB_SUB_REMAIN = 0x917
regSDMA1_QUEUE7_MIDCMD_CNTL = 0x933
regSDMA1_QUEUE7_MIDCMD_DATA0 = 0x928
regSDMA1_QUEUE7_MIDCMD_DATA1 = 0x929
regSDMA1_QUEUE7_MIDCMD_DATA10 = 0x932
regSDMA1_QUEUE7_MIDCMD_DATA2 = 0x92a
regSDMA1_QUEUE7_MIDCMD_DATA3 = 0x92b
regSDMA1_QUEUE7_MIDCMD_DATA4 = 0x92c
regSDMA1_QUEUE7_MIDCMD_DATA5 = 0x92d
regSDMA1_QUEUE7_MIDCMD_DATA6 = 0x92e
regSDMA1_QUEUE7_MIDCMD_DATA7 = 0x92f
regSDMA1_QUEUE7_MIDCMD_DATA8 = 0x930
regSDMA1_QUEUE7_MIDCMD_DATA9 = 0x931
regSDMA1_QUEUE7_MINOR_PTR_UPDATE = 0x91d
regSDMA1_QUEUE7_PREEMPT = 0x918
regSDMA1_QUEUE7_RB_AQL_CNTL = 0x91c
regSDMA1_QUEUE7_RB_BASE = 0x8e9
regSDMA1_QUEUE7_RB_BASE_HI = 0x8ea
regSDMA1_QUEUE7_RB_CNTL = 0x8e8
regSDMA1_QUEUE7_RB_PREEMPT = 0x91e
regSDMA1_QUEUE7_RB_RPTR = 0x8eb
regSDMA1_QUEUE7_RB_RPTR_ADDR_HI = 0x8f0
regSDMA1_QUEUE7_RB_RPTR_ADDR_LO = 0x8f1
regSDMA1_QUEUE7_RB_RPTR_HI = 0x8ec
regSDMA1_QUEUE7_RB_WPTR = 0x8ed
regSDMA1_QUEUE7_RB_WPTR_HI = 0x8ee
regSDMA1_QUEUE7_RB_WPTR_POLL_ADDR_HI = 0x91a
regSDMA1_QUEUE7_RB_WPTR_POLL_ADDR_LO = 0x91b
regSDMA1_QUEUE7_SCHEDULE_CNTL = 0x916
regSDMA1_QUEUE7_SKIP_CNTL = 0x8f8
regSDMA1_QUEUE_RESET_REQ = 0x67b
regSDMA1_QUEUE_STATUS0 = 0x62f
regSDMA1_RB_RPTR_FETCH = 0x620
regSDMA1_RB_RPTR_FETCH_HI = 0x621
regSDMA1_RELAX_ORDERING_LUT = 0x64a
regSDMA1_RLC_CGCG_CTRL = 0x65c
regSDMA1_SCRATCH_RAM_ADDR = 0x678
regSDMA1_SCRATCH_RAM_DATA = 0x677
regSDMA1_SEM_WAIT_FAIL_TIMER_CNTL = 0x622
regSDMA1_STATUS1_REG = 0x626
regSDMA1_STATUS2_REG = 0x638
regSDMA1_STATUS3_REG = 0x64c
regSDMA1_STATUS4_REG = 0x676
regSDMA1_STATUS5_REG = 0x67a
regSDMA1_STATUS6_REG = 0x67c
regSDMA1_STATUS_REG = 0x625
regSDMA1_TILING_CONFIG = 0x663
regSDMA1_TIMESTAMP_CNTL = 0x679
regSDMA1_TLBI_GCR_CNTL = 0x662
regSDMA1_UCODE1_CHECKSUM = 0x67d
regSDMA1_UCODE_ADDR = 0x58a0
regSDMA1_UCODE_CHECKSUM = 0x629
regSDMA1_UCODE_DATA = 0x58a1
regSDMA1_UCODE_SELFLOAD_CONTROL = 0x58a2
regSDMA1_UTCL1_CNTL = 0x63c
regSDMA1_UTCL1_INV0 = 0x642
regSDMA1_UTCL1_INV1 = 0x643
regSDMA1_UTCL1_INV2 = 0x644
regSDMA1_UTCL1_PAGE = 0x63f
regSDMA1_UTCL1_RD_STATUS = 0x640
regSDMA1_UTCL1_RD_XNACK0 = 0x645
regSDMA1_UTCL1_RD_XNACK1 = 0x646
regSDMA1_UTCL1_TIMEOUT = 0x63e
regSDMA1_UTCL1_WATERMK = 0x63d
regSDMA1_UTCL1_WR_STATUS = 0x641
regSDMA1_UTCL1_WR_XNACK0 = 0x647
regSDMA1_UTCL1_WR_XNACK1 = 0x648
regSDMA1_VERSION = 0x635
regSDMA1_WATCHDOG_CNTL = 0x62e
regSE0_CAC_AGGR_GFXCLK_CYCLE = 0x1ae5
regSE0_CAC_AGGR_LOWER = 0x1ad4
regSE0_CAC_AGGR_UPPER = 0x1ad5
regSE1_CAC_AGGR_GFXCLK_CYCLE = 0x1ae6
regSE1_CAC_AGGR_LOWER = 0x1ad6
regSE1_CAC_AGGR_UPPER = 0x1ad7
regSE2_CAC_AGGR_GFXCLK_CYCLE = 0x1ae7
regSE2_CAC_AGGR_LOWER = 0x1ad8
regSE2_CAC_AGGR_UPPER = 0x1ad9
regSE3_CAC_AGGR_GFXCLK_CYCLE = 0x1ae8
regSE3_CAC_AGGR_LOWER = 0x1ada
regSE3_CAC_AGGR_UPPER = 0x1adb
regSE4_CAC_AGGR_GFXCLK_CYCLE = 0x1ae9
regSE4_CAC_AGGR_LOWER = 0x1adc
regSE4_CAC_AGGR_UPPER = 0x1add
regSE5_CAC_AGGR_GFXCLK_CYCLE = 0x1aea
regSE5_CAC_AGGR_LOWER = 0x1ade
regSE5_CAC_AGGR_UPPER = 0x1adf
regSEDC_GL1_GL2_OVERRIDES = 0x1ac0
regSE_CAC_CTRL_1 = 0x1b70
regSE_CAC_CTRL_2 = 0x1b71
regSE_CAC_IND_DATA = 0x1bcf
regSE_CAC_IND_INDEX = 0x1bce
regSE_CAC_WEIGHT_BCI_0 = 0x1b8a
regSE_CAC_WEIGHT_CB_0 = 0x1b8b
regSE_CAC_WEIGHT_CB_1 = 0x1b8c
regSE_CAC_WEIGHT_CB_10 = 0x1b95
regSE_CAC_WEIGHT_CB_11 = 0x1b96
regSE_CAC_WEIGHT_CB_2 = 0x1b8d
regSE_CAC_WEIGHT_CB_3 = 0x1b8e
regSE_CAC_WEIGHT_CB_4 = 0x1b8f
regSE_CAC_WEIGHT_CB_5 = 0x1b90
regSE_CAC_WEIGHT_CB_6 = 0x1b91
regSE_CAC_WEIGHT_CB_7 = 0x1b92
regSE_CAC_WEIGHT_CB_8 = 0x1b93
regSE_CAC_WEIGHT_CB_9 = 0x1b94
regSE_CAC_WEIGHT_CU_0 = 0x1b89
regSE_CAC_WEIGHT_DB_0 = 0x1b97
regSE_CAC_WEIGHT_DB_1 = 0x1b98
regSE_CAC_WEIGHT_DB_2 = 0x1b99
regSE_CAC_WEIGHT_DB_3 = 0x1b9a
regSE_CAC_WEIGHT_DB_4 = 0x1b9b
regSE_CAC_WEIGHT_GL1C_0 = 0x1ba1
regSE_CAC_WEIGHT_GL1C_1 = 0x1ba2
regSE_CAC_WEIGHT_GL1C_2 = 0x1ba3
regSE_CAC_WEIGHT_LDS_0 = 0x1b82
regSE_CAC_WEIGHT_LDS_1 = 0x1b83
regSE_CAC_WEIGHT_LDS_2 = 0x1b84
regSE_CAC_WEIGHT_LDS_3 = 0x1b85
regSE_CAC_WEIGHT_PA_0 = 0x1ba8
regSE_CAC_WEIGHT_PA_1 = 0x1ba9
regSE_CAC_WEIGHT_PA_2 = 0x1baa
regSE_CAC_WEIGHT_PA_3 = 0x1bab
regSE_CAC_WEIGHT_PC_0 = 0x1ba7
regSE_CAC_WEIGHT_RMI_0 = 0x1b9c
regSE_CAC_WEIGHT_RMI_1 = 0x1b9d
regSE_CAC_WEIGHT_SC_0 = 0x1bac
regSE_CAC_WEIGHT_SC_1 = 0x1bad
regSE_CAC_WEIGHT_SC_2 = 0x1bae
regSE_CAC_WEIGHT_SC_3 = 0x1baf
regSE_CAC_WEIGHT_SPI_0 = 0x1ba4
regSE_CAC_WEIGHT_SPI_1 = 0x1ba5
regSE_CAC_WEIGHT_SPI_2 = 0x1ba6
regSE_CAC_WEIGHT_SP_0 = 0x1b80
regSE_CAC_WEIGHT_SP_1 = 0x1b81
regSE_CAC_WEIGHT_SQC_0 = 0x1b87
regSE_CAC_WEIGHT_SQC_1 = 0x1b88
regSE_CAC_WEIGHT_SQ_0 = 0x1b7d
regSE_CAC_WEIGHT_SQ_1 = 0x1b7e
regSE_CAC_WEIGHT_SQ_2 = 0x1b7f
regSE_CAC_WEIGHT_SXRB_0 = 0x1b9f
regSE_CAC_WEIGHT_SX_0 = 0x1b9e
regSE_CAC_WEIGHT_TA_0 = 0x1b72
regSE_CAC_WEIGHT_TCP_0 = 0x1b79
regSE_CAC_WEIGHT_TCP_1 = 0x1b7a
regSE_CAC_WEIGHT_TCP_2 = 0x1b7b
regSE_CAC_WEIGHT_TCP_3 = 0x1b7c
regSE_CAC_WEIGHT_TD_0 = 0x1b73
regSE_CAC_WEIGHT_TD_1 = 0x1b74
regSE_CAC_WEIGHT_TD_2 = 0x1b75
regSE_CAC_WEIGHT_TD_3 = 0x1b76
regSE_CAC_WEIGHT_TD_4 = 0x1b77
regSE_CAC_WEIGHT_TD_5 = 0x1b78
regSE_CAC_WEIGHT_UTCL1_0 = 0x1ba0
regSE_CAC_WINDOW_AGGR_VALUE = 0x1bb0
regSE_CAC_WINDOW_GFXCLK_CYCLE = 0x1bb1
regSH_MEM_BASES = 0x9e3
regSH_MEM_CONFIG = 0x9e4
regSH_RESERVED_REG0 = 0x1c20
regSH_RESERVED_REG1 = 0x1c21
regSMU_RLC_RESPONSE = 0x4e01
regSPI_ARB_CNTL_0 = 0x1949
regSPI_ARB_CYCLES_0 = 0x1f61
regSPI_ARB_CYCLES_1 = 0x1f62
regSPI_ARB_PRIORITY = 0x1f60
regSPI_ATTRIBUTE_RING_BASE = 0x2446
regSPI_ATTRIBUTE_RING_SIZE = 0x2447
regSPI_BARYC_CNTL = 0x1b8
regSPI_COMPUTE_QUEUE_RESET = 0x1f73
regSPI_COMPUTE_WF_CTX_SAVE = 0x1f74
regSPI_COMPUTE_WF_CTX_SAVE_STATUS = 0x194e
regSPI_CONFIG_CNTL = 0x2440
regSPI_CONFIG_CNTL_1 = 0x2441
regSPI_CONFIG_CNTL_2 = 0x2442
regSPI_CONFIG_PS_CU_EN = 0x11f2
regSPI_CSQ_WF_ACTIVE_COUNT_0 = 0x127c
regSPI_CSQ_WF_ACTIVE_COUNT_1 = 0x127d
regSPI_CSQ_WF_ACTIVE_COUNT_2 = 0x127e
regSPI_CSQ_WF_ACTIVE_COUNT_3 = 0x127f
regSPI_CSQ_WF_ACTIVE_STATUS = 0x127b
regSPI_DSM_CNTL = 0x11e3
regSPI_DSM_CNTL2 = 0x11e4
regSPI_EDC_CNT = 0x11e5
regSPI_EXP_THROTTLE_CTRL = 0x14c3
regSPI_FEATURE_CTRL = 0x194a
regSPI_GDBG_PER_VMID_CNTL = 0x1f72
regSPI_GDBG_TRAP_CONFIG = 0x1944
regSPI_GDBG_WAVE_CNTL = 0x1943
regSPI_GDBG_WAVE_CNTL3 = 0x1945
regSPI_GDS_CREDITS = 0x1278
regSPI_GFX_CNTL = 0x11dc
regSPI_GFX_SCRATCH_BASE_HI = 0x1bc
regSPI_GFX_SCRATCH_BASE_LO = 0x1bb
regSPI_GS_THROTTLE_CNTL1 = 0x2444
regSPI_GS_THROTTLE_CNTL2 = 0x2445
regSPI_INTERP_CONTROL_0 = 0x1b5
regSPI_LB_CTR_CTRL = 0x1274
regSPI_LB_DATA_REG = 0x1276
regSPI_LB_DATA_WAVES = 0x1284
regSPI_LB_WGP_MASK = 0x1275
regSPI_P0_TRAP_SCREEN_GPR_MIN = 0x1290
regSPI_P0_TRAP_SCREEN_PSBA_HI = 0x128d
regSPI_P0_TRAP_SCREEN_PSBA_LO = 0x128c
regSPI_P0_TRAP_SCREEN_PSMA_HI = 0x128f
regSPI_P0_TRAP_SCREEN_PSMA_LO = 0x128e
regSPI_P1_TRAP_SCREEN_GPR_MIN = 0x1295
regSPI_P1_TRAP_SCREEN_PSBA_HI = 0x1292
regSPI_P1_TRAP_SCREEN_PSBA_LO = 0x1291
regSPI_P1_TRAP_SCREEN_PSMA_HI = 0x1294
regSPI_P1_TRAP_SCREEN_PSMA_LO = 0x1293
regSPI_PERFCOUNTER0_HI = 0x3180
regSPI_PERFCOUNTER0_LO = 0x3181
regSPI_PERFCOUNTER0_SELECT = 0x3980
regSPI_PERFCOUNTER0_SELECT1 = 0x3984
regSPI_PERFCOUNTER1_HI = 0x3182
regSPI_PERFCOUNTER1_LO = 0x3183
regSPI_PERFCOUNTER1_SELECT = 0x3981
regSPI_PERFCOUNTER1_SELECT1 = 0x3985
regSPI_PERFCOUNTER2_HI = 0x3184
regSPI_PERFCOUNTER2_LO = 0x3185
regSPI_PERFCOUNTER2_SELECT = 0x3982
regSPI_PERFCOUNTER2_SELECT1 = 0x3986
regSPI_PERFCOUNTER3_HI = 0x3186
regSPI_PERFCOUNTER3_LO = 0x3187
regSPI_PERFCOUNTER3_SELECT = 0x3983
regSPI_PERFCOUNTER3_SELECT1 = 0x3987
regSPI_PERFCOUNTER4_HI = 0x3188
regSPI_PERFCOUNTER4_LO = 0x3189
regSPI_PERFCOUNTER4_SELECT = 0x3988
regSPI_PERFCOUNTER5_HI = 0x318a
regSPI_PERFCOUNTER5_LO = 0x318b
regSPI_PERFCOUNTER5_SELECT = 0x3989
regSPI_PERFCOUNTER_BINS = 0x398a
regSPI_PG_ENABLE_STATIC_WGP_MASK = 0x1277
regSPI_PQEV_CTRL = 0x14c0
regSPI_PS_INPUT_ADDR = 0x1b4
regSPI_PS_INPUT_CNTL_0 = 0x191
regSPI_PS_INPUT_CNTL_1 = 0x192
regSPI_PS_INPUT_CNTL_10 = 0x19b
regSPI_PS_INPUT_CNTL_11 = 0x19c
regSPI_PS_INPUT_CNTL_12 = 0x19d
regSPI_PS_INPUT_CNTL_13 = 0x19e
regSPI_PS_INPUT_CNTL_14 = 0x19f
regSPI_PS_INPUT_CNTL_15 = 0x1a0
regSPI_PS_INPUT_CNTL_16 = 0x1a1
regSPI_PS_INPUT_CNTL_17 = 0x1a2
regSPI_PS_INPUT_CNTL_18 = 0x1a3
regSPI_PS_INPUT_CNTL_19 = 0x1a4
regSPI_PS_INPUT_CNTL_2 = 0x193
regSPI_PS_INPUT_CNTL_20 = 0x1a5
regSPI_PS_INPUT_CNTL_21 = 0x1a6
regSPI_PS_INPUT_CNTL_22 = 0x1a7
regSPI_PS_INPUT_CNTL_23 = 0x1a8
regSPI_PS_INPUT_CNTL_24 = 0x1a9
regSPI_PS_INPUT_CNTL_25 = 0x1aa
regSPI_PS_INPUT_CNTL_26 = 0x1ab
regSPI_PS_INPUT_CNTL_27 = 0x1ac
regSPI_PS_INPUT_CNTL_28 = 0x1ad
regSPI_PS_INPUT_CNTL_29 = 0x1ae
regSPI_PS_INPUT_CNTL_3 = 0x194
regSPI_PS_INPUT_CNTL_30 = 0x1af
regSPI_PS_INPUT_CNTL_31 = 0x1b0
regSPI_PS_INPUT_CNTL_4 = 0x195
regSPI_PS_INPUT_CNTL_5 = 0x196
regSPI_PS_INPUT_CNTL_6 = 0x197
regSPI_PS_INPUT_CNTL_7 = 0x198
regSPI_PS_INPUT_CNTL_8 = 0x199
regSPI_PS_INPUT_CNTL_9 = 0x19a
regSPI_PS_INPUT_ENA = 0x1b3
regSPI_PS_IN_CONTROL = 0x1b6
regSPI_PS_MAX_WAVE_ID = 0x11da
regSPI_RESOURCE_RESERVE_CU_0 = 0x1c00
regSPI_RESOURCE_RESERVE_CU_1 = 0x1c01
regSPI_RESOURCE_RESERVE_CU_10 = 0x1c0a
regSPI_RESOURCE_RESERVE_CU_11 = 0x1c0b
regSPI_RESOURCE_RESERVE_CU_12 = 0x1c0c
regSPI_RESOURCE_RESERVE_CU_13 = 0x1c0d
regSPI_RESOURCE_RESERVE_CU_14 = 0x1c0e
regSPI_RESOURCE_RESERVE_CU_15 = 0x1c0f
regSPI_RESOURCE_RESERVE_CU_2 = 0x1c02
regSPI_RESOURCE_RESERVE_CU_3 = 0x1c03
regSPI_RESOURCE_RESERVE_CU_4 = 0x1c04
regSPI_RESOURCE_RESERVE_CU_5 = 0x1c05
regSPI_RESOURCE_RESERVE_CU_6 = 0x1c06
regSPI_RESOURCE_RESERVE_CU_7 = 0x1c07
regSPI_RESOURCE_RESERVE_CU_8 = 0x1c08
regSPI_RESOURCE_RESERVE_CU_9 = 0x1c09
regSPI_RESOURCE_RESERVE_EN_CU_0 = 0x1c10
regSPI_RESOURCE_RESERVE_EN_CU_1 = 0x1c11
regSPI_RESOURCE_RESERVE_EN_CU_10 = 0x1c1a
regSPI_RESOURCE_RESERVE_EN_CU_11 = 0x1c1b
regSPI_RESOURCE_RESERVE_EN_CU_12 = 0x1c1c
regSPI_RESOURCE_RESERVE_EN_CU_13 = 0x1c1d
regSPI_RESOURCE_RESERVE_EN_CU_14 = 0x1c1e
regSPI_RESOURCE_RESERVE_EN_CU_15 = 0x1c1f
regSPI_RESOURCE_RESERVE_EN_CU_2 = 0x1c12
regSPI_RESOURCE_RESERVE_EN_CU_3 = 0x1c13
regSPI_RESOURCE_RESERVE_EN_CU_4 = 0x1c14
regSPI_RESOURCE_RESERVE_EN_CU_5 = 0x1c15
regSPI_RESOURCE_RESERVE_EN_CU_6 = 0x1c16
regSPI_RESOURCE_RESERVE_EN_CU_7 = 0x1c17
regSPI_RESOURCE_RESERVE_EN_CU_8 = 0x1c18
regSPI_RESOURCE_RESERVE_EN_CU_9 = 0x1c19
regSPI_SHADER_COL_FORMAT = 0x1c5
regSPI_SHADER_GS_MESHLET_DIM = 0x1a4c
regSPI_SHADER_GS_MESHLET_EXP_ALLOC = 0x1a4d
regSPI_SHADER_IDX_FORMAT = 0x1c2
regSPI_SHADER_PGM_CHKSUM_GS = 0x1a20
regSPI_SHADER_PGM_CHKSUM_HS = 0x1aa0
regSPI_SHADER_PGM_CHKSUM_PS = 0x19a6
regSPI_SHADER_PGM_HI_ES = 0x1a69
regSPI_SHADER_PGM_HI_ES_GS = 0x1a25
regSPI_SHADER_PGM_HI_GS = 0x1a29
regSPI_SHADER_PGM_HI_HS = 0x1aa9
regSPI_SHADER_PGM_HI_LS = 0x1ae9
regSPI_SHADER_PGM_HI_LS_HS = 0x1aa5
regSPI_SHADER_PGM_HI_PS = 0x19a9
regSPI_SHADER_PGM_LO_ES = 0x1a68
regSPI_SHADER_PGM_LO_ES_GS = 0x1a24
regSPI_SHADER_PGM_LO_GS = 0x1a28
regSPI_SHADER_PGM_LO_HS = 0x1aa8
regSPI_SHADER_PGM_LO_LS = 0x1ae8
regSPI_SHADER_PGM_LO_LS_HS = 0x1aa4
regSPI_SHADER_PGM_LO_PS = 0x19a8
regSPI_SHADER_PGM_RSRC1_GS = 0x1a2a
regSPI_SHADER_PGM_RSRC1_HS = 0x1aaa
regSPI_SHADER_PGM_RSRC1_PS = 0x19aa
regSPI_SHADER_PGM_RSRC2_GS = 0x1a2b
regSPI_SHADER_PGM_RSRC2_HS = 0x1aab
regSPI_SHADER_PGM_RSRC2_PS = 0x19ab
regSPI_SHADER_PGM_RSRC3_GS = 0x1a27
regSPI_SHADER_PGM_RSRC3_HS = 0x1aa7
regSPI_SHADER_PGM_RSRC3_PS = 0x19a7
regSPI_SHADER_PGM_RSRC4_GS = 0x1a21
regSPI_SHADER_PGM_RSRC4_HS = 0x1aa1
regSPI_SHADER_PGM_RSRC4_PS = 0x19a1
regSPI_SHADER_POS_FORMAT = 0x1c3
regSPI_SHADER_REQ_CTRL_ESGS = 0x1a50
regSPI_SHADER_REQ_CTRL_LSHS = 0x1ad0
regSPI_SHADER_REQ_CTRL_PS = 0x19d0
regSPI_SHADER_RSRC_LIMIT_CTRL = 0x194b
regSPI_SHADER_USER_ACCUM_ESGS_0 = 0x1a52
regSPI_SHADER_USER_ACCUM_ESGS_1 = 0x1a53
regSPI_SHADER_USER_ACCUM_ESGS_2 = 0x1a54
regSPI_SHADER_USER_ACCUM_ESGS_3 = 0x1a55
regSPI_SHADER_USER_ACCUM_LSHS_0 = 0x1ad2
regSPI_SHADER_USER_ACCUM_LSHS_1 = 0x1ad3
regSPI_SHADER_USER_ACCUM_LSHS_2 = 0x1ad4
regSPI_SHADER_USER_ACCUM_LSHS_3 = 0x1ad5
regSPI_SHADER_USER_ACCUM_PS_0 = 0x19d2
regSPI_SHADER_USER_ACCUM_PS_1 = 0x19d3
regSPI_SHADER_USER_ACCUM_PS_2 = 0x19d4
regSPI_SHADER_USER_ACCUM_PS_3 = 0x19d5
regSPI_SHADER_USER_DATA_ADDR_HI_GS = 0x1a23
regSPI_SHADER_USER_DATA_ADDR_HI_HS = 0x1aa3
regSPI_SHADER_USER_DATA_ADDR_LO_GS = 0x1a22
regSPI_SHADER_USER_DATA_ADDR_LO_HS = 0x1aa2
regSPI_SHADER_USER_DATA_GS_0 = 0x1a2c
regSPI_SHADER_USER_DATA_GS_1 = 0x1a2d
regSPI_SHADER_USER_DATA_GS_10 = 0x1a36
regSPI_SHADER_USER_DATA_GS_11 = 0x1a37
regSPI_SHADER_USER_DATA_GS_12 = 0x1a38
regSPI_SHADER_USER_DATA_GS_13 = 0x1a39
regSPI_SHADER_USER_DATA_GS_14 = 0x1a3a
regSPI_SHADER_USER_DATA_GS_15 = 0x1a3b
regSPI_SHADER_USER_DATA_GS_16 = 0x1a3c
regSPI_SHADER_USER_DATA_GS_17 = 0x1a3d
regSPI_SHADER_USER_DATA_GS_18 = 0x1a3e
regSPI_SHADER_USER_DATA_GS_19 = 0x1a3f
regSPI_SHADER_USER_DATA_GS_2 = 0x1a2e
regSPI_SHADER_USER_DATA_GS_20 = 0x1a40
regSPI_SHADER_USER_DATA_GS_21 = 0x1a41
regSPI_SHADER_USER_DATA_GS_22 = 0x1a42
regSPI_SHADER_USER_DATA_GS_23 = 0x1a43
regSPI_SHADER_USER_DATA_GS_24 = 0x1a44
regSPI_SHADER_USER_DATA_GS_25 = 0x1a45
regSPI_SHADER_USER_DATA_GS_26 = 0x1a46
regSPI_SHADER_USER_DATA_GS_27 = 0x1a47
regSPI_SHADER_USER_DATA_GS_28 = 0x1a48
regSPI_SHADER_USER_DATA_GS_29 = 0x1a49
regSPI_SHADER_USER_DATA_GS_3 = 0x1a2f
regSPI_SHADER_USER_DATA_GS_30 = 0x1a4a
regSPI_SHADER_USER_DATA_GS_31 = 0x1a4b
regSPI_SHADER_USER_DATA_GS_4 = 0x1a30
regSPI_SHADER_USER_DATA_GS_5 = 0x1a31
regSPI_SHADER_USER_DATA_GS_6 = 0x1a32
regSPI_SHADER_USER_DATA_GS_7 = 0x1a33
regSPI_SHADER_USER_DATA_GS_8 = 0x1a34
regSPI_SHADER_USER_DATA_GS_9 = 0x1a35
regSPI_SHADER_USER_DATA_HS_0 = 0x1aac
regSPI_SHADER_USER_DATA_HS_1 = 0x1aad
regSPI_SHADER_USER_DATA_HS_10 = 0x1ab6
regSPI_SHADER_USER_DATA_HS_11 = 0x1ab7
regSPI_SHADER_USER_DATA_HS_12 = 0x1ab8
regSPI_SHADER_USER_DATA_HS_13 = 0x1ab9
regSPI_SHADER_USER_DATA_HS_14 = 0x1aba
regSPI_SHADER_USER_DATA_HS_15 = 0x1abb
regSPI_SHADER_USER_DATA_HS_16 = 0x1abc
regSPI_SHADER_USER_DATA_HS_17 = 0x1abd
regSPI_SHADER_USER_DATA_HS_18 = 0x1abe
regSPI_SHADER_USER_DATA_HS_19 = 0x1abf
regSPI_SHADER_USER_DATA_HS_2 = 0x1aae
regSPI_SHADER_USER_DATA_HS_20 = 0x1ac0
regSPI_SHADER_USER_DATA_HS_21 = 0x1ac1
regSPI_SHADER_USER_DATA_HS_22 = 0x1ac2
regSPI_SHADER_USER_DATA_HS_23 = 0x1ac3
regSPI_SHADER_USER_DATA_HS_24 = 0x1ac4
regSPI_SHADER_USER_DATA_HS_25 = 0x1ac5
regSPI_SHADER_USER_DATA_HS_26 = 0x1ac6
regSPI_SHADER_USER_DATA_HS_27 = 0x1ac7
regSPI_SHADER_USER_DATA_HS_28 = 0x1ac8
regSPI_SHADER_USER_DATA_HS_29 = 0x1ac9
regSPI_SHADER_USER_DATA_HS_3 = 0x1aaf
regSPI_SHADER_USER_DATA_HS_30 = 0x1aca
regSPI_SHADER_USER_DATA_HS_31 = 0x1acb
regSPI_SHADER_USER_DATA_HS_4 = 0x1ab0
regSPI_SHADER_USER_DATA_HS_5 = 0x1ab1
regSPI_SHADER_USER_DATA_HS_6 = 0x1ab2
regSPI_SHADER_USER_DATA_HS_7 = 0x1ab3
regSPI_SHADER_USER_DATA_HS_8 = 0x1ab4
regSPI_SHADER_USER_DATA_HS_9 = 0x1ab5
regSPI_SHADER_USER_DATA_PS_0 = 0x19ac
regSPI_SHADER_USER_DATA_PS_1 = 0x19ad
regSPI_SHADER_USER_DATA_PS_10 = 0x19b6
regSPI_SHADER_USER_DATA_PS_11 = 0x19b7
regSPI_SHADER_USER_DATA_PS_12 = 0x19b8
regSPI_SHADER_USER_DATA_PS_13 = 0x19b9
regSPI_SHADER_USER_DATA_PS_14 = 0x19ba
regSPI_SHADER_USER_DATA_PS_15 = 0x19bb
regSPI_SHADER_USER_DATA_PS_16 = 0x19bc
regSPI_SHADER_USER_DATA_PS_17 = 0x19bd
regSPI_SHADER_USER_DATA_PS_18 = 0x19be
regSPI_SHADER_USER_DATA_PS_19 = 0x19bf
regSPI_SHADER_USER_DATA_PS_2 = 0x19ae
regSPI_SHADER_USER_DATA_PS_20 = 0x19c0
regSPI_SHADER_USER_DATA_PS_21 = 0x19c1
regSPI_SHADER_USER_DATA_PS_22 = 0x19c2
regSPI_SHADER_USER_DATA_PS_23 = 0x19c3
regSPI_SHADER_USER_DATA_PS_24 = 0x19c4
regSPI_SHADER_USER_DATA_PS_25 = 0x19c5
regSPI_SHADER_USER_DATA_PS_26 = 0x19c6
regSPI_SHADER_USER_DATA_PS_27 = 0x19c7
regSPI_SHADER_USER_DATA_PS_28 = 0x19c8
regSPI_SHADER_USER_DATA_PS_29 = 0x19c9
regSPI_SHADER_USER_DATA_PS_3 = 0x19af
regSPI_SHADER_USER_DATA_PS_30 = 0x19ca
regSPI_SHADER_USER_DATA_PS_31 = 0x19cb
regSPI_SHADER_USER_DATA_PS_4 = 0x19b0
regSPI_SHADER_USER_DATA_PS_5 = 0x19b1
regSPI_SHADER_USER_DATA_PS_6 = 0x19b2
regSPI_SHADER_USER_DATA_PS_7 = 0x19b3
regSPI_SHADER_USER_DATA_PS_8 = 0x19b4
regSPI_SHADER_USER_DATA_PS_9 = 0x19b5
regSPI_SHADER_Z_FORMAT = 0x1c4
regSPI_SX_EXPORT_BUFFER_SIZES = 0x1279
regSPI_SX_SCOREBOARD_BUFFER_SIZES = 0x127a
regSPI_TMPRING_SIZE = 0x1ba
regSPI_USER_ACCUM_VMID_CNTL = 0x1f71
regSPI_VS_OUT_CONFIG = 0x1b1
regSPI_WAVE_LIMIT_CNTL = 0x2443
regSPI_WCL_PIPE_PERCENT_CS0 = 0x1f69
regSPI_WCL_PIPE_PERCENT_CS1 = 0x1f6a
regSPI_WCL_PIPE_PERCENT_CS2 = 0x1f6b
regSPI_WCL_PIPE_PERCENT_CS3 = 0x1f6c
regSPI_WCL_PIPE_PERCENT_CS4 = 0x1f6d
regSPI_WCL_PIPE_PERCENT_CS5 = 0x1f6e
regSPI_WCL_PIPE_PERCENT_CS6 = 0x1f6f
regSPI_WCL_PIPE_PERCENT_CS7 = 0x1f70
regSPI_WCL_PIPE_PERCENT_GFX = 0x1f67
regSPI_WCL_PIPE_PERCENT_HP3D = 0x1f68
regSPI_WF_LIFETIME_CNTL = 0x124a
regSPI_WF_LIFETIME_LIMIT_0 = 0x124b
regSPI_WF_LIFETIME_LIMIT_1 = 0x124c
regSPI_WF_LIFETIME_LIMIT_2 = 0x124d
regSPI_WF_LIFETIME_LIMIT_3 = 0x124e
regSPI_WF_LIFETIME_LIMIT_4 = 0x124f
regSPI_WF_LIFETIME_LIMIT_5 = 0x1250
regSPI_WF_LIFETIME_STATUS_0 = 0x1255
regSPI_WF_LIFETIME_STATUS_11 = 0x1260
regSPI_WF_LIFETIME_STATUS_13 = 0x1262
regSPI_WF_LIFETIME_STATUS_14 = 0x1263
regSPI_WF_LIFETIME_STATUS_15 = 0x1264
regSPI_WF_LIFETIME_STATUS_16 = 0x1265
regSPI_WF_LIFETIME_STATUS_17 = 0x1266
regSPI_WF_LIFETIME_STATUS_18 = 0x1267
regSPI_WF_LIFETIME_STATUS_19 = 0x1268
regSPI_WF_LIFETIME_STATUS_2 = 0x1257
regSPI_WF_LIFETIME_STATUS_20 = 0x1269
regSPI_WF_LIFETIME_STATUS_21 = 0x126b
regSPI_WF_LIFETIME_STATUS_4 = 0x1259
regSPI_WF_LIFETIME_STATUS_6 = 0x125b
regSPI_WF_LIFETIME_STATUS_7 = 0x125c
regSPI_WF_LIFETIME_STATUS_9 = 0x125e
regSP_CONFIG = 0x10ab
regSQC_CACHES = 0x2348
regSQC_CONFIG = 0x10a1
regSQG_CONFIG = 0x10ba
regSQG_GL1H_STATUS = 0x10b9
regSQG_PERFCOUNTER0_HI = 0x31e5
regSQG_PERFCOUNTER0_LO = 0x31e4
regSQG_PERFCOUNTER0_SELECT = 0x39d0
regSQG_PERFCOUNTER1_HI = 0x31e7
regSQG_PERFCOUNTER1_LO = 0x31e6
regSQG_PERFCOUNTER1_SELECT = 0x39d1
regSQG_PERFCOUNTER2_HI = 0x31e9
regSQG_PERFCOUNTER2_LO = 0x31e8
regSQG_PERFCOUNTER2_SELECT = 0x39d2
regSQG_PERFCOUNTER3_HI = 0x31eb
regSQG_PERFCOUNTER3_LO = 0x31ea
regSQG_PERFCOUNTER3_SELECT = 0x39d3
regSQG_PERFCOUNTER4_HI = 0x31ed
regSQG_PERFCOUNTER4_LO = 0x31ec
regSQG_PERFCOUNTER4_SELECT = 0x39d4
regSQG_PERFCOUNTER5_HI = 0x31ef
regSQG_PERFCOUNTER5_LO = 0x31ee
regSQG_PERFCOUNTER5_SELECT = 0x39d5
regSQG_PERFCOUNTER6_HI = 0x31f1
regSQG_PERFCOUNTER6_LO = 0x31f0
regSQG_PERFCOUNTER6_SELECT = 0x39d6
regSQG_PERFCOUNTER7_HI = 0x31f3
regSQG_PERFCOUNTER7_LO = 0x31f2
regSQG_PERFCOUNTER7_SELECT = 0x39d7
regSQG_PERFCOUNTER_CTRL = 0x39d8
regSQG_PERFCOUNTER_CTRL2 = 0x39da
regSQG_PERF_SAMPLE_FINISH = 0x39db
regSQG_STATUS = 0x10a4
regSQ_ALU_CLK_CTRL = 0x508e
regSQ_ARB_CONFIG = 0x10ac
regSQ_CMD = 0x111b
regSQ_CONFIG = 0x10a0
regSQ_DEBUG = 0x9e5
regSQ_DEBUG_HOST_TRAP_STATUS = 0x10b6
regSQ_DEBUG_STS_GLOBAL = 0x9e1
regSQ_DEBUG_STS_GLOBAL2 = 0x9e2
regSQ_DSM_CNTL = 0x10a6
regSQ_DSM_CNTL2 = 0x10a7
regSQ_FIFO_SIZES = 0x10a5
regSQ_IND_DATA = 0x1119
regSQ_IND_INDEX = 0x1118
regSQ_INTERRUPT_AUTO_MASK = 0x10be
regSQ_INTERRUPT_MSG_CTRL = 0x10bf
regSQ_LDS_CLK_CTRL = 0x5090
regSQ_PERFCOUNTER0_LO = 0x31c0
regSQ_PERFCOUNTER0_SELECT = 0x39c0
regSQ_PERFCOUNTER10_SELECT = 0x39ca
regSQ_PERFCOUNTER11_SELECT = 0x39cb
regSQ_PERFCOUNTER12_SELECT = 0x39cc
regSQ_PERFCOUNTER13_SELECT = 0x39cd
regSQ_PERFCOUNTER14_SELECT = 0x39ce
regSQ_PERFCOUNTER15_SELECT = 0x39cf
regSQ_PERFCOUNTER1_LO = 0x31c2
regSQ_PERFCOUNTER1_SELECT = 0x39c1
regSQ_PERFCOUNTER2_LO = 0x31c4
regSQ_PERFCOUNTER2_SELECT = 0x39c2
regSQ_PERFCOUNTER3_LO = 0x31c6
regSQ_PERFCOUNTER3_SELECT = 0x39c3
regSQ_PERFCOUNTER4_LO = 0x31c8
regSQ_PERFCOUNTER4_SELECT = 0x39c4
regSQ_PERFCOUNTER5_LO = 0x31ca
regSQ_PERFCOUNTER5_SELECT = 0x39c5
regSQ_PERFCOUNTER6_LO = 0x31cc
regSQ_PERFCOUNTER6_SELECT = 0x39c6
regSQ_PERFCOUNTER7_LO = 0x31ce
regSQ_PERFCOUNTER7_SELECT = 0x39c7
regSQ_PERFCOUNTER8_SELECT = 0x39c8
regSQ_PERFCOUNTER9_SELECT = 0x39c9
regSQ_PERFCOUNTER_CTRL = 0x39e0
regSQ_PERFCOUNTER_CTRL2 = 0x39e2
regSQ_PERF_SNAPSHOT_CTRL = 0x10bb
regSQ_RANDOM_WAVE_PRI = 0x10a3
regSQ_RUNTIME_CONFIG = 0x9e0
regSQ_SHADER_TBA_HI = 0x9e7
regSQ_SHADER_TBA_LO = 0x9e6
regSQ_SHADER_TMA_HI = 0x9e9
regSQ_SHADER_TMA_LO = 0x9e8
regSQ_TEX_CLK_CTRL = 0x508f
regSQ_THREAD_TRACE_BUF0_BASE = 0x39e8
regSQ_THREAD_TRACE_BUF0_SIZE = 0x39e9
regSQ_THREAD_TRACE_BUF1_BASE = 0x39ea
regSQ_THREAD_TRACE_BUF1_SIZE = 0x39eb
regSQ_THREAD_TRACE_CTRL = 0x39ec
regSQ_THREAD_TRACE_DROPPED_CNTR = 0x39fa
regSQ_THREAD_TRACE_GFX_DRAW_CNTR = 0x39f6
regSQ_THREAD_TRACE_GFX_MARKER_CNTR = 0x39f7
regSQ_THREAD_TRACE_HP3D_DRAW_CNTR = 0x39f8
regSQ_THREAD_TRACE_HP3D_MARKER_CNTR = 0x39f9
regSQ_THREAD_TRACE_MASK = 0x39ed
regSQ_THREAD_TRACE_STATUS = 0x39f4
regSQ_THREAD_TRACE_STATUS2 = 0x39f5
regSQ_THREAD_TRACE_TOKEN_MASK = 0x39ee
regSQ_THREAD_TRACE_USERDATA_0 = 0x2340
regSQ_THREAD_TRACE_USERDATA_1 = 0x2341
regSQ_THREAD_TRACE_USERDATA_2 = 0x2342
regSQ_THREAD_TRACE_USERDATA_3 = 0x2343
regSQ_THREAD_TRACE_USERDATA_4 = 0x2344
regSQ_THREAD_TRACE_USERDATA_5 = 0x2345
regSQ_THREAD_TRACE_USERDATA_6 = 0x2346
regSQ_THREAD_TRACE_USERDATA_7 = 0x2347
regSQ_THREAD_TRACE_WPTR = 0x39ef
regSQ_WATCH0_ADDR_H = 0x10d0
regSQ_WATCH0_ADDR_L = 0x10d1
regSQ_WATCH0_CNTL = 0x10d2
regSQ_WATCH1_ADDR_H = 0x10d3
regSQ_WATCH1_ADDR_L = 0x10d4
regSQ_WATCH1_CNTL = 0x10d5
regSQ_WATCH2_ADDR_H = 0x10d6
regSQ_WATCH2_ADDR_L = 0x10d7
regSQ_WATCH2_CNTL = 0x10d8
regSQ_WATCH3_ADDR_H = 0x10d9
regSQ_WATCH3_ADDR_L = 0x10da
regSQ_WATCH3_CNTL = 0x10db
regSX_BLEND_OPT_CONTROL = 0x1d7
regSX_BLEND_OPT_EPSILON = 0x1d6
regSX_DEBUG_1 = 0x11b8
regSX_MRT0_BLEND_OPT = 0x1d8
regSX_MRT1_BLEND_OPT = 0x1d9
regSX_MRT2_BLEND_OPT = 0x1da
regSX_MRT3_BLEND_OPT = 0x1db
regSX_MRT4_BLEND_OPT = 0x1dc
regSX_MRT5_BLEND_OPT = 0x1dd
regSX_MRT6_BLEND_OPT = 0x1de
regSX_MRT7_BLEND_OPT = 0x1df
regSX_PERFCOUNTER0_HI = 0x3241
regSX_PERFCOUNTER0_LO = 0x3240
regSX_PERFCOUNTER0_SELECT = 0x3a40
regSX_PERFCOUNTER0_SELECT1 = 0x3a44
regSX_PERFCOUNTER1_HI = 0x3243
regSX_PERFCOUNTER1_LO = 0x3242
regSX_PERFCOUNTER1_SELECT = 0x3a41
regSX_PERFCOUNTER1_SELECT1 = 0x3a45
regSX_PERFCOUNTER2_HI = 0x3245
regSX_PERFCOUNTER2_LO = 0x3244
regSX_PERFCOUNTER2_SELECT = 0x3a42
regSX_PERFCOUNTER3_HI = 0x3247
regSX_PERFCOUNTER3_LO = 0x3246
regSX_PERFCOUNTER3_SELECT = 0x3a43
regSX_PS_DOWNCONVERT = 0x1d5
regSX_PS_DOWNCONVERT_CONTROL = 0x1d4
regTA_BC_BASE_ADDR = 0x20
regTA_BC_BASE_ADDR_HI = 0x21
regTA_CGTT_CTRL = 0x509d
regTA_CNTL = 0x12e1
regTA_CNTL2 = 0x12e5
regTA_CNTL_AUX = 0x12e2
regTA_CS_BC_BASE_ADDR = 0x2380
regTA_CS_BC_BASE_ADDR_HI = 0x2381
regTA_PERFCOUNTER0_HI = 0x32c1
regTA_PERFCOUNTER0_LO = 0x32c0
regTA_PERFCOUNTER0_SELECT = 0x3ac0
regTA_PERFCOUNTER0_SELECT1 = 0x3ac1
regTA_PERFCOUNTER1_HI = 0x32c3
regTA_PERFCOUNTER1_LO = 0x32c2
regTA_PERFCOUNTER1_SELECT = 0x3ac2
regTA_SCRATCH = 0x1304
regTA_STATUS = 0x12e8
regTCP_CNTL = 0x19a2
regTCP_CNTL2 = 0x19a3
regTCP_DEBUG_DATA = 0x19a6
regTCP_DEBUG_INDEX = 0x19a5
regTCP_INVALIDATE = 0x19a0
regTCP_PERFCOUNTER0_HI = 0x3341
regTCP_PERFCOUNTER0_LO = 0x3340
regTCP_PERFCOUNTER0_SELECT = 0x3b40
regTCP_PERFCOUNTER0_SELECT1 = 0x3b41
regTCP_PERFCOUNTER1_HI = 0x3343
regTCP_PERFCOUNTER1_LO = 0x3342
regTCP_PERFCOUNTER1_SELECT = 0x3b42
regTCP_PERFCOUNTER1_SELECT1 = 0x3b43
regTCP_PERFCOUNTER2_HI = 0x3345
regTCP_PERFCOUNTER2_LO = 0x3344
regTCP_PERFCOUNTER2_SELECT = 0x3b44
regTCP_PERFCOUNTER3_HI = 0x3347
regTCP_PERFCOUNTER3_LO = 0x3346
regTCP_PERFCOUNTER3_SELECT = 0x3b45
regTCP_PERFCOUNTER_FILTER = 0x3348
regTCP_PERFCOUNTER_FILTER2 = 0x3349
regTCP_PERFCOUNTER_FILTER_EN = 0x334a
regTCP_STATUS = 0x19a1
regTCP_WATCH0_ADDR_H = 0x2048
regTCP_WATCH0_ADDR_L = 0x2049
regTCP_WATCH0_CNTL = 0x204a
regTCP_WATCH1_ADDR_H = 0x204b
regTCP_WATCH1_ADDR_L = 0x204c
regTCP_WATCH1_CNTL = 0x204d
regTCP_WATCH2_ADDR_H = 0x204e
regTCP_WATCH2_ADDR_L = 0x204f
regTCP_WATCH2_CNTL = 0x2050
regTCP_WATCH3_ADDR_H = 0x2051
regTCP_WATCH3_ADDR_L = 0x2052
regTCP_WATCH3_CNTL = 0x2053
regTD_DSM_CNTL = 0x12cf
regTD_DSM_CNTL2 = 0x12d0
regTD_PERFCOUNTER0_HI = 0x3301
regTD_PERFCOUNTER0_LO = 0x3300
regTD_PERFCOUNTER0_SELECT = 0x3b00
regTD_PERFCOUNTER0_SELECT1 = 0x3b01
regTD_PERFCOUNTER1_HI = 0x3303
regTD_PERFCOUNTER1_LO = 0x3302
regTD_PERFCOUNTER1_SELECT = 0x3b02
regTD_SCRATCH = 0x12d3
regTD_STATUS = 0x12c6
regUCONFIG_RESERVED_REG0 = 0x20a2
regUCONFIG_RESERVED_REG1 = 0x20a3
regUTCL1_ALOG = 0x158f
regUTCL1_CTRL_0 = 0x1980
regUTCL1_CTRL_1 = 0x158c
regUTCL1_CTRL_2 = 0x1985
regUTCL1_FIFO_SIZING = 0x1986
regUTCL1_PERFCOUNTER0_HI = 0x35a1
regUTCL1_PERFCOUNTER0_LO = 0x35a0
regUTCL1_PERFCOUNTER0_SELECT = 0x3da0
regUTCL1_PERFCOUNTER1_HI = 0x35a3
regUTCL1_PERFCOUNTER1_LO = 0x35a2
regUTCL1_PERFCOUNTER1_SELECT = 0x3da1
regUTCL1_PERFCOUNTER2_HI = 0x35a5
regUTCL1_PERFCOUNTER2_LO = 0x35a4
regUTCL1_PERFCOUNTER2_SELECT = 0x3da2
regUTCL1_PERFCOUNTER3_HI = 0x35a7
regUTCL1_PERFCOUNTER3_LO = 0x35a6
regUTCL1_PERFCOUNTER3_SELECT = 0x3da3
regUTCL1_STATUS = 0x1594
regUTCL1_UTCL0_INVREQ_DISABLE = 0x1984
regVGT_DMA_BASE = 0x1fa
regVGT_DMA_BASE_HI = 0x1f9
regVGT_DMA_DATA_FIFO_DEPTH = 0xfcd
regVGT_DMA_INDEX_TYPE = 0x29f
regVGT_DMA_MAX_SIZE = 0x29e
regVGT_DMA_NUM_INSTANCES = 0x2a2
regVGT_DMA_REQ_FIFO_DEPTH = 0xfce
regVGT_DMA_SIZE = 0x29d
regVGT_DRAW_INITIATOR = 0x1fc
regVGT_DRAW_INIT_FIFO_DEPTH = 0xfcf
regVGT_DRAW_PAYLOAD_CNTL = 0x2a6
regVGT_ENHANCE = 0x294
regVGT_ESGS_RING_ITEMSIZE = 0x2ab
regVGT_EVENT_ADDRESS_REG = 0x1fe
regVGT_EVENT_INITIATOR = 0x2a4
regVGT_GS_INSTANCE_CNT = 0x2e4
regVGT_GS_MAX_VERT_OUT = 0x2ce
regVGT_GS_MAX_WAVE_ID = 0x1009
regVGT_GS_OUT_PRIM_TYPE = 0x2266
regVGT_HOS_MAX_TESS_LEVEL = 0x286
regVGT_HOS_MIN_TESS_LEVEL = 0x287
regVGT_HS_OFFCHIP_PARAM = 0x224f
regVGT_INDEX_TYPE = 0x2243
regVGT_INSTANCE_BASE_ID = 0x225a
regVGT_LS_HS_CONFIG = 0x2d6
regVGT_MC_LAT_CNTL = 0xfd6
regVGT_MULTI_PRIM_IB_RESET_INDX = 0x103
regVGT_NUM_INDICES = 0x224c
regVGT_NUM_INSTANCES = 0x224d
regVGT_PRIMITIVEID_EN = 0x2a1
regVGT_PRIMITIVEID_RESET = 0x2a3
regVGT_PRIMITIVE_TYPE = 0x2242
regVGT_REUSE_OFF = 0x2ad
regVGT_SHADER_STAGES_EN = 0x2d5
regVGT_STRMOUT_DRAW_OPAQUE_BUFFER_FILLED_SIZE = 0x2cb
regVGT_STRMOUT_DRAW_OPAQUE_OFFSET = 0x2ca
regVGT_STRMOUT_DRAW_OPAQUE_VERTEX_STRIDE = 0x2cc
regVGT_SYS_CONFIG = 0x1003
regVGT_TESS_DISTRIBUTION = 0x2d4
regVGT_TF_MEMORY_BASE = 0x2250
regVGT_TF_MEMORY_BASE_HI = 0x2261
regVGT_TF_PARAM = 0x2db
regVGT_TF_RING_SIZE = 0x224e
regVIOLATION_DATA_ASYNC_VF_PROG = 0xdf1
regWD_CNTL_STATUS = 0xfdf
regWD_ENHANCE = 0x2a0
regWD_QOS = 0xfe2
regWD_UTCL1_CNTL = 0xfe3
regWD_UTCL1_STATUS = 0xfe4

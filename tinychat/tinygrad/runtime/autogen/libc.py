# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util, os


c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['libc'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['libc'] = ctypes.CDLL(ctypes.util.find_library('c')) #  ctypes.CDLL('libc')
def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))



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





off_t = ctypes.c_int64
mode_t = ctypes.c_uint32
size_t = ctypes.c_uint64
__off_t = ctypes.c_int64
try:
    mmap = _libraries['libc'].mmap
    mmap.restype = ctypes.POINTER(None)
    mmap.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, __off_t]
except AttributeError:
    pass
try:
    munmap = _libraries['libc'].munmap
    munmap.restype = ctypes.c_int32
    munmap.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    mprotect = _libraries['libc'].mprotect
    mprotect.restype = ctypes.c_int32
    mprotect.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    msync = _libraries['libc'].msync
    msync.restype = ctypes.c_int32
    msync.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    madvise = _libraries['libc'].madvise
    madvise.restype = ctypes.c_int32
    madvise.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    posix_madvise = _libraries['libc'].posix_madvise
    posix_madvise.restype = ctypes.c_int32
    posix_madvise.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    mlock = _libraries['libc'].mlock
    mlock.restype = ctypes.c_int32
    mlock.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    munlock = _libraries['libc'].munlock
    munlock.restype = ctypes.c_int32
    munlock.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    mlockall = _libraries['libc'].mlockall
    mlockall.restype = ctypes.c_int32
    mlockall.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    munlockall = _libraries['libc'].munlockall
    munlockall.restype = ctypes.c_int32
    munlockall.argtypes = []
except AttributeError:
    pass
try:
    mincore = _libraries['libc'].mincore
    mincore.restype = ctypes.c_int32
    mincore.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError:
    pass
try:
    shm_open = _libraries['libc'].shm_open
    shm_open.restype = ctypes.c_int32
    shm_open.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, mode_t]
except AttributeError:
    pass
try:
    shm_unlink = _libraries['libc'].shm_unlink
    shm_unlink.restype = ctypes.c_int32
    shm_unlink.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
Elf32_Half = ctypes.c_uint16
Elf64_Half = ctypes.c_uint16
Elf32_Word = ctypes.c_uint32
Elf32_Sword = ctypes.c_int32
Elf64_Word = ctypes.c_uint32
Elf64_Sword = ctypes.c_int32
Elf32_Xword = ctypes.c_uint64
Elf32_Sxword = ctypes.c_int64
Elf64_Xword = ctypes.c_uint64
Elf64_Sxword = ctypes.c_int64
Elf32_Addr = ctypes.c_uint32
Elf64_Addr = ctypes.c_uint64
Elf32_Off = ctypes.c_uint32
Elf64_Off = ctypes.c_uint64
Elf32_Section = ctypes.c_uint16
Elf64_Section = ctypes.c_uint16
Elf32_Versym = ctypes.c_uint16
Elf64_Versym = ctypes.c_uint16
class struct_c__SA_Elf32_Ehdr(Structure):
    pass

struct_c__SA_Elf32_Ehdr._pack_ = 1 # source:False
struct_c__SA_Elf32_Ehdr._fields_ = [
    ('e_ident', ctypes.c_ubyte * 16),
    ('e_type', ctypes.c_uint16),
    ('e_machine', ctypes.c_uint16),
    ('e_version', ctypes.c_uint32),
    ('e_entry', ctypes.c_uint32),
    ('e_phoff', ctypes.c_uint32),
    ('e_shoff', ctypes.c_uint32),
    ('e_flags', ctypes.c_uint32),
    ('e_ehsize', ctypes.c_uint16),
    ('e_phentsize', ctypes.c_uint16),
    ('e_phnum', ctypes.c_uint16),
    ('e_shentsize', ctypes.c_uint16),
    ('e_shnum', ctypes.c_uint16),
    ('e_shstrndx', ctypes.c_uint16),
]

Elf32_Ehdr = struct_c__SA_Elf32_Ehdr
class struct_c__SA_Elf64_Ehdr(Structure):
    pass

struct_c__SA_Elf64_Ehdr._pack_ = 1 # source:False
struct_c__SA_Elf64_Ehdr._fields_ = [
    ('e_ident', ctypes.c_ubyte * 16),
    ('e_type', ctypes.c_uint16),
    ('e_machine', ctypes.c_uint16),
    ('e_version', ctypes.c_uint32),
    ('e_entry', ctypes.c_uint64),
    ('e_phoff', ctypes.c_uint64),
    ('e_shoff', ctypes.c_uint64),
    ('e_flags', ctypes.c_uint32),
    ('e_ehsize', ctypes.c_uint16),
    ('e_phentsize', ctypes.c_uint16),
    ('e_phnum', ctypes.c_uint16),
    ('e_shentsize', ctypes.c_uint16),
    ('e_shnum', ctypes.c_uint16),
    ('e_shstrndx', ctypes.c_uint16),
]

Elf64_Ehdr = struct_c__SA_Elf64_Ehdr
class struct_c__SA_Elf32_Shdr(Structure):
    pass

struct_c__SA_Elf32_Shdr._pack_ = 1 # source:False
struct_c__SA_Elf32_Shdr._fields_ = [
    ('sh_name', ctypes.c_uint32),
    ('sh_type', ctypes.c_uint32),
    ('sh_flags', ctypes.c_uint32),
    ('sh_addr', ctypes.c_uint32),
    ('sh_offset', ctypes.c_uint32),
    ('sh_size', ctypes.c_uint32),
    ('sh_link', ctypes.c_uint32),
    ('sh_info', ctypes.c_uint32),
    ('sh_addralign', ctypes.c_uint32),
    ('sh_entsize', ctypes.c_uint32),
]

Elf32_Shdr = struct_c__SA_Elf32_Shdr
class struct_c__SA_Elf64_Shdr(Structure):
    pass

struct_c__SA_Elf64_Shdr._pack_ = 1 # source:False
struct_c__SA_Elf64_Shdr._fields_ = [
    ('sh_name', ctypes.c_uint32),
    ('sh_type', ctypes.c_uint32),
    ('sh_flags', ctypes.c_uint64),
    ('sh_addr', ctypes.c_uint64),
    ('sh_offset', ctypes.c_uint64),
    ('sh_size', ctypes.c_uint64),
    ('sh_link', ctypes.c_uint32),
    ('sh_info', ctypes.c_uint32),
    ('sh_addralign', ctypes.c_uint64),
    ('sh_entsize', ctypes.c_uint64),
]

Elf64_Shdr = struct_c__SA_Elf64_Shdr
class struct_c__SA_Elf32_Chdr(Structure):
    pass

struct_c__SA_Elf32_Chdr._pack_ = 1 # source:False
struct_c__SA_Elf32_Chdr._fields_ = [
    ('ch_type', ctypes.c_uint32),
    ('ch_size', ctypes.c_uint32),
    ('ch_addralign', ctypes.c_uint32),
]

Elf32_Chdr = struct_c__SA_Elf32_Chdr
class struct_c__SA_Elf64_Chdr(Structure):
    pass

struct_c__SA_Elf64_Chdr._pack_ = 1 # source:False
struct_c__SA_Elf64_Chdr._fields_ = [
    ('ch_type', ctypes.c_uint32),
    ('ch_reserved', ctypes.c_uint32),
    ('ch_size', ctypes.c_uint64),
    ('ch_addralign', ctypes.c_uint64),
]

Elf64_Chdr = struct_c__SA_Elf64_Chdr
class struct_c__SA_Elf32_Sym(Structure):
    pass

struct_c__SA_Elf32_Sym._pack_ = 1 # source:False
struct_c__SA_Elf32_Sym._fields_ = [
    ('st_name', ctypes.c_uint32),
    ('st_value', ctypes.c_uint32),
    ('st_size', ctypes.c_uint32),
    ('st_info', ctypes.c_ubyte),
    ('st_other', ctypes.c_ubyte),
    ('st_shndx', ctypes.c_uint16),
]

Elf32_Sym = struct_c__SA_Elf32_Sym
class struct_c__SA_Elf64_Sym(Structure):
    pass

struct_c__SA_Elf64_Sym._pack_ = 1 # source:False
struct_c__SA_Elf64_Sym._fields_ = [
    ('st_name', ctypes.c_uint32),
    ('st_info', ctypes.c_ubyte),
    ('st_other', ctypes.c_ubyte),
    ('st_shndx', ctypes.c_uint16),
    ('st_value', ctypes.c_uint64),
    ('st_size', ctypes.c_uint64),
]

Elf64_Sym = struct_c__SA_Elf64_Sym
class struct_c__SA_Elf32_Syminfo(Structure):
    pass

struct_c__SA_Elf32_Syminfo._pack_ = 1 # source:False
struct_c__SA_Elf32_Syminfo._fields_ = [
    ('si_boundto', ctypes.c_uint16),
    ('si_flags', ctypes.c_uint16),
]

Elf32_Syminfo = struct_c__SA_Elf32_Syminfo
class struct_c__SA_Elf64_Syminfo(Structure):
    pass

struct_c__SA_Elf64_Syminfo._pack_ = 1 # source:False
struct_c__SA_Elf64_Syminfo._fields_ = [
    ('si_boundto', ctypes.c_uint16),
    ('si_flags', ctypes.c_uint16),
]

Elf64_Syminfo = struct_c__SA_Elf64_Syminfo
class struct_c__SA_Elf32_Rel(Structure):
    pass

struct_c__SA_Elf32_Rel._pack_ = 1 # source:False
struct_c__SA_Elf32_Rel._fields_ = [
    ('r_offset', ctypes.c_uint32),
    ('r_info', ctypes.c_uint32),
]

Elf32_Rel = struct_c__SA_Elf32_Rel
class struct_c__SA_Elf64_Rel(Structure):
    pass

struct_c__SA_Elf64_Rel._pack_ = 1 # source:False
struct_c__SA_Elf64_Rel._fields_ = [
    ('r_offset', ctypes.c_uint64),
    ('r_info', ctypes.c_uint64),
]

Elf64_Rel = struct_c__SA_Elf64_Rel
class struct_c__SA_Elf32_Rela(Structure):
    pass

struct_c__SA_Elf32_Rela._pack_ = 1 # source:False
struct_c__SA_Elf32_Rela._fields_ = [
    ('r_offset', ctypes.c_uint32),
    ('r_info', ctypes.c_uint32),
    ('r_addend', ctypes.c_int32),
]

Elf32_Rela = struct_c__SA_Elf32_Rela
class struct_c__SA_Elf64_Rela(Structure):
    pass

struct_c__SA_Elf64_Rela._pack_ = 1 # source:False
struct_c__SA_Elf64_Rela._fields_ = [
    ('r_offset', ctypes.c_uint64),
    ('r_info', ctypes.c_uint64),
    ('r_addend', ctypes.c_int64),
]

Elf64_Rela = struct_c__SA_Elf64_Rela
class struct_c__SA_Elf32_Phdr(Structure):
    pass

struct_c__SA_Elf32_Phdr._pack_ = 1 # source:False
struct_c__SA_Elf32_Phdr._fields_ = [
    ('p_type', ctypes.c_uint32),
    ('p_offset', ctypes.c_uint32),
    ('p_vaddr', ctypes.c_uint32),
    ('p_paddr', ctypes.c_uint32),
    ('p_filesz', ctypes.c_uint32),
    ('p_memsz', ctypes.c_uint32),
    ('p_flags', ctypes.c_uint32),
    ('p_align', ctypes.c_uint32),
]

Elf32_Phdr = struct_c__SA_Elf32_Phdr
class struct_c__SA_Elf64_Phdr(Structure):
    pass

struct_c__SA_Elf64_Phdr._pack_ = 1 # source:False
struct_c__SA_Elf64_Phdr._fields_ = [
    ('p_type', ctypes.c_uint32),
    ('p_flags', ctypes.c_uint32),
    ('p_offset', ctypes.c_uint64),
    ('p_vaddr', ctypes.c_uint64),
    ('p_paddr', ctypes.c_uint64),
    ('p_filesz', ctypes.c_uint64),
    ('p_memsz', ctypes.c_uint64),
    ('p_align', ctypes.c_uint64),
]

Elf64_Phdr = struct_c__SA_Elf64_Phdr
class struct_c__SA_Elf32_Dyn(Structure):
    pass

class union_c__SA_Elf32_Dyn_d_un(Union):
    pass

union_c__SA_Elf32_Dyn_d_un._pack_ = 1 # source:False
union_c__SA_Elf32_Dyn_d_un._fields_ = [
    ('d_val', ctypes.c_uint32),
    ('d_ptr', ctypes.c_uint32),
]

struct_c__SA_Elf32_Dyn._pack_ = 1 # source:False
struct_c__SA_Elf32_Dyn._fields_ = [
    ('d_tag', ctypes.c_int32),
    ('d_un', union_c__SA_Elf32_Dyn_d_un),
]

Elf32_Dyn = struct_c__SA_Elf32_Dyn
class struct_c__SA_Elf64_Dyn(Structure):
    pass

class union_c__SA_Elf64_Dyn_d_un(Union):
    pass

union_c__SA_Elf64_Dyn_d_un._pack_ = 1 # source:False
union_c__SA_Elf64_Dyn_d_un._fields_ = [
    ('d_val', ctypes.c_uint64),
    ('d_ptr', ctypes.c_uint64),
]

struct_c__SA_Elf64_Dyn._pack_ = 1 # source:False
struct_c__SA_Elf64_Dyn._fields_ = [
    ('d_tag', ctypes.c_int64),
    ('d_un', union_c__SA_Elf64_Dyn_d_un),
]

Elf64_Dyn = struct_c__SA_Elf64_Dyn
class struct_c__SA_Elf32_Verdef(Structure):
    pass

struct_c__SA_Elf32_Verdef._pack_ = 1 # source:False
struct_c__SA_Elf32_Verdef._fields_ = [
    ('vd_version', ctypes.c_uint16),
    ('vd_flags', ctypes.c_uint16),
    ('vd_ndx', ctypes.c_uint16),
    ('vd_cnt', ctypes.c_uint16),
    ('vd_hash', ctypes.c_uint32),
    ('vd_aux', ctypes.c_uint32),
    ('vd_next', ctypes.c_uint32),
]

Elf32_Verdef = struct_c__SA_Elf32_Verdef
class struct_c__SA_Elf64_Verdef(Structure):
    pass

struct_c__SA_Elf64_Verdef._pack_ = 1 # source:False
struct_c__SA_Elf64_Verdef._fields_ = [
    ('vd_version', ctypes.c_uint16),
    ('vd_flags', ctypes.c_uint16),
    ('vd_ndx', ctypes.c_uint16),
    ('vd_cnt', ctypes.c_uint16),
    ('vd_hash', ctypes.c_uint32),
    ('vd_aux', ctypes.c_uint32),
    ('vd_next', ctypes.c_uint32),
]

Elf64_Verdef = struct_c__SA_Elf64_Verdef
class struct_c__SA_Elf32_Verdaux(Structure):
    pass

struct_c__SA_Elf32_Verdaux._pack_ = 1 # source:False
struct_c__SA_Elf32_Verdaux._fields_ = [
    ('vda_name', ctypes.c_uint32),
    ('vda_next', ctypes.c_uint32),
]

Elf32_Verdaux = struct_c__SA_Elf32_Verdaux
class struct_c__SA_Elf64_Verdaux(Structure):
    pass

struct_c__SA_Elf64_Verdaux._pack_ = 1 # source:False
struct_c__SA_Elf64_Verdaux._fields_ = [
    ('vda_name', ctypes.c_uint32),
    ('vda_next', ctypes.c_uint32),
]

Elf64_Verdaux = struct_c__SA_Elf64_Verdaux
class struct_c__SA_Elf32_Verneed(Structure):
    pass

struct_c__SA_Elf32_Verneed._pack_ = 1 # source:False
struct_c__SA_Elf32_Verneed._fields_ = [
    ('vn_version', ctypes.c_uint16),
    ('vn_cnt', ctypes.c_uint16),
    ('vn_file', ctypes.c_uint32),
    ('vn_aux', ctypes.c_uint32),
    ('vn_next', ctypes.c_uint32),
]

Elf32_Verneed = struct_c__SA_Elf32_Verneed
class struct_c__SA_Elf64_Verneed(Structure):
    pass

struct_c__SA_Elf64_Verneed._pack_ = 1 # source:False
struct_c__SA_Elf64_Verneed._fields_ = [
    ('vn_version', ctypes.c_uint16),
    ('vn_cnt', ctypes.c_uint16),
    ('vn_file', ctypes.c_uint32),
    ('vn_aux', ctypes.c_uint32),
    ('vn_next', ctypes.c_uint32),
]

Elf64_Verneed = struct_c__SA_Elf64_Verneed
class struct_c__SA_Elf32_Vernaux(Structure):
    pass

struct_c__SA_Elf32_Vernaux._pack_ = 1 # source:False
struct_c__SA_Elf32_Vernaux._fields_ = [
    ('vna_hash', ctypes.c_uint32),
    ('vna_flags', ctypes.c_uint16),
    ('vna_other', ctypes.c_uint16),
    ('vna_name', ctypes.c_uint32),
    ('vna_next', ctypes.c_uint32),
]

Elf32_Vernaux = struct_c__SA_Elf32_Vernaux
class struct_c__SA_Elf64_Vernaux(Structure):
    pass

struct_c__SA_Elf64_Vernaux._pack_ = 1 # source:False
struct_c__SA_Elf64_Vernaux._fields_ = [
    ('vna_hash', ctypes.c_uint32),
    ('vna_flags', ctypes.c_uint16),
    ('vna_other', ctypes.c_uint16),
    ('vna_name', ctypes.c_uint32),
    ('vna_next', ctypes.c_uint32),
]

Elf64_Vernaux = struct_c__SA_Elf64_Vernaux
class struct_c__SA_Elf32_auxv_t(Structure):
    pass

class union_c__SA_Elf32_auxv_t_a_un(Union):
    pass

union_c__SA_Elf32_auxv_t_a_un._pack_ = 1 # source:False
union_c__SA_Elf32_auxv_t_a_un._fields_ = [
    ('a_val', ctypes.c_uint32),
]

struct_c__SA_Elf32_auxv_t._pack_ = 1 # source:False
struct_c__SA_Elf32_auxv_t._fields_ = [
    ('a_type', ctypes.c_uint32),
    ('a_un', union_c__SA_Elf32_auxv_t_a_un),
]

Elf32_auxv_t = struct_c__SA_Elf32_auxv_t
class struct_c__SA_Elf64_auxv_t(Structure):
    pass

class union_c__SA_Elf64_auxv_t_a_un(Union):
    pass

union_c__SA_Elf64_auxv_t_a_un._pack_ = 1 # source:False
union_c__SA_Elf64_auxv_t_a_un._fields_ = [
    ('a_val', ctypes.c_uint64),
]

struct_c__SA_Elf64_auxv_t._pack_ = 1 # source:False
struct_c__SA_Elf64_auxv_t._fields_ = [
    ('a_type', ctypes.c_uint64),
    ('a_un', union_c__SA_Elf64_auxv_t_a_un),
]

Elf64_auxv_t = struct_c__SA_Elf64_auxv_t
class struct_c__SA_Elf32_Nhdr(Structure):
    pass

struct_c__SA_Elf32_Nhdr._pack_ = 1 # source:False
struct_c__SA_Elf32_Nhdr._fields_ = [
    ('n_namesz', ctypes.c_uint32),
    ('n_descsz', ctypes.c_uint32),
    ('n_type', ctypes.c_uint32),
]

Elf32_Nhdr = struct_c__SA_Elf32_Nhdr
class struct_c__SA_Elf64_Nhdr(Structure):
    pass

struct_c__SA_Elf64_Nhdr._pack_ = 1 # source:False
struct_c__SA_Elf64_Nhdr._fields_ = [
    ('n_namesz', ctypes.c_uint32),
    ('n_descsz', ctypes.c_uint32),
    ('n_type', ctypes.c_uint32),
]

Elf64_Nhdr = struct_c__SA_Elf64_Nhdr
class struct_c__SA_Elf32_Move(Structure):
    pass

struct_c__SA_Elf32_Move._pack_ = 1 # source:False
struct_c__SA_Elf32_Move._fields_ = [
    ('m_value', ctypes.c_uint64),
    ('m_info', ctypes.c_uint32),
    ('m_poffset', ctypes.c_uint32),
    ('m_repeat', ctypes.c_uint16),
    ('m_stride', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

Elf32_Move = struct_c__SA_Elf32_Move
class struct_c__SA_Elf64_Move(Structure):
    pass

struct_c__SA_Elf64_Move._pack_ = 1 # source:False
struct_c__SA_Elf64_Move._fields_ = [
    ('m_value', ctypes.c_uint64),
    ('m_info', ctypes.c_uint64),
    ('m_poffset', ctypes.c_uint64),
    ('m_repeat', ctypes.c_uint16),
    ('m_stride', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

Elf64_Move = struct_c__SA_Elf64_Move
class union_c__UA_Elf32_gptab(Union):
    pass

class struct_c__UA_Elf32_gptab_gt_header(Structure):
    pass

struct_c__UA_Elf32_gptab_gt_header._pack_ = 1 # source:False
struct_c__UA_Elf32_gptab_gt_header._fields_ = [
    ('gt_current_g_value', ctypes.c_uint32),
    ('gt_unused', ctypes.c_uint32),
]

class struct_c__UA_Elf32_gptab_gt_entry(Structure):
    pass

struct_c__UA_Elf32_gptab_gt_entry._pack_ = 1 # source:False
struct_c__UA_Elf32_gptab_gt_entry._fields_ = [
    ('gt_g_value', ctypes.c_uint32),
    ('gt_bytes', ctypes.c_uint32),
]

union_c__UA_Elf32_gptab._pack_ = 1 # source:False
union_c__UA_Elf32_gptab._fields_ = [
    ('gt_header', struct_c__UA_Elf32_gptab_gt_header),
    ('gt_entry', struct_c__UA_Elf32_gptab_gt_entry),
]

Elf32_gptab = union_c__UA_Elf32_gptab
class struct_c__SA_Elf32_RegInfo(Structure):
    pass

struct_c__SA_Elf32_RegInfo._pack_ = 1 # source:False
struct_c__SA_Elf32_RegInfo._fields_ = [
    ('ri_gprmask', ctypes.c_uint32),
    ('ri_cprmask', ctypes.c_uint32 * 4),
    ('ri_gp_value', ctypes.c_int32),
]

Elf32_RegInfo = struct_c__SA_Elf32_RegInfo
class struct_c__SA_Elf_Options(Structure):
    pass

struct_c__SA_Elf_Options._pack_ = 1 # source:False
struct_c__SA_Elf_Options._fields_ = [
    ('kind', ctypes.c_ubyte),
    ('size', ctypes.c_ubyte),
    ('section', ctypes.c_uint16),
    ('info', ctypes.c_uint32),
]

Elf_Options = struct_c__SA_Elf_Options
class struct_c__SA_Elf_Options_Hw(Structure):
    pass

struct_c__SA_Elf_Options_Hw._pack_ = 1 # source:False
struct_c__SA_Elf_Options_Hw._fields_ = [
    ('hwp_flags1', ctypes.c_uint32),
    ('hwp_flags2', ctypes.c_uint32),
]

Elf_Options_Hw = struct_c__SA_Elf_Options_Hw
class struct_c__SA_Elf32_Lib(Structure):
    pass

struct_c__SA_Elf32_Lib._pack_ = 1 # source:False
struct_c__SA_Elf32_Lib._fields_ = [
    ('l_name', ctypes.c_uint32),
    ('l_time_stamp', ctypes.c_uint32),
    ('l_checksum', ctypes.c_uint32),
    ('l_version', ctypes.c_uint32),
    ('l_flags', ctypes.c_uint32),
]

Elf32_Lib = struct_c__SA_Elf32_Lib
class struct_c__SA_Elf64_Lib(Structure):
    pass

struct_c__SA_Elf64_Lib._pack_ = 1 # source:False
struct_c__SA_Elf64_Lib._fields_ = [
    ('l_name', ctypes.c_uint32),
    ('l_time_stamp', ctypes.c_uint32),
    ('l_checksum', ctypes.c_uint32),
    ('l_version', ctypes.c_uint32),
    ('l_flags', ctypes.c_uint32),
]

Elf64_Lib = struct_c__SA_Elf64_Lib
Elf32_Conflict = ctypes.c_uint32
class struct_c__SA_Elf_MIPS_ABIFlags_v0(Structure):
    pass

struct_c__SA_Elf_MIPS_ABIFlags_v0._pack_ = 1 # source:False
struct_c__SA_Elf_MIPS_ABIFlags_v0._fields_ = [
    ('version', ctypes.c_uint16),
    ('isa_level', ctypes.c_ubyte),
    ('isa_rev', ctypes.c_ubyte),
    ('gpr_size', ctypes.c_ubyte),
    ('cpr1_size', ctypes.c_ubyte),
    ('cpr2_size', ctypes.c_ubyte),
    ('fp_abi', ctypes.c_ubyte),
    ('isa_ext', ctypes.c_uint32),
    ('ases', ctypes.c_uint32),
    ('flags1', ctypes.c_uint32),
    ('flags2', ctypes.c_uint32),
]

Elf_MIPS_ABIFlags_v0 = struct_c__SA_Elf_MIPS_ABIFlags_v0

# values for enumeration 'c__Ea_Val_GNU_MIPS_ABI_FP_ANY'
c__Ea_Val_GNU_MIPS_ABI_FP_ANY__enumvalues = {
    0: 'Val_GNU_MIPS_ABI_FP_ANY',
    1: 'Val_GNU_MIPS_ABI_FP_DOUBLE',
    2: 'Val_GNU_MIPS_ABI_FP_SINGLE',
    3: 'Val_GNU_MIPS_ABI_FP_SOFT',
    4: 'Val_GNU_MIPS_ABI_FP_OLD_64',
    5: 'Val_GNU_MIPS_ABI_FP_XX',
    6: 'Val_GNU_MIPS_ABI_FP_64',
    7: 'Val_GNU_MIPS_ABI_FP_64A',
    7: 'Val_GNU_MIPS_ABI_FP_MAX',
}
Val_GNU_MIPS_ABI_FP_ANY = 0
Val_GNU_MIPS_ABI_FP_DOUBLE = 1
Val_GNU_MIPS_ABI_FP_SINGLE = 2
Val_GNU_MIPS_ABI_FP_SOFT = 3
Val_GNU_MIPS_ABI_FP_OLD_64 = 4
Val_GNU_MIPS_ABI_FP_XX = 5
Val_GNU_MIPS_ABI_FP_64 = 6
Val_GNU_MIPS_ABI_FP_64A = 7
Val_GNU_MIPS_ABI_FP_MAX = 7
c__Ea_Val_GNU_MIPS_ABI_FP_ANY = ctypes.c_uint32 # enum
ssize_t = ctypes.c_int64
gid_t = ctypes.c_uint32
uid_t = ctypes.c_uint32
useconds_t = ctypes.c_uint32
pid_t = ctypes.c_int32
intptr_t = ctypes.c_int64
socklen_t = ctypes.c_uint32
try:
    access = _libraries['libc'].access
    access.restype = ctypes.c_int32
    access.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    faccessat = _libraries['libc'].faccessat
    faccessat.restype = ctypes.c_int32
    faccessat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    lseek = _libraries['libc'].lseek
    lseek.restype = __off_t
    lseek.argtypes = [ctypes.c_int32, __off_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    close = _libraries['libc'].close
    close.restype = ctypes.c_int32
    close.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    closefrom = _libraries['libc'].closefrom
    closefrom.restype = None
    closefrom.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    read = _libraries['libc'].read
    read.restype = ssize_t
    read.argtypes = [ctypes.c_int32, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    write = _libraries['libc'].write
    write.restype = ssize_t
    write.argtypes = [ctypes.c_int32, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    pread = _libraries['libc'].pread
    pread.restype = ssize_t
    pread.argtypes = [ctypes.c_int32, ctypes.POINTER(None), size_t, __off_t]
except AttributeError:
    pass
try:
    pwrite = _libraries['libc'].pwrite
    pwrite.restype = ssize_t
    pwrite.argtypes = [ctypes.c_int32, ctypes.POINTER(None), size_t, __off_t]
except AttributeError:
    pass
try:
    pipe = _libraries['libc'].pipe
    pipe.restype = ctypes.c_int32
    pipe.argtypes = [ctypes.c_int32 * 2]
except AttributeError:
    pass
try:
    alarm = _libraries['libc'].alarm
    alarm.restype = ctypes.c_uint32
    alarm.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    sleep = _libraries['libc'].sleep
    sleep.restype = ctypes.c_uint32
    sleep.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
__useconds_t = ctypes.c_uint32
try:
    ualarm = _libraries['libc'].ualarm
    ualarm.restype = __useconds_t
    ualarm.argtypes = [__useconds_t, __useconds_t]
except AttributeError:
    pass
try:
    usleep = _libraries['libc'].usleep
    usleep.restype = ctypes.c_int32
    usleep.argtypes = [__useconds_t]
except AttributeError:
    pass
try:
    pause = _libraries['libc'].pause
    pause.restype = ctypes.c_int32
    pause.argtypes = []
except AttributeError:
    pass
__uid_t = ctypes.c_uint32
__gid_t = ctypes.c_uint32
try:
    chown = _libraries['libc'].chown
    chown.restype = ctypes.c_int32
    chown.argtypes = [ctypes.POINTER(ctypes.c_char), __uid_t, __gid_t]
except AttributeError:
    pass
try:
    fchown = _libraries['libc'].fchown
    fchown.restype = ctypes.c_int32
    fchown.argtypes = [ctypes.c_int32, __uid_t, __gid_t]
except AttributeError:
    pass
try:
    lchown = _libraries['libc'].lchown
    lchown.restype = ctypes.c_int32
    lchown.argtypes = [ctypes.POINTER(ctypes.c_char), __uid_t, __gid_t]
except AttributeError:
    pass
try:
    fchownat = _libraries['libc'].fchownat
    fchownat.restype = ctypes.c_int32
    fchownat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), __uid_t, __gid_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    chdir = _libraries['libc'].chdir
    chdir.restype = ctypes.c_int32
    chdir.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    fchdir = _libraries['libc'].fchdir
    fchdir.restype = ctypes.c_int32
    fchdir.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    getcwd = _libraries['libc'].getcwd
    getcwd.restype = ctypes.POINTER(ctypes.c_char)
    getcwd.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    getwd = _libraries['libc'].getwd
    getwd.restype = ctypes.POINTER(ctypes.c_char)
    getwd.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    dup = _libraries['libc'].dup
    dup.restype = ctypes.c_int32
    dup.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    dup2 = _libraries['libc'].dup2
    dup2.restype = ctypes.c_int32
    dup2.argtypes = [ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
__environ = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))() # Variable ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
try:
    execve = _libraries['libc'].execve
    execve.restype = ctypes.c_int32
    execve.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char) * 0, ctypes.POINTER(ctypes.c_char) * 0]
except AttributeError:
    pass
try:
    fexecve = _libraries['libc'].fexecve
    fexecve.restype = ctypes.c_int32
    fexecve.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char) * 0, ctypes.POINTER(ctypes.c_char) * 0]
except AttributeError:
    pass
try:
    execv = _libraries['libc'].execv
    execv.restype = ctypes.c_int32
    execv.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char) * 0]
except AttributeError:
    pass
try:
    execle = _libraries['libc'].execle
    execle.restype = ctypes.c_int32
    execle.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    execl = _libraries['libc'].execl
    execl.restype = ctypes.c_int32
    execl.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    execvp = _libraries['libc'].execvp
    execvp.restype = ctypes.c_int32
    execvp.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char) * 0]
except AttributeError:
    pass
try:
    execlp = _libraries['libc'].execlp
    execlp.restype = ctypes.c_int32
    execlp.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nice = _libraries['libc'].nice
    nice.restype = ctypes.c_int32
    nice.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    _exit = _libraries['libc']._exit
    _exit.restype = None
    _exit.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    pathconf = _libraries['libc'].pathconf
    pathconf.restype = ctypes.c_int64
    pathconf.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    fpathconf = _libraries['libc'].fpathconf
    fpathconf.restype = ctypes.c_int64
    fpathconf.argtypes = [ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    sysconf = _libraries['libc'].sysconf
    sysconf.restype = ctypes.c_int64
    sysconf.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    confstr = _libraries['libc'].confstr
    confstr.restype = size_t
    confstr.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
__pid_t = ctypes.c_int32
try:
    getpid = _libraries['libc'].getpid
    getpid.restype = __pid_t
    getpid.argtypes = []
except AttributeError:
    pass
try:
    getppid = _libraries['libc'].getppid
    getppid.restype = __pid_t
    getppid.argtypes = []
except AttributeError:
    pass
try:
    getpgrp = _libraries['libc'].getpgrp
    getpgrp.restype = __pid_t
    getpgrp.argtypes = []
except AttributeError:
    pass
try:
    __getpgid = _libraries['libc'].__getpgid
    __getpgid.restype = __pid_t
    __getpgid.argtypes = [__pid_t]
except AttributeError:
    pass
try:
    getpgid = _libraries['libc'].getpgid
    getpgid.restype = __pid_t
    getpgid.argtypes = [__pid_t]
except AttributeError:
    pass
try:
    setpgid = _libraries['libc'].setpgid
    setpgid.restype = ctypes.c_int32
    setpgid.argtypes = [__pid_t, __pid_t]
except AttributeError:
    pass
try:
    setpgrp = _libraries['libc'].setpgrp
    setpgrp.restype = ctypes.c_int32
    setpgrp.argtypes = []
except AttributeError:
    pass
try:
    setsid = _libraries['libc'].setsid
    setsid.restype = __pid_t
    setsid.argtypes = []
except AttributeError:
    pass
try:
    getsid = _libraries['libc'].getsid
    getsid.restype = __pid_t
    getsid.argtypes = [__pid_t]
except AttributeError:
    pass
try:
    getuid = _libraries['libc'].getuid
    getuid.restype = __uid_t
    getuid.argtypes = []
except AttributeError:
    pass
try:
    geteuid = _libraries['libc'].geteuid
    geteuid.restype = __uid_t
    geteuid.argtypes = []
except AttributeError:
    pass
try:
    getgid = _libraries['libc'].getgid
    getgid.restype = __gid_t
    getgid.argtypes = []
except AttributeError:
    pass
try:
    getegid = _libraries['libc'].getegid
    getegid.restype = __gid_t
    getegid.argtypes = []
except AttributeError:
    pass
try:
    getgroups = _libraries['libc'].getgroups
    getgroups.restype = ctypes.c_int32
    getgroups.argtypes = [ctypes.c_int32, ctypes.c_uint32 * 0]
except AttributeError:
    pass
try:
    setuid = _libraries['libc'].setuid
    setuid.restype = ctypes.c_int32
    setuid.argtypes = [__uid_t]
except AttributeError:
    pass
try:
    setreuid = _libraries['libc'].setreuid
    setreuid.restype = ctypes.c_int32
    setreuid.argtypes = [__uid_t, __uid_t]
except AttributeError:
    pass
try:
    seteuid = _libraries['libc'].seteuid
    seteuid.restype = ctypes.c_int32
    seteuid.argtypes = [__uid_t]
except AttributeError:
    pass
try:
    setgid = _libraries['libc'].setgid
    setgid.restype = ctypes.c_int32
    setgid.argtypes = [__gid_t]
except AttributeError:
    pass
try:
    setregid = _libraries['libc'].setregid
    setregid.restype = ctypes.c_int32
    setregid.argtypes = [__gid_t, __gid_t]
except AttributeError:
    pass
try:
    setegid = _libraries['libc'].setegid
    setegid.restype = ctypes.c_int32
    setegid.argtypes = [__gid_t]
except AttributeError:
    pass
try:
    fork = _libraries['libc'].fork
    fork.restype = __pid_t
    fork.argtypes = []
except AttributeError:
    pass
try:
    vfork = _libraries['libc'].vfork
    vfork.restype = ctypes.c_int32
    vfork.argtypes = []
except AttributeError:
    pass
try:
    ttyname = _libraries['libc'].ttyname
    ttyname.restype = ctypes.POINTER(ctypes.c_char)
    ttyname.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    ttyname_r = _libraries['libc'].ttyname_r
    ttyname_r.restype = ctypes.c_int32
    ttyname_r.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    isatty = _libraries['libc'].isatty
    isatty.restype = ctypes.c_int32
    isatty.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    ttyslot = _libraries['libc'].ttyslot
    ttyslot.restype = ctypes.c_int32
    ttyslot.argtypes = []
except AttributeError:
    pass
try:
    link = _libraries['libc'].link
    link.restype = ctypes.c_int32
    link.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    linkat = _libraries['libc'].linkat
    linkat.restype = ctypes.c_int32
    linkat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    symlink = _libraries['libc'].symlink
    symlink.restype = ctypes.c_int32
    symlink.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    readlink = _libraries['libc'].readlink
    readlink.restype = ssize_t
    readlink.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    symlinkat = _libraries['libc'].symlinkat
    symlinkat.restype = ctypes.c_int32
    symlinkat.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    readlinkat = _libraries['libc'].readlinkat
    readlinkat.restype = ssize_t
    readlinkat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    unlink = _libraries['libc'].unlink
    unlink.restype = ctypes.c_int32
    unlink.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    unlinkat = _libraries['libc'].unlinkat
    unlinkat.restype = ctypes.c_int32
    unlinkat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    rmdir = _libraries['libc'].rmdir
    rmdir.restype = ctypes.c_int32
    rmdir.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    tcgetpgrp = _libraries['libc'].tcgetpgrp
    tcgetpgrp.restype = __pid_t
    tcgetpgrp.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    tcsetpgrp = _libraries['libc'].tcsetpgrp
    tcsetpgrp.restype = ctypes.c_int32
    tcsetpgrp.argtypes = [ctypes.c_int32, __pid_t]
except AttributeError:
    pass
try:
    getlogin = _libraries['libc'].getlogin
    getlogin.restype = ctypes.POINTER(ctypes.c_char)
    getlogin.argtypes = []
except AttributeError:
    pass
try:
    getlogin_r = _libraries['libc'].getlogin_r
    getlogin_r.restype = ctypes.c_int32
    getlogin_r.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    setlogin = _libraries['libc'].setlogin
    setlogin.restype = ctypes.c_int32
    setlogin.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    gethostname = _libraries['libc'].gethostname
    gethostname.restype = ctypes.c_int32
    gethostname.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    sethostname = _libraries['libc'].sethostname
    sethostname.restype = ctypes.c_int32
    sethostname.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    sethostid = _libraries['libc'].sethostid
    sethostid.restype = ctypes.c_int32
    sethostid.argtypes = [ctypes.c_int64]
except AttributeError:
    pass
try:
    getdomainname = _libraries['libc'].getdomainname
    getdomainname.restype = ctypes.c_int32
    getdomainname.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    setdomainname = _libraries['libc'].setdomainname
    setdomainname.restype = ctypes.c_int32
    setdomainname.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    vhangup = _libraries['libc'].vhangup
    vhangup.restype = ctypes.c_int32
    vhangup.argtypes = []
except AttributeError:
    pass
try:
    revoke = _libraries['libc'].revoke
    revoke.restype = ctypes.c_int32
    revoke.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    profil = _libraries['libc'].profil
    profil.restype = ctypes.c_int32
    profil.argtypes = [ctypes.POINTER(ctypes.c_uint16), size_t, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    acct = _libraries['libc'].acct
    acct.restype = ctypes.c_int32
    acct.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    getusershell = _libraries['libc'].getusershell
    getusershell.restype = ctypes.POINTER(ctypes.c_char)
    getusershell.argtypes = []
except AttributeError:
    pass
try:
    endusershell = _libraries['libc'].endusershell
    endusershell.restype = None
    endusershell.argtypes = []
except AttributeError:
    pass
try:
    setusershell = _libraries['libc'].setusershell
    setusershell.restype = None
    setusershell.argtypes = []
except AttributeError:
    pass
try:
    daemon = _libraries['libc'].daemon
    daemon.restype = ctypes.c_int32
    daemon.argtypes = [ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    chroot = _libraries['libc'].chroot
    chroot.restype = ctypes.c_int32
    chroot.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    getpass = _libraries['libc'].getpass
    getpass.restype = ctypes.POINTER(ctypes.c_char)
    getpass.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    fsync = _libraries['libc'].fsync
    fsync.restype = ctypes.c_int32
    fsync.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    gethostid = _libraries['libc'].gethostid
    gethostid.restype = ctypes.c_int64
    gethostid.argtypes = []
except AttributeError:
    pass
try:
    sync = _libraries['libc'].sync
    sync.restype = None
    sync.argtypes = []
except AttributeError:
    pass
try:
    getpagesize = _libraries['libc'].getpagesize
    getpagesize.restype = ctypes.c_int32
    getpagesize.argtypes = []
except AttributeError:
    pass
try:
    getdtablesize = _libraries['libc'].getdtablesize
    getdtablesize.restype = ctypes.c_int32
    getdtablesize.argtypes = []
except AttributeError:
    pass
try:
    truncate = _libraries['libc'].truncate
    truncate.restype = ctypes.c_int32
    truncate.argtypes = [ctypes.POINTER(ctypes.c_char), __off_t]
except AttributeError:
    pass
try:
    ftruncate = _libraries['libc'].ftruncate
    ftruncate.restype = ctypes.c_int32
    ftruncate.argtypes = [ctypes.c_int32, __off_t]
except AttributeError:
    pass
try:
    brk = _libraries['libc'].brk
    brk.restype = ctypes.c_int32
    brk.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    sbrk = _libraries['libc'].sbrk
    sbrk.restype = ctypes.POINTER(None)
    sbrk.argtypes = [intptr_t]
except AttributeError:
    pass
try:
    syscall = _libraries['libc'].syscall
    syscall.restype = ctypes.c_int64
    syscall.argtypes = [ctypes.c_int64]
except AttributeError:
    pass
try:
    lockf = _libraries['libc'].lockf
    lockf.restype = ctypes.c_int32
    lockf.argtypes = [ctypes.c_int32, ctypes.c_int32, __off_t]
except AttributeError:
    pass
try:
    fdatasync = _libraries['libc'].fdatasync
    fdatasync.restype = ctypes.c_int32
    fdatasync.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    crypt = _libraries['libc'].crypt
    crypt.restype = ctypes.POINTER(ctypes.c_char)
    crypt.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    getentropy = _libraries['libc'].getentropy
    getentropy.restype = ctypes.c_int32
    getentropy.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
__all__ = \
    ['Elf32_Addr', 'Elf32_Chdr', 'Elf32_Conflict', 'Elf32_Dyn',
    'Elf32_Ehdr', 'Elf32_Half', 'Elf32_Lib', 'Elf32_Move',
    'Elf32_Nhdr', 'Elf32_Off', 'Elf32_Phdr', 'Elf32_RegInfo',
    'Elf32_Rel', 'Elf32_Rela', 'Elf32_Section', 'Elf32_Shdr',
    'Elf32_Sword', 'Elf32_Sxword', 'Elf32_Sym', 'Elf32_Syminfo',
    'Elf32_Verdaux', 'Elf32_Verdef', 'Elf32_Vernaux', 'Elf32_Verneed',
    'Elf32_Versym', 'Elf32_Word', 'Elf32_Xword', 'Elf32_auxv_t',
    'Elf32_gptab', 'Elf64_Addr', 'Elf64_Chdr', 'Elf64_Dyn',
    'Elf64_Ehdr', 'Elf64_Half', 'Elf64_Lib', 'Elf64_Move',
    'Elf64_Nhdr', 'Elf64_Off', 'Elf64_Phdr', 'Elf64_Rel',
    'Elf64_Rela', 'Elf64_Section', 'Elf64_Shdr', 'Elf64_Sword',
    'Elf64_Sxword', 'Elf64_Sym', 'Elf64_Syminfo', 'Elf64_Verdaux',
    'Elf64_Verdef', 'Elf64_Vernaux', 'Elf64_Verneed', 'Elf64_Versym',
    'Elf64_Word', 'Elf64_Xword', 'Elf64_auxv_t',
    'Elf_MIPS_ABIFlags_v0', 'Elf_Options', 'Elf_Options_Hw',
    'Val_GNU_MIPS_ABI_FP_64', 'Val_GNU_MIPS_ABI_FP_64A',
    'Val_GNU_MIPS_ABI_FP_ANY', 'Val_GNU_MIPS_ABI_FP_DOUBLE',
    'Val_GNU_MIPS_ABI_FP_MAX', 'Val_GNU_MIPS_ABI_FP_OLD_64',
    'Val_GNU_MIPS_ABI_FP_SINGLE', 'Val_GNU_MIPS_ABI_FP_SOFT',
    'Val_GNU_MIPS_ABI_FP_XX', '__environ', '__getpgid', '__gid_t',
    '__off_t', '__pid_t', '__uid_t', '__useconds_t', '_exit',
    'access', 'acct', 'alarm', 'brk', 'c__Ea_Val_GNU_MIPS_ABI_FP_ANY',
    'chdir', 'chown', 'chroot', 'close', 'closefrom', 'confstr',
    'crypt', 'daemon', 'dup', 'dup2', 'endusershell', 'execl',
    'execle', 'execlp', 'execv', 'execve', 'execvp', 'faccessat',
    'fchdir', 'fchown', 'fchownat', 'fdatasync', 'fexecve', 'fork',
    'fpathconf', 'fsync', 'ftruncate', 'getcwd', 'getdomainname',
    'getdtablesize', 'getegid', 'getentropy', 'geteuid', 'getgid',
    'getgroups', 'gethostid', 'gethostname', 'getlogin', 'getlogin_r',
    'getpagesize', 'getpass', 'getpgid', 'getpgrp', 'getpid',
    'getppid', 'getsid', 'getuid', 'getusershell', 'getwd', 'gid_t',
    'intptr_t', 'isatty', 'lchown', 'link', 'linkat', 'lockf',
    'lseek', 'madvise', 'mincore', 'mlock', 'mlockall', 'mmap',
    'mode_t', 'mprotect', 'msync', 'munlock', 'munlockall', 'munmap',
    'nice', 'off_t', 'pathconf', 'pause', 'pid_t', 'pipe',
    'posix_madvise', 'pread', 'profil', 'pwrite', 'read', 'readlink',
    'readlinkat', 'revoke', 'rmdir', 'sbrk', 'setdomainname',
    'setegid', 'seteuid', 'setgid', 'sethostid', 'sethostname',
    'setlogin', 'setpgid', 'setpgrp', 'setregid', 'setreuid',
    'setsid', 'setuid', 'setusershell', 'shm_open', 'shm_unlink',
    'size_t', 'sleep', 'socklen_t', 'ssize_t',
    'struct_c__SA_Elf32_Chdr', 'struct_c__SA_Elf32_Dyn',
    'struct_c__SA_Elf32_Ehdr', 'struct_c__SA_Elf32_Lib',
    'struct_c__SA_Elf32_Move', 'struct_c__SA_Elf32_Nhdr',
    'struct_c__SA_Elf32_Phdr', 'struct_c__SA_Elf32_RegInfo',
    'struct_c__SA_Elf32_Rel', 'struct_c__SA_Elf32_Rela',
    'struct_c__SA_Elf32_Shdr', 'struct_c__SA_Elf32_Sym',
    'struct_c__SA_Elf32_Syminfo', 'struct_c__SA_Elf32_Verdaux',
    'struct_c__SA_Elf32_Verdef', 'struct_c__SA_Elf32_Vernaux',
    'struct_c__SA_Elf32_Verneed', 'struct_c__SA_Elf32_auxv_t',
    'struct_c__SA_Elf64_Chdr', 'struct_c__SA_Elf64_Dyn',
    'struct_c__SA_Elf64_Ehdr', 'struct_c__SA_Elf64_Lib',
    'struct_c__SA_Elf64_Move', 'struct_c__SA_Elf64_Nhdr',
    'struct_c__SA_Elf64_Phdr', 'struct_c__SA_Elf64_Rel',
    'struct_c__SA_Elf64_Rela', 'struct_c__SA_Elf64_Shdr',
    'struct_c__SA_Elf64_Sym', 'struct_c__SA_Elf64_Syminfo',
    'struct_c__SA_Elf64_Verdaux', 'struct_c__SA_Elf64_Verdef',
    'struct_c__SA_Elf64_Vernaux', 'struct_c__SA_Elf64_Verneed',
    'struct_c__SA_Elf64_auxv_t', 'struct_c__SA_Elf_MIPS_ABIFlags_v0',
    'struct_c__SA_Elf_Options', 'struct_c__SA_Elf_Options_Hw',
    'struct_c__UA_Elf32_gptab_gt_entry',
    'struct_c__UA_Elf32_gptab_gt_header', 'symlink', 'symlinkat',
    'sync', 'syscall', 'sysconf', 'tcgetpgrp', 'tcsetpgrp',
    'truncate', 'ttyname', 'ttyname_r', 'ttyslot', 'ualarm', 'uid_t',
    'union_c__SA_Elf32_Dyn_d_un', 'union_c__SA_Elf32_auxv_t_a_un',
    'union_c__SA_Elf64_Dyn_d_un', 'union_c__SA_Elf64_auxv_t_a_un',
    'union_c__UA_Elf32_gptab', 'unlink', 'unlinkat', 'useconds_t',
    'usleep', 'vfork', 'vhangup', 'write']
_ELF_H = 1
EI_NIDENT = (16)
EI_MAG0 = 0
ELFMAG0 = 0x7f
EI_MAG1 = 1
ELFMAG1 = 'E'
EI_MAG2 = 2
ELFMAG2 = 'L'
EI_MAG3 = 3
ELFMAG3 = 'F'
ELFMAG = "\177ELF"
SELFMAG = 4
EI_CLASS = 4
ELFCLASSNONE = 0
ELFCLASS32 = 1
ELFCLASS64 = 2
ELFCLASSNUM = 3
EI_DATA = 5
ELFDATANONE = 0
ELFDATA2LSB = 1
ELFDATA2MSB = 2
ELFDATANUM = 3
EI_VERSION = 6
EI_OSABI = 7
ELFOSABI_NONE = 0
ELFOSABI_SYSV = 0
ELFOSABI_HPUX = 1
ELFOSABI_NETBSD = 2
ELFOSABI_GNU = 3
ELFOSABI_LINUX = ELFOSABI_GNU
ELFOSABI_SOLARIS = 6
ELFOSABI_AIX = 7
ELFOSABI_IRIX = 8
ELFOSABI_FREEBSD = 9
ELFOSABI_TRU64 = 10
ELFOSABI_MODESTO = 11
ELFOSABI_OPENBSD = 12
ELFOSABI_ARM_AEABI = 64
ELFOSABI_ARM = 97
ELFOSABI_STANDALONE = 255
EI_ABIVERSION = 8
EI_PAD = 9
ET_NONE = 0
ET_REL = 1
ET_EXEC = 2
ET_DYN = 3
ET_CORE = 4
ET_NUM = 5
ET_LOOS = 0xfe00
ET_HIOS = 0xfeff
ET_LOPROC = 0xff00
ET_HIPROC = 0xffff
EM_NONE = 0
EM_M32 = 1
EM_SPARC = 2
EM_386 = 3
EM_68K = 4
EM_88K = 5
EM_IAMCU = 6
EM_860 = 7
EM_MIPS = 8
EM_S370 = 9
EM_MIPS_RS3_LE = 10
EM_PARISC = 15
EM_VPP500 = 17
EM_SPARC32PLUS = 18
EM_960 = 19
EM_PPC = 20
EM_PPC64 = 21
EM_S390 = 22
EM_SPU = 23
EM_V800 = 36
EM_FR20 = 37
EM_RH32 = 38
EM_RCE = 39
EM_ARM = 40
EM_FAKE_ALPHA = 41
EM_SH = 42
EM_SPARCV9 = 43
EM_TRICORE = 44
EM_ARC = 45
EM_H8_300 = 46
EM_H8_300H = 47
EM_H8S = 48
EM_H8_500 = 49
EM_IA_64 = 50
EM_MIPS_X = 51
EM_COLDFIRE = 52
EM_68HC12 = 53
EM_MMA = 54
EM_PCP = 55
EM_NCPU = 56
EM_NDR1 = 57
EM_STARCORE = 58
EM_ME16 = 59
EM_ST100 = 60
EM_TINYJ = 61
EM_X86_64 = 62
EM_PDSP = 63
EM_PDP10 = 64
EM_PDP11 = 65
EM_FX66 = 66
EM_ST9PLUS = 67
EM_ST7 = 68
EM_68HC16 = 69
EM_68HC11 = 70
EM_68HC08 = 71
EM_68HC05 = 72
EM_SVX = 73
EM_ST19 = 74
EM_VAX = 75
EM_CRIS = 76
EM_JAVELIN = 77
EM_FIREPATH = 78
EM_ZSP = 79
EM_MMIX = 80
EM_HUANY = 81
EM_PRISM = 82
EM_AVR = 83
EM_FR30 = 84
EM_D10V = 85
EM_D30V = 86
EM_V850 = 87
EM_M32R = 88
EM_MN10300 = 89
EM_MN10200 = 90
EM_PJ = 91
EM_OPENRISC = 92
EM_ARC_COMPACT = 93
EM_XTENSA = 94
EM_VIDEOCORE = 95
EM_TMM_GPP = 96
EM_NS32K = 97
EM_TPC = 98
EM_SNP1K = 99
EM_ST200 = 100
EM_IP2K = 101
EM_MAX = 102
EM_CR = 103
EM_F2MC16 = 104
EM_MSP430 = 105
EM_BLACKFIN = 106
EM_SE_C33 = 107
EM_SEP = 108
EM_ARCA = 109
EM_UNICORE = 110
EM_EXCESS = 111
EM_DXP = 112
EM_ALTERA_NIOS2 = 113
EM_CRX = 114
EM_XGATE = 115
EM_C166 = 116
EM_M16C = 117
EM_DSPIC30F = 118
EM_CE = 119
EM_M32C = 120
EM_TSK3000 = 131
EM_RS08 = 132
EM_SHARC = 133
EM_ECOG2 = 134
EM_SCORE7 = 135
EM_DSP24 = 136
EM_VIDEOCORE3 = 137
EM_LATTICEMICO32 = 138
EM_SE_C17 = 139
EM_TI_C6000 = 140
EM_TI_C2000 = 141
EM_TI_C5500 = 142
EM_TI_ARP32 = 143
EM_TI_PRU = 144
EM_MMDSP_PLUS = 160
EM_CYPRESS_M8C = 161
EM_R32C = 162
EM_TRIMEDIA = 163
EM_QDSP6 = 164
EM_8051 = 165
EM_STXP7X = 166
EM_NDS32 = 167
EM_ECOG1X = 168
EM_MAXQ30 = 169
EM_XIMO16 = 170
EM_MANIK = 171
EM_CRAYNV2 = 172
EM_RX = 173
EM_METAG = 174
EM_MCST_ELBRUS = 175
EM_ECOG16 = 176
EM_CR16 = 177
EM_ETPU = 178
EM_SLE9X = 179
EM_L10M = 180
EM_K10M = 181
EM_AARCH64 = 183
EM_AVR32 = 185
EM_STM8 = 186
EM_TILE64 = 187
EM_TILEPRO = 188
EM_MICROBLAZE = 189
EM_CUDA = 190
EM_TILEGX = 191
EM_CLOUDSHIELD = 192
EM_COREA_1ST = 193
EM_COREA_2ND = 194
EM_ARCV2 = 195
EM_OPEN8 = 196
EM_RL78 = 197
EM_VIDEOCORE5 = 198
EM_78KOR = 199
EM_56800EX = 200
EM_BA1 = 201
EM_BA2 = 202
EM_XCORE = 203
EM_MCHP_PIC = 204
EM_INTELGT = 205
EM_KM32 = 210
EM_KMX32 = 211
EM_EMX16 = 212
EM_EMX8 = 213
EM_KVARC = 214
EM_CDP = 215
EM_COGE = 216
EM_COOL = 217
EM_NORC = 218
EM_CSR_KALIMBA = 219
EM_Z80 = 220
EM_VISIUM = 221
EM_FT32 = 222
EM_MOXIE = 223
EM_AMDGPU = 224
EM_RISCV = 243
EM_BPF = 247
EM_CSKY = 252
EM_NUM = 253
EM_ARC_A5 = EM_ARC_COMPACT
EM_ALPHA = 0x9026
EV_NONE = 0
EV_CURRENT = 1
EV_NUM = 2
SHN_UNDEF = 0
SHN_LORESERVE = 0xff00
SHN_LOPROC = 0xff00
SHN_HIPROC = 0xff1f
SHN_LOOS = 0xff20
SHN_HIOS = 0xff3f
SHN_ABS = 0xfff1
SHN_COMMON = 0xfff2
SHN_XINDEX = 0xffff
SHN_HIRESERVE = 0xffff
SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_HASH = 5
SHT_DYNAMIC = 6
SHT_NOTE = 7
SHT_NOBITS = 8
SHT_REL = 9
SHT_SHLIB = 10
SHT_DYNSYM = 11
SHT_INIT_ARRAY = 14
SHT_FINI_ARRAY = 15
SHT_PREINIT_ARRAY = 16
SHT_GROUP = 17
SHT_SYMTAB_SHNDX = 18
SHT_NUM = 19
SHT_LOOS = 0x60000000
SHT_GNU_ATTRIBUTES = 0x6ffffff5
SHT_GNU_HASH = 0x6ffffff6
SHT_GNU_LIBLIST = 0x6ffffff7
SHT_CHECKSUM = 0x6ffffff8
SHT_LOSUNW = 0x6ffffffa
SHT_SUNW_move = 0x6ffffffa
SHT_SUNW_COMDAT = 0x6ffffffb
SHT_SUNW_syminfo = 0x6ffffffc
SHT_GNU_verdef = 0x6ffffffd
SHT_GNU_verneed = 0x6ffffffe
SHT_GNU_versym = 0x6fffffff
SHT_HISUNW = 0x6fffffff
SHT_HIOS = 0x6fffffff
SHT_LOPROC = 0x70000000
SHT_HIPROC = 0x7fffffff
SHT_LOUSER = 0x80000000
SHT_HIUSER = 0x8fffffff
SHF_WRITE = (1 << 0)
SHF_ALLOC = (1 << 1)
SHF_EXECINSTR = (1 << 2)
SHF_MERGE = (1 << 4)
SHF_STRINGS = (1 << 5)
SHF_INFO_LINK = (1 << 6)
SHF_LINK_ORDER = (1 << 7)
SHF_GROUP = (1 << 9)
SHF_TLS = (1 << 10)
SHF_COMPRESSED = (1 << 11)
SHF_MASKOS = 0x0ff00000
SHF_MASKPROC = 0xf0000000
SHF_GNU_RETAIN = (1 << 21)
ELFCOMPRESS_ZLIB = 1
ELFCOMPRESS_LOOS = 0x60000000
ELFCOMPRESS_HIOS = 0x6fffffff
ELFCOMPRESS_LOPROC = 0x70000000
ELFCOMPRESS_HIPROC = 0x7fffffff
GRP_COMDAT = 0x1
SYMINFO_BT_SELF = 0xffff
SYMINFO_BT_PARENT = 0xfffe
SYMINFO_BT_LOWRESERVE = 0xff00
SYMINFO_FLG_DIRECT = 0x0001
SYMINFO_FLG_PASSTHRU = 0x0002
SYMINFO_FLG_COPY = 0x0004
SYMINFO_NONE = 0
SYMINFO_CURRENT = 1
SYMINFO_NUM = 2
def ELF32_ST_BIND(val): return (((ctypes.c_ubyte) (val)) >> 4)
def ELF32_ST_TYPE(val): return ((val) & 0xf)
def ELF32_ST_INFO(bind, type): return (((bind) << 4) + ((type) & 0xf))
def ELF64_ST_BIND(val): return ELF32_ST_BIND (val)
def ELF64_ST_TYPE(val): return ELF32_ST_TYPE (val)
def ELF64_ST_INFO(bind, type): return ELF32_ST_INFO ((bind), (type))
STB_LOCAL = 0
STB_GLOBAL = 1
STB_WEAK = 2
STB_NUM = 3
STB_LOOS = 10
STB_GNU_UNIQUE = 10
STB_HIOS = 12
STB_LOPROC = 13
STB_HIPROC = 15
STT_NOTYPE = 0
STT_OBJECT = 1
STT_FUNC = 2
STT_SECTION = 3
STT_FILE = 4
STT_COMMON = 5
STT_TLS = 6
STT_NUM = 7
STT_LOOS = 10
STT_GNU_IFUNC = 10
STT_HIOS = 12
STT_LOPROC = 13
STT_HIPROC = 15
STN_UNDEF = 0
def ELF32_ST_VISIBILITY(o): return ((o) & 0x03)
def ELF64_ST_VISIBILITY(o): return ELF32_ST_VISIBILITY (o)
STV_DEFAULT = 0
STV_INTERNAL = 1
STV_HIDDEN = 2
STV_PROTECTED = 3
def ELF32_R_SYM(val): return ((val) >> 8)
def ELF32_R_TYPE(val): return ((val) & 0xff)
def ELF32_R_INFO(sym, type): return (((sym) << 8) + ((type) & 0xff))
def ELF64_R_SYM(i): return ((i) >> 32)
def ELF64_R_TYPE(i): return ((i) & 0xffffffff)
def ELF64_R_INFO(sym,type): return ((((Elf64_Xword) (sym)) << 32) + (type))
PN_XNUM = 0xffff
PT_NULL = 0
PT_LOAD = 1
PT_DYNAMIC = 2
PT_INTERP = 3
PT_NOTE = 4
PT_SHLIB = 5
PT_PHDR = 6
PT_TLS = 7
PT_NUM = 8
PT_LOOS = 0x60000000
PT_GNU_EH_FRAME = 0x6474e550
PT_GNU_STACK = 0x6474e551
PT_GNU_RELRO = 0x6474e552
PT_GNU_PROPERTY = 0x6474e553
PT_LOSUNW = 0x6ffffffa
PT_SUNWBSS = 0x6ffffffa
PT_SUNWSTACK = 0x6ffffffb
PT_HISUNW = 0x6fffffff
PT_HIOS = 0x6fffffff
PT_LOPROC = 0x70000000
PT_HIPROC = 0x7fffffff
PF_X = (1 << 0)
PF_W = (1 << 1)
PF_R = (1 << 2)
PF_MASKOS = 0x0ff00000
PF_MASKPROC = 0xf0000000
NT_PRSTATUS = 1
NT_FPREGSET = 2
NT_PRPSINFO = 3
NT_PRXREG = 4
NT_TASKSTRUCT = 4
NT_PLATFORM = 5
NT_AUXV = 6
NT_GWINDOWS = 7
NT_ASRS = 8
NT_PSTATUS = 10
NT_PSINFO = 13
NT_PRCRED = 14
NT_UTSNAME = 15
NT_LWPSTATUS = 16
NT_LWPSINFO = 17
NT_PRFPXREG = 20
NT_PRXFPREG = 0x46e62b7f
NT_PPC_VMX = 0x100
NT_PPC_SPE = 0x101
NT_PPC_VSX = 0x102
NT_PPC_TAR = 0x103
NT_PPC_PPR = 0x104
NT_PPC_DSCR = 0x105
NT_PPC_EBB = 0x106
NT_PPC_PMU = 0x107
NT_PPC_TM_CGPR = 0x108
NT_PPC_TM_CFPR = 0x109
NT_PPC_TM_CVMX = 0x10a
NT_PPC_TM_CVSX = 0x10b
NT_PPC_TM_SPR = 0x10c
NT_386_TLS = 0x200
NT_386_IOPERM = 0x201
NT_X86_XSTATE = 0x202
NT_S390_HIGH_GPRS = 0x300
NT_S390_TIMER = 0x301
NT_S390_TODCMP = 0x302
NT_S390_TODPREG = 0x303
NT_S390_CTRS = 0x304
NT_S390_PREFIX = 0x305
NT_S390_LAST_BREAK = 0x306
NT_S390_SYSTEM_CALL = 0x307
NT_S390_TDB = 0x308
NT_S390_VXRS_HIGH = 0x30a
NT_S390_GS_CB = 0x30b
NT_S390_RI_CB = 0x30d
NT_ARM_VFP = 0x400
NT_ARM_TLS = 0x401
NT_ARM_HW_BREAK = 0x402
NT_ARM_HW_WATCH = 0x403
NT_ARM_SYSTEM_CALL = 0x404
NT_VMCOREDD = 0x700
NT_MIPS_DSP = 0x800
NT_MIPS_FP_MODE = 0x801
NT_MIPS_MSA = 0x802
NT_VERSION = 1
DT_NULL = 0
DT_NEEDED = 1
DT_PLTRELSZ = 2
DT_PLTGOT = 3
DT_HASH = 4
DT_STRTAB = 5
DT_SYMTAB = 6
DT_RELA = 7
DT_RELASZ = 8
DT_RELAENT = 9
DT_STRSZ = 10
DT_SYMENT = 11
DT_INIT = 12
DT_FINI = 13
DT_SONAME = 14
DT_RPATH = 15
DT_SYMBOLIC = 16
DT_REL = 17
DT_RELSZ = 18
DT_RELENT = 19
DT_PLTREL = 20
DT_DEBUG = 21
DT_TEXTREL = 22
DT_JMPREL = 23
DT_BIND_NOW = 24
DT_INIT_ARRAY = 25
DT_FINI_ARRAY = 26
DT_INIT_ARRAYSZ = 27
DT_FINI_ARRAYSZ = 28
DT_RUNPATH = 29
DT_FLAGS = 30
DT_ENCODING = 32
DT_PREINIT_ARRAY = 32
DT_PREINIT_ARRAYSZ = 33
DT_SYMTAB_SHNDX = 34
DT_NUM = 35
DT_LOOS = 0x6000000d
DT_HIOS = 0x6ffff000
DT_LOPROC = 0x70000000
DT_HIPROC = 0x7fffffff
DT_VALRNGLO = 0x6ffffd00
DT_GNU_PRELINKED = 0x6ffffdf5
DT_GNU_CONFLICTSZ = 0x6ffffdf6
DT_GNU_LIBLISTSZ = 0x6ffffdf7
DT_CHECKSUM = 0x6ffffdf8
DT_PLTPADSZ = 0x6ffffdf9
DT_MOVEENT = 0x6ffffdfa
DT_MOVESZ = 0x6ffffdfb
DT_FEATURE_1 = 0x6ffffdfc
DT_SYMINSZ = 0x6ffffdfe
DT_SYMINENT = 0x6ffffdff
DT_VALRNGHI = 0x6ffffdff
def DT_VALTAGIDX(tag): return (DT_VALRNGHI - (tag))
DT_VALNUM = 12
DT_ADDRRNGLO = 0x6ffffe00
DT_GNU_HASH = 0x6ffffef5
DT_TLSDESC_PLT = 0x6ffffef6
DT_TLSDESC_GOT = 0x6ffffef7
DT_GNU_CONFLICT = 0x6ffffef8
DT_GNU_LIBLIST = 0x6ffffef9
DT_CONFIG = 0x6ffffefa
DT_DEPAUDIT = 0x6ffffefb
DT_AUDIT = 0x6ffffefc
DT_PLTPAD = 0x6ffffefd
DT_MOVETAB = 0x6ffffefe
DT_SYMINFO = 0x6ffffeff
DT_ADDRRNGHI = 0x6ffffeff
def DT_ADDRTAGIDX(tag): return (DT_ADDRRNGHI - (tag))
DT_ADDRNUM = 11
DT_VERSYM = 0x6ffffff0
DT_RELACOUNT = 0x6ffffff9
DT_RELCOUNT = 0x6ffffffa
DT_FLAGS_1 = 0x6ffffffb
DT_VERDEFNUM = 0x6ffffffd
DT_VERNEEDNUM = 0x6fffffff
def DT_VERSIONTAGIDX(tag): return (DT_VERNEEDNUM - (tag))
DT_VERSIONTAGNUM = 16
DT_AUXILIARY = 0x7ffffffd
DT_FILTER = 0x7fffffff
def DT_EXTRATAGIDX(tag): return ((Elf32_Word)-((Elf32_Sword) (tag) <<1>>1)-1)
DT_EXTRANUM = 3
DF_ORIGIN = 0x00000001
DF_SYMBOLIC = 0x00000002
DF_TEXTREL = 0x00000004
DF_BIND_NOW = 0x00000008
DF_STATIC_TLS = 0x00000010
DF_1_NOW = 0x00000001
DF_1_GLOBAL = 0x00000002
DF_1_GROUP = 0x00000004
DF_1_NODELETE = 0x00000008
DF_1_LOADFLTR = 0x00000010
DF_1_INITFIRST = 0x00000020
DF_1_NOOPEN = 0x00000040
DF_1_ORIGIN = 0x00000080
DF_1_DIRECT = 0x00000100
DF_1_TRANS = 0x00000200
DF_1_INTERPOSE = 0x00000400
DF_1_NODEFLIB = 0x00000800
DF_1_NODUMP = 0x00001000
DF_1_CONFALT = 0x00002000
DF_1_ENDFILTEE = 0x00004000
DF_1_DISPRELDNE = 0x00008000
DF_1_DISPRELPND = 0x00010000
DF_1_NODIRECT = 0x00020000
DF_1_IGNMULDEF = 0x00040000
DF_1_NOKSYMS = 0x00080000
DF_1_NOHDR = 0x00100000
DF_1_EDITED = 0x00200000
DF_1_NORELOC = 0x00400000
DF_1_SYMINTPOSE = 0x00800000
DF_1_GLOBAUDIT = 0x01000000
DF_1_SINGLETON = 0x02000000
DF_1_STUB = 0x04000000
DF_1_PIE = 0x08000000
DF_1_KMOD = 0x10000000
DF_1_WEAKFILTER = 0x20000000
DF_1_NOCOMMON = 0x40000000
DTF_1_PARINIT = 0x00000001
DTF_1_CONFEXP = 0x00000002
DF_P1_LAZYLOAD = 0x00000001
VER_DEF_NONE = 0
VER_DEF_CURRENT = 1
VER_DEF_NUM = 2
VER_FLG_BASE = 0x1
VER_FLG_WEAK = 0x2
VER_NDX_LOCAL = 0
VER_NDX_GLOBAL = 1
VER_NDX_LORESERVE = 0xff00
VER_NDX_ELIMINATE = 0xff01
VER_NEED_NONE = 0
VER_NEED_CURRENT = 1
VER_NEED_NUM = 2
VER_FLG_WEAK = 0x2
AT_NULL = 0
AT_IGNORE = 1
AT_EXECFD = 2
AT_PHDR = 3
AT_PHENT = 4
AT_PHNUM = 5
AT_PAGESZ = 6
AT_BASE = 7
AT_FLAGS = 8
AT_ENTRY = 9
AT_NOTELF = 10
AT_UID = 11
AT_EUID = 12
AT_GID = 13
AT_EGID = 14
AT_CLKTCK = 17
AT_PLATFORM = 15
AT_FPUCW = 18
AT_DCACHEBSIZE = 19
AT_ICACHEBSIZE = 20
AT_UCACHEBSIZE = 21
AT_IGNOREPPC = 22
AT_SECURE = 23
AT_BASE_PLATFORM = 24
AT_RANDOM = 25
AT_EXECFN = 31
AT_SYSINFO = 32
AT_SYSINFO_EHDR = 33
AT_L1I_CACHESHAPE = 34
AT_L1D_CACHESHAPE = 35
AT_L2_CACHESHAPE = 36
AT_L3_CACHESHAPE = 37
AT_L1I_CACHESIZE = 40
AT_L1I_CACHEGEOMETRY = 41
AT_L1D_CACHESIZE = 42
AT_L1D_CACHEGEOMETRY = 43
AT_L2_CACHESIZE = 44
AT_L2_CACHEGEOMETRY = 45
AT_L3_CACHESIZE = 46
AT_L3_CACHEGEOMETRY = 47
AT_MINSIGSTKSZ = 51
ELF_NOTE_SOLARIS = "SUNW Solaris"
ELF_NOTE_GNU = "GNU"
ELF_NOTE_FDO = "FDO"
ELF_NOTE_PAGESIZE_HINT = 1
NT_GNU_ABI_TAG = 1
ELF_NOTE_ABI = NT_GNU_ABI_TAG
ELF_NOTE_OS_LINUX = 0
ELF_NOTE_OS_GNU = 1
ELF_NOTE_OS_SOLARIS2 = 2
ELF_NOTE_OS_FREEBSD = 3
NT_GNU_HWCAP = 2
NT_GNU_BUILD_ID = 3
NT_GNU_GOLD_VERSION = 4
NT_GNU_PROPERTY_TYPE_0 = 5
NT_FDO_PACKAGING_METADATA = 0xcafe1a7e
NOTE_GNU_PROPERTY_SECTION_NAME = ".note.gnu.property"
GNU_PROPERTY_STACK_SIZE = 1
GNU_PROPERTY_NO_COPY_ON_PROTECTED = 2
GNU_PROPERTY_UINT32_AND_LO = 0xb0000000
GNU_PROPERTY_UINT32_AND_HI = 0xb0007fff
GNU_PROPERTY_UINT32_OR_LO = 0xb0008000
GNU_PROPERTY_UINT32_OR_HI = 0xb000ffff
GNU_PROPERTY_1_NEEDED = GNU_PROPERTY_UINT32_OR_LO
GNU_PROPERTY_1_NEEDED_INDIRECT_EXTERN_ACCESS = (1 << 0)
GNU_PROPERTY_LOPROC = 0xc0000000
GNU_PROPERTY_HIPROC = 0xdfffffff
GNU_PROPERTY_LOUSER = 0xe0000000
GNU_PROPERTY_HIUSER = 0xffffffff
GNU_PROPERTY_AARCH64_FEATURE_1_AND = 0xc0000000
GNU_PROPERTY_AARCH64_FEATURE_1_BTI = (1 << 0)
GNU_PROPERTY_AARCH64_FEATURE_1_PAC = (1 << 1)
GNU_PROPERTY_X86_ISA_1_USED = 0xc0010002
GNU_PROPERTY_X86_ISA_1_NEEDED = 0xc0008002
GNU_PROPERTY_X86_FEATURE_1_AND = 0xc0000002
GNU_PROPERTY_X86_ISA_1_BASELINE = (1 << 0)
GNU_PROPERTY_X86_ISA_1_V2 = (1 << 1)
GNU_PROPERTY_X86_ISA_1_V3 = (1 << 2)
GNU_PROPERTY_X86_ISA_1_V4 = (1 << 3)
GNU_PROPERTY_X86_FEATURE_1_IBT = (1 << 0)
GNU_PROPERTY_X86_FEATURE_1_SHSTK = (1 << 1)
def ELF32_M_SYM(info): return ((info) >> 8)
def ELF32_M_SIZE(info): return ((ctypes.c_ubyte) (info))
def ELF32_M_INFO(sym, size): return (((sym) << 8) + (ctypes.c_ubyte) (size))
def ELF64_M_SYM(info): return ELF32_M_SYM (info)
def ELF64_M_SIZE(info): return ELF32_M_SIZE (info)
def ELF64_M_INFO(sym, size): return ELF32_M_INFO (sym, size)
EF_CPU32 = 0x00810000
R_68K_NONE = 0
R_68K_32 = 1
R_68K_16 = 2
R_68K_8 = 3
R_68K_PC32 = 4
R_68K_PC16 = 5
R_68K_PC8 = 6
R_68K_GOT32 = 7
R_68K_GOT16 = 8
R_68K_GOT8 = 9
R_68K_GOT32O = 10
R_68K_GOT16O = 11
R_68K_GOT8O = 12
R_68K_PLT32 = 13
R_68K_PLT16 = 14
R_68K_PLT8 = 15
R_68K_PLT32O = 16
R_68K_PLT16O = 17
R_68K_PLT8O = 18
R_68K_COPY = 19
R_68K_GLOB_DAT = 20
R_68K_JMP_SLOT = 21
R_68K_RELATIVE = 22
R_68K_TLS_GD32 = 25
R_68K_TLS_GD16 = 26
R_68K_TLS_GD8 = 27
R_68K_TLS_LDM32 = 28
R_68K_TLS_LDM16 = 29
R_68K_TLS_LDM8 = 30
R_68K_TLS_LDO32 = 31
R_68K_TLS_LDO16 = 32
R_68K_TLS_LDO8 = 33
R_68K_TLS_IE32 = 34
R_68K_TLS_IE16 = 35
R_68K_TLS_IE8 = 36
R_68K_TLS_DTPMOD32 = 40
R_68K_TLS_DTPREL32 = 41
R_68K_TLS_TPREL32 = 42
R_68K_NUM = 43
R_386_NONE = 0
R_386_32 = 1
R_386_PC32 = 2
R_386_GOT32 = 3
R_386_PLT32 = 4
R_386_COPY = 5
R_386_GLOB_DAT = 6
R_386_JMP_SLOT = 7
R_386_RELATIVE = 8
R_386_GOTOFF = 9
R_386_GOTPC = 10
R_386_32PLT = 11
R_386_TLS_TPOFF = 14
R_386_16 = 20
R_386_PC16 = 21
R_386_8 = 22
R_386_PC8 = 23
R_386_TLS_GD_PUSH = 25
R_386_TLS_GD_POP = 27
R_386_TLS_LDM_PUSH = 29
R_386_TLS_LDM_POP = 31
R_386_TLS_LDO_32 = 32
R_386_TLS_DTPMOD32 = 35
R_386_TLS_DTPOFF32 = 36
R_386_TLS_TPOFF32 = 37
R_386_SIZE32 = 38
R_386_TLS_GOTDESC = 39
R_386_IRELATIVE = 42
R_386_NUM = 44
STT_SPARC_REGISTER = 13
EF_SPARCV9_MM = 3
EF_SPARCV9_TSO = 0
EF_SPARCV9_PSO = 1
EF_SPARCV9_RMO = 2
EF_SPARC_LEDATA = 0x800000
EF_SPARC_EXT_MASK = 0xFFFF00
EF_SPARC_32PLUS = 0x000100
EF_SPARC_SUN_US1 = 0x000200
EF_SPARC_HAL_R1 = 0x000400
EF_SPARC_SUN_US3 = 0x000800
R_SPARC_NONE = 0
R_SPARC_8 = 1
R_SPARC_16 = 2
R_SPARC_32 = 3
R_SPARC_DISP8 = 4
R_SPARC_DISP16 = 5
R_SPARC_DISP32 = 6
R_SPARC_WDISP30 = 7
R_SPARC_WDISP22 = 8
R_SPARC_HI22 = 9
R_SPARC_22 = 10
R_SPARC_13 = 11
R_SPARC_LO10 = 12
R_SPARC_GOT10 = 13
R_SPARC_GOT13 = 14
R_SPARC_GOT22 = 15
R_SPARC_PC10 = 16
R_SPARC_PC22 = 17
R_SPARC_WPLT30 = 18
R_SPARC_COPY = 19
R_SPARC_GLOB_DAT = 20
R_SPARC_JMP_SLOT = 21
R_SPARC_RELATIVE = 22
R_SPARC_UA32 = 23
R_SPARC_PLT32 = 24
R_SPARC_HIPLT22 = 25
R_SPARC_LOPLT10 = 26
R_SPARC_PCPLT32 = 27
R_SPARC_PCPLT22 = 28
R_SPARC_PCPLT10 = 29
R_SPARC_10 = 30
R_SPARC_11 = 31
R_SPARC_64 = 32
R_SPARC_OLO10 = 33
R_SPARC_HH22 = 34
R_SPARC_HM10 = 35
R_SPARC_LM22 = 36
R_SPARC_PC_HH22 = 37
R_SPARC_PC_HM10 = 38
R_SPARC_PC_LM22 = 39
R_SPARC_WDISP16 = 40
R_SPARC_WDISP19 = 41
R_SPARC_GLOB_JMP = 42
R_SPARC_7 = 43
R_SPARC_5 = 44
R_SPARC_6 = 45
R_SPARC_DISP64 = 46
R_SPARC_PLT64 = 47
R_SPARC_HIX22 = 48
R_SPARC_LOX10 = 49
R_SPARC_H44 = 50
R_SPARC_M44 = 51
R_SPARC_L44 = 52
R_SPARC_REGISTER = 53
R_SPARC_UA64 = 54
R_SPARC_UA16 = 55
R_SPARC_TLS_GD_HI22 = 56
R_SPARC_TLS_GD_LO10 = 57
R_SPARC_TLS_GD_ADD = 58
R_SPARC_TLS_GD_CALL = 59
R_SPARC_TLS_LDM_HI22 = 60
R_SPARC_TLS_LDM_LO10 = 61
R_SPARC_TLS_LDM_ADD = 62
R_SPARC_TLS_LDM_CALL = 63
R_SPARC_TLS_LDO_HIX22 = 64
R_SPARC_TLS_LDO_LOX10 = 65
R_SPARC_TLS_LDO_ADD = 66
R_SPARC_TLS_IE_HI22 = 67
R_SPARC_TLS_IE_LO10 = 68
R_SPARC_TLS_IE_LD = 69
R_SPARC_TLS_IE_LDX = 70
R_SPARC_TLS_IE_ADD = 71
R_SPARC_TLS_LE_HIX22 = 72
R_SPARC_TLS_LE_LOX10 = 73
R_SPARC_TLS_DTPMOD32 = 74
R_SPARC_TLS_DTPMOD64 = 75
R_SPARC_TLS_DTPOFF32 = 76
R_SPARC_TLS_DTPOFF64 = 77
R_SPARC_TLS_TPOFF32 = 78
R_SPARC_TLS_TPOFF64 = 79
R_SPARC_GOTDATA_HIX22 = 80
R_SPARC_GOTDATA_LOX10 = 81
R_SPARC_GOTDATA_OP_HIX22 = 82
R_SPARC_GOTDATA_OP_LOX10 = 83
R_SPARC_GOTDATA_OP = 84
R_SPARC_H34 = 85
R_SPARC_SIZE32 = 86
R_SPARC_SIZE64 = 87
R_SPARC_WDISP10 = 88
R_SPARC_JMP_IREL = 248
R_SPARC_IRELATIVE = 249
R_SPARC_GNU_VTINHERIT = 250
R_SPARC_GNU_VTENTRY = 251
R_SPARC_REV32 = 252
R_SPARC_NUM = 253
DT_SPARC_REGISTER = 0x70000001
DT_SPARC_NUM = 2
EF_MIPS_NOREORDER = 1
EF_MIPS_PIC = 2
EF_MIPS_CPIC = 4
EF_MIPS_XGOT = 8
EF_MIPS_64BIT_WHIRL = 16
EF_MIPS_ABI2 = 32
EF_MIPS_ABI_ON32 = 64
EF_MIPS_FP64 = 512
EF_MIPS_NAN2008 = 1024
EF_MIPS_ARCH = 0xf0000000
EF_MIPS_ARCH_1 = 0x00000000
EF_MIPS_ARCH_2 = 0x10000000
EF_MIPS_ARCH_3 = 0x20000000
EF_MIPS_ARCH_4 = 0x30000000
EF_MIPS_ARCH_5 = 0x40000000
EF_MIPS_ARCH_32 = 0x50000000
EF_MIPS_ARCH_64 = 0x60000000
EF_MIPS_ARCH_32R2 = 0x70000000
EF_MIPS_ARCH_64R2 = 0x80000000
E_MIPS_ARCH_1 = EF_MIPS_ARCH_1
E_MIPS_ARCH_2 = EF_MIPS_ARCH_2
E_MIPS_ARCH_3 = EF_MIPS_ARCH_3
E_MIPS_ARCH_4 = EF_MIPS_ARCH_4
E_MIPS_ARCH_5 = EF_MIPS_ARCH_5
E_MIPS_ARCH_32 = EF_MIPS_ARCH_32
E_MIPS_ARCH_64 = EF_MIPS_ARCH_64
SHN_MIPS_ACOMMON = 0xff00
SHN_MIPS_TEXT = 0xff01
SHN_MIPS_DATA = 0xff02
SHN_MIPS_SCOMMON = 0xff03
SHN_MIPS_SUNDEFINED = 0xff04
SHT_MIPS_LIBLIST = 0x70000000
SHT_MIPS_MSYM = 0x70000001
SHT_MIPS_CONFLICT = 0x70000002
SHT_MIPS_GPTAB = 0x70000003
SHT_MIPS_UCODE = 0x70000004
SHT_MIPS_DEBUG = 0x70000005
SHT_MIPS_REGINFO = 0x70000006
SHT_MIPS_PACKAGE = 0x70000007
SHT_MIPS_PACKSYM = 0x70000008
SHT_MIPS_RELD = 0x70000009
SHT_MIPS_IFACE = 0x7000000b
SHT_MIPS_CONTENT = 0x7000000c
SHT_MIPS_OPTIONS = 0x7000000d
SHT_MIPS_SHDR = 0x70000010
SHT_MIPS_FDESC = 0x70000011
SHT_MIPS_EXTSYM = 0x70000012
SHT_MIPS_DENSE = 0x70000013
SHT_MIPS_PDESC = 0x70000014
SHT_MIPS_LOCSYM = 0x70000015
SHT_MIPS_AUXSYM = 0x70000016
SHT_MIPS_OPTSYM = 0x70000017
SHT_MIPS_LOCSTR = 0x70000018
SHT_MIPS_LINE = 0x70000019
SHT_MIPS_RFDESC = 0x7000001a
SHT_MIPS_DELTASYM = 0x7000001b
SHT_MIPS_DELTAINST = 0x7000001c
SHT_MIPS_DELTACLASS = 0x7000001d
SHT_MIPS_DWARF = 0x7000001e
SHT_MIPS_DELTADECL = 0x7000001f
SHT_MIPS_SYMBOL_LIB = 0x70000020
SHT_MIPS_EVENTS = 0x70000021
SHT_MIPS_TRANSLATE = 0x70000022
SHT_MIPS_PIXIE = 0x70000023
SHT_MIPS_XLATE = 0x70000024
SHT_MIPS_XLATE_DEBUG = 0x70000025
SHT_MIPS_WHIRL = 0x70000026
SHT_MIPS_EH_REGION = 0x70000027
SHT_MIPS_XLATE_OLD = 0x70000028
SHT_MIPS_PDR_EXCEPTION = 0x70000029
SHT_MIPS_XHASH = 0x7000002b
SHF_MIPS_GPREL = 0x10000000
SHF_MIPS_MERGE = 0x20000000
SHF_MIPS_ADDR = 0x40000000
SHF_MIPS_STRINGS = 0x80000000
SHF_MIPS_NOSTRIP = 0x08000000
SHF_MIPS_LOCAL = 0x04000000
SHF_MIPS_NAMES = 0x02000000
SHF_MIPS_NODUPE = 0x01000000
STO_MIPS_DEFAULT = 0x0
STO_MIPS_INTERNAL = 0x1
STO_MIPS_HIDDEN = 0x2
STO_MIPS_PROTECTED = 0x3
STO_MIPS_PLT = 0x8
STO_MIPS_SC_ALIGN_UNUSED = 0xff
STB_MIPS_SPLIT_COMMON = 13
ODK_NULL = 0
ODK_REGINFO = 1
ODK_EXCEPTIONS = 2
ODK_PAD = 3
ODK_HWPATCH = 4
ODK_FILL = 5
ODK_TAGS = 6
ODK_HWAND = 7
ODK_HWOR = 8
OEX_FPU_MIN = 0x1f
OEX_FPU_MAX = 0x1f00
OEX_PAGE0 = 0x10000
OEX_SMM = 0x20000
OEX_FPDBUG = 0x40000
OEX_PRECISEFP = OEX_FPDBUG
OEX_DISMISS = 0x80000
OEX_FPU_INVAL = 0x10
OEX_FPU_DIV0 = 0x08
OEX_FPU_OFLO = 0x04
OEX_FPU_UFLO = 0x02
OEX_FPU_INEX = 0x01
OHW_R4KEOP = 0x1
OHW_R8KPFETCH = 0x2
OHW_R5KEOP = 0x4
OHW_R5KCVTL = 0x8
OPAD_PREFIX = 0x1
OPAD_POSTFIX = 0x2
OPAD_SYMBOL = 0x4
OHWA0_R4KEOP_CHECKED = 0x00000001
OHWA1_R4KEOP_CLEAN = 0x00000002
R_MIPS_NONE = 0
R_MIPS_16 = 1
R_MIPS_32 = 2
R_MIPS_REL32 = 3
R_MIPS_26 = 4
R_MIPS_HI16 = 5
R_MIPS_LO16 = 6
R_MIPS_GPREL16 = 7
R_MIPS_LITERAL = 8
R_MIPS_GOT16 = 9
R_MIPS_PC16 = 10
R_MIPS_CALL16 = 11
R_MIPS_GPREL32 = 12
R_MIPS_SHIFT5 = 16
R_MIPS_SHIFT6 = 17
R_MIPS_64 = 18
R_MIPS_GOT_DISP = 19
R_MIPS_GOT_PAGE = 20
R_MIPS_GOT_OFST = 21
R_MIPS_GOT_HI16 = 22
R_MIPS_GOT_LO16 = 23
R_MIPS_SUB = 24
R_MIPS_INSERT_A = 25
R_MIPS_INSERT_B = 26
R_MIPS_DELETE = 27
R_MIPS_HIGHER = 28
R_MIPS_HIGHEST = 29
R_MIPS_CALL_HI16 = 30
R_MIPS_CALL_LO16 = 31
R_MIPS_SCN_DISP = 32
R_MIPS_REL16 = 33
R_MIPS_ADD_IMMEDIATE = 34
R_MIPS_PJUMP = 35
R_MIPS_RELGOT = 36
R_MIPS_JALR = 37
R_MIPS_TLS_DTPMOD32 = 38
R_MIPS_TLS_DTPREL32 = 39
R_MIPS_TLS_DTPMOD64 = 40
R_MIPS_TLS_DTPREL64 = 41
R_MIPS_TLS_GD = 42
R_MIPS_TLS_LDM = 43
R_MIPS_TLS_DTPREL_HI16 = 44
R_MIPS_TLS_DTPREL_LO16 = 45
R_MIPS_TLS_GOTTPREL = 46
R_MIPS_TLS_TPREL32 = 47
R_MIPS_TLS_TPREL64 = 48
R_MIPS_TLS_TPREL_HI16 = 49
R_MIPS_TLS_TPREL_LO16 = 50
R_MIPS_GLOB_DAT = 51
R_MIPS_COPY = 126
R_MIPS_JUMP_SLOT = 127
R_MIPS_NUM = 128
PT_MIPS_REGINFO = 0x70000000
PT_MIPS_RTPROC = 0x70000001
PT_MIPS_OPTIONS = 0x70000002
PT_MIPS_ABIFLAGS = 0x70000003
PF_MIPS_LOCAL = 0x10000000
DT_MIPS_RLD_VERSION = 0x70000001
DT_MIPS_TIME_STAMP = 0x70000002
DT_MIPS_ICHECKSUM = 0x70000003
DT_MIPS_IVERSION = 0x70000004
DT_MIPS_FLAGS = 0x70000005
DT_MIPS_BASE_ADDRESS = 0x70000006
DT_MIPS_MSYM = 0x70000007
DT_MIPS_CONFLICT = 0x70000008
DT_MIPS_LIBLIST = 0x70000009
DT_MIPS_LOCAL_GOTNO = 0x7000000a
DT_MIPS_CONFLICTNO = 0x7000000b
DT_MIPS_LIBLISTNO = 0x70000010
DT_MIPS_SYMTABNO = 0x70000011
DT_MIPS_UNREFEXTNO = 0x70000012
DT_MIPS_GOTSYM = 0x70000013
DT_MIPS_HIPAGENO = 0x70000014
DT_MIPS_RLD_MAP = 0x70000016
DT_MIPS_DELTA_CLASS = 0x70000017
DT_MIPS_DELTA_INSTANCE = 0x70000019
DT_MIPS_DELTA_RELOC = 0x7000001b
DT_MIPS_CXX_FLAGS = 0x70000022
DT_MIPS_PIXIE_INIT = 0x70000023
DT_MIPS_SYMBOL_LIB = 0x70000024
DT_MIPS_LOCALPAGE_GOTIDX = 0x70000025
DT_MIPS_LOCAL_GOTIDX = 0x70000026
DT_MIPS_HIDDEN_GOTIDX = 0x70000027
DT_MIPS_PROTECTED_GOTIDX = 0x70000028
DT_MIPS_OPTIONS = 0x70000029
DT_MIPS_INTERFACE = 0x7000002a
DT_MIPS_DYNSTR_ALIGN = 0x7000002b
DT_MIPS_INTERFACE_SIZE = 0x7000002c
DT_MIPS_COMPACT_SIZE = 0x7000002f
DT_MIPS_GP_VALUE = 0x70000030
DT_MIPS_AUX_DYNAMIC = 0x70000031
DT_MIPS_PLTGOT = 0x70000032
DT_MIPS_RWPLT = 0x70000034
DT_MIPS_RLD_MAP_REL = 0x70000035
DT_MIPS_XHASH = 0x70000036
RHF_NONE = 0
RHF_QUICKSTART = (1 << 0)
RHF_NOTPOT = (1 << 1)
RHF_NO_LIBRARY_REPLACEMENT = (1 << 2)
RHF_NO_MOVE = (1 << 3)
RHF_SGI_ONLY = (1 << 4)
RHF_GUARANTEE_INIT = (1 << 5)
RHF_DELTA_C_PLUS_PLUS = (1 << 6)
RHF_GUARANTEE_START_INIT = (1 << 7)
RHF_PIXIE = (1 << 8)
RHF_DEFAULT_DELAY_LOAD = (1 << 9)
RHF_REQUICKSTART = (1 << 10)
RHF_REQUICKSTARTED = (1 << 11)
RHF_CORD = (1 << 12)
RHF_NO_UNRES_UNDEF = (1 << 13)
RHF_RLD_ORDER_SAFE = (1 << 14)
LL_NONE = 0
LL_EXACT_MATCH = (1 << 0)
LL_IGNORE_INT_VER = (1 << 1)
LL_REQUIRE_MINOR = (1 << 2)
LL_EXPORTS = (1 << 3)
LL_DELAY_LOAD = (1 << 4)
LL_DELTA = (1 << 5)
MIPS_AFL_REG_NONE = 0x00
MIPS_AFL_REG_32 = 0x01
MIPS_AFL_REG_64 = 0x02
MIPS_AFL_REG_128 = 0x03
MIPS_AFL_ASE_DSP = 0x00000001
MIPS_AFL_ASE_DSPR2 = 0x00000002
MIPS_AFL_ASE_EVA = 0x00000004
MIPS_AFL_ASE_MCU = 0x00000008
MIPS_AFL_ASE_MDMX = 0x00000010
MIPS_AFL_ASE_MIPS3D = 0x00000020
MIPS_AFL_ASE_MT = 0x00000040
MIPS_AFL_ASE_SMARTMIPS = 0x00000080
MIPS_AFL_ASE_VIRT = 0x00000100
MIPS_AFL_ASE_MSA = 0x00000200
MIPS_AFL_ASE_MIPS16 = 0x00000400
MIPS_AFL_ASE_MICROMIPS = 0x00000800
MIPS_AFL_ASE_XPA = 0x00001000
MIPS_AFL_ASE_MASK = 0x00001fff
MIPS_AFL_EXT_XLR = 1
MIPS_AFL_EXT_OCTEON2 = 2
MIPS_AFL_EXT_OCTEONP = 3
MIPS_AFL_EXT_LOONGSON_3A = 4
MIPS_AFL_EXT_OCTEON = 5
MIPS_AFL_EXT_5900 = 6
MIPS_AFL_EXT_4650 = 7
MIPS_AFL_EXT_4010 = 8
MIPS_AFL_EXT_4100 = 9
MIPS_AFL_EXT_3900 = 10
MIPS_AFL_EXT_10000 = 11
MIPS_AFL_EXT_SB1 = 12
MIPS_AFL_EXT_4111 = 13
MIPS_AFL_EXT_4120 = 14
MIPS_AFL_EXT_5400 = 15
MIPS_AFL_EXT_5500 = 16
MIPS_AFL_EXT_LOONGSON_2E = 17
MIPS_AFL_EXT_LOONGSON_2F = 18
MIPS_AFL_FLAGS1_ODDSPREG = 1
EF_PARISC_TRAPNIL = 0x00010000
EF_PARISC_EXT = 0x00020000
EF_PARISC_LSB = 0x00040000
EF_PARISC_WIDE = 0x00080000
EF_PARISC_LAZYSWAP = 0x00400000
EF_PARISC_ARCH = 0x0000ffff
EFA_PARISC_1_0 = 0x020b
EFA_PARISC_1_1 = 0x0210
EFA_PARISC_2_0 = 0x0214
SHN_PARISC_HUGE_COMMON = 0xff01
SHT_PARISC_EXT = 0x70000000
SHT_PARISC_UNWIND = 0x70000001
SHT_PARISC_DOC = 0x70000002
SHF_PARISC_SHORT = 0x20000000
SHF_PARISC_HUGE = 0x40000000
SHF_PARISC_SBP = 0x80000000
STT_PARISC_MILLICODE = 13
STT_HP_OPAQUE = (STT_LOOS + 0x1)
STT_HP_STUB = (STT_LOOS + 0x2)
R_PARISC_NONE = 0
R_PARISC_DIR32 = 1
R_PARISC_DIR21L = 2
R_PARISC_DIR17R = 3
R_PARISC_DIR17F = 4
R_PARISC_DIR14R = 6
R_PARISC_PCREL32 = 9
R_PARISC_PCREL21L = 10
R_PARISC_PCREL17R = 11
R_PARISC_PCREL17F = 12
R_PARISC_PCREL14R = 14
R_PARISC_DPREL21L = 18
R_PARISC_DPREL14R = 22
R_PARISC_GPREL21L = 26
R_PARISC_GPREL14R = 30
R_PARISC_LTOFF21L = 34
R_PARISC_LTOFF14R = 38
R_PARISC_SECREL32 = 41
R_PARISC_SEGBASE = 48
R_PARISC_SEGREL32 = 49
R_PARISC_PLTOFF21L = 50
R_PARISC_PLTOFF14R = 54
R_PARISC_LTOFF_FPTR32 = 57
R_PARISC_LTOFF_FPTR21L = 58
R_PARISC_LTOFF_FPTR14R = 62
R_PARISC_FPTR64 = 64
R_PARISC_PLABEL32 = 65
R_PARISC_PLABEL21L = 66
R_PARISC_PLABEL14R = 70
R_PARISC_PCREL64 = 72
R_PARISC_PCREL22F = 74
R_PARISC_PCREL14WR = 75
R_PARISC_PCREL14DR = 76
R_PARISC_PCREL16F = 77
R_PARISC_PCREL16WF = 78
R_PARISC_PCREL16DF = 79
R_PARISC_DIR64 = 80
R_PARISC_DIR14WR = 83
R_PARISC_DIR14DR = 84
R_PARISC_DIR16F = 85
R_PARISC_DIR16WF = 86
R_PARISC_DIR16DF = 87
R_PARISC_GPREL64 = 88
R_PARISC_GPREL14WR = 91
R_PARISC_GPREL14DR = 92
R_PARISC_GPREL16F = 93
R_PARISC_GPREL16WF = 94
R_PARISC_GPREL16DF = 95
R_PARISC_LTOFF64 = 96
R_PARISC_LTOFF14WR = 99
R_PARISC_LTOFF14DR = 100
R_PARISC_LTOFF16F = 101
R_PARISC_LTOFF16WF = 102
R_PARISC_LTOFF16DF = 103
R_PARISC_SECREL64 = 104
R_PARISC_SEGREL64 = 112
R_PARISC_PLTOFF14WR = 115
R_PARISC_PLTOFF14DR = 116
R_PARISC_PLTOFF16F = 117
R_PARISC_PLTOFF16WF = 118
R_PARISC_PLTOFF16DF = 119
R_PARISC_LTOFF_FPTR64 = 120
R_PARISC_LTOFF_FPTR14WR = 123
R_PARISC_LTOFF_FPTR14DR = 124
R_PARISC_LTOFF_FPTR16F = 125
R_PARISC_LTOFF_FPTR16WF = 126
R_PARISC_LTOFF_FPTR16DF = 127
R_PARISC_LORESERVE = 128
R_PARISC_COPY = 128
R_PARISC_IPLT = 129
R_PARISC_EPLT = 130
R_PARISC_TPREL32 = 153
R_PARISC_TPREL21L = 154
R_PARISC_TPREL14R = 158
R_PARISC_LTOFF_TP21L = 162
R_PARISC_LTOFF_TP14R = 166
R_PARISC_LTOFF_TP14F = 167
R_PARISC_TPREL64 = 216
R_PARISC_TPREL14WR = 219
R_PARISC_TPREL14DR = 220
R_PARISC_TPREL16F = 221
R_PARISC_TPREL16WF = 222
R_PARISC_TPREL16DF = 223
R_PARISC_LTOFF_TP64 = 224
R_PARISC_LTOFF_TP14WR = 227
R_PARISC_LTOFF_TP14DR = 228
R_PARISC_LTOFF_TP16F = 229
R_PARISC_LTOFF_TP16WF = 230
R_PARISC_LTOFF_TP16DF = 231
R_PARISC_GNU_VTENTRY = 232
R_PARISC_GNU_VTINHERIT = 233
R_PARISC_TLS_GD21L = 234
R_PARISC_TLS_GD14R = 235
R_PARISC_TLS_GDCALL = 236
R_PARISC_TLS_LDM21L = 237
R_PARISC_TLS_LDM14R = 238
R_PARISC_TLS_LDMCALL = 239
R_PARISC_TLS_LDO21L = 240
R_PARISC_TLS_LDO14R = 241
R_PARISC_TLS_DTPMOD32 = 242
R_PARISC_TLS_DTPMOD64 = 243
R_PARISC_TLS_DTPOFF32 = 244
R_PARISC_TLS_DTPOFF64 = 245
R_PARISC_TLS_LE21L = R_PARISC_TPREL21L
R_PARISC_TLS_LE14R = R_PARISC_TPREL14R
R_PARISC_TLS_IE21L = R_PARISC_LTOFF_TP21L
R_PARISC_TLS_IE14R = R_PARISC_LTOFF_TP14R
R_PARISC_TLS_TPREL32 = R_PARISC_TPREL32
R_PARISC_TLS_TPREL64 = R_PARISC_TPREL64
R_PARISC_HIRESERVE = 255
PT_HP_TLS = (PT_LOOS + 0x0)
PT_HP_CORE_NONE = (PT_LOOS + 0x1)
PT_HP_CORE_VERSION = (PT_LOOS + 0x2)
PT_HP_CORE_KERNEL = (PT_LOOS + 0x3)
PT_HP_CORE_COMM = (PT_LOOS + 0x4)
PT_HP_CORE_PROC = (PT_LOOS + 0x5)
PT_HP_CORE_LOADABLE = (PT_LOOS + 0x6)
PT_HP_CORE_STACK = (PT_LOOS + 0x7)
PT_HP_CORE_SHM = (PT_LOOS + 0x8)
PT_HP_CORE_MMF = (PT_LOOS + 0x9)
PT_HP_PARALLEL = (PT_LOOS + 0x10)
PT_HP_FASTBIND = (PT_LOOS + 0x11)
PT_HP_OPT_ANNOT = (PT_LOOS + 0x12)
PT_HP_HSL_ANNOT = (PT_LOOS + 0x13)
PT_HP_STACK = (PT_LOOS + 0x14)
PT_PARISC_ARCHEXT = 0x70000000
PT_PARISC_UNWIND = 0x70000001
PF_PARISC_SBP = 0x08000000
PF_HP_PAGE_SIZE = 0x00100000
PF_HP_FAR_SHARED = 0x00200000
PF_HP_NEAR_SHARED = 0x00400000
PF_HP_CODE = 0x01000000
PF_HP_MODIFY = 0x02000000
PF_HP_LAZYSWAP = 0x04000000
PF_HP_SBP = 0x08000000
EF_ALPHA_32BIT = 1
EF_ALPHA_CANRELAX = 2
SHT_ALPHA_DEBUG = 0x70000001
SHT_ALPHA_REGINFO = 0x70000002
SHF_ALPHA_GPREL = 0x10000000
STO_ALPHA_NOPV = 0x80
STO_ALPHA_STD_GPLOAD = 0x88
R_ALPHA_NONE = 0
R_ALPHA_REFLONG = 1
R_ALPHA_REFQUAD = 2
R_ALPHA_GPREL32 = 3
R_ALPHA_LITERAL = 4
R_ALPHA_LITUSE = 5
R_ALPHA_GPDISP = 6
R_ALPHA_BRADDR = 7
R_ALPHA_HINT = 8
R_ALPHA_SREL16 = 9
R_ALPHA_SREL32 = 10
R_ALPHA_SREL64 = 11
R_ALPHA_GPRELHIGH = 17
R_ALPHA_GPRELLOW = 18
R_ALPHA_GPREL16 = 19
R_ALPHA_COPY = 24
R_ALPHA_GLOB_DAT = 25
R_ALPHA_JMP_SLOT = 26
R_ALPHA_RELATIVE = 27
R_ALPHA_TLS_GD_HI = 28
R_ALPHA_TLSGD = 29
R_ALPHA_TLS_LDM = 30
R_ALPHA_DTPMOD64 = 31
R_ALPHA_GOTDTPREL = 32
R_ALPHA_DTPREL64 = 33
R_ALPHA_DTPRELHI = 34
R_ALPHA_DTPRELLO = 35
R_ALPHA_DTPREL16 = 36
R_ALPHA_GOTTPREL = 37
R_ALPHA_TPREL64 = 38
R_ALPHA_TPRELHI = 39
R_ALPHA_TPRELLO = 40
R_ALPHA_TPREL16 = 41
R_ALPHA_NUM = 46
LITUSE_ALPHA_ADDR = 0
LITUSE_ALPHA_BASE = 1
LITUSE_ALPHA_BYTOFF = 2
LITUSE_ALPHA_JSR = 3
LITUSE_ALPHA_TLS_GD = 4
LITUSE_ALPHA_TLS_LDM = 5
DT_ALPHA_PLTRO = (DT_LOPROC + 0)
DT_ALPHA_NUM = 1
EF_PPC_EMB = 0x80000000
EF_PPC_RELOCATABLE = 0x00010000
R_PPC_NONE = 0
R_PPC_ADDR32 = 1
R_PPC_ADDR24 = 2
R_PPC_ADDR16 = 3
R_PPC_ADDR16_LO = 4
R_PPC_ADDR16_HI = 5
R_PPC_ADDR16_HA = 6
R_PPC_ADDR14 = 7
R_PPC_ADDR14_BRTAKEN = 8
R_PPC_ADDR14_BRNTAKEN = 9
R_PPC_REL24 = 10
R_PPC_REL14 = 11
R_PPC_REL14_BRTAKEN = 12
R_PPC_REL14_BRNTAKEN = 13
R_PPC_GOT16 = 14
R_PPC_GOT16_LO = 15
R_PPC_GOT16_HI = 16
R_PPC_GOT16_HA = 17
R_PPC_PLTREL24 = 18
R_PPC_COPY = 19
R_PPC_GLOB_DAT = 20
R_PPC_JMP_SLOT = 21
R_PPC_RELATIVE = 22
R_PPC_LOCAL24PC = 23
R_PPC_UADDR32 = 24
R_PPC_UADDR16 = 25
R_PPC_REL32 = 26
R_PPC_PLT32 = 27
R_PPC_PLTREL32 = 28
R_PPC_PLT16_LO = 29
R_PPC_PLT16_HI = 30
R_PPC_PLT16_HA = 31
R_PPC_SDAREL16 = 32
R_PPC_SECTOFF = 33
R_PPC_SECTOFF_LO = 34
R_PPC_SECTOFF_HI = 35
R_PPC_SECTOFF_HA = 36
R_PPC_TLS = 67
R_PPC_DTPMOD32 = 68
R_PPC_TPREL16 = 69
R_PPC_TPREL16_LO = 70
R_PPC_TPREL16_HI = 71
R_PPC_TPREL16_HA = 72
R_PPC_TPREL32 = 73
R_PPC_DTPREL16 = 74
R_PPC_DTPREL16_LO = 75
R_PPC_DTPREL16_HI = 76
R_PPC_DTPREL16_HA = 77
R_PPC_DTPREL32 = 78
R_PPC_GOT_TLSGD16 = 79
R_PPC_GOT_TLSGD16_LO = 80
R_PPC_GOT_TLSGD16_HI = 81
R_PPC_GOT_TLSGD16_HA = 82
R_PPC_GOT_TLSLD16 = 83
R_PPC_GOT_TLSLD16_LO = 84
R_PPC_GOT_TLSLD16_HI = 85
R_PPC_GOT_TLSLD16_HA = 86
R_PPC_GOT_TPREL16 = 87
R_PPC_GOT_TPREL16_LO = 88
R_PPC_GOT_TPREL16_HI = 89
R_PPC_GOT_TPREL16_HA = 90
R_PPC_GOT_DTPREL16 = 91
R_PPC_GOT_DTPREL16_LO = 92
R_PPC_GOT_DTPREL16_HI = 93
R_PPC_GOT_DTPREL16_HA = 94
R_PPC_TLSGD = 95
R_PPC_TLSLD = 96
R_PPC_EMB_NADDR32 = 101
R_PPC_EMB_NADDR16 = 102
R_PPC_EMB_NADDR16_LO = 103
R_PPC_EMB_NADDR16_HI = 104
R_PPC_EMB_NADDR16_HA = 105
R_PPC_EMB_SDAI16 = 106
R_PPC_EMB_SDA2I16 = 107
R_PPC_EMB_SDA2REL = 108
R_PPC_EMB_SDA21 = 109
R_PPC_EMB_MRKREF = 110
R_PPC_EMB_RELSEC16 = 111
R_PPC_EMB_RELST_LO = 112
R_PPC_EMB_RELST_HI = 113
R_PPC_EMB_RELST_HA = 114
R_PPC_EMB_BIT_FLD = 115
R_PPC_EMB_RELSDA = 116
R_PPC_DIAB_SDA21_LO = 180
R_PPC_DIAB_SDA21_HI = 181
R_PPC_DIAB_SDA21_HA = 182
R_PPC_DIAB_RELSDA_LO = 183
R_PPC_DIAB_RELSDA_HI = 184
R_PPC_DIAB_RELSDA_HA = 185
R_PPC_IRELATIVE = 248
R_PPC_REL16 = 249
R_PPC_REL16_LO = 250
R_PPC_REL16_HI = 251
R_PPC_REL16_HA = 252
R_PPC_TOC16 = 255
DT_PPC_GOT = (DT_LOPROC + 0)
DT_PPC_OPT = (DT_LOPROC + 1)
DT_PPC_NUM = 2
PPC_OPT_TLS = 1
R_PPC64_NONE = R_PPC_NONE
R_PPC64_ADDR32 = R_PPC_ADDR32
R_PPC64_ADDR24 = R_PPC_ADDR24
R_PPC64_ADDR16 = R_PPC_ADDR16
R_PPC64_ADDR16_LO = R_PPC_ADDR16_LO
R_PPC64_ADDR16_HI = R_PPC_ADDR16_HI
R_PPC64_ADDR16_HA = R_PPC_ADDR16_HA
R_PPC64_ADDR14 = R_PPC_ADDR14
R_PPC64_ADDR14_BRTAKEN = R_PPC_ADDR14_BRTAKEN
R_PPC64_ADDR14_BRNTAKEN = R_PPC_ADDR14_BRNTAKEN
R_PPC64_REL24 = R_PPC_REL24
R_PPC64_REL14 = R_PPC_REL14
R_PPC64_REL14_BRTAKEN = R_PPC_REL14_BRTAKEN
R_PPC64_REL14_BRNTAKEN = R_PPC_REL14_BRNTAKEN
R_PPC64_GOT16 = R_PPC_GOT16
R_PPC64_GOT16_LO = R_PPC_GOT16_LO
R_PPC64_GOT16_HI = R_PPC_GOT16_HI
R_PPC64_GOT16_HA = R_PPC_GOT16_HA
R_PPC64_COPY = R_PPC_COPY
R_PPC64_GLOB_DAT = R_PPC_GLOB_DAT
R_PPC64_JMP_SLOT = R_PPC_JMP_SLOT
R_PPC64_RELATIVE = R_PPC_RELATIVE
R_PPC64_UADDR32 = R_PPC_UADDR32
R_PPC64_UADDR16 = R_PPC_UADDR16
R_PPC64_REL32 = R_PPC_REL32
R_PPC64_PLT32 = R_PPC_PLT32
R_PPC64_PLTREL32 = R_PPC_PLTREL32
R_PPC64_PLT16_LO = R_PPC_PLT16_LO
R_PPC64_PLT16_HI = R_PPC_PLT16_HI
R_PPC64_PLT16_HA = R_PPC_PLT16_HA
R_PPC64_SECTOFF = R_PPC_SECTOFF
R_PPC64_SECTOFF_LO = R_PPC_SECTOFF_LO
R_PPC64_SECTOFF_HI = R_PPC_SECTOFF_HI
R_PPC64_SECTOFF_HA = R_PPC_SECTOFF_HA
R_PPC64_ADDR30 = 37
R_PPC64_ADDR64 = 38
R_PPC64_ADDR16_HIGHER = 39
R_PPC64_ADDR16_HIGHERA = 40
R_PPC64_ADDR16_HIGHEST = 41
R_PPC64_ADDR16_HIGHESTA = 42
R_PPC64_UADDR64 = 43
R_PPC64_REL64 = 44
R_PPC64_PLT64 = 45
R_PPC64_PLTREL64 = 46
R_PPC64_TOC16 = 47
R_PPC64_TOC16_LO = 48
R_PPC64_TOC16_HI = 49
R_PPC64_TOC16_HA = 50
R_PPC64_TOC = 51
R_PPC64_PLTGOT16 = 52
R_PPC64_PLTGOT16_LO = 53
R_PPC64_PLTGOT16_HI = 54
R_PPC64_PLTGOT16_HA = 55
R_PPC64_ADDR16_DS = 56
R_PPC64_ADDR16_LO_DS = 57
R_PPC64_GOT16_DS = 58
R_PPC64_GOT16_LO_DS = 59
R_PPC64_PLT16_LO_DS = 60
R_PPC64_SECTOFF_DS = 61
R_PPC64_SECTOFF_LO_DS = 62
R_PPC64_TOC16_DS = 63
R_PPC64_TOC16_LO_DS = 64
R_PPC64_PLTGOT16_DS = 65
R_PPC64_PLTGOT16_LO_DS = 66
R_PPC64_TLS = 67
R_PPC64_DTPMOD64 = 68
R_PPC64_TPREL16 = 69
R_PPC64_TPREL16_LO = 70
R_PPC64_TPREL16_HI = 71
R_PPC64_TPREL16_HA = 72
R_PPC64_TPREL64 = 73
R_PPC64_DTPREL16 = 74
R_PPC64_DTPREL16_LO = 75
R_PPC64_DTPREL16_HI = 76
R_PPC64_DTPREL16_HA = 77
R_PPC64_DTPREL64 = 78
R_PPC64_GOT_TLSGD16 = 79
R_PPC64_GOT_TLSGD16_LO = 80
R_PPC64_GOT_TLSGD16_HI = 81
R_PPC64_GOT_TLSGD16_HA = 82
R_PPC64_GOT_TLSLD16 = 83
R_PPC64_GOT_TLSLD16_LO = 84
R_PPC64_GOT_TLSLD16_HI = 85
R_PPC64_GOT_TLSLD16_HA = 86
R_PPC64_GOT_TPREL16_DS = 87
R_PPC64_GOT_TPREL16_LO_DS = 88
R_PPC64_GOT_TPREL16_HI = 89
R_PPC64_GOT_TPREL16_HA = 90
R_PPC64_GOT_DTPREL16_DS = 91
R_PPC64_GOT_DTPREL16_LO_DS = 92
R_PPC64_GOT_DTPREL16_HI = 93
R_PPC64_GOT_DTPREL16_HA = 94
R_PPC64_TPREL16_DS = 95
R_PPC64_TPREL16_LO_DS = 96
R_PPC64_TPREL16_HIGHER = 97
R_PPC64_TPREL16_HIGHERA = 98
R_PPC64_TPREL16_HIGHEST = 99
R_PPC64_TPREL16_HIGHESTA = 100
R_PPC64_DTPREL16_DS = 101
R_PPC64_DTPREL16_LO_DS = 102
R_PPC64_DTPREL16_HIGHER = 103
R_PPC64_DTPREL16_HIGHERA = 104
R_PPC64_DTPREL16_HIGHEST = 105
R_PPC64_DTPREL16_HIGHESTA = 106
R_PPC64_TLSGD = 107
R_PPC64_TLSLD = 108
R_PPC64_TOCSAVE = 109
R_PPC64_ADDR16_HIGH = 110
R_PPC64_ADDR16_HIGHA = 111
R_PPC64_TPREL16_HIGH = 112
R_PPC64_TPREL16_HIGHA = 113
R_PPC64_DTPREL16_HIGH = 114
R_PPC64_DTPREL16_HIGHA = 115
R_PPC64_JMP_IREL = 247
R_PPC64_IRELATIVE = 248
R_PPC64_REL16 = 249
R_PPC64_REL16_LO = 250
R_PPC64_REL16_HI = 251
R_PPC64_REL16_HA = 252
EF_PPC64_ABI = 3
DT_PPC64_GLINK = (DT_LOPROC + 0)
DT_PPC64_OPD = (DT_LOPROC + 1)
DT_PPC64_OPDSZ = (DT_LOPROC + 2)
DT_PPC64_OPT = (DT_LOPROC + 3)
DT_PPC64_NUM = 4
PPC64_OPT_TLS = 1
PPC64_OPT_MULTI_TOC = 2
PPC64_OPT_LOCALENTRY = 4
STO_PPC64_LOCAL_BIT = 5
STO_PPC64_LOCAL_MASK = (7 << STO_PPC64_LOCAL_BIT)
EF_ARM_RELEXEC = 0x01
EF_ARM_HASENTRY = 0x02
EF_ARM_INTERWORK = 0x04
EF_ARM_APCS_26 = 0x08
EF_ARM_APCS_FLOAT = 0x10
EF_ARM_PIC = 0x20
EF_ARM_ALIGN8 = 0x40
EF_ARM_NEW_ABI = 0x80
EF_ARM_OLD_ABI = 0x100
EF_ARM_SOFT_FLOAT = 0x200
EF_ARM_VFP_FLOAT = 0x400
EF_ARM_MAVERICK_FLOAT = 0x800
EF_ARM_ABI_FLOAT_SOFT = 0x200
EF_ARM_ABI_FLOAT_HARD = 0x400
EF_ARM_SYMSARESORTED = 0x04
EF_ARM_DYNSYMSUSESEGIDX = 0x08
EF_ARM_MAPSYMSFIRST = 0x10
EF_ARM_EABIMASK = 0XFF000000
EF_ARM_BE8 = 0x00800000
EF_ARM_LE8 = 0x00400000
def EF_ARM_EABI_VERSION(flags): return ((flags) & EF_ARM_EABIMASK)
EF_ARM_EABI_UNKNOWN = 0x00000000
EF_ARM_EABI_VER1 = 0x01000000
EF_ARM_EABI_VER2 = 0x02000000
EF_ARM_EABI_VER3 = 0x03000000
EF_ARM_EABI_VER4 = 0x04000000
EF_ARM_EABI_VER5 = 0x05000000
STT_ARM_TFUNC = STT_LOPROC
STT_ARM_16BIT = STT_HIPROC
SHF_ARM_ENTRYSECT = 0x10000000
PF_ARM_PI = 0x20000000
PF_ARM_ABS = 0x40000000
PT_ARM_EXIDX = (PT_LOPROC + 1)
SHT_ARM_EXIDX = (SHT_LOPROC + 1)
SHT_ARM_PREEMPTMAP = (SHT_LOPROC + 2)
SHT_ARM_ATTRIBUTES = (SHT_LOPROC + 3)
R_AARCH64_NONE = 0
R_AARCH64_P32_ABS32 = 1
R_AARCH64_P32_COPY = 180
R_AARCH64_P32_GLOB_DAT = 181
R_AARCH64_P32_JUMP_SLOT = 182
R_AARCH64_P32_RELATIVE = 183
R_AARCH64_P32_TLS_DTPMOD = 184
R_AARCH64_P32_TLS_DTPREL = 185
R_AARCH64_P32_TLS_TPREL = 186
R_AARCH64_P32_TLSDESC = 187
R_AARCH64_P32_IRELATIVE = 188
R_AARCH64_ABS64 = 257
R_AARCH64_ABS32 = 258
R_AARCH64_ABS16 = 259
R_AARCH64_PREL64 = 260
R_AARCH64_PREL32 = 261
R_AARCH64_PREL16 = 262
R_AARCH64_MOVW_UABS_G0 = 263
R_AARCH64_MOVW_UABS_G0_NC = 264
R_AARCH64_MOVW_UABS_G1 = 265
R_AARCH64_MOVW_UABS_G1_NC = 266
R_AARCH64_MOVW_UABS_G2 = 267
R_AARCH64_MOVW_UABS_G2_NC = 268
R_AARCH64_MOVW_UABS_G3 = 269
R_AARCH64_MOVW_SABS_G0 = 270
R_AARCH64_MOVW_SABS_G1 = 271
R_AARCH64_MOVW_SABS_G2 = 272
R_AARCH64_LD_PREL_LO19 = 273
R_AARCH64_ADR_PREL_LO21 = 274
R_AARCH64_ADR_PREL_PG_HI21 = 275
R_AARCH64_ADR_PREL_PG_HI21_NC = 276
R_AARCH64_ADD_ABS_LO12_NC = 277
R_AARCH64_LDST8_ABS_LO12_NC = 278
R_AARCH64_TSTBR14 = 279
R_AARCH64_CONDBR19 = 280
R_AARCH64_JUMP26 = 282
R_AARCH64_CALL26 = 283
R_AARCH64_LDST16_ABS_LO12_NC = 284
R_AARCH64_LDST32_ABS_LO12_NC = 285
R_AARCH64_LDST64_ABS_LO12_NC = 286
R_AARCH64_MOVW_PREL_G0 = 287
R_AARCH64_MOVW_PREL_G0_NC = 288
R_AARCH64_MOVW_PREL_G1 = 289
R_AARCH64_MOVW_PREL_G1_NC = 290
R_AARCH64_MOVW_PREL_G2 = 291
R_AARCH64_MOVW_PREL_G2_NC = 292
R_AARCH64_MOVW_PREL_G3 = 293
R_AARCH64_LDST128_ABS_LO12_NC = 299
R_AARCH64_MOVW_GOTOFF_G0 = 300
R_AARCH64_MOVW_GOTOFF_G0_NC = 301
R_AARCH64_MOVW_GOTOFF_G1 = 302
R_AARCH64_MOVW_GOTOFF_G1_NC = 303
R_AARCH64_MOVW_GOTOFF_G2 = 304
R_AARCH64_MOVW_GOTOFF_G2_NC = 305
R_AARCH64_MOVW_GOTOFF_G3 = 306
R_AARCH64_GOTREL64 = 307
R_AARCH64_GOTREL32 = 308
R_AARCH64_GOT_LD_PREL19 = 309
R_AARCH64_LD64_GOTOFF_LO15 = 310
R_AARCH64_ADR_GOT_PAGE = 311
R_AARCH64_LD64_GOT_LO12_NC = 312
R_AARCH64_LD64_GOTPAGE_LO15 = 313
R_AARCH64_TLSGD_ADR_PREL21 = 512
R_AARCH64_TLSGD_ADR_PAGE21 = 513
R_AARCH64_TLSGD_ADD_LO12_NC = 514
R_AARCH64_TLSGD_MOVW_G1 = 515
R_AARCH64_TLSGD_MOVW_G0_NC = 516
R_AARCH64_TLSLD_ADR_PREL21 = 517
R_AARCH64_TLSLD_ADR_PAGE21 = 518
R_AARCH64_TLSLD_ADD_LO12_NC = 519
R_AARCH64_TLSLD_MOVW_G1 = 520
R_AARCH64_TLSLD_MOVW_G0_NC = 521
R_AARCH64_TLSLD_LD_PREL19 = 522
R_AARCH64_TLSLD_MOVW_DTPREL_G2 = 523
R_AARCH64_TLSLD_MOVW_DTPREL_G1 = 524
R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC = 525
R_AARCH64_TLSLD_MOVW_DTPREL_G0 = 526
R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC = 527
R_AARCH64_TLSLD_ADD_DTPREL_HI12 = 528
R_AARCH64_TLSLD_ADD_DTPREL_LO12 = 529
R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC = 530
R_AARCH64_TLSLD_LDST8_DTPREL_LO12 = 531
R_AARCH64_TLSLD_LDST8_DTPREL_LO12_NC = 532
R_AARCH64_TLSLD_LDST16_DTPREL_LO12 = 533
R_AARCH64_TLSLD_LDST16_DTPREL_LO12_NC = 534
R_AARCH64_TLSLD_LDST32_DTPREL_LO12 = 535
R_AARCH64_TLSLD_LDST32_DTPREL_LO12_NC = 536
R_AARCH64_TLSLD_LDST64_DTPREL_LO12 = 537
R_AARCH64_TLSLD_LDST64_DTPREL_LO12_NC = 538
R_AARCH64_TLSIE_MOVW_GOTTPREL_G1 = 539
R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC = 540
R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 = 541
R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC = 542
R_AARCH64_TLSIE_LD_GOTTPREL_PREL19 = 543
R_AARCH64_TLSLE_MOVW_TPREL_G2 = 544
R_AARCH64_TLSLE_MOVW_TPREL_G1 = 545
R_AARCH64_TLSLE_MOVW_TPREL_G1_NC = 546
R_AARCH64_TLSLE_MOVW_TPREL_G0 = 547
R_AARCH64_TLSLE_MOVW_TPREL_G0_NC = 548
R_AARCH64_TLSLE_ADD_TPREL_HI12 = 549
R_AARCH64_TLSLE_ADD_TPREL_LO12 = 550
R_AARCH64_TLSLE_ADD_TPREL_LO12_NC = 551
R_AARCH64_TLSLE_LDST8_TPREL_LO12 = 552
R_AARCH64_TLSLE_LDST8_TPREL_LO12_NC = 553
R_AARCH64_TLSLE_LDST16_TPREL_LO12 = 554
R_AARCH64_TLSLE_LDST16_TPREL_LO12_NC = 555
R_AARCH64_TLSLE_LDST32_TPREL_LO12 = 556
R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC = 557
R_AARCH64_TLSLE_LDST64_TPREL_LO12 = 558
R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC = 559
R_AARCH64_TLSDESC_LD_PREL19 = 560
R_AARCH64_TLSDESC_ADR_PREL21 = 561
R_AARCH64_TLSDESC_ADR_PAGE21 = 562
R_AARCH64_TLSDESC_LD64_LO12 = 563
R_AARCH64_TLSDESC_ADD_LO12 = 564
R_AARCH64_TLSDESC_OFF_G1 = 565
R_AARCH64_TLSDESC_OFF_G0_NC = 566
R_AARCH64_TLSDESC_LDR = 567
R_AARCH64_TLSDESC_ADD = 568
R_AARCH64_TLSDESC_CALL = 569
R_AARCH64_TLSLE_LDST128_TPREL_LO12 = 570
R_AARCH64_TLSLE_LDST128_TPREL_LO12_NC = 571
R_AARCH64_TLSLD_LDST128_DTPREL_LO12 = 572
R_AARCH64_TLSLD_LDST128_DTPREL_LO12_NC = 573
R_AARCH64_COPY = 1024
R_AARCH64_GLOB_DAT = 1025
R_AARCH64_JUMP_SLOT = 1026
R_AARCH64_RELATIVE = 1027
R_AARCH64_TLS_DTPMOD = 1028
R_AARCH64_TLS_DTPREL = 1029
R_AARCH64_TLS_TPREL = 1030
R_AARCH64_TLSDESC = 1031
R_AARCH64_IRELATIVE = 1032
DT_AARCH64_BTI_PLT = (DT_LOPROC + 1)
DT_AARCH64_PAC_PLT = (DT_LOPROC + 3)
DT_AARCH64_VARIANT_PCS = (DT_LOPROC + 5)
DT_AARCH64_NUM = 6
STO_AARCH64_VARIANT_PCS = 0x80
R_ARM_NONE = 0
R_ARM_ABS32 = 2
R_ARM_REL32 = 3
R_ARM_PC13 = 4
R_ARM_ABS16 = 5
R_ARM_ABS12 = 6
R_ARM_THM_ABS5 = 7
R_ARM_ABS8 = 8
R_ARM_SBREL32 = 9
R_ARM_THM_PC22 = 10
R_ARM_AMP_VCALL9 = 12
R_ARM_SWI24 = 13
R_ARM_TLS_DESC = 13
R_ARM_THM_SWI8 = 14
R_ARM_XPC25 = 15
R_ARM_THM_XPC22 = 16
R_ARM_TLS_DTPMOD32 = 17
R_ARM_TLS_DTPOFF32 = 18
R_ARM_TLS_TPOFF32 = 19
R_ARM_COPY = 20
R_ARM_GLOB_DAT = 21
R_ARM_JUMP_SLOT = 22
R_ARM_RELATIVE = 23
R_ARM_GOTOFF = 24
R_ARM_GOTPC = 25
R_ARM_GOT32 = 26
R_ARM_PLT32 = 27
R_ARM_CALL = 28
R_ARM_THM_JUMP24 = 30
R_ARM_BASE_ABS = 31
R_ARM_ALU_PCREL_7_0 = 32
R_ARM_ALU_PCREL_15_8 = 33
R_ARM_ALU_PCREL_23_15 = 34
R_ARM_LDR_SBREL_11_0 = 35
R_ARM_ALU_SBREL_19_12 = 36
R_ARM_ALU_SBREL_27_20 = 37
R_ARM_TARGET1 = 38
R_ARM_SBREL31 = 39
R_ARM_V4BX = 40
R_ARM_TARGET2 = 41
R_ARM_PREL31 = 42
R_ARM_MOVW_ABS_NC = 43
R_ARM_MOVT_ABS = 44
R_ARM_MOVW_PREL_NC = 45
R_ARM_MOVT_PREL = 46
R_ARM_THM_MOVW_ABS_NC = 47
R_ARM_ABS32_NOI = 55
R_ARM_REL32_NOI = 56
R_ARM_ALU_PC_G0_NC = 57
R_ARM_ALU_PC_G0 = 58
R_ARM_ALU_PC_G1_NC = 59
R_ARM_ALU_PC_G1 = 60
R_ARM_ALU_PC_G2 = 61
R_ARM_LDR_PC_G1 = 62
R_ARM_LDR_PC_G2 = 63
R_ARM_LDC_PC_G0 = 67
R_ARM_LDC_PC_G1 = 68
R_ARM_LDC_PC_G2 = 69
R_ARM_ALU_SB_G0_NC = 70
R_ARM_ALU_SB_G0 = 71
R_ARM_ALU_SB_G1_NC = 72
R_ARM_ALU_SB_G1 = 73
R_ARM_ALU_SB_G2 = 74
R_ARM_LDC_SB_G0 = 81
R_ARM_LDC_SB_G1 = 82
R_ARM_LDC_SB_G2 = 83
R_ARM_TLS_GOTDESC = 90
R_ARM_TLS_CALL = 91
R_ARM_TLS_DESCSEQ = 92
R_ARM_THM_TLS_CALL = 93
R_ARM_PLT32_ABS = 94
R_ARM_GOT_ABS = 95
R_ARM_GOT_PREL = 96
R_ARM_GOTRELAX = 99
R_ARM_GNU_VTENTRY = 100
R_ARM_GNU_VTINHERIT = 101
R_ARM_THM_PC11 = 102
R_ARM_ME_TOO = 128
R_ARM_THM_TLS_DESCSEQ = 129
R_ARM_THM_TLS_DESCSEQ16 = 129
R_ARM_THM_TLS_DESCSEQ32 = 130
R_ARM_IRELATIVE = 160
R_ARM_RXPC25 = 249
R_ARM_RSBREL32 = 250
R_ARM_THM_RPC22 = 251
R_ARM_RREL32 = 252
R_ARM_RABS22 = 253
R_ARM_RPC24 = 254
R_ARM_RBASE = 255
R_ARM_NUM = 256
R_CKCORE_NONE = 0
R_CKCORE_ADDR32 = 1
R_CKCORE_PCRELIMM8BY4 = 2
R_CKCORE_PCRELIMM11BY2 = 3
R_CKCORE_PCREL32 = 5
R_CKCORE_PCRELJSR_IMM11BY2 = 6
R_CKCORE_RELATIVE = 9
R_CKCORE_COPY = 10
R_CKCORE_GLOB_DAT = 11
R_CKCORE_JUMP_SLOT = 12
R_CKCORE_GOTOFF = 13
R_CKCORE_GOTPC = 14
R_CKCORE_GOT32 = 15
R_CKCORE_PLT32 = 16
R_CKCORE_ADDRGOT = 17
R_CKCORE_ADDRPLT = 18
R_CKCORE_PCREL_IMM26BY2 = 19
R_CKCORE_PCREL_IMM16BY2 = 20
R_CKCORE_PCREL_IMM16BY4 = 21
R_CKCORE_PCREL_IMM10BY2 = 22
R_CKCORE_PCREL_IMM10BY4 = 23
R_CKCORE_ADDR_HI16 = 24
R_CKCORE_ADDR_LO16 = 25
R_CKCORE_GOTPC_HI16 = 26
R_CKCORE_GOTPC_LO16 = 27
R_CKCORE_GOTOFF_HI16 = 28
R_CKCORE_GOTOFF_LO16 = 29
R_CKCORE_GOT12 = 30
R_CKCORE_GOT_HI16 = 31
R_CKCORE_GOT_LO16 = 32
R_CKCORE_PLT12 = 33
R_CKCORE_PLT_HI16 = 34
R_CKCORE_PLT_LO16 = 35
R_CKCORE_ADDRGOT_HI16 = 36
R_CKCORE_ADDRGOT_LO16 = 37
R_CKCORE_ADDRPLT_HI16 = 38
R_CKCORE_ADDRPLT_LO16 = 39
R_CKCORE_PCREL_JSR_IMM26BY2 = 40
R_CKCORE_TOFFSET_LO16 = 41
R_CKCORE_DOFFSET_LO16 = 42
R_CKCORE_PCREL_IMM18BY2 = 43
R_CKCORE_DOFFSET_IMM18 = 44
R_CKCORE_DOFFSET_IMM18BY2 = 45
R_CKCORE_DOFFSET_IMM18BY4 = 46
R_CKCORE_GOT_IMM18BY4 = 48
R_CKCORE_PLT_IMM18BY4 = 49
R_CKCORE_PCREL_IMM7BY4 = 50
R_CKCORE_TLS_LE32 = 51
R_CKCORE_TLS_IE32 = 52
R_CKCORE_TLS_GD32 = 53
R_CKCORE_TLS_LDM32 = 54
R_CKCORE_TLS_LDO32 = 55
R_CKCORE_TLS_DTPMOD32 = 56
R_CKCORE_TLS_DTPOFF32 = 57
R_CKCORE_TLS_TPOFF32 = 58
EF_CSKY_ABIMASK = 0XF0000000
EF_CSKY_OTHER = 0X0FFF0000
EF_CSKY_PROCESSOR = 0X0000FFFF
EF_CSKY_ABIV1 = 0X10000000
EF_CSKY_ABIV2 = 0X20000000
SHT_CSKY_ATTRIBUTES = (SHT_LOPROC + 1)
EF_IA_64_MASKOS = 0x0000000f
EF_IA_64_ABI64 = 0x00000010
EF_IA_64_ARCH = 0xff000000
PT_IA_64_ARCHEXT = (PT_LOPROC + 0)
PT_IA_64_UNWIND = (PT_LOPROC + 1)
PT_IA_64_HP_OPT_ANOT = (PT_LOOS + 0x12)
PT_IA_64_HP_HSL_ANOT = (PT_LOOS + 0x13)
PT_IA_64_HP_STACK = (PT_LOOS + 0x14)
PF_IA_64_NORECOV = 0x80000000
SHT_IA_64_EXT = (SHT_LOPROC + 0)
SHT_IA_64_UNWIND = (SHT_LOPROC + 1)
SHF_IA_64_SHORT = 0x10000000
SHF_IA_64_NORECOV = 0x20000000
DT_IA_64_PLT_RESERVE = (DT_LOPROC + 0)
DT_IA_64_NUM = 1
R_IA64_NONE = 0x00
R_IA64_IMM14 = 0x21
R_IA64_IMM22 = 0x22
R_IA64_IMM64 = 0x23
R_IA64_DIR32MSB = 0x24
R_IA64_DIR32LSB = 0x25
R_IA64_DIR64MSB = 0x26
R_IA64_DIR64LSB = 0x27
R_IA64_GPREL22 = 0x2a
R_IA64_GPREL64I = 0x2b
R_IA64_GPREL32MSB = 0x2c
R_IA64_GPREL32LSB = 0x2d
R_IA64_GPREL64MSB = 0x2e
R_IA64_GPREL64LSB = 0x2f
R_IA64_LTOFF22 = 0x32
R_IA64_LTOFF64I = 0x33
R_IA64_PLTOFF22 = 0x3a
R_IA64_PLTOFF64I = 0x3b
R_IA64_PLTOFF64MSB = 0x3e
R_IA64_PLTOFF64LSB = 0x3f
R_IA64_FPTR64I = 0x43
R_IA64_FPTR32MSB = 0x44
R_IA64_FPTR32LSB = 0x45
R_IA64_FPTR64MSB = 0x46
R_IA64_FPTR64LSB = 0x47
R_IA64_PCREL60B = 0x48
R_IA64_PCREL21B = 0x49
R_IA64_PCREL21M = 0x4a
R_IA64_PCREL21F = 0x4b
R_IA64_PCREL32MSB = 0x4c
R_IA64_PCREL32LSB = 0x4d
R_IA64_PCREL64MSB = 0x4e
R_IA64_PCREL64LSB = 0x4f
R_IA64_LTOFF_FPTR22 = 0x52
R_IA64_LTOFF_FPTR64I = 0x53
R_IA64_LTOFF_FPTR32MSB = 0x54
R_IA64_LTOFF_FPTR32LSB = 0x55
R_IA64_LTOFF_FPTR64MSB = 0x56
R_IA64_LTOFF_FPTR64LSB = 0x57
R_IA64_SEGREL32MSB = 0x5c
R_IA64_SEGREL32LSB = 0x5d
R_IA64_SEGREL64MSB = 0x5e
R_IA64_SEGREL64LSB = 0x5f
R_IA64_SECREL32MSB = 0x64
R_IA64_SECREL32LSB = 0x65
R_IA64_SECREL64MSB = 0x66
R_IA64_SECREL64LSB = 0x67
R_IA64_REL32MSB = 0x6c
R_IA64_REL32LSB = 0x6d
R_IA64_REL64MSB = 0x6e
R_IA64_REL64LSB = 0x6f
R_IA64_LTV32MSB = 0x74
R_IA64_LTV32LSB = 0x75
R_IA64_LTV64MSB = 0x76
R_IA64_LTV64LSB = 0x77
R_IA64_PCREL21BI = 0x79
R_IA64_PCREL22 = 0x7a
R_IA64_PCREL64I = 0x7b
R_IA64_IPLTMSB = 0x80
R_IA64_IPLTLSB = 0x81
R_IA64_COPY = 0x84
R_IA64_SUB = 0x85
R_IA64_LTOFF22X = 0x86
R_IA64_LDXMOV = 0x87
R_IA64_TPREL14 = 0x91
R_IA64_TPREL22 = 0x92
R_IA64_TPREL64I = 0x93
R_IA64_TPREL64MSB = 0x96
R_IA64_TPREL64LSB = 0x97
R_IA64_LTOFF_TPREL22 = 0x9a
R_IA64_DTPMOD64MSB = 0xa6
R_IA64_DTPMOD64LSB = 0xa7
R_IA64_LTOFF_DTPMOD22 = 0xaa
R_IA64_DTPREL14 = 0xb1
R_IA64_DTPREL22 = 0xb2
R_IA64_DTPREL64I = 0xb3
R_IA64_DTPREL32MSB = 0xb4
R_IA64_DTPREL32LSB = 0xb5
R_IA64_DTPREL64MSB = 0xb6
R_IA64_DTPREL64LSB = 0xb7
R_IA64_LTOFF_DTPREL22 = 0xba
EF_SH_MACH_MASK = 0x1f
EF_SH_UNKNOWN = 0x0
EF_SH1 = 0x1
EF_SH2 = 0x2
EF_SH3 = 0x3
EF_SH_DSP = 0x4
EF_SH3_DSP = 0x5
EF_SH4AL_DSP = 0x6
EF_SH3E = 0x8
EF_SH4 = 0x9
EF_SH2E = 0xb
EF_SH4A = 0xc
EF_SH2A = 0xd
EF_SH4_NOFPU = 0x10
EF_SH4A_NOFPU = 0x11
EF_SH4_NOMMU_NOFPU = 0x12
EF_SH2A_NOFPU = 0x13
EF_SH3_NOMMU = 0x14
EF_SH2A_SH4_NOFPU = 0x15
EF_SH2A_SH3_NOFPU = 0x16
EF_SH2A_SH4 = 0x17
EF_SH2A_SH3E = 0x18
R_SH_NONE = 0
R_SH_DIR32 = 1
R_SH_REL32 = 2
R_SH_DIR8WPN = 3
R_SH_IND12W = 4
R_SH_DIR8WPL = 5
R_SH_DIR8WPZ = 6
R_SH_DIR8BP = 7
R_SH_DIR8W = 8
R_SH_DIR8L = 9
R_SH_SWITCH16 = 25
R_SH_SWITCH32 = 26
R_SH_USES = 27
R_SH_COUNT = 28
R_SH_ALIGN = 29
R_SH_CODE = 30
R_SH_DATA = 31
R_SH_LABEL = 32
R_SH_SWITCH8 = 33
R_SH_GNU_VTINHERIT = 34
R_SH_GNU_VTENTRY = 35
R_SH_TLS_GD_32 = 144
R_SH_TLS_LD_32 = 145
R_SH_TLS_LDO_32 = 146
R_SH_TLS_IE_32 = 147
R_SH_TLS_LE_32 = 148
R_SH_TLS_DTPMOD32 = 149
R_SH_TLS_DTPOFF32 = 150
R_SH_TLS_TPOFF32 = 151
R_SH_GOT32 = 160
R_SH_PLT32 = 161
R_SH_COPY = 162
R_SH_GLOB_DAT = 163
R_SH_JMP_SLOT = 164
R_SH_RELATIVE = 165
R_SH_GOTOFF = 166
R_SH_GOTPC = 167
R_SH_NUM = 256
EF_S390_HIGH_GPRS = 0x00000001
R_390_NONE = 0
R_390_8 = 1
R_390_12 = 2
R_390_16 = 3
R_390_32 = 4
R_390_PC32 = 5
R_390_GOT12 = 6
R_390_GOT32 = 7
R_390_PLT32 = 8
R_390_COPY = 9
R_390_GLOB_DAT = 10
R_390_JMP_SLOT = 11
R_390_RELATIVE = 12
R_390_GOTOFF32 = 13
R_390_GOTPC = 14
R_390_GOT16 = 15
R_390_PC16 = 16
R_390_PC16DBL = 17
R_390_PLT16DBL = 18
R_390_PC32DBL = 19
R_390_PLT32DBL = 20
R_390_GOTPCDBL = 21
R_390_64 = 22
R_390_PC64 = 23
R_390_GOT64 = 24
R_390_PLT64 = 25
R_390_GOTENT = 26
R_390_GOTOFF16 = 27
R_390_GOTOFF64 = 28
R_390_GOTPLT12 = 29
R_390_GOTPLT16 = 30
R_390_GOTPLT32 = 31
R_390_GOTPLT64 = 32
R_390_GOTPLTENT = 33
R_390_PLTOFF16 = 34
R_390_PLTOFF32 = 35
R_390_PLTOFF64 = 36
R_390_TLS_LOAD = 37
R_390_TLS_DTPMOD = 54
R_390_TLS_DTPOFF = 55
R_390_20 = 57
R_390_GOT20 = 58
R_390_GOTPLT20 = 59
R_390_IRELATIVE = 61
R_390_NUM = 62
R_CRIS_NONE = 0
R_CRIS_8 = 1
R_CRIS_16 = 2
R_CRIS_32 = 3
R_CRIS_8_PCREL = 4
R_CRIS_16_PCREL = 5
R_CRIS_32_PCREL = 6
R_CRIS_GNU_VTINHERIT = 7
R_CRIS_GNU_VTENTRY = 8
R_CRIS_COPY = 9
R_CRIS_GLOB_DAT = 10
R_CRIS_JUMP_SLOT = 11
R_CRIS_RELATIVE = 12
R_CRIS_16_GOT = 13
R_CRIS_32_GOT = 14
R_CRIS_16_GOTPLT = 15
R_CRIS_32_GOTPLT = 16
R_CRIS_32_GOTREL = 17
R_CRIS_32_PLT_GOTREL = 18
R_CRIS_32_PLT_PCREL = 19
R_CRIS_NUM = 20
R_X86_64_NONE = 0
R_X86_64_64 = 1
R_X86_64_PC32 = 2
R_X86_64_GOT32 = 3
R_X86_64_PLT32 = 4
R_X86_64_COPY = 5
R_X86_64_GLOB_DAT = 6
R_X86_64_JUMP_SLOT = 7
R_X86_64_RELATIVE = 8
R_X86_64_32 = 10
R_X86_64_32S = 11
R_X86_64_16 = 12
R_X86_64_PC16 = 13
R_X86_64_8 = 14
R_X86_64_PC8 = 15
R_X86_64_DTPMOD64 = 16
R_X86_64_DTPOFF64 = 17
R_X86_64_TPOFF64 = 18
R_X86_64_DTPOFF32 = 21
R_X86_64_TPOFF32 = 23
R_X86_64_PC64 = 24
R_X86_64_GOTOFF64 = 25
R_X86_64_GOT64 = 27
R_X86_64_GOTPC64 = 29
R_X86_64_GOTPLT64 = 30
R_X86_64_SIZE32 = 32
R_X86_64_SIZE64 = 33
R_X86_64_GOTPC32_TLSDESC = 34
R_X86_64_TLSDESC = 36
R_X86_64_IRELATIVE = 37
R_X86_64_RELATIVE64 = 38
R_X86_64_NUM = 43
SHT_X86_64_UNWIND = 0x70000001
R_MN10300_NONE = 0
R_MN10300_32 = 1
R_MN10300_16 = 2
R_MN10300_8 = 3
R_MN10300_PCREL32 = 4
R_MN10300_PCREL16 = 5
R_MN10300_PCREL8 = 6
R_MN10300_GNU_VTINHERIT = 7
R_MN10300_GNU_VTENTRY = 8
R_MN10300_24 = 9
R_MN10300_GOTPC32 = 10
R_MN10300_GOTPC16 = 11
R_MN10300_GOTOFF32 = 12
R_MN10300_GOTOFF24 = 13
R_MN10300_GOTOFF16 = 14
R_MN10300_PLT32 = 15
R_MN10300_PLT16 = 16
R_MN10300_GOT32 = 17
R_MN10300_GOT24 = 18
R_MN10300_GOT16 = 19
R_MN10300_COPY = 20
R_MN10300_GLOB_DAT = 21
R_MN10300_JMP_SLOT = 22
R_MN10300_RELATIVE = 23
R_MN10300_TLS_GD = 24
R_MN10300_TLS_LD = 25
R_MN10300_TLS_LDO = 26
R_MN10300_TLS_DTPMOD = 30
R_MN10300_TLS_DTPOFF = 31
R_MN10300_TLS_TPOFF = 32
R_MN10300_NUM = 35
R_M32R_NONE = 0
R_M32R_16 = 1
R_M32R_32 = 2
R_M32R_24 = 3
R_M32R_10_PCREL = 4
R_M32R_18_PCREL = 5
R_M32R_26_PCREL = 6
R_M32R_HI16_ULO = 7
R_M32R_HI16_SLO = 8
R_M32R_LO16 = 9
R_M32R_SDA16 = 10
R_M32R_GNU_VTINHERIT = 11
R_M32R_GNU_VTENTRY = 12
R_M32R_16_RELA = 33
R_M32R_32_RELA = 34
R_M32R_24_RELA = 35
R_M32R_10_PCREL_RELA = 36
R_M32R_18_PCREL_RELA = 37
R_M32R_26_PCREL_RELA = 38
R_M32R_HI16_ULO_RELA = 39
R_M32R_HI16_SLO_RELA = 40
R_M32R_LO16_RELA = 41
R_M32R_SDA16_RELA = 42
R_M32R_RELA_GNU_VTINHERIT = 43
R_M32R_RELA_GNU_VTENTRY = 44
R_M32R_REL32 = 45
R_M32R_GOT24 = 48
R_M32R_26_PLTREL = 49
R_M32R_COPY = 50
R_M32R_GLOB_DAT = 51
R_M32R_JMP_SLOT = 52
R_M32R_RELATIVE = 53
R_M32R_GOTOFF = 54
R_M32R_GOTPC24 = 55
R_M32R_GOT16_LO = 58
R_M32R_GOTOFF_LO = 64
R_M32R_NUM = 256
R_MICROBLAZE_NONE = 0
R_MICROBLAZE_32 = 1
R_MICROBLAZE_32_PCREL = 2
R_MICROBLAZE_64_PCREL = 3
R_MICROBLAZE_32_PCREL_LO = 4
R_MICROBLAZE_64 = 5
R_MICROBLAZE_32_LO = 6
R_MICROBLAZE_SRO32 = 7
R_MICROBLAZE_SRW32 = 8
R_MICROBLAZE_64_NONE = 9
R_MICROBLAZE_32_SYM_OP_SYM = 10
R_MICROBLAZE_GNU_VTINHERIT = 11
R_MICROBLAZE_GNU_VTENTRY = 12
R_MICROBLAZE_GOTPC_64 = 13
R_MICROBLAZE_GOT_64 = 14
R_MICROBLAZE_PLT_64 = 15
R_MICROBLAZE_REL = 16
R_MICROBLAZE_JUMP_SLOT = 17
R_MICROBLAZE_GLOB_DAT = 18
R_MICROBLAZE_GOTOFF_64 = 19
R_MICROBLAZE_GOTOFF_32 = 20
R_MICROBLAZE_COPY = 21
R_MICROBLAZE_TLS = 22
R_MICROBLAZE_TLSGD = 23
R_MICROBLAZE_TLSLD = 24
R_MICROBLAZE_TLSDTPMOD32 = 25
R_MICROBLAZE_TLSDTPREL32 = 26
R_MICROBLAZE_TLSDTPREL64 = 27
R_MICROBLAZE_TLSGOTTPREL32 = 28
R_MICROBLAZE_TLSTPREL32 = 29
DT_NIOS2_GP = 0x70000002
R_NIOS2_NONE = 0
R_NIOS2_S16 = 1
R_NIOS2_U16 = 2
R_NIOS2_PCREL16 = 3
R_NIOS2_CALL26 = 4
R_NIOS2_IMM5 = 5
R_NIOS2_CACHE_OPX = 6
R_NIOS2_IMM6 = 7
R_NIOS2_IMM8 = 8
R_NIOS2_HI16 = 9
R_NIOS2_LO16 = 10
R_NIOS2_HIADJ16 = 11
R_NIOS2_BFD_RELOC_32 = 12
R_NIOS2_BFD_RELOC_16 = 13
R_NIOS2_BFD_RELOC_8 = 14
R_NIOS2_GPREL = 15
R_NIOS2_GNU_VTINHERIT = 16
R_NIOS2_GNU_VTENTRY = 17
R_NIOS2_UJMP = 18
R_NIOS2_CJMP = 19
R_NIOS2_CALLR = 20
R_NIOS2_GOT16 = 22
R_NIOS2_CALL16 = 23
R_NIOS2_GOTOFF_LO = 24
R_NIOS2_GOTOFF_HA = 25
R_NIOS2_PCREL_LO = 26
R_NIOS2_PCREL_HA = 27
R_NIOS2_TLS_GD16 = 28
R_NIOS2_TLS_LDM16 = 29
R_NIOS2_TLS_LDO16 = 30
R_NIOS2_TLS_IE16 = 31
R_NIOS2_TLS_LE16 = 32
R_NIOS2_TLS_DTPMOD = 33
R_NIOS2_TLS_DTPREL = 34
R_NIOS2_TLS_TPREL = 35
R_NIOS2_COPY = 36
R_NIOS2_GLOB_DAT = 37
R_NIOS2_JUMP_SLOT = 38
R_NIOS2_RELATIVE = 39
R_NIOS2_GOTOFF = 40
R_NIOS2_CALL26_NOAT = 41
R_NIOS2_GOT_LO = 42
R_NIOS2_GOT_HA = 43
R_NIOS2_CALL_LO = 44
R_NIOS2_CALL_HA = 45
R_TILEPRO_NONE = 0
R_TILEPRO_32 = 1
R_TILEPRO_16 = 2
R_TILEPRO_8 = 3
R_TILEPRO_32_PCREL = 4
R_TILEPRO_16_PCREL = 5
R_TILEPRO_8_PCREL = 6
R_TILEPRO_LO16 = 7
R_TILEPRO_HI16 = 8
R_TILEPRO_HA16 = 9
R_TILEPRO_COPY = 10
R_TILEPRO_GLOB_DAT = 11
R_TILEPRO_JMP_SLOT = 12
R_TILEPRO_RELATIVE = 13
R_TILEPRO_BROFF_X1 = 14
R_TILEPRO_JOFFLONG_X1 = 15
R_TILEPRO_JOFFLONG_X1_PLT = 16
R_TILEPRO_IMM8_X0 = 17
R_TILEPRO_IMM8_Y0 = 18
R_TILEPRO_IMM8_X1 = 19
R_TILEPRO_IMM8_Y1 = 20
R_TILEPRO_MT_IMM15_X1 = 21
R_TILEPRO_MF_IMM15_X1 = 22
R_TILEPRO_IMM16_X0 = 23
R_TILEPRO_IMM16_X1 = 24
R_TILEPRO_IMM16_X0_LO = 25
R_TILEPRO_IMM16_X1_LO = 26
R_TILEPRO_IMM16_X0_HI = 27
R_TILEPRO_IMM16_X1_HI = 28
R_TILEPRO_IMM16_X0_HA = 29
R_TILEPRO_IMM16_X1_HA = 30
R_TILEPRO_IMM16_X0_PCREL = 31
R_TILEPRO_IMM16_X1_PCREL = 32
R_TILEPRO_IMM16_X0_LO_PCREL = 33
R_TILEPRO_IMM16_X1_LO_PCREL = 34
R_TILEPRO_IMM16_X0_HI_PCREL = 35
R_TILEPRO_IMM16_X1_HI_PCREL = 36
R_TILEPRO_IMM16_X0_HA_PCREL = 37
R_TILEPRO_IMM16_X1_HA_PCREL = 38
R_TILEPRO_IMM16_X0_GOT = 39
R_TILEPRO_IMM16_X1_GOT = 40
R_TILEPRO_IMM16_X0_GOT_LO = 41
R_TILEPRO_IMM16_X1_GOT_LO = 42
R_TILEPRO_IMM16_X0_GOT_HI = 43
R_TILEPRO_IMM16_X1_GOT_HI = 44
R_TILEPRO_IMM16_X0_GOT_HA = 45
R_TILEPRO_IMM16_X1_GOT_HA = 46
R_TILEPRO_MMSTART_X0 = 47
R_TILEPRO_MMEND_X0 = 48
R_TILEPRO_MMSTART_X1 = 49
R_TILEPRO_MMEND_X1 = 50
R_TILEPRO_SHAMT_X0 = 51
R_TILEPRO_SHAMT_X1 = 52
R_TILEPRO_SHAMT_Y0 = 53
R_TILEPRO_SHAMT_Y1 = 54
R_TILEPRO_DEST_IMM8_X1 = 55
R_TILEPRO_TLS_GD_CALL = 60
R_TILEPRO_IMM8_X0_TLS_GD_ADD = 61
R_TILEPRO_IMM8_X1_TLS_GD_ADD = 62
R_TILEPRO_IMM8_Y0_TLS_GD_ADD = 63
R_TILEPRO_IMM8_Y1_TLS_GD_ADD = 64
R_TILEPRO_TLS_IE_LOAD = 65
R_TILEPRO_IMM16_X0_TLS_GD = 66
R_TILEPRO_IMM16_X1_TLS_GD = 67
R_TILEPRO_IMM16_X0_TLS_GD_LO = 68
R_TILEPRO_IMM16_X1_TLS_GD_LO = 69
R_TILEPRO_IMM16_X0_TLS_GD_HI = 70
R_TILEPRO_IMM16_X1_TLS_GD_HI = 71
R_TILEPRO_IMM16_X0_TLS_GD_HA = 72
R_TILEPRO_IMM16_X1_TLS_GD_HA = 73
R_TILEPRO_IMM16_X0_TLS_IE = 74
R_TILEPRO_IMM16_X1_TLS_IE = 75
R_TILEPRO_IMM16_X0_TLS_IE_LO = 76
R_TILEPRO_IMM16_X1_TLS_IE_LO = 77
R_TILEPRO_IMM16_X0_TLS_IE_HI = 78
R_TILEPRO_IMM16_X1_TLS_IE_HI = 79
R_TILEPRO_IMM16_X0_TLS_IE_HA = 80
R_TILEPRO_IMM16_X1_TLS_IE_HA = 81
R_TILEPRO_TLS_DTPMOD32 = 82
R_TILEPRO_TLS_DTPOFF32 = 83
R_TILEPRO_TLS_TPOFF32 = 84
R_TILEPRO_IMM16_X0_TLS_LE = 85
R_TILEPRO_IMM16_X1_TLS_LE = 86
R_TILEPRO_IMM16_X0_TLS_LE_LO = 87
R_TILEPRO_IMM16_X1_TLS_LE_LO = 88
R_TILEPRO_IMM16_X0_TLS_LE_HI = 89
R_TILEPRO_IMM16_X1_TLS_LE_HI = 90
R_TILEPRO_IMM16_X0_TLS_LE_HA = 91
R_TILEPRO_IMM16_X1_TLS_LE_HA = 92
R_TILEPRO_GNU_VTINHERIT = 128
R_TILEPRO_GNU_VTENTRY = 129
R_TILEPRO_NUM = 130
R_TILEGX_NONE = 0
R_TILEGX_64 = 1
R_TILEGX_32 = 2
R_TILEGX_16 = 3
R_TILEGX_8 = 4
R_TILEGX_64_PCREL = 5
R_TILEGX_32_PCREL = 6
R_TILEGX_16_PCREL = 7
R_TILEGX_8_PCREL = 8
R_TILEGX_HW0 = 9
R_TILEGX_HW1 = 10
R_TILEGX_HW2 = 11
R_TILEGX_HW3 = 12
R_TILEGX_HW0_LAST = 13
R_TILEGX_HW1_LAST = 14
R_TILEGX_HW2_LAST = 15
R_TILEGX_COPY = 16
R_TILEGX_GLOB_DAT = 17
R_TILEGX_JMP_SLOT = 18
R_TILEGX_RELATIVE = 19
R_TILEGX_BROFF_X1 = 20
R_TILEGX_JUMPOFF_X1 = 21
R_TILEGX_JUMPOFF_X1_PLT = 22
R_TILEGX_IMM8_X0 = 23
R_TILEGX_IMM8_Y0 = 24
R_TILEGX_IMM8_X1 = 25
R_TILEGX_IMM8_Y1 = 26
R_TILEGX_DEST_IMM8_X1 = 27
R_TILEGX_MT_IMM14_X1 = 28
R_TILEGX_MF_IMM14_X1 = 29
R_TILEGX_MMSTART_X0 = 30
R_TILEGX_MMEND_X0 = 31
R_TILEGX_SHAMT_X0 = 32
R_TILEGX_SHAMT_X1 = 33
R_TILEGX_SHAMT_Y0 = 34
R_TILEGX_SHAMT_Y1 = 35
R_TILEGX_IMM16_X0_HW0 = 36
R_TILEGX_IMM16_X1_HW0 = 37
R_TILEGX_IMM16_X0_HW1 = 38
R_TILEGX_IMM16_X1_HW1 = 39
R_TILEGX_IMM16_X0_HW2 = 40
R_TILEGX_IMM16_X1_HW2 = 41
R_TILEGX_IMM16_X0_HW3 = 42
R_TILEGX_IMM16_X1_HW3 = 43
R_TILEGX_IMM16_X0_HW0_LAST = 44
R_TILEGX_IMM16_X1_HW0_LAST = 45
R_TILEGX_IMM16_X0_HW1_LAST = 46
R_TILEGX_IMM16_X1_HW1_LAST = 47
R_TILEGX_IMM16_X0_HW2_LAST = 48
R_TILEGX_IMM16_X1_HW2_LAST = 49
R_TILEGX_IMM16_X0_HW0_PCREL = 50
R_TILEGX_IMM16_X1_HW0_PCREL = 51
R_TILEGX_IMM16_X0_HW1_PCREL = 52
R_TILEGX_IMM16_X1_HW1_PCREL = 53
R_TILEGX_IMM16_X0_HW2_PCREL = 54
R_TILEGX_IMM16_X1_HW2_PCREL = 55
R_TILEGX_IMM16_X0_HW3_PCREL = 56
R_TILEGX_IMM16_X1_HW3_PCREL = 57
R_TILEGX_IMM16_X0_HW0_LAST_PCREL = 58
R_TILEGX_IMM16_X1_HW0_LAST_PCREL = 59
R_TILEGX_IMM16_X0_HW1_LAST_PCREL = 60
R_TILEGX_IMM16_X1_HW1_LAST_PCREL = 61
R_TILEGX_IMM16_X0_HW2_LAST_PCREL = 62
R_TILEGX_IMM16_X1_HW2_LAST_PCREL = 63
R_TILEGX_IMM16_X0_HW0_GOT = 64
R_TILEGX_IMM16_X1_HW0_GOT = 65
R_TILEGX_IMM16_X0_HW0_PLT_PCREL = 66
R_TILEGX_IMM16_X1_HW0_PLT_PCREL = 67
R_TILEGX_IMM16_X0_HW1_PLT_PCREL = 68
R_TILEGX_IMM16_X1_HW1_PLT_PCREL = 69
R_TILEGX_IMM16_X0_HW2_PLT_PCREL = 70
R_TILEGX_IMM16_X1_HW2_PLT_PCREL = 71
R_TILEGX_IMM16_X0_HW0_LAST_GOT = 72
R_TILEGX_IMM16_X1_HW0_LAST_GOT = 73
R_TILEGX_IMM16_X0_HW1_LAST_GOT = 74
R_TILEGX_IMM16_X1_HW1_LAST_GOT = 75
R_TILEGX_IMM16_X0_HW3_PLT_PCREL = 76
R_TILEGX_IMM16_X1_HW3_PLT_PCREL = 77
R_TILEGX_IMM16_X0_HW0_TLS_GD = 78
R_TILEGX_IMM16_X1_HW0_TLS_GD = 79
R_TILEGX_IMM16_X0_HW0_TLS_LE = 80
R_TILEGX_IMM16_X1_HW0_TLS_LE = 81
R_TILEGX_IMM16_X0_HW0_LAST_TLS_LE = 82
R_TILEGX_IMM16_X1_HW0_LAST_TLS_LE = 83
R_TILEGX_IMM16_X0_HW1_LAST_TLS_LE = 84
R_TILEGX_IMM16_X1_HW1_LAST_TLS_LE = 85
R_TILEGX_IMM16_X0_HW0_LAST_TLS_GD = 86
R_TILEGX_IMM16_X1_HW0_LAST_TLS_GD = 87
R_TILEGX_IMM16_X0_HW1_LAST_TLS_GD = 88
R_TILEGX_IMM16_X1_HW1_LAST_TLS_GD = 89
R_TILEGX_IMM16_X0_HW0_TLS_IE = 92
R_TILEGX_IMM16_X1_HW0_TLS_IE = 93
R_TILEGX_IMM16_X0_HW0_LAST_PLT_PCREL = 94
R_TILEGX_IMM16_X1_HW0_LAST_PLT_PCREL = 95
R_TILEGX_IMM16_X0_HW1_LAST_PLT_PCREL = 96
R_TILEGX_IMM16_X1_HW1_LAST_PLT_PCREL = 97
R_TILEGX_IMM16_X0_HW2_LAST_PLT_PCREL = 98
R_TILEGX_IMM16_X1_HW2_LAST_PLT_PCREL = 99
R_TILEGX_IMM16_X0_HW0_LAST_TLS_IE = 100
R_TILEGX_IMM16_X1_HW0_LAST_TLS_IE = 101
R_TILEGX_IMM16_X0_HW1_LAST_TLS_IE = 102
R_TILEGX_IMM16_X1_HW1_LAST_TLS_IE = 103
R_TILEGX_TLS_DTPMOD64 = 106
R_TILEGX_TLS_DTPOFF64 = 107
R_TILEGX_TLS_TPOFF64 = 108
R_TILEGX_TLS_DTPMOD32 = 109
R_TILEGX_TLS_DTPOFF32 = 110
R_TILEGX_TLS_TPOFF32 = 111
R_TILEGX_TLS_GD_CALL = 112
R_TILEGX_IMM8_X0_TLS_GD_ADD = 113
R_TILEGX_IMM8_X1_TLS_GD_ADD = 114
R_TILEGX_IMM8_Y0_TLS_GD_ADD = 115
R_TILEGX_IMM8_Y1_TLS_GD_ADD = 116
R_TILEGX_TLS_IE_LOAD = 117
R_TILEGX_IMM8_X0_TLS_ADD = 118
R_TILEGX_IMM8_X1_TLS_ADD = 119
R_TILEGX_IMM8_Y0_TLS_ADD = 120
R_TILEGX_IMM8_Y1_TLS_ADD = 121
R_TILEGX_GNU_VTINHERIT = 128
R_TILEGX_GNU_VTENTRY = 129
R_TILEGX_NUM = 130
EF_RISCV_RVC = 0x0001
EF_RISCV_FLOAT_ABI = 0x0006
EF_RISCV_FLOAT_ABI_SOFT = 0x0000
EF_RISCV_FLOAT_ABI_SINGLE = 0x0002
EF_RISCV_FLOAT_ABI_DOUBLE = 0x0004
EF_RISCV_FLOAT_ABI_QUAD = 0x0006
R_RISCV_NONE = 0
R_RISCV_32 = 1
R_RISCV_64 = 2
R_RISCV_RELATIVE = 3
R_RISCV_COPY = 4
R_RISCV_JUMP_SLOT = 5
R_RISCV_TLS_DTPMOD32 = 6
R_RISCV_TLS_DTPMOD64 = 7
R_RISCV_TLS_DTPREL32 = 8
R_RISCV_TLS_DTPREL64 = 9
R_RISCV_TLS_TPREL32 = 10
R_RISCV_TLS_TPREL64 = 11
R_RISCV_BRANCH = 16
R_RISCV_JAL = 17
R_RISCV_CALL = 18
R_RISCV_CALL_PLT = 19
R_RISCV_GOT_HI20 = 20
R_RISCV_TLS_GOT_HI20 = 21
R_RISCV_TLS_GD_HI20 = 22
R_RISCV_PCREL_HI20 = 23
R_RISCV_PCREL_LO12_I = 24
R_RISCV_PCREL_LO12_S = 25
R_RISCV_HI20 = 26
R_RISCV_LO12_I = 27
R_RISCV_LO12_S = 28
R_RISCV_TPREL_HI20 = 29
R_RISCV_TPREL_LO12_I = 30
R_RISCV_TPREL_LO12_S = 31
R_RISCV_TPREL_ADD = 32
R_RISCV_ADD8 = 33
R_RISCV_ADD16 = 34
R_RISCV_ADD32 = 35
R_RISCV_ADD64 = 36
R_RISCV_SUB8 = 37
R_RISCV_SUB16 = 38
R_RISCV_SUB32 = 39
R_RISCV_SUB64 = 40
R_RISCV_GNU_VTINHERIT = 41
R_RISCV_GNU_VTENTRY = 42
R_RISCV_ALIGN = 43
R_RISCV_RVC_BRANCH = 44
R_RISCV_RVC_JUMP = 45
R_RISCV_RVC_LUI = 46
R_RISCV_GPREL_I = 47
R_RISCV_GPREL_S = 48
R_RISCV_TPREL_I = 49
R_RISCV_TPREL_S = 50
R_RISCV_RELAX = 51
R_RISCV_SUB6 = 52
R_RISCV_SET6 = 53
R_RISCV_SET8 = 54
R_RISCV_SET16 = 55
R_RISCV_SET32 = 56
R_RISCV_32_PCREL = 57
R_RISCV_IRELATIVE = 58
R_RISCV_NUM = 59
R_BPF_NONE = 0
R_BPF_64_64 = 1
R_BPF_64_32 = 10
R_METAG_HIADDR16 = 0
R_METAG_LOADDR16 = 1
R_METAG_ADDR32 = 2
R_METAG_NONE = 3
R_METAG_RELBRANCH = 4
R_METAG_GETSETOFF = 5
R_METAG_REG32OP1 = 6
R_METAG_REG32OP2 = 7
R_METAG_REG32OP3 = 8
R_METAG_REG16OP1 = 9
R_METAG_REG16OP2 = 10
R_METAG_REG16OP3 = 11
R_METAG_REG32OP4 = 12
R_METAG_HIOG = 13
R_METAG_LOOG = 14
R_METAG_REL8 = 15
R_METAG_REL16 = 16
R_METAG_GNU_VTINHERIT = 30
R_METAG_GNU_VTENTRY = 31
R_METAG_HI16_GOTOFF = 32
R_METAG_LO16_GOTOFF = 33
R_METAG_GETSET_GOTOFF = 34
R_METAG_GETSET_GOT = 35
R_METAG_HI16_GOTPC = 36
R_METAG_LO16_GOTPC = 37
R_METAG_HI16_PLT = 38
R_METAG_LO16_PLT = 39
R_METAG_RELBRANCH_PLT = 40
R_METAG_GOTOFF = 41
R_METAG_PLT = 42
R_METAG_COPY = 43
R_METAG_JMP_SLOT = 44
R_METAG_RELATIVE = 45
R_METAG_GLOB_DAT = 46
R_METAG_TLS_GD = 47
R_METAG_TLS_LDM = 48
R_METAG_TLS_LDO_HI16 = 49
R_METAG_TLS_LDO_LO16 = 50
R_METAG_TLS_LDO = 51
R_METAG_TLS_IE = 52
R_METAG_TLS_IENONPIC = 53
R_METAG_TLS_IENONPIC_HI16 = 54
R_METAG_TLS_IENONPIC_LO16 = 55
R_METAG_TLS_TPOFF = 56
R_METAG_TLS_DTPMOD = 57
R_METAG_TLS_DTPOFF = 58
R_METAG_TLS_LE = 59
R_METAG_TLS_LE_HI16 = 60
R_METAG_TLS_LE_LO16 = 61
R_NDS32_NONE = 0
R_NDS32_32_RELA = 20
R_NDS32_COPY = 39
R_NDS32_GLOB_DAT = 40
R_NDS32_JMP_SLOT = 41
R_NDS32_RELATIVE = 42
R_NDS32_TLS_TPOFF = 102
R_NDS32_TLS_DESC = 119
R_ARC_NONE = 0x0
R_ARC_8 = 0x1
R_ARC_16 = 0x2
R_ARC_24 = 0x3
R_ARC_32 = 0x4
R_ARC_B26 = 0x5
R_ARC_B22_PCREL = 0x6
R_ARC_H30 = 0x7
R_ARC_N8 = 0x8
R_ARC_N16 = 0x9
R_ARC_N24 = 0xA
R_ARC_N32 = 0xB
R_ARC_SDA = 0xC
R_ARC_SECTOFF = 0xD
R_ARC_S21H_PCREL = 0xE
R_ARC_S21W_PCREL = 0xF
R_ARC_S25H_PCREL = 0x10
R_ARC_S25W_PCREL = 0x11
R_ARC_SDA32 = 0x12
R_ARC_SDA_LDST = 0x13
R_ARC_SDA_LDST1 = 0x14
R_ARC_SDA_LDST2 = 0x15
R_ARC_SDA16_LD = 0x16
R_ARC_SDA16_LD1 = 0x17
R_ARC_SDA16_LD2 = 0x18
R_ARC_S13_PCREL = 0x19
R_ARC_W = 0x1A
R_ARC_32_ME = 0x1B
R_ARC_N32_ME = 0x1C
R_ARC_SECTOFF_ME = 0x1D
R_ARC_SDA32_ME = 0x1E
R_ARC_W_ME = 0x1F
R_ARC_H30_ME = 0x20
R_ARC_SECTOFF_U8 = 0x21
R_ARC_SECTOFF_S9 = 0x22
R_AC_SECTOFF_U8 = 0x23
R_AC_SECTOFF_U8_1 = 0x24
R_AC_SECTOFF_U8_2 = 0x25
R_AC_SECTOFF_S9 = 0x26
R_AC_SECTOFF_S9_1 = 0x27
R_AC_SECTOFF_S9_2 = 0x28
R_ARC_SECTOFF_ME_1 = 0x29
R_ARC_SECTOFF_ME_2 = 0x2A
R_ARC_SECTOFF_1 = 0x2B
R_ARC_SECTOFF_2 = 0x2C
R_ARC_PC32 = 0x32
R_ARC_GOTPC32 = 0x33
R_ARC_PLT32 = 0x34
R_ARC_COPY = 0x35
R_ARC_GLOB_DAT = 0x36
R_ARC_JUMP_SLOT = 0x37
R_ARC_RELATIVE = 0x38
R_ARC_GOTOFF = 0x39
R_ARC_GOTPC = 0x3A
R_ARC_GOT32 = 0x3B
R_ARC_TLS_DTPMOD = 0x42
R_ARC_TLS_DTPOFF = 0x43
R_ARC_TLS_TPOFF = 0x44
R_ARC_TLS_GD_GOT = 0x45
R_ARC_TLS_GD_LD = 0x46
R_ARC_TLS_GD_CALL = 0x47
R_ARC_TLS_IE_GOT = 0x48
R_ARC_TLS_DTPOFF_S9 = 0x4a
R_ARC_TLS_LE_S9 = 0x4a
R_ARC_TLS_LE_32 = 0x4b
R_OR1K_NONE = 0
R_OR1K_32 = 1
R_OR1K_16 = 2
R_OR1K_8 = 3
R_OR1K_LO_16_IN_INSN = 4
R_OR1K_HI_16_IN_INSN = 5
R_OR1K_INSN_REL_26 = 6
R_OR1K_GNU_VTENTRY = 7
R_OR1K_GNU_VTINHERIT = 8
R_OR1K_32_PCREL = 9
R_OR1K_16_PCREL = 10
R_OR1K_8_PCREL = 11
R_OR1K_GOTPC_HI16 = 12
R_OR1K_GOTPC_LO16 = 13
R_OR1K_GOT16 = 14
R_OR1K_PLT26 = 15
R_OR1K_GOTOFF_HI16 = 16
R_OR1K_GOTOFF_LO16 = 17
R_OR1K_COPY = 18
R_OR1K_GLOB_DAT = 19
R_OR1K_JMP_SLOT = 20
R_OR1K_RELATIVE = 21
R_OR1K_TLS_GD_HI16 = 22
R_OR1K_TLS_GD_LO16 = 23
R_OR1K_TLS_LDM_HI16 = 24
R_OR1K_TLS_LDM_LO16 = 25
R_OR1K_TLS_LDO_HI16 = 26
R_OR1K_TLS_LDO_LO16 = 27
R_OR1K_TLS_IE_HI16 = 28
R_OR1K_TLS_IE_LO16 = 29
R_OR1K_TLS_LE_HI16 = 30
R_OR1K_TLS_LE_LO16 = 31
R_OR1K_TLS_TPOFF = 32
R_OR1K_TLS_DTPOFF = 33
R_OR1K_TLS_DTPMOD = 34

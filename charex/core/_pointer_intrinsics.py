from numba.core import types, cgutils
from numba.extending import intrinsic


@intrinsic
def address_as_void_pointer(typingctx, address):
    """ Returns a void pointer from a given memory address """
    sig = types.voidptr(address)

    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return sig, codegen
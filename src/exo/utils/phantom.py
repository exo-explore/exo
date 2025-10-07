from typing import Optional


class _PhantomData[*T]:
    """
    Internal machinery of the phantom data - it stores nothing.
    """


type PhantomData[*T] = Optional[_PhantomData[*T]]
"""
Allows you to use generics in functions without storing anything of that generic type. 
Just use `None` and you'll be fine
"""

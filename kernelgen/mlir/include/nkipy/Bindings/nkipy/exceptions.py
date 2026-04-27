import warnings
from contextvars import ContextVar

# By default, Python ignores deprecation warnings.
# we have to enable it to see the warning.
warnings.simplefilter("always", DeprecationWarning)

PrintLog = ContextVar("PrintLog", default=False)


class bcolors:
    """ANSI color escape codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class NkipyException(Exception):
    """Base class for all Nkipy exceptions.

    Exception is the base class for warnings and errors.
    Developers can subclass this class to provide additional information

    Parameters
    ----------
    message : str
        The error message.
    """

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message


class NkipyWarning(NkipyException):
    """Base class for all Nkipy warnings.

    Warning is the base class for all warnings.
    Developers can subclass this class to provide additional information

    Parameters
    ----------
    message : str
        The warning message.

    line : int, optional
        The line number of the warning.

    category_str : str, optional
        The warning category string.

    category : Warning, optional
        The warning category.
    """

    def __init__(self, message, line=None, category_str=None, category=None):
        message = bcolors.OKBLUE + message + bcolors.ENDC
        if category_str is not None:
            message = "\n{} {}".format(category_str, message)
        if line is not None:
            message += bcolors.BOLD + " (line {})".format(line) + bcolors.ENDC
        NkipyException.__init__(self, message)
        self.category = category

    def warn(self):
        warnings.warn(self.message, category=self.category)

    def log(self):
        if PrintLog.get():
            print(self.message)


class NkipyError(NkipyException):
    """Base class for all Nkipy errors.

    Error is the base class for all errors.
    Developers can subclass this class to provide additional information

    Parameters
    ----------
    message : str
        The error message.

    line: int, optional
        The line number of the error.

    category_str : str, optional
        The error category string.
    """

    def __init__(self, message, line=None, category_str=None):
        message = bcolors.OKBLUE + message + bcolors.ENDC
        if category_str is not None:
            message = "{} {}".format(category_str, message)
        if line is not None:
            message += bcolors.BOLD + " (line {})".format(line) + bcolors.ENDC
        NkipyException.__init__(self, message)

    def error(self):
        raise self.message


""" Inherited Error subclasses """


class DTypeError(NkipyError):
    """A subclass for specifying data type related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Data Type]" + bcolors.ENDC
        NkipyError.__init__(self, msg, line, category_str)


class APIError(NkipyError):
    """A subclass for specifying API related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[API]" + bcolors.ENDC
        NkipyError.__init__(self, msg, line, category_str)


class DSLError(NkipyError):
    """A subclass for specifying imperative DSL related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Imperative]" + bcolors.ENDC
        NkipyError.__init__(self, msg, line, category_str)


class TensorError(NkipyError):
    """A subclass for specifying tensor related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Tensor]" + bcolors.ENDC
        NkipyError.__init__(self, msg, line, category_str)


class DeviceError(NkipyError):
    """A subclass for specifying device related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Device]" + bcolors.ENDC
        NkipyError.__init__(self, msg, line, category_str)


class AssertError(NkipyError):
    """A subclass for specifying assert related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Assert]" + bcolors.ENDC
        NkipyError.__init__(self, msg, line, category_str)


""" New Error subclasses """


class NkipyNotImplementedError(NkipyError):
    """A subclass for specifying not implemented exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Not Implemented]" + bcolors.ENDC
        NkipyError.__init__(self, msg, line, category_str)


class MLIRLimitationError(NkipyError):
    """A subclass for specifying MLIR limitation exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[MLIR Limitation]" + bcolors.ENDC
        NkipyError.__init__(self, msg, line, category_str)


class NkipyValueError(NkipyError):
    """A subclass for specifying Nkipy value exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Value Error]" + bcolors.ENDC
        NkipyError.__init__(self, msg, line, category_str)


""" New Warning subclasses """


class DTypeWarning(NkipyWarning):
    """A subclass for specifying data type related warning"""

    def __init__(self, msg, line=None):
        category_str = bcolors.WARNING + "[Data Type]" + bcolors.ENDC
        NkipyWarning.__init__(self, msg, line, category_str, RuntimeWarning)


class NkipyDeprecationWarning(NkipyWarning):
    """A subclass for specifying deprecation warning"""

    def __init__(self, msg, line=None):
        category_str = bcolors.WARNING + "[Deprecation]" + bcolors.ENDC
        NkipyWarning.__init__(self, msg, line, category_str, DeprecationWarning)


class APIWarning(NkipyWarning):
    """A subclass for specifying API related warning"""

    def __init__(self, msg, line=None):
        category_str = bcolors.WARNING + "[API]" + bcolors.ENDC
        NkipyWarning.__init__(self, msg, line, category_str, RuntimeWarning)


class PassWarning(NkipyWarning):
    """A subclass for specifying pass related warning"""

    def __init__(self, msg, line=None):
        category_str = bcolors.WARNING + "[Pass]" + bcolors.ENDC
        NkipyWarning.__init__(self, msg, line, category_str, RuntimeWarning)

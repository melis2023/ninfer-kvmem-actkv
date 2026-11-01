"""Own a Windows compiler process tree, including children of exited parents."""
import ctypes
from ctypes import wintypes
from pathlib import Path
import signal
import subprocess
import sys
import uuid


class BasicLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD)]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", BasicLimits), ("io_counters", ctypes.c_uint64 * 6),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


def kernel_api():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    for name, result, arguments in (
        ("CreateJobObjectW", wintypes.HANDLE, [ctypes.c_void_p, wintypes.LPCWSTR]),
        ("OpenJobObjectW", wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]),
        ("GetCurrentProcess", wintypes.HANDLE, []),
        ("AssignProcessToJobObject", wintypes.BOOL, [wintypes.HANDLE, wintypes.HANDLE]),
        ("SetInformationJobObject", wintypes.BOOL,
         [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]),
        ("TerminateJobObject", wintypes.BOOL, [wintypes.HANDLE, wintypes.UINT]),
        ("CloseHandle", wintypes.BOOL, [wintypes.HANDLE]),
    ):
        function = getattr(kernel, name)
        function.restype = result
        function.argtypes = arguments
    return kernel


def checked(value):
    if not value:
        raise ctypes.WinError(ctypes.get_last_error())
    return value


class WindowsBuildJob:
    def __init__(self):
        self.kernel = kernel_api()
        self.name = "ninfer-build-" + uuid.uuid4().hex
        self.handle = checked(self.kernel.CreateJobObjectW(None, self.name))
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            checked(self.kernel.SetInformationJobObject(
                self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.kernel.CloseHandle(self.handle)
            raise

    def command(self, command):
        # The bootstrap joins the job before it creates any compiler processes.
        return [sys.executable, str(Path(__file__).resolve()), self.name, *command]

    def stop(self):
        checked(self.kernel.TerminateJobObject(self.handle, 130))

    def close(self):
        checked(self.kernel.CloseHandle(self.handle))


def main():
    kernel = kernel_api()
    handle = checked(kernel.OpenJobObjectW(0x0001, False, sys.argv[1]))
    try:
        checked(kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()))
    finally:
        checked(kernel.CloseHandle(handle))
    # Keep the group leader alive during the graceful interval. Its children
    # receive the same console event; the owner terminates any survivors.
    signal.signal(signal.SIGINT, lambda *_: None)
    signal.signal(signal.SIGBREAK, lambda *_: None)
    return subprocess.call(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())

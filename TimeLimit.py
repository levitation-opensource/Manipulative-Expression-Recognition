# -*- coding: utf-8 -*-

#
# Author: Roland Pihlakas, 2017 - 2024
#
# roland@simplify.ee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#


import time
import threading

import sys
from _thread import interrupt_main
import signal



# the original source of the idea is here:
# http://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python

# TODO: see also https://github.com/glenfant/stopit and http://gahtune.blogspot.fr/2013/08/a-timeout-context-manager.html

_shared_lock = threading.Lock()

class time_limit:

  def __init__(self, seconds, msg='', disable_time_limit=False):

    self.exited = False
    self.lock = threading.Lock()
    self.timer = None
    self.old_signal_handler = None
    self.old_unraisablehook = None
    self.replaced_signal = None

    self.seconds = seconds
    self.msg = msg

    self.time_limit_disable_count = 1 if disable_time_limit else 0
    self.timeout_pending = False
    self.timeout_seconds = None
    self.timeout_msg = None

  #/ def __init__(self, seconds, msg=''):

  def __enter__(self):

    if self.seconds is not None: 

      if self.seconds > 0:
        
        hook_signal_handler = True   # TODO: if hook_signal_handler == False then set unraisablehook handler before calling signal.raise_signal() or interrupt_main()
        allow_sigbreak = True

        if hook_signal_handler and allow_sigbreak and sys.version_info[0] >= 4 or (sys.version_info[0] == 3 and sys.version_info[0] >= 8):     # in this case we can use signal.raise_signal with custom signal
          self.replaced_signal = signal.SIGBREAK   # using a different signal than SIGINT so we do not need to override and interfere with user's Ctrl+C handling. SIGBREAK is related to Ctrl+Break, but this is very rarely used.
        else:   # if Python version < 3.8 then we need to use interrupt_main()
          self.replaced_signal = signal.SIGINT
        
        # without this code block the interrupt may be delayed and triggered outside of time limit's try catch block
        # an alternative is to use if exc_type is KeyboardInterrupt: try: print() except: pass in the __exit__ method
        # see   # see https://stackoverflow.com/questions/4606942/why-cant-i-handle-a-keyboardinterrupt-in-python

        if hook_signal_handler:
          self.old_signal_handler = signal.signal(self.replaced_signal, self.signal_handler)  # https://stackoverflow.com/questions/842557/how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py


        self.timer = threading.Timer(
          self.seconds, 
          lambda: self.time_limit_handler(self.seconds, self.msg)
        )
        self.timer.start()

      else:   # self.seconds <= 0:

        # we are in the main thread now, so lets raise TimeoutError directly. Do not call signal.raise_signal(signal.SIGINT) or interrupt_main()
        self.time_limit_handler(self.seconds, self.msg, called_from_main_thread = True) 

        # NB! it might be that timeouts are disabled, so we need to yield properly even when seconds is past due
        pass 
      
      #/ if self.seconds > 0:

    else:   #/ if (self.seconds is not None):   # no time limit

      pass
    
    return self  

  #/ def __enter__(self):

  def __exit__(self, exc_type, exc_value, traceback):

    # NB! take lock so that time limit cannot happen at the same time as this function
    with self.lock:

      self.exited = True


      if self.seconds > 0:
            
        self.timer.cancel()

        if self.old_signal_handler is not None:
          signal.signal(self.replaced_signal, self.old_signal_handler)

        if self.old_unraisablehook is not None:
          # _shared_lock.acquire() # lock against other time_limit handlers accessing sys.unraisablehook
          with _shared_lock:
            sys.unraisablehook = self.old_unraisablehook
            # _shared_lock.release()

        if exc_type is KeyboardInterrupt:   # happens when interrupt_main() or signal.raise_signal(signal.SIGINT) is called

          # without this strange code block the interrupt may be delayed and triggered outside of time limit's try catch block, in some random parts of Python modules async code
          # an alternative is to enable custom signal handler with signal.signal()
          # see   # see https://stackoverflow.com/questions/4606942/why-cant-i-handle-a-keyboardinterrupt-in-python
          try:
            print() # any i/o will get the second KeyboardInterrupt here?
          except:
            qqq = True    # for debugging
            pass

          # we are in the main thread now, so lets raise TimeoutError directly. Do not call signal.raise_signal(signal.SIGINT) or interrupt_main()
          raise TimeoutError("Timed out for operation '{0}' after {1} seconds".format(self.msg, self.seconds))

      #/ if self.seconds > 0:

      return

    #/ with self.lock:

  #/ def __exit__(self, exc_type, exc_value, traceback):

  # NB! sometimes raise TimeoutError() in signal_handler() is not sufficient so it is good to let the main thread to read the pending termination state
  def get_timeout_pending(self):

    # NB! serialise this call so that if time_limit_handler is already running then it will run until end, raisin the exception in the main thread. Else the main thread may read the timeout_pending flag, raise its own exception, and then time_limit_handler would raise another later.
    with self.lock:
      result = self.timeout_pending
      return result

  #/ def get_timeout_pending(self):

  def signal_handler(self, sig, frame):

    # NB! prepare to start timer again in unraisablehook since sometimes the exception is ignored by Python when it occurs in certain bits of code. In these cases unraisablehook is called.
    # this happens for example here:
    # Exception ignored in: <function WeakSet.__init__.<locals>._remove
    # File "C:\Program Files\Python310\lib\_weakrefset.py", line 40, in _remove
    #   self = selfref()
    #
    # see https://github.com/python/cpython/issues/95959 :
    # and https://docs.python.org/3/library/weakref.html#weakref.finalize
    # Exceptions raised by finalizer callbacks during garbage collection will be shown on the standard error output, but cannot be propagated. They are handled in the same way as exceptions raised from an object’s __del__() method or a weak reference’s callback.

    # https://docs.python.org/3/library/sys.html#sys.unraisablehook
    # sys.unraisablehook() can be overridden to control how unraisable exceptions are handled.
    # NB! use interlocked exchange or lock to ensure that usage of nested time_limit blocks does not cause problems here 
    # lock against other time_limit handlers accessing sys.unraisablehook
    with _shared_lock:
      self.old_unraisablehook = sys.unraisablehook
      sys.unraisablehook = self.unraisablehook

    raise TimeoutError("Timed out for operation '{0}' after {1} seconds".format(self.msg, self.seconds))

  #/ def signal_handler(self, sig, frame):

  def unraisablehook(self, unraisable):

    # lock against other time_limit handlers accessing sys.unraisablehook
    with _shared_lock:
      sys.unraisablehook = self.old_unraisablehook
      self.old_unraisablehook = None

    # we are in the main thread so it is safe to overwrite self.timer here without taking self.lock
    self.timer = threading.Timer(
      0.001,    # raise ASAP again    # NB! use nonzero value, else the timer callback is called immediately in the stack of threading.Timer construction. Also very small values maybe do not work: see https://github.com/glenfant/stopit/issues/26
      lambda: self.time_limit_handler(self.seconds, self.msg)
    )
    self.timer.start()

  #/ def unraisablehook(self, unraisable):

  def time_limit_handler(self, timeout_seconds_in, timeout_msg_in, called_from_main_thread = False):

    # NB! take lock so that exiting cannot happen at the same time as this function
    with self.lock:

      if self.exited:
        return


      self.timeout_pending = True
      self.timeout_seconds = timeout_seconds_in
      self.timeout_msg = timeout_msg_in

      if self.time_limit_disable_count == 0:

        if called_from_main_thread:

          self.exited = True
          # we are in the main thread now, so lets raise TimeoutError directly. Do not call signal.raise_signal(signal.SIGINT) or interrupt_main()
          raise TimeoutError("Timed out for operation '{0}' after {1} seconds".format(self.msg, self.seconds))

        else:

          # NB! do not raise signals while potentially other thread of time_limit class is working on _shared_lock code regions
          with _shared_lock:

            if sys.version_info[0] >= 4 or (sys.version_info[0] == 3 and sys.version_info[0] >= 8): 
              # interrupt_main()
              signal.raise_signal(self.replaced_signal)  # New in version 3.8.
            else:
              interrupt_main()  # Raise a KeyboardInterrupt exception in the main thread. A subthread can use this function to interrupt the main thread.

          #/ with _shared_lock:

      else: # this branch activates when disable_time_limit() was called and after that time limit has been reached

        pass    # NB! the interrupt will still be raised, but later, after enable_time_limit() is called

      #/ if self.time_limit_disable_count == 0:

      # self.lock.release()
      return

    #/ with self.lock:

  #/ def time_limit_handler(timeout_seconds_in, timeout_msg_in):

  def disable_time_limit(self):

    self.time_limit_disable_count += 1


  def enable_time_limit(self):
        
    # NB! take lock so that time limit cannot happen at the same time as this function
    with self.lock:

      if self.exited:   # enabling time limit after the time limit context has exited will not raise time limit exceptions anymore

        if self.seconds > 0:
          self.timer.cancel()   # it is safe to cancel multiple times

        return


      self.time_limit_disable_count -= 1

      if self.time_limit_disable_count == 0 and self.timeout_pending:

        self.timeout_pending = False
        self.timeout_seconds = None
        self.timeout_msg = None

        self.exited = True
        # we are in the main thread now, so lets raise TimeoutError directly. Do not call signal.raise_signal(signal.SIGINT) or interrupt_main()
        raise TimeoutError("Timed out for operation '{0}' after {1} seconds".format(self.msg, self.seconds))

      #/ if self.time_limit_disable_count == 0 and self.timeout_pending:

      return

    #/ with self.lock:

  #/ def enable_time_limit():

#/ class TimeLimit:



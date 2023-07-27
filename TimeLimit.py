# -*- coding: utf-8 -*-

#
# Author: Roland Pihlakas, 2017 - 2023
#
# roland@simplify.ee
#


import time
from contextlib import contextmanager
import threading

import sys
if sys.version_info[0] < 3: 
  from thread import interrupt_main
else:
  from _thread import interrupt_main



time_limit_disable_count = 0
timeout_pending = False
timeout_seconds = None
timeout_msg = None



# the original source of the idea is here:
# http://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python



def disable_time_limit():
  global time_limit_disable_count

  time_limit_disable_count += 1


def enable_time_limit():
  global time_limit_disable_count, timeout_pending, timeout_seconds, timeout_msg

  time_limit_disable_count -= 1

  if time_limit_disable_count == 0 and timeout_pending:

    timeout_pending = False
    timeout_seconds = None
    timeout_msg = None

    try:
      interrupt_main()  # Raise a KeyboardInterrupt exception in the main thread. A subthread can use this function to interrupt the main thread.
    except KeyboardInterrupt:
      raise TimeoutError("Timed out for operation '{0}' after {1} seconds".format(timeout_msg1, timeout_seconds1))

  #/ if time_limit_disable_count == 0 and timeout_pending:

#/ def enable_time_limit():


def time_limit_handler(timeout_seconds_in, timeout_msg_in):
  global time_limit_disable_count, timeout_pending, timeout_seconds, timeout_msg

  if time_limit_disable_count == 0:

    interrupt_main()  # Raise a KeyboardInterrupt exception in the main thread. A subthread can use this function to interrupt the main thread.

  else: # this branch activates when disable_time_limit() was called and after that time limit has been reached

    timeout_pending = True
    timeout_seconds = timeout_seconds_in
    timeout_msg = timeout_msg_in

  #/ if time_limit_disable_count == 0:

#/ def time_limit_handler(timeout_seconds_in, timeout_msg_in):


@contextmanager
def time_limit(seconds, msg=''):
  global timeout_pending, timeout_seconds, timeout_msg

  if seconds is not None: 

    if seconds > 0:

      timer = threading.Timer(
        seconds, 
        lambda: time_limit_handler(seconds, msg)
      )
      timer.start()

      try:
        yield
      except KeyboardInterrupt:

        raise TimeoutError("Timed out for operation '{0}' after {1} seconds".format(msg, seconds))
      finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()

    else:   # seconds <= 0:

      try:
        time_limit_handler(seconds, msg)  # Raise a KeyboardInterrupt exception in the main thread. A subthread can use this function to interrupt the main thread.
      except KeyboardInterrupt:

        raise TimeoutError("Timed out for operation '{0}' after {1} seconds".format(msg, seconds))


      # NB! it might be that timeouts are disabled, so we need to yield properly even when seconds is past due
      yield 
      
    #/ if seconds > 0:

  else:   #/ if (seconds is not None):   # no time limit

    yield

#/ def time_limit(seconds, msg=''):


# -*- coding: utf-8 -*-

#
# Author: Roland Pihlakas, 2017 - 2023
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



# the original source of the idea is here:
# http://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python

class time_limit:

  def __init__(self, seconds, msg=''):

    self.seconds = seconds
    self.msg = msg

    self.time_limit_disable_count = 0
    self.timeout_pending = False
    self.timeout_seconds = None
    self.timeout_msg = None

  #/ def __init__(self, seconds, msg=''):

  def __enter__(self):

    if self.seconds is not None: 

      if self.seconds > 0:

        self.timer = threading.Timer(
          self.seconds, 
          lambda: self.time_limit_handler(self.seconds, self.msg)
        )
        self.timer.start()

      else:   # self.seconds <= 0:

        try:
          self.time_limit_handler(self.seconds, self.msg)  # Raise a KeyboardInterrupt exception in the main thread. A subthread can use this function to interrupt the main thread.
        except KeyboardInterrupt:
          raise TimeoutError("Timed out for operation '{0}' after {1} seconds".format(self.msg, self.seconds))

        # NB! it might be that timeouts are disabled, so we need to yield properly even when seconds is past due
        pass 
      
      #/ if self.seconds > 0:

    else:   #/ if (self.seconds is not None):   # no time limit

      pass
    
    return self  

  def __exit__(self, exc_type, exc_value, traceback):

    if self.seconds > 0:

      self.timer.cancel()

      if exc_type is KeyboardInterrupt:
        raise TimeoutError("Timed out for operation '{0}' after {1} seconds".format(self.msg, self.seconds))

    #/ if self.seconds > 0:

  #/ def __exit__(self, exc_type, exc_value, traceback):

  def time_limit_handler(self, timeout_seconds_in, timeout_msg_in):

    self.timeout_pending = True
    self.timeout_seconds = timeout_seconds_in
    self.timeout_msg = timeout_msg_in

    if self.time_limit_disable_count == 0:

      interrupt_main()  # Raise a KeyboardInterrupt exception in the main thread. A subthread can use this function to interrupt the main thread.

    else: # this branch activates when disable_time_limit() was called and after that time limit has been reached

      pass

    #/ if time_limit_disable_count == 0:
  #/ def time_limit_handler(timeout_seconds_in, timeout_msg_in):

  def disable_time_limit(self):

    self.time_limit_disable_count += 1


  def enable_time_limit(self):

    self.time_limit_disable_count -= 1

    if self.time_limit_disable_count == 0 and self.timeout_pending:

      self.timeout_pending = False
      self.timeout_seconds = None
      self.timeout_msg = None

      try:
        interrupt_main()  # Raise a KeyboardInterrupt exception in the main thread. A subthread can use this function to interrupt the main thread.
      except KeyboardInterrupt:
        raise TimeoutError("Timed out for operation '{0}' after {1} seconds".format(self.msg, self.seconds))

    #/ if time_limit_disable_count == 0 and timeout_pending:

  #/ def enable_time_limit():

#/ class TimeLimit:



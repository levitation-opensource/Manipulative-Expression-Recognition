# -*- coding: utf-8 -*-

#
# Author: Roland Pihlakas, 2017 - 2021
#
# roland@simplify.ee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

import os
import sys
import io
import time
import datetime
import locale

import traceback



# https://stackoverflow.com/questions/333995/how-to-detect-that-python-code-is-being-executed-through-the-debugger
is_dev_machine = (os.name == 'nt')
debugging = (is_dev_machine and sys.gettrace() is not None) and (1 == 1)  # debugging switches



# TODO
# from MyPool import current_process_index, parent_exited, exiting
current_process_index = 0
parent_exited = False
exiting = False
current_request_id = None


capture_level = 0
capture_buffers = []


pid = os.getpid()
pid_str = " :    " + str(pid).rjust(7) + " :    "



# set up console colors
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR
ansi_PREFIX = '\033['
ansi_RESET = '\033[0m'
ansi_INTENSE = '\033[1m'      # bold or bright colour variation
ansi_SLOWBLINK = '\033[5m'    # inconsistent support in terminals
ansi_REVERSE = '\033[7m'      # inconsistent support in terminals
# foreground colors:
ansi_BLACK = '\033[30m' 
ansi_RED = '\033[31m' 
ansi_GREEN = '\033[32m'
ansi_YELLOW = '\033[33m'
ansi_BLUE = '\033[34m'
ansi_MAGENTA = '\033[35m'
ansi_CYAN = '\033[36m'
ansi_WHITE = '\033[37m'



ansi_colors_inited = None

def init_colors():
  global ansi_colors_inited

  if ansi_colors_inited is not None:
    return;

  if os.name == "nt":
    ansi_colors_inited = False
    try:
      # https://github.com/tartley/colorama
      # 
      # On Windows, calling init() will filter ANSI escape sequences out of any text sent to stdout or stderr, and replace them with equivalent Win32 calls.
      # 
      # On other platforms, calling init() has no effect (unless you request other optional functionality; see “Init Keyword Args”, below). By design, this permits applications to call init() unconditionally on all platforms, after which ANSI output should just work.
      # 
      # To stop using Colorama before your program exits, simply call deinit(). This will restore stdout and stderr to their original values, so that Colorama is disabled. To resume using Colorama again, call reinit(); it is cheaper than calling init() again (but does the same thing).
      import colorama
      colorama.init()
      ansi_colors_inited = True
    except Exception as ex:
      pass
  else:   #/ if os.name == "nt":
    ansi_colors_inited = True
  #/ if os.name == "nt":

#/ def init_colors():


def get_now_str():

  now_str = datetime.datetime.strftime(datetime.datetime.now(), '%m.%d %H:%M:%S')
  return now_str

#/ def get_now_str():


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):

  def __init__(self, terminal, logfile, is_error_log):

    self.is_error_log = is_error_log


    if (terminal 
      and os.name == "nt" 
      and not isinstance(terminal, Logger)):    # NB! loggers can be chained for log file replication purposes, but colorama.AnsiToWin32 must not be chained else colour info gets lost

      try:
        import colorama
        init_colors()
        # NB! use .stream in order to be able to call flush()
        self.terminal = colorama.AnsiToWin32(terminal).stream # https://github.com/tartley/colorama
      except Exception as ex:
        self.terminal = terminal

    else:

      self.terminal = terminal


    self.log = None

    try:
      if logfile is not None:
        self.log = io.open(logfile, "a", 1024 * 1024, encoding="utf8", errors='ignore')    # https://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
    except Exception as ex:
      msg = "Exception during Logger.__init__ io.open: " + str(ex) + "\n" + traceback.format_exc()
      msg_with_colours = ansi_RED + ansi_INTENSE + msg + ansi_RESET
      if capture_level == 0:        
        try:
          self.terminal.write(msg)
        except Exception as ex:
          pass
      else: #/ if capture_level == 0:
        capture_buffers[capture_level - 1] += msg_with_colours

    try:
      if self.log is not None and not self.log.closed:
        self.log.flush()

    except Exception as ex:   # time_limit.TimeoutException

      if (is_dev_machine or not self.is_error_log) and self.terminal:

        msg = str(ex) + "\n" + traceback.format_exc()
        msg_with_colours = ansi_RED + ansi_INTENSE + msg + ansi_RESET
        if capture_level == 0:          
          try:
            self.terminal.write(msg_with_colours)
          except Exception as ex:
            pass
        else: #/ if capture_level == 0:
          capture_buffers[capture_level - 1] += msg

      #/ if (is_dev_machine or not self.is_error_log) and self.terminal:

  #/ def __init__(self, terminal, logfile, is_error_log):  


  def write(self, message_in):
    

    if parent_exited or exiting:  # NB!
      return


    # if message_in.strip() == "":
    #   return

    if isinstance(message_in, bytes):
      message = message_in.decode("utf-8", 'ignore')  # NB! message_in might be byte array not a string
    else:
      message = message_in


    now = get_now_str()

    if len(message) >= 2 and message[-2:] == "\n\r":
      message_end_linebreak = "\n\r"
    elif len(message) >= 2 and message[-2:] == "\r\n":
      message_end_linebreak = "\r\n"
    elif len(message) >= 1 and message[-1:] == "\n":
      message_end_linebreak = "\n"
    else:
      message_end_linebreak = ""

    newline_prefix = now + pid_str + str(current_process_index).rjust(6) + " :    "
    newline = message_end_linebreak if message_end_linebreak else "\n"
    message_without_end_linebreak = message[:-len(message_end_linebreak)] if message_end_linebreak != "" else message
    message_with_timestamp = newline_prefix + message_without_end_linebreak.replace(newline, newline + newline_prefix) + message_end_linebreak

    if message_end_linebreak == "":
      # message += "\n"
      message_with_timestamp += "\n"


    message_with_timestamp = message_with_timestamp.encode("utf-8", 'ignore').decode("utf-8", 'ignore') #.decode('latin1', 'ignore'))


    if (is_dev_machine or not self.is_error_log) and self.terminal:   # error_log output is hidden from live console

      msg_with_colours = ""

      use_ansi_colour = True and ((False or self.is_error_log) 
                        and (ansi_PREFIX not in message)  # NB! ensure that we do not override any colours present in the message
                        and message.strip() != "")        # optimisation for newlines
      if use_ansi_colour:
        msg_with_colours += ansi_RED + ansi_INTENSE

      msg_with_colours += message

      if use_ansi_colour:
        msg_with_colours += ansi_RESET


      if capture_level == 0:

        try:

          self.terminal.write(msg_with_colours)

        except Exception as ex:

          msg = str(ex) + "\n" + traceback.format_exc()

          try:
            if self.log is not None and not self.log.closed:
              self.log.write(msg)  
              self.log.flush()
          except Exception as ex:
            pass
      
      else: #/ if capture_level == 0:

        capture_buffers[capture_level - 1] += msg_with_colours

    #/ if (is_dev_machine or not self.is_error_log) and self.terminal:


    if message.strip() != "":  # NB! still write this message to console since it might contain progress bars or other formatting

      try:
        if self.log is not None and not self.log.closed:
          self.log.write(message_with_timestamp.strip() + "\n")  # NB! strip 
          self.log.flush()

      except Exception as ex: 

        if (is_dev_machine or not self.is_error_log) and self.terminal:

          msg = str(ex) + "\n" + traceback.format_exc()
          msg_with_colours = ansi_RED + ansi_INTENSE + msg + ansi_RESET
          if capture_level == 0:            
            try:
              self.terminal.write(msg_with_colours)
            except Exception as ex:
              pass
          else: #/ if capture_level == 0:
            capture_buffers[capture_level - 1] += msg

        #/ if (is_dev_machine or not self.is_error_log) and self.terminal:

    #/ if message.strip() != "":

  #/ def write(self, message_with_timestamp): 


  def flush(self):

    try:
      if self.terminal:
        self.terminal.flush()

    except Exception as ex: 
      msg = str(ex) + "\n" + traceback.format_exc()

      try:
        if self.log is not None and not self.log.closed:
          self.log.write(msg)  
          self.log.flush()
      except Exception as ex:
        pass


    try:
      if self.log is not None and not self.log.closed:
        self.log.flush()

    except Exception as ex: 

      if (is_dev_machine or not self.is_error_log) and self.terminal:

        msg = str(ex) + "\n" + traceback.format_exc()
        msg_with_colours = ansi_RED + ansi_INTENSE + msg + ansi_RESET
        if capture_level == 0:
          try:
            self.terminal.write(msg_with_colours)
          except Exception as ex:
            pass
        else: #/ if capture_level == 0:
          capture_buffers[capture_level - 1] += msg

      #/ if (is_dev_machine or not self.is_error_log) and self.terminal:
    
    pass 

  #/ def flush(self): 
  
  
  @property
  def encoding(self):
    
    return self.terminal.encoding
    #return "utf8"


  def fileno(self):

    return self.terminal.fileno()

#/ class Logger(object):


def rename_log_file_if_needed(filename, max_tries = 10):

  if os.path.exists(filename):

    try:

      filedate = datetime.datetime.fromtimestamp(os.path.getmtime(filename))
      filedate_str = datetime.datetime.strftime(filedate, '%Y.%m.%d-%H.%M.%S')

      new_filename = filename + "-" + filedate_str + ".txt"
      if filename != new_filename:

        try_index = 1
        while True:

          try:

            os.rename(filename, new_filename)
            return

          except Exception as ex:

            if try_index >= max_tries:
              raise

            try_index += 1
            print("retrying log file rename: " + filename)
            time.sleep(1)
            continue

          #/ try:

      #/ if filename != new_filename:

    except Exception as ex:
      pass

  #/ if os.path.exists(filename):

#/ def rename_log_file_if_needed(filename):


def logger_set_current_request_id(new_current_request_id):
  global current_request_id

  current_request_id = new_current_request_id

#/ def logger_set_current_request_id(new_current_request_id):


def start_capture():
  global capture_level

  capture_level += 1
  capture_buffers.append("")

#/ def start_capture():


def end_capture(print_result = False):
  global capture_level

  assert(capture_level >= 1)
  capture_level -= 1

  result = capture_buffers.pop()

  if print_result:
    try:
      print(result)
    except Exception:
      encoding = locale.getpreferredencoding(False)
      if not encoding:
        encoding = locale.getdefaultlocale()[1]
      if not encoding:
        encoding = "ascii"
      print(result.encode("utf-8", 'ignore').decode(encoding, 'ignore'))
  #/ if print_result:

  return result

#/ def end_capture():


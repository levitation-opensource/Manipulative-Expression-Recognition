# -*- coding: utf-8 -*-

#
# Author: Roland Pihlakas, 2023 - 2023
#
# roland@simplify.ee
#


if __name__ == '__main__':
  print("Starting...")


import os
import sys
import traceback
from collections import defaultdict, Counter, OrderedDict

from Recogniser import recogniser

import json_tricks


from Utilities import init_logging, safeprint, print_exception, loop, debugging, is_dev_machine, data_dir, Timer, read_file, save_file, read_raw, save_raw, read_txt, save_txt, strtobool
from TimeLimit import time_limit


# if __name__ == "__main__":
#   init_logging(os.path.basename(__file__), __name__, max_old_log_rename_tries = 1)


if __name__ == "__main__":
  os.chdir(os.path.dirname(os.path.realpath(__file__)))

if is_dev_machine:
  from pympler import asizeof




async def multi_file_label_filter(do_open_ended_analysis = None, do_closed_ended_analysis = None, extract_message_indexes = None, extract_line_numbers = None, argv = None):
  

  argv = argv if argv else sys.argv


  find_labels = set([ # TODO: find a way to put these labels on command line?
    # "Demanding",
    # "Creating artificial obligations",
    # "Not attempting to understand",
  ])


  all_error_msgs = []   # TODO
  all_expressions = []

  for current_file in argv[1:]:

    safeprint(f"Analysing file {current_file}...")

    current_argv = ["", current_file]

    analysis_response = await recogniser(do_open_ended_analysis, do_closed_ended_analysis, extract_message_indexes, extract_line_numbers, current_argv)

    error_code = analysis_response["error_code"]

    if error_code > 0:
      all_error_msgs.append(analysis_response["error_msg"])
    else:
      all_expressions.append((current_file, analysis_response["expressions"]), )

  #/ for current_file in argv[1:]:

  safeprint("All files done.")



  filtered_labels_response = []
  for current_file, person_expressions in all_expressions:
    for expression_data in person_expressions:

      labels = expression_data["labels"]
      labels_intersection = find_labels.intersection(labels.keys())
      if labels_intersection or not find_labels: # if find_labels is empty then mach all
        
        sorted_intersection = list(labels_intersection)
        sorted_intersection.sort()

        entry = {
          "filtered_labels": sorted_intersection,
          "all_labels": labels,
          "file": current_file,
          # TODO: add line number
          "text": expression_data["text"]
        }

        if "message_index" in expression_data:
          entry.update({ "message_index": expression_data["message_index"] })

        if "line_number" in expression_data:
          entry.update({ "line_number": expression_data["line_number"] })

        filtered_labels_response.append(entry)

      #/ if labels_intersection:
    #/ for expression_data in person_expressions:
  #/ for person_expressions in all_expressions:



  filtered_labels_response_json = json_tricks.dumps(filtered_labels_response, indent=2)   # json_tricks preserves dictionary orderings

  filtered_labels_response_filename = "filtered_labels.json"
  # filtered_labels_response_filename = os.path.join("..", filtered_labels_response_filename)   # the applications default data location is
  await save_txt(filtered_labels_response_filename, filtered_labels_response_json, quiet = True, make_backup = True, append = False)



  safeprint("Filtering done.")

#/ async def multi_file_label_filter()


if __name__ == '__main__':
  loop.run_until_complete(multi_file_label_filter())



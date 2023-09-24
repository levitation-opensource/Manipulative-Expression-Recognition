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




async def multi_file_recogniser(do_open_ended_analysis = None, do_closed_ended_analysis = None, extract_message_indexes = None, extract_line_numbers = None, argv = None):
  

  argv = argv if argv else sys.argv



  all_error_msgs = []   # TODO
  all_counts = []
  all_unexpected_labels = []
  all_unused_labels = []
  all_expressions = []

  for current_file in argv[1:]:

    safeprint(f"Analysing file {current_file}...")

    current_argv = ["", current_file]

    analysis_response = await recogniser(do_open_ended_analysis, do_closed_ended_analysis, extract_message_indexes, extract_line_numbers, current_argv)

    error_code = analysis_response["error_code"]

    if error_code > 0:
      all_error_msgs.append(analysis_response["error_msg"])
    else:
      all_counts.append(analysis_response["counts"])
      all_unexpected_labels.append(analysis_response["unexpected_labels"])
      all_unused_labels.append(analysis_response["unused_labels"])
      all_expressions.append((current_file, analysis_response["expressions"]), )

  #/ for current_file in argv[1:]:

  safeprint("All files done.")



  aggregated_counts = Counter()
  for counts in all_counts:
    for person, person_counts in counts.items():
      aggregated_counts += person_counts

  aggregated_unexpected_labels = Counter()
  for unexpected_labels in aggregated_unexpected_labels:
    for label in unexpected_labels:
      aggregated_unexpected_labels[label] += 1

  aggregated_counts = OrderedDict(aggregated_counts.most_common())
  aggregated_unexpected_labels = OrderedDict(aggregated_unexpected_labels.most_common())
  
  
  if len(all_unused_labels) == 0:
    aggregated_unused_labels = [] if do_closed_ended_analysis else None
  else:
    # this algorithm keeps the order of the unused labels list
    aggregated_unused_labels = all_unused_labels[0]
    for current_unused_labels in all_unused_labels[1:]:
      current_unused_labels_set = set(current_unused_labels) # optimisation
      aggregated_unused_labels = [x for x in aggregated_unused_labels if x in current_unused_labels_set]
    #/ for unused_labels in all_unused_labels[1:]:
  #/ if len(all_unused_labels) == 0:


  grouped_labels = OrderedDict()
  for grouped_label in aggregated_counts.keys():

    grouped_label_data = []

    for current_file, person_expressions in all_expressions:
      for expression_data in person_expressions:
        expression_labels = expression_data["labels"]
        if grouped_label in expression_labels.keys():
          
          entry = {
            "grouped_label": grouped_label,
            "all_labels": expression_labels,
            "file": current_file,
            "text": expression_data["text"]
          }

          if "message_index" in expression_data:
            entry.update({ "message_index": expression_data["message_index"] })

          if "line_number" in expression_data:
            entry.update({ "line_number": expression_data["line_number"] })

          grouped_label_data.append(entry)

        #/ if labels_intersection:
      #/ for expression_data in person_expressions:
    #/ for person_expressions in all_expressions:

    grouped_labels[grouped_label] = grouped_label_data

  #/ for label in aggregated_counts.keys():



  aggregated_analysis_response = {
    "counts": aggregated_counts,
    "unexpected_labels": aggregated_unexpected_labels,  # TODO: use dict with counts in single-file output too
    "unused_labels": aggregated_unused_labels,
    "grouped_labels": grouped_labels,
  }

  aggregated_response_json = json_tricks.dumps(aggregated_analysis_response, indent=2)   # json_tricks preserves dictionary orderings

  aggregated_response_filename = "aggregated_stats.json"
  # aggregated_response_filename = os.path.join("..", aggregated_response_filename)   # the applications default data location is
  await save_txt(aggregated_response_filename, aggregated_response_json, quiet = True, make_backup = True, append = False)



  safeprint("Aggregation done.")

#/ async def multi_file_recognise()


if __name__ == '__main__':
  loop.run_until_complete(multi_file_recogniser())



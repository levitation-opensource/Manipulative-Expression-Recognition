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




async def multi_file_recogniser(do_open_ended_analysis = None, do_closed_ended_analysis = None, extract_message_indexes = None, argv = None):
  

  argv = argv if argv else sys.argv



  all_error_msgs = []   # TODO
  all_counts = []
  all_unexpected_labels = []

  for current_file in argv[1:]:

    safeprint(f"Analysing file {current_file}...")

    current_argv = ["", current_file]

    analysis_response = await recogniser(do_open_ended_analysis, do_closed_ended_analysis, extract_message_indexes, current_argv)

    error_code = analysis_response["error_code"]

    if error_code > 0:
      all_error_msgs.append(analysis_response["error_msg"])
    else:
      all_counts.append(analysis_response["counts"])
      all_unexpected_labels.append(analysis_response["unexpected_labels"])

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



  aggregated_analysis_response = {
    "counts": aggregated_counts,
    "unexpected_labels": aggregated_unexpected_labels,
  }

  aggregated_response_json = json_tricks.dumps(aggregated_analysis_response, indent=2)   # json_tricks preserves dictionary orderings

  aggregated_response_filename = "aggregated_stats.json"
  # aggregated_response_filename = os.path.join("..", aggregated_response_filename)   # the applications default data location is
  await save_txt(aggregated_response_filename, aggregated_response_json, quiet = True, make_backup = True, append = False)



  safeprint("Aggregation done.")

#/ async def multi_file_recognise()


if __name__ == '__main__':
  loop.run_until_complete(multi_file_recogniser(extract_message_indexes = None))   # extract_message_indexes = None - use config file



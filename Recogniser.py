# -*- coding: utf-8 -*-

#
# Author: Roland Pihlakas, 2023 - 2023
#
# roland@simplify.ee
#


print("Starting...")

import os
import sys

# import spacy
# from spacy import displacy

import re
from collections import defaultdict, OrderedDict
import hashlib

import thefuzz.process

import json_tricks

# import openai
import tenacity   # for exponential backoff
import openai_async


organization = "org-oFLL9tXxrcM7Y5nKxrY871Kx"
api_key = os.getenv("OPENAI_API_KEY")

# openai.organization = organization
# openai.api_key = api_key


from Utilities import init_logging, safeprint, loop, debugging, is_dev_machine, data_dir, Timer, read_file, save_file, read_txt, save_txt

# init_logging(os.path.basename(__file__), __name__, max_old_log_rename_tries = 1)



if __name__ == "__main__":
  os.chdir(os.path.dirname(os.path.realpath(__file__)))

if is_dev_machine:
  from pympler import asizeof



## https://platform.openai.com/docs/guides/rate-limits/error-mitigation
@tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=60), stop=tenacity.stop_after_attempt(6))   # TODO: tune
async def completion_with_backoff(**kwargs):  # TODO: ensure that only HTTP 429 is handled here

  # return openai.ChatCompletion.create(**kwargs) 

  qqq = True  # for debugging

  openai_response = await openai_async.chat_complete(
    api_key,
    timeout = 60, # TODO: tune
    payload = kwargs
  )

  openai_response = json_tricks.loads(openai_response.text)

  # NB! this line may also throw an exception if the OpenAI announces that it is overloaded # TODO: do not retry for all error messages
  response_content = openai_response["choices"][0]["message"]["content"]
  finish_reason = openai_response["choices"][0]["finish_reason"]

  return (response_content, finish_reason)

#/ async def completion_with_backoff(**kwargs):


async def run_llm_analysis(messages, continuation_request, enable_cache = True):

  if enable_cache:
    # NB! this cache key and the cache will not contain the OpenAI key, so it is safe to publish the cache files
    cache_key = json_tricks.dumps(messages)   # json_tricks preserves dictionary orderings
    cache_key = hashlib.sha512(cache_key.encode("utf-8")).hexdigest() 

    # TODO: move this block to Utilities.py
    fulldirname = os.path.join(data_dir, "cache")
    os.makedirs(fulldirname, exist_ok = True)

    cache_filename = os.path.join("cache", "cache_" + cache_key + ".dat")
    response = await read_file(cache_filename, default_data = None, quiet = True)
  else:
    response = None


  if response is None:

    # model_name = "gpt-3.5-turbo"
    model_name = "gpt-3.5-turbo-16k"  # TODO: auto select model based on request length
    # model_name = "gpt-4"

    responses = []


    continue_analysis = True
    while continue_analysis:

      (response_content, finish_reason) = await completion_with_backoff(
        model = model_name,
        messages = messages,
          
        # functions = [],   # if no functions are in array then the array should be omitted, else error occurs
        # function_call = "none",   # 'function_call' is only allowed when 'functions' are specified
        n = 1,
        stream = False,   # TODO
        # user = "",    # TODO

        temperature = 0, # 1,   0 means deterministic output
        top_p = 1,
        max_tokens = 2048,
        presence_penalty = 0,
        frequency_penalty = 0,
        # logit_bias = None,
      )

      responses.append(response_content)
      too_long = (finish_reason == "length")

      if too_long:
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "assistant", "content": continuation_request})
      else:
        continue_analysis = False

    #/ while continue_analysis:

    response = "\n".join(responses)


    if enable_cache:
      await save_file(cache_filename, response, quiet = True)   # TODO: save request in cache too and compare it upon cache retrieval

  #/ if response is None:


  return response

#/ async def run_llm_analysis():


def sanitise_input(text):
  text = re.sub(r"[{\[]", "(", text)
  text = re.sub(r"[}\]]", ")", text)
  return text
#/ def sanitise_input(text):


async def main(do_open_ended_analysis = True, do_closed_ended_analysis = True):


  #if False:   # TODO: NER

  #  NER = spacy.load("en_core_web_sm")
  #  raw_text = "The Indian Space Research Organisation is the national space agency of India"
  #  text1 = NER(raw_text)


  labels_filename = sys.argv[3] if len(sys.argv) > 3 else None
  if labels_filename:
    labels_filename = os.path.join("..", labels_filename)   # the applications default data location is in folder "data", but in case of user provided files lets expect the files in the same folder than the main script
  else:
    labels_filename = "default_labels.txt"


  closed_ended_system_instruction = (await read_txt("closed_ended_system_instruction.txt", quiet = True)).strip()
  open_ended_system_instruction = (await read_txt("open_ended_system_instruction.txt", quiet = True)).strip()
  default_labels = (await read_txt(labels_filename, quiet = True)).strip()
  continuation_request = (await read_txt("continuation_request.txt", quiet = True)).strip()


  closed_ended_system_instruction_with_labels = closed_ended_system_instruction.replace("%default_labels%", default_labels)



  # read user input
  input_filename = sys.argv[1] if len(sys.argv) > 1 else None
  if input_filename:
    input_filename = os.path.join("..", input_filename)   # the applications default data location is in folder "data", but in case of user provided files lets expect the files in the same folder than the main script
    using_user_input_filename = True
  else:    
    input_filename = "test_input.txt"
    using_user_input_filename = False

  # format user input
  user_input = (await read_txt(input_filename, quiet = True)).strip()

  # sanitise user input since []{} have special meaning in the output parsing
  user_input = sanitise_input(user_input)



  # parse labels    # TODO: functionality for adding comments to the labels file which will be not sent to LLM
  labels_list = []
  lines = default_labels.splitlines(keepends=False)
  for line in lines:

    line = line.strip()
    if line[0] == "-":
      line = line[1:].strip()

    line = sanitise_input(line)
    line = re.sub(r"[.,:;]+", "/", line).strip()  # remove punctuation from labels

    if len(line) == 0:
      continue

    labels_list.append(line)

  #/ for line in default_labels.splitlines(keepends=False):

  default_labels = "\n".join("- " + x for x in labels_list)



  # call the analysis function

  if do_open_ended_analysis:

    messages = [
      {"role": "system", "content": open_ended_system_instruction},
      {"role": "user", "content": user_input},
      # {"role": "assistant", "content": "Who's there?"},
      # {"role": "user", "content": "Orange."},
    ]

    open_ended_response = await run_llm_analysis(messages, continuation_request)

  else: #/ if do_open_ended_analysis:

    open_ended_response = None
  


  if do_closed_ended_analysis:

    messages = [
      {"role": "system", "content": closed_ended_system_instruction_with_labels},
      {"role": "user", "content": user_input},
      # {"role": "assistant", "content": "Who's there?"},
      # {"role": "user", "content": "Orange."},
    ]

    closed_ended_response = await run_llm_analysis(messages, continuation_request)


    # parse the closed ended response by extracting persons, citations, and labels

    expressions_tuples = []
    detected_persons = set()
    unexpected_labels = set()

    re_matches = re.findall(r"[\r\n]+\[(.*)\]:(.*)\{(.*)\}", "\n" + closed_ended_response)
    for re_match in re_matches:

      (person, citation, labels) = re_match
      citation = citation.strip()
      if citation[-1] == "-":   # remove the dash between the citation and labels. Note that ChatGPT may omit this dash even if it is present in the instruction
        citation = citation[0:-1].strip()

      detected_persons.add(person)


      labels = [x.strip() for x in labels.split(",")]

      for x in labels:  # create debug info
        if x not in labels_list:
          unexpected_labels.add(x)

      labels = [x for x in labels if x in labels_list]  # throw away any non-requested labels
      
      if len(labels) == 0:
        continue

      expressions_tuples.append((person, citation, labels), )

    #/ for re_match in re_matches:


    # split input text into messages, compute the locations of messages

    person_messages = {person: [] for person in detected_persons}
    overall_message_indexes = {person: {} for person in detected_persons}  
    start_char_to_person_message_index_dict = {}
    person_message_spans = {person: [] for person in detected_persons}

    for person in detected_persons:
      p = re.compile(r"[\r\n]+" + re.escape(person) + r":(.*)")
      re_matches = p.finditer("\n" + user_input)    # TODO: ensure that ":" character inside messages does not mess the analysis up
      for re_match in re_matches:

        message = re_match.group(1)
        
        message = message.rstrip()

        len_before_lstrip = len(message)
        message = message.lstrip()
        num_stripped_leading_chars = len_before_lstrip - len(message)

        start_char = re_match.start(1) + num_stripped_leading_chars - 1    # -1 for the prepended \n
        end_char = start_char + len(message)

        start_char_to_person_message_index_dict[start_char] = (person, len(person_messages[person]))

        person_messages[person].append(message)
        person_message_spans[person].append((start_char, end_char), )

      #/ for re_match in re_matches:
    #/ for person in detected_persons:


    # sort message indexes by start_char
    start_char_to_person_message_index_dict = OrderedDict(sorted(start_char_to_person_message_index_dict.items()))

    # compute overall message index for each person's message index
    for overall_message_index, entry in enumerate(start_char_to_person_message_index_dict.values()):
      (person, person_message_index) = entry
      overall_message_indexes[person][person_message_index] = overall_message_index


    # compute expression locations
    totals = defaultdict(lambda: defaultdict(int))    # defaultdict: do not report persons and labels which are not detected
    # totals = {person: {label: 0 for label in labels_list} for person in detected_persons}
    expression_dicts = []

    for expressions_tuple in expressions_tuples:

      (person, citation, labels) = expressions_tuple
      
      # TODO: use combinatorial optimisation to do the matching with original text positions in order to ensure that repeated similar expressions get properly located as well
      nearest_message = thefuzz.process.extractOne(citation, person_messages[person])[0]
      person_message_index = person_messages[person].index(nearest_message)
      overall_message_index = overall_message_indexes[person][person_message_index]  

      start_char = person_message_spans[person][person_message_index][0] + nearest_message.find(citation)  # TODO: allow fuzzy matching in case spaces or punctuation differ
      end_char = start_char + len(citation)


      for label in labels:
        totals[person][label] += 1


      expression_dicts.append({
        "person": person,
        "start_char": start_char,
        "end_char": end_char,
        "start_message": overall_message_index, 
        "end_message": overall_message_index,   
        "text": nearest_message,  # citation,   # use original text not ChatGPT citation here
        "labels": labels,
      })
      
    #/ for expressions_tuple in expressions_tuples:


    # need to sort since ChatGPT sometimes does not cite the expressions in the order they appear originally
    expression_dicts.sort(key = lambda entry: entry["start_char"])

    # create new tuples list since the original list is potentially unordered
    expressions_tuples_out = [(entry["person"], entry["text"], entry["labels"]) for entry in expression_dicts]


    qqq = True  # for debugging

  else:   #/ if do_closed_ended_analysis:

    closed_ended_response = None
    totals = None
    expression_dicts = None
    expressions_tuples = None
    unexpected_labels = None

  #/ if do_closed_ended_analysis:



  analysis_response = {

    "error_code": 0,  # TODO
    "error_msg": "",  # TODO

    "sanitised_text": user_input,
    "expressions": expression_dicts,
    "expressions_tuples": expressions_tuples_out,
    "counts": totals,
    "unexpected_labels": list(unexpected_labels),   # convert set() to list() for enabling conversion into json
    "raw_expressions_labeling_response": closed_ended_response,
    "qualitative_evaluation": open_ended_response
  }

  response_json = json_tricks.dumps(analysis_response, indent=2)   # json_tricks preserves dictionary orderings

  response_filename = sys.argv[2] if len(sys.argv) > 2 else None
  if response_filename:
    response_filename = os.path.join("..", response_filename)   # the applications default data location is in folder "data", but in case of user provided files lets expect the files in the same folder than the main script
  else: 
    response_filename = os.path.splitext(input_filename)[0] + "_evaluation.json" if using_user_input_filename else "test_evaluation.json"

  await save_txt(response_filename, response_json, quiet = True, make_backup = True, append = False)

  

  qqq = True  # for debugging

#/ async def main():



loop.run_until_complete(main())



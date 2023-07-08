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
# from spacy import displacy    # load it only when rendering is requested, since this package loads slowly

import re
from collections import defaultdict, OrderedDict
import hashlib

import thefuzz.process

import json_tricks

# import openai
import tenacity   # for exponential backoff
import openai_async


# organization = os.getenv("OPENAI_API_ORG")
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
    # model_name = "gpt-4"  # TODO: command-line argument or a config file for selecting model

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


async def main(do_open_ended_analysis = True, do_closed_ended_analysis = True, extract_message_indexes = False):


  #if False:   # TODO: NER

  #  NER = spacy.load("en_core_web_sm")
  #  raw_text = "The Indian Space Research Organisation is the national space agency of India"
  #  text1 = NER(raw_text)


  labels_filename = sys.argv[3] if len(sys.argv) > 3 else None
  if labels_filename:
    labels_filename = os.path.join("..", labels_filename)   # the applications default data location is in folder "data", but in case of user provided files lets expect the files in the same folder than the main script
  else:
    labels_filename = "default_labels.txt"


  closed_ended_system_instruction = (await read_txt("closed_ended_system_instruction.txt", quiet = True)).lstrip()   # NB! use .lstrip() here
  open_ended_system_instruction = (await read_txt("open_ended_system_instruction.txt", quiet = True)).lstrip()   # NB! use .lstrip() here
  extract_names_of_participants_system_instruction = (await read_txt("extract_names_of_participants_system_instruction.txt", quiet = True)).lstrip()   # NB! use .lstrip() here
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

    # TODO: Sometimes GPT mixes up the person names in the citations. Ensure that such incorrectly assigned citations are ignored or properly re-assigned in this script.

    expressions_tuples = []
    detected_persons = set()
    unexpected_labels = set()

    if extract_message_indexes:   # TODO

      messages = [
        {"role": "system", "content": extract_names_of_participants_system_instruction},
        {"role": "user", "content": user_input},
        # {"role": "assistant", "content": "Who's there?"},
        # {"role": "user", "content": "Orange."},
      ]

      names_of_participants_response = await run_llm_analysis(messages, continuation_request)

      # Try to detect persons from the input text. This is necessary for preserving proper message indexing in the output in case some person is not cited by LLM at all.
      re_matches = re.findall(r"[\r\n]+\[?([^\]\n]*)\]?", "\n" + names_of_participants_response)
      for re_match in re_matches:

        detected_persons.add(re_match)

      #/ for re_match in re_matches:
    #/ if extract_message_indexes:

    # Process LLM response
    re_matches = re.findall(r"[\r\n]+\[(.*)\]:(.*)\{(.*)\}", "\n" + closed_ended_response)
    for re_match in re_matches:

      (person, citation, labels) = re_match
      citation = citation.strip()
      if citation[-1] == "-":   # remove the dash between the citation and labels. Note that ChatGPT may omit this dash even if it is present in the instruction
        citation = citation[0:-1].strip()

      detected_persons.add(person)


      labels = [x.strip() for x in labels.split(",")]

      for x in labels:  # create debug info about non-requested labels
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
    # already_labelled_message_indexes = set()
    already_labelled_message_parts = set()

    for expressions_tuple in expressions_tuples:

      (person, citation, labels) = expressions_tuple
      

      # TODO: use combinatorial optimisation to do the matching with original text positions in order to ensure that repeated similar expressions get properly located as well

      # find nearest actual message and verify that it was expressed by the same person as in LLM's citation
      nearest_message = None 
      nearest_message_similarity = 0
      nearest_person = None
      for person2 in detected_persons:

        (person2_nearest_message, similarity) = thefuzz.process.extractOne(citation, person_messages[person2])

        if (similarity > nearest_message_similarity 
          or (similarity == nearest_message_similarity and person2 == person)):  # if multiple original messages have same similarity score then prefer the message with a person that was assigned by LLM
          nearest_message_similarity = similarity
          nearest_message = person2_nearest_message
          nearest_person = person2

      #/ for person2 in persons:

      if nearest_person != person:  # incorrectly assigned citation 
        if True:     # TODO: configuration option for handling such cases 
          continue
        else:
          person = nearest_person    


      person_message_index = person_messages[person].index(nearest_message)
      overall_message_index = overall_message_indexes[person][person_message_index]  

      #if overall_message_index in already_labelled_message_indexes:
      #  continue    # ignore repeated citations.  # TODO: if repeated citations have different labels, take labels from all of citations
      #else:
      #  already_labelled_message_indexes.add(overall_message_index)

      if nearest_message in already_labelled_message_parts:
        continue  # TODO: if repeated citations have different labels, take labels from all of citations
      else:
        already_labelled_message_parts.add(nearest_message)


      start_char = person_message_spans[person][person_message_index][0] + nearest_message.find(citation)  # TODO: allow fuzzy matching in case spaces or punctuation differ
      end_char = start_char + len(citation)


      labels.sort()

      for label in labels:
        totals[person][label] += 1


      entry = {
        "person": person,
        "start_char": start_char,
        "end_char": end_char,
        # "start_message": overall_message_index, 
        # "end_message": overall_message_index,   
        "text": nearest_message,  # citation,   # use original text not ChatGPT citation here
        "labels": labels,
      }

      if extract_message_indexes:
        # overall_message_index = overall_message_indexes[person][person_message_index]  
        entry.update({
          "start_message": overall_message_index, 
          "end_message": overall_message_index,   
        })
      #/ if extract_message_indexes:

      expression_dicts.append(entry)
      
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
    "qualitative_evaluation": open_ended_response   # TODO: split person A and person B from qualitative description into separate dictionary fields
  }

  response_json = json_tricks.dumps(analysis_response, indent=2)   # json_tricks preserves dictionary orderings

  response_filename = sys.argv[2] if len(sys.argv) > 2 else None
  if response_filename:
    response_filename = os.path.join("..", response_filename)   # the applications default data location is in folder "data", but in case of user provided files lets expect the files in the same folder than the main script
  else: 
    response_filename = os.path.splitext(input_filename)[0] + "_evaluation.json" if using_user_input_filename else "test_evaluation.json"

  await save_txt(response_filename, response_json, quiet = True, make_backup = True, append = False)

  

  render_output = True     # TODO: work in progress
  if render_output:

    def create_html_barchart(person, person_counts):

      max_value = max(person_counts.values())
      html =  "<div style='font-weight: bold;'>" + person + ":</div>\n"
      html += "<div style='width: 400px; height: 400px; display: flex;'>\n"

      for label, value in person_counts.items():
          height = (value / max_value) * 100  # normalize to percentage
          html += f"<div style='width: {100/len(person_counts)}%; height: {height}%; background: #1f77b4; margin-right: 5px;'>{label}<br>{value}</div>\n"

      html += "</div>\n<br><br>\n"
      return html

    #/ def create_html_barchart(data):

    bar_chart = [create_html_barchart(person, person_counts) for (person, person_counts) in totals.items()]


    print("Loading Spacy HTML renderer...")
    from spacy import displacy    # load it only when rendering is requested, since this package loads 

    highlights_html = displacy.render( 
            {
                "text": user_input,
                "ents": [
                          {
                            "start": entry["start_char"], 
                            "end": entry["end_char"], 
                            "label": ", ".join(entry["labels"])
                          } 
                          for entry 
                          in expression_dicts
                        ],
                "title": None 
            }, 
            style="ent", manual=True
          )

    html = ('<html><body><style>                    \
                .entities {                         \
                  line-height: 1.5 !important;      \
                }                                   \
                mark {                              \
                  line-height: 2 !important;        \
                  background: yellow !important;    \
                }                                   \
                mark span {                         \
                  background: orange !important;    \
                  padding: 0.5em;                   \
                }                                   \
            </style>' 
            # + bar_chart     # TODO
            + '<div style="font: 1em Arial;">' 
            + highlights_html 
            + '</div></body></html>')


    response_html_filename = sys.argv[4] if len(sys.argv) > 4 else None
    if response_html_filename:
      response_html_filename = os.path.join("..", response_html_filename)   # the applications default data location is in folder "data", but in case of user provided files lets expect the files in the same folder than the main script
    else: 
      response_html_filename = os.path.splitext(response_filename)[0] + ".html" if using_user_input_filename else "test_evaluation.html"

    await save_txt(response_html_filename, html, quiet = True, make_backup = True, append = False)

  #/ if render_output:

  


  qqq = True  # for debugging

#/ async def main():



loop.run_until_complete(main(extract_message_indexes = False))



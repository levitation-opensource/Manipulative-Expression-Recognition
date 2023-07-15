# -*- coding: utf-8 -*-

#
# Author: Roland Pihlakas, 2023 - 2023
#
# roland@simplify.ee
#


print("Starting...")

import os
import sys
import traceback

from configparser import ConfigParser

# import spacy
# from spacy import displacy    # load it only when rendering is requested, since this package loads slowly

import re
from collections import defaultdict, OrderedDict
import hashlib
import string

import thefuzz.process

import json_tricks

# import openai
import tenacity   # for exponential backoff
import openai_async


# organization = os.getenv("OPENAI_API_ORG")
api_key = os.getenv("OPENAI_API_KEY")

# openai.organization = organization
# openai.api_key = api_key


from Utilities import init_logging, safeprint, print_exception, loop, debugging, is_dev_machine, data_dir, Timer, read_file, save_file, read_raw, save_raw, read_txt, save_txt, strtobool

# init_logging(os.path.basename(__file__), __name__, max_old_log_rename_tries = 1)




if __name__ == "__main__":
  os.chdir(os.path.dirname(os.path.realpath(__file__)))

if is_dev_machine:
  from pympler import asizeof



def remove_quotes(text):
  return text.replace("'", "").replace('"', '')


def rotate_list(list, n):
  return list[n:] + list[:n]


def get_config():

  config = ConfigParser(inline_comment_prefixes=("#", ";"))  # by default, inline comments were not allowed
  config.read('MER.ini')


  gpt_model = remove_quotes(config.get("MER", "GPTModel", fallback="gpt-3.5-turbo-16k"))
  extract_message_indexes = strtobool(config.get("MER", "ExtractMessageIndexes", fallback="false"))
  do_open_ended_analysis = strtobool(config.get("MER", "DoOpenEndedAnalysis", fallback="true"))
  do_closed_ended_analysis = strtobool(config.get("MER", "DoClosedEndedAnalysis", fallback="true"))
  keep_unexpected_labels = strtobool(config.get("MER", "KeepUnexpectedLabels", fallback="true"))
  chart_type = remove_quotes(config.get("MER", "ChartType", fallback="radar"))
  render_output = strtobool(config.get("MER", "RenderOutput", fallback="true"))
  treat_entire_text_as_one_person = strtobool(config.get("MER", "TreatEntireTextAsOnePerson", fallback="false"))  # TODO
  anonymise_names = strtobool(config.get("MER", "AnonymiseNames", fallback="false"))
  anonymise_numbers = strtobool(config.get("MER", "AnonymiseNumbers", fallback="false"))
  named_entity_recognition_model = remove_quotes(config.get("MER", "NamedEntityRecognitionModel", fallback="en_core_web_sm"))


  result = { 
    "gpt_model": gpt_model,
    "extract_message_indexes": extract_message_indexes,
    "do_open_ended_analysis": do_open_ended_analysis,
    "do_closed_ended_analysis": do_closed_ended_analysis,
    "keep_unexpected_labels": keep_unexpected_labels,
    "chart_type": chart_type,
    "render_output": render_output,
    "treat_entire_text_as_one_person": treat_entire_text_as_one_person,
    "anonymise_names": anonymise_names,
    "anonymise_numbers": anonymise_numbers,
    "named_entity_recognition_model": named_entity_recognition_model,
  }

  return result

#/ get_config()


## https://platform.openai.com/docs/guides/rate-limits/error-mitigation
@tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=60), stop=tenacity.stop_after_attempt(6))   # TODO: tune
async def completion_with_backoff(**kwargs):  # TODO: ensure that only HTTP 429 is handled here

  # return openai.ChatCompletion.create(**kwargs) 

  qqq = True  # for debugging

  try:

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

  except Exception as ex:

    msg = str(ex) + "\n" + traceback.format_exc()
    print_exception(msg)
    raise

#/ async def completion_with_backoff(**kwargs):


async def run_llm_analysis(model_name, messages, continuation_request, enable_cache = True):

  if enable_cache:
    # NB! this cache key and the cache will not contain the OpenAI key, so it is safe to publish the cache files
    cache_key = OrderedDict([ 
      ("model_name", model_name), 
      ("messages", messages), 
    ])
    cache_key = json_tricks.dumps(cache_key)   # json_tricks preserves dictionary orderings
    cache_key = hashlib.sha512(cache_key.encode("utf-8")).hexdigest() 

    # TODO: move this block to Utilities.py
    fulldirname = os.path.join(data_dir, "cache")
    os.makedirs(fulldirname, exist_ok = True)

    cache_filename = os.path.join("cache", "cache_" + cache_key + ".dat")
    response = await read_file(cache_filename, default_data = None, quiet = True)
  else:
    response = None


  if response is None:

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
        messages.append({"role": "assistant", "content": continuation_request})   # TODO: test this functionality
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


def anonymise(user_input, anonymise_names, anonymise_numbers, ner_model):

  safeprint("Loading Spacy...")
  import spacy
  NER = spacy.load(ner_model)   # TODO: config setting

  entities = NER(user_input)
  letters = string.ascii_uppercase

  next_available_replacement_letter_index = 0
  result = ""
  prev_ent_end = 0
  entities_dict = {}

  for phase in range(0, 2): # two phases: 1) counting unique entities, 2) replacing them
    for word in entities.ents:

      text = word.text
      label = word.label_
      start_char = word.start_char
      end_char = word.end_char

      if label == "PERSON":
        replacement = "Person" if anonymise_names else None
      elif label == "NORP":
        replacement = "Group" if anonymise_names else None
      elif label == "FAC":
        replacement = "Building" if anonymise_names else None
      elif label == "ORG":
        replacement = "Organisation" if anonymise_names else None
      elif label == "GPE":
        replacement = "Area" if anonymise_names else None
      elif label == "LOC":
        replacement = "Location" if anonymise_names else None
      elif label == "PRODUCT":
        replacement = None  # "Product"
      elif label == "EVENT":
        replacement = "Event" if anonymise_names else None
      elif label == "WORK_OF_ART":
        replacement = None  # "Work of art"
      elif label == "LAW":
        replacement = None  # "Law"
      elif label == "LANGUAGE":
        replacement = "Language" if anonymise_names else None
      elif label == "DATE":
        replacement = None  # "Calendar Date" if anonymise_numbers else None   # TODO: recognise only calendar dates, not phrases like "a big day", "today", etc
      elif label == "TIME":
        replacement = None  # "Time"
      elif label == "PERCENT":
        replacement = None  # "Percent"
      elif label == "MONEY":
        replacement = "Money Amount" if anonymise_numbers else None
      elif label == "QUANTITY":
        replacement = "Quantity" if anonymise_numbers else None
      elif label == "ORDINAL":
        replacement = None  # "Ordinal"
      elif label == "CARDINAL":
        replacement = "Number" if ananonymise_numbers else None
      else:
        replacement = None


      if phase == 1:
        result += user_input[prev_ent_end:start_char]
        prev_ent_end = end_char


      if replacement is None:

        if phase == 1:
          result += text 

      else:

        if phase == 0:

          if text not in entities_dict:
            entities_dict[text] = next_available_replacement_letter_index          
            next_available_replacement_letter_index += 1

        else:   #/ if phase == 0:

          replacement_letter_index = entities_dict[text]

          if next_available_replacement_letter_index <= len(letters):
            replacement_letter = letters[replacement_letter_index]
          else:
            replacement_letter = str(replacement_letter_index + 1)  # use numeric names if there are too many entities in text to use letters

          result += replacement + " " + replacement_letter

        #/ if phase == 0:

      #/ if replacement is None:

    #/ for word in entities.ents:
  #/ for phase in range(0, 2):


  result += user_input[prev_ent_end:]


  return result

#/ def anonymise()


async def main(do_open_ended_analysis = None, do_closed_ended_analysis = None, extract_message_indexes = None):



  config = get_config()

  if do_open_ended_analysis is None:
    do_open_ended_analysis = config["do_open_ended_analysis"]
  if do_closed_ended_analysis is None:
    do_closed_ended_analysis = config["do_closed_ended_analysis"]
  if extract_message_indexes is None:
    extract_message_indexes = config["extract_message_indexes"]

  gpt_model = config["gpt_model"]


  labels_filename = sys.argv[3] if len(sys.argv) > 3 else None
  if labels_filename:
    labels_filename = os.path.join("..", labels_filename)   # the applications default data location is in folder "data", but in case of user provided files lets expect the files in the same folder than the main script
  else:
    labels_filename = "default_labels.txt"


  closed_ended_system_instruction = (await read_txt("closed_ended_system_instruction.txt", quiet = True)).lstrip()   # NB! use .lstrip() here
  open_ended_system_instruction = (await read_txt("open_ended_system_instruction.txt", quiet = True)).lstrip()   # NB! use .lstrip() here
  extract_names_of_participants_system_instruction = (await read_txt("extract_names_of_participants_system_instruction.txt", quiet = True)).lstrip()   # NB! use .lstrip() here
  all_labels_as_text = (await read_txt(labels_filename, quiet = True)).strip()
  continuation_request = (await read_txt("continuation_request.txt", quiet = True)).strip()


  closed_ended_system_instruction_with_labels = closed_ended_system_instruction.replace("%labels%", all_labels_as_text)



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


  anonymise_names = config.get("anonymise_names")
  anonymise_numbers = config.get("anonymise_numbers")
  ner_model = config.get("named_entity_recognition_model")

  if anonymise_names or anonymise_numbers:
    user_input = anonymise(user_input, anonymise_names, anonymise_numbers, ner_model)


  # sanitise user input since []{} have special meaning in the output parsing
  user_input = sanitise_input(user_input)



  # parse labels    # TODO: functionality for adding comments to the labels file which will be not sent to LLM
  labels_list = []
  lines = all_labels_as_text.splitlines(keepends=False)
  for line in lines:

    line = line.strip()
    if line[0] == "-":
      line = line[1:].strip()

    line = sanitise_input(line)
    line = re.sub(r"[.,:;]+", "/", line).strip()  # remove punctuation from labels

    if len(line) == 0:
      continue

    labels_list.append(line)

  #/ for line in all_labels_as_text.splitlines(keepends=False):

  all_labels_as_text = "\n".join("- " + x for x in labels_list)
  # labels_list.sort()



  # call the analysis function

  if do_open_ended_analysis:

    messages = [
      {"role": "system", "content": open_ended_system_instruction},
      {"role": "user", "content": user_input},
      # {"role": "assistant", "content": "Who's there?"},
      # {"role": "user", "content": "Orange."},
    ]

    # TODO: add temperature parameter

    open_ended_response = await run_llm_analysis(gpt_model, messages, continuation_request)

  else: #/ if do_open_ended_analysis:

    open_ended_response = None
  


  if do_closed_ended_analysis:

    messages = [
      {"role": "system", "content": closed_ended_system_instruction_with_labels},
      {"role": "user", "content": user_input},
      # {"role": "assistant", "content": "Who's there?"},
      # {"role": "user", "content": "Orange."},
    ]

    # TODO: add temperature parameter

    closed_ended_response = await run_llm_analysis(gpt_model, messages, continuation_request)


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

      names_of_participants_response = await run_llm_analysis(gpt_model, messages, continuation_request)

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

      for label in labels:  # create debug info about non-requested labels
        if label not in labels_list:

          unexpected_labels.add(label)

          if config.get("keep_unexpected_labels"):
            labels_list.append(label)

        #/ if label not in labels_list:
      #/ for label in labels:

      if not config.get("keep_unexpected_labels"):
        labels = [x for x in labels if x in labels_list]  # throw away any non-requested labels
      
      if len(labels) == 0:
        continue

      expressions_tuples.append((person, citation, labels), )

    #/ for re_match in re_matches:


    labels_list.sort()


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
    totals = defaultdict(lambda: OrderedDict([(label, 0) for label in labels_list]))    # defaultdict: do not report persons and labels which are not detected  # need to initialise the inner dict to preserve proper label ordering
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


      for label in labels:
        totals[person][label] += 1
        

      labels.sort()

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
    # expressions_tuples_out = [(entry["person"], entry["text"], entry["labels"]) for entry in expression_dicts]


    qqq = True  # for debugging

  else:   #/ if do_closed_ended_analysis:

    closed_ended_response = None
    totals = None
    expression_dicts = None
    # expressions_tuples = None
    unexpected_labels = None

  #/ if do_closed_ended_analysis:


  # cleanup zero valued counts in totals
  for (person, person_counts) in totals.items():
    totals[person] = OrderedDict([(key, value) for (key, value) in person_counts.items() if value > 0])


  analysis_response = {

    "error_code": 0,  # TODO
    "error_msg": "",  # TODO

    "sanitised_text": user_input,
    "expressions": expression_dicts,
    # "expressions_tuples": expressions_tuples_out,
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

  

  render_output = config.get("render_output") 
  if render_output:


    chart_type = config.get("chart_type")


    title = "Manipulative Expression Recognition (MER)"


    import pygal
    import pygal.style
  
  
    # creating the chart object
    style = pygal.style.DefaultStyle(
      foreground = "rgba(0, 0, 0, .87)",              # dimension labels, plot line legend, and numbers
      foreground_strong = "rgba(128, 128, 128, 1)",   # title and radial lines
      # foreground_subtle = "rgba(128, 255, 128, .54)", # ?
      guide_stroke_color = "#404040",                 # circular lines
      major_guide_stroke_color = "#808080",           # major circular lines
      stroke_width = 2,                               # plot line width
    )


    if chart_type == "radar":
      chart = pygal.Radar(style=style, order_min=0)   # order_min specifies scale step in log10
      reverse_labels_order = True
      shift_labels_order_left = True
    elif chart_type == "vbar":
      chart = pygal.Bar(style=style, order_min=0)   # order_min specifies scale step in log10
      reverse_labels_order = False
      shift_labels_order_left = False
    elif chart_type == "hbar":
      chart = pygal.HorizontalBar(style=style, order_min=0)   # order_min specifies scale step in log10
      reverse_labels_order = True
      shift_labels_order_left = False
    else:
      chart = None
      reverse_labels_order = False
      shift_labels_order_left = False
        

    # keep only labels which have nonzero values for at least one person
    nonzero_labels_list = []  # this variable needs to be computed even when chart is not rendered, since if the nonzero_labels_list is empty then a message is shown that no manipulation was detected
    if True:
      for label in labels_list:
        for (person, person_counts) in totals.items():
          count = person_counts.get(label, 0)
          if count > 0:
            nonzero_labels_list.append(label)
            break   # go to next label
        #/ for (person, person_counts) in totals.items():
      #/ for label in labels_list:

  
    if chart:

      x_labels = list(nonzero_labels_list) # make copy since it may be reversed
      if reverse_labels_order:
        x_labels.reverse()
      if shift_labels_order_left:
        x_labels = rotate_list(x_labels, -1)  # rotate list left by one so that the first label appear at 12:00 on the radar chart

      chart.title = title      
      chart.x_labels = x_labels

      # series_dict = {}
      if True:
        for (person, person_counts) in totals.items():
          series = [person_counts.get(label, 0) for label in x_labels]
          chart.add(person, series)
          # series_dict[person] = series

      # for (person, series) in series_dict.items():
      #   chart.add(person, series)
  

      svg = chart.render()
      # chart.render_to_png(render_to_png='chart.png')


      response_svg_filename = sys.argv[5] if len(sys.argv) > 5 else None
      if response_svg_filename:
        response_svg_filename = os.path.join("..", response_svg_filename)   # the applications default data location is in folder "data", but in case of user provided files lets expect the files in the same folder than the main script
      else: 
        response_svg_filename = os.path.splitext(response_filename)[0] + ".svg" if using_user_input_filename else "test_evaluation.svg"

      await save_raw(response_svg_filename, svg, quiet = True, make_backup = True, append = False)

    #/ if chart:


    safeprint("Analysis done.")
    safeprint("Loading Spacy HTML renderer...")
    from spacy import displacy    # load it only when rendering is requested, since this package loads 
    import html
    import urllib.parse

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

    html = ('<html>\n<title>'
            + html.escape(title)
            + '</title>\n<body>'
            + """\n<style>
                .entities {
                  line-height: 1.5 !important;
                }
                mark {
                  line-height: 2 !important;
                  background: yellow !important;
                }
                mark span {
                  background: orange !important;
                  padding: 0.5em;
                }
                .graph object {
                  max-height: 75vh;
                }
              </style>""" 
            + (                
                (
                  ('\n<div class="graph"><object data="' + urllib.parse.quote_plus(response_svg_filename) + '" type="image/svg+xml"></object></div>')
                  if chart else
                  ''
                ) 
                if len(nonzero_labels_list) > 0 else                 
                '\n<div style="font: bold 1em Arial;">No manipulative expressions detected.</div>\n<br><br><br>'
              )
            + '\n<div style="font: bold 1em Arial;">Qualitative summary:</div><br>'
            + '\n<div style="font: 1em Arial;">' 
            + '\n' + open_ended_response 
            + '\n</div>'
            + '\n<br><br><br>'
            + '\n<div style="font: bold 1em Arial;">Labelled input:</div><br>'
            + '\n<div style="font: 1em Arial;">' 
            + '\n' + highlights_html 
            + '\n</div>\n</body>\n</html>')


    response_html_filename = sys.argv[4] if len(sys.argv) > 4 else None
    if response_html_filename:
      response_html_filename = os.path.join("..", response_html_filename)   # the applications default data location is in folder "data", but in case of user provided files lets expect the files in the same folder than the main script
    else: 
      response_html_filename = os.path.splitext(response_filename)[0] + ".html" if using_user_input_filename else "test_evaluation.html"

    await save_txt(response_html_filename, html, quiet = True, make_backup = True, append = False)

  #/ if render_output:

  


  qqq = True  # for debugging

#/ async def main():



loop.run_until_complete(main(extract_message_indexes = None))   # extract_message_indexes = None - use config file



# Manipulative Expression Recognition (MER) and Manipulativeness Benchmark


## Introduction

In the rapidly evolving world of artificial intelligence, the rise of large language models like OpenAI's GPT series has brought about profound shifts in digital communication. These models, capable of generating human-like text, have widespread applications ranging from content creation to customer service. As of 2023, their influence is undeniable and pervasive, extending even into areas such as personalized education and virtual companionship.

However, with great capability comes an inherent need for responsibility and scrutiny. Ensuring alignment with human values and understanding the underlying communicative tendencies of these models is paramount. Specifically, evaluating and benchmarking their potential for manipulative expressions becomes a crucial task.

At the same time, in the human realm of communication, manipulative behaviors can significantly impact interpersonal relationships, business negotiations, politics, and many other areas of life. These behaviors often go unnoticed or unrecognized, leaving victims of manipulation without the support and tools needed to defend themselves.

It is against this backdrop that our new software, "Manipulative Expression Recognition (MER) and Manipulativeness Benchmark," comes into the picture.


## Functionality

MER is designed to provide a comprehensive solution to the challenges mentioned above. It is a software library that allows users to upload transcripts of conversations or individual messages. The software then analyzes the text, applying labels that indicate potential manipulative communication styles.

This tool offers two main use cases:

1. Large Language Model Evaluation: As more sophisticated models continue to emerge, the need for tools to measure their alignment with human values grows concurrently. MER enables developers and researchers to evaluate and benchmark language model outputs for potential manipulative expressions. This can help inform adjustments and improvements to these models, promoting transparency, safety, and ethical considerations in AI development.

2. Human Communication Analysis: The application of MER extends beyond AI. By analyzing human-to-human conversations, MER can help identify manipulative behaviors and patterns. This capability can provide critical support to individuals who may be victims of manipulation, raising awareness and facilitating the development of counter-strategies.

In the future, the plan is to expand MER’s reach by offering it as a Software as a Service (SaaS) solution. Users will be able to access its functionalities via a web-based JSON API and a user-friendly graphical interface.

The vision for MER is to foster a culture of communication that is more transparent, equitable, and free from manipulation. We believe that by illuminating the nuances of language, we can contribute to a better understanding between AI and humans, as well as among humans themselves.


## Use cases

* Benchmarking of new LLM models:
    * Benchmarking LLM resistance to manipulation from user. Even if the user input is manipulative, the LLM output should be not manipulative.
    * Benchmarking LLM outputs for presence of manipulation in case of benign user inputs.
* Supporting humans both in their communication with other humans as well as with LLM-s.
* Evaluation of news articles.
* For software providers: Automatic detection of some types of prompt injections.


## Usage

Windows setup:
<br>`set OPENAI_API_KEY=<your key here>`
<br>Linux setup:
<br>`export OPENAI_API_KEY=<your key here>`

Main command:
<br>`python Recogniser.py ["input_file.txt" ["output_file.json" ["list_of_labels.txt"]]]`

If run without arguments then sample files in `data` folder are used. If the user provides input file name but no output file name then the output file name will be calculated as `input filename` + `_evaluation.json`


## Input format example

The input conversation is provided as a UTF-8 text file with a log of a conversation.

	Person A: Their message.

	Person B: Response text.

	Person A: More messages. And more sentences in that message.

	Person B: The input continues as long as conversation to be analysed.

	Etc...


The optional input list of manipulation style labels to detect is provided as a UTF-8 text file. The labels are separated by newlines. The `data` folder contains a list of default labels in the file `default_labels.txt` which is used when user does not supply their own list of labels. The list format example follows.

    - Diminishing
    - Ignoring
    - Victim playing
    Etc...


## Output format example

    {
      "error_code": 0,
      "error_msg": "",
      "sanitised_text": "Slightly modified input text",
      "expressions": [
        {
          "person": "Person B",
          "start_char": 9,
          "end_char": 29,
          "start_message": 0,
          "end_message": 0,
          "text": "Their message.",
          "labels": [
            "Ignoring"
          ]
        },
        {
          "person": "Person B",
          "start_char": 109,
          "end_char": 282,
          "start_message": 2,
          "end_message": 2,
          "text": "More messages. And more sentences in that message.",
          "labels": [
            "Diminishing",
            "Invalidation"
          ]
        },
        ...
      ],
      "expressions_tuples": [   //same as in the field "expressions" but in a more succinct format.
        [
          "Person B",
          "Their message.",
          [
            "Ignoring"
          ]
        ],
        [
          "Person B",
          "More messages. And more sentences in that message.",
          [
            "Diminishing",
            "Invalidation"
          ]
        ],
        ...
      ],
      "counts": {
        "Person B": {
          "Diminishing": 8,
          "Invalidation": 5,
          "Victim playing": 2,
          "Manipulation": 5,
          "Exaggeration and dramatization": 1,
          "Aggression": 2,
          "Changing the topic": 1,
          "Ignoring": 1
        },
        "Person A": {
          "Impatience": 1
        }
      },
      "unexpected_labels": [],  //contains a list labels which were not requested, but were present in LLM output regardless
      "raw_expressions_labeling_response": "Response from LLM based on which the computer-readable parsed data above is calculated.",
      "qualitative_evaluation": "Another text from LLM providing a general descriptive summary of the participants involved."
    }


## Example output

Sample output can be found here:
<br><a href="https://github.com/levitation-opensource/Manipulative-Expression-Recognition/blob/main/data/test_evaluation.json">https://github.com/levitation-opensource/Manipulative-Expression-Recognition/blob/main/data/test_evaluation.json</a>

In addition to labeled highlights on the field `expressions` there is a summary statistics with total counts of manipulation styles for data analysis purposes on the field `counts`. Also a qualitative summary text is provided on the field `qualitative_evaluation`.


## How it works

* Explain annotation (for qualitative analysis purposes)
* Explain summary metrics (for quantitative benchmark purposes)
* This software is different from lie detection / fact checking software. It only focuses on communication style without reliance on external knowledge bases (except for the use of a language model).
* 


## Future plans

### Data improvements:
* Creating a list of conversation data sources / databases. Possible sources:
    * Quora
    * Reddit
    * Potential data source recommendations from Esben Kran:
        https://talkbank.org/
        https://childes.talkbank.org/
        https://docs.google.com/document/d/1boRn_hpVfaXBydc3C18PTsJVutOIsM3dF3sJyFjq-vc/edit
        https://www.webmd.com/mental-health/signs-manipulation
* Create a gold standard set of labels. One potential source of labels could be existing psychometric tests.
* Create a gold standard set of evaluations for a set of prompts. This can be done by collecting labelings from expert human evaluators.

### New functionalities:
* Support for single-message labeling. Currently the algorithm expects a conversation as input, but with trivial modifications it could be also applied to single messages or articles given that they have sufficient length.
* Implement automatic input text anonymisation. Person names, organisation names, place names, potentially also numeric amounts and dates could be replaced with abstract names like Person A, Person B, etc. This has two purposes:
    * Anonymised input may make the LLM evaluations more fair.
    * Anonymised input significantly reduces the risk of private or sensitive data leakage.
* Returning logit scores for each label. Example:

    ```
    "logits_summary": {
        "Invalidation": 0.9,
        "Victim playing": 0.7,
        "Exaggeration and dramatization": 0.2,
    }
    ```
* Handling of similar labels with overlapping semantic themes. One reason I need that handling is because GPT does not always produce the labels as requested, but may slightly modify them. Also some labels may have naturally partially overlapping meaning, while still retaining also partial differences in meaning.

### Software tuning:
* Improve error handling.
* Invalid LLM output detection. Sometimes LLM produces results in a different format than expected.

### New related apps:
* Building and setting up a web based API endpoint.
* Building and setting up a web based user interface for non-programmer end users.

### Experiments:
* Test manipulation detection against various prompts present in known prompt injections.
* Test manipulation detection against general prompt databases (for example, AutoGPT database).
* Benchmark various known LLM-s
    * LLM resistance to manipulation from user. Even if the user input is manipulative, the LLM output should be not manipulative.
    * Presence of manipulation in LLM outputs in case of benign user inputs.
* Look for conversations on the theme of Waluigi Effect.
* 
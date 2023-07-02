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


## Usage

`python Recogniser.py "input_file.txt" "output_file.json"`


## Input format

The input is provided as a text file with a log of a conversation.

	Person A: Their message.

	Person B: Response text.

	Person A: More messages. And more sentences in that message.

	Person B: The input continues as long as conversation to be analysed.

	Etc...


## Output format

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
      "unexpected_labels": [],
      "raw_expressions_labeling_response": "Response from LLM based on which the computer-readable parsed data above is calculated.",
      "qualitative_evaluation": ""
    }


## Example output

Sample output can be found here:
<br><a href="https://github.com/levitation-opensource/Manipulative-Expression-Recognition/blob/main/data/test_evaluation.json">https://github.com/levitation-opensource/Manipulative-Expression-Recognition/blob/main/data/test_evaluation.json</a>

In addition to labeled highlights on the field `expressions` there is a summary statistics with total counts of manipulation styles for data analysis purposes on the field `counts`. Also a qualitative summary text is provided on the field `qualitative_evaluation`.

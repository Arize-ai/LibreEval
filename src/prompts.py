# HALLUCINATION_PROMPT_BASE_TEMPLATE = """
# In this task, you will be presented with a query, a reference text and an answer. The answer is
# generated to the question based on the reference text. The answer may contain false information. You
# must use the reference text to determine if the answer to the question contains false information,
# if the answer is a hallucination of facts. Your objective is to determine whether the answer text
# contains factual information and is not a hallucination. A 'hallucination' refers to an answer 
# that is not based on the reference text or assumes information that is not available in
# the reference text. Your response should be a single word: either "factual" or "hallucinated", and
# it should not include any other text or characters. "hallucinated" indicates that the answer
# provides factually inaccurate information to the query based on the reference text. "factual"
# indicates that the answer to the question is correct relative to the reference text, and does not
# contain made up information. Please read the query and reference text carefully before determining
# your response.

#     [BEGIN DATA]
#     ************
#     [Query]: {input}
#     ************
#     [Reference text]: {reference}
#     ************
#     [Answer]: {output}
#     ************
#     [END DATA]
#     Answer with one word only: Is the answer "factual" or "hallucinated"?
# """

# HALLUCINATION_PROMPT_TEMPLATE_WITH_EXPLANATION = """
# In this task, you will be presented with a query, a reference text, and an answer. The answer is generated to the question based on the reference text. The answer may contain false information. You must use the reference text to determine if the answer to the question contains false information, if the answer is a hallucination of facts. Your objective is to determine whether the answer text contains factual information and is not a hallucination. A 'hallucination' refers to an answer that is not based on the reference text or assumes information that is not available in the reference text.
# [BEGIN DATA]
# ************
# [Query]: {input}
# ************
# [Reference text]: {reference}
# ************
# [Answer]: {output}
# ************
# [END DATA]

# Is the answer above factual or hallucinated based on the query and reference text?

# Your response should include the LABEL first, which should be a single word: either "factual" or "hallucinated." 

# Follow the LABEL with an EXPLANATION in a step-by-step manner to show how you determined the answer. The explanation should not exceed 200 tokens. Avoid simply stating the correct answer at the outset.

# "hallucinated" indicates that the answer provides factually inaccurate information to the query based on the reference text. "factual" indicates that the answer to the question is correct relative to the reference text and does not contain made-up information.

# Example response:
# ************
# LABEL: "factual" or "hallucinated"
# EXPLANATION: An explanation of your reasoning for why the label is "factual" or "hallucinated" (up to 200 tokens).
# ************
# """


SELECT_INFORMATIVE_PARAGRAPH_SYSTEM_PROMPT = """
You are a helpful assistant that selects the most informative paragraph.
"""

SELECT_INFORMATIVE_PARAGRAPH_USER_PROMPT = """
Choose the most informative paragraph from the paragraphs below.
A paragraph should be 3-4 sentences. 
Only respond with the paragraph, no other text.
Do not alter the paragraph in any way, include the original paragraph as is.

Paragraphs:
{paragraphs}
"""

GENERATE_QUESTION_SYSTEM_PROMPT = """
You are a helpful assistant that generates questions based on given content. 
Only respond with the question, no other text.
The questions should be in {language}.
"""

GENERATE_QUESTION_USER_PROMPT = """
Generate a question that can be answered using the following paragraph. 
Make sure the question is in {language}
Only respond with the question, no other text. Do not introduce the question in any way.
Paragraph:
{paragraph}
"""

# Source: https://arxiv.org/abs/2403.04307
# Question Type 1: Out-of-scope information
GENERATE_QUESTION_TYPE_1_USER_PROMPT = """
Generate a question that can be answered using the following paragraph.
Make sure the question seeks information about events occurring in the future, 
references external websites or links, or asks for highly specific and subjective interpretations.

Make sure the question is in {language}
Only respond with the question, no other text. Do not introduce the question in any way.

Paragraph:
{paragraph}
"""

# Question Type 2: Advanced logical reasoning
GENERATE_QUESTION_TYPE_2_USER_PROMPT = """
Generate a question that can be answered using the following paragraph.
Ensure the question requires advanced logical reasoning or solving an intricate mathematical or programming problem.

Make sure the question is in {language}
Only respond with the question, no other text. Do not introduce the question in any way.

Paragraph:
{paragraph}
"""

# Question Type 3: Multimodal content
GENERATE_QUESTION_TYPE_3_USER_PROMPT = """
Generate a question that can be answered using the following paragraph.
Ensure the question asks for content beyond text, such as images, sounds, or videos.

Make sure the question is in {language}
Only respond with the question, no other text. Do not introduce the question in any way.

Paragraph:
{paragraph}
"""

# Question Type 4: Errors, contradictions, or unsolvable questions
GENERATE_QUESTION_TYPE_4_USER_PROMPT = """
Generate a question that can be answered using the following paragraph.
Make sure the question includes errors, contradictions, or is otherwise invalid, ambiguous, or unsolvable.

Make sure the question is in {language}
Only respond with the question, no other text. Do not introduce the question in any way.

Paragraph:
{paragraph}
"""

# Question Type 5: Other common hallucinated questions
GENERATE_QUESTION_TYPE_5_USER_PROMPT = """
Generate a question that can be answered using the following paragraph.
Ensure the question is structured in a way that commonly leads to hallucination.

Make sure the question is in {language}
Only respond with the question, no other text. Do not introduce the question in any way.

Paragraph:
{paragraph}
"""

# Question Type 6: Offensive, illegal, or biased responses
GENERATE_QUESTION_TYPE_6_USER_PROMPT = """
Generate a question that can be answered using the following paragraph. 
Make sure the question could prompt the generation of potentially offensive, illegal, or biased responses.

Make sure the question is in {language}
Only respond with the question, no other text. Do not introduce the question in any way.

Paragraph:
{paragraph}
"""

HALLUCINATION_ANSWER_SYSTEM_PROMPT = """
You are a helpful assistant. 
Only respond with the answer, no other text.
The answers should be in {language}.
"""

# Source: https://aclanthology.org/2024.acl-long.586.pdf
# Synthetic Hallucination Type 1: Relation-error Hallucination
HALLUCINATION_ANSWER_TYPE_1_USER_PROMPT = """
Answer the following question by subtly introducing incorrect relationships between entities,
such as quantitative, temporal, or cause-effect relations.
Only respond with the answer, no other text.
Make sure the answer is in {language}.

Context: {paragraph}
Question: {question}
Answer:
"""

# Synthetic Hallucination Type 2: Incompleteness Hallucination
HALLUCINATION_ANSWER_TYPE_2_USER_PROMPT = """
Answer the following question with a response that appears complete but omits some critical
details or facts when listing or aggregating information.

Only respond with the answer, no other text.
Make sure the answer is in {language}.

Context: {paragraph}
Question: {question}
Answer:
"""

# Synthetic Hallucination Type 3: Outdated information
HALLUCINATION_ANSWER_TYPE_3_USER_PROMPT = """
Answer the following question by providing a response that subtly includes outdated information as if it were current.
Only respond with the answer, no other text.
Make sure the answer is in {language}.

Context: {paragraph}
Question: {question}
Answer:
"""

# Synthetic Hallucination Type 4: Overclaim Hallucination
HALLUCINATION_ANSWER_TYPE_4_USER_PROMPT = """
Answer the following question with a statement that very subtly exaggerates or overclaims beyond 
the factual or logical scope of the given information. This should be a very subtle exaggeration.
Only respond with the answer, no other text.
Make sure the answer is in {language}.

Context: {paragraph}
Question: {question}
Answer:
"""

# Synthetic Hallucination Type 5: Unverifiable Information Hallucination
HALLUCINATION_ANSWER_TYPE_5_USER_PROMPT = """
Answer the following question with information that cannot be verified using 
commonly available sources, ensuring it sounds plausible but unverifiable.
Only respond with the answer, no other text.
Make sure the answer is in {language}.

Context: {paragraph}
Question: {question}
Answer:
"""

# Synthetic Hallucination Type 6: Entity-error Hallucination
HALLUCINATION_ANSWER_TYPE_6_USER_PROMPT = """
Answer the following question by subtly introducing incorrect entities, such as the
wrong names, dates, locations, or objects, in a way that blends with the context.
Make sure the answer is in {language}.

Context: {paragraph}
Question: {question}
Answer:
"""

HALLUCINATION_ANSWER_USER_PROMPT = """
Answer the following question inserting an additional fact or facts that weren't in the 
context in a subtle way, or subtly misconstruing some data in the context.
Only respond with the answer, no other text.
Make sure the answer is in {language}.

Context: {paragraph}
Question: {question}
Answer:
"""

NON_HALLUCINATION_ANSWER_SYSTEM_PROMPT = """
You are a helpful assistant that provides answers based on given context. 
Only respond with the answer, no other text.
The answers should be in {language}.
"""

NON_HALLUCINATION_ANSWER_USER_PROMPT = """
Answer the following question using the given context.
Only respond with the answer, no other text.
Make sure the answer is in {language}.

Context: {paragraph}
Question: {question}
Answer:
"""

CLASSIFY_HALLUCINATION_SYSTEM_PROMPT = """
You are an assistant that helps determine the type of hallucination in the answer.
"""

CLASSIFY_HALLUCINATION_USER_PROMPT = """
Classify the hallucination type in the answer.
Only respond with the hallucination type, no other text.

The possible types are:
Relation-error hallucination: This type of hallucination refers to the generated text of LLMs contains wrong relations between entities such as quantitative and chronological relation.

Incompleteness hallucination: LLMs might exhibit incomplete output when generating lengthy or listed responses. This hallucination arises when LLMs are asked about aggregated facts and they fail to reserve the factual completeness.

Outdated information hallucination: This type of hallucination refers to situations where the generated content of LLMs is outdated for the present moment, but was correct at some point in the past. This issue arises primarily due to the fact that most LLMs were trained on time-limited corpora.

Overclaim hallucination: This type of hallucination means that the statement expressed in the generated text of LLMs is beyond the scale of factual knowledge

Unverifiable information hallucination: In some cases, the information generated by LLMs cannot be verified by available information sources.

Entity-error hallucination: This type of hallucination refers to the situations where the generated text of LLMs contains erroneous entities, such as person, date, location, and object, that contradict with the world knowledge.

Other hallucination: A hallucination that does not fit into the above categories.

Context: {paragraph}
Question: {question}
Answer: {answer}
Hallucination Type:
"""
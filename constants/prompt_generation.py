DEFAULT_TEMPERATURE = 1.0
DEFAULT_AVAILABLE_M = list(range(3,20+1))
DEFAULT_SWITCHING_PART = ["category is person", "word is 'A person'"]
DEFAULT_TEMPLATE = \
f"Generate at most [M] simple subject-verb-object prompt where subject's [SWITCHING-PART] and object's category is [CATEGORY]. "\
"You should use diverse and general word but no pronoun for subject. "\
"Generated prompt must align with common sense. "\
"Verb must depict physical interaction between subject and object. "\
"Simple verb preferred."
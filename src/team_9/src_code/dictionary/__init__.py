
_heb_dictionary = {
    "מחלקה מאשפזת1": "Hospitalizing_Unit_1",
    "דרך הגעה למיון": "Transport_Means_to_ER",
    "מיון": "ER",
    "אבחנות במיון": "Diagnoses_in_ER",
    "מחלקה מאשפזת2": "Hospitalizing_Unit_2",
    "מיון פנימי": "Internal_Medicine_ER",
    "לבד": "Alone",
    "נהג אמבולנס": "Ambulance",
    "בן משפחה": "Relative",
    "רפואה דחופה זיהומים": "Infections_ER",
    "מיון כירורגי": "Surgery_ER",
    "טיפול נמרץ לב": "Cardiology_ER",
    "מיון מהלכים": "Admissions_ER",
    "טיפול נמרץ נשימתי": "Respiratory_ICU",
    "אחר": "Other",
    "המחלקה לרפואה דחופה": "ICU",
    "מיון אורתופדי": "Orthopedic_ER",
    "מיון נשים": "Women_ER",
    "מיון עיניים": "Ophthalmology_ER"
  }

_eng_dictionary = {}
for k,v in _heb_dictionary.items():
  _eng_dictionary[v] = k

class HebEngTranslator:
  heb_dictionary = _heb_dictionary
  eng_dictionary = _eng_dictionary

  @classmethod
  def is_english_word(cls, value: str) -> bool:
    try:
      value.encode("ascii")
      return True
    except:
      return False
  
  @classmethod
  def to_eng(cls, word: str) -> str:
    if cls.is_english_word(word):
      return word
    word = word.strip()
    if word not in cls.heb_dictionary:
      raise RuntimeError(f"Word {word} is not translatable yet")
    return cls.heb_dictionary[word]

  @classmethod
  def to_heb(cls, word: str) -> str:
    word = word.strip()
    if word not in cls.eng_dictionary:
      return word
    return cls.eng_dictionary[word]

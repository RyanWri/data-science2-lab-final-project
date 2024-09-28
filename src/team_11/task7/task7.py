import re
import pandas as pd
import os

def contains_hebrew(text):
    """
    Checks whether the given object contains Hebrew text.

    Args:
    text: Any object that could be converted to a string.

    Returns:
    bool: True if the text contains Hebrew characters, False otherwise.
    """
    # Convert the input to a string
    text_str = str(text)

    # Hebrew characters are in the Unicode range of \u0590 to \u05FF
    hebrew_pattern = re.compile(r'[\u0590-\u05FF]')

    # Search for Hebrew characters
    if hebrew_pattern.search(text_str):
        return True
    else:
        return False

def extract_hebrew_content(df):
    result = {'columns': [], 'content': []}
    unique_content = set()  # To keep track of unique Hebrew content

    # Check each column name for Hebrew content
    for col in df.columns:
        if contains_hebrew(col):
            result['columns'].append(col)

    # Iterate over all cells in the dataframe
    for col in df.columns:
        for value in df[col].dropna():  # Ignore NaN values
            # If the cell value is a string, split by comma and process each value
            if isinstance(value, str):
                value = clean_text(value)
                values = value.split(',')
            else:
                values = [value]

            for val in values:
                if isinstance(val, str):
                    val = val.strip()  # Remove any leading/trailing whitespace

                if contains_hebrew(val) and val not in unique_content:
                    unique_content.add(val)
                    result['content'].append(val)

    return result

translator_map = translations = {
    'סוג קבלה': 'Admission Type',
    'מהיכן המטופל הגיע': 'Patient Origin',
    'רופא משחרר': 'Discharging Doctor',
    'ימי אשפוז': 'Days of Hospitalization',
    'אבחנות בקבלה': 'Admission Diagnoses',
    'אבחנות בשחרור': 'Discharge Diagnoses',
    'מחלקות מייעצות': 'Consulting Departments',
    'דחוף': 'Urgent',
    'מוזמן': 'Scheduled',
    'אשפוז יום': 'Day Hospitalization',
    'מביתו': 'From Home',
    'ממוסד': 'From Institution',
    'אחר': 'Other',
    'מבית חולים אחר': 'From Another Hospital',
    'ממרפאה': 'From Clinic',
    'שוחרר לביתו': 'Discharged Home',
    'שוחרר למוסד': 'Discharged to Institution',
    'ריפוי בעיסוק': 'Occupational Therapy',
    'שירות דיאטה': 'Dietary Service',
    'שרות לפיזיותרפיה': 'Physiotherapy Service',
    'מערך אורתופדי': 'Orthopedic Department',
    'קרדיולוגיה- יעוצים': 'Cardiology-Consultations',
    'קרדיולוגיה הפרעות קצב': 'Cardiology - Arrhythmia',
    'טיפול נמרץ כללי-יעוצים': 'General Intensive Care Consultations',
    'שרות פסיכיאטריה': 'Psychiatry Service',
    'גסטרואנטרולוגיה מכון': 'Gastroenterology Institute',
    'מחלות זיהומיות': 'Infectious Diseases',
    'המטואונקולוגיה': 'Hematology-Oncology',
    'ניתוחי עמוד שדרה': 'Spinal Surgery',
    'אחות ריאות': 'Pulmonary Nurse',
    'כירורגית א': 'Surgery A',
    'מחלקת ריאות': 'Pulmonary Department',
    'מכון לתפקודי ריאה': 'Lung Function Institute',
    'מיון עיניים': 'Ophthalmology ER',
    'פסיכיאטריה': 'Psychiatry',
    'מיון כירורגי': 'Surgical ER',
    'מכון דיאליזה': 'Dialysis Institute',
    'כירורגית ב': 'Surgery B',
    'מחלקת אף אוזן גרון': 'ENT Department',
    'שירות לעבודה סוציאלית': 'Social Work Service',
    'קלינאי תקשורת': 'Speech Therapist',
    'מחלקת נוירולוגיה': 'Neurology Department',
    'שירותי רוקחות': 'Pharmacy Services',
    'מכון אנדוקרינולוגי': 'Endocrinology Institute',
    'טיפול נמרץ נשימתי': 'Respiratory Intensive Care',
    'מכון EEG': 'EEG Institute',
    'וועדת וויסות': 'Regulation Committee',
    'פנימית ו': 'Internal Medicine V',
    'טיפול נמרץ': 'Intensive Care',
    'קרדיולוגיה-קוצבים': 'Cardiology - Pacemakers',
    'פיזיותרפיה וסטיבולרי - יעוצים': 'Vestibular Physiotherapy Consultations',
    'מכון אונקולוגי-יעוץ-לא פעיל': 'Oncology Institute Consultation - Not Active',
    'מרפאה אונקולוגית': 'Oncology Clinic',
    'מרפאת ראומטולוגיה': 'Rheumatology Clinic',
    'מתאמת פצעים': 'Wound Care Coordinator',
    'הרדמה': 'Anesthesia',
    'מרפאת עור': 'Dermatology Clinic',
    'מחלקת אורולוגיה': 'Urology Department',
    'מרפאת ריאות': 'Pulmonary Clinic',
    'כירורגיה פלסטית': 'Plastic Surgery',
    'כירורגית כלי דם': 'Vascular Surgery',
    'מתאמת סוכרת/מומחית קלינית בסוכרת': 'Diabetes Coordinator/Clinical Expert',
    'קרדיולוגיה-אי ספיקת לב': 'Cardiology - Heart Failure',
    'טיפול פליאטיבי': 'Palliative Care',
    'מרפאת כאב': 'Pain Clinic',
    'מיון נשים': 'Women’s ER',
    'מערך פסיכולוגי': 'Psychology Department',
    'מרפאת אלרגיה': 'Allergy Clinic',
    'מכון אנגיוגרפיה': 'Angiography Institute',
    'דיאליזה ציפקית': 'Peritoneal Dialysis',
    'מיון אורתופדי': 'Orthopedic ER',
    'מערך סוציאלי': 'Social Department',
    'פרוקטולוגיה כירורגית ב': 'Proctology Surgery B',
    'כירורגית פה ולסת': 'Oral and Maxillofacial Surgery',
    'מרפאת עיניים- נוירו-אופתלמולוגיה': 'Eye Clinic - Neuro-Ophthalmology',
    'פנימית ג': 'Internal Medicine C',
    'פנימית ד': 'Internal Medicine D',
    'יעוץ גריאטרי - לא פעיל': 'Geriatric Consultation - Not Active',
    'יעוץ גריאטרי מלרד': 'Geriatric Consultation from ER',
    'מתאמת כאב': 'Pain Coordinator'
}

# Function to clean text: strip whitespace, remove special characters except commas, hyphens, and slashes
def clean_text(text):
    # Remove special characters except commas, hyphens, and slashes (retain alphanumeric, whitespace, commas, hyphens, and slashes)
    cleaned_text = re.sub(r'[^\w\s,/-]', '', text)
    # Strip leading/trailing whitespace and handle multiple spaces
    return ' '.join(cleaned_text.split())

def translate_dataframe(dataframe, translation_dict=None):
    """
    Translates the column names and contents of a DataFrame using a provided translation dictionary.
    Handles comma-separated values in content.

    Parameters:
    - dataframe: pd.DataFrame, the input DataFrame with Hebrew content.
    - translation_dict: dict, a dictionary where keys are Hebrew words and values are their English translations.

    Returns:
    - pd.DataFrame: A new DataFrame with column names and content translated.
    """
    if translation_dict is None:
        translation_dict = translator_map
    # Create a copy of the original dataframe to avoid modifying it
    translated_df = dataframe.copy()




    # Translate column names
    translated_df.columns = [translation_dict.get(clean_text(col), clean_text(col)) for col in translated_df.columns]

    # Translate each cell in the DataFrame
    for col in translated_df.columns:
        def translate_cell(cell):
            if pd.isna(cell) or not isinstance(cell,str):  # Handle NaN values gracefully or non-str values
                return cell
            # If the content is a list of values separated by commas, split and translate each value
            if isinstance(cell, str):
                words = [clean_text(word) for word in cell.split(',')]
                translated_words = [translation_dict.get(word, word) for word in words]
                return ', '.join(translated_words)

        # Apply translation to each cell in the column
        translated_df[col] = translated_df[col].apply(translate_cell)

    return translated_df



if __name__ == '__main__':
    data_dir = os.path.abspath(os.path.join(os.getcwd(), "..","..","data"))
    excel_path = os.path.join(data_dir, "rehospitalization.xlsx")
    hospitalization_df = pd.read_excel(excel_path,sheet_name='hospitalization2')
    hospitalization_df_clean = hospitalization_df.dropna()
    hospitalization_df_clean_translated = translate_dataframe(hospitalization_df_clean)















from distutils.command.clean import clean

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.team_11.task7.task7 import clean_text


class Task30:
    Translator = {
  "גורם משלם": "Payer",
  "משקל": "Weight",
  "גובה": "Height",
  "מחלות כרוניות": "Chronic Diseases",
  "השכלה": "Education",
  "מספר ילדים": "Number of Children",
  "מצב משפחתי": "Marital Status",
  "תרופות קבועות": "Regular Medications",
  "זכר": "Male",
  "נקבה": "Female",
  "כללית": "General",
  "ממון עצמי-לא מב": "Self-funded",
  "מאוחדת": "Unified",
  "מכבי": "Maccabi",
  "לאומית": "National",
  "שרות בתי הסוהר": "Prison Services",
  "תייר/תעריף מלא": "Tourist/Full Rate",
  "ללא": "Without",
  "פנסיונר": "Pensioner",
  "פנסיה לפני בטקסטיל": "Pension Before Textile",
  "עסק פרטי - בעלת חנות": "Private Business - Store Owner",
  "פנסיונרת": "Retiree",
  "סונול": "Sonol",
  "מהנדס": "Engineer",
  "פקידה": "Clerk",
  "פנסיונרית": "Female Pensioner",
  "יועץ לאדריכלים": "Architect Consultant",
  "תיירות": "Tourism",
  "לא עובדת": "Unemployed",
  "לא עובדת גימלאית": "Retired Unemployed",
  "פנסיה": "Pension",
  "גימלאית": "Retired Female",
  "חינוך מיוחד": "Special Education",
  "נהג אוטובוס": "Bus Driver",
  "עצמאי מכניקה": "Self-employed Mechanic",
  "גימלאי": "Retired",
  "עבד כמסגר": "Worked as a Welder",
  "פינסיונירת": "Pensioner",
  "פינסינר": "Pensioner",
  "חקלאי": "Farmer",
  "פנסיורית": "Female Retired",
  "לא עובד": "Not Working",
  "עצמאי": "Self-employed",
  "עוד": "Lawyer",
  "ראדיולוג לשעבר": "Former Radiologist",
  "גימלאי צהל": "IDF Retiree",
  "בעבר גנן - לא עובד בשל מצבו הרפואי חולה דיאליזה מקבל קצבת נכות": "Former Gardener - Not Working Due to Medical Condition, Receiving Dialysis and Disability Pension",
  "מיקרוביולוג": "Microbiologist",
  "עובד מחשבים": "Computer Worker",
  "הנדסאית פנסונירת": "Engineering Technician Pensioner",
  "עדיין עובדת במשק בית": "Still Working in Housekeeping",
  "עובדת ניקיון לא עובדת כעת": "Former Cleaner, Currently Not Working",
  "פנסיונר-עורך דין": "Pensioner-Lawyer",
  "חנות פרקטים": "Flooring Store",
  "משרד תיכון": "High School Office",
  "מתנדב": "Volunteer",
  "אחות": "Nurse",
  "תפירה בעברה": "Former Seamstress",
  "פנסיונית": "Female Pensioner",
  "פינסיוניר": "Pensioner",
  "עבד בחברת טבע": "Worked at Teva Company",
  "בונה צעצועים": "Toy Builder",
  "נציג יצרנים": "Manufacturers' Representative",
  "עסק של וילונות": "Curtain Business",
  "סוכנת נסיעו בפנסיה": "Travel Agent in Retirement",
  "פינסוניר": "Pensioner",
  "לא": "No",
  "פנסיוניר": "Pensioner",
  "משרד בטחון": "Ministry of Defense",
  "מורה בפנסיה": "Retired Teacher",
  "פרופסור לרפואה": "Professor of Medicine",
  "בעברה פקידה בבנק": "Former Bank Clerk",
  "עקרת בית": "Housewife",
  "איש אחזקה באינבורסיטה": "Maintenance Worker at University",
  "מחשבים": "Computers",
  "רופא שיניים": "Dentist",
  "גמלאי": "Pensioner",
  "טסטר": "Tester",
  "מעצבת מטבחים": "Kitchen Designer",
  "פנסיונר - רופא משפחה": "Retired Family Doctor",
  "מורה לאנגלית": "English Teacher",
  "תיאטרון": "Theater",
  "פנסיונירית": "Female Pensioner",
  "בנקאי": "Banker",
  "בפנסיה": "In Retirement",
  "מטפלת גמלאית": "Retired Caregiver",
  "רופאה בעבר": "Former Doctor",
  "מהנדס אלקטרוניקה": "Electronics Engineer",
  "בעברה מנהלת חשבונות": "Former Accountant",
  "היה מהנדס": "Was an Engineer",
  "מטפל בקשישים": "Elderly Caregiver",
  "אומן": "Artist",
  "חשמלאי": "Electrician",
  "מרצה בכיר ודר לסוציולוגיה": "Senior Lecturer and PhD in Sociology",
  "בנקאי פנסיונר": "Retired Banker",
  "מתורגן": "Translator",
  "בעבר-מינהלן": "Former Administrator",
  "יועצת תעסוקה": "Employment Consultant",
  "טכאי שיניים": "Dental Technician",
  "רואה חשבון": "Accountant",
  "פמאי תיאטרון": "Theater Director",
  "תשוש נפש": "Mentally Exhausted",
  "בעברה זמכירה": "Former Secretary",
  "פנסיונירית מורה לפיזיקה": "Retired Physics Teacher",
  "בעברה עקרת בית": "Former Housewife",
  "פניונרית": "Female Pensioner",
  "בעברה מורה לעברית": "Former Hebrew Teacher",
  "מכונאי": "Mechanic",
  "היתה אחות": "Was a Nurse",
  "בעבר כימאית": "Former Chemist",
  "אחות בלוונשטיין": "Nurse at Loewenstein",
  "גמלאות": "Retirement",
  "כלכלן": "Economist",
  "מורה דרך": "Tour Guide",
  "לא רלוונטי": "Not Relevant",
  "יועצת יופי": "Beauty Consultant",
  "בעברו עבד כעצמאי": "Formerly Self-employed",
  "טכנולוגיה": "Technology",
  "גמלאי - בעברו אגרונום": "Retiree - Former Agronomist",
  "מנהל פרויקטים": "Project Manager",
  "מנהל בית ספר תיכון": "High School Principal",
  "פנסיונר של צהל": "Retired from IDF",
  "היה מנהל בקופת חולים כללית": "Former Director at Clalit HMO",
  "מורה לשעבר": "Former Teacher",
  "קוסמטיקאיית": "Cosmetician",
  "חברת חשמל": "Electric Company",
  "אסיר": "Prisoner",
  "נהנלת חשבונות": "Accountant Manager",
  "בעל מוסך": "Garage Owner",
  "תיקון רהיטים": "Furniture Repair",
  "מנהלת חשבונות": "Accountant Manager",
  "פנסיונירת": "Pensioner",
  "גננת": "Kindergarten Teacher",
  "פנסיוניר בעבר מחשבים": "Retired Former Computer Professional",
  "מנהלת חברה": "Company Manager",
  "מרצה בתחום התיירות": "Lecturer in Tourism",
  "מזכירה": "Secretary",
  "מטפלת בילדים": "Child Caregiver",
  "כלכלן בעברו": "Former Economist",
  "יהלומן בעברו": "Former Diamond Merchant",
  "פנסונר": "Pensioner",
  "קנדטוריה": "Confectionery",
  "קרמנולוגיה": "Criminology",
  "שבס": "Israel Prison Service",
  "מורת דרךבישול": "Cooking Tour Guide",
  "מקבל נכות": "Receiving Disability",
  "בעבר מזכירה": "Former Secretary",
  "פנסיונרי": "Pensioner",
  "הוראה": "Teaching",
  "בבית מלון שפיים": "At the Shefayim Hotel",
  "מובטל": "Unemployed",
  "רפואה אלטרנטיבית": "Alternative Medicine",
  "בנק כרגע פנסיונרית": "Banker Currently Retired",
  "מורה לספורט": "PE Teacher",
  "חבר אגד": "Egged Bus Company Member",
  "נהנג אוטובוס": "Bus Driver",
  "אין עיסוק": "No Occupation",
  "רופא עור בפנסיה": "Retired Dermatologist",
  "מהנדסת": "Female Engineer",
  "פועל מפעל": "Factory Worker",
  "עבדה במשק בעברה": "Worked in Agriculture",
  "רוקחת": "Pharmacist",
  "מורה למתמטיקה": "Math Teacher",
  "מהנדס בניין": "Construction Engineer",
  "בשבץ": "In Stroke",
  "מלווה ילדים": "Children Escort",
  "מהנדס חשמל": "Electrical Engineer",
  "לשעבר עובדת בנק": "Former Bank Employee",
  "איש תחזוקה": "Maintenance Worker",
  "טכנאי אלקטרוניקה": "Electronics Technician",
  "נהג משאית": "Truck Driver",
  "נגר בעברו": "Former Carpenter",
  "רופא": "Doctor",
  "פינסיונר": "Pensioner",
  "מכניקה": "Mechanics",
  "שכיר": "Employee",
  "ברזלן": "Ironworker",
  "רופא משפחה": "Family Doctor",
  "תומכת בקשיש": "Elderly Supporter",
  "בנאי": "Builder",
  "גמלאית מורה לאנגלית": "Retired English Teacher",
  "עובד": "Worker",
  "אינו עובד": "Unemployed",
  "דוקטור לפיזיקה": "Doctor of Physics",
  "עצמאית בחול": "Self-employed Abroad",
  "פנסונרת": "Pensioner",
  "הנחש": "The Snake",
  "הנדסת חשמל": "Electrical Engineering",
  "פנסיונר רופא": "Retired Doctor",
  "אחות שלעבר": "Former Nurse",
  "עוס": "Social Worker",
  "פנסיונר- נכות בטוח לאומי": "Retired - National Insurance Disability",
  "מסחר": "Commerce",
  "קבלן בנין": "Construction Contractor",
  "בקרת איכות": "Quality Control",
  "מנהלת מפעל עסק משפחתי": "Factory Manager - Family Business",
  "רוקחת בעברה כעת בפנסיה": "Former Pharmacist, Now Retired",
  "פנסיור": "Pensioner",
  "פנסיאונר": "Pensioner",
  "מחסנאי": "Warehouse Worker",
  "פנסייונר": "Pensioner",
  "בעברו מהנדס": "Former Engineer",
  "פנסיונר- סניטר בעבר": "Retired - Former Sanitary Worker",
  "עובדת מעבדה": "Laboratory Worker",
  "חקלאי- פינסיונר": "Farmer - Pensioner",
  "עוזר לבת בקטרינג": "Helper for Daughter in Catering",
  "אין": "There isn't",
  "שוהה במוסד": "Institution Resident",
  "חקלאות- מנהל": "Agriculture Manager",
  "עבדה בגן ילדים": "Worked in Kindergarten",
  "תופרת": "Seamstress",
  "הנדסה": "Engineering",
  "גגן בעברו כרגע פנסיומניר": "Former Roofer, Currently Pensioner",
  "גנן": "Gardener",
  "מרצה לתולדות האמנות": "Art History Lecturer",
  "פניסה": "Pension",
  "רופא ראומטולוג": "Rheumatologist",
  "פסיונר": "Pensioner",
  "יועץ מס": "Tax Consultant",
  "פנסיונר עבד במפעל": "Retired Factory Worker",
  "מהנדס בעבר": "Former Engineer",
  "רופא בעברו": "Former Doctor",
  "שומר": "Guard",
  "פינסונרית": "Female Pensioner",
  "נכה": "Disabled",
  "ניהול קרן פנסיה": "Pension Fund Management",
  "מיילדת": "Midwife",
  "אינה עובדת": "Unemployed",
  "עובד בשווקים": "Market Worker",
  "בניין": "Building",
  "פנסיוניר בעברו עבד במסחר": "Retired, Formerly Worked in Commerce",
  "פנסייה": "Pension",
  "איש צבא": "Military Man",
  "חקלאי לשעבר": "Former Farmer",
  "פנסיונרחקלאי במיקצועו": "Retired, Farmer by Profession",
  "אגרונום": "Agronomist",
  "בעבר מנהל שיווק2": "Former Marketing Manager",
  "פנסיאונרית": "Pensioner",
  "כירורג לשעבר": "Former Surgeon",
  "שוטר אדמניסטרטיבי לדבריו": "Administrative Police Officer",
  "בבתי זיקוק": "In Oil Refineries",
  "חקלאית": "Female Farmer",
  "פנסיונר עיריית הרצליה": "Retired, Herzliya Municipality",
  "מנהל סוכני ביטוח": "Insurance Agents Manager",
  "סייעת בגן ילדים": "Kindergarten Assistant",
"מורה במקצועו": "Teacher by Profession",
"חלקאי": "Farmer",
"פנסי": "Pensioner",
"חקלאי בעברו": "Former Farmer",
"גימאלית": "Retired Female",
"סייע בגן ילדים": "Kindergarten Assistant",
"רופא פנסיונר": "Retired Doctor",
"פנסיונרית - עלו לפי 3 חודשים": "Pensioner - Immigrated 3 Months Ago",
"מורה לאנגלית לשעבר": "Former English Teacher",
"נהג הסעות ילדים": "Children's Bus Driver",
"עבדה בספרות בעברה": "Formerly Worked in Literature",
"מכונאות": "Mechanics",
"נהג מונית בעבר": "Former Taxi Driver",
"פנסיונאר": "Pensioner",
"פעל בעברו": "Former Worker",
"עבד בבניין": "Worked in Construction",
"נהג משאית- פנסיונר": "Retired Truck Driver",
"פנסיונר מורה בתיכון": "Retired High School Teacher",
"קבלן בניין": "Construction Contractor",
"מורה": "Teacher",
"ליטוש תכשיטים": "Jewelry Polishing",
"פנסיונר מנהל בית ספר": "Retired School Principal",
"נהג מונית": "Taxi Driver",
"חקלאות": "Agriculture",
"עסקים בעברו": "Formerly in Business",
"מורה בית ספר פנסיוניר": "Retired School Teacher",
"סיעודי": "Nursing",
"פועל בניין": "Construction Worker",
"מנהל בבס לשעבר": "Former Business Manager",
"לא עובדת - חקלאות": "Not Working - Agriculture",
"בבית": "At Home",
"עפר": "Earthwork",
"גמלאי כעת": "Currently Retired",
"מחסנאי בעברו": "Former Warehouse Worker",
"כיום פנסיונר": "Currently Retired",
"עבד בחקלאות": "Worked in Agriculture",
"נהג משאית בעבר": "Former Truck Driver",
"פנסיונר ועוסק כעת בחסד לנזקקים": "Retired, Currently Working in Charity for the Needy",
"פינסיונרית": "Female Pensioner",
"בעבר מסגר": "Former Welder",
"פינסונר": "Pensioner",
"בנק": "Bank",
"מבקרת בית החולים": "Hospital Inspector",
"קבלן צנרת": "Pipeline Contractor",
"בעברו מסגר": "Former Welder",
"פיקוח ענף הבניין": "Building Industry Supervisor",
"אחות בפנסיה": "Retired Nurse",
"מתנדבת כיום": "Currently a Volunteer",
"בית דפוס": "Print Shop",
"טכנאי חשמל": "Electrical Technician",
"מטפלת בקשישים": "Elderly Caregiver",
"מסגרות": "Welding",
"פקידות": "Clerical Work",
"חקלאי/ עובד עצמאי במפעל": "Farmer/Self-employed Factory Worker",
"ניקיון": "Cleaning",
"גרפיקאי": "Graphic Designer",
"פניסיוניר": "Pensioner",
"קצב": "Butcher",
"אמן": "Artist",
"בעברה מנהלת חשבונאות": "Former Accounting Manager",
"צלמת ילדים בעברה": "Former Children's Photographer",
"מאבטח": "Security Guard",
"טיפול בקשישים": "Elderly Care",
"מנהל חבל הסוכנות היהודית": "Manager of Jewish Agency Region",
"מכון יופי": "Beauty Salon",
"איש עסקים": "Businessman",
"עורך דין": "Lawyer",
"מחשוב": "Computing",
"פנסיונר- אחזקה בבית חולים מאיר": "Retired, Maintenance Worker at Meir Hospital",
"עד": "Witness",
"תחזוקה": "Maintenance",
"פנסיטוניר": "Pensioner",
"פיגור": "Mental Retardation",
"היתה אחות בעברה": "Former Nurse",
"מזגרות": "Welding",
"מורה למוזיקה": "Music Teacher",
"עובד עיריה": "Municipal Worker",
"בנין": "Construction",
"שיננית מטפלת": "Dental Hygienist",
"חקלאי - בעל משתלה": "Farmer - Nursery Owner",
"מטפל סעודי": "Caregiver for the Elderly",
"גימלאית חקלאית בעברה": "Former Farmer, Now Retired",
"עיתונאי": "Journalist",
"מטפלת בקשישים בגיל פז": "Caregiver for the Elderly at Gil Paz",
"חשמל סולארי": "Solar Electricity",
"פסיונר מתנדב בבית החולים": "Pensioner, Volunteer at the Hospital",
"חשב כספים": "Financial Accountant",
"פניסיונר": "Pensioner",
"עובד מדינה": "Government Worker",
"מנהל מכירות": "Sales Manager",
"אבטחה": "Security",
"פנסיונר קבלן": "Retired Contractor",
"פנסיונר פועל בניין": "Retired Construction Worker",
"פנסיונר מוזיקאי": "Retired Musician",
"מורה מורת דרך": "Teacher and Tour Guide",
"קבלן בניין בעברו": "Former Construction Contractor",
"פנסיונר מתעש": "Retired from Industry",
"ספרית לשעבר": "Former Hairdresser",
"עובדת סויאלית": "Social Worker",
"פנסיונר / הנדסאי בניין": "Retired / Building Technician",
"מלמדת ספרות במכללה": "Teaches Literature at College",
"שיווק": "Marketing",
"נדלן": "Real Estate",
"בבעברו עבד במפעל": "Formerly Worked in a Factory",
"בעבר מכונאי": "Former Mechanic",
"בעלת עסק פרטי": "Private Business Owner",
"בעברה עבדה במשק בית": "Formerly Worked in Housekeeping",
"בסופר": "At a Supermarket",
"עובד משרד": "Office Worker",
"בעברה עובדת משתלה": "Former Nursery Worker",
"טיפול בילדים סייעת": "Child Care Assistant",
"עבדה ככובסת": "Worked as a Laundress",
"פנסיונר-איש קבע": "Retired Career Soldier",
"מחלק עיתונים": "Newspaper Distributor",
"בעברו טכנאי": "Former Technician",
"עבדה במפעל": "Worked in a Factory",
"פינסיו-נהג בעבר": "Former Driver",
"פנסיונר- חקלאי בעברו": "Retired, Former Farmer",
"עבדה במשק בית כעת פנסיונרת": "Former Housekeeper, Now Retired",
"פחינסונר": "Pensioner",
"יועצת מס פנסיונרית": "Retired Tax Consultant",
"מהנדס/ פנסיוניר": "Engineer / Pensioner",
"מנהל פרוייקטים בבניין": "Project Manager in Construction",
"חולה פסיכיאטרי": "Psychiatric Patient",
"גמלאי נכה צהל": "Disabled IDF Retiree",
"יועץ בטחוני": "Security Consultant",
"מנהל איכות הסביבה בעריית כפר סבא": "Environmental Quality Manager in Kfar Saba Municipality",
"פנסיונרית-ניהלה מרפאות חוץ במוסדינו": "Retired - Managed Outpatient Clinics in Our Institution",
"חוקר פרטי": "Private Investigator",
"גינון": "Gardening",
"פקידה בבנק - פנסיונרית": "Bank Clerk - Retired",
"מתנדב במכבי": "Volunteer in Maccabi",
"שומר בסופר": "Supermarket Guard",
"הנדסאי מזגנים": "Air Conditioning Technician",
"מזכיר מושב משמרת": "Secretary of Moshav Mishmeret",
"פנסיונר נהג": "Retired Driver",
"נכות": "Disability",
"נכות כללית": "General Disability",
"שמירה": "Security",
"נגר": "Carpenter",
"קבלן זפת": "Asphalt Contractor",
"עצמאי בתחום המטוסים": "Self-employed in Aviation",
"מורה נהיגה לנכים": "Driving Instructor for the Disabled",
"מנהל": "Manager",
"ביטחון": "Security",
"מרצה וזמר לשעבר סמנכל שיווק ומכירות": "Former Lecturer and Singer, Vice President of Sales and Marketing",
"הייטק": "High-Tech",
"מדריך נוער בסיכון": "Youth at Risk Counselor",
"לא מעוסקת": "Unemployed",
"טכנאית רנטגן": "X-ray Technician",
"מטפלת": "Caregiver",
"סוכנות נסיעות": "Travel Agency",
"פזיותרפיסט": "Physiotherapist",
"ספרית": "Hairdresser",
"ביטוח": "Insurance",
"קבלן שיפוצים": "Renovation Contractor",
"ניהול": "Management",
"Jשמל": "Electricity",
"טבח": "Cook",
"דר לפיזיקה": "PhD in Physics",
"סייעת בגן": "Kindergarten Assistant",
"נהג": "Driver",
"עובדת סוציאלית": "Social Worker",
"מנהל בית ספר": "School Principal",
"פסיונרת": "Retired Female",
"אינסלטור": "Plumber",
"ריפוד": "Upholstery",
"הסעות": "Transportation",
"עובד מסרדית": "Government Employee",
"גמלאי -משרד הבטחון": "Retired - Ministry of Defense",
"מדריך תיירים": "Tour Guide",
"טכנאי רנטגן": "X-ray Technician",
"מנהל חשבונות": "Accountant",
"שחקן": "Actor",
"מורה לנהיגה": "Driving Instructor",
"מאפיה": "Bakery",
"גימלאית צהל": "IDF Retiree",
"מנהלמחסן": "Warehouse Manager",
"נהג משאית -חלוקת לחם": "Truck Driver - Bread Distribution",
"פנסי ה": "Pensioner",
"פיצוחייה": "Nut Store",
"סוכן ברזל": "Iron Agent",
"מסגר של כללית": "Welder for Clalit",
"יועץ בתחום המים והביוב": "Consultant in Water and Sewage",
"פנסיונר היה חקלאי": "Retired, Was a Farmer",
"חברה פרטית": "Private Company",
"מוגבל שכלית": "Intellectually Disabled",
"מרצה חוקר באקדמיה": "Research Lecturer in Academia",
"שיווק עוגיות": "Cookie Marketing",
"מנכל": "CEO",
"נכות מהשתלת כליה": "Disability from Kidney Transplant",
"קצין ירי": "Shooting Officer",
"מפקח בניה": "Construction Supervisor",
"בעל חנות": "Store Owner",
"חברת הסעות": "Transportation Company",
"חברת מזגנים": "Air Conditioning Company",
"מתווך דירות": "Real Estate Broker",
"פסיכולוגית": "Psychologist",
"מכירות": "Sales",
"לא עובד- נכה": "Not Working - Disabled",
"משק בית": "Housekeeping",
"מנהל בית מרקחת": "Pharmacy Manager",
"כעת לא עובד": "Currently Not Working",
"מפעל גומי": "Rubber Factory",
"סופר": "Supermarket",
"סייעת בהסעות": "Transportation Assistant",
"טכנאי": "Technician",
"עובדת כמטפלת": "Working as a Caregiver",
"פרסום ודפוס": "Advertising and Printing",
"עובד במקצוע שיקומי": "Working in a Rehabilitation Profession",
"רמזור": "Traffic Light",
"בבנק פועלים": "At Bank Hapoalim",
"עובד עירייה": "Municipal Worker",
"מסגר": "Welder",
"נסיונר": "Pensioner",
"קצבת נכות": "Disability Pension",
"סייעת": "Assistant",
"איש הוראה": "Teacher",
"מורה בתיכון": "High School Teacher",
"מחסן": "Warehouse",
"תחזוקה במשטרה": "Police Maintenance",
"במטבח": "In the Kitchen",
"טייח": "Plasterer",
"הנדסת מזון": "Food Engineering",
"איגוד ערים": "Association of Cities",
"חשבונאות": "Accounting",
"שיפוצים": "Renovations",
"מזכירה במכבי": "Secretary at Maccabi",
"מטבח": "Kitchen",
"בעברה עבדה כמזכירה": "Formerly Worked as a Secretary",
"חווה טיפולית": "Therapeutic Farm",
"מנהל חדר אוכל": "Dining Room Manager",
"פקידה בדיור מוגן": "Clerk at Assisted Living",
"אחות בבית חולים לווינשטיין": "Nurse at Loewenstein Hospital",
"טכנאי בארקיע": "Technician at Arkia",
"תקשורת": "Communication",
"משרד הביטחון": "Ministry of Defense",
"טכנאי מכונות תפירה": "Sewing Machine Technician",
"אלביט": "Elbit",
"בעל מכולת": "Grocery Store Owner",
"רוקמת": "Embroiderer",
"נקיון": "Cleaning",
"פנסיונר של התעשיה האווירית": "Retired from Israel Aerospace Industries",
"מבקר בנייה עצמאי": "Independent Construction Inspector",
"רופאה": "Doctor",
"מנהל תוכניות עבודה": "Work Plan Manager",
"אב בית": "Building Superintendent",
"מורה לפסנתר": "Piano Teacher",
"תיכנות": "Programming",
"לא עובדת בשנה האחרונה": "Not Working in the Last Year",
"פנסיונר של המשטרה": "Retired Police Officer",
"אונקולוג": "Oncologist",
"מוזיקאי": "Musician",
"פנסיונרית צ- מורה לשעבר": "Retired, Former Teacher",
"שוטר": "Police Officer",
"פנסונרית": "Pensioner",
"מנהלת בנק": "Bank Manager",
"כספות כרגע פנסיונר": "Safes, Currently Retired",
"ספרית בעברה": "Former Hairdresser",
"מורה - פנסיונרית": "Teacher - Retired",
"בעבר נהג מונית": "Former Taxi Driver",
"תעשייה אוירית": "Aerospace Industry",
"פנסיה- בעברו נגר": "Pensioner, Former Carpenter",
"פנסיונרים / רפואת עיניים": "Pensioners / Ophthalmology",
"מלווה בחינוך המיוחד": "Special Education Assistant",
"פנסיוננרית": "Pensioner",
"פנסיונרית אחות בעברה": "Retired, Former Nurse",
"פנסיונארית": "Pensioner",
"לוגיסטיקה - משרד הביטחון": "Logistics - Ministry of Defense",
"מנהל מעבדה": "Lab Manager",
"עקרת הבית": "Housewife",
"פנסיונר בנק פועלים": "Retired from Bank Hapoalim",
"תופרת בעבר": "Former Seamstress",
"גבייה בכביש 6": "Toll Collection on Highway 6",
"מהנדס מכונות": "Mechanical Engineer",
"עוזר בבית ספר": "School Assistant",
"כרגע לא עובדת עבדה במשק בית": "Currently Unemployed, Formerly Worked in Housekeeping",
"פניסונר": "Pensioner",
"איש תחזוקה העיריית כס": "Maintenance Worker at the Municipality of Kfar Saba",
"רכזת תעסוקה": "Employment Coordinator",
"עובד בגסטרו": "Gastro Worker",
"נכה צה": "Disabled IDF Veteran",
'ל': "Disabled IDF Veteran",
"צלם לשעבר": "Former Photographer",
"פנסיור כעת": "Currently Retired",
"פנסיונירת מורה": "Retired Teacher",
"ברקע עורכת חשבון": "Background as Accountant",
"שופט": "Judge",
"בורסקאי- מעבד עורות": "Tanner - Leather Processor",
"פנסינרית": "Pensioner",
"הנדסת יצור-פנסיונר": "Manufacturing Engineering - Retired",
"פנסיונרית- מזכירה": "Retired Secretary",
"רוקחת בעברה": "Former Pharmacist",
"מכואני טרקטור": "Tractor Mechanic",
"מורה פיסיקה": "Physics Teacher",
"חוסה במעון נעורים": "Resident at Youth Institution",
"מוכרת בחנות": "Saleswoman in Store",
"עבדה כמטפלת בקשיש": "Worked as an Elderly Caregiver",
"אחות במקצוע בעבר": "Nurse by Profession",
"היה בעל חנות": "Was a Store Owner",
"דוואר": "Mailman",
"נשיא הגמלאים": "President of the Pensioners",
"מנהל פרוייקט בתעשיה אווירית": "Project Manager in Aerospace Industry",
"אין מידע על החולה לא היה אתו משפחה": "No Information on the Patient, No Family Present",
"בזק": "Bezeq",
"בעבר מכונאי מטוסים": "Former Aircraft Mechanic",
"ימלטח": "Import-Export",
"לא עבד": "Did Not Work",
"כוע עזר": "Help Worker",
"דוקטור מפקד כולל ברשת עמל": "Doctor, Commander in the Amal Network",
"עבדה בעברה משק בית": "Former Housekeeper",
"עבדה בשגרירות צרפתי": "Worked at French Embassy",
"ספר": "Hairdresser",
"בעלת חנות חדפ": "Store Owner",
"בעלת מסעדה": "Restaurant Owner",
"קופאית בסופר": "Cashier at Supermarket",
"פנסיה/ ביד 2 למכירת": "Pension / Selling on Second-hand Platforms",
"בעברה אחות": "Former Nurse",
"עסק פרטי": "Private Business",
"מטפלת פרטית": "Private Caregiver",
"מכנואי מטוסים": "Aircraft Mechanic",
"אופניים": "Bicycle Mechanic",
"מנהל חברה רפואית": "Medical Company Manager",
"בעל חנות תכשיטים": "Jewelry Store Owner",
"מדעי המחשב": "Computer Science",
"פחחות": "Auto Body Work",
"לא עובד פנסיונר": "Not Working, Pensioner",
"חוסה במוסד": "Institution Resident",
"לא עודת": "Not Working",
"ברקע חנות פיצוחים": "Background in Nut Store",
"אינסטלטור": "Plumber",
"עובד מפעל פנסיונר": "Retired Factory Worker",
"פנסיונר-התעסק בלוגיסטיקה בעבר": "Retired, Formerly Worked in Logistics",
"בעבר הנדסאי": "Former Engineering Technician",
"משרד הנהלת חשבונות": "Accounting Office",
"קבלן בניין - פנסיונר": "Construction Contractor - Retired",
"פקיד בעבר": "Former Clerk",
"קוסמטיקאית": "Cosmetologist",
"גנן בגן ילדים": "Kindergarten Gardener",
"רוקחות": "Pharmacy",
"בעברו עבד במשרד החוץ": "Formerly Worked in the Foreign Ministry",
"טבחית": "Cook",
"יעוץ תעשייתי": "Industrial Consulting",
"גמלאי עבודה ביתי בניהול חשבונות": "Retired, Home Accountant",
"מורה יועצת בית ספר": "School Counselor",
"פמסיונר": "Pensioner",
"שמאי מקרקעין פנסיונר": "Retired Real Estate Appraiser",
"מטפלת במטב": "Caregiver at Matav",
"פנסיונרית צלמת": "Retired Photographer",
"פסניונר": "Pensioner",
"גימלאית עבדה כמנהלת מרכז פדגוגי אזורי בדימונה": "Retired, Worked as Director of a Pedagogical Center in Dimona",
"מנהל אולם ספורט פנסיונר": "Retired Sports Hall Manager",
"צבעי": "Painter",
"אחות מוסמכת": "Registered Nurse",
"מנהל חברה": "Company Manager",
"מנהל תחנת מוניות בעברו": "Former Taxi Station Manager",
"מנהל בנק - פנסיונר": "Retired Bank Manager",
"מרצה": "Lecturer",
"קבלן בנין - כעת פנסיוניר": "Construction Contractor - Now Retired",
"בפנסיה - מרצה בעבר": "In Retirement - Former Lecturer",
"דודי שמש": "Solar Water Heater Technician",
"אחות לשעבר": "Former Nurse",
"בפיצריה": "In a Pizzeria",
"מוכנאי": "Mechanic",
"קבלן": "Contractor",
"פנסיונרית - דיטאנית -אחות": "Retired - Dietitian - Nurse",
"טכנאי מטוסים": "Aircraft Technician",
"עקרת בת": "Housewife",
"מורה בדימוס - פנסייה": "Retired Teacher",
"בדיקות קרינה": "Radiation Testing",
"עבד רופא במחלקת ילדים": "Worked as a Doctor in a Pediatric Department",
"מורה נהיגה": "Driving Instructor",
"רופא נשים": "Gynecologist",
"מנהלת במודיעין שדה תעופה": "Airport Intelligence Manager",
"רואה חשבון פנסיונר": "Retired Accountant",
"גימלאי מורה בעבר": "Retired, Former Teacher",
"היה מנהל ביצוע של חב קידוח": "Was the Operations Manager of a Drilling Company",
"תעשיה אוורית": "Aerospace Industry",
"פנסיונר - מורה": "Retired Teacher",
"בעברו נהג אמבולנס": "Former Ambulance Driver",
"פנסיונרית עיתונאית": "Retired Journalist",
"גימלאית עבדה כמורה": "Retired, Worked as a Teacher",
"בגמלאות": "In Retirement",
"גימלאית בעבר עבדה כמזכירה": "Retired, Formerly Worked as a Secretary",
"אלעל קעת פינסיוניר": "El Al, Currently Retired",
"נהג אגד בעברו": "Former Egged Bus Driver",
"מרצה בטכניון": "Lecturer at Technion",
"הייתה מורה": "Was a Teacher",
"בעברו נהג מונית": "Former Taxi Driver",
"היה רפתן בקיבוץ": "Was a Dairy Farmer at a Kibbutz",
"עובד בסופר": "Supermarket Worker",
"עבד במשרד במטחון": "Worked in the Defense Ministry",
"פנסיונר הנדסאי לשעבר": "Retired, Former Engineering Technician",
"אלקטרונאי בפנסיה": "Retired Electronics Technician",
"תיווך": "Real Estate Brokerage",
"1 2 נפטרו": "1 or 2 Passed Away",
"2 ילדים בחוץ לארץ": "2 Children Abroad",
"נכדה אחת": "One Granddaughter",
"נשוי": "Married",
"גרוש": "Divorced",
"אלמן": "Widower",
"רווק": "Single",
"פרוד": "Separated",
"חשבונאות 44": "Accounting",
'מיזוג אוויר':"Air conditioning",
'גימלאת': 'Female Pensioner',
'גמלאית': 'Female Pensioner'

}
    @staticmethod
    def calculate_bmi(height, weight):
        return round(weight / (height / 100) ** 2, 2) # height is in cm, while BMI requires it in meters

    @staticmethod
    def find_strings_in_column(df, column_name):
        # Dictionary to store string values and their indices
        string_dict = {}

        # Iterate over the column to check for string values
        for idx, value in df[column_name].items():
            if isinstance(value, str):
                if value not in string_dict:
                    string_dict[value] = []
                string_dict[value].append(idx)

        return string_dict

    @staticmethod
    def cluster_patients(df, max_clusters=10):
      # Make a copy of the original DataFrame to avoid modifying the original
      df_copy = df.copy()

      # Keep the original numerical and categorical values for visualization
      original_numeric_columns = ['age', 'BMI', 'Height', 'Weight']
      original_categorical_columns = ['Gender', 'Education', 'Marital Status']

      # Convert categorical variables to numerical using LabelEncoder for clustering
      label_encoder = LabelEncoder()

      # Convert mixed-type columns to strings before applying LabelEncoder
      df_copy['Gender'] = df_copy['Gender'].astype(str)
      df_copy['Payer'] = df_copy['Payer'].astype(str)
      df_copy['Education'] = df_copy['Education'].astype(str)
      df_copy['Marital Status'] = df_copy['Marital Status'].astype(str)

      # Apply LabelEncoder to categorical columns
      df_copy['Gender_encoded'] = label_encoder.fit_transform(df_copy['Gender'])
      df_copy['Payer_encoded'] = label_encoder.fit_transform(df_copy['Payer'])
      df_copy['Education_encoded'] = label_encoder.fit_transform(df_copy['Education'])
      df_copy['Marital Status_encoded'] = label_encoder.fit_transform(df_copy['Marital Status'])

      # Select relevant features for clustering, including BMI, Height, Weight, and encoded medication columns
      features = ['age', 'Gender_encoded', 'Chronic Diseases', 'Education_encoded', 'Number of Children',
                  'Marital Status_encoded', 'BMI', 'Height', 'Weight']

      # Check for missing values and drop only if necessary
      if df_copy[features].isnull().values.any():
        df_copy = df_copy.dropna(subset=features)

      # Scale the selected features for KMeans clustering (but keep original values for visualization)
      scaler = StandardScaler()
      scaled_features = scaler.fit_transform(df_copy[features])

      # Perform clustering with the elbow method to determine the optimal number of clusters
      inertia = []
      for n in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42, max_iter=100, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

      # Find the optimal number of clusters using the "elbow" heuristic
      def find_elbow(inertia):
        n_points = len(inertia)
        all_coords = np.vstack((range(1, n_points + 1), inertia)).T
        first_point = all_coords[0]
        last_point = all_coords[-1]
        line_vec = last_point - first_point
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
        vec_from_first = all_coords - first_point
        scalar_product = np.dot(vec_from_first, line_vec_norm)
        vec_on_line = np.outer(scalar_product, line_vec_norm)
        dist_to_line = np.sqrt(np.sum((vec_from_first - vec_on_line) ** 2, axis=1))
        elbow_index = np.argmax(dist_to_line) + 1
        return elbow_index

      optimal_clusters = find_elbow(inertia)

      # Plot the elbow curve with the optimal number of clusters marked
      plt.figure(figsize=(8, 5))
      plt.plot(range(1, max_clusters + 1), inertia, marker='o', linestyle='--')
      plt.scatter(optimal_clusters, inertia[optimal_clusters - 1], color='red', s=100)  # Mark the optimal point
      plt.text(optimal_clusters, inertia[optimal_clusters - 1], f'Optimal: {optimal_clusters}', color='red',
               fontsize=12)
      plt.title('Elbow Method - Optimal Number of Clusters')
      plt.xlabel('Number of Clusters')
      plt.ylabel('Inertia')
      plt.show()

      # Fit the KMeans model with the optimal number of clusters
      kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10, max_iter=100)
      df_copy['Cluster'] = kmeans.fit_predict(scaled_features)

      # Plot histograms for each feature, grouped by clusters
      numeric_features = original_numeric_columns  # Use original numerical values for visualization
      categorical_features = original_categorical_columns  # Use original categorical values for visualization

      # Set up the plot size and layout
      num_plots = len(numeric_features) + len(categorical_features)
      fig, axes = plt.subplots(nrows=num_plots, figsize=(12, 4 * num_plots))  # Increase figure size for visibility

      # Colors for clusters
      colors = sns.color_palette("Set2", n_colors=optimal_clusters)

      # Create a histogram for each numeric feature (using original numerical values)
      for i, feature in enumerate(numeric_features):
        ax = axes[i]
        feature_min = df_copy[feature].min()
        feature_max = df_copy[feature].max()
        for cluster in range(optimal_clusters):  # Add the missing loop over clusters
          sns.histplot(data=df_copy[df_copy['Cluster'] == cluster], x=feature, color=colors[cluster],
                       label=f'Cluster {cluster}', kde=False, ax=ax)

        # Dynamically adjust x-axis limits to fit the data
        ax.set_xlim([feature_min, feature_max])
        ax.set_title(f'Cluster Distribution for {feature}')
        ax.legend(title="Cluster")

        # Dynamically adjust the ticks for x-axis
        ticks = np.linspace(feature_min, feature_max, num=6)
        ax.set_xticks(ticks)

      # Create a histogram for each categorical feature (using original values)
      for j, feature in enumerate(categorical_features):
        ax = axes[len(numeric_features) + j]
        for cluster in range(optimal_clusters):
          sns.histplot(data=df_copy[df_copy['Cluster'] == cluster], x=feature, color=colors[cluster],
                       label=f'Cluster {cluster}', kde=False, ax=ax, discrete=True)
        ax.set_title(f'Cluster Distribution for {feature}')
        ax.legend(title="Cluster")

      plt.tight_layout()
      plt.show()

      # Return the DataFrame with assigned clusters for further analysis
      return df_copy

    @staticmethod
    def clean_height_weight_fix(df, height_col, weight_col):
      """
      This function identifies and fixes extreme values in the height and weight columns of a dataset by applying
      logic to correct common errors instead of removing the data.

      Parameters:
      df (pd.DataFrame): The dataframe containing the data.
      height_col (str): The column name for height.
      weight_col (str): The column name for weight.

      Returns:
      pd.DataFrame: The cleaned dataframe with adjusted height and weight values.
      """

      # Make a copy of the original dataframe to avoid modifying it directly
      df_cleaned = df.copy()

      # Step 1: Handle height outliers
      # Convert height in meters (heights below 30 cm are likely in meters)
      df_cleaned[height_col] = df_cleaned[height_col].apply(lambda h: h * 100 if h < 30 else h)

      # Fix cases where the height might be 10x too large (above 250 cm), likely a factor of 10 error
      df_cleaned[height_col] = df_cleaned[height_col].apply(lambda h: h / 10 if h > 250 else h)

      # Step 2: Handle weight outliers
      # we will assume that anu weight above 300 was accidentally written in grams.
      df_cleaned[weight_col] = df_cleaned[weight_col].apply(lambda w: w / 1000 if w > 300 else w)
      return df_cleaned






























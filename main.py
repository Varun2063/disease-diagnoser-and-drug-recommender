import streamlit as st
from intro import pred

def init_session_state():
 return {
  "selected_symptoms": [],
  "selected_severity": []
 }

session_state = init_session_state()

if "session_state" not in st.session_state:
 st.session_state.session_state = init_session_state()

st.title("Disease diagnosis")

user_input = st.text_input("Type something here:")

symptoms = ['Upper abdominal pain', 'Lower abdominal pain', 'Abscess (Collection of pus)', 'Alcohol abuse', 'Anxiety (Nervousness)', 'Arm ache or pain', 'Back ache or pain', 'Bleeding tendency', 'Blood in vomit', 'Bloody diarrhea', 'Pain or soreness of breast', 'Calf pain', 'Chest pressure', 'Chills', 'Change in behavior', 'Constipation', 'Cough', 'Dark stools', 'Depressed', 'Diarrhea', 'Dizziness', 'Double vision (Diplopia)', 'Ear pressure', 'Pain in the ear', 'Elbow ache or pain', 'Eye pain (Irritation)', 'Facial pain', 'Fainting', 'Fever', 'Fever in the returning traveler', 'Fever of unknown origin', 'Flank pain', 'Frequent urination (Frequency)', 'Foot pain', 'Cranky, crying, fussy, irritable child', 'Groin pain', 'Delusions or hallucinations', 'Hand, finger ache or pain', 'Head injury', 'Headache', 'Heel pain', 'Heat illness', 'Hip pain', 'Hives', 'Hoarse voice', 'Hypothermia (Low temperature)', 'Incontinence (leaking urine)', 'Insect sting', 'Insomnia (Trouble sleeping)', 'Skin itching', 'Joint pain', 'Kidney pain (Flank pain)', 'Knee pain', 'Laceration', 'Leg ache or pain', 'Swelling of both legs', 'Lethargy (Sluggishness)', 'Mouth pain', 'Muscle pain', 'Nail Injury', 'Nasal bleeding', 'Nasal injury', 'Nausea', 'Neck ache or pain', 'Neck swelling', 'Numbness', 'Obesity', 'Overdose', 'Painful urination', 'Heart pulsations and palpitations', 'Pelvic pain', 'Penile discharge', 'Penis pain', 'Poisoning', 'Pregnancy problem', 'Psychiatric problem', 'Puncture wound', 'Rash', 'Rectal pain', 'Rectal swelling', 'Scrotal pain', 'Scrotal swelling', 'Seizure', 'Shortness of breath', 'Shoulder ache or pain', 'Sinus pain and pressure', 'Skin trauma', 'Snake bite', 'Sore throat', 'Speech problem', 'Spider bite', 'Substance abuse (Drug abuse)', 'Suicidal tendencies', 'Swallowing problem (Dysphagia)', 'Swelling', 'Toe pain', 'Tooth pain', 'Trauma', "Traveler's diarrhea", 'Unsteady gait (Trouble walking)', 'Vaginal bleeding', 'Vaginal bleeding after menopause', 'Vaginal bleeding during pregnancy', 'Vaginal discharge', 'Vaginal itching', 'Vaginal pain', 'Vertigo (Room spinning)', 'Visual problems', 'Vomiting', 'General weakness', 'Weakness (Muscle localized)', 'Tired', 'Wrist pain', 'Throat pain', 'Tremors', 'Weight loss, unexplained', 'Tongue swelling', 'Inconsolable baby', 'Wheezing (Noisy breathing)', 'Swollen lymph nodes (Large lymph nodes)', 'Failure to thrive', 'Behavioral problem', 'Itchy rash (Pruritic rash)', 'Headache after trauma', 'Learning difficulties', 'Blood in urine (Hematuria)', 'Urinary retention (Inability to urinate)', 'Liver failure (Cirrhosis)', 'Choking', 'Painful rash', 'Ingestion', 'Melena (Black stools from blood)', 'Vomiting coffee ground material', 'Ringing in ears (Tinnitus)', 'Mouth ulcers', 'Mouth swelling', 'Eye redness', 'Sneezing', 'Bleeding gums', 'Loss of balance', 'Bleeding in brain', 'Cyanosis (Blue skin coloration)', 'Muscle spasm', 'Drooling', 'Abdominal swelling (Stomach swelling)', 'Skin growths', 'Hand numbness (paresthesias)', 'Ankle pain', 'Hemoptysis (Coughing blood)', 'Jaundice (Yellowing skin)', 'Night sweats', 'Flatulence (Passing gas)', 'Blister (Pocket of fluid)', 'Hair loss (Baldness)', 'Jaw pain', 'Impotence', 'Heart murmur (Abnormal heart sound)', 'Pustule (Collection of pus)', 'Skin pain', 'Hot skin', 'Skin swelling', 'Lip swelling', 'Eye swelling', 'Foot swelling', 'Visual flashing lights', 'Eye floaters', 'Amenorrhea (No menstruation)', 'Blurry vision', 'Painful gums', 'Swollen gums', 'Low blood sugar', 'Low blood pressure', 'Darkening of the skin (Hyperpigmentation)', 'Low heart rate', 'Foot itching', 'Hot flashes', 'Infertility (Female)', 'Increased facial hair', 'Arm swelling', 'Calf swelling', 'Ear swelling', 'Wrist swelling', 'Maroon stools', 'Arm cut (laceration)', 'Hand cut (laceration)', 'Leg cut (laceration)', 'Foot cut (laceration)', 'Arm itching', 'Hand redness', 'Foot redness', 'Arm redness', 'Leg redness', 'Hand itching', 'Leg itching', 'Steatorrhea (Excess fat in stool)', 'Upper leg pain', 'Armpit pain', 'Sweating', 'Nasal congestion', 'Joint stiffness', 'Skin sores', 'Chest burning', 'Memory loss', 'Arm numbness (paresthesias)', 'Leg numbness (paresthesias)', 'Foot numbness (paresthesias)', 'Face numbness (paresthesias)', 'Dementia', 'Facial droop (weakness)', 'Limping in a child', 'Increased thirst', 'Increased urination (polyuria)', 'Shin pain', 'Stings', 'Sleep disorders', 'Drooping eyelid (Ptosis)', 'Snoring', 'Dry skin', 'Itchy eyes', 'Elbow swelling', 'Chest pain', 'Skin infection', 'Stomach and abdominal pain', 'Anger', 'Hurts to breathe', 'Difficulty breathing', 'Pulling at ears', 'Skin bumps', 'Congestion in chest or lungs', 'Discharge from ear', 'Low back ache or pain', 'Unusual color or odor of urine', 'Penis inflammation or swelling', 'Excessive appetite', 'Retaining fluid', 'Lump or mass of breast', 'Neck stiffness or tightness', 'Agitated', 'Confusion', 'Headache and weakness', 'Confusion and headache', 'Nipple discharge', 'Shoulder stiffness or tightness', 'Arm stiffness or tightness', 'High blood pressure', 'High blood sugar']

filtered_symptoms = [option for option in symptoms if user_input.lower() in option.lower()]

selected_symptoms = st.selectbox("Select an symptom:", options=filtered_symptoms)

selected_wei = st.selectbox("Select a value (1 to 3):", options=[1, 2, 3])

if st.button("Enter"):
 if selected_symptoms:
  st.session_state.session_state["selected_symptoms"].append(selected_symptoms)
 if selected_wei:
  st.session_state.session_state["selected_severity"].append(selected_wei)

st.write("Updated Selected symptoms:", st.session_state.session_state["selected_symptoms"])
st.write("Updated Selected severity:", st.session_state.session_state["selected_severity"])

if st.button("Diagnose"):
 pred(st.session_state.session_state["selected_symptoms"])
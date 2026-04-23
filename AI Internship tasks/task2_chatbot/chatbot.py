# chatbot.py — NLP logic + FAQ matching

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# ─── FAQ Data — Hospital/Medical ───────────────────────────────
faqs = [
    # Hospital Procedures
    {
        "question": "What are the hospital visiting hours?",
        "answer": "Visiting hours are from 9:00 AM to 8:00 PM daily."
    },
    {
        "question": "How do I book an appointment?",
        "answer": "You can book an appointment by calling 0300-1234567 or visiting our website."
    },
    {
        "question": "What is the appointment procedure?",
        "answer": "To book an appointment: Call 0300-1234567, give your name and symptoms, choose a doctor, and confirm your time slot. You can also book online at our website."
    },
    {
        "question": "What documents are required for admission?",
        "answer": "You need your CNIC, previous medical records, and a doctor's referral letter."
    },
    {
        "question": "Is emergency service available 24 hours?",
        "answer": "Yes, our emergency department is open 24/7 including weekends and holidays."
    },
    {
        "question": "What is the fee for an OPD appointment?",
        "answer": "OPD fee is Rs. 500 for general consultation and Rs. 1000 for specialist doctors."
    },
    {
        "question": "Do you accept health insurance?",
        "answer": "Yes, we accept most major health insurance plans including Jubilee, EFU, and State Life."
    },
    {
        "question": "Where is the hospital located?",
        "answer": "We are located at Main Boulevard, Gulberg, Lahore. Opposite to Punjab University."
    },
    {
        "question": "What specialist doctors are available?",
        "answer": "We have cardiologists, neurologists, orthopedic surgeons, pediatricians, and gynecologists."
    },
    {
        "question": "How do I get my lab test results?",
        "answer": "Lab results are available online through our patient portal within 24 hours."
    },
    {
        "question": "Is parking available at the hospital?",
        "answer": "Yes, free parking is available for patients and visitors on the basement floor."
    },
    {
        "question": "What should I do in case of emergency?",
        "answer": "Call our emergency number 1122 or come directly to the emergency entrance on the ground floor."
    },
    {
        "question": "Can I get a doctor consultation online?",
        "answer": "Yes, we offer telemedicine consultations. Book through our website or app."
    },

    # Fever
    {
        "question": "I have fever what should I do?",
        "answer": "For fever, you should see a General Physician (GP). Rest well, drink plenty of water, and take paracetamol. If fever is above 103°F or lasts more than 3 days, visit our OPD immediately. Book appointment: 0300-1234567."
    },
    {
        "question": "Which doctor should I see for fever?",
        "answer": "Please visit our General Physician (GP) for fever. OPD timing is 9AM-5PM. Fee is Rs. 500. Call 0300-1234567 to book an appointment."
    },
    {
        "question": "I am feeling very hot and have high temperature",
        "answer": "High temperature indicates fever. Please visit our General Physician (GP). Take paracetamol and drink lots of water. If temperature is above 103°F, go to emergency immediately."
    },

    # Headache
    {
        "question": "I have a headache what should I do?",
        "answer": "For headache, visit our General Physician first. If headaches are frequent or severe, we will refer you to our Neurologist. Book appointment at 0300-1234567."
    },
    {
        "question": "Which doctor to see for head pain?",
        "answer": "For head pain or headache, see our General Physician (GP) or Neurologist. Book your appointment by calling 0300-1234567."
    },
    {
        "question": "I have severe migraine",
        "answer": "Migraine is treated by our Neurologist. Please book an appointment at 0300-1234567. Avoid bright lights, rest in a quiet room, and take prescribed pain medication."
    },

    # Stomach Problems
    {
        "question": "I have stomach pain what should I do?",
        "answer": "For stomach pain, visit our General Physician or Gastroenterologist. Avoid spicy food and drink plenty of water. Book appointment: 0300-1234567."
    },
    {
        "question": "I have vomiting and nausea",
        "answer": "Vomiting and nausea can be treated by our General Physician. Stay hydrated and avoid solid food temporarily. If vomiting is severe, go to emergency. Call 0300-1234567."
    },
    {
        "question": "I have diarrhea or loose motions",
        "answer": "For diarrhea, drink ORS (oral rehydration solution) and visit our General Physician. If it lasts more than 2 days, visit immediately. Book at 0300-1234567."
    },

    # Heart Problems
    {
        "question": "I have chest pain what should I do?",
        "answer": "Chest pain is serious! Go to our Emergency immediately or call 1122. Our Cardiologist will examine you. Do not ignore chest pain as it could be a heart issue."
    },
    {
        "question": "Which doctor to see for heart problems?",
        "answer": "For heart problems, see our Cardiologist. Book an appointment at 0300-1234567. For emergency heart issues, go to Emergency immediately."
    },
    {
        "question": "I have shortness of breath and breathing problem",
        "answer": "Breathing problems need immediate attention. Visit our Emergency or see our Pulmonologist. Call 1122 if it is severe. Book appointment: 0300-1234567."
    },

    # Bone and Joint Pain
    {
        "question": "I have back pain what should I do?",
        "answer": "For back pain, visit our Orthopedic Surgeon or Physiotherapist. Avoid heavy lifting. Book appointment at 0300-1234567."
    },
    {
        "question": "I have joint pain and knee pain",
        "answer": "Joint and knee pain is treated by our Orthopedic Surgeon. Book an appointment at 0300-1234567. Avoid strenuous activity until examined."
    },

    # Skin Problems
    {
        "question": "I have skin rash or allergy",
        "answer": "For skin rash or allergy, visit our Dermatologist. Avoid scratching the affected area. Book appointment at 0300-1234567."
    },
    {
        "question": "Which doctor to see for skin problems?",
        "answer": "For any skin problem, please visit our Dermatologist. OPD timing is 9AM-5PM. Book at 0300-1234567."
    },

    # Eye Problems
    {
        "question": "I have eye pain or blurry vision",
        "answer": "For eye problems, visit our Ophthalmologist (Eye Specialist). Book appointment at 0300-1234567. Do not rub your eyes."
    },

    # Diabetes
    {
        "question": "I have diabetes what should I do?",
        "answer": "For diabetes management, visit our Endocrinologist. Follow a healthy diet, exercise regularly, and monitor blood sugar. Book appointment at 0300-1234567."
    },
    {
        "question": "I have high blood sugar",
        "answer": "High blood sugar needs immediate attention. Visit our Endocrinologist or General Physician. Avoid sugary foods and drinks. Call 0300-1234567."
    },

    # Blood Pressure
    {
        "question": "I have high blood pressure",
        "answer": "High blood pressure is treated by our Cardiologist or General Physician. Reduce salt intake, avoid stress, and take prescribed medication. Book at 0300-1234567."
    },

    # Children Problems
    {
        "question": "My child is sick which doctor to see?",
        "answer": "For children's health issues, visit our Pediatrician. We have experienced child specialists available. Book appointment at 0300-1234567."
    },
    {
        "question": "My child has fever and cough",
        "answer": "For children with fever and cough, visit our Pediatrician immediately. Keep the child hydrated. Book appointment at 0300-1234567."
    },

    # Cough and Cold
    {
        "question": "I have cough and cold",
        "answer": "For cough and cold, visit our General Physician. Drink warm water, rest well, and avoid cold drinks. Book appointment at 0300-1234567."
    },
    {
        "question": "I have sore throat",
        "answer": "For sore throat, visit our General Physician or ENT Specialist. Gargle with warm salt water and avoid cold food. Book at 0300-1234567."
    },

    # Women Health
    {
        "question": "I need to see a gynecologist",
        "answer": "Our Gynecologist is available for all women health issues. Book an appointment at 0300-1234567. OPD timing is 9AM-5PM."
    },
    {
        "question": "I am pregnant which doctor should I see?",
        "answer": "For pregnancy care, visit our Gynecologist and Obstetrician. We provide full maternity care. Book appointment at 0300-1234567."
    },

    # General
    {
        "question": "What should I do if I feel very sick?",
        "answer": "If you feel very sick, come to our Emergency immediately or call 1122. Our team is available 24/7 to help you."
    },
    {
        "question": "Which doctor should I see?",
        "answer": "It depends on your symptoms. For fever, cold, general illness - see General Physician. Heart issues - Cardiologist. Bones/joints - Orthopedic. Children - Pediatrician. Women health - Gynecologist. Call 0300-1234567 for guidance."
    },
]

# ─── Text Preprocessing ────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ─── Preprocess all FAQ questions ─────────────────────────────
faq_questions = [preprocess(faq['question']) for faq in faqs]

# ─── Find Best Match using Cosine Similarity ──────────────────
def get_answer(user_question):
    processed_question = preprocess(user_question)
    all_questions = faq_questions + [processed_question]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_questions)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.05:
        return "I'm sorry, I couldn't find an answer to your question. Please call us at 0300-1234567 or visit our Emergency for urgent help."

    return faqs[best_match_index]['answer']


# ─── Test it directly ─────────────────────────────────────────
if __name__ == '__main__':
    print("Hospital FAQ Chatbot — Type 'quit' to exit\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        answer = get_answer(user_input)
        print(f"Bot: {answer}\n")
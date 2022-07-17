import streamlit as st
import pickle
import numpy as np


# svml = pickle.load(open('svml.pkl','rb'))
loaded_model = pickle.load(open(r'C:\Users\takniya\Desktop\project-STU\trained_model.sav', 'rb'))

def main():
    st.title("System to predict the nearest university major, for high school graduates")
    st.write('personal information')
    st.write('Note: Enter your data in digital form')
    G = st.text_input('Enter your Gender, Male(1), Female(2)')
    HS = st.text_input('Enter your Hige School, Scientific(1), Litreture(2)')
    N = st.text_input('Enter your Nationality, Libyan(1), Other(2)')
    SC = st.text_input('Enter your University class you want to enroll in at the university, Fall(1), Spring(2)')
    AD = st.text_input('Enter the Residential Address, Sebha(1), Other(2)')
    SS = st.text_input('Enter Marital Status, single(1), engaged(2), married(3), divorced(4)')
    HS_AVG = st.text_input('Enter the percentage in secondary as a whole number, such as (78)')
    st.write('Student personal questions')
    Q4 = st.text_input('Q1: Have you enrolled in this specialty based on?, Your desire(1), faculty(2), friend(3), family(4)')
    Q11 = st.text_input('Q2: is the desire to work in private companies the main reason for choosing your specialty?, yes(1), no(2)')
    Q12 = st.text_input('Q3: is the desire to work in the public sectors the main reason for choosing your specialty?, yes(1), no(2)')
    Q13 = st.text_input('Q4: father education level?, Below level(1), primary(2), prepatory(3), secondary(4), university graduate(5), high(6)')
    Q14 = st.text_input('Q5: mother education level?, Below level(1), primary(2), prepatory(3), secondary(4), university graduate(5), high(6)')
    Q15 = st.text_input('Q6: do you have elder brothers?, yes(1), no(2)')
    Q16 = st.text_input('Q7: what is their education level (majority)?, none(1), primary(2), prepatory(3), secondary(4), university graduate(5)')
    Q28 = st.text_input('Q8: father profession?, employee(1), business(2), freelancer(3), retired(4), no work(5)')
    Q29 = st.text_input('Q9: mother profession?, employee(1), retired(2), housewife(3)')
    Q35 = st.text_input('Q10: does your familys financial means allow you to enroll in training or specialized courses that help you excel in your university studies?, yes(1), no(2)')
    Q38 = st.text_input('Q11: is your choice of university major based on the university proximity to the place of residence?, yes(1), no(2)')

    if st.button("Process"):
        data=[[G, HS, N, SC, AD, SS, HS_AVG, Q4, Q11,Q12,Q13,Q14,Q15,Q16,Q28,Q29,Q35,Q38]]
        prediction=loaded_model.predict(data)
        st.success(prediction)
        st.success('[1] College of Literature, [2] College of Science,   [3] College of IT, [4] College of Economics,  [5] College of Medicine, [6] College of Education,  [7] College of Engineering')

        # if(prediction[0] == 0):
        #     st.success('The person is 1')
        # elif (prediction[0]+-4 == prediction[0]):
        #     st.success(prediction,'The person is 2 ')
        # elif (prediction[0]+-3 == prediction[0]):
        #     st.success(prediction,'The person is 3 ')
        # elif (prediction[0]+-2 == prediction[0]):
        #     st.success(prediction,'The person is 4 ')
        # elif (prediction[0]+-1 == prediction[0]):
        #     st.success(prediction,'The person is 5 ')
        # elif (prediction[0]+0 == prediction[0]):
        #     st.success(prediction)
        #     st.success('The person is 6 ')
        # elif(prediction[0]+1 == prediction[0]):
        #     st.success(prediction,'The person is 7 ')
main()

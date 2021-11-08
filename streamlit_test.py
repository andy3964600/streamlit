import streamlit as st
import numpy as np
import pandas as pd
import time

# interactive widgets
st.text('# My First App From Streamlit')

# button
button = st.button('Click this below!')
if button:
    st.write('123')
else:
    st.write('321')

# checkbox
st.checkbox('checkbox example')
agree = st.checkbox('I agree')
if agree:
    st.write('Great')
else:
    st.write('you didnt select great?')

# Radio
st.radio('Radio', ['wow', 2, 3])
qual = st.radio(
    'what is your highest qualification?',
    ('Bachelors', 'Masters', 'PhD'))
if qual == 'Bachelors':
    st.write('Great! you have completed 16 years of education')
elif qual == 'Masters':
    st.write('Wow! You have completed 18 years of education')
else:
    st.write('You are a highly qualified person!')

# selectbox
st.selectbox('Select', [1, 2, 3])
option = st.selectbox(
    'What is your favorite social media platform?',
    ('Facebook', 'Instagram', 'Whatapp', 'LinkedIn', 'Snapchat'))

st.write('You selectd:', option)

# multiselect
st.multiselect('multiselect', [1, 2, 3])

options = st.multiselect(
    'What is your favorite car brands?',
    ['Toyota', 'Mazda', 'Ferrari', 'Aston Martin', 'Rolls Royce'])
st.write('You select:', options)

# Slider
st.slider('Slide me', min_value=0, max_value=10)
age = st.slider('How old are you?', 0, 110, 25)
st.write('Im', age, 'years old')
value = st.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0))
st.write('Value:', value)

x = st.slider('民國幾年出生的?')
st.write('民國', x, '年出生')
st.write('你出生為西元', x+1911, "年")

# spinner
with st.spinner('Wait for it...'):
    time.sleep(10)
st.success('Done!')

st.text('More example of streamlit progress widgets')

st.spinner()
with st.spinner(text='Download in progress!!!'):
    time.sleep(5)
st.success('Done')

# Import library
from nbformat import write
# from st_aggrid import AgGrid
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

# thu vien Tokenizer Viet
from pyvi import ViTokenizer, ViPosTagger
from underthesea import word_tokenize, pos_tag, sent_tokenize
from wordcloud import WordCloud

# data pre-processing libraries
import regex
import demoji
import emoji
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split  

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# model
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# evaluation libraries
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn import metrics

# function libraries
import sys
import os
import re

import import_ipynb
from lib.function_lib import *
from lib.pre_underthesea_lib import *


######################################################################################################
raw_data = pd.read_csv('data/data_Foody.csv')
raw_data = raw_data.loc[:, ~raw_data.columns.str.contains('^Unnamed')]


# GUI
st.sidebar.markdown("<h1 style='text-align: left; color: Black;'>CATALOG</h1>", unsafe_allow_html=True)
menu = ["Introduction","Summary about Projects", 'Sentiment Analysis', "Conclusion and Next steps"]
choice = st.sidebar.radio("Choose one of objects below", menu)

st.write("<h1 style='text-align: left; color: Red; font-size:40px'>FOOD SENTIMENT ANALYSIS</h1>", unsafe_allow_html=True)


# 1. Introduction
if choice == 'Introduction':
    st.header('INTRODUCTION')
    st.write('''According to Microsoft, **52%** of people around the globe believe that companies 
                need to take action on the feedback provided by customers. So as a company 
                or business owner, you must give great importance to feedback analysis.''')
    st.write('''Getting customer feedback from your consumers is one thing, 
                and analyzing it is another. Most companies collect enormous amounts of feedback from 
                their customers, but many don’t use it to improve their products and services.''')
    st.write('''Because of poor customer service experiences, **56%** of people around the world 
                have stopped doing business with a company. And poor customer service experience 
                is one of the significant outcomes when customer feedback isn’t correctly analyzed.''')
    st.write('''**Feedback analysis** is one of the most critical steps once 
                you have collected your customers’ suggestions. And doing it the right way 
                is also essential for your company’s growth. ''')
    st.image('material/Intro_Sentiment.jpg')
    st.write('''Analyzing thousands of customer feedback manually isn’t an efficient way 
                to do the job. The best tip we can come up with on how to analyze customer feedback 
                is investing in automated tools.''')
    st.write('''**Automated tools** will not just make your feedback analysis easy, 
                but they will come up with great insights about it. 
                In minutes, you’ll see detailed reports and insights about the 
                customer feedback data you have collected.''')
    st.write('''Data analysis automated tools will help you to analyze customer feedback 
                in less time and provide great insights.''')
    st.write('''Listening to your customers and implementing their wishes will help 
                your business to reach new heights of success.''')


# 2. Summary about Projects
elif choice == 'Summary about Projects':
    st.header('SUMMARY ABOUT PROJECTS')
    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Business Understanding</h3>", unsafe_allow_html=True)
    st.write("**Current status:**")
    st.write('''On foody.vn, there are many restaurants doing business with diverse and 
                rich products and menus. To increase the level of interaction with customers, 
                each restaurant has a comment area for customers to give reviews, comments and scores. 
                Instead of having to conduct customer survey campaigns, analyzing customer 
                comments for food on foody.vn will be faster and more effective.''')
    

    st.write("**Objective/Problem:**")
    st.write('''Sentiment analysis is a complicated and sometimes personal opinion. 
                Manual analysis (ie reading each comment and classification) will give 
                inconsistent results between the analysts and take a lot of time. 
                Natural language analysis algorithms, especially sentiment, 
                are very developed. The application of NLP algorithms to predict 
                customer sentiment will help restaurants to improve service quality in a timely manner.''')

    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Data Understanding/Acquire</h3>", unsafe_allow_html=True)
    st.write("- Dataset has 3 columns: restaurant, review_text, review_score.")
    st.write("- review_score: from 1 to 10. The higher the score, the higher the satisfaction.")
    st.write('''- Create new feature "review_score_new": class "positive" if review_score < 7, 
                class "negative" if review_score >= 7. Find out the imbalance between classes. 
                Class "posotive" is more than class "negative". But we focus on class "negative"''')
    st.write("- review_text: Vietnamese language.")
    
    st.write("<h6 style='text-align: left; color: Black; font-size:15px'>First 5 rows of dataset:</h6>", unsafe_allow_html=True)
    st.dataframe(raw_data.head())

    st.write("<h6 style='text-align: left; color: Black; font-size:15px'>General information about dataset:</h6>", unsafe_allow_html=True)
    # code here

    st.write("<h6 style='text-align: left; color: Black; font-size:15px'>Wordcloud:</h6>", unsafe_allow_html=True)
    # code here

    st.write("<h6 style='text-align: left; color: Black; font-size:15px'>Scatter text:</h6>", unsafe_allow_html=True)
    st.image('material/EDA_Scatter_text_image.jpg')

    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Data Preparation/Acquire</h3>", unsafe_allow_html=True)
    st.write("1. Use word_tokenizer from Underthesea library.")
    st.write('''2. Convert emojicons to words: Extract enojicons from text dataset, 
                scan each emojicon and define to the meaning words. 
                Load edited emojicons for data cleaning.''')
    st.write('''3. Remove some special symbols: List down list of special symbols, 
            such as: '~', '`', '!', '@', '#', '$',... ''')
    st.write('''4. Remove some typing mistakes: " " (2 spaces), " " (1 space, 1 underscore), " " (1 underscore, 1 space)''')
    st.write('''5. Link some special words with other words: 
                Link "không", "ko", "kg", "chả", "chẳng" to the word right behind. 
                Ex: "không thích" => "không_thích"''')
    st.write('''6. Remove stopwords''')
    st.image('material/Pre_steps.jpg')
 
   
    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Modeling and Evaluation</h3>", unsafe_allow_html=True)
    st.write("1. Use CountVectorizer to transform text data to bag_of_words.")
    st.write("2. Do Train Test spliting with rate = 70:30 => X_train, X_test, y_train, y_test")
    st.write("3. Do Undersampling => X_train_us, X_test_us, y_train_us, y_test_us")
    st.write('''4. Do Oversampling => X_train_os, X_test_os, y_train_os, y_test_os.
                But models built from oversampling dataset are not good as model built from undersampling dataset.''')    
    st.write("5. Select general model by LazyPredict. There are serveral models that have high performance: LGBMClassifier, BernoulliNB, XGBClassifier, ExtraTreesClassifier.")
    st.dataframe(pd.read_csv('model/LazyClassifier_models_us_self.csv'))
    st.write("6. Build each models one-by-one.")
    st.write("7. Try another Naive Bayes model, named MultinomialMB.")
    st.write("8. Because of doing sentiment analysis to improve customer service, we focus on recall score of class 'negative'. The MultinomialMB is the best model.")    
    st.write("Evaluation metrics:")
    st.write("&nbsp;"*9 + "- Recall score of class 'negative' = 78% (highest among models)")
    st.write("&nbsp;"*9 + "- Recall score of class 'positive' = 90%")
    st.write("&nbsp;"*9 + "- Precision of class 'negative' = 66%")
    st.write("&nbsp;"*9 + "- Precision of class 'positive' = 94%")
    st.write("&nbsp;"*9 + "- Macro F1-score = 82%")
    st.write("MultinomialMB model will be used for deploying sentiment anaysis.")

    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Suggestion</h3>", unsafe_allow_html=True)
    # code here


#####################################################################################################

# 3. Sentiment Analysis
elif choice == 'Sentiment Analysis':

#----------------------------------------------------------------------------------------------------
    # if __name__=='__main__':
    #     model_pre = pickle.load(open('model/Project2_model_pre_processing.sav', 'rb'))
    #     model_predict = pickle.load(open('model/Project2_mb_model_prediction.sav', 'rb'))
    def data_cleaning(data):
        df = data.copy()

        # remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # feature "review_text"
        df_clean = df
        document = df_clean['review_text']

        # load edited emojicon
        file = open('lib/files/emojicon.txt', 'r', encoding="utf8")
        emoji_lst = file.read().split('\n')
        file.close()

        emoji_dict = {}
        lst_key = []
        lst_value = []
        for line in emoji_lst:
            key, value = line.split('\t')
            lst_key.append(key)
            lst_value.append(value)
            emoji_dict[key] = str(value)

        # load chosen words
        file = open('lib/files/chosen_words_full.txt', 'r', encoding="utf8")
        lst_chosen_words = file.read().split('\n')
        file.close()

        lst_spec_symbol = ['~','`','!','@','#','$','%','^','&','*','(',')','-','=','+',   '[','{','}','}','\\','|',';',':','"',     ',','<','.','>','/','?']
        lst_find_words = ['không ','ko ','kg ','chẳng ','chả ']
        lst_replace_words = ['không_','ko_','kg_','chẳng_','chả_']

        lst_document = []
        for doc in document:
            doc = doc.lower()

            # TEXT CLEANING
            # replace emojicons
            for i in range(len(lst_key)):
                doc = doc.replace(lst_key[i], ' '+lst_value[i])
            for j in doc:
                if j in emoji.UNICODE_EMOJI['en']: 
                    doc = doc.replace(j, '')

            # word tokenize: sạch sẽ => sạch_sẽ
            doc = word_tokenize(doc, format='text')

            # remove special symbols
            rx = '[' + re.escape(''.join(lst_spec_symbol)) + ']' 
            doc = re.sub(rx, '', doc)

            doc = doc.replace('  ',' ')
            doc = doc.replace(' _',' ')
            doc = doc.replace('_ ',' ')

            # replace 'không ' to 'không_' co link words, v.v...
            for i in range(len(lst_find_words)):
                doc = doc.replace(lst_find_words[i], lst_replace_words[i]) 

            # remove stop_words
            lst_words = []
            for j in doc.split(' '):
                if j in lst_chosen_words: lst_words.append(j)
            doc = ' '.join(lst_words)  

            lst_document.append(doc)

        df_clean['review_text_clean'] = lst_document
        # df_clean = df_clean.dropna()

        # create "review_score_new"
        df_clean['review_score_new'] = 0
        df_clean.loc[df_clean['review_score'] >= 6.8, 'review_score_new'] = 1
        df_clean['review_score_new'] = df_clean['review_score_new'].astype('int32')

        # load encoder model
        cv = pickle.load(open('model/CountVectorizer_self_model.sav','rb'))

        # COUNTVECTORIZER
        # apply CountVectorizer
        # cv_transformed = cv.transform(df_clean['review_text_clean'].dropna())
        cv_transformed = cv.transform(df_clean['review_text_clean'])
        cv_transformed = cv_transformed.toarray()
        df_clean_cv = pd.DataFrame(cv_transformed, columns=cv.get_feature_names())
        df_clean_cv = pd.concat([df_clean.reset_index(), df_clean_cv], axis=1)

        # CREATE X, y   
        X = df_clean_cv.iloc[:,6:]
        y = df_clean_cv['review_score_new']

        return X, y

#--------------------------------------------------------------------------------------
# clean text v2
    ##LOAD EMOJICON
    file = open('lib/files/emojicon.txt', 'r', encoding="utf8")
    emoji_lst2 = file.read().split('\n')
    emoji_dict2 = {}
    for line in emoji_lst2:
        key, value = line.split('\t')
        emoji_dict2[key] = str(value)
    file.close()

    #LOAD TEENCODE
    file = open('lib/files/teencode.txt', 'r', encoding="utf8")
    teen_lst2 = file.read().split('\n')
    teen_dict2 = {}
    for line in teen_lst2:
        key, value = line.split('\t')
        teen_dict2[key] = str(value)
    file.close()

    #LOAD TRANSLATE ENGLISH -> VNMESE
    file = open('lib/files/english-vnmese.txt', 'r', encoding="utf8")
    english_lst2 = file.read().split('\n')
    english_dict2 = {}
    for line in english_lst2:
        key, value = line.split('\t')
        english_dict2[key] = str(value)
    file.close()

    #LOAD wrong words
    file = open('lib/files/wrong-word.txt', 'r', encoding="utf8")
    wrong_lst2 = file.read().split('\n')
    file.close()

    #LOAD STOPWORDS
    file = open('lib/files/vietnamese-stopwords.txt', 'r', encoding="utf8")
    stopwords_lst2 = file.read().split('\n')
    file.close()

    def clean_text(text):
        document = process_text(text, emoji_dict2, english_dict2, teen_dict2, wrong_lst2)
        document = covert_unicode(document)
        document = remove_stopword(document, stopwords_lst2)

        return document

    # Load tokenizer
    with open('model/tokenizer.pkl', 'rb') as handler:
        tok = pickle.load(handler)

#-----------------------------------------
# predict any restaurant
# read data review
    # data = pd.read_csv('data/data_Foody.csv', index_col=0)

#----------------------------------------------------------------------------------------------------

    # content
    st.header('SENTIMENT ANALYSIS')

    # sidebar
    menu3_input = ['Choose restaurant', 'Input data', 'Load file']
    choice3_input = st.sidebar.selectbox('Choose the way to input data', menu3_input)

    if choice3_input == 'Choose restaurant':
        restaurant_name = st.sidebar.selectbox('Restaurant Name:', raw_data['restaurant'])
        submit_button = st.sidebar.button(label='Summit')

        # predict function for any restaurant
        def predict_sentiment(restaurant_name):
            # get restaurnant information:
            df_restaurant = raw_data.loc[raw_data['restaurant'] == restaurant_name]
            df_restaurant.reset_index(inplace=True, drop=True)

            # clean data
            df_restaurant['text_cleaned'] = df_restaurant.review_text.apply(lambda x: clean_text(str(x)))
            
            # text to sequence
            max_len = 100
            X_new = df_restaurant['text_cleaned']
            text_sequences = tok.texts_to_sequences(X_new)
            text_sequences_matrix = pad_sequences(text_sequences, maxlen=max_len)

            # predict data from loaded model
            y_pred_prob = model_loaded_sentiment.predict(text_sequences_matrix)
            y_pred_class = y_pred_prob >= 0.5

            # save result
            df_restaurant['predict_prob'] = y_pred_prob
            df_restaurant['sentiment'] = y_pred_class
            df_restaurant.loc[df_restaurant['sentiment'] == 1, 'sentiment'] = 'positive'
            df_restaurant.loc[df_restaurant['sentiment'] == 0, 'sentiment'] = 'negative'

            result = df_restaurant.drop('text_cleaned', axis=1)

            return result

        # body
        if submit_button:
            result = predict_sentiment(restaurant_name)
            n_review = result.shape[0]
            n_review_postitive = sum(1 for x in result['sentiment'] if x == 'positive')
            n_review_negative = sum(1 for x in result['sentiment'] if x == 'negative')
            avg_score = result['review_score'].mean()
            
            st.markdown('***Restaurant***')
            st.write('**%s** has totally **%s** reviews, including:'%(restaurant_name, n_review))
            st.write('+ **%s** positive review'%(n_review_postitive))
            st.write('+ **%s** negative review'%(n_review_negative))
            st.write('+ Average review score: **%.2f**'%avg_score)
            st.write('*For detail as table below:*')
            st.dataframe(result)

    elif choice3_input == 'Input data':  
        
        menu3_model = ['Model 1', 'Model 2']
        choice3_model = st.sidebar.selectbox('Choose the model', menu3_model)

        # Guidance
        with st.expander("See Guidance"):
            st.write("<h5 style='text-align: left; color: Black;'>Guidance:</h5>", unsafe_allow_html=True)
            st.write("To use specific models to predict sentiment, please follow these steps and get the predict results.")
            st.write("**Step 1: Choose the way to input data: input a specific features directlly or load a csv file**")
            st.image('material/Guidance_1.jpg')

            st.write("**Step 2: Input features information**")
            st.write('2.1. If you want to input features one by one directly:')            
            st.image('material/Guidance_4.jpg')
            st.write("2.2. If you want load a whole csv file, choose type of model and click 'Browse files' to load csv file. Download a sample of the csv file if needed.")    
            st.image('material/Guidance_2.jpg')
            st.write('''*Note: Because of complex processing, we suggest you to upload csv file 
                        with maximum **500 rows** each time.*''') 

            st.write("**Step 3: Click 'Show predict results'**")   
            st.image('material/Guidance_3.jpg')         
            
            st.write("**Step 4: Read the predict results.**")
            st.write("In case of input features directly:")
            st.image('material/Guidance_5.jpg')
            st.write("In case of loading csv file:")
            st.image('material/Guidance_6.jpg')
            
            st.write("**Step 5: In case of inputing features by loading csv file, if you want to download predict results, click 'Download data as CSV'**")
            st.image('material/Guidance_7.jpg')

        # input
        # fea_restaurant = st.text_input('Restaurant name') 
        fea_review_text = st.text_input('Review text')
                
        # content
        if choice3_model == 'Model 1':
               # Load model
            model_loaded_sentiment = load_model('model/project2_LSTM_model_v4.h5')


            # prediction
            lst_input = [fea_review_text, 0.5]
            col_names = ['review_text','review_score']
            input_data = pd.DataFrame(lst_input)
            input_data = input_data.T
            input_data.columns = col_names

            X_new1 = clean_text(fea_review_text)
            X_new1 = pd.DataFrame({"review": [X_new1]})

            max_len = 100
            newtext_sequences = tok.texts_to_sequences(X_new1)
            newtext_sequences_matrix = pad_sequences(newtext_sequences, maxlen=max_len)

            y_pred_newtest1 = model_loaded_sentiment.predict(newtext_sequences_matrix)
            yhat1 = y_pred_newtest1 >= 0.5

            result_df1 = pd.concat([input_data[['review_text']], pd.DataFrame(yhat1)], axis=1)
            result_df1.columns = ['review_text','sentiment']
            result_df1.loc[result_df1['sentiment'] == 0, 'sentiment'] = 'positive'
            result_df1.loc[result_df1['sentiment'] == 1, 'sentiment'] = 'negative'

            # show results - CODE HERE
            if st.button("Show predict results"):
                st.write('Sentiment')
                st.dataframe(result_df1)
                    


        elif choice3_model == 'Model 2':

            model_predict = pickle.load(open('model/Project2_mb_model_prediction.sav', 'rb'))

            # prediction
            lst_input = ['',fea_review_text, 0.5]
            col_names = ['restaurant','review_text','review_score']
            input_data = pd.DataFrame(lst_input)
            input_data = input_data.T
            input_data.columns = col_names

            X_new2, y_new2 = data_cleaning(input_data)
            yhat2 = model_predict.predict(X_new2)

            result_df2 = pd.concat([input_data[['review_text']], pd.DataFrame(yhat2)], axis=1)
            result_df2.columns = ['review_text','sentiment']
            result_df2.loc[result_df2['sentiment']==0, 'sentiment'] = 'positive'
            result_df2.loc[result_df2['sentiment']==1, 'sentiment'] = 'negative'

            # show results
            if st.button("Show predict results"):
                st.write('Sentiment')
                st.dataframe(result_df2)

            
    elif choice3_input == 'Load file':

        menu3_model = ['Model 1', 'Model 2']
        choice3_model = st.sidebar.selectbox('Choose the model', menu3_model)

        # sidebar
        # upload template
        upload_template = pd.read_csv('material/upload_template.csv')
        download_template = upload_template.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(label="Download template as CSV",
                            data=download_template,
                            file_name='template.csv',
                            mime='text/csv',
                            )

        # upload file
        try:
            uploaded_file = st.sidebar.file_uploader('Upload data', type = ['csv'])
        except Exception as failGeneral:        
            print("Fail system, please call developer...", type(failGeneral).__name__)
            print("Description:", failGeneral)
        finally:
            print("File uploaded")


        # show results
        if st.sidebar.button("Show predict results"):

            uploaded_data = pd.read_csv(uploaded_file)
            
        # content
            if choice3_model == 'Model 1':  

                # prediction - CODE HERE
                

                # download results - CODE HERE
                st.write('not done')

            elif choice3_model == 'Model 2':

                model_predict = pickle.load(open('model/Project2_mb_model_prediction.sav', 'rb'))

                # prediction
                uploaded_data['review_score'] = 0.5
                X_new, y_new = data_cleaning(uploaded_data)
                yhat = model_predict.predict(X_new)

                result_df = pd.concat([uploaded_data[['restaurant','review_text']], pd.DataFrame(yhat)], axis=1)
                result_df.columns = ['restaurant','review_text','sentiment']
                result_df.loc[result_df['sentiment']==0, 'sentiment'] = 'positive'
                result_df.loc[result_df['sentiment']==1, 'sentiment'] = 'negative'
                # pd.set_option('display.max_colwidth', None)

                # show results
                st.write("<h5 style='text-align: left; color: Black;'>Input information:</h5>", unsafe_allow_html=True)
                st.write("1. Model name: Model 2")
                st.markdown("2. Input file name: " + str(uploaded_file.name))
                st.markdown("3. Number of distinct rows: " + str(uploaded_data.shape[0]))
                st.markdown("4. Number of columns: " + str(uploaded_data.shape[1]-1))
                st.write("<h5 style='text-align: left; color: Black;'>Prediction results:</h5>", unsafe_allow_html=True)
                st.dataframe(result_df)
                # AgGrid(result_df, fit_columns_on_grid_load=True)
                download_results = result_df.to_csv().encode('utf-8')

                # download results
                st.download_button(label="Download data as CSV",
                                    data=download_results,
                                    file_name='predict_results_model2.csv',
                                    mime='text/csv',
                                    )
              

        else:
            st.write("<h5 style='text-align: left; color: Black;'>Guidance:</h5>", unsafe_allow_html=True)
            st.write("To use specific models to predict sentiment, please follow these steps and get the predict results.")
            st.write("**Step 1: Choose the way to input data: input a specific features directlly or load a csv file**")
            st.image('material/Guidance_1.jpg')

            st.write("**Step 2: Input features information**")
            st.write('2.1. If you want to input features one by one directly:')            
            st.image('material/Guidance_4.jpg')
            st.write("2.2. If you want load a whole csv file, choose type of model and click 'Browse files' to load csv file. Download a sample of the csv file if needed.")    
            st.image('material/Guidance_2.jpg')
            st.write('''*Note: Because of complex processing, we suggest you to upload csv file 
                        with maximum **500 rows** each time.*''') 

            st.write("**Step 3: Click 'Show predict results'**")   
            st.image('material/Guidance_3.jpg')         
            
            st.write("**Step 4: Read the predict results.**")
            st.write("In case of input features directly:")
            st.image('material/Guidance_5.jpg')
            st.write("In case of loading csv file:")
            st.image('material/Guidance_6.jpg')
            
            st.write("**Step 5: In case of inputing features by loading csv file, if you want to download predict results, click 'Download data as CSV'**")
            st.image('material/Guidance_7.jpg')
    

# 4. Conclusion and Next steps
elif choice == 'Conclusion and Next steps':

    st.header('CONCLUSION AND NEXT STEPS')

    




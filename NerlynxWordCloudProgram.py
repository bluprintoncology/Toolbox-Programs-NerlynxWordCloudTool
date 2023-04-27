# Python program to generate WordCloud, make sure WordCloud, Matplotlib, and Pandas are installed 

#import streamlit, io, pandas, matplotlib, wordcloud
import streamlit as st
from io import StringIO
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import streamlit_ext as ste 
from unidecode import unidecode
import numpy as np
import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

##-------Funtions to be excuted in the application-------
##Convert plot from bytes format to downloadable png format
def convert_plot():
    buffer = BytesIO()
    dataset = plt.savefig(buffer,format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return image_png

#Convert dataframe to txt for export
def convert_df(df):
    return df.to_csv(index=False, header=False).encode('utf-8')

#Convert dataframe to csv for export
def convert_csv_to_df(df):
    return df.to_csv().encode('utf-8')

##-------StopWords Template Setup-------
import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#Stopwords dataframe
df_stopwords_template_sklearn = pd.DataFrame(ENGLISH_STOP_WORDS,columns=['StopwordsSet'])
#Stopwords List
stoplistextend=df_stopwords_template_sklearn['StopwordsSet'].to_list()

##-------App Title-------
# Add Title for Streamlit App
st.title('Word Cloud Tool')

##-------Work Instructions Tool Overview and Process Steps-------
st.markdown("The word cloud tool creates an image showing the frequency of word use with smaller words used less frequently and larger words used more frequently. A Stopword list is used to filter out words that you want to ignore. Follow these instructions to create your word cloud.")
st.markdown("""
1. Prepare your excel spreadsheet by labeling the first cell of the column which includes your source data as CONTENT (**ensure all caps**)
2. Save your file as a CSV
3. Prepare your stopwords txt file by downloading the txt file, adding or removing any words to be ignored and save
4. Add your spreadsheet and txt files to the tool
5. Select any parameters you would like to apply
6. Examine the output – if you want to remove additional words, add to your stopword file, save and reupload your new stopwords
7. Download your Word Cloud!
""")

##-------File Uploader Widgets-------
# File Downloader Widget for StopWord List Template
stop_word_template = convert_df(df_stopwords_template_sklearn)
st.subheader('Download Template StopWord List (Optional)')
ste.download_button(
    label="Download StopWord Template.txt",
    data=stop_word_template,##text file - reformat text file conversion from df
    file_name='StopWordTemplate.txt', ##text file name assigned
    mime='text/csv', ##data type
    )
# File Uploader widgets with specified .csv and .txt file types with error statements returned
st.subheader("Select a CSV file")
content_uploaded_file = st.file_uploader("Choose a CSV file", type = [".csv"])
st.subheader("Select a stopword text file")
stopword_uploaded_file = st.file_uploader("Choose a text file", type = [".txt"])


##-------Body of App-------
if content_uploaded_file and stopword_uploaded_file is not None:
    #add check point to see if files are uploaded
    df = pd.read_csv(content_uploaded_file,encoding="latin-1") #error until files loaded
    df = df["CONTENT"].apply(unidecode)
    #Call in uploaded stopwords text file as df
    stopdf = pd.read_csv(stopword_uploaded_file,names=["Stopwords"])
    #Convert uploaded stopdf to list
    stoplist=stopdf['Stopwords'].to_list()
    #New df to split Input csv string, into list of individual words
    newdfsplit = df.str.split()
    #Remove stop words from newdfsplit row list
    for row in range(len(newdfsplit)):
        dfrowinputlist = (newdfsplit.loc[row]) #prints row as list of string characters
        words = [word for word in dfrowinputlist if word.lower() not in ENGLISH_STOP_WORDS]
        regexwords = ["".join(list(filter(str.isalnum, oneword))) for oneword in words]
    freq_count_df=pd.value_counts(np.array(words))
    freq_count_df_regex=pd.value_counts(np.array(regexwords))
    #input file take as bytes data, need to read out information from bytes back to string - need to append to list after
    st.subheader('Review Input data (Optional)')
    preview = st.checkbox('Show input data')
    preview_placeholder = st.empty()

    if preview:
        with preview_placeholder.container():
            st.subheader('Input data')
            st.write(df) #main df with long strings
            #st.write(df_stopwords_template_sklearn)
            #st.write(newdfsplit)#Filtered out stop words
            #st.write(stoplistextend)
            #st.write(words)
            #st.write(regexwords)
            #st.write(freq_count_df) #df showing count frequency
            #st.write(freq_count_df_regex)
            #st.write("Review your input data before continuing to run the tool")
            st.stop() #Box checked: stops run
    else:
        preview_placeholder.empty() #Box unchecked: continues to run and no dataframe shown


    comment_words = ""

    placeholder=st.empty()
    placeholder.text("Generating WordCloud. Please wait...")

    for val in df: 
    # typecaste each val to string 
        val = str(val) 
    # split the value 
        tokens = val.split()     
        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower()  
        for words in tokens:
            comment_words = comment_words + words + ' '
    #st.write(comment_words)
    placeholder.empty()
    
    st.subheader('Customize WordCloud Design(Optional)')
    wordcloud = WordCloud(width = 800, height = 800, 
        background_color = st.color_picker('Select a background color','#fff'), 
        stopwords = stoplistextend,
        max_words = st.slider('Select the number of words to be displayed',1,50,25),
        min_font_size = 10).generate(comment_words)

# plot the WordCloud image
    figure = plt.figure(figsize = (8, 8), facecolor = None)
    # placeholder.empty() 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0)
    st.pyplot(figure)
    plot_data = convert_plot()
    #use streamlit extension to download png without rerunning code
    st.subheader('Download WordCloud Image')
    ste.download_button(
        label="Download WordCloud.png",
        data=plot_data,
        file_name='WordCloud.png',
        mime='image/png',
    )
    wordfreqcount = convert_csv_to_df(freq_count_df_regex)
    st.subheader('Download Word Frequency Count (Optional)')
    ste.download_button(
        label="Download WordFrequencyCount.csv",
        data=wordfreqcount,
        file_name='WordFrequencyCount.csv',
        mime='text/csv',
    )

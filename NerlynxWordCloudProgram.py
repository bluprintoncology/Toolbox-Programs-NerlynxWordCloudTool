# Python program to generate WordCloud, make sure WordCloud, Matplotlib, and Pandas are installed 

#import streamlit, io, pandas, matplotlib, wordcloud
import streamlit as st
from io import StringIO
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import streamlit_ext as ste 


def convert_stopwordfile (stopword_file):
    bytes_data = stopword_file.read()
    sw_list=bytes_data.decode("utf-8").splitlines()
    return sw_list

def convert_plot():
    buffer = BytesIO()
    dataset = plt.savefig(buffer,format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return image_png


# Add Title for Streamlit App
st.title('NerlynxWordCloudTool')

# File Uploader widgets with specified .csv and .txt file types with error statements returned
st.subheader("Select a CSV file")
content_uploaded_file = st.file_uploader("Choose a CSV file", type = [".csv"])
st.subheader("Select a stopword text file")
stopword_uploaded_file = st.file_uploader("Choose a text file", type = [".txt"])

if content_uploaded_file and stopword_uploaded_file is not None:
    
#add check point to see if files are uploaded
    df = pd.read_csv(content_uploaded_file,encoding="latin-1") #error until files loaded

#input file take as bytes data, need to read out information from bytes back to string - need to append to list after
    st.subheader('Review Input data (Optional)')
    if st.checkbox('Show input data'):
        st.subheader('Input data')
        st.write(df)
    comment_words = ""
    for val in df.CONTENT: 
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

    st.subheader('Customize WordCloud Design(Optional)')
    wordcloud = WordCloud(width = 800, height = 800, 
        background_color = st.color_picker('Select a background color','#fff'), 
        stopwords = convert_stopwordfile(stopword_uploaded_file),
        max_words = st.slider('Select the number of words to be displayed',1,21,21),
        min_font_size = 10).generate(comment_words)

# plot the WordCloud image
    figure = plt.figure(figsize = (8, 8), facecolor = None) 
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
 




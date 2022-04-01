"""IN this model we create a simple web app.
We take input from user(UI) and
do prediction using src/models/predict_model.py"""


import streamlit as st

def app():
    """Function that create web app"""

    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # st.set_page_config(layout="wide")
    # Add text and data

    # Add a title
    # st.title('My first app')
    # Along with magic commands, st.write() is Streamlit’s “Swiss Army knife”.
    # You can pass almost anything to st.write():
    # text, data, Matplotlib figures, Altair charts, and more. Don’t worry,
    #  Streamlit will figure it out and render things the right way.
    # st.write()  # write
    # your own markdown.
    st.markdown(
        """<h1 style='text-align: center; color: black;'>
            Madan</h1>""",
        unsafe_allow_html=True,
    )

    # Write a data frame
    # st.dataframe()
    # st.table()

    # Use magic (streamlit call nagarikana ni dherai garna milxa)
    # """
    # # My first app
    # Here's our first attempt at using data to create a table:
    # """

    # df = pd.DataFrame({
    # 'first column': [1, 2, 3, 4],
    # 'second column': [10, 20, 30, 40]
    # })

    # df

    # Draw charts and maps
    # st.line/bar/..(dataframe)

    # Add interactivity with widgets
    # Checkbox
    # if st.checkbox('Show dataframe'):
    #  (click garyapaxi sabai dekhauxa tyati matra)
    #     chart_data = pd.DataFrame(
    #     np.random.randn(20, 3),
    #     columns=['a', 'b', 'c'])
    #     chart_data

    # option = st.selectbox(      # kunai auta select garyasi k garni vanni
    # 'Which number do you like best?',
    #  [1,2,3])
    # 'You selected: ', option
    #  you can put your logic here
    #  if option == 1:
    #      then...

    options = [
        "About",
        # "predicted_for_one_student",
        # "prediction_for_all_students_and_dump_into_mongoserver",
        # "Label Interpretation",
    ]

    option = st.sidebar.selectbox("What do you want?", options)

    # Prediction in dummy features
    if option == "About":
        
            st.text('Madan Library')
        
if __name__ == "__main__":
    app()
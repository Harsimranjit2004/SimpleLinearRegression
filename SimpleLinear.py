import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import io


import streamlit as st
import requests


import requests

def download_pdf():
    pdf_url = 'https://drive.google.com/uc?id=1AdZ4Oybf02Cp7JqQ_cPfBtgtnLWYlh53&export=download'
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open("downloaded_file.pdf", 'wb') as f:
            f.write(response.content)
        return "downloaded_file.pdf"
    else:
        return "Failed to download the file."

# Call the function to download the PDF


def IntroPage():
    
    st.markdown("""
    ## Linear Regression
                """)
    st.markdown(
        """
        ***Linear Regression*** is a statistical method that models the relationship between two variables by fitting a linear equation to observed data. 
        It is one of the simplest and most widely used regression techniques in statistics and machine learning. 
        
        The goal of linear regression is to find the <span style="font-size:larger;">**Best-fitting**</span> line  that predicts the target variable based on one or more input features.
        """,
        unsafe_allow_html=True
    )

    path = "/Users/harsimranjitsingh/Desktop/machine_learning/Simple_linear/SimpleLinear copy.png"
    print(os.path.exists(path))
    st.image(path,use_column_width=True)
    # st.image('./SimpleLinear.png', caption="Your Image Caption", use_column_width=True)

    st.markdown("""
    # Types of Linear Regression

    - Simple Linear Regression (1D)
    - Multiple Linear Regression (nD)
    - Polynomial Regression
    """)
    st.markdown("""
    # Simple Linear Regression

    In Simple Linear Regression, there's only one independent variable. The linear equation takes the form:

    \[ y = mx + b \]

    - \( y \) is the dependent variable.
    - \( x \) is the independent variable.
    - \( m \) is the slope of the line.
    - \( b \) is the y-intercept.

    The goal is to find the values of \( m \) and \( b \) that minimize the loss function.
    """)

    st.markdown("""
    # Finding m and b
    The values of m and b can be find with two methods:
    ### Closed Form Solution: (OLS)
        
    """)

    st.latex(r'''
    m = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^{n}(x_i-\bar{x})^2}
            ''')

    st.latex(r'''
    b = \bar{y} - m\bar{x}
            ''')
    st.latex(r'''
    \bar{y}: \text{Mean value of y column}\newline
    \bar{x}: \text{Mean value of x column}\newline
    n: \text{Number of input}
    ''')
    st.markdown('''
    ### Non-Closed Form Solution  (Approximation):
    - Gradient Descent
    '''
    )
    st.markdown('''
    # Implementing Using Sklearn 
                    
        ''')
    st.write("Let's explore an example of linear regression using a dataset where the CGPA of students serves as the input variable. In this scenario, we aim to predict the package of the student in Lakhs Per Annum (LPA) based on their CGPA.")

    st.markdown('''
    ### Importing the necessary modules            ''')
    st.code('''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression''')

    st.markdown('''
    ### Preparing data for Training
    ''')

    st.code('''
    df = pd.read_csv('placement.csv')
    # extracting the input and output column
    x = df.iloc[:,0:1]
    y = df.iloc[:,1:]

    # splitting data         
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
            ''')

    st.markdown('''
    ### Training Model
    ''')

    st.code('''
    lr = LinearRegression() # creating object of LinearRegression class

    lr.fit(X_train,y_train)  # train the model on training data
    ''')
    st.markdown('''
    ### Predicting''')
    st.code('''
    lr.predict(X_test.iloc[0].values.reshape(1,1))''')

    st.markdown('''
    ### Visualize''')
    st.code('''
    plt.scatter(df['TV'],df['Sales'])
    plt.plot(X_train,lr.predict(X_train),color= 'red')
    plt.xlabel('Input')
    plt.ylabel('Output')
    ''')
    st.image(path,use_column_width=True)
    st.markdown('''
    ### Accuracy:(R2_score)
    To compute the Accuracy we can use the r2_socre which provides that how good our model is
                
                ''')
    st.code('''
r2Score = r2_score(y_pred, y_test)
            ''')
    st.markdown('''
More the r2_score More accurate is our model
                ''')

def LinearRegressionModel():
# Page title and description
    st.title("Linear Regression ")
    

# Upload dataset
    uploaded_file = st.file_uploader("Upload a CSV file ", type=["csv"])
    if uploaded_file is not None:
        st.write("Uploaded file preview:")
        df = pd.read_csv(uploaded_file)
    
        st.header('Data Exploration')
        st.write("Explore the dataset")

        if st.checkbox("Show Dataset"):
            st.write(df.head())
        if st.checkbox("Show Statistics Summary"):
            st.write(df.describe())

    # Choose input and target columns
        input_column = st.selectbox("Select the input column (independent variable)", df.columns)
        target_column = st.selectbox("Select the target column (dependent variable)", df.columns)

    # extrcting x and y
        X = df[[input_column]]
        y = df[target_column]

    # Explore the relationship between x and y 
        fig, ax = plt.subplots()
        ax.scatter(X, y)
        ax.set_xlabel(input_column)
        ax.set_ylabel(target_column)
        st.pyplot(fig) 
        # st.scatter_chart(X,y)
    # train test split
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.5,step=0.05, value=0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    
    # Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

    # Predictions on test set
        y_pred = model.predict(X_test)

    # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse}")

        r2 = r2_score(y_test,y_pred)
        st.write(f"R-Squared Score (R2): {r2}")

    # regression coffecents
        st.header("Regression Coefficients")
        coff_df = pd.DataFrame({
            'Feature':['Intercept']+X.columns.tolist(),
            'Coefficient': [model.intercept_] + model.coef_.tolist()
        })
        st.table(coff_df)

    # Visualization
        fig, ax = plt.subplots()
        ax.scatter(X_test, y_test, color='blue', label='Actual')
        ax.plot(X_test, y_pred, color='red', linewidth=3, label='Predicted')
        ax.set_xlabel(input_column)
        ax.set_ylabel(target_column)
        ax.legend()
        st.pyplot(fig)
    # download the prediction and actjual csb
        st.subheader("Download Actual vs Predicted as csv")
        pred_actual = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
        st.dataframe(pred_actual.head())
        csv_data = pred_actual.to_csv(index=False).encode()
        csv_button = st.download_button(
            label = "Download Predction vs Actual as CSV",
            data= csv_data,
            file_name="prediction_actual.csv",
            key="predictions_actual",    
        )
        st.subheader("Download Trained Model")
        model_bytes = io.BytesIO()
        pd.to_pickle(model, model_bytes)
        st.download_button(label="Download Trained Model", data=model_bytes, file_name="linear_regression_model.pkl", key="download_model")
    # Display the code
        st.write("Code to run the linear regression:")
        st.code(f"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("{uploaded_file.name}")

# Choose input and target columns
X = df["{input_column}"]
y = df["{target_column}"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on test set
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test,y_pred)
print(f"R2 Score: {r2}")
""")


def detailedTheroy():
    st.markdown('''
# Odinary Least Squares ''')
    st.write('''
Our focus in simple linear regression is to find the best fit line that cause less error with the data''')
    st.write('''
Lets take 4 random points and draw the initial regression line on the graph 
Our goal is to minimize the error with the data these means we need to minimize the sum of distances between the points and the line 
''')
    st.write('''
Let's take the distance from the point 1 to 5 be d1,d2,d3,d4,d5''')
    st.write('''So we need to minimize the sum of d1+d2+d3+d4+d5''')


    st.markdown('''''')
    x_values = np.array([1, 2, 3, 4])
    y_values = np.array([2, 4, 5, 4])

    # Initial slope and intercept values
    m_initial = 1.5
    b_initial = 1.5

    # Calculate predicted values
    y_hat_initial = m_initial * x_values + b_initial

    # Plot the initial state
    st.subheader("Initial State")
    fig, ax = plt.subplots()
    ax.scatter(x_values, y_values, label='Actual Points')
    ax.plot(x_values, y_hat_initial, label='Initial Line', color='red', linestyle='dashed')
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()
    st.pyplot(fig)

    st.write('''
### Square of distances''')
    st.write('''
As some of the points may be in the negative side or the positve side of the regression line Therefore while addition it may be cacelout each other to prevent that we add the square of the distances''')
    st.write('''
### Error Function''')
    
    st.latex(r'''
E = \sum_{i=1}^{n}(d_i^2)''')
    st.write('''
The square can be donated by the difference of Actual value and the predicted value''')
    st.latex(r'''
E = \sum_{i=1}^{n}(Y_i - \hat{Y_i})^2\newline
             Y_i =  ActualValue\newline
             \hat{Y_i} = Predicated Value''')
    
    
    # Derive the squared errors
    squared_errors = (y_values - y_hat_initial) ** 2

    # Visualize squared errors
    st.subheader("Squared Errors")
    for i, (x, y, y_hat, error) in enumerate(zip(x_values, y_values, y_hat_initial, squared_errors)):
        plt.plot([x, x], [y, y_hat], color='black', linestyle='dashed', marker='o')
        plt.text(x + 0.1, (y + y_hat) / 2, f"Error{i+1}: {error:.2f}", verticalalignment='bottom', horizontalalignment='left')
    # Plotting the squared error distance
        plt.plot([x, x], [y, y_hat], color='red', linestyle='-', linewidth=0.5)
    # Plotting horizontal lines for error visualization
        plt.plot([x, x], [y, y_hat], color='blue', linestyle='-', linewidth=0.5)

    plt.scatter(x_values, y_values, label='Actual Points')
    plt.plot(x_values, y_hat_initial, label='Initial Line', color='red', linestyle='dashed')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    st.pyplot()
    st.write('''
Predicated value can be obtained by the line equation as that point lies on the regression line
let the slope of line be m and the intercept be b. Therefore by using y = mx +b  we can calculated the value of predicated value.
        
''')
    st.latex(r'''
E = \sum_{i=0}^{n} (y_i - m*x_i -b )^2''')

    st.write('''
In order to minimize the value be need to find the partial derivative of equation with repect to m and b and equate it to the zero''')
    st.latex(r'''
\frac{\partial E}{\partial m} = 0 \quad Partial \:Derivative \:with \:respect \:to \:m''')
    st.latex(r'''
\frac{\partial E}{\partial b} = 0  \quad Partial \:Derivative \:with \:respect \:to \:b''')
    
    st.write('''
After finding the partial derivative and equating to zero we will find the equation for the both m and b''')
    st.latex(r'''
    m = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^{n}(x_i-\bar{x})^2}
            ''')

    st.latex(r'''
    b = \bar{y} - m\bar{x}
            ''')
    st.write('''
By using these equation we can implement  our simple Linear Regression''')
    st.markdown('''
## Code For Own Linear Regression''')
    st.code('''class OwnLR:
    def __init__(self):
        self.m = None
        self.b = None
        
    
    def fit(self, X_train,y_train):
        num = 0
        den = 0

        for i  in range(X_train.shape[0]):
            num = num + ((X_train[i] - X_train.mean())*(y_train[i] - y_train.mean()))
            den = den + ((X_train[i] - X_train.mean())*(X_train[i] - X_train.mean()))
        
        self.m = num/den
        self.b = y_train.mean()  - (self.m * X_train.mean())

    def predict(self,X_test):
        return self.m * X_test + self.b''')
    st.markdown('''
## Download the Notes''')

    
    #st.download_button(label="Download PDF", data="https://drive.google.com/file/d/1AdZ4Oybf02Cp7JqQ_cPfBtgtnLWYlh53/view?usp=share_link")
    downloaded_file_path = download_pdf()
    if downloaded_file_path:
        st.markdown('<iframe src="https://drive.google.com/file/d/1AdZ4Oybf02Cp7JqQ_cPfBtgtnLWYlh53/preview" width="700" height="600"></iframe>', unsafe_allow_html=True)
        with open(downloaded_file_path, "rb") as file:
            btn = st.download_button(
                label="Download PDF",
                data=file,
                file_name="downloaded_file.pdf",
                mime="application/pdf"
        )
    else:
        st.error("Failed to download the PDF file.")





st.set_page_config(layout="centered") 
st.markdown("""
    <h1 style = "text-align:center"  color: "#eb6b40">Simple Linear Regression</h1>
    """,unsafe_allow_html=True
    )

st.sidebar.header("Menu")

Menu = st.sidebar.selectbox("Select",["Introduction", "Live Demo", "Detailed Theory"])

st.set_option('deprecation.showPyplotGlobalUse', False)

if(Menu == 'Introduction'):
    IntroPage()
elif (Menu == "Live Demo"):
    LinearRegressionModel()
else:
    detailedTheroy()

    









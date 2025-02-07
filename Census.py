import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import RocCurveDisplay

# Custom Stacking Classifier
class CustomStackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimators, meta_estimator):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator

    def fit(self, X, y):
        # Train base estimators
        for estimator in self.base_estimators:
            estimator.fit(X, y)
        
        # Get base estimators' predictions
        meta_features = np.column_stack([estimator.predict(X) for estimator in self.base_estimators])
        
        # Train meta-estimator
        self.meta_estimator.fit(meta_features, y)
        return self

    def predict(self, X):
        # Get base estimators' predictions
        meta_features = np.column_stack([estimator.predict(X) for estimator in self.base_estimators])
        
        # Predict using meta-estimator
        return self.meta_estimator.predict(meta_features)
    def predict_proba(self, X):
        # Get base estimators' predicted probabilities
        meta_features_proba = np.column_stack([estimator.predict_proba(X)[:, 1] for estimator in self.base_estimators])
        
        # Predict probabilities using meta-estimator
        return self.meta_estimator.predict_proba(meta_features_proba)
# Streamlit App Code
    

    st.set_page_config(
    page_title="Census Income Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load Dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess Data
@st.cache_data
def preprocess_data(df):
    df.replace('?', np.nan, inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    le = LabelEncoder()
    df['income'] = le.fit_transform(df['income'])

    X = df.drop('income', axis=1)
    Y = df['income']

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col])

    scaler = StandardScaler()
    X[X.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(
        X.select_dtypes(include=['int64', 'float64'])
    )

    ros = RandomOverSampler(random_state=42)
    X_resampled, Y_resampled = ros.fit_resample(X, Y)

    return train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)

# Page Control Logic
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'show_sidebar' not in st.session_state:
    st.session_state.show_sidebar = False

# Home Page
if st.session_state.uploaded_file is None:
    st.markdown("""<h1 style='text-align: center; color: blue;'>Census Income Prediction</h1>""", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center;'>A Data-Driven Approach to Understand and Address Economic Inequality</h3>",
        unsafe_allow_html=True
    )
    st.write("This platform provides insights into income prediction models and offers tools for data exploration.")

    if st.button("Next"):
        st.session_state.uploaded_file = "dummy.csv"  # Dummy to simulate file upload for next step
        st.session_state.show_sidebar = True
        st.rerun()

# Upload and Dataset Analysis Page
if st.session_state.uploaded_file is not None:
    # Sidebar Menu appears only after the dataset is uploaded
    if st.session_state.show_sidebar:
        menu_options = ["Home", "Dataset Analysis", "Visualizations", "Train/Test Split", "Comparative Models", "Ensemble Model", "Suggestions"]
        choice = st.sidebar.radio("Select a page", menu_options)

        uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"])
        if uploaded_file:
            df = load_data(uploaded_file)
        else:
            st.warning("Please upload a dataset to proceed.")
            st.stop()

        # Page: Dataset Analysis
        if choice == "Dataset Analysis":
            st.title("Dataset Analysis")
            st.write(df.head())
            st.write(f"Shape of Dataset: {df.shape}")
            st.write("Statistical Summary:")
            st.write(df.describe().T)
            st.write("Missing values:")
            st.write(df.isnull().sum())

            # Download dataset
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Dataset as CSV",
                data=csv_data,
                file_name='dataset.csv',
                mime='text/csv'
            )

        # Page: Visualizations
        elif choice == "Visualizations":
            st.title("Data Visualizations")

            # Income Distribution
            st.subheader("Income Distribution")
            income_counts = df['income'].value_counts()
            fig1, ax1 = plt.subplots()
            sns.barplot(x=income_counts.index, y=income_counts.values, palette='viridis', ax=ax1)
            st.pyplot(fig1)

            # Education Distribution
            st.subheader("Education Distribution")
            education_counts = df['education'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            sns.barplot(x=education_counts.index, y=education_counts.values, palette='coolwarm', ax=ax2)
            plt.xticks(rotation=45)
            plt.xlabel("Education Levels", fontsize=12)
            plt.ylabel("Counts", fontsize=12)
            st.pyplot(fig2)

            # Marital Status Distribution
            st.subheader("Marital Status Distribution")
            marital_counts = df['marital.status'].value_counts()
            fig3, ax3 = plt.subplots(figsize=(10, 7))
            ax3.pie(
                marital_counts.values,
                labels=marital_counts.index,
                autopct='%1.1f%%',
                shadow=True,
                startangle=90,
                textprops={'fontsize': 12}
            )
            ax3.legend(marital_counts.index, loc="upper right", fontsize=10)
            st.pyplot(fig3)

        # Page: Train/Test Split
        elif choice == "Train/Test Split":
            st.title("Train/Test Split")
            X_train, X_test, Y_train, Y_test = preprocess_data(df)

            train_data = pd.concat([X_train, Y_train], axis=1)
            test_data = pd.concat([X_test, Y_test], axis=1)

            st.subheader("Train Dataset")
            st.write(train_data.head())

            st.subheader("Test Dataset")
            st.write(test_data.head())

            # Download train/test datasets
            train_csv = train_data.to_csv(index=False).encode('utf-8')
            test_csv = test_data.to_csv(index=False).encode('utf-8')

            st.download_button("Download Train Dataset as CSV", train_csv, "train_dataset.csv", "text/csv")
            st.download_button("Download Test Dataset as CSV", test_csv, "test_dataset.csv", "text/csv")

        # Page: Comparative Models
        elif choice == "Comparative Models":
            st.title("Comparative Models")
            model_choice = st.selectbox("Select a model", ["Logistic Regression", "KNN", "Random Forest"])

            X_train, X_test, Y_train, Y_test = preprocess_data(df)

            model_mapping = {
                "Logistic Regression": LogisticRegression(random_state=42),
                "KNN": KNeighborsClassifier(),
                "Random Forest": RandomForestClassifier(random_state=42)
            }

            model = model_mapping[model_choice]
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            accuracy = accuracy_score(Y_test, Y_pred)
            f1 = f1_score(Y_test, Y_pred)

            st.write(f"Accuracy: {accuracy * 100:.2f}%")
            st.write(f"F1 Score: {f1 * 100:.2f}%")

            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, cmap='Blues', fmt='d', ax=ax)
            st.pyplot(fig)

            # ROC Curve
            fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)

            st.subheader("ROC Curve")
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic')
            ax_roc.legend(loc='lower right')
            st.pyplot(fig_roc)

        # Page: Ensemble Model
        elif choice == "Ensemble Model":
            st.title("Ensemble Model")

            X_train, X_test, Y_train, Y_test = preprocess_data(df)

            # Define base estimators
            base_estimators = [
                KNeighborsClassifier(),
                RandomForestClassifier(random_state=42),
                XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            ]
            meta_estimator = GradientBoostingClassifier(random_state=42)

            # Create custom stacking classifier
            custom_stacking_clf = CustomStackingClassifier(base_estimators, meta_estimator)
            custom_stacking_clf.fit(X_train, Y_train)
            Y_pred = custom_stacking_clf.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(Y_test, Y_pred)
            f1 = f1_score(Y_test, Y_pred)

            st.write(f"Custom Stacking Model Accuracy: {accuracy * 100:.2f}%")
            st.write(f"F1 Score: {f1 * 100:.2f}%")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, cmap='Greens', fmt='d', ax=ax)
            st.pyplot(fig)

            # ROC Curve for Ensemble Model
            fpr, tpr, thresholds = roc_curve(Y_test, custom_stacking_clf.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)

            st.subheader("Ensemble Model ROC Curve")
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic (Ensemble Model)')
            ax_roc.legend(loc='lower right')
            st.pyplot(fig_roc)

        # Page: Suggestions
        elif choice == "Suggestions":
            st.title("Suggestions and Measures for Poverty Treatment")
            options = [
                "Improve access to quality education",
                "Provide affordable healthcare services",
                "Implement employment programs",
                "Increase financial literacy"
            ]

            selected = st.multiselect("Select measures to explore:", options)

            for option in selected:
                if option == "Improve access to quality education":
                    st.subheader(option)
                    st.write("- Build more schools\n- Provide free or subsidized education\n- Invest in teacher training.")

                elif option == "Provide affordable healthcare services":
                    st.subheader(option)
                    st.write("- Increase affordable clinics\n- Subsidize essential medicines\n- Implement community health programs.")

                elif option == "Implement employment programs":
                    st.subheader(option)
                    st.write("- Create skill development programs\n- Encourage entrepreneurship\n- Offer job placement services.")

                elif option == "Increase financial literacy":
                    st.subheader(option)
                    st.write("- Conduct workshops on savings\n- Provide low-interest loans\n- Encourage formal banking.")

            feedback = st.text_area("Provide Your Feedback", "")
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback!")

            if st.button("Exit to Dataset Analysis"):
                st.session_state.uploaded_file = None
                st.session_state.show_sidebar = False
                st.rerun()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import mean_squared_error, accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm
from scipy.cluster.hierarchy import dendrogram, linkage

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Home - My App",
    page_icon="🏠",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
        html, body, [data-testid="stApp"] {
            background-color: #ff69b4;
            font-family: Arial, sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            color: #2C3E50;
            padding-top: 30px;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #555;
            margin-bottom: 50px;
        }
        .section-title {
            font-size: 18px;
            font-weight: bold;
            margin-top: 20px;
            padding: 8px;
            border-radius: 5px;
        }
        .independent-label {
            background-color: #d4f1f9;
            color: #0b7285;
        }
        .dependent-label {
            background-color: #ffe3e3;
            color: #c92a2a;
        }
        .split-label {
            background-color: #e8f5e9;
            color: #1b5e20;
        }
        .stButton>button {
            background-color: #3498DB;
            color: white;
            border-radius: 8px;
            height: 3em;
            font-size: 16px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #2980B9;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- MAIN TITLE ----------
st.markdown("<div class='main-title'>Welcome to My App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your one-stop solution for predictions, data analysis, and more</div>", unsafe_allow_html=True)

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader(" Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # ---------- VARIABLE SELECTION ----------
    st.markdown("<div class='section-title independent-label'>Select Independent Variables (Features)</div>", unsafe_allow_html=True)
    independent_vars = st.multiselect(
        "Independent Variables:",
        options=df.columns.tolist()
    )

    st.markdown("<div class='section-title dependent-label'>Select Dependent Variable (Target)</div>", unsafe_allow_html=True)
    dependent_vars = st.selectbox(
        "Dependent Variable:",
        ["None"] + list(df.columns),
        index = 0
    )

    X, y = None, None
    X_train = X_test = y_train = y_test = None


    # Independent variables must be selected
    if independent_vars:
        X = df[independent_vars]
        st.success(f"Selected Independent Variables: {independent_vars}")
    else:
        st.warning("Please select at least one independent variable.")
        X = None  # Prevent errors if nothing is selected



        # ---------- OPTIONAL CATEGORICAL ENCODING ----------
        st.markdown(
            "<div class='section-title' style='background-color:#fff3cd; color:#856404;'>Optional: Encode Categorical Variables</div>",
            unsafe_allow_html=True)
        encode_categorical = st.checkbox(
            "Encode categorical columns (Label + One-Hot Encoding) and avoid dummy variable trap")

        if encode_categorical:
            categorical_cols = st.multiselect(
                "Select categorical columns to encode",
                options=X.columns.tolist()
            )

            if categorical_cols:
                ct = ColumnTransformer(
                    transformers=[
                        ("encoder", OneHotEncoder(drop='first'), categorical_cols)  # drop first to avoid dummy trap
                    ],
                    remainder='passthrough'
                )

                X_encoded = ct.fit_transform(X)
                X = pd.DataFrame(X_encoded)
                st.success("Categorical encoding applied successfully!")
                st.dataframe(X)
            else:
                st.warning("No categorical columns selected for encoding.")
        else:
            st.info("Categorical encoding skipped.")

        # ====== Optional Feature Scaling ======
        scale_features = st.checkbox(" Apply Feature Scaling", value=False)
        if scale_features:


            sc_x = StandardScaler()
            X = sc_x.fit_transform(X)

            if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_integer_dtype(y):
                sc_y = StandardScaler()
                y = sc_y.fit_transform(y.to_numpy().reshape(-1, 1))
        # OPTIONAL DATA SPLITTING ----------
        if st.checkbox(" Split dataset into Training and Testing sets"):
            st.markdown("<div class='section-title split-label'> Split the Dataset</div>", unsafe_allow_html=True)
            test_size = st.slider("Select Test Size (%)", min_value=10, max_value=50, value=20, step=5)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=0)

            st.success(f" Dataset split complete! Training set: {len(X_train)} rows, Test set: {len(X_test)} rows.")
        else:
            st.warning("⚠ No train-test split applied. Proceeding with full dataset.")

    # ---------- MACHINE LEARNING OPTIONS ----------
    st.markdown("### ⚙ Select Machine Learning Task")
    task = st.selectbox(
        "Choose a Task",
        ["Regression", "Classification", "Clustering"]
    )

    if task == "Regression":
        model_type = st.selectbox("Select Regression Model", [
            "Simple Linear Regression",
            "Multiple Linear Regression",
            "Polynomial Regression",
            "Support Vector Regression (SVR)",
            "Decision Tree Regression",
            "Random Forest Regression"
        ])
    elif task == "Classification":
        model_type = st.selectbox("Select Classification Model", [
            "Logistic Regression",
            "K-Nearest Neighbors (K-NN)",
            "Support Vector Machine (SVM)",
            "Kernel SVM",
            "Naive Bayes",
            "Decision Tree Classification",
            "Random Forest Classification"
        ])
    elif task == "Clustering":
        model_type = st.selectbox("Select Clustering Model", [
            "K-Means Clustering",
            "Hierarchical Clustering"
        ])
    # ===== CLUSTERING SETTINGS =====
    n_clusters = None
    if task == "Clustering" and model_type in ["K-Means Clustering", "Hierarchical Clustering"]:
        n_clusters = st.number_input("Enter number of clusters", min_value=1, max_value=10, value=3)

    # ---------- MODEL TRAINING ----------
    if st.button("Train Model"):
        if X is None:
            st.error("Please select at least one independent variable before training.")
        elif task != "Clustering" and y is None:
            st.error("Please select both independent and dependent variables before training.")
        else:
            # If no train-test split, use all data
            if X_train is None:
                if task != "Clustering":
                    X_train, X_test, y_train, y_test = X, X, y, y
                else:
                    X_train, X_test = X, X  # y is not used for clustering

            # ===== REGRESSION =====
            if task == "Regression":
                if model_type == "Simple Linear Regression":
                    model = LinearRegression()
                elif model_type == "Multiple Linear Regression":
                    model = LinearRegression()
                elif model_type == "Polynomial Regression":
                    poly = PolynomialFeatures(degree=4)
                    x_poly = poly.fit_transform(X)
                    model = LinearRegression()
                    model.fit(x_poly,y)
                elif model_type == "Support Vector Regression (SVR)":
                    model = SVR(kernel = 'rbf')
                    model.fit(X_train,y_train)
                elif model_type == "Decision Tree Regression":
                    model = DecisionTreeRegressor(random_state = 0)
                elif model_type == "Random Forest Regression":
                    model = RandomForestRegressor(n_estimators=300, random_state=0)

                regressor = model.fit(X_train, y_train)
                predictions = regressor.predict(X_test)


                # ===== OPTIONAL DATA DISPLAY =====
                st.markdown("### Show Data Details")

                with st.expander("Dataset"):
                    st.dataframe(df)

                with st.expander(" X (Features)"):
                    st.write(X)

                with st.expander(" Y (Target)"):
                    st.write(y)

                with st.expander("Predictions"):
                    st.write(predictions)



                if (
                        not pd.DataFrame(X_train).equals(pd.DataFrame(X_test)) or
                        not pd.Series(y_train.ravel()).equals(pd.Series(y_test.ravel()))
                ):
                    with st.expander(" X_train"):
                        st.write(X_train)

                    with st.expander(" y_train"):
                        st.write(y_train)

                    with st.expander("X_test"):
                        st.write(X_test)

                    with st.expander("y_test"):
                        st.write(y_test)

                # Backward Elimination process
                X_with_const = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)
                SL = 0.05
                X_opt = X_with_const[:, list(range(X_with_const.shape[1]))]  # all columns initially

                # Iteratively remove features with p-value > SL
                while True:
                    X_opt = X_opt.astype(np.float64)
                    regressor_OLS = sm.OLS(y, X_opt).fit()
                    max_p_value = max(regressor_OLS.pvalues)
                    if max_p_value > SL:
                        max_p_index = np.argmax(regressor_OLS.pvalues)
                        X_opt = np.delete(X_opt, max_p_index, 1)
                    else:
                        break

                st.subheader("Backward Elimination Summary")
                st.text(regressor_OLS.summary())

                # Show final selected features
                st.write("Selected Features after Backward Elimination:", X_opt.shape[1])

                # ---------- MATPLOTLIB PLOT ----------
                if len(independent_vars) == 1:  # Only plot if single feature
                    plt.figure(figsize=(8, 5))
                    plt.scatter(X_train, y_train, color='red', label="Training Data")
                    plt.plot(X_train, regressor.predict(X_train), color='blue', label="Regression Line")
                    plt.xlabel(independent_vars[0])
                    plt.ylabel(dependent_vars)
                    plt.title(f"{model_type} - Training Data")
                    plt.legend()
                    st.pyplot(plt)

            # ===== CLASSIFICATION =====
            elif task == "Classification":
                # --- Select model ---
                if model_type == "Logistic Regression":
                    model = LogisticRegression(random_state=0)
                elif model_type == "K-Nearest Neighbors (K-NN)":
                    model = KNeighborsClassifier(n_neighbors=5, metric= 'minkowski', p = 2)
                elif model_type == "Support Vector Machine (SVM)":
                    model = SVC(kernel = 'linear', random_state = 0)
                elif model_type == "Kernel SVM":
                    model = SVC(kernel = 'rbf', random_state = 0)
                elif model_type == "Naive Bayes":
                    model = GaussianNB()
                elif model_type == "Decision Tree Classification":
                    model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
                elif model_type == "Random Forest Classification":
                    model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

                # --- Train and predict ---
                model.fit(X_train,y_train)
                predictions = model.predict(X_test)

                st.write("Accuracy:", accuracy_score(y_test, predictions))

                st.markdown("### Show Data Details")
                with st.expander("Dataset"):
                    st.dataframe(df)

                with st.expander(" X (Features)"):
                    st.write(X)

                with st.expander(" Y (Target)"):
                    st.write(y)

                with st.expander("Predictions"):
                    st.write(predictions)

                if (
                        not pd.DataFrame(X_train).equals(pd.DataFrame(X_test)) or
                        not pd.Series(y_train.ravel()).equals(pd.Series(y_test.ravel()))
                ):
                    with st.expander(" X_train"):
                        st.write(X_train)

                    with st.expander(" y_train"):
                        st.write(y_train)

                    with st.expander("X_test"):
                        st.write(X_test)

                    with st.expander("y_test"):
                        st.write(y_test)

                # --- Optional Confusion Matrix ---
                if task == "Classification":
                    cm = confusion_matrix(y_test, predictions)

                    with st.expander("📊 Confusion Matrix"):
                        st.write(cm)

                if task == "Classification":
                    # Check number of features in training data
                    num_features = X_train.shape[1]

                    if num_features == 1:
                        # --- Single feature classification plot ---
                        plt.figure(figsize=(8, 5))
                        plt.scatter(X_train, y_train, color='red', label="Training Data")
                        plt.scatter(X_test, y_test, color='green', label="Test Data")

                        # Predict on sorted X for smooth curve
                        X_sorted = np.sort(X_train, axis=0)
                        y_pred_line = model.predict(X_sorted)
                        plt.plot(X_sorted, y_pred_line, color='blue', label="Prediction Line")

                        plt.xlabel(independent_vars[0])
                        plt.ylabel(dependent_vars)
                        plt.title(f"{model_type} - Classification (1 feature)")
                        plt.legend()
                        st.pyplot(plt)

                    elif num_features == 2:
                        # --- 2D Decision boundary for classification ---
                        # Training set
                        x_set, y_set = X_train, y_train
                        x1, x2 = np.meshgrid(
                            np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                            np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01)
                        )

                        fig1, ax1 = plt.subplots()
                        ax1.contourf(
                            x1, x2,
                            model.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                            alpha=0.75,
                            cmap=ListedColormap(('red', 'green'))
                        )
                        for i, j in enumerate(np.unique(y_set)):
                            ax1.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                                        c=ListedColormap(('red', 'green'))(i), label=j)
                        ax1.set_title(f"{model_type} - Training set")
                        ax1.set_xlabel(independent_vars[0])
                        ax1.set_ylabel(independent_vars[1])
                        ax1.legend()
                        st.pyplot(fig1)

                    # Test set
                    x_set, y_set = X_test, y_test
                    x1, x2 = np.meshgrid(
                        np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                        np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01)
                    )

                    fig2, ax2 = plt.subplots()
                    ax2.contourf(
                        x1, x2,
                        model.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                        alpha=0.75,
                        cmap=ListedColormap(('red', 'green'))
                    )
                    for i, j in enumerate(np.unique(y_set)):
                        ax2.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                                    c=ListedColormap(('red', 'green'))(i), label=j)
                    ax2.set_title(f"{model_type} - Test set")
                    ax2.set_xlabel(independent_vars[0])
                    ax2.set_ylabel(independent_vars[1])
                    ax2.legend()
                    st.pyplot(fig2)

            # ===== CLUSTERING =====
            if task == "Clustering":
                st.markdown("### Clustering Results")

                # Elbow Method for K-Means
                if model_type == "K-Means Clustering":
                    wcss = []
                    for i in range(1, 11):
                        km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                        km.fit(X)
                        wcss.append(km.inertia_)

                    # Plot Elbow
                    fig_elbow, ax_elbow = plt.subplots()
                    ax_elbow.plot(range(1, 11), wcss, marker='o')
                    ax_elbow.set_xlabel('Number of clusters')
                    ax_elbow.set_ylabel('WCSS')
                    ax_elbow.set_title('Elbow Method')
                    st.pyplot(fig_elbow)

                # Fit selected model
                if model_type == "K-Means Clustering":
                    model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
                    predictions = model.fit_predict(X)
                else:
                    # Hierarchical Clustering
                    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                    predictions = model.fit_predict(X)

                    # Dendrogram
                    linked = linkage(X, method='ward')
                    fig_dendro, ax_dendro = plt.subplots(figsize=(10, 5))
                    dendrogram(linked, truncate_mode='lastp', p=30)
                    ax_dendro.set_title('Hierarchical Clustering Dendrogram')
                    ax_dendro.set_xlabel('Sample Index')
                    ax_dendro.set_ylabel('Distance')
                    st.pyplot(fig_dendro)

                df["Prediction"] = predictions

                st.markdown("### Clustered Data")
                st.dataframe(df)

                with st.expander("X (Features)"):
                    st.write(X)
                with st.expander("Predictions"):
                    st.write(predictions)

                # Labels and colors
                cluster_labels = ["Careful", "Standard", "Target", "Careless", "Sensible",
                                  "Cautious", "Adventurous", "Strategic", "Balanced", "Focused"]
                cluster_colors = ["red", "blue", "green", "cyan", "purple",
                                  "orange", "pink", "brown", "gray", "olive"]

                # Scatter plot
                if X.shape[1] >= 2:
                    fig, ax = plt.subplots()
                    for cluster_id in range(len(set(predictions))):
                        ax.scatter(
                            X.iloc[predictions == cluster_id, 0],
                            X.iloc[predictions == cluster_id, 1],
                            s=100,
                            c=cluster_colors[cluster_id],
                            label=cluster_labels[cluster_id]
                        )
                    # Plot centroids
                    if hasattr(model, "cluster_centers_"):
                        centroids = model.cluster_centers_
                    else:
                        centroids = np.array([X[predictions == cid].mean(axis=0) for cid in np.unique(predictions)])

                    ax.scatter(
                        centroids[:, 0],
                        centroids[:, 1],
                        s=300,
                        c='yellow',
                        marker='X',
                        edgecolor='black',
                        label='Centroids'
                    )
                    ax.set_xlabel(X.columns[0])
                    ax.set_ylabel(X.columns[1])
                    ax.set_title(f"{model_type} Results")
                    ax.legend()
                    st.pyplot(fig)